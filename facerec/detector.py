"""人脸检测：YOLOv8-face（ultralytics 单类检测器）。

职责：在一帧 BGR 图像中定位所有人脸，返回带置信度的边界框，并按需外扩裁剪
出用于识别的 RGB/BGR 人脸小图。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class Detection:
    """单个人脸检测结果（像素坐标，左上/右下）。"""

    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def crop(self, frame: np.ndarray, padding: float = config.CROP_PADDING) -> np.ndarray:
        """按外扩比例裁剪人脸区域，并做边界钳位。"""
        h, w = frame.shape[:2]
        pw, ph = int(self.width * padding), int(self.height * padding)
        x1 = max(0, self.x1 - pw)
        y1 = max(0, self.y1 - ph)
        x2 = min(w, self.x2 + pw)
        y2 = min(h, self.y2 + ph)
        return frame[y1:y2, x1:x2]


def resolve_device(device: Optional[str] = None) -> str:
    """在 auto 模式下自动选择 cuda/cpu。"""
    device = device or config.DEVICE
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch 缺失时仍可使用（例如纯 onnxruntime 部署）
        return "cpu"


def _load_yolo(model_path: str):
    """加载 ultralytics YOLO 模型。

    兼容 PyTorch >= 2.6 默认 weights_only=True 导致的旧权重加载失败。
    """
    import torch
    from ultralytics import YOLO

    original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        return YOLO(model_path)
    finally:
        torch.load = original_load


class FaceDetector:
    """YOLOv8-face 推理封装。

    Parameters
    ----------
    model_path : 权重路径，默认 ``config.DETECT_MODEL``
    device : ``"auto"`` / ``"cpu"`` / ``"cuda:0"``
    conf, iou, min_face_size : 检测超参数
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf: float = config.DETECT_CONF,
        iou: float = config.DETECT_IOU,
        imgsz: int = config.DETECT_IMGSZ,
        min_face_size: int = config.MIN_FACE_SIZE,
    ) -> None:
        self.model_path = str(model_path or config.DETECT_MODEL)
        self.device = resolve_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.min_face_size = min_face_size
        self.last_infer_ms = 0.0
        self.model = _load_yolo(self.model_path)
        self.model.to(self.device)

    # ------------------------------------------------------------ 推理
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """检测一帧中的所有人脸，按面积从大到小排序。"""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        import time

        t0 = time.perf_counter()
        results = self.model(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), score in zip(boxes, confs):
                det = Detection(int(x1), int(y1), int(x2), int(y2), float(score))
                if det.width < self.min_face_size or det.height < self.min_face_size:
                    continue
                detections.append(det)

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    def detect_and_crop(
        self, frame: np.ndarray, padding: float = config.CROP_PADDING
    ) -> Tuple[List[Detection], List[np.ndarray]]:
        """检测并返回 (检测框列表, 人脸小图列表)。"""
        detections = self.detect(frame)
        crops = [d.crop(frame, padding) for d in detections]
        return detections, crops

    def detect_largest(
        self, frame: np.ndarray, padding: float = config.CROP_PADDING
    ) -> Tuple[Optional[Detection], Optional[np.ndarray]]:
        """只取画面中最大的人脸（注册、单人核验场景常用）。"""
        detections = self.detect(frame)
        if not detections:
            return None, None
        return detections[0], detections[0].crop(frame, padding)

    # ------------------------------------------------------------ 可视化
    @staticmethod
    def draw(
        frame: np.ndarray,
        detections: Sequence[Detection],
        labels: Optional[Sequence[str]] = None,
        colors: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """在图像副本上绘制人脸框与标签（BGR）。"""
        vis = frame.copy()
        for i, det in enumerate(detections):
            color = (0, 200, 0) if colors is None else colors[i]
            label = f"face {det.conf:.2f}" if labels is None else labels[i]
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(0, det.y1 - th - 8)
            cv2.rectangle(vis, (det.x1, top), (det.x1 + tw + 6, det.y1), color, -1)
            cv2.putText(
                vis, label, (det.x1 + 3, det.y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA,
            )
        return vis
