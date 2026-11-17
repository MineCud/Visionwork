"""人脸识别：ArcFace (w600k_r50, ONNX) 特征提取 + 特征库检索。

流程：人脸小图 -> 112×112 -> RGB -> (x-127.5)/128 -> NCHW -> ONNX -> 512 维
特征 -> L2 归一化 -> 与库内特征做内积（余弦相似度）-> 阈值判定身份。
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config
from .gallery import FaceGallery


def _providers() -> List[str]:
    """按优先级选择 onnxruntime 执行后端。"""
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]
    wanted = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    return wanted or ["CPUExecutionProvider"]


class FaceRecognizer:
    """ArcFace 特征提取与身份判定。

    Parameters
    ----------
    model_path : ONNX 模型路径，默认 ``config.ARCFACE_MODEL``
    gallery : 特征库实例；缺省时使用默认目录新建
    threshold : 余弦相似度阈值，默认 ``config.SIM_THRESHOLD``
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        gallery: Optional[FaceGallery] = None,
        threshold: Optional[float] = None,
    ) -> None:
        import onnxruntime as ort

        self.model_path = str(model_path or config.ARCFACE_MODEL)
        self.gallery = gallery if gallery is not None else FaceGallery()
        self.threshold = float(config.SIM_THRESHOLD if threshold is None else threshold)

        self.session = ort.InferenceSession(self.model_path, providers=_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]
        self.last_infer_ms = 0.0

    # ------------------------------------------------------------ 特征提取
    @staticmethod
    def preprocess(face_bgr: np.ndarray) -> np.ndarray:
        """BGR 人脸图 -> ArcFace 输入张量 (1, 3, 112, 112)。"""
        img = cv2.resize(face_bgr, (config.FACE_SIZE, config.FACE_SIZE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = np.transpose((rgb - 127.5) / 128.0, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(blob, dtype=np.float32)

    def extract(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """提取单张人脸的 512 维归一化特征；输入无效时返回 None。"""
        if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
            return None
        if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
            return None
        try:
            t0 = time.perf_counter()
            feature = self.session.run(None, {self.input_name: self.preprocess(face_bgr)})[0][0]
            self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
            feature = np.asarray(feature, dtype=np.float32)
            return feature / (np.linalg.norm(feature) + 1e-8)
        except Exception as exc:  # 单张失败不阻断整条流水线
            print(f"[recognizer] 特征提取失败: {exc}")
            return None

    def extract_batch(self, crops: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
        return [self.extract(c) for c in crops]

    # ------------------------------------------------------------ 检索与判定
    def search(self, feature: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        return self.gallery.search(feature, top_k=top_k)

    def identify(self, face_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        """1:N 识别：返回 (student_id, 相似度)；低于阈值时 student_id 为 None。"""
        feature = self.extract(face_bgr)
        if feature is None:
            return None, 0.0
        candidates = self.search(feature)
        if not candidates:
            return None, 0.0
        student_id, score = candidates[0]
        return (student_id, score) if score >= self.threshold else (None, score)

    def verify(self, face_a: np.ndarray, face_b: np.ndarray) -> Tuple[bool, float]:
        """1:1 核验：判断两张人脸是否同一人。"""
        fa, fb = self.extract(face_a), self.extract(face_b)
        if fa is None or fb is None:
            return False, 0.0
        score = float(np.dot(fa, fb))
        return score >= self.threshold, score

    # ------------------------------------------------------------ 注册
    def enroll(
        self,
        student_id: str,
        crops: Sequence[np.ndarray],
        name: str = "",
        class_name: str = "",
        min_ratio: float = 0.5,
    ) -> Tuple[bool, str]:
        """多帧平均注册：对多张人脸特征取均值并 L2 归一化后入库。"""
        crops = [c for c in crops if c is not None and getattr(c, "size", 0)]
        if not crops:
            return False, "没有可用的人脸图像"

        features = [f for f in self.extract_batch(crops) if f is not None]
        need = max(1, int(np.ceil(len(crops) * min_ratio)))
        if len(features) < need:
            return False, f"有效人脸过少：{len(features)}/{len(crops)}，至少需要 {need} 张"

        mean = np.mean(features, axis=0).astype(np.float32)
        mean /= np.linalg.norm(mean) + 1e-8
        self.gallery.add(student_id, mean, name=name, class_name=class_name, save=True)
        return True, f"注册成功：{student_id}（{len(features)}/{len(crops)} 帧平均）"

    def remove(self, student_id: str) -> bool:
        return self.gallery.remove(student_id)

    def reload(self) -> None:
        """重新从磁盘载入特征库（Web 端注册后热更新用）。"""
        self.gallery.load()

    @property
    def gallery_size(self) -> int:
        return self.gallery.size
