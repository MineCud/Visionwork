"""端到端流水线：检测 -> 识别 -> 去重 -> 考勤事件 -> 可视化。

``FacePipeline`` 是核心 MVP 的门面：一帧图像进，带标注的画面与结构化考勤事件出。
不依赖 Web 框架、数据库或前端，便于在脚本、Notebook、摄像头 Demo 中复用。
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from . import config
from .attendance import AttendanceEvent, AttendanceLogger, AttendanceTracker
from .detector import Detection, FaceDetector
from .imageio import imread, imwrite
from .recognizer import FaceRecognizer


def _ascii_safe(text: str, fallback: str) -> str:
    """OpenCV 默认字体不支持中文，非 ASCII 姓名回退为学号。"""
    return text if text and all(ord(ch) < 128 for ch in text) else fallback


class FacePipeline:
    """人脸识别考勤流水线。

    Parameters
    ----------
    detector, recognizer, tracker, logger : 可注入的组件，缺省按配置新建
    process_fps : 主循环最大处理帧率，超出直接回显（跳帧节流）
    save_snapshots : 是否在签到成功时保存现场抓拍
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        tracker: Optional[AttendanceTracker] = None,
        logger: Optional[AttendanceLogger] = None,
        process_fps: float = config.PROCESS_FPS,
        save_snapshots: bool = False,
        snapshot_dir: Optional[str | Path] = None,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.recognizer = recognizer or FaceRecognizer()
        self.tracker = tracker or AttendanceTracker()
        self.logger = logger or AttendanceLogger()
        self.process_fps = float(process_fps)
        self.save_snapshots = save_snapshots
        self.snapshot_dir = Path(snapshot_dir or config.SNAPSHOT_DIR)

        self._frame_interval = 1.0 / max(1.0, self.process_fps)
        self._last_tick = 0.0
        self.recent_events: List[AttendanceEvent] = []

    # ------------------------------------------------------------ 对外接口
    def process_frame(self, frame: np.ndarray, now: Optional[datetime] = None) -> Dict:
        """处理一帧（带帧率节流），返回可视化结果与本次新增事件。

        节流使用单调时钟，业务时间（时段/冷却/时间戳）使用 ``now``，
        两者分离使得离线回放历史时间线时不受真实帧率影响。
        """
        now = now or datetime.now()
        tick = time.perf_counter()
        if self._last_tick and (tick - self._last_tick) < self._frame_interval:
            return {
                "vis_frame": frame, "detections": [], "labels": [], "events": [],
                "face_count": 0, "latency_ms": 0.0, "throttled": True,
            }
        self._last_tick = tick
        return self._process(frame, now, honor_cooldown=True)

    def process_image(
        self,
        image: Union[np.ndarray, str, Path],
        now: Optional[datetime] = None,
        honor_cooldown: bool = True,
        save_snapshot: bool = False,
    ) -> Dict:
        """处理单张图片（不做帧率节流，默认仍遵守签到冷却）。"""
        if isinstance(image, (str, Path)):
            frame = imread(image)
            if frame is None:
                raise FileNotFoundError(f"无法读取图片: {image}")
        else:
            frame = image
        result = self._process(frame, now or datetime.now(), honor_cooldown, save_snapshot)
        result["throttled"] = False
        return result

    def reset(self) -> None:
        """清空冷却状态与事件缓存（离线批处理/演示时常用）。"""
        self.tracker.reset()
        self.recent_events.clear()
        self._last_tick = 0.0

    def reload_gallery(self) -> None:
        self.recognizer.reload()

    def pop_recent_events(self) -> List[AttendanceEvent]:
        events, self.recent_events = self.recent_events, []
        return events

    # ------------------------------------------------------------ 核心流程
    def _process(
        self,
        frame: np.ndarray,
        now: datetime,
        honor_cooldown: bool = True,
        save_snapshot: Optional[bool] = None,
    ) -> Dict:
        t0 = time.perf_counter()
        detections, crops = self.detector.detect_and_crop(frame)

        labels: List[str] = []
        colors: List[tuple] = []
        events: List[AttendanceEvent] = []
        want_snapshot = self.save_snapshots if save_snapshot is None else save_snapshot

        for det, crop in zip(detections, crops):
            student_id, score = self.recognizer.identify(crop)

            # 1) 库中无人匹配：陌生人
            if student_id is None:
                labels.append(f"Unknown {score:.2f}")
                colors.append((0, 0, 220))
                continue

            meta = self.recognizer.gallery.metadata(student_id)
            display = _ascii_safe(meta.get("name", ""), student_id)

            # 2) 冷却中：识别成功但不重复记录
            if honor_cooldown and not self.tracker.checkin(student_id, now):
                labels.append(f"{display} wait {self.tracker.remaining(student_id, now):.0f}s")
                colors.append((0, 200, 255))
                continue

            # 3) 记一次签到
            period = self.tracker.period_at(now)
            event = AttendanceEvent(
                student_id=student_id,
                name=meta.get("name", ""),
                class_name=meta.get("class_name", ""),
                status=self.tracker.status_at(now),
                period=(period or {}).get("name", ""),
                similarity=float(score),
                timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
                snapshot_path=self._save_snapshot(frame, student_id, now) if want_snapshot else "",
            )
            self.logger.append(event)
            events.append(event)
            self.recent_events.append(event)
            labels.append(f"{display} {event.status} {score:.2f}")
            colors.append((0, 200, 0))

        self.recent_events = self.recent_events[-200:]
        vis = FaceDetector.draw(frame, detections, labels, colors)
        cv2.putText(
            vis, now.strftime("%Y-%m-%d %H:%M:%S"), (12, vis.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )
        return {
            "vis_frame": vis,
            "detections": detections,
            "labels": labels,
            "events": events,
            "face_count": len(detections),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "throttled": False,
        }

    # ------------------------------------------------------------ 抓拍
    def _save_snapshot(self, frame: np.ndarray, student_id: str, now: datetime) -> str:
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = self.snapshot_dir / f"{student_id}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            imwrite(path, frame)
            return str(path)
        except Exception:
            return ""
