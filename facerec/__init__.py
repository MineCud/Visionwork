"""facerec —— 基于 YOLOv8-face + ArcFace 的人脸识别考勤核心 MVP。

只保留「检测 -> 特征提取 -> 1:N 检索 -> 考勤判定」主链路，
不含 Web 前端、HTTP 接口与数据库，便于阅读、复现与二次开发。

快速上手::

    from facerec import FacePipeline

    pipe = FacePipeline()
    result = pipe.process_image("data/sample/202210040130_20260523_085929.jpg")
    print(result["events"])
"""
from .attendance import AttendanceEvent, AttendanceLogger, AttendanceTracker
from .detector import Detection, FaceDetector
from .gallery import FaceGallery
from .imageio import imread, imread_rgb, imwrite
from .pipeline import FacePipeline
from .recognizer import FaceRecognizer

__version__ = "0.1.0"

__all__ = [
    "Detection",
    "FaceDetector",
    "FaceRecognizer",
    "FaceGallery",
    "imread",
    "imwrite",
    "imread_rgb",
    "AttendanceTracker",
    "AttendanceLogger",
    "AttendanceEvent",
    "FacePipeline",
    "__version__",
]
