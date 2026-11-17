"""全局配置：路径、模型、检测/识别阈值与考勤时段。

所有路径都以项目根目录（face_mvp/）为基准，可用环境变量覆盖，便于在
Notebook、脚本或容器中复用同一套代码。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("FACEREC_DATA_DIR", ROOT_DIR / "data"))
GALLERY_DIR = DATA_DIR / "gallery"      # 人脸特征库（*.npy + index.csv）
SAMPLE_DIR = DATA_DIR / "sample"        # 示例/评测图片
SNAPSHOT_DIR = DATA_DIR / "snapshots"   # 考勤抓拍
LOG_DIR = DATA_DIR / "logs"             # 考勤事件 CSV

MODEL_DIR = Path(os.getenv("FACEREC_MODEL_DIR", ROOT_DIR / "models"))
OUTPUT_DIR = Path(os.getenv("FACEREC_OUTPUT_DIR", ROOT_DIR / "outputs"))

for _d in (GALLERY_DIR, SAMPLE_DIR, SNAPSHOT_DIR, LOG_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DETECT_MODEL = MODEL_DIR / "yolov8n-face.pt"
ARCFACE_MODEL = MODEL_DIR / "arcface_w600k_r50.onnx"

GALLERY_INDEX = GALLERY_DIR / "index.csv"
ATTENDANCE_LOG = LOG_DIR / "attendance.csv"

# ---------------------------------------------------------------- 设备
# "auto" -> 有 CUDA 用 GPU，否则 CPU；也可写 "cpu" / "cuda:0"
DEVICE = os.getenv("FACEREC_DEVICE", "auto")

# ---------------------------------------------------------------- 人脸检测
DETECT_CONF = 0.5       # 置信度阈值：低于该值的候选框被丢弃
DETECT_IOU = 0.45       # NMS IoU 阈值
DETECT_IMGSZ = 640      # 推理输入边长（YOLO 内部 letterbox，越小越快）
MIN_FACE_SIZE = 40      # 最小人脸边长（像素），过滤远处误检
CROP_PADDING = 0.15     # 人脸框外扩比例，保留下巴/额头上下文

# ---------------------------------------------------------------- 人脸识别
FACE_SIZE = 112         # ArcFace 输入尺寸
EMBED_DIM = 512         # 特征维度
# 相似度阈值：由报告 4.2 节的 1:1 验证实验确定（FAR≈1% 时 TAR≈82%）
SIM_THRESHOLD = 0.40

# ---------------------------------------------------------------- 考勤业务
COOLDOWN_SECONDS = 1800  # 同一人重复签到的冷却时间（秒）

# 签到时段；late_after 之后签到记为迟到
ATTENDANCE_PERIODS = (
    {"name": "Morning", "start": "07:30", "end": "08:00", "late_after": "07:45"},
    {"name": "Class-1", "start": "08:00", "end": "08:30", "late_after": "08:10"},
    {"name": "Afternoon", "start": "14:00", "end": "14:30", "late_after": "14:10"},
)

STATUS_NORMAL = "Normal"
STATUS_LATE = "Late"
STATUS_OFF_HOURS = "Off-hours"

# ---------------------------------------------------------------- 流水线
PROCESS_FPS = 10        # 主循环最多处理的帧率，超出则直接回显原帧
