# face_mvp · 基于 YOLOv8-face + ArcFace 的课堂人脸识别考勤（核心 MVP）

从课程项目「校园人脸识别考勤系统」中抽出的**核心识别链路**：不含 Web 前端、HTTP 接口和数据库，
只保留 *人脸检测 → 特征提取 → 1:N 检索 → 考勤判定* 这条主线，方便阅读、复现和二次开发。

![demo](docs/images/demo.jpg)

## 目录

- [核心特性](#核心特性)
- [系统结构](#系统结构)
- [快速开始](#快速开始)
- [命令行与 Python API](#命令行与-python-api)
- [配置项](#配置项)
- [实验与指标](#实验与指标)
- [设计说明](#设计说明)
- [已知不足](#已知不足)
- [许可](#许可)

## 核心特性

- **检测**：YOLOv8-face（单类目标检测），支持置信度 / IoU / 输入分辨率 / 最小人脸尺寸过滤。
- **识别**：ArcFace `w600k_r50` ONNX，512 维归一化特征，内积即余弦相似度。
- **检索**：FAISS `IndexFlatIP` 精确检索，未安装 FAISS 时自动退化为 NumPy 矩阵检索。
- **注册**：多帧特征平均 + 二次归一化，单张照片也能注册。
- **考勤**：时段判定（正常/迟到/非签到时段）+ 冷却去重（默认 30 分钟），事件写入 CSV。
- **零框架依赖**：核心只依赖 OpenCV / NumPy / onnxruntime / ultralytics，可作为库被脚本、Notebook 或服务复用。
- **可测**：业务时间通过参数注入，签到时段逻辑可在任意时间点离线回放。

## 系统结构

```
输入帧(BGR)
   │
   ▼
FaceDetector (YOLOv8-face) ──► Detection(x1,y1,x2,y2,conf) + 人脸小图(外扩15%)
   │
   ▼
FaceRecognizer (ArcFace ONNX) ──► 512维特征 (L2 归一化)
   │
   ▼
FaceGallery (FAISS/NumPy) ──► top-1 (student_id, 余弦相似度)
   │
   ├── 相似度 < 阈值 ──► Unknown（不记录）
   ▼
AttendanceTracker ──► 冷却检查 + 时段/迟到判定
   │
   ▼
AttendanceLogger ──► data/logs/attendance.csv + 抓拍
```

```
face_mvp/
├── facerec/                 # 核心包
│   ├── config.py            # 路径、阈值、时段等集中配置
│   ├── detector.py          # YOLOv8-face 检测 + 绘制
│   ├── recognizer.py        # ArcFace 特征提取 / 1:N 识别 / 1:1 核验 / 注册
│   ├── gallery.py           # 特征库（*.npy + index.csv），FAISS/NumPy 双后端
│   ├── attendance.py        # 签到时段、冷却去重、CSV 落盘
│   ├── pipeline.py          # 端到端流水线（检测→识别→考勤→标注）
│   ├── cli.py / __main__.py # python -m facerec
├── scripts/
│   ├── enroll.py            # 人脸注册（图片目录 / 摄像头）
│   └── download_models.py   # 模型下载与就绪检查
├── examples/                # 最小可运行示例
├── tests/test_smoke.py      # 冒烟测试（无需 pytest）
├── docs/                    # 架构说明与效果图
├── data/                    # gallery / sample / snapshots / logs（不入库）
└── models/                  # 权重（不入库）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> CPU 环境即可运行；有 GPU 时把 `onnxruntime` 换成 `onnxruntime-gpu`，并把 `FACEREC_DEVICE` 设为 `cuda:0`。

### 2. 下载模型

```bash
python scripts/download_models.py          # 下载 yolov8n-face.pt 与 arcface_w600k_r50.onnx 到 models/
python scripts/download_models.py --check  # 只检查是否就绪
```

### 3. 注册人脸

```bash
# 从图片目录注册（多帧取平均）
python scripts/enroll.py --student_id 202210040130 --name Huang --class_name CS2401 \
    --images data/sample/202210040130

# 从摄像头注册
python scripts/enroll.py --student_id 202210040130 --name Huang --camera --frames 10
```

### 4. 跑起来

```bash
python -m facerec --source data/sample/202210040130_20260523_085929.jpg
python -m facerec --source data/sample --out outputs/batch      # 批量图片
python -m facerec --source 0                                    # 摄像头实时
python -m facerec --source demo.mp4 --out outputs/demo.mp4      # 视频
```

## 命令行与 Python API

```python
from facerec import FacePipeline

pipe = FacePipeline(save_snapshots=True)
result = pipe.process_image("data/sample/202210040130_20260523_085929.jpg", now=None)

print(result["face_count"], result["latency_ms"])
for event in result["events"]:
    print(event.student_id, event.name, event.status, event.similarity)
```

| 组件 | 主要方法 | 说明 |
|------|----------|------|
| `FaceDetector` | `detect` / `detect_and_crop` / `detect_largest` | 返回 `Detection` 列表，按人脸面积降序 |
| `FaceRecognizer` | `extract` / `identify` / `verify` / `enroll` | 1:N 识别、1:1 核验、多帧平均注册 |
| `FaceGallery` | `add` / `remove` / `search` / `as_dataframe` | 特征库持久化与精确检索 |
| `AttendanceTracker` | `checkin` / `period_at` / `status_at` | 冷却去重与时段判定，时间可注入 |
| `FacePipeline` | `process_frame` / `process_image` / `reset` | 帧流（带节流）与单图两套入口 |

## 配置项

全部集中在 `facerec/config.py`，关键项如下（也可用环境变量覆盖数据/模型目录）：

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `DETECT_CONF` | `0.5` | 人脸检测置信度阈值 |
| `DETECT_IMGSZ` | `640` | YOLO 推理输入边长，480/320 可提速 |
| `MIN_FACE_SIZE` | `40` | 最小人脸边长（像素） |
| `CROP_PADDING` | `0.15` | 人脸框外扩比例，实验证明对精度影响显著 |
| `SIM_THRESHOLD` | `0.40` | 余弦相似度阈值，由 1:1 验证实验确定（FAR≈1%） |
| `COOLDOWN_SECONDS` | `1800` | 同一人重复签到冷却时间 |
| `ATTENDANCE_PERIODS` | 见文件 | 签到时段与迟到阈值 |
| `PROCESS_FPS` | `10` | 视频流最大处理帧率 |

## 实验与指标

在 53 张自采集样本（7 个注册账号 / 5 位真实人物，跨 2 个月、不同光照）上的结果：

| 指标 | 数值 |
|------|------|
| 检测召回率（单人人脸图像） | 53/53 = 100% |
| 检测置信度（均值 / 最低） | 0.882 / 0.644 |
| 1:1 验证 AUC | 0.992 |
| TAR @ FAR=1% | 79.2%（阈值 0.41） |
| TAR @ FAR=0.1% | 61.6%（阈值 0.46） |
| 1:N rank-1（按真实人物） | 46/46 = 100% |
| 1:N rank-1（按注册账号） | 37.0%（同一人被重复注册为 3 个账号） |
| 检测耗时（1280×720, CPU, imgsz=640） | 41.5 ms/帧 |
| 特征提取（112×112, CPU） | 32 ms/张 |
| 端到端（单脸） | 84.3 ms → 11.9 FPS |

> 完整实验（预处理消融、阈值曲线、失败案例）见配套课程报告 Notebook。

## 设计说明

- **为什么去掉数据库**：考勤事件的本质是一条带时间戳的记录，CSV 已足够支撑离线分析；把存储换成 `AttendanceLogger`
  即可无缝替换为 MySQL/Redis，核心识别逻辑不受影响。
- **为什么去掉 Web**：`FacePipeline.process_frame` 输入一帧、输出带标注帧与事件，本身与传输层无关，
  需要时可套 FastAPI/MJPEG 而不改核心代码。
- **为什么保留 FAISS**：注册规模上千后暴力检索仍是 O(N·d)，`IndexFlatIP` 提供统一接口，
  后续可平滑替换为 `IndexIVFFlat`；小规模下 NumPy 后端让环境零依赖。
- **时间注入**：`now` 参数贯穿流线，使「迟到判定」「冷却去重」可在 Notebook 中离线回放与单元测试。

## 已知不足

1. 未做人脸对齐（5 点关键点仿射），仅按检测框裁剪缩放，同人相似度分布偏低。
2. 同一人被重复注册为多个学号时，1:N 会稳定匹配到其中一个，需要在注册环节做查重。
3. 检测与识别逐帧全量执行，多人场景下延迟线性增长，可利用跟踪结果复用特征。
4. 阈值固定，未按光照/清晰度自适应。

## 许可

MIT License，见 [LICENSE](LICENSE)。
