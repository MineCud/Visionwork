"""模型下载脚本：YOLOv8-face 检测权重 + ArcFace w600k_r50 (ONNX) 识别权重。

用法::

    python scripts/download_models.py            # 下载到 models/
    python scripts/download_models.py --check    # 只检查是否就绪
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facerec import config  # noqa: E402

MODELS = (
    {
        "name": "yolov8n-face.pt",
        "dest": config.DETECT_MODEL,
        "urls": (
            "https://hf-mirror.com/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt",
            "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
        ),
    },
    {
        "name": "arcface_w600k_r50.onnx",
        "dest": config.ARCFACE_MODEL,
        "urls": (
            "https://hf-mirror.com/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
        ),
    },
)


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  下载 {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as fh:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done * 100 // total:3d}%  {done // 1048576}MB", end="", flush=True)
        print()
        return True
    except Exception as exc:
        print(f"\n  [失败] {exc}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="下载人脸检测/识别模型")
    parser.add_argument("--check", action="store_true", help="只检查模型是否就绪")
    args = parser.parse_args()

    missing = []
    for model in MODELS:
        dest: Path = model["dest"]
        if dest.exists():
            print(f"[就绪] {model['name']}  {dest.stat().st_size / 1048576:.1f}MB  {dest}")
            continue
        missing.append(model)

    if args.check:
        return 0 if not missing else 1

    for model in missing:
        dest: Path = model["dest"]
        print(f"\n[下载] {model['name']}")
        for url in model["urls"]:
            if download(url, dest):
                print(f"  [完成] {dest}")
                break
        else:
            print(f"  [错误] 所有镜像均失败，请手动放置到 {dest}")

    left = [m["name"] for m in MODELS if not Path(m["dest"]).exists()]
    print("\n全部模型就绪。" if not left else f"\n仍缺少: {left}")
    return 0 if not left else 1


if __name__ == "__main__":
    sys.exit(main())
