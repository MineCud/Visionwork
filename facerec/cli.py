"""命令行入口：单张图片 / 图片目录 / 视频文件 / 摄像头。

用法示例::

    python -m facerec --source data/sample/202210040130_20260523_085929.jpg
    python -m facerec --source data/sample --out outputs/batch
    python -m facerec --source demo.mp4 --out outputs/demo_annotated.mp4
    python -m facerec --source 0            # 摄像头
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import cv2

from . import config
from .attendance import AttendanceEvent
from .imageio import imread, imwrite
from .pipeline import FacePipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _print_events(events: List[AttendanceEvent]) -> None:
    for ev in events:
        print(f"  [签到] {ev.timestamp}  {ev.student_id} {ev.name} "
              f"状态={ev.status} 时段={ev.period} 相似度={ev.similarity:.3f}")


def run_image(pipe: FacePipeline, path: Path, out: Path) -> None:
    result = pipe.process_image(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out, result["vis_frame"])
    print(f"[图片] {path.name} 人脸数={result['face_count']} 耗时={result['latency_ms']:.1f}ms -> {out}")
    _print_events(result["events"])


def run_directory(pipe: FacePipeline, folder: Path, out_dir: Path) -> None:
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[目录] 未找到图片: {folder}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    all_events: List[AttendanceEvent] = []
    t0 = time.perf_counter()
    for path in images:
        result = pipe.process_image(path)
        imwrite(out_dir / f"{path.stem}_annotated.jpg", result["vis_frame"])
        all_events.extend(result["events"])
    elapsed = time.perf_counter() - t0
    print(f"[目录] 处理 {len(images)} 张，总耗时 {elapsed:.2f}s，"
          f"平均 {elapsed / len(images) * 1000:.1f}ms/张，签到 {len(all_events)} 次")
    _print_events(all_events)


def run_video(pipe: FacePipeline, path: Path, out: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    events: List[AttendanceEvent] = []
    frames = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipe.process_image(frame)  # 离线视频：不按真实时间节流
            writer.write(result["vis_frame"])
            events.extend(result["events"])
            frames += 1
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t0
    print(f"[视频] {frames} 帧，耗时 {elapsed:.2f}s，处理速度 {frames / max(elapsed, 1e-6):.1f} FPS -> {out}")
    _print_events(events)


def run_camera(pipe: FacePipeline, index: int) -> None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头 index={index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("摄像头已启动，按 q 退出。")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipe.process_frame(frame)
            _print_events(result["events"])
            cv2.imshow("facerec - press q to quit", result["vis_frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8-face + ArcFace 人脸识别考勤 Demo")
    parser.add_argument("--source", required=True,
                        help="图片/图片目录/视频路径，或摄像头序号（0）")
    parser.add_argument("--out", default=None, help="输出路径（图片、视频或目录）")
    parser.add_argument("--threshold", type=float, default=config.SIM_THRESHOLD,
                        help=f"相似度阈值，默认 {config.SIM_THRESHOLD}")
    parser.add_argument("--gallery-dir", default=str(config.GALLERY_DIR), help="特征库目录")
    parser.add_argument("--fps", type=float, default=config.PROCESS_FPS, help="摄像头模式处理帧率")
    parser.add_argument("--save-snapshots", action="store_true", help="签到成功时保存抓拍")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .gallery import FaceGallery
    from .recognizer import FaceRecognizer

    gallery = FaceGallery(args.gallery_dir)
    if gallery.size == 0:
        print("提示：特征库为空，请先运行 scripts/enroll.py 注册人脸。")
        print("      python scripts/enroll.py --student_id S001 --name ZhangSan --images data/sample")

    pipe = FacePipeline(
        recognizer=FaceRecognizer(gallery=gallery, threshold=args.threshold),
        process_fps=args.fps,
        save_snapshots=args.save_snapshots,
    )

    source = args.source
    if source.isdigit():
        run_camera(pipe, int(source))
        return 0

    path = Path(source)
    if not path.exists():
        raise SystemExit(f"路径不存在: {path}")

    if path.is_dir():
        run_directory(pipe, path, Path(args.out or config.OUTPUT_DIR / "batch"))
    elif path.suffix.lower() in VIDEO_EXTS:
        run_video(pipe, path, Path(args.out or config.OUTPUT_DIR / f"{path.stem}_annotated.mp4"))
    elif path.suffix.lower() in IMAGE_EXTS:
        run_image(pipe, path, Path(args.out or config.OUTPUT_DIR / f"{path.stem}_annotated.jpg"))
    else:
        raise SystemExit(f"不支持的文件类型: {path.suffix}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
