"""人脸注册脚本：从图片目录或摄像头采集多帧，取平均特征写入特征库。

用法::

    python scripts/enroll.py --student_id 202210040130 --name Huang --class_name CS2401 \
        --images data/sample/202210040130

    python scripts/enroll.py --student_id 202210040130 --name Huang --camera --frames 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from facerec import config  # noqa: E402
from facerec.detector import FaceDetector  # noqa: E402
from facerec.gallery import FaceGallery  # noqa: E402
from facerec.recognizer import FaceRecognizer  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_from_images(detector: FaceDetector, folder: Path, limit: int | None = None) -> List:
    """从图片目录中裁剪每张图里最大的人脸。"""
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if limit:
        files = files[:limit]
    crops = []
    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  [跳过] 无法读取 {path.name}")
            continue
        det, crop = detector.detect_largest(frame)
        if crop is None:
            print(f"  [跳过] 未检测到人脸 {path.name}")
            continue
        print(f"  [采集] {path.name}  conf={det.conf:.2f}  size={det.width}x{det.height}")
        crops.append(crop)
    return crops


def collect_from_camera(detector: FaceDetector, index: int, target: int, interval: float = 0.3) -> List:
    """实时采集：每 interval 秒抓一帧最大人脸，按 q 提前结束。"""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头 index={index}")
    crops, last_t = [], 0.0
    print(f"采集 {target} 帧人脸，按 q 结束。")
    try:
        while len(crops) < target:
            ok, frame = cap.read()
            if not ok:
                break
            det, crop = detector.detect_largest(frame)
            vis = frame.copy()
            cv2.putText(vis, f"collected {len(crops)}/{target}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if crop is not None:
                cv2.rectangle(vis, det.box, (0, 255, 0), 2)
                now = time.time()
                if now - last_t >= interval:
                    crops.append(crop)
                    last_t = now
            cv2.imshow("enroll - press q to quit", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return crops


def main() -> int:
    parser = argparse.ArgumentParser(description="注册人脸到特征库")
    parser.add_argument("--student_id", required=True, help="学号（特征库主键）")
    parser.add_argument("--name", default="", help="姓名")
    parser.add_argument("--class_name", default="", help="班级")
    parser.add_argument("--images", default=None, help="图片目录")
    parser.add_argument("--camera", action="store_true", help="使用摄像头采集")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=10, help="摄像头采集帧数")
    parser.add_argument("--limit", type=int, default=None, help="图片目录最多使用张数")
    parser.add_argument("--gallery-dir", default=str(config.GALLERY_DIR))
    args = parser.parse_args()

    if not args.images and not args.camera:
        parser.error("需要指定 --images 或 --camera")

    detector = FaceDetector()
    if args.camera:
        crops = collect_from_camera(detector, args.camera_index, args.frames)
    else:
        folder = Path(args.images)
        if not folder.is_dir():
            raise SystemExit(f"图片目录不存在: {folder}")
        crops = collect_from_images(detector, folder, args.limit)

    print(f"\n共采集 {len(crops)} 张有效人脸")
    if not crops:
        return 1

    gallery = FaceGallery(args.gallery_dir)
    recognizer = FaceRecognizer(gallery=gallery)
    ok, message = recognizer.enroll(args.student_id, crops, name=args.name, class_name=args.class_name)
    print(message)
    if not ok:
        return 1

    print(f"\n特征库当前 {gallery.size} 人（backend={gallery.backend}，目录={gallery.dir}）")
    print(gallery.as_dataframe().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
