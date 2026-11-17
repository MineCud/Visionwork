"""摄像头实时 Demo：检测框 + 身份 + 考勤状态实时显示。

运行::

    python examples/webcam_demo.py            # 默认摄像头 0
    python examples/webcam_demo.py --index 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from facerec import FacePipeline  # noqa: E402
from facerec import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="摄像头实时人脸签到 Demo")
    parser.add_argument("--index", type=int, default=0, help="摄像头序号")
    parser.add_argument("--threshold", type=float, default=config.SIM_THRESHOLD)
    parser.add_argument("--gallery-dir", default=str(config.GALLERY_DIR))
    args = parser.parse_args()

    from facerec import FaceGallery, FaceRecognizer

    gallery = FaceGallery(args.gallery_dir)
    if gallery.size == 0:
        print("特征库为空，请先运行 scripts/enroll.py 注册人脸。")
        return 1
    print(f"特征库 {gallery.size} 人，阈值 {args.threshold}，按 q 退出。")

    pipeline = FacePipeline(
        recognizer=FaceRecognizer(gallery=gallery, threshold=args.threshold),
        save_snapshots=True,
    )

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        print(f"无法打开摄像头 index={args.index}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process_frame(frame)
            for event in result["events"]:
                print(f"[签到] {event.timestamp} {event.student_id} {event.name} "
                      f"{event.status} 相似度={event.similarity:.3f}")
            cv2.imshow("facerec webcam - press q", result["vis_frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
