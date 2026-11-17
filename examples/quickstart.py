"""最小示例：注册一张人脸 + 识别一张图片。

运行::

    python examples/quickstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from facerec import FaceDetector, FaceGallery, FacePipeline, FaceRecognizer  # noqa: E402
from facerec.imageio import imread, imwrite  # noqa: E402
from facerec import config  # noqa: E402

GALLERY_DIR = config.OUTPUT_DIR / "quickstart_gallery"
AVATAR = config.TESTSET_DIR / "P1_01.jpg"
PROBE = config.TESTSET_DIR / "P1_02.jpg"


def main() -> int:
    if not AVATAR.exists() or not PROBE.exists():
        print(f"示例图片缺失，请确认 {config.TESTSET_DIR} 中存在 P1_01.jpg / P1_02.jpg")
        return 1

    detector = FaceDetector()
    gallery = FaceGallery(GALLERY_DIR)
    recognizer = FaceRecognizer(gallery=gallery)

    # 1) 注册：从一张证件照裁剪人脸并取特征
    det, crop = detector.detect_largest(imread(AVATAR))
    ok, message = recognizer.enroll("P1", [crop], name="Huang", class_name="CS2401")
    print(message)

    # 2) 识别：对现场照片做 1:N 检索
    pipeline = FacePipeline(detector=detector, recognizer=recognizer)
    result = pipeline.process_image(PROBE)
    print(f"人脸数={result['face_count']} 耗时={result['latency_ms']:.1f}ms")
    for event in result["events"]:
        print(f"  识别为 {event.student_id} {event.name} 相似度={event.similarity:.3f} "
              f"状态={event.status} 时段={event.period}")

    out = config.OUTPUT_DIR / "quickstart_annotated.jpg"
    imwrite(out, result["vis_frame"])
    print(f"可视化结果: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
