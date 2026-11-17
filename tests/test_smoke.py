"""冒烟测试：不依赖 pytest，也可被 pytest 收集。

运行::

    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from facerec import config  # noqa: E402
from facerec.attendance import AttendanceTracker  # noqa: E402
from facerec.gallery import FaceGallery  # noqa: E402

EPS = 1e-6


def test_gallery_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gallery = FaceGallery(tmp, dim=8)
        vec = np.random.RandomState(0).randn(8).astype(np.float32)
        gallery.add("S001", vec, name="Zhang", class_name="CS2401")
        assert gallery.size == 1 and "S001" in gallery

        reloaded = FaceGallery(tmp, dim=8)
        assert reloaded.size == 1
        assert reloaded.name_of("S001") == "Zhang"
        top = reloaded.search(vec, top_k=1)
        assert top and top[0][0] == "S001" and abs(top[0][1] - 1.0) < 1e-4

        assert reloaded.remove("S001") is True
        assert FaceGallery(tmp, dim=8).size == 0


def test_gallery_prefers_faiss_when_available() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gallery = FaceGallery(tmp, dim=8)
        gallery.add("A", np.ones(8, dtype=np.float32))
        gallery.add("B", np.eye(8, dtype=np.float32)[0])
        assert gallery.backend in {"faiss", "numpy"}
        assert gallery.search(np.ones(8, dtype=np.float32))[0][0] == "A"


def test_tracker_cooldown_and_periods() -> None:
    tracker = AttendanceTracker(cooldown_seconds=1800)
    t0 = datetime(2026, 3, 19, 7, 40, 0)   # Morning 时段，未迟到
    assert tracker.period_at(t0)["name"] == "Morning"
    assert tracker.status_at(t0) == "Normal"
    assert tracker.status_at(datetime(2026, 3, 19, 7, 50, 0)) == "Late"

    assert tracker.checkin("S001", t0) is True
    assert tracker.checkin("S001", t0 + timedelta(seconds=60)) is False
    assert tracker.checkin("S001", t0 + timedelta(seconds=1801)) is True
    assert tracker.status_at(datetime(2026, 3, 19, 12, 0, 0)) == "Off-hours"
    assert abs(tracker.remaining("S002", t0)) < EPS


def test_detector_and_pipeline() -> None:
    if not config.DETECT_MODEL.exists() or not config.ARCFACE_MODEL.exists():
        print("  [跳过] 模型缺失，先运行 python scripts/download_models.py")
        return
    avatar = config.TESTSET_DIR / "P1_01.jpg"
    probe = config.TESTSET_DIR / "P1_02.jpg"
    if not avatar.exists() or not probe.exists():
        print("  [跳过] 测试集图片缺失")
        return

    import cv2

    from facerec.detector import FaceDetector
    from facerec.imageio import imread
    from facerec.pipeline import FacePipeline
    from facerec.recognizer import FaceRecognizer

    detector = FaceDetector()
    det, crop = detector.detect_largest(imread(avatar))
    assert det is not None and crop is not None and crop.size > 0

    with tempfile.TemporaryDirectory() as tmp:
        gallery = FaceGallery(tmp)
        recognizer = FaceRecognizer(gallery=gallery, threshold=0.30)
        ok, _ = recognizer.enroll("P1", [crop], name="Huang", class_name="CS2401")
        assert ok

        pipeline = FacePipeline(detector=detector, recognizer=recognizer)
        result = pipeline.process_image(probe, now=datetime(2026, 3, 19, 7, 40, 0))
        assert result["face_count"] >= 1
        assert len(result["events"]) == 1
        event = result["events"][0]
        assert event.student_id == "P1"
        assert event.status in {"Normal", "Late", "Off-hours"}
        assert event.similarity > 0.30


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for func in tests:
        try:
            func()
            print(f"[PASS] {func.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {func.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
