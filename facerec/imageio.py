"""跨平台 / 中文路径安全的图像读写。

OpenCV 的 ``cv2.imread`` / ``cv2.imwrite`` 在 Windows 上走窄字符接口，
项目路径含中文（如 ``E:\\视觉作业\\...``）时会静默失败并返回 None。
这里统一改用 ``np.fromfile + cv2.imdecode`` 与 ``cv2.imencode + tofile``，
在任何路径下都能工作。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def imread(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """读取图像，失败返回 None（与 cv2.imread 行为一致，但支持中文路径）。"""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, flags)


def imwrite(path: PathLike, image: np.ndarray, quality: int = 95) -> bool:
    """写出图像，自动创建父目录；JPEG 使用给定质量。"""
    if image is None or getattr(image, "size", 0) == 0:
        return False
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ext = target.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        return False
    try:
        buffer.tofile(str(target))
        return True
    except OSError:
        return False


def imread_rgb(path: PathLike) -> Optional[np.ndarray]:
    """读取为 RGB（供 matplotlib 直接显示）。"""
    image = imread(path)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
