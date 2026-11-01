"""Ultralytics compatibility patch for RGB-D NPZ images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_npz_image(file_path: str | Path, mode: str = 'rgbd') -> np.ndarray:
    """Load an RGB-D NPZ image in the channel order requested by ``mode``.

    :param file_path: path to an NPZ file containing ``orthophoto`` and ``ndsm``.
    :param mode: channel order, for example ``rgbd``, ``rgb`` or ``depth``.
    :return: image array in the requested channel order.
    """
    with np.load(file_path, allow_pickle=False) as data:
        missing = {'orthophoto', 'ndsm'} - set(data.files)
        if missing:
            raise ValueError(f'NPZ image is missing arrays: {sorted(missing)}')
        rgbd = np.dstack((data['orthophoto'], data['ndsm']))

    normalized_mode = 'ddd' if mode == 'depth' else mode
    indexes = {'r': 0, 'g': 1, 'b': 2, 'd': 3}
    try:
        return np.dstack([rgbd[:, :, indexes[channel]] for channel in normalized_mode])
    except KeyError as error:
        raise ValueError(f'Unsupported RGB-D channel mode: {mode!r}') from error


def apply_monkey_patch(color_mode: str = 'rgbd') -> None:
    """Teach Ultralytics, OpenCV, and Pillow to read RGB-D NPZ images.

    :param color_mode: channel order used by the training dataset.
    """
    from PIL import Image
    import cv2
    from ultralytics.data.base import IMG_FORMATS

    if not hasattr(cv2, 'original_imread'):
        cv2.original_imread = cv2.imread
    if not hasattr(Image, 'original_open'):
        Image.original_open = Image.open

    def fake_cv2_imread(
        file_path: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | None:
        if str(file_path).lower().endswith('.npz'):
            return load_npz_image(file_path, color_mode)
        return cv2.original_imread(file_path, *args, **kwargs)

    def fake_pil_open(file_path: str | Path, *args: Any, **kwargs: Any) -> Image.Image:
        if str(file_path).lower().endswith('.npz'):
            image = load_npz_image(file_path, color_mode)
            pil_mode = 'RGBA' if image.shape[2] == 4 else 'RGB'
            result = Image.fromarray(image, mode=pil_mode)
            result.format = 'NPZ'
            return result
        return Image.original_open(file_path, *args, **kwargs)

    IMG_FORMATS.add('npz')
    cv2.imread = fake_cv2_imread
    Image.open = fake_pil_open
