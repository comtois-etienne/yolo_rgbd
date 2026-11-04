"""Ultralytics compatibility patch for RGB-D NPZ images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

def load_image(file_path: str | Path, mode: str = 'rgbd') -> np.ndarray:
    """Load an RGB-D NPZ or PNG image in the channel order requested by ``mode``.

    :param file_path: path to an NPZ or four-channel PNG image.
    :param mode: channel order, for example ``rgbd``, ``rgb`` or ``depth``.
    :return: image array in the requested channel order.
    """
    if str(file_path).lower().endswith('.npz'):
        rgbd = _load_npz_data(file_path)
    else:
        from PIL import Image

        with Image.open(file_path) as image:
            rgbd = np.asarray(image)

    if rgbd.ndim == 4 and rgbd.shape[0] == 1:
        rgbd = rgbd[0]
    if rgbd.ndim != 3 or rgbd.shape[2] != 4:
        raise ValueError('RGB-D image data must have shape (height, width, 4)')
    if rgbd.dtype != np.uint8 or np.any(rgbd > 255):
        raise ValueError('Each RGB-D image layer must be uint8 with max value 255')

    normalized_mode = 'ddd' if mode == 'depth' else mode
    indexes = {'r': 0, 'g': 1, 'b': 2, 'd': 3}
    try:
        return np.dstack([rgbd[:, :, indexes[channel]] for channel in normalized_mode])
    except KeyError as error:
        raise ValueError(f'Unsupported RGB-D channel mode: {mode!r}') from error


def _load_npz_data(file_path: str | Path) -> np.ndarray:
    with np.load(file_path, allow_pickle=False) as data:
        if 'data' in data.files:
            return data['data']
        missing = {'orthophoto', 'ndsm'} - set(data.files)
        if missing:
            raise ValueError(
                f'NPZ image must contain data or orthophoto and ndsm; '
                f'missing arrays: {sorted(missing)}'
            )
        return np.dstack((data['orthophoto'], data['ndsm']))


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
            return load_image(file_path, color_mode)
        return cv2.original_imread(file_path, *args, **kwargs)

    def fake_pil_open(file_path: str | Path, *args: Any, **kwargs: Any) -> Image.Image:
        if str(file_path).lower().endswith('.npz'):
            image = load_image(file_path, color_mode)
            pil_mode = 'RGBA' if image.shape[2] == 4 else 'RGB'
            result = Image.fromarray(image, mode=pil_mode)
            result.format = 'NPZ'
            return result
        return Image.original_open(file_path, *args, **kwargs)

    IMG_FORMATS.add('npz')
    cv2.imread = fake_cv2_imread
    Image.open = fake_pil_open
