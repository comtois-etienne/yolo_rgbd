from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yolo_rgbd.prediction import clean_prediction, load_npz, read_prediction
from yolo_rgbd.monkey_patch import read_image


FIXTURE = Path(__file__).parents[1] / 'test_files'
COMBINED_FIXTURE = FIXTURE / '40_299000_5037480 (2026-11-02).npz'


def test_load_npz_rejects_non_uint8_layers() -> None:
    with pytest.raises(ValueError, match='uint8'):
        load_npz(COMBINED_FIXTURE)


def test_load_image_accepts_four_channel_png(tmp_path: Path) -> None:
    expected = np.zeros((4, 5, 4), dtype=np.uint8)
    expected[:, :, 3] = 7
    image_path = tmp_path / 'image.png'
    Image.fromarray(expected, mode='RGBA').save(image_path)

    image = read_image(image_path)

    np.testing.assert_array_equal(image, expected)


def test_load_image_mode_selects_channels(tmp_path: Path) -> None:
    expected = np.zeros((4, 5, 4), dtype=np.uint8)
    expected[:, :, 3] = 7
    image_path = tmp_path / 'image.npz'
    np.savez(image_path, data=expected)

    rgb = read_image(image_path, mode='rgb')
    depth = read_image(image_path, mode='depth')

    assert rgb.shape == (4, 5, 3)
    assert depth.shape == (4, 5, 3)
    assert rgb.dtype == depth.dtype == np.uint8
    np.testing.assert_array_equal(depth[:, :, 0], depth[:, :, 1])
    np.testing.assert_array_equal(depth[:, :, 1], depth[:, :, 2])


def test_clean_prediction_assigns_instances_and_classes() -> None:
    masks = [np.pad(np.ones((10, 10), dtype=np.uint8), ((2, 0), (2, 0)))]

    instances, semantics, confidences = clean_prediction(masks, [0.8], [2])

    assert instances.max() == 1
    assert np.all(semantics[2:12, 2:12] == 3)
    assert np.all(confidences[2:12, 2:12] == 0.8)


def test_read_prediction_returns_saved_arrays(tmp_path: Path) -> None:
    path = tmp_path / 'prediction.npz'
    expected = (
        np.ones((2, 2), dtype=np.uint16),
        np.full((2, 2), 2, dtype=np.uint16),
        np.full((2, 2), 0.5, dtype=np.float32),
    )
    np.savez_compressed(
        path,
        instance_labels=expected[0],
        semantic_labels=expected[1],
        confidence_labels=expected[2],
    )

    result = read_prediction(path)

    for actual, wanted in zip(result, expected):
        np.testing.assert_array_equal(actual, wanted)
