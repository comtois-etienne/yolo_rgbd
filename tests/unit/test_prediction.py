from pathlib import Path

import numpy as np

from yolo_rgbd.prediction import clean_prediction, load_npz


FIXTURE = Path(__file__).parents[1] / 'test_files' / 'fold_1' / 'images'


def test_load_npz_returns_rgbd_image() -> None:
    image_path = next(FIXTURE.glob('*.npz'))

    image = load_npz(image_path)

    assert image.shape == (2000, 2000, 4)
    assert image.dtype == np.uint8


def test_clean_prediction_assigns_instances_and_classes() -> None:
    masks = [np.pad(np.ones((10, 10), dtype=np.uint8), ((2, 0), (2, 0)))]

    instances, semantics, confidences = clean_prediction(masks, [0.8], [2])

    assert instances.max() == 1
    assert np.all(semantics[2:12, 2:12] == 3)
    assert np.all(confidences[2:12, 2:12] == 0.8)
