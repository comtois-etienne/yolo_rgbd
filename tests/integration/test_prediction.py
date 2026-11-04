from pathlib import Path

import pytest

from yolo_rgbd.prediction import predict_instances
from ultralytics import YOLO


pytest.importorskip('ultralytics')

ROOT = Path(__file__).parents[1]
IMAGE = next((ROOT / 'test_files' / 'fold_1' / 'images').glob('*.npz'))
MODEL = ROOT / 'test_files' / 'models' / 'yolo11n_instance_rgbd_256_f1_60e.pt'


def test_prediction_with_sample_rgbd_model() -> None:
    model = YOLO(str(MODEL))
    masks, confidences, classes = predict_instances(model, IMAGE, confidence=0.1)

    assert len(masks) == len(confidences) == len(classes)
    assert masks[0].shape == (2000, 2000)
