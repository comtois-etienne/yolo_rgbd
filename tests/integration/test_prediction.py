from pathlib import Path

import numpy as np
import pytest

from yolo_rgbd.prediction import (
    predict_image,
    predict_folder,
    predict_instances,
    read_prediction,
)
from ultralytics import YOLO


pytest.importorskip('ultralytics')

ROOT = Path(__file__).parents[1]
IMAGE = next((ROOT / 'test_files' / 'fold_1' / 'images').glob('*.npz'))
MODEL = ROOT / 'test_files' / 'models' / 'yolo11n_instance_rgbd_256_f1_60e.pt'
NPZ_DIR = ROOT / 'test_files' / 'npz'
RGBA_DIR = ROOT / 'test_files' / 'rgba'


def test_prediction_with_sample_rgbd_model() -> None:
    model = YOLO(str(MODEL))
    masks, confidences, classes = predict_instances(model, IMAGE, confidence=0.1)

    assert len(masks) == len(confidences) == len(classes)
    assert masks[0].shape == (2000, 2000)


def test_predict_file_writes_compressed_prediction(tmp_path: Path) -> None:
    output_path = tmp_path / 'single-output.npz'

    predict_image(NPZ_DIR / '40_298960_5037560 (2026-11-01).npz', output_path, MODEL)

    assert not output_path.with_suffix('.txt').exists()
    with np.load(output_path) as prediction:
        assert set(prediction.files) == {
            'instance_labels',
            'semantic_labels',
            'confidence_labels',
        }
    arrays = read_prediction(output_path)
    assert len(arrays) == 3


def test_predict_folder_preserves_input_stems(tmp_path: Path) -> None:
    output_dir = tmp_path / 'predictions'

    predict_folder(RGBA_DIR, output_dir, MODEL)

    input_stems = {path.stem for path in RGBA_DIR.glob('*.png')}
    output_stems = {path.stem for path in output_dir.glob('*.npz')}
    assert output_stems == input_stems
