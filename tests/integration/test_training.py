from pathlib import Path
import shutil

import pytest

from yolo_rgbd.training import train


pytest.importorskip('ultralytics')

ROOT = Path(__file__).parents[1]
IMAGE = next((ROOT / 'test_files' / 'fold_1' / 'images').glob('*.npz'))
MODEL = ROOT / 'test_files' / 'models' / 'yolo11n_instance_rgbd_256_f1_60e.pt'
LABEL = ROOT / 'test_files' / 'fold_1' / 'instances' / f'{IMAGE.stem}.txt'


def test_training_with_sample_dataset(tmp_path: Path) -> None:
    image_dir = tmp_path / 'images'
    label_dir = tmp_path / 'labels'
    image_dir.mkdir()
    label_dir.mkdir()
    shutil.copy2(IMAGE, image_dir / IMAGE.name)
    shutil.copy2(LABEL, label_dir / LABEL.name)
    dataset = tmp_path / 'dataset.yaml'
    dataset.write_text(
        f'path: {tmp_path}\n'
        'train: images\n'
        'val: images\n'
        'names:\n  0: tree\nnc: 1\nch: 4\n'
    )

    result = train(
        dataset,
        MODEL,
        tmp_path / 'runs',
        epochs=1,
        batch=1,
        image_size=256,
    )

    assert result is not None
