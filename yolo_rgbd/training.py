"""Training entry points for RGB/RGB-D Ultralytics segmentation models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def train(
    dataset_yaml: str | Path,
    model: str | Path,
    output_dir: str | Path,
    *,
    color_mode: str = 'rgbd',
    epochs: int = 100,
    batch: int = 4,
    image_size: int = 640,
    name: str = 'yolo_rgbd',
) -> Any:
    """Train a segmentation model from an existing Ultralytics dataset YAML.

    :param dataset_yaml: existing dataset configuration; dataset creation is external.
    :param model: model YAML or weights path.
    :param output_dir: directory for training runs.
    :param color_mode: NPZ channel order used by the dataset.
    :param epochs: number of training epochs.
    :param batch: training batch size.
    :param image_size: training image size.
    :param name: name of the training run.
    :return: Ultralytics training result.
    """
    from ultralytics import YOLO

    from .monkey_patch import apply_monkey_patch

    apply_monkey_patch(color_mode)
    trainer = YOLO(str(model))
    return trainer.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        project=str(output_dir),
        name=name,
        exist_ok=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_yaml', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--color-mode', default='rgbd')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--name', default='yolo_rgbd')
    return parser.parse_args()


def main() -> None:
    """Parse command-line arguments and start training."""
    args = _parse_args()
    train(
        args.dataset_yaml,
        args.model,
        args.output_dir,
        color_mode=args.color_mode,
        epochs=args.epochs,
        batch=args.batch,
        image_size=args.image_size,
        name=args.name,
    )


if __name__ == '__main__':
    main()
