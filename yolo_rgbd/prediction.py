"""Prediction helpers for models trained on RGB or RGB-D NPZ images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
MIN_MASK_SIZE = 100
ImageInput = np.ndarray | str | Path
Prediction = tuple[list[np.ndarray], list[float], list[int]]


def load_yolo_model(model_path: str | Path) -> Any:
    """
    Load an Ultralytics YOLO model from a path.

    :param model_path: path to the YOLO model weights or model definition.
    :return: loaded Ultralytics YOLO model.
    """
    from ultralytics import YOLO

    return YOLO(str(model_path))


def load_npz(file_path: str | Path, mode: str = 'rgbd') -> np.ndarray:
    """
    Load an RGB-D image from an NPZ file.

    :param file_path: path to an NPZ file with ``data`` or ``orthophoto`` and
        ``ndsm``.
    :param mode: channel order, for example ``rgbd``, ``rgb`` or ``depth``.
    :return: uint8 image with shape ``(height, width, 4)``.
    """
    from .monkey_patch import load_image

    return load_image(file_path, mode)


def predict_instances(
    model: Any,
    image: ImageInput,
    confidence: float = CONFIDENCE_THRESHOLD,
) -> Prediction:
    """
    Predict instance masks for an RGB or RGB-D array or NPZ image.

    :param model: injected Ultralytics YOLO model.
    :param image: image array or NPZ path.
    :param confidence: confidence threshold for predictions.
    :return: masks, confidence scores, and class indices.
    """
    source = load_npz(image) if isinstance(image, (str, Path)) else image
    results = model.predict(source=source, conf=confidence)
    if results[0].masks is None:
        return [np.zeros(source.shape[:2], dtype=np.uint16)], [0.0], [0]
    masks = results[0].masks.data.cpu().numpy()
    masks = [_resize_mask(mask, source.shape[:2]) for mask in masks]
    confs = results[0].boxes.conf.cpu().numpy().tolist()
    classes = [int(value) for value in results[0].boxes.cls.cpu().numpy()]
    return masks, confs, classes


def predict_from_array(
    models: Any | list[Any],
    image: np.ndarray,
    pred_mode: str = 'rgbd',
    confidence: float = CONFIDENCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict and merge instance segmentation results for one image.

    :param models: injected YOLO model or models whose predictions are merged.
    :param image: RGB-D or RGB array.
    :param pred_mode: ``rgbd`` to use all channels or ``rgb`` to use RGB only.
    :param confidence: confidence threshold for each model prediction.
    :return: instance labels, semantic labels, and per-pixel confidence.
    """
    model_list = models if isinstance(models, list) else [models]
    source = _select_prediction_channels(image, pred_mode)
    masks: list[np.ndarray] = []
    confs: list[float] = []
    classes: list[int] = []
    for model in model_list:
        prediction = predict_instances(model, source, confidence)
        model_masks, model_confs, model_classes = prediction
        masks.extend(model_masks)
        classes.extend(model_classes)
        confs.extend(_adjust_confidences(model_confs, model_classes, confidence))

    instances, semantics, confidences = clean_prediction(masks, confs, classes)
    confidences[confidences < confidence] *= 1 / confidence
    return instances, semantics, confidences


def _select_prediction_channels(image: np.ndarray, pred_mode: str) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError('image must have shape (height, width, 3 or 4)')
    if pred_mode == 'rgb':
        return image[:, :, :3]
    if pred_mode == 'rgbd' and image.shape[2] == 4:
        return image
    raise ValueError("pred_mode must be 'rgb' for RGB/RGB-D or 'rgbd' for RGB-D")


def _adjust_confidences(
    confidences: list[float],
    classes: list[int],
    threshold: float,
) -> list[float]:
    return [
        confidence * threshold if class_index == 0 else confidence
        for confidence, class_index in zip(confidences, classes)
    ]


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    from scipy.ndimage import zoom

    factors = (shape[0] / mask.shape[0], shape[1] / mask.shape[1])
    return zoom(mask, factors, order=0)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def _nms(masks: list[np.ndarray], scores: list[float]) -> list[int]:
    remaining = sorted(range(len(masks)), key=scores.__getitem__, reverse=True)
    kept: list[int] = []
    while remaining:
        index = remaining.pop(0)
        kept.append(index)
        remaining = [
            other for other in remaining
            if _mask_iou(masks[index], masks[other]) < IOU_THRESHOLD
        ]
    return kept


def clean_prediction(
    masks: list[np.ndarray],
    confs: list[float],
    classes: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge predicted masks into instance, semantic, and confidence arrays.

    :param masks: predicted binary masks.
    :param confs: confidence for each mask.
    :param classes: class index for each mask.
    :return: instance labels, semantic labels, and per-pixel confidence.
    """
    if not masks:
        raise ValueError('At least one prediction mask is required')
    processed = [_largest_component(mask) for mask in masks]
    kept_indices = _nms(processed, confs)
    shape = masks[0].shape
    instances = np.zeros(shape, dtype=np.uint16)
    for index, mask_index in enumerate(kept_indices, start=1):
        instances[processed[mask_index] > 0] = index
    instances = _remove_small_masks(instances)
    semantics = np.zeros(shape, dtype=np.uint16)
    confidence = np.zeros(shape, dtype=np.float32)
    for index, mask_index in enumerate(kept_indices, start=1):
        visible = instances == index
        semantics[visible] = classes[mask_index] + 1
        confidence[visible] = confs[mask_index]
    return _relabel(instances), semantics, confidence


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from skimage.measure import label

    labels = label(mask > 0)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    return labels == int(np.argmax(sizes[1:]) + 1)


def _remove_small_masks(instances: np.ndarray) -> np.ndarray:
    result = instances.copy()
    for index in np.unique(instances):
        if index and np.count_nonzero(instances == index) < MIN_MASK_SIZE:
            result[instances == index] = 0
    return result


def _relabel(instances: np.ndarray) -> np.ndarray:
    result = np.zeros_like(instances)
    for new_index, old_index in enumerate(np.unique(instances)[1:], start=1):
        result[instances == old_index] = new_index
    return result


class ObjectDetector:
    """Small stateful wrapper around an Ultralytics model."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.results: Any = None

    def detect(self, image: np.ndarray, verbose: bool = False, imgsz: int = 640) -> Any:
        """Run detection on an image."""
        self.results = self.model(image, verbose=verbose, imgsz=imgsz)
        return self.results

    def get_objs(self, score: float = 0.5) -> pd.DataFrame:
        """Return detected boxes above a confidence threshold."""
        if self.results is None:
            raise RuntimeError('detect must be called before get_objs')
        rows = []
        for result in self.results:
            for index in range(len(result.boxes.cls)):
                box = result.boxes.xyxy[index].cpu().numpy().astype(int)
                rows.append({
                    'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3],
                    'xcenter': int((box[0] + box[2]) / 2),
                    'score': float(result.boxes.conf[index]),
                    'class': int(result.boxes.cls[index]),
                })
        columns = ['xmin', 'ymin', 'xmax', 'ymax', 'xcenter', 'score', 'class']
        objects = pd.DataFrame(rows, columns=columns)
        return objects[objects['score'] >= score]

    def get_objs_by_class(self, class_index: int, score: float = 0.5) -> pd.DataFrame:
        """Return detected boxes for one numeric class index."""
        objects = self.get_objs(score)
        return objects[objects['class'] == class_index]
