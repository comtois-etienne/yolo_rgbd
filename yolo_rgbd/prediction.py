"""Prediction helpers for models trained on RGB or RGB-D NPZ images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
MIN_MASK_RATIO = 0.006 # 133 -> 10x10 ; 2000 -> 150x150
ImageInput = np.ndarray | str | Path
Prediction = tuple[list[np.ndarray], list[float], list[int]]
MergedPrediction = tuple[np.ndarray, np.ndarray, np.ndarray]
ModelInput = Any | str | Path


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
    from .monkey_patch import read_image

    return read_image(file_path, mode)


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
    array: np.ndarray,
    pred_mode: str = 'rgbd',
    confidence: float = CONFIDENCE_THRESHOLD,
    min_mask_ratio: float = MIN_MASK_RATIO,
) -> MergedPrediction:
    """
    Predict and merge instance segmentation results for one array.

    :param models: injected YOLO model or models whose predictions are merged.
    :param array: RGB-D or RGB array.
    :param pred_mode: ``rgbd`` to use all channels or ``rgb`` to use RGB only.
    :param confidence: confidence threshold for each model prediction.
    :param min_mask_ratio: minimum mask area ratio to retain.
    :return: instance labels, semantic labels, and per-pixel confidence.
    """
    model_list = models if isinstance(models, list) else [models]
    source = _select_prediction_channels(array, pred_mode)
    masks: list[np.ndarray] = []
    confs: list[float] = []
    classes: list[int] = []
    for model in model_list:
        prediction = predict_instances(model, source, confidence)
        model_masks, model_confs, model_classes = prediction
        masks.extend(model_masks)
        classes.extend(model_classes)
        confs.extend(_adjust_confidences(model_confs, model_classes, confidence))

    instances, semantics, confidences = clean_prediction(
        masks, confs, classes, min_mask_ratio
    )
    confidences[confidences < confidence] *= 1 / confidence
    return instances, semantics, confidences


def predict_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    model: ModelInput,
    pred_mode: str = 'rgbd',
    confidence: float = CONFIDENCE_THRESHOLD,
    min_mask_ratio: float = MIN_MASK_RATIO,
) -> None:
    """
    Predict every PNG and NPZ image in a folder.

    :param input_dir: directory containing PNG or NPZ images.
    :param output_dir: directory where compressed prediction files are saved.
    :param model: injected YOLO model or path to a YOLO model.
    :param pred_mode: channel mode passed to ``predict_from_array``.
    :param confidence: confidence threshold passed to ``predict_from_array``.
    :param min_mask_ratio: minimum mask area ratio passed to predictions.
    :return: None.
    """
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    loaded_model = _load_model_input(model)
    for input_path in _image_paths(source_dir):
        output_path = destination_dir / f'{input_path.stem}.npz'
        predict_image(
            input_path,
            output_path,
            loaded_model,
            pred_mode=pred_mode,
            confidence=confidence,
            min_mask_ratio=min_mask_ratio,
        )


def predict_image(
    input_file_path: str | Path,
    output_file_path: str | Path,
    model: ModelInput,
    pred_mode: str = 'rgbd',
    confidence: float = CONFIDENCE_THRESHOLD,
    min_mask_ratio: float = MIN_MASK_RATIO,
) -> None:
    """
    Predict one PNG or NPZ image and save its compressed prediction.

    :param input_file_path: path to the PNG or NPZ input image.
    :param output_file_path: path to the compressed NPZ prediction.
    :param model: injected YOLO model or path to a YOLO model.
    :param pred_mode: channel mode passed to ``predict_from_array``.
    :param confidence: confidence threshold passed to ``predict_from_array``.
    :param min_mask_ratio: minimum mask area ratio passed to predictions.
    :return: None.
    """
    from .monkey_patch import read_image

    image = read_image(input_file_path, pred_mode)
    predictions = predict_from_array(
        _load_model_input(model),
        image,
        pred_mode=pred_mode,
        confidence=confidence,
        min_mask_ratio=min_mask_ratio,
    )
    _write_prediction(output_file_path, predictions)


def _load_model_input(model: ModelInput) -> Any:
    if isinstance(model, (str, Path)):
        return load_yolo_model(model)
    return model


def _image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {'.npz', '.png'}
    )


def _write_prediction(
    output_file_path: str | Path,
    prediction: MergedPrediction,
) -> None:
    instances, semantics, confidences = prediction
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        instance_labels=instances,
        semantic_labels=semantics,
        confidence_labels=confidences,
    )


def read_prediction(
    npz_path: str | Path,
) -> MergedPrediction:
    """
    Read a compressed prediction file.

    :param npz_path: path to a compressed prediction NPZ file.
    :return: instance labels, semantic labels, and confidence labels.
    """
    with np.load(npz_path, allow_pickle=False) as prediction:
        keys = {'instance_labels', 'semantic_labels', 'confidence_labels'}
        missing = keys - set(prediction.files)
        if missing:
            raise ValueError(f'Prediction is missing arrays: {sorted(missing)}')
        return (
            prediction['instance_labels'],
            prediction['semantic_labels'],
            prediction['confidence_labels'],
        )


def _select_prediction_channels(array: np.ndarray, pred_mode: str) -> np.ndarray:
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError('image must have shape (height, width, 3 or 4)')
    if pred_mode == 'depth' and array.shape[2] != 4:
        raise ValueError('depth prediction requires a four-channel RGB-D array')

    if pred_mode == 'depth' or pred_mode == 'ddd':
        d = array[:, :, 3]
        return np.dstack((d, d, d))
    if pred_mode == 'rgb':
        return array[:, :, :3]
    if pred_mode == 'rgbd' and array.shape[2] == 4:
        return array
    raise ValueError("pred_mode must be 'rgb' for RGB/RGB-D or 'rgbd' for RGB-D, or 'depth' for depth-only prediction")


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


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    :param mask_a: np.ndarray, first binary mask
    :param mask_b: np.ndarray, second binary mask
    :return: float, IoU value
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return intersection / union


def nms(masks: list[np.ndarray], confidences: list[float], iou_threshold: float = 0.5) -> list[np.ndarray]:
    """
    Perform Non-Maximum Suppression (NMS) on a list of binary masks based on their IoU.

    :param masks: list of np.ndarray, list of binary masks
    :param confidences: list of float, list of confidence scores corresponding to each mask
    :param iou_threshold: float, IoU threshold for suppression
    :return: list of np.ndarray, list of masks after NMS (from most to least confident)
    """
    ordered_indices = np.argsort(confidences)[::-1] # high to low
    masks = [masks[i] for i in ordered_indices]
    confidences = [confidences[i] for i in ordered_indices]

    kept_masks = []

    while len(masks) > 0:
        mask = masks.pop(0)
        kept_masks.append(mask)
        ious = [iou(mask, m) for m in masks]
        masks = [m for i, m in enumerate(masks) if ious[i] < iou_threshold]

    return kept_masks


def _is_mask_touching_border(mask: np.ndarray, border_width=1, top=True, bottom=True, left=True, right=True) -> bool:
    """
    Check if a binary mask is touching the borders of the array.

    :param mask: 2D numpy array (binary mask)
    :param border_width: int, width of the border to check (default: 1)
    :param top: bool, check top border (default: True)
    :param bottom: bool, check bottom border (default: True)
    :param left: bool, check left border (default: True)
    :param right: bool, check right border (default: True)
    :return: True if the mask touches any of the selected borders, False otherwise
    """
    end = (1 + border_width)
    if top and np.any(mask[0:border_width, :]):
        return True
    if bottom and np.any(mask[-end:-1, :]):
        return True
    if left and np.any(mask[:, 0:border_width]):
        return True
    if right and np.any(mask[:, -end:-1]):
        return True
    return False


def _are_overlapping(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """
    True if the two masks touch eachother by overlapping each other.

    :param mask_a: 2D binary numpy array of the first mask
    :param mask_b: 2D binary numpy array of the second mask
    :return: True if the masks touch eachother, False otherwise
    """
    return np.sum(np.logical_and(mask_a, mask_b)) > 0


from skimage.morphology import disk
from skimage.morphology import dilation as binary_dilation

def _remove_holes(mask: np.ndarray) -> np.ndarray:
    """
    Remove holes of any size from the binary mask.
    The holes are merged with the labeled region.

    :param mask: np.ndarray, binay mask (containing one instance)
    :return: np.ndarray, the same binary mask with all holes removed
    """
    mask = mask.astype(int)
    labeled = label(mask, background=-1)
    mask = mask.astype(bool)

    for v in np.unique(labeled):
        hole = (labeled == v)
        if _is_mask_touching_border(hole):
            continue
        hole_dilated = binary_dilation(hole, disk(1))
        if _are_overlapping(mask, hole_dilated):
            mask = np.logical_or(mask, hole)

    return mask


from skimage.measure import label

def _get_biggest_mask(masks: np.array) -> np.ndarray:
    """
    Gives the biggest mask that is not the background (0).
    Returns empty mask if no masks are found.

    :param masks: split binary mask, or labeled mask
    :return: binary mask of the biggest mask
    """
    masks = label(masks)
    areas = [np.sum(masks == i) for i in range(1, masks.max() + 1)]
    if len(areas) == 0:
        return np.zeros_like(masks, dtype=bool)
    biggest_mask = np.argmax(areas) + 1
    return (masks == biggest_mask)


def clean_prediction(
    masks: list[np.ndarray],
    confs: list[float],
    classes: list[int],
    min_mask_ratio: float = MIN_MASK_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge predicted masks into instance, semantic, and confidence arrays.

    :param masks: predicted binary masks.
    :param confs: confidence for each mask.
    :param classes: class index for each mask.
    :param min_mask_ratio: minimum percentage of image area a mask must cover.
    :return: instance labels, semantic labels, and per-pixel confidence.
    """
    if not masks:
        raise ValueError('At least one prediction mask is required')
    kept_masks = nms(masks, confs, IOU_THRESHOLD)
    kept_indices = [_mask_index(masks, mask) for mask in kept_masks]
    shape = masks[0].shape
    instances = np.zeros(shape, dtype=np.uint16)
    for index, mask_index in enumerate(kept_indices, start=1):
        instances[kept_masks[index - 1] > 0] = index
    instances = _remove_small_masks(instances, min_mask_ratio)
    instances = _clean_instance_parts(instances)

    semantics = np.zeros(shape, dtype=np.uint16)
    confidence = np.zeros(shape, dtype=np.float32)
    for index, mask_index in enumerate(kept_indices, start=1):
        visible = instances == index
        semantics[visible] = classes[mask_index] + 1
        confidence[visible] = confs[mask_index]
    return _relabel(instances), semantics, confidence


def _mask_index(masks: list[np.ndarray], target: np.ndarray) -> int:
    for index, mask in enumerate(masks):
        if mask is target:
            return index
    raise ValueError('NMS returned a mask that was not in the input')


def _clean_instance_parts(instances: np.ndarray) -> np.ndarray:
    cleaned = np.zeros_like(instances)
    for index in np.unique(instances):
        if index == 0:
            continue
        instance = _get_biggest_mask(instances == index)
        cleaned[_remove_holes(instance)] = index
    return cleaned


def _remove_small_masks(instances: np.ndarray, min_mask_ratio: float) -> np.ndarray:
    if not 0 <= min_mask_ratio <= 1:
        raise ValueError('min_mask_ratio must be between 0 and 1')
    minimum_area = instances.size * min_mask_ratio
    result = instances.copy()
    for index in np.unique(instances):
        if index and np.count_nonzero(instances == index) < minimum_area:
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

    def detect(self, array: np.ndarray, verbose: bool = False, imgsz: int = 640) -> Any:
        """Run detection on an image."""
        self.results = self.model(array, verbose=verbose, imgsz=imgsz)
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
