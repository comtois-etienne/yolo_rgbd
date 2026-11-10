"""Reusable image-mask and prediction utilities."""

from __future__ import annotations

import numpy as np
from skimage.measure import label
from skimage.morphology import dilation as binary_dilation
from skimage.morphology import disk


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


def remove_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask."""
    labeled = label(mask.astype(int), background=-1)
    result = mask.astype(bool)
    for value in np.unique(labeled):
        hole = labeled == value
        if _is_mask_touching_border(hole):
            continue
        if _are_overlapping(result, binary_dilation(hole, disk(1))):
            result = np.logical_or(result, hole)
    return result


def get_biggest_mask(mask: np.ndarray) -> np.ndarray:
    """Return the biggest connected component of a mask."""
    labeled = label(mask)
    areas = [np.sum(labeled == index) for index in range(1, labeled.max() + 1)]
    if not areas:
        return np.zeros_like(labeled, dtype=bool)
    return labeled == int(np.argmax(areas) + 1)


def clean_instance_parts(instances: np.ndarray) -> np.ndarray:
    """Keep the biggest part and fill holes in every instance."""
    cleaned = np.zeros_like(instances)
    for index in np.unique(instances):
        if index:
            mask = get_biggest_mask(instances == index)
            cleaned[remove_holes(mask)] = index
    return cleaned


def remove_small_masks(instances: np.ndarray, min_mask_ratio: float) -> np.ndarray:
    """Remove instances smaller than a ratio of the image area."""
    if not 0 <= min_mask_ratio <= 1:
        raise ValueError('min_mask_ratio must be between 0 and 1')
    minimum_area = instances.size * min_mask_ratio
    result = instances.copy()
    for index in np.unique(instances):
        if index and np.count_nonzero(instances == index) < minimum_area:
            result[instances == index] = 0
    return result


def relabel(instances: np.ndarray) -> np.ndarray:
    """Relabel non-zero instances consecutively from one."""
    result = np.zeros_like(instances)
    for new_index, old_index in enumerate(np.unique(instances)[1:], start=1):
        result[instances == old_index] = new_index
    return result


def display_prediction(
    rgbd_array: np.ndarray,
    instances: np.ndarray,
    semantics: np.ndarray,
    confidences: np.ndarray,
    colorbar: bool = False,
) -> None:
    """Display an RGB-D image and its prediction arrays."""
    import matplotlib.pyplot as plt

    plt.subplots(1, 5, figsize=(20, 10))
    plt.subplot(1, 5, 1)
    plt.title('orthophoto')
    plt.imshow(rgbd_array[:, :, :3])
    plt.subplot(1, 5, 2)
    plt.title(f'ndsm={np.max(rgbd_array[:, :, 3]):.1f}')
    plt.imshow(rgbd_array[:, :, 3], cmap='terrain')
    plt.subplot(1, 5, 3)
    plt.title(f'instance_labels={len(np.unique(instances)) - 1}')
    plt.imshow(instances, cmap='tab20b', interpolation='nearest')
    plt.subplot(1, 5, 4)
    plt.title(f'semantic_labels={len(np.unique(semantics)) - 1}')
    plt.imshow(semantics, vmin=0, vmax=20, cmap='tab20', interpolation='nearest')
    if colorbar:
        plt.colorbar(ticks=range(21), label='semantic class')
    plt.subplot(1, 5, 5)
    plt.title('confidence')
    plt.imshow(confidences, vmin=0, vmax=1, cmap='hot', interpolation='nearest')
    if colorbar:
        plt.colorbar(ticks=[0, 0.5, 1], label='confidence')
    plt.show()


def _is_mask_touching_border(mask: np.ndarray, border_width: int = 1) -> bool:
    end = 1 + border_width
    return any((
        np.any(mask[:border_width, :]),
        np.any(mask[-end:-1, :]),
        np.any(mask[:, :border_width]),
        np.any(mask[:, -end:-1]),
    ))


def _are_overlapping(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(np.logical_and(mask_a, mask_b).sum())