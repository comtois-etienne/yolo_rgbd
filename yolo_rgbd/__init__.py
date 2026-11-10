"""RGB-D YOLO training and prediction utilities."""

from .prediction import (
    ObjectDetector,
    clean_prediction,
    load_yolo_model,
    load_npz,
    predict_image,
    predict_folder,
    predict_from_array,
    predict_instances,
    read_prediction,
)

from .monkey_patch import read_image


def display_prediction(rgbd_array, instances, semantics, confs, colorbar=False) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rgb = rgbd_array[:, :, :3]
    d = rgbd_array[:, :, 3]

    plt.subplots(1, 5, figsize=(20, 10))

    plt.subplot(1, 5, 1)
    plt.title(f'orthophoto')
    plt.imshow(rgb)

    plt.subplot(1, 5, 2)
    max_v = np.max(d)
    plt.title(f'ndsm={max_v:.1f}')
    plt.imshow(d, cmap='terrain')

    plt.subplot(1, 5, 3)
    unique = len(np.unique(instances))
    plt.title(f'instance_labels={unique - 1}')
    plt.imshow(instances, cmap='tab20b', interpolation='nearest')

    plt.subplot(1, 5, 4)
    unique = len(np.unique(semantics)) - 1
    plt.title(f'semantic_labels={unique}')
    plt.imshow(semantics, vmin=0, vmax=20, cmap='tab20', interpolation='nearest')
    plt.colorbar(ticks=range(21), label='semantic class') if colorbar else None

    plt.subplot(1, 5, 5)
    plt.title(f'confidence')
    plt.imshow(confs, vmin=0, vmax=1, cmap='hot', interpolation='nearest')
    plt.colorbar(ticks=[0, 0.5, 1], label='confidence') if colorbar else None

    plt.show()


__all__ = [
    'ObjectDetector',
    'clean_prediction',
    'load_yolo_model',
    'load_npz',
    'predict_image',
    'predict_folder',
    'predict_from_array',
    'predict_instances',
    'read_image',
    'read_prediction',
    'display_prediction',
]
