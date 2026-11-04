# yolo_rgbd

Utilities for training and predicting with Ultralytics segmentation models on
RGB-D images stored as NumPy `.npz` files. This project deliberately does not
read geotiffs or create datasets from tiles; those responsibilities belong to
[`pydsm`](https://github.com/comtois-etienne/pydsm).

## Installation

The code requires Python 3.10 or newer and **Ultralytics 8.3.40 exactly**.

### pip virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### conda environment

```bash
conda create -n yolo_rgbd python=3.12
conda activate yolo_rgbd
python -m pip install -e .
```

The same environment is used for training and prediction. GPU-enabled PyTorch
can be installed according to the CUDA version available on the machine before
installing this project.

## NPZ format

Each NPZ image file must contain either one `data` array or two separate arrays:

* `data`: an `H x W x 4` RGB-D `uint8` array, in RGB-D channel order
* `orthophoto`: an `H x W x 3` RGB `uint8` array, together with
  `ndsm`: an `H x W` depth `uint8` array

Four-channel PNG images are also supported. All four layers must be `uint8`
with values no greater than 255; values are not normalized during loading.

For the compatibility with the sample model in the tests files, the height in meters of the ndsm is multiplied by 10. The maximum height is 25.5m and is represented by 255 (int). 

Labels are ordinary Ultralytics text files next to the images. Dataset and
label creation is provided by `pydsm`; pass the resulting `dataset.yaml` to
the training command.

## Training

```bash
yolo-rgbd-train dataset.yaml path/to/model.yaml runs \
  --color-mode rgbd --epochs 100 --batch 4 --image-size 640
```

`yolo_rgbd.training.train()` is the equivalent Python API. It applies the
required OpenCV/Pillow/Ultralytics monkey patch before training so `.npz`
images are accepted by the data loader.

## Prediction

```python
from yolo_rgbd.prediction import predict_instances

masks, confidences, classes = predict_instances(
    'runs/segment/weights/best.pt',
    'path/to/image.npz',
)
```

Prediction also accepts an `H x W x 3` or `H x W x 4` NumPy array. Use
`clean_prediction()` to merge model masks into instance, semantic, and
confidence arrays.

## Tests

```bash
python -m pytest
```

The integration tests use the fixtures under `tests/test_files`, including the
sample RGB-D models. They are skipped when the optional Ultralytics/PyTorch
runtime is unavailable.
