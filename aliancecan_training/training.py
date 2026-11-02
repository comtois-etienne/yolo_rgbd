#!~/venv_ultralytics/bin/env python

"""
Dependencies:
- ultralytics
"""


import datetime
import sys
import os



########### Training Variables ###########

source_dataset = sys.argv[1]
runs_dir = sys.argv[2]
class_mode = sys.argv[3]  # 'instance' or 'species'
model_size = sys.argv[4]  # 'n' or 's' or 'm' or 'l' or 'x'
fold = int(sys.argv[5])
epoch = int(sys.argv[6])  # number of epochs
batch = int(sys.argv[7])
imgsz = int(sys.argv[8])  # image size
color_mode = sys.argv[9]

# epoch = 100 #todo remove

COLOR_MODE = color_mode  # 'rgbd' or 'rgb' or 'depth'
VALIDATION = False # YOLO validation phase
PLOTTING = False # whether to plot images during training

# DATASET_NAME = f'dataset_1250_{tile_size}m_{CLASS_MODE}'
model_name = f'yolo11{model_size}'
MODEL = f'{source_dataset}/yolo-seg/{model_name}-seg.pt'
FOLD_NO = fold

EPOCHS = epoch
IMGSZ = imgsz
BATCH_SIZE = batch
SAVE_PERIOD = 0

# DROP_DEPTH = 0.00 # probability to drop depth channel during training for rgbd mode
# DROP_RGB = 0.00

start_time = datetime.datetime.now().strftime("%Y%m%d %Hh%M")
# YAML = f'{source_dataset}/{DATASET_NAME}/cv/fold_{FOLD_NO}/tree-{COLOR_MODE}.yaml'
YAML = f'{source_dataset}/dataset.yaml'
run_name = f'{model_name}_{class_mode}_{COLOR_MODE}_{IMGSZ}_f{FOLD_NO}_{EPOCHS}e_({start_time})'

log_path = f'{runs_dir}/{run_name}/console.log'
os.makedirs(os.path.dirname(log_path), exist_ok=True)
log_file = open(log_path, 'w')
sys.stdout = log_file


########### RGBD Monkey Patching ###########
"""
Optional, for viewing images during training
in ultralytics.utils.plotting.py
change original line in function `plot_images` :
`mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0) # Original`

to this line :
`mosaic[y : y + h, x : x + w, :] = images[i][[3,2,1]].transpose(1, 2, 0) # Modified`
"""

import numpy as np
from ultralytics.data.base import IMG_FORMATS
from ultralytics import YOLO # ensure YOLO is imported before monkey patching
from PIL import Image
import cv2
import scipy.stats


def _get_tile_mean_std() -> dict:
    """
    Based on 1000 tiles
    """
    return {
        'r_mean': 68.806,
        'r_std': 45.867,
        'g_mean': 91.519,
        'g_std': 42.803,
        'b_mean': 50.144,
        'b_std': 44.72,
        'd_mean': 4.800,
        'd_std': 5.030
    }


def _my_distribution(mean, std, min_val=0, max_val=255):
    # from `https://stackoverflow.com/questions/50626710/generating-random-numbers-with-predefined-mean-std-min-and-max`
    scale = max_val - min_val
    location = min_val
    unscaled_mean = (mean - min_val) / scale
    unscaled_var = (std / scale) ** 2
    t = unscaled_mean / (1 - unscaled_mean)
    beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
    alpha = beta * t
    return scipy.stats.beta(alpha, beta, scale=scale, loc=location)


def _random_layer(mean, std, size=128):
    dist = _my_distribution(mean, std)
    layer = dist.rvs(size=(size, size)).astype(np.uint8)
    layer = np.clip(layer, 0, 255)
    return layer


def random_tile(size=128):
    """
    :param size: int, size of the tile (size x size)
    """
    mean_std = _get_tile_mean_std()
    r = _random_layer(mean_std['r_mean'], mean_std['r_std'], size)
    g = _random_layer(mean_std['g_mean'], mean_std['g_std'], size)
    b = _random_layer(mean_std['b_mean'], mean_std['b_std'], size)
    d = _random_layer(mean_std['d_mean']*10, mean_std['d_std']*10, size)
    return np.dstack((r, g, b, d))


def convert(rgbd: np.ndarray, mode='rgbd'):
    """
    :param rgbd: np.ndarray, must be in the rgbd order
    :param mode: string, defines the order of the layers (rgb, bgr, ddd, bgrd...)
    :return: np.nparray, the reordered array
    """
    indexes = { 'r' : 0, 'g' : 1, 'b' : 2, 'd' : 3 }
    arrays = [rgbd[:,:,(indexes[m])] for m in list(mode)]
    return np.dstack(arrays)


def open_tile_npz(npz_path: str, mode: str) -> np.ndarray:
    """
    Opens a tile that has been saved as a compressed numpy array.  
    The npz dtype must be uint8 and contain 'orthophoto' and 'ndsm' arrays.  

    :param npz_path: str, path to the tile
    :param mode: str, 'bgrd', 'rgbd', 'rgb', 'bgr', 'depth', ...
    :return: np.ndarray, the tile as a numpy array
    """
    mode = 'ddd' if mode == 'depth' else mode
    npz = np.load(npz_path, allow_pickle=True)
    rgbd = np.dstack((npz['orthophoto'], npz['ndsm']))
    return convert(rgbd, mode)


if cv2.__dict__.get('original_imread', None) is None:
    print("Monkey Patching 🐒 cv2.imread to add .npz tiles support for YOLO RGBD")
    cv2.original_imread = cv2.imread


if Image.__dict__.get('original_open', None) is None:
    print("Monkey Patching 🐒 PIL.Image.open to add .npz tiles support for YOLO RGBD")
    Image.original_open = Image.open


def fake_cv2_imread(file_path, *args, **kwargs):
    """
    This function open the tiles during the training process.  
    cv2.imread replacement to open .npz tiles as rgbd or rgb images.
    cv2 uses BGR color mode by default.
    """
    if isinstance(file_path, str) and file_path.endswith(".npz"):
        tile = open_tile_npz(file_path, COLOR_MODE)
        # rand = np.random.rand()
        # if COLOR_MODE == 'rgbd' and rand < (DROP_DEPTH + DROP_RGB):
        #     fake_tile = random_tile(size=tile.shape[0])
        #     if rand < DROP_DEPTH: tile[:, :, 3] = fake_tile[:, :, 3]
        #     else: tile[:, :, 0:3] = fake_tile[:, :, 0:3]
        return tile
    else:
        return cv2.original_imread(file_path, *args, **kwargs)


def fake_pil_open(file_path):
    """
    This function open the tiles during the dataset integrity check.  
    PIL.Image.open replacement to open .npz tiles as rgbd or rgb images. 
    PIL uses RGB color mode by default.
    """
    if file_path[-3:] == 'npz':
        rgbd = open_tile_npz(file_path, COLOR_MODE)
        mode = 'RGBA' if len(COLOR_MODE) == 4 else 'RGB'
        pil_im = Image.fromarray(rgbd, mode=mode)
        pil_im.format = 'NPZ'
        return pil_im
    else:
        return Image.original_open(file_path)


IMG_FORMATS.add("npz")
cv2.imread = fake_cv2_imread
Image.open = fake_pil_open




########### Training ###########

print(f'{MODEL=}')
print(f'{COLOR_MODE=}')
print(f'{YAML=}')
print(f'{EPOCHS=}')
print(f'{IMGSZ=}')
print(f'{BATCH_SIZE=}')
print(f'{SAVE_PERIOD=}')
with open(YAML, 'r') as f:
    print('*'*20)
    print(f.read())
    print('*'*20)

# print(f'{DROP_DEPTH=}')
# print(f'{DROP_RGB=}')

model = YOLO(YAML).load(MODEL)
results = model.train(
    data=YAML, 
    epochs=EPOCHS, 
    imgsz=IMGSZ, 
    batch=BATCH_SIZE, 
    save_period=SAVE_PERIOD,
    project=runs_dir,
    name=run_name,
    plots=PLOTTING,
    val=VALIDATION,
    exist_ok=True, # overwrite existing run with same name
)
print(results.results_dict)
print(results.fitness)

log_file.close()

