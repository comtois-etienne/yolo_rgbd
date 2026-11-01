#!/bin/bash
#SBATCH --account=ACCOUNT_NAME
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-10:00     # DD-HH:MM:SS


############################# VARIABLES ######
FOLD=1 # 1 to 5, for cross-validation
CLASS_MODE="species" # "instances" or "species"
MODEL_SIZE="l"
HSL_V=0.2

EPOCH=100
BATCH=4
IMGSZ=1984
COLOR_MODE="rgbd" # "rgb" or "rgbd"

#####################################################################

############################## DATASET ##############################

# the source directory should contain a directory named "dataset" with the following files:
# - fold_1.tar
# - fold_2.tar
# - fold_3.tar
# - fold_4.tar
# - fold_5.tar

SOURCEDIR=/path/to/user/directory/aliancecan_training

# Prepare virtualenv
module purge
module load python/3.12.4
module load opencv/4.11.0
module load cuda cudnn

source $SOURCEDIR/venv_ultralytics/bin/activate
echo $VIRTUAL_ENV
pip show numpy


# Prepare data
mkdir $SLURM_TMPDIR/dataset

tar xf $SOURCEDIR/dataset/fold_1.tar -C $SLURM_TMPDIR/dataset
tar xf $SOURCEDIR/dataset/fold_2.tar -C $SLURM_TMPDIR/dataset
tar xf $SOURCEDIR/dataset/fold_3.tar -C $SLURM_TMPDIR/dataset
tar xf $SOURCEDIR/dataset/fold_4.tar -C $SLURM_TMPDIR/dataset
tar xf $SOURCEDIR/dataset/fold_5.tar -C $SLURM_TMPDIR/dataset

tar xf $SOURCEDIR/dataset/yolo-seg.tar -C $SLURM_TMPDIR/dataset


# FOLD dataset
python - $FOLD $CLASS_MODE "$SLURM_TMPDIR/dataset" << 'EOF'

import os
import sys
import shutil


val_fold = int(sys.argv[1])
dataset_mode = sys.argv[2]
dataset_mode = 'instances' if dataset_mode == 'instance' else dataset_mode
base_dir = sys.argv[3]

os.makedirs(os.path.join(base_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'train', 'labels'), exist_ok=True)

os.makedirs(os.path.join(base_dir, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'val', 'labels'), exist_ok=True)

for i in range(1, 6):
    fold_name = f'fold_{i}'
    fold_dir = os.path.join(base_dir, fold_name)
    split = 'val' if i == val_fold else 'train'

    src_images = os.path.join(fold_dir, 'images')
    src_labels = os.path.join(fold_dir, dataset_mode)

    dst_images = os.path.join(base_dir, split, 'images')
    dst_labels = os.path.join(base_dir, split, 'labels')

    for fname in os.listdir(src_images):
        shutil.move(
            os.path.join(src_images, fname),
            os.path.join(dst_images, fname),
        )

    for fname in os.listdir(src_labels):
        shutil.move(
            os.path.join(src_labels, fname),
            os.path.join(dst_labels, fname),
        )

    shutil.rmtree(fold_dir)

EOF



# Prepare YAML file for training
python - "$SLURM_TMPDIR/dataset" $CLASS_MODE $MODEL_SIZE $HSL_V $COLOR_MODE << 'EOF'

def get_header(base_dir, hsv_h=0.2):
    return f"""
path: {base_dir}
train: train/images
val: val/images
test:

hsv_h: {hsv_h} # default: 0.4
"""


def get_species(channels=4):
    return f"""
names:
  0: tree
  1: acpl
  2: acsa
  3: gltr
  4: frpe
  5: ul
  6: pipu
  7: syre
  8: ceoc
  9: quru
  10: tico
  11: ma
  12: po
  13: gi
  14: gy
  15: pn

nc: 17
ch: {channels}
"""


def get_instances(channels=4):
    return f"""
names:
  0: tree

nc: 1
ch: {channels}
"""


def get_names(class_mode, channels):
    if class_mode == 'instance' or class_mode == 'instances':
        return get_instances(channels)
    elif class_mode == 'species':
        return get_species(channels)


architecture = \
"""
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]]

  - [[15, 18, 21], 1, Segment, [nc, 32, 256]]
"""


def get_scales(model_size='n'):
    if model_size == 'n':
        return 'scales:\n  n: [0.33, 0.25, 1024]\n'
    elif model_size == 's':
        return 'scales:\n  s: [0.33, 0.50, 1024]\n'
    elif model_size == 'm':
        return 'scales:\n  m: [0.67, 0.75, 768]\n'
    elif model_size == 'l':
        return 'scales:\n  l: [1.00, 1.00, 512]\n'
    return 'scales:\n  x: [1.00, 1.25, 512]\n'


def get_yaml(dataset_dir, dataset_mode='instances', model_size='n', hsv_h=0.2, channels=4):
    return get_header(dataset_dir, hsv_h) + get_names(dataset_mode, channels) + get_scales(model_size) + architecture


import os
import sys

base_dir = sys.argv[1]
dataset_mode = sys.argv[2]
model_size = sys.argv[3]
hsv_h = float(sys.argv[4])
color_mode = sys.argv[5]
channels = 4 if color_mode == 'rgbd' else 3

yaml = get_yaml(base_dir, dataset_mode, model_size, hsv_h, channels)

with open(os.path.join(base_dir, 'dataset.yaml'), 'w') as f:
    f.write(yaml)

EOF


#####################################################################




# Start training
chmod +x $SOURCEDIR/training.py
python3 $SOURCEDIR/training.py $SLURM_TMPDIR/dataset $SOURCEDIR/runs $CLASS_MODE $MODEL_SIZE $FOLD $EPOCH $BATCH $IMGSZ $COLOR_MODE
