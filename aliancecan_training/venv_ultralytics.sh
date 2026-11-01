#!/bin/bash

module load python/3.12.4
module load opencv/4.11.0

virtualenv --no-download venv_ultralytics
source venv_ultralytics/bin/activate

pip install --no-index --upgrade pip
pip install --no-index numpy pillow pyyaml matplotlib scipy
pip install --no-index ultralytics==8.3.51+computecanada


echo "Creating backup of plotting.py"
PYDIR=$(ls -d venv_ultralytics/lib/python*)
cp \
"$PYDIR/site-packages/ultralytics/utils/plotting.py" \
"$PYDIR/site-packages/ultralytics/utils/plotting.py.bak"


echo "Patching plotting.py for RGB-D support"
python << 'EOF'
from pathlib import Path

path = Path("venv_ultralytics/lib") \
    .glob("python*/site-packages/ultralytics/utils/plotting.py")
path = next(path)

text = path.read_text()

start = text.index("# Build Image")
end = text.index("# Resize (optional)", start)

replacement = """# Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        # mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0) # Original
        if images[i].shape[0] == 3:  # Modified by Etienne
            mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)
        elif images[i].shape[0] == 4:  # Modified by Etienne
            mosaic[y : y + h, x : x + w, :] = images[i][[3,2,1]].transpose(1, 2, 0) # Modified by Etienne

"""

path.write_text(text[:start] + replacement + text[end:])
EOF

# source venv_ultralytics/bin/activate


