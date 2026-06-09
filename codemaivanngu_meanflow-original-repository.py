import os, sys
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB2")

os.environ["WANDB_API_KEY"] = secret_value_0  # force it into the env so the SDK can see it
os.environ['MPLBACKEND'] = 'agg'  # or del os.environ['MPLBACKEND'] if a specific backend is not neccessary
os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1' # tpu v5e mới quá dùng jax hơi cũ nên phải setup
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'


!pip install -q tfds apache_beam mlcroissant
!curl -LsSf https://astral.sh/uv/install.sh | sh
os.environ["PATH"] += ":/root/.local/bin"


!git clone https://github.com/Gsunshine/meanflow
%cd meanflow


!pip --version 


%%bash
uv python install 3.10
uv python pin 3.10
uv venv .venv --python 3.10
uv init
uv add pip
uv run bash


with open("/kaggle/working/meanflow/scripts/install.sh", "w") as file:
    file.write(r"""uv run pip install jax[tpu]==0.4.27 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv run pip install jaxlib==0.4.27 "flax>=0.8"
uv run pip install pillow clu tensorflow==2.15.0 "keras<3" "torch<=2.4" torchvision tensorflow_datasets matplotlib==3.9.2
uv run pip install orbax-checkpoint==0.4.4 ml-dtypes==0.5.0 tensorstore==0.1.67
uv run pip install diffusers dm-tree cached_property""")
# with open("/kaggle/working/meanflow/scripts/install.sh", "w") as file:
#     file.write(r"""uv run pip install jax[tpu]==0.4.13 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# uv run pip install jaxlib==0.4.13 "flax>=0.7.2"
# uv run pip install pillow clu tensorflow==2.13.1 "keras<3" "torch<=2.4" torchvision tensorflow_datasets matplotlib==3.7.5
# uv run pip install orbax-checkpoint==0.4.4 ml-dtypes==0.5.0 tensorstore==0.1.67
# uv run pip install diffusers dm-tree cached_property""")





cat /kaggle/working/meanflow/scripts/prepare_data.sh


file_path = "/kaggle/working/meanflow/scripts/prepare_data.sh"

with open(file_path,"w") as file:
    file.write("""#!/bin/bash

# Configuration for data preparation
export IMAGENET_ROOT="YOUR_IMAGENET_ROOT"
export OUTPUT_DIR="YOUR_OUTPUT_DIR"
export LOG_DIR="YOUR_LOG_DIR"

# Validate required environment variables
if [ "$IMAGENET_ROOT" = "YOUR_IMAGENET_ROOT" ] || [ "$OUTPUT_DIR" = "YOUR_OUTPUT_DIR" ] || [ "$LOG_DIR" = "YOUR_LOG_DIR" ]; then
    echo "ERROR: Please update the environment variables at the top of this script:"
    echo "  - IMAGENET_ROOT: Path to your ImageNet dataset"
    echo "  - OUTPUT_DIR: Path where to save the processed data"
    echo "  - LOG_DIR: Path where to save logs"
    exit 1
fi

export BATCH_SIZE=128
export VAE_TYPE="mse"

export now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
export JOBNAME=prepare_data_${now}_${salt}_$1
export LOG_DIR=$LOG_DIR/$USER/$JOBNAME

sudo mkdir -p ${LOG_DIR}
sudo chmod 777 -R ${LOG_DIR}

# Image size configuration (common sizes: 256, 512, 1024)
# Corresponding latent sizes will be: 32x32, 64x64, 128x128
IMAGE_SIZE=${IMAGE_SIZE:-256}  # Can be overridden via environment variable

# Computation flags (can be overridden via environment variables)
COMPUTE_LATENT=${COMPUTE_LATENT:-True}  # Whether to compute latent dataset
COMPUTE_FID=${COMPUTE_FID:-False}       # Whether to compute FID statistics

# Calculate latent size for display
LATENT_SIZE=$((IMAGE_SIZE / 8))

echo "=============================================="
echo "Data Preparation Configuration"
echo "=============================================="
echo "ImageNet Root: $IMAGENET_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "VAE Type: $VAE_TYPE"
echo "Image Size: $IMAGE_SIZE -> Latent Size: ${LATENT_SIZE}x${LATENT_SIZE}"
echo "Compute Latent: $COMPUTE_LATENT"
echo "Compute FID: $COMPUTE_FID"
if [ "$COMPUTE_FID" = "True" ]; then
    echo "FID: Using ALL training samples"
fi
echo "=============================================="

uv run prepare_dataset.py \
    --imagenet_root=\"$IMAGENET_ROOT\" \
    --output_dir=\"$OUTPUT_DIR\" \
    --batch_size=$BATCH_SIZE \
    --vae_type=\"$VAE_TYPE\" \
    --image_size=$IMAGE_SIZE \
    --compute_latent=$COMPUTE_LATENT \
    --compute_fid=$COMPUTE_FID \
    --overwrite=False \
    2>&1 | tee -a $LOG_DIR/output.log

echo "=============================================="
echo "Data preparation completed!"
echo "Check logs at: $LOG_DIR/output.log"
if [ "$COMPUTE_LATENT" = "True" ]; then
    echo "Latent dataset saved to: $OUTPUT_DIR"
fi
if [ "$COMPUTE_FID" = "True" ]; then
    echo "FID stats saved to: $OUTPUT_DIR/imagenet_${IMAGE_SIZE}_fid_stats.npz"
fi
echo "==============================================" """)
!chmod +x {file_path}
!bash {file_path}


cat /kaggle/working/meanflow/prepare_dataset.py


import os
import re
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC")
TRAIN_SRC = INPUT_ROOT / "train"
VAL_SRC   = INPUT_ROOT / "val"
MAP_FILE  = Path("/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv")

WORK_ROOT = Path("/kaggle/working/imagenet")
WORK_TRAIN = WORK_ROOT / "train"
WORK_VAL   = WORK_ROOT / "val"

assert TRAIN_SRC.exists(), f"Missing: {TRAIN_SRC}"
assert VAL_SRC.exists(), f"Missing: {VAL_SRC}"
assert MAP_FILE.exists(), f"Missing: {MAP_FILE}"

WORK_ROOT.mkdir(parents=True, exist_ok=True)

def symlink_if_needed(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.as_posix(), dst.as_posix())

def parse_loc_val_solution(map_path: Path):
    """
    Trả về dict: {ImageId(without .JPEG): wnid}
    Hỗ trợ:
      - CSV chuẩn: ImageId,PredictionString
      - Kiểu 2 dòng: ImageId \n "wnid x1 y1 x2 y2 ..."
    """
    mapping = {}

    # Đọc vài dòng đầu để đoán format
    head_lines = []
    with map_path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            head_lines.append(line.strip())

    is_csv_header = any("ImageId" in l for l in head_lines) and any("," in l for l in head_lines)

    if is_csv_header:
        # CSV dạng Kaggle phổ biến
        import csv
        with map_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if "ImageId" not in reader.fieldnames:
                raise ValueError(f"CSV header không có ImageId: {reader.fieldnames}")
            # Thường là PredictionString
            pred_col = "PredictionString" if "PredictionString" in reader.fieldnames else None
            if pred_col is None:
                # fallback: lấy cột thứ 2
                cols = [c for c in reader.fieldnames if c != "ImageId"]
                if not cols:
                    raise ValueError(f"Không tìm thấy cột prediction trong CSV: {reader.fieldnames}")
                pred_col = cols[0]

            for row in reader:
                imgid = (row.get("ImageId") or "").strip()
                pred  = (row.get(pred_col) or "").strip()
                if not imgid or not pred:
                    continue
                # PredictionString có thể chứa nhiều object: wnid x1 y1 x2 y2 wnid x1 y1 x2 y2 ...
                wnid = pred.split()[0]
                if re.fullmatch(r"n\d{8}", wnid):
                    mapping[imgid] = wnid
        return mapping

    # Fallback: parse kiểu “2 dòng” như bạn paste
    current_imgid = None
    with map_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Nếu là ImageId (không có khoảng trắng), ví dụ ILSVRC2012_val_00048981
            if re.fullmatch(r"ILSVRC2012_val_\d{8}", line):
                current_imgid = line
                continue

            # Nếu là dòng prediction bắt đầu bằng wnid
            if current_imgid is not None:
                first = line.split()[0]
                if re.fullmatch(r"n\d{8}", first):
                    mapping[current_imgid] = first
                    current_imgid = None

    if not mapping:
        raise ValueError("Không parse được LOC_val_solution.csv. Hãy mở vài dòng đầu của file để kiểm tra format.")
    return mapping

# 1) Symlink train
if not WORK_TRAIN.exists():
    os.symlink(TRAIN_SRC.as_posix(), WORK_TRAIN.as_posix())
print("train symlink:", WORK_TRAIN, "->", os.readlink(WORK_TRAIN))

# 2) Parse map
imgid2wnid = parse_loc_val_solution(MAP_FILE)
print("Parsed mappings:", len(imgid2wnid))

# 3) Tạo trước 1000 folder wnid theo train để đảm bảo class order nhất quán
WORK_VAL.mkdir(parents=True, exist_ok=True)
train_wnids = sorted([p.name for p in TRAIN_SRC.iterdir() if p.is_dir() and re.fullmatch(r"n\d{8}", p.name)])
print("Train wnids:", len(train_wnids))
for wnid in train_wnids:
    (WORK_VAL / wnid).mkdir(parents=True, exist_ok=True)

# 4) Symlink val ảnh vào đúng wnid folder
missing_img = 0
linked = 0

for imgid, wnid in imgid2wnid.items():
    src = VAL_SRC / f"{imgid}.JPEG"
    if not src.exists():
        # đôi khi extension khác case
        src = VAL_SRC / f"{imgid}.jpeg"
    if not src.exists():
        missing_img += 1
        continue

    dst = WORK_VAL / wnid / src.name
    if not dst.exists():
        os.symlink(src.as_posix(), dst.as_posix())
        linked += 1

print("Linked val images:", linked)
print("Missing val images:", missing_img)

# 5) Sanity check nhanh
val_count = sum(1 for _ in WORK_VAL.rglob("*.JPEG")) + sum(1 for _ in WORK_VAL.rglob("*.jpeg"))
print("WORK_VAL total images:", val_count)
print("WORK_ROOT:", WORK_ROOT)



%%writefile pyproject.toml
[project]
name = "shortcut-models"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = "==3.11.6"
dependencies = [
    "absl-py>=2.3.1",
    "chex>=0.1.86",
    "cython<3",
    "diffusers>=0.35.2",
    "distrax==0.1.4",
    "einops>=0.8.1",
    "fabric>=3.2.2",
    "flax>=0.8.3",
    "imageio>=2.37.0",
    "jax[tpu]==0.5.3",
    "jaxtyping>=0.3.3",
    "libtmux>=0.46.2",
    "matplotlib>=3.10.7",
    "ml-collections>=1.1.0",
    "moviepy>=2.2.1",
    "numba>=0.62.1",
    "numpy>=1.26.4",
    "opensimplex>=0.4.5.1",
    "opt-einsum>=3.4.0",
    "optax<=0.2.4",
    "orbax==0.1.9",
    "plotly>=6.3.1",
    "protobuf<=3.20.3",
    "pygame>=2.6.1",
    "scipy>1.12.0",
    "tabulate>=0.9.0",
    "tensorflow-cpu>=2.16.0",
    "tensorflow-datasets>=4.9.9",
    "tensorflow-probability==0.22.0",
    "termcolor>=3.1.0",
    "threadpoolctl==3.1.0",
    "typeguard>=4.0.0",
    "wandb>=0.19.11",
    "wheel>=0.45.1",
]




