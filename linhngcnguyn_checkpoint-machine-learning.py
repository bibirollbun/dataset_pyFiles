!git clone https://github.com/rangha26/ML_nhom7.git



%cd /kaggle/working/ML_nhom7
!ls



!pip install -r requirements.txt


%%bash
cd /kaggle/working/RSNA2024

CLEARML_OFFLINE_MODE=1 TMPDIR=/dev/shm python - << 'PY'
import sys, os
sys.path.insert(0, ".")

sys.argv = [
    "train.py",
    "--data_dir", "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification",
    "--workdir", "/kaggle/working/workdir",
    "--epochs", "20", 
    "--batch_size", "2",
    "--accumulation_steps", "2",
    "--num_workers", "0",
]

# patch pandas.read_csv để chỉ lấy ít dòng -> giảm dataset -> giảm cache output
import pandas as pd
_real_read_csv = pd.read_csv
def _patched_read_csv(path, *args, **kwargs):
    df = _real_read_csv(path, *args, **kwargs)
    if str(path).endswith("train.csv"):
        df = df.sample(n=min(300, len(df)), random_state=42).reset_index(drop=True)  # <= đổi 300 thành 100/200 nếu muốn nhẹ hơn
    return df
pd.read_csv = _patched_read_csv

import config
config.args.num_workers = int(config.args.num_workers)
config.args.batch_size = int(config.args.batch_size)
config.args.epochs = int(config.args.epochs)
config.args.accumulation_steps = int(config.args.accumulation_steps)

config.args.image_size = (32, 128, 256)

# đưa cache sang RAM để không ăn /kaggle/working
config.args.cache_dir = "/dev/shm/rsna_cache"

import runpy
runpy.run_path("train.py", run_name="__main__")
PY


!cp "/kaggle/working/workdir/SEResNext101_custom_[no_resample]_[augs1]_32x128x256/model_best.pth" \
    "/kaggle/working/model_best.pth"



!ls -lh /kaggle/working/model_best.pth


