!ls /kaggle/input/byu-locating-bacterial-flagellar-motors-2025


!ls /kaggle/input/byu2025-ray03j-dataset-3


!ls /kaggle/working/


!python /kaggle/input/byu2025-ray03j-dataset-3/01_create_dataset.py \
--input_dir /kaggle/input/byu-locating-bacterial-flagellar-motors-2025 \
--output_dir /kaggle/working/


!python /kaggle/input/byu2025-ray03j-dataset-3/01_create_dataset.py \
--input_dir /kaggle/input/byu-locating-bacterial-flagellar-motors-2025 \
--output_dir /kaggle/working/ \
--mode test


# !python /kaggle/input/byu2025-ray03j-dataset-3/02_train.py --config /kaggle/input/byu-base0525-3/src/config.yaml


import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

class TomoDataPreparer:
    def __init__(self, input_dir: Path, val_output_dir: Path, val_ratio: float = 0.3):
        self.input_dir = input_dir
        self.val_output_dir = val_output_dir
        self.val_ratio = val_ratio
        self.train_ids = []
        self.val_ids = []

    def prepare_data(self):
        """train_labels.csv ã‚’èª­ã�¿è¾¼ã‚“ã�§ã€�valãƒ‡ãƒ¼ã‚¿ã‚’åˆ†å‰²ãƒ»å‡ºåŠ›"""
        df = pd.read_csv(self.input_dir / "train_labels.csv")
        tomo_ids = df["tomo_id"].unique()

        # train/val ã�®åˆ†å‰²
        self.train_ids, self.val_ids = train_test_split(
            tomo_ids, test_size=self.val_ratio, random_state=42
        )

        # val ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã‚’ä½œæˆ�
        self.val_output_dir.mkdir(parents=True, exist_ok=True)

        train_dir = self.input_dir / "train"
        for tomo_id in self.val_ids:
            src_dir = train_dir / tomo_id
            dst_dir = self.val_output_dir / tomo_id

            if src_dir.exists() and not dst_dir.exists():
                dst_dir.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(src_dir, dst_dir)
                    print(f"Created symlink for {tomo_id}: {src_dir} -> {dst_dir}")
                except OSError as e:
                    print(f"Failed to create symlink for {tomo_id}: {e}")

# --- Kaggle Notebook å›ºå®šãƒ‘ã‚¹è¨­å®š ---
input_dir = Path("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025")
val_output_dir = Path("/kaggle/working/val")
val_ratio = 0.3

# --- å®Ÿè¡Œ ---
preparer = TomoDataPreparer(
    input_dir=input_dir,
    val_output_dir=val_output_dir,
    val_ratio=val_ratio
)
preparer.prepare_data()



!python /kaggle/input/byu2025-ray03j-dataset-3/06_predict.py \
--config /kaggle/input/byu2025-ray03j-dataset-3/src/config.yaml \
--checkpoint /kaggle/input/byu0516/byu0516_epoch031_val_loss0.0035.ckpt \
--output_dir /kaggle/working \
--val_output_dir /kaggle/working/val \
opts data.batch_size=8


from omegaconf import OmegaConf
from collections import defaultdict
import numpy as np
from pathlib import Path
import csv
import sys

# ==== å›ºå®šãƒ‘ã‚¹ï¼ˆKaggle Notebookã�§ä½¿ã�†å‰�æ��ï¼‰ ====
CONFIG_PATH = "/kaggle/input/byu2025-ray03j-dataset-3/src/config.yaml"
OUTPUT_DIR = Path("/kaggle/working")

# ==== config èª­ã�¿è¾¼ã�¿ ====
cfg = OmegaConf.load(CONFIG_PATH)
cfg.data.output_dir = str(OUTPUT_DIR)

# ==== ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªä½œæˆ� ====
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== å…¥å‡ºåŠ›ãƒ•ã‚¡ã‚¤ãƒ« ====
input_file = OUTPUT_DIR / "output.csv"
output_file = OUTPUT_DIR / "submission.csv"

# ==== å…¥åŠ›ãƒ•ã‚¡ã‚¤ãƒ«ãƒ�ã‚§ãƒƒã‚¯ ====
if not input_file.exists():
    print(f"â�Œ å…¥åŠ›ãƒ•ã‚¡ã‚¤ãƒ«ã�Œå­˜åœ¨ã�—ã�¾ã�›ã‚“: {input_file}", file=sys.stderr)
    sys.exit(1)

print(f"âœ… å…¥åŠ›ãƒ•ã‚¡ã‚¤ãƒ«ã‚’èª­ã�¿è¾¼ã�¿ä¸­: {input_file}")

# ==== ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿ ====
tomo_coords = defaultdict(list)
with open(input_file, newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # ãƒ˜ãƒƒãƒ€ãƒ¼ã‚’ã‚¹ã‚­ãƒƒãƒ—
    for row in reader:
        tomo_id = row[0]
        coords = list(map(float, row[1:]))
        tomo_coords[tomo_id].append(coords)

# ==== å¹³å�‡ã‚’è¨ˆç®—ã�— submission.csv å‡ºåŠ› ====
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
    for tomo_id, coords_list in tomo_coords.items():
        mean_coords = np.mean(coords_list, axis=0)
        mean_coords = np.round(mean_coords, 5)
        writer.writerow([tomo_id] + mean_coords.tolist())

print(f"ğŸ�‰ å®Œäº†: submission.csv ã�Œç”Ÿæˆ�ã�•ã‚Œã�¾ã�—ã�Ÿ â†’ {output_file}")


! head -n 10 submission.csv
! wc -l submission.csv

