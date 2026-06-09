# === Install required Python packages from your uploaded requirements.txt ===
!pip install /kaggle/input/keyvulee_v3/pytorch/default/9/biopython-1.85-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --quiet
!pip install /kaggle/input/keyvulee_v3/pytorch/default/9/ptflops-0.7.4-py3-none-any.whl --no-index --quiet --no-deps

# === Standard imports ===
import sys, os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
# === Add the path to your uploaded code files ===
sys.path.append("/kaggle/input/keyvulee_v3/pytorch/default/9/src_v3")

# === Import your custom modules ===
from config_v3 import get_config
from dataset_v2 import RNAFMDataset, rna_collate_fn
from model_v3 import create_model

print("✅ Direct imports successful!")


ckpt_path = "/kaggle/input/keyvulee_v3/pytorch/default/9/epoch1-step1284.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model(get_config())
model = model.to(device)
model.eval()

checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
model.load_state_dict(state_dict, strict=False)

print("✅ Model loaded.")


config = get_config()
config.data.test_sequences = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
config.data.use_cache = False
config.output_dir = "/kaggle/working"

test_dataset = RNAFMDataset(
    sequences_csv=config.data.test_sequences,
    coords_csv=None,
    max_len=config.data.max_len,
    device="cpu",
    use_cache=False,
    cache_dir=config.data.cache_dir,
    msa_dir=config.data.msa_dir,
    window_size=config.data.window_size,
    stride=config.data.stride,
)

from torch.utils.data import DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=rna_collate_fn,
)

print("✅ Dataset loaded and ready.")


import numpy as np
import pandas as pd

rows = []

# Optional: controls how many noisy decoys are created per structure
def generate_decoys(pred, num_decoys=5, noise_scale=0.5):
    return [pred + np.random.normal(scale=noise_scale, size=pred.shape) for _ in range(num_decoys)]

with torch.no_grad():
    for batch in test_loader:
        emb = batch['embeddings'].to(device)
        mask = batch['mask'].to(device)
        ids = batch['id']
        preds = model(emb).cpu().numpy()

        for i, seq_id in enumerate(ids):
            L = mask[i].sum().item()
            pred_coords = preds[i][:L]
            decoys = generate_decoys(pred_coords)

            # Get original sequence
            seq = test_dataset.seq_dict.get(seq_id, "A" * L)

            for j in range(L):
                row = {
                    "ID": f"{seq_id}_{j+1}",
                    "resname": seq[j] if j < len(seq) else "A",
                    "resid": j+1
                }
                for d, dec in enumerate(decoys):
                    x, y, z = dec[j]
                    row[f"x_{d+1}"] = x
                    row[f"y_{d+1}"] = y
                    row[f"z_{d+1}"] = z
                rows.append(row)

# Save as submission.csv
submission_df = pd.DataFrame(rows)
submission_df.to_csv("/kaggle/working/submission.csv", index=False)

print("✅ submission.csv saved and ready to submit!")


