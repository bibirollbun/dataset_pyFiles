import sys  # sys module
import os
import torch
import pandas as pd
import numpy as np
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm



# Load datasets:
train_seq = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_lab = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
val_seq = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
val_lab = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
test_seq =pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")


# Adding the user-defined module path
sys.path.append("/kaggle/input/modules/")

from modules.dataset import RNAInferenceDataset
from modules.models.baseline import RNABaselineBiLSTM
from modules.utils.tm_score import compute_tm_scores_for_multiple_structures

# âœ… Config
class Config:
    def __init__(self):
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = "/kaggle/input/rna_model_baseline/pytorch/default/1/rna_model_baseline.pth"
        self.test_dir = "/kaggle/input/preprocessed-test"
        self.val_labels_path = "/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
        self.sample_submission_path = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"
        self.submission_path = "./submission.csv"
        self.hidden_dim = 128
        self.num_layers = 2
        self.dropout = 0.1

# âœ… Load the model
def load_model(config):
    model = RNABaselineBiLSTM(
        input_dim=5,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(checkpoint)
    model.to(config.device)
    model.eval()
    return model

# âœ… Inference (updated for overall progress only)
def run_inference(model, dataloader, device):
    predictions = []
    total = len(dataloader.dataset)
    with torch.no_grad():
        with tqdm(total=total, desc="ğŸ”� Running Inference", unit="seq") as pbar:
            for batch in dataloader:
                input_ids = batch[0].to(device)
                output = model(input_ids)  # [B, L, 5, 3]
                predictions.extend(output.cpu().numpy())
                pbar.update(input_ids.size(0))
    return predictions

# âœ… Save submission
def save_submission_from_test_seq(sample_submission_path, target_ids, predictions, output_path):
    df = pd.read_csv(sample_submission_path)

    id2row = {}
    tid2idx = {tid: i for i, tid in enumerate(target_ids)}
    bad_ids = []

    for _, row in df.iterrows():
        full_id = row["ID"]
        target_id, resid = full_id.split("_")
        resid = int(resid)

        if target_id not in tid2idx:
            raise ValueError(f"â�Œ target_id {target_id} not found")

        i = tid2idx[target_id]
        coords = predictions[i][resid - 1]  # [5, 3]

        if np.isnan(coords).any() or np.isinf(coords).any() or (coords == -1e+18).any():
            bad_ids.append(full_id)
            continue

        res_char = row["resname"]
        new_row = [full_id, res_char, resid] + coords.flatten().tolist()
        id2row[full_id] = new_row

    if bad_ids:
        print(f"âš ï¸� {len(bad_ids)} entries have invalid coordinates and were skipped. Examples: {bad_ids[:5]}")

    header = ['ID', 'resname', 'resid'] + [f"{axis}_{f}" for f in range(1, 6) for axis in ['x', 'y', 'z']]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar=' ')
        writer.writerow(header)
        for full_id in df["ID"]:
            if full_id not in id2row:
                raise ValueError(f"â�Œ Missing ID: {full_id}")
            writer.writerow(id2row[full_id])

    print(f"âœ… submission.csv saved successfully: {output_path} ({len(id2row)} rows)")

# âœ… Main ìˆ˜ì •
def main():
    config = Config()
    config.sample_submission_path = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"
    config.submission_path = "./submission.csv"

    print("ğŸ“¦ Loading model...")
    model = load_model(config)

    print("ğŸ“‚ Loading .pt test data...")
    X = torch.load(os.path.join(config.test_dir, "input_seqs.pt"), map_location=config.device)
    target_ids = torch.load(os.path.join(config.test_dir, "target_ids.pkl"), map_location=config.device)

    try:
        mask = torch.load(os.path.join(config.test_dir, "masks.pt"), map_location=config.device)
        if mask.numel() == 0 or mask.shape[0] == 0:
            raise ValueError("Empty mask detected")
    except:
        print("âš ï¸� Using dummy mask")
        mask = torch.ones(X.shape[:2], dtype=torch.float32)

    print("ğŸ“Š Inference Dataset shapes:")
    print("X:", X.shape)
    print("mask:", mask.shape)

    test_dataset = RNAInferenceDataset(X, mask)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0)

    print("ğŸš€ Running inference...")
    predictions = run_inference(model, test_loader, config.device)

    print("ğŸ’¾ Saving submission (test set)...")
    save_submission_from_test_seq(
        config.sample_submission_path,
        target_ids,
        predictions,
        config.submission_path
    )

    print("ğŸ�¯ Submission ready for Kaggle upload!")

if __name__ == "__main__":
    main()


from config import Config  # ì�´ë¯¸ ë�˜ì–´ ì�ˆë‹¤ë©´ ìƒ�ë�µ
config = Config()

config.test_dir = "/kaggle/input/preprocessed-test"  # ë˜�ëŠ” ë„ˆê°€ ì‚¬ìš©í•˜ëŠ” validation ê²½ë¡œ

target_ids = torch.load(os.path.join(config.test_dir, "target_ids.pkl"))

print("ğŸ“� total target_ids:", len(target_ids))
print("ğŸ§¬ unique:", len(set(target_ids)))
print("ğŸ§¾ sample:", target_ids[:5])



import pandas as pd

# Specify paths for submission files
submission_path = './submission.csv'  # Path to your submission file
sample_submission_path = '/kaggle/input/stanford-rna-3d-folding/sample_submission.csv'  # Path to sample submission file

# Read both submission and sample submission files
sub = pd.read_csv(submission_path)
sample = pd.read_csv(sample_submission_path)

# 1. Check the number of rows
assert len(sub) == len(sample), f"â�Œ Row count mismatch: submission ({len(sub)}) vs sample ({len(sample)})"
print(f"âœ… Row count match: {len(sub)}")

# 2. Check for missing IDs
missing_ids = set(sample["ID"]) - set(sub["ID"])
if len(missing_ids) > 0:
    print(f"â�Œ Missing IDs: {missing_ids}")
else:
    print("âœ… All IDs are present in the submission file.")

# 3. Check column order and names
expected_cols = ['ID', 'resname', 'resid'] + [f"{a}_{i}" for i in range(1, 6) for a in 'xyz']
assert list(sub.columns) == expected_cols, "â�Œ Column order/names mismatch"
print("âœ… Column order and names match")

# 4. Check for NaN values
assert sub.isnull().sum().sum() == 0, "â�Œ NaN values found"
print("âœ… No NaN values")

# 5. Verify the data type of 'resid'
assert pd.api.types.is_integer_dtype(sub['resid']), "â�Œ 'resid' type error"
print("âœ… 'resid' column type is correct")

# 6. Verify coordinate data types (example: x_1 column)
assert pd.api.types.is_float_dtype(sub['x_1']), "â�Œ Coordinate type error"
print("âœ… Coordinate type is correct")

# 7. Verify no missing IDs
assert set(sample["ID"]) - set(sub["ID"]) == set(), "â�Œ Missing IDs found"
print("âœ… No missing IDs")

# 8. Compare data types of all columns
print("\nSubmission file data types:")
print(sub.dtypes)

print("\nSample submission file data types:")
print(sample.dtypes)

# Check for any mismatched data types
mismatched_columns = []
for col in sub.columns:
    if sub[col].dtype != sample[col].dtype:
        mismatched_columns.append(col)

if mismatched_columns:
    print(f"â�Œ Mismatched data types in columns: {mismatched_columns}")
else:
    print("âœ… All column data types match")

print("ğŸ�‰ Submission file format check complete!")


