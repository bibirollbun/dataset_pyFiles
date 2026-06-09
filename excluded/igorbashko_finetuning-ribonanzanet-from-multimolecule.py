%pip install git+https://github.com/dls5-omics/multimolecule@develop --quiet


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from multimolecule import RnaTokenizer, RibonanzaNetModel

class RibonanzaNetWith3DHead(nn.Module):
    """
    A wrapper model that uses a pre-trained RibonanzaNetModel,
    removes BOS/EOS token embeddings, and predicts 3D coordinates.
    """
    def __init__(self, pretrained_model_name="multimolecule/ribonanzanet"):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.base_model = RibonanzaNetModel.from_pretrained(self.pretrained_model_name)
        if hasattr(self.base_model, 'config') and hasattr(self.base_model.config, 'hidden_size'):
             self.hidden_size = self.base_model.config.hidden_size
             print(f"Detected hidden size: {self.hidden_size}")
        else:
            # Fallback or raise error if config/hidden_size is not found
            print("Warning: Could not automatically determine hidden size from model config. Assuming 256.")
            self.hidden_size = 256 # User mentioned 256, use as fallback
        self.coord_head = nn.Linear(self.hidden_size, 3) # Output size 3 for (x, y, z)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass through the base model and the coordinate head.
        Accepts tokenized input (input_ids, attention_mask, etc.)
        """
        base_model_args = {'input_ids': input_ids}
        if attention_mask is not None:
            base_model_args['attention_mask'] = attention_mask.float()
        outputs = self.base_model(**base_model_args)
        if hasattr(outputs, 'last_hidden_state'):
            all_embeddings = outputs.last_hidden_state
        else:
            raise AttributeError("Model output object does not have 'last_hidden_state'. Inspect the output structure.")
        # Slice to remove BOS (index 0) and EOS (index -1) embeddings
        # Shape becomes: (batch_size, sequence_length_nucleotides, hidden_size)
        # Note: This assumes BOS is always first and EOS is always last.
        nucleotide_embeddings = all_embeddings[:, 1:-1, :]

        # Handle potential empty sequence after slicing if input was just [BOS, EOS]
        if nucleotide_embeddings.shape[1] == 0:
             # Return an empty tensor with the correct dimensions or handle as needed
             # For prediction, maybe return shape (batch_size, 0, 3)
             # During training, this case might need special loss handling (e.g., ignore sample)
             print("Warning: Sequence length is zero after removing BOS/EOS.")
             # Example: return empty tensor
             return torch.zeros(nucleotide_embeddings.shape[0], 0, 3, device=nucleotide_embeddings.device)
        # Pass nucleotide embeddings through the coordinate prediction head
        # Shape: (batch_size, sequence_length_nucleotides, 3)
        predicted_coords = self.coord_head(nucleotide_embeddings)

        return predicted_coords


# Load CSVs
from tqdm import tqdm
import warnings
train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

# Create a pdb_id field
train_labels["pdb_id"] = train_labels["ID"].apply(
    lambda x: x.split("_")[0] + "_" + x.split("_")[1]
)

# Collect xyz data for each sequence
# Add warning ignore to make readable output
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    all_xyz = []
    for pdb_id in tqdm(train_sequences["target_id"], desc="Collecting XYZ data"):
        df = train_labels[train_labels["pdb_id"] == pdb_id]
        xyz = df[["x_1", "y_1", "z_1"]].to_numpy().astype("float32")
        xyz[xyz < -1e17] = float("nan")
        all_xyz.append(xyz)


valid_indices = []
max_len_seen = 0

for i, xyz in enumerate(all_xyz):
    # Track the maximum length
    if len(xyz) > max_len_seen:
        max_len_seen = len(xyz)

    nan_ratio = np.isnan(xyz).mean()
    seq_len = len(xyz)
    # Keep sequence if it meets criteria
    if (nan_ratio <= 0.5) and (10 < seq_len < 99999999):
        valid_indices.append(i)

print(f"Longest sequence in train: {max_len_seen}")

# Filter sequences & xyz based on valid_indices
train_sequences = train_sequences.loc[valid_indices].reset_index(drop=True)
all_xyz = [all_xyz[i] for i in valid_indices]

# Prepare final data dictionary
data = {
    "sequence": train_sequences["sequence"].tolist(),
    "temporal_cutoff": train_sequences["temporal_cutoff"].tolist(),
    "description": train_sequences["description"].tolist(),
    "all_sequences": train_sequences["all_sequences"].tolist(),
    "xyz": all_xyz,
}


cutoff_date = "2020-01-01"
test_cutoff_date = "2022-05-01"
cutoff_date = pd.Timestamp(cutoff_date)
test_cutoff_date = pd.Timestamp(test_cutoff_date)
train_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if pd.Timestamp(date_str) <= cutoff_date]
test_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if cutoff_date < pd.Timestamp(date_str) <= test_cutoff_date]


from torch.utils.data import Dataset, DataLoader
class RNA3D_Dataset(Dataset):
    """
    A PyTorch Dataset for 3D RNA structures.
    """
    def __init__(self, indices, data_dict,  tokenizer, max_len=128):
        self.indices = indices
        self.data = data_dict
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        sequence = tokenizer(self.data["sequence"][data_idx], return_tensors="pt", padding=True)
        sequence["input_ids"] = sequence["input_ids"].squeeze()
        sequence["attention_mask"] = sequence["attention_mask"].squeeze()
        if torch.cuda.is_available():
            sequence = {k: v.cuda() for k, v in sequence.items()}
        # Convert xyz to torch tensor
        xyz = torch.tensor(self.data["xyz"][data_idx], dtype=torch.float32)

        # If sequence is longer than max_len, randomly crop
        if len(sequence["input_ids"]) > self.max_len:
            crop_start = np.random.randint(len(sequence["input_ids"]) - self.max_len)
            crop_end = crop_start + self.max_len
            sequence["input_ids"] = sequence["input_ids"][crop_start:crop_end]
            sequence["attention_mask"] = sequence["attention_mask"][crop_start:crop_end]
            sequence["input_ids"][0] = 1
            sequence["input_ids"][127] = 2 #127 is 128-1. Just save some computation
            xyz = xyz[crop_start+1:crop_end-1]

        return {"sequence": sequence, "xyz": xyz, "shape": sequence["input_ids"].shape}

tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
train_dataset = RNA3D_Dataset(train_indices, data, tokenizer) # max_len was 384, I changed to 128 becuase 384 was throwind out of memory error on Kaggle's GPU
val_dataset = RNA3D_Dataset(test_indices, data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # Leave batch_size=1 so far, change in the future
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


def calculate_distance_matrix(X, Y, epsilon=1e-4):
    """
    Calculate pairwise distances between every point in X and every point in Y.
    Shape: (len(X), len(Y))
    """
    return (torch.square(X[:,None]-Y[None,:])+epsilon).sum(-1).sqrt()

def dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=None):
    """
    Distance-based RMSD.
    pred_x, pred_y: predicted coordinates (usually the same tensor for X and Y).
    gt_x, gt_y: ground truth coordinates.
    """
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff_sq = (pred_dm[mask] - gt_dm[mask])**2 + epsilon
    if d_clamp is not None:
        diff_sq = diff_sq.clamp(max=d_clamp**2)

    return diff_sq.sqrt().mean() / Z

def local_dRMSD(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10, d_clamp=30):
    """
    Local distance-based RMSD, ignoring distances above a clamp threshold.
    """
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = (~torch.isnan(gt_dm)) & (gt_dm < d_clamp)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff_sq = (pred_dm[mask] - gt_dm[mask])**2 + epsilon
    return diff_sq.sqrt().mean() / Z

def dRMAE(pred_x, pred_y, gt_x, gt_y, epsilon=1e-4, Z=10):
    """
    Distance-based Mean Absolute Error.
    """
    pred_dm = calculate_distance_matrix(pred_x, pred_y)
    gt_dm = calculate_distance_matrix(gt_x, gt_y)

    mask = ~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0], device=mask.device).bool()] = False

    diff = torch.abs(pred_dm[mask] - gt_dm[mask])
    return diff.mean() / Z

def align_svd_mae(input_coords, target_coords, Z=10):
    """
    Align input_coords to target_coords via SVD (Kabsch algorithm) and compute MAE.
    """
    assert input_coords.shape == target_coords.shape, "Input and target must have the same shape"

    # Create mask for valid points
    mask = ~torch.isnan(target_coords.sum(dim=-1))
    input_coords = input_coords[mask]
    target_coords = target_coords[mask]
    
    # Compute centroids
    centroid_input = input_coords.mean(dim=0, keepdim=True)
    centroid_target = target_coords.mean(dim=0, keepdim=True)

    # Center the points
    input_centered = input_coords - centroid_input
    target_centered = target_coords - centroid_target

    # Compute covariance matrix
    cov_matrix = input_centered.T @ target_centered

    # SVD to find optimal rotation
    U, S, Vt = torch.svd(cov_matrix)
    R = Vt @ U.T

    # Ensure a proper rotation (determinant R == 1)
    if torch.det(R) < 0:
        Vt_adj = Vt.clone()   # Clone to avoid in-place modification issues
        Vt_adj[-1, :] = -Vt_adj[-1, :]
        R = Vt_adj @ U.T

    # Rotate input and compute mean absolute error
    aligned_input = (input_centered @ R.T) + centroid_target
    return torch.abs(aligned_input - target_coords).mean() / Z


def train_model(model, train_dl, val_dl, epochs=1, cos_epoch=35, lr=3e-4, clip=1):
    """Train the model with a CosineAnnealingLR after `cos_epoch` epochs."""
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(epochs - cos_epoch) * len(train_dl),
    )

    best_val_loss = float("inf")
    best_preds = None

    for epoch in range(epochs):
        model.train()
        train_pbar = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        for idx, batch in enumerate(train_pbar):
            #sequence = batch["sequence"].cuda()
            sequence = batch["sequence"]
            gt_xyz = batch["xyz"].cuda()
            gt_xyz = gt_xyz.squeeze()

            pred_xyz = model(**sequence).squeeze()

            # Combine two distance-based losses
            loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz) + align_svd_mae(pred_xyz, gt_xyz)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

            if (epoch + 1) > cos_epoch:
                scheduler.step()

            running_loss += loss.item()
            avg_loss = running_loss / (idx + 1)
            train_pbar.set_description(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for idx, batch in enumerate(val_dl):
                sequence = batch["sequence"]
                gt_xyz = batch["xyz"].cuda()
                gt_xyz = gt_xyz.squeeze()

                pred_xyz = model(**sequence).squeeze()
                loss = dRMAE(pred_xyz, pred_xyz, gt_xyz, gt_xyz)
                val_loss += loss.item()

                val_preds.append((gt_xyz.cpu().numpy(), pred_xyz.cpu().numpy()))

            val_loss /= len(val_dl)
            print(f"Validation Loss (Epoch {epoch+1}): {val_loss:.4f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds = val_preds
                torch.save(model.state_dict(),"RibonanzaNet_multimolecule_fine_tuned_good_val.pt")
                print(f"  -> New best model saved at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), "RibonanzaNet_multimolecule_fine_tuned.pt")
    return best_val_loss, best_preds


model = RibonanzaNetWith3DHead(pretrained_model_name="multimolecule/ribonanzanet").cuda()


if __name__ == "__main__":
    best_loss, best_predictions = train_model(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        epochs=10,         # or config["epochs"]
        cos_epoch=35,      # or config["cos_epoch"]
        lr=3e-4,
        clip=1
    )
    print(f"Best Validation Loss: {best_loss:.4f}")

