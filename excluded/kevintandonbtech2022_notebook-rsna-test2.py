import os, gc, warnings
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ====================================================
# Constants
# ====================================================
ID_COL = "SeriesInstanceUID"
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# ====================================================
# Config
# ====================================================
class CFG:
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    size = 384
    batch_size = 8
    epochs = 5
    lr = 2e-4
    n_folds = 5
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================
# Dummy dataset class (replace with your dataset)
# ====================================================
class RSNADataset(Dataset):
    def __init__(self, df, labels):
        self.df = df
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # X = image tensor [3, H, W] (replace with DICOM preproc if needed)
        X = torch.randn(3, CFG.size, CFG.size)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y

# ====================================================
# Model
# ====================================================
def get_model():
    model = timm.create_model(
        CFG.model_name,
        pretrained=False,   # ⬅️ change this to False (no HF download)
        in_chans=3,
        num_classes=len(LABEL_COLS),
        drop_rate=0.2,
        drop_path_rate=0.2
    )
    return model


# ====================================================
# Training loop
# ====================================================
def train_and_save_oof(train_df, targets):
    oof_preds = np.zeros((len(train_df), len(LABEL_COLS)))

    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, targets[:, -1])):  # stratify on "Aneurysm Present"
        print(f"===== Fold {fold} =====")

        model = get_model().to(CFG.device)
        optimizer = AdamW(model.parameters(), lr=CFG.lr)
        criterion = nn.BCEWithLogitsLoss()

        trn_dataset = RSNADataset(train_df.iloc[trn_idx], targets[trn_idx])
        val_dataset = RSNADataset(train_df.iloc[val_idx], targets[val_idx])

        trn_loader = DataLoader(trn_dataset, batch_size=CFG.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)

        best_auc = 0
        for epoch in range(CFG.epochs):
            model.train()
            for X, y in trn_loader:
                X, y = X.to(CFG.device), y.to(CFG.device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(CFG.device)
                    out = model(X)
                    val_preds.append(torch.sigmoid(out).cpu().numpy())
                    val_targets.append(y.numpy())
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)

            auc = roc_auc_score(val_targets[:, -1], val_preds[:, -1])
            print(f"Epoch {epoch} - Fold {fold} - Aneurysm AUC: {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), f"fold{fold}_best.pth")

        # Save OOF predictions
        model.load_state_dict(torch.load(f"fold{fold}_best.pth"))
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X, _ in val_loader:
                X = X.to(CFG.device)
                out = model(X)
                val_preds.append(torch.sigmoid(out).cpu().numpy())
        val_preds = np.concatenate(val_preds)
        oof_preds[val_idx] = val_preds

    # Save to CSV
    oof_df = pd.DataFrame(oof_preds, columns=LABEL_COLS)
    oof_df.insert(0, ID_COL, train_df[ID_COL].values)
    oof_df["target"] = targets[:, -1]  # ground truth for aneurysm
    oof_df.to_csv("/kaggle/working/oof.csv", index=False)
    print("Saved OOF predictions to /kaggle/working/oof.csv")
'''
# Example usage (replace with your real train dataframe + labels)
dummy_df = pd.DataFrame({"SeriesInstanceUID": [f"id_{i}" for i in range(100)]})
dummy_labels = np.random.randint(0, 2, (100, len(LABEL_COLS)))
train_and_save_oof(dummy_df, dummy_labels)
'''


# fit_calibrators.py
def_calibrate():
    import pandas as pd
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from pathlib import Path
    
    # Paths
    oof_path = Path("/kaggle/working/oof.csv")
    out_path = Path("/kaggle/working/calibrators.pkl")
    
    LABEL_COLS = [
        'Left Infraclinoid Internal Carotid Artery',
        'Right Infraclinoid Internal Carotid Artery',
        'Left Supraclinoid Internal Carotid Artery',
        'Right Supraclinoid Internal Carotid Artery',
        'Left Middle Cerebral Artery',
        'Right Middle Cerebral Artery',
        'Anterior Communicating Artery',
        'Left Anterior Cerebral Artery',
        'Right Anterior Cerebral Artery',
        'Left Posterior Communicating Artery',
        'Right Posterior Communicating Artery',
        'Basilar Tip',
        'Other Posterior Circulation',
        'Aneurysm Present',
    ]
    '''
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF not found: {oof_path}")
        
    oof = pd.read_csv(oof_path)
    '''
    print("Loaded OOF:", oof.shape)
    print("OOF columns:", oof.columns.tolist()[:30])
    
    # Determine ground-truth column name(s)
    # We'll assume the OOF has a 'target' or 'Aneurysm_gt' or similar.
    # Try to detect a ground truth column automatically.
    possible_gt_names = ["target", "AneurysmPresent_gt", "aneurysm_label", "Aneurysm_gt",
                         "Aneurysm Present_gt", "aneurysm", "y_true"]
    gt_col = None
    for cand in possible_gt_names:
        if cand in oof.columns:
            gt_col = cand
            break

    # Heuristic: if an explicit per-row ground truth column not found, try common patterns:
    if gt_col is None:
        # If oof contains a column matching 'Aneurysm Present' with integer values 0/1,
        # we may assume that is the prediction column, and underlying ground truth may be stored as 'target'
        # Fallback: if there's a 'target' column use it.
        if "target" in oof.columns:
            gt_col = "target"
    
    if gt_col is None:
        # Let user know and raise error for clarity.
        raise ValueError("Could not find a ground-truth column in OOF. Please ensure your OOF has a column named 'target' (or similar).")
    
    print("Using ground-truth column:", gt_col)
    
    calibrators = {}
    
    for col in LABEL_COLS:
        if col not in oof.columns:
            print(f"[Skip] Predictions column not found in OOF: {col}")
            calibrators[col] = None
            continue

        y_scores = oof[col].values
        y_true = oof[gt_col].values  # NOTE: uses same GT for all labels; replace per-label GT if you have them
    
        # Check that y_true contains at least two classes
        unique = np.unique(y_true)
        if unique.shape[0] < 2:
            print(f"[Skip] Ground truth for column '{col}' has only one class ({unique}). Skipping calibrator.")
            calibrators[col] = None
            continue
    
        # Also ensure scores are not degenerate
        if np.allclose(y_scores, y_scores[0]):
            print(f"[Skip] Predictions for '{col}' are constant; skipping calibrator.")
            calibrators[col] = None
            continue

        # Prefer Platt scaling (LogisticRegression). If it fails, fallback to isotonic
        try:
            lr = LogisticRegression(max_iter=2000)
            lr.fit(y_scores.reshape(-1, 1), y_true)
            calibrators[col] = ("platt", lr)
            print(f"[OK] Trained Platt calibrator for '{col}'")
        except Exception as e:
            print(f"[Warn] Platt failed for '{col}' ({e}), trying isotonic.")
            try:
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(y_scores, y_true)
                calibrators[col] = ("isotonic", iso)
                print(f"[OK] Trained isotonic calibrator for '{col}'")
            except Exception as e2:
                print(f"[Skip] Both calibrators failed for '{col}': {e2}. Saving None.")
                calibrators[col] = None
    
    # Save calibrators dictionary
    joblib.dump(calibrators, out_path)
    print("Saved calibrators to:", out_path) 


# ---------------------------
# Build and save submission.parquet (robust, runs predict() per series)
# Paste this into your inference notebook AFTER models are loaded
# and predict(series_path) is defined.
# ---------------------------

import os
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm

# Make sure LABEL_COLS matches your model output order
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# Conservative fallback prediction if a series fails
FALLBACK_ROW = [0.1] * len(LABEL_COLS)
'''
def find_series_dirs(search_roots=None):
    """Search for DICOM series directories (contains at least one .dcm file)."""
    roots = search_roots or ['/kaggle/input', '/kaggle/working']
    series_dirs = []
    for root in roots:
        if not os.path.exists(root): 
            continue
        for entry in Path(root).rglob('*'):
            if entry.is_dir():
                try:
                    # quick check: any .dcm file in directory (non-recursive)
                    for f in entry.iterdir():
                        if f.is_file() and f.suffix.lower() == '.dcm':
                            series_dirs.append(str(entry))
                            break
                except Exception:
                    # skip unreadable dirs
                    continue
    return sorted(set(series_dirs))

# Find candidates automatically
series_dirs = find_series_dirs()

# If none found automatically, also try direct children of /kaggle/input
if not series_dirs:
    # fallback: any subdirectory of input that contains files
    for root in ['/kaggle/input', '/kaggle/working']:
        if not os.path.exists(root): continue
        for d in Path(root).iterdir():
            if d.is_dir():
                # treat the folder as a series if it has files
                files = list(d.rglob('*'))
                if files:
                    series_dirs.append(str(d))
    series_dirs = sorted(set(series_dirs))

print(f"Found {len(series_dirs)} candidate series directories (first 10):")
for d in series_dirs[:10]:
    print("  ", d)

if not series_dirs:
    raise RuntimeError("No series directories found. If your test series are located in a different path, set `series_dirs` manually.")

# Run predictions and collect rows
rows = []
ids = []

for series_path in tqdm(series_dirs, desc="Predicting series"):
    sid = os.path.basename(series_path.rstrip('/')) or series_path  # fallback UID
    try:
        # predict should return a polars.DataFrame with columns = LABEL_COLS (no ID col)
        pred_df = predict(series_path)  # your existing function
        # If function returns a polars.DataFrame with ID column, handle that
        if isinstance(pred_df, pl.DataFrame):
            pdf = pred_df.to_pandas()
        elif isinstance(pred_df, pd.DataFrame):
            pdf = pred_df
        else:
            # if predict returns a numpy array
            if isinstance(pred_df, np.ndarray):
                arr = pred_df
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                pdf = pd.DataFrame(arr, columns=LABEL_COLS)
        # If pdf has the ID column, drop it and use our sid
        if 'ID' in pdf.columns or 'SeriesInstanceUID' in pdf.columns:
            # find label columns intersection
            label_cols_present = [c for c in LABEL_COLS if c in pdf.columns]
            row_values = pdf[label_cols_present].iloc[0].astype(float).values
            # ensure correct column order; if some missing, fill with fallback
            final_row = []
            for col in LABEL_COLS:
                if col in pdf.columns:
                    final_row.append(float(pdf[col].iloc[0]))
                else:
                    final_row.append(0.1)
        else:
            # assume pdf columns are exactly LABEL_COLS order
            try:
                row_values = pdf.values.flatten().astype(float)
                if row_values.size != len(LABEL_COLS):
                    # mismatch — try to map by column names
                    final_row = []
                    for col in LABEL_COLS:
                        if col in pdf.columns:
                            final_row.append(float(pdf[col].iloc[0]))
                        else:
                            final_row.append(0.1)
                else:
                    final_row = row_values.tolist()
            except Exception:
                final_row = FALLBACK_ROW
        # ensure numeric and length-correct
        if len(final_row) != len(LABEL_COLS):
            final_row = FALLBACK_ROW
        rows.append(final_row)
        ids.append(sid)
    except Exception as e:
        # on any failure, log and append fallback
        print(f"[Error] series {sid} failed: {e}")
        traceback.print_exc()
        rows.append(FALLBACK_ROW)
        ids.append(sid)

# Build submission DataFrame
sub_df = pd.DataFrame(rows, columns=LABEL_COLS)
sub_df.insert(0, "ID", ids)

# Save as submission.parquet exactly where Kaggle expects it
out_path = "/kaggle/working/submission.parquet"
sub_df.to_parquet(out_path, index=False)
print(f"✅ Saved submission to {out_path} (rows: {len(sub_df)})")
print(sub_df.head())'''



import os
import shutil
from collections import defaultdict

import pandas as pd
import polars as pl
import pydicom

import kaggle_evaluation.rsna_inference_server





ID_COL = 'SeriesInstanceUID'

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# All tags (other than PixelData and SeriesInstanceUID) that may be in a test set dcm file
DICOM_TAG_ALLOWLIST = [
    'BitsAllocated',
    'BitsStored',
    'Columns',
    'FrameOfReferenceUID',
    'HighBit',
    'ImageOrientationPatient',
    'ImagePositionPatient',
    'InstanceNumber',
    'Modality',
    'PatientID',
    'PhotometricInterpretation',
    'PixelRepresentation',
    'PixelSpacing',
    'PlanarConfiguration',
    'RescaleIntercept',
    'RescaleSlope',
    'RescaleType',
    'Rows',
    'SOPClassUID',
    'SOPInstanceUID',
    'SamplesPerPixel',
    'SliceThickness',
    'SpacingBetweenSlices',
    'StudyInstanceUID',
    'TransferSyntaxUID',
]

# Replace this function with your inference code.
# You can return either a Pandas or Polars dataframe, though Polars is recommended.
# Each prediction (except the very first) must be returned within 30 minutes of the series being provided.
def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # --------- Replace this section with your own prediction code ---------
    series_id = os.path.basename(series_path)
    
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    # Collect tags from the dicoms
    tags = defaultdict(list)
    tags['SeriesInstanceUID'] = series_id
    global dcms
    for filepath in all_filepaths:
        ds = pydicom.dcmread(filepath, force=True)
        tags['filepath'].append(filepath)
        for tag in DICOM_TAG_ALLOWLIST:
            tags[tag].append(getattr(ds, tag, None))
        # The image is in ds.PixelData

    # ... do some machine learning magic ...
    predictions = pl.DataFrame(
        data=[[series_id] + [0.5] * len(LABEL_COLS)],
        schema=[ID_COL, *LABEL_COLS],
        orient='row',
    )
    # ----------------------------------------------------------------------

    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == [ID_COL, *LABEL_COLS]
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == [ID_COL, *LABEL_COLS]).all()
    else:
        raise TypeError('The predict function must return a DataFrame')

    # ----------------------------- IMPORTANT ------------------------------
    # You MUST have the following code in your `predict` function
    # to prevent "out of disk space" errors. This is a temporary workaround
    # as we implement improvements to our evaluation system.
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    # ----------------------------------------------------------------------
    
    return predictions.drop(ID_COL)


inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))




