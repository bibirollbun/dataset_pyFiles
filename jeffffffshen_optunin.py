!pip install optuna -q

import optuna
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# --- 1. CONFIGURATION: UPDATE THESE PATHS TO MATCH YOUR INPUTS ---

# This script assumes your files are in three locations. Add or remove paths as needed.

# Path to the outputs from your first run (e.g., Folds 1-2)
# This might be a dataset you uploaded.
RUN1_OUTPUT_DIR = Path("/kaggle/input/your-dataset-name-for-run1/stage_2_multihead/") 

# Path to the outputs from your second run (e.g., Fold 3)
# Based on your image, this looks like a dataset.
RUN2_OUTPUT_DIR = Path("/kaggle/input/iter1of6headsmultiheader/stage_2_multihead_resumed/") 

# Path to the outputs from your final notebook run (e.g., Folds 4-5)
# Based on your image, this is the output from the 'multi-header' notebook.
RUN3_OUTPUT_DIR = Path("/kaggle/input/multi-header/stage_2_multihead_final_folds/")

# List all directories the script needs to search for prediction files
ALL_RUN_DIRS = [RUN1_OUTPUT_DIR, RUN2_OUTPUT_DIR, RUN3_OUTPUT_DIR]

# This dictionary tells the script how many folds' worth of test predictions are in each run's .pkl file
# This is crucial for correctly averaging the test predictions.
FOLDS_PER_RUN = {
    RUN1_OUTPUT_DIR: 2,
    RUN2_OUTPUT_DIR: 1,
    RUN3_OUTPUT_DIR: 2,
}

# The model types you trained
MODEL_TYPES = [
    "cls_token", "mean_pool", "max_pool",
    "attention_pool", "concat_pool", "multiscale_attention"
]

# Path to the original competition data
DATA_DIR = Path("/kaggle/input/fake-or-real-the-impostor-hunt/data")
FINAL_SUBMISSION_PATH = "submission.csv"


# --- 2. HELPER FUNCTIONS ---
def normalize_folder_name(raw_id: str) -> str:
    raw = str(raw_id)
    if raw.startswith("article_"): return raw
    if raw.isdigit(): return f"article_{int(raw):04d}"
    return raw

def load_test_pairs(data_dir: str) -> pd.DataFrame:
    test_folder = Path(data_dir) / "test"
    rows = []
    if test_folder.exists():
        for aid_folder in sorted(test_folder.iterdir()):
            if aid_folder.is_dir() and not aid_folder.name.startswith("."):
                folder_name = normalize_folder_name(aid_folder.name)
                for idx in (1, 2):
                    file_path = aid_folder / f"file_{idx}.txt"
                    if file_path.exists():
                        text = file_path.read_text(encoding="utf-8")
                        rows.append({"id": folder_name, "file_idx": idx, "text": text})
    return pd.DataFrame(rows)


# --- 3. LOAD & COMBINE PREDICTIONS FROM ALL RUNS ---
print("Loading and combining predictions from all runs...")

# Part A: Combine Out-of-Fold (OOF) predictions for tuning
all_oof = {}
for model_type in MODEL_TYPES:
    oof_dfs = [pd.read_csv(run_dir / f"oof_{model_type}.csv") for run_dir in ALL_RUN_DIRS if (run_dir / f"oof_{model_type}.csv").exists()]
    if oof_dfs:
        all_oof[model_type] = pd.concat(oof_dfs, ignore_index=True)
if not all_oof: raise ValueError("No OOF files found. Please check your RUN directory paths.")
ground_truth_df = all_oof[MODEL_TYPES[0]][['id', 'file_idx', 'label']].copy()
print("OOF predictions loaded successfully.")

# Part B: Combine Test predictions for final submission
final_test_preds = {}
total_folds = sum(FOLDS_PER_RUN.values())
for model_type in MODEL_TYPES:
    weighted_preds = None
    for run_dir, num_folds in FOLDS_PER_RUN.items():
        pkl_path = run_dir / "test_predictions.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                run_preds = pickle.load(f)
                if model_type in run_preds:
                    if weighted_preds is None:
                        weighted_preds = run_preds[model_type] * num_folds
                    else:
                        weighted_preds += run_preds[model_type] * num_folds
    if weighted_preds is not None:
        final_test_preds[model_type] = weighted_preds / total_folds
print("Test predictions loaded and combined successfully.")


# --- 4. OPTUNA SETUP & EXECUTION ---
def calculate_per_article_accuracy(oof_df: pd.DataFrame) -> float:
    correct, total = 0, 0
    for _, group in oof_df.groupby("id"):
        if group.loc[group["prob_real"].idxmax()]["label"] == 1:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def objective(trial: optuna.Trial) -> float:
    trial_df = ground_truth_df.copy()
    weighted_probs, total_weight = np.zeros(len(trial_df)), 0
    for model_type in MODEL_TYPES:
        weight = trial.suggest_float(f"w_{model_type}", 0.0, 1.0)
        weighted_probs += weight * all_oof[model_type]['prob_real'].values
        total_weight += weight
    if total_weight > 0:
        trial_df['prob_real'] = weighted_probs / total_weight
    else: return 0.0
    return calculate_per_article_accuracy(trial_df)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    print("\n" + "="*50)
    print("ğŸš€ OPTUNA STUDY COMPLETE ğŸš€")
    print("="*50)
    print(f"Best OOF Accuracy: {study.best_value:.6f}")
    print("\nOptimal weights for the ensemble:")
    best_weights = sorted(study.best_params.items(), key=lambda item: item[1], reverse=True)
    for param, value in best_weights:
        print(f"  - {param:25s}: {value:.4f}")

    # --- 5. GENERATE FINAL SUBMISSION ---
    print("\n" + "="*50)
    print("ğŸ“� Generating final submission file...")
    print("="*50)

    test_df = load_test_pairs(DATA_DIR)
    if not test_df.empty and final_test_preds:
        best_params = study.best_params
        final_probs, total_weight = np.zeros(len(test_df)), 0
        
        for model_type in MODEL_TYPES:
            weight = best_params.get(f"w_{model_type}", 0)
            if model_type in final_test_preds:
                final_probs += weight * final_test_preds[model_type]
                total_weight += weight

        if total_weight > 0:
            test_df['prob_real'] = final_probs / total_weight
        else:
            test_df['prob_real'] = 0.5
        
        submission_rows = []
        for article_id, group in test_df.groupby('id'):
            chosen_file = group.loc[group['prob_real'].idxmax()]
            numeric_id = int(article_id.replace("article_", ""))
            submission_rows.append({"id": numeric_id, "real_text_id": int(chosen_file['file_idx'])})
            
        submission_df = pd.DataFrame(submission_rows).sort_values("id")
        submission_df.to_csv(FINAL_SUBMISSION_PATH, index=False)
        print(f"\nâœ… Submission file saved successfully to: {FINAL_SUBMISSION_PATH}")
    else:
        print("\nCould not generate submission: No test data or test predictions found.")

