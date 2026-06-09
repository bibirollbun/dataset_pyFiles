import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score
from datetime import datetime
import sys

# --- Add Your Custom Scripts to the Python Path ---
# This ensures the notebook can find your .py files on Kaggle
SCRIPTS_DIR = "/kaggle/input/monigarr-kaggle-jigsaw2025-utility-scripts"
sys.path.append(SCRIPTS_DIR)

# --- Import Custom Utilities & Config ---
from data_utils import load_data, create_folds, prepare_and_tokenize_data
from training_utils import run_experiment
from system_utils import check_gpu, check_tokenizer_files
from configs import RedditRoBERTaBaseKaggleConfig as CFG # Use a Kaggle-specific config

# Instantiate the config
cfg = CFG()


# --- System and Path Verification ---
check_gpu()
check_tokenizer_files(cfg)


# --- Data Pipeline ---
DATA_DIR = '/kaggle/input/jigsaw-agile-community-rules/'

# Load, prepare, and tokenize all data using our utility functions
train_df, test_df, _ = load_data(DATA_DIR)
train_df = create_folds(train_df, cfg)
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)
train_df, test_df, tokenizer = prepare_and_tokenize_data(train_df, test_df, cfg, tokenizer)

print("\nData pipeline complete. DataFrame prepared for training.")


# --- Run the Entire 5-Fold Fine-Tuning Experiment ---
oof_preds, test_preds = run_experiment(cfg, train_df, test_df, tokenizer)

print("\nFull 5-fold training and prediction complete.")


# --- Create Submission File ---
submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': test_preds
})

# Save the final submission file required by the competition
submission_df.to_csv('submission.csv', index=False)

print("submission.csv created successfully!")
submission_df.head()

