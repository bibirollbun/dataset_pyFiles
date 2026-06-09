!pip install -U transformers


!pip install torch datasets scikit-learn


# ===========================
# ENHANCED JIGSAW SOLUTION v2 
# ===========================

import random
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import warnings

warnings.filterwarnings('ignore')

# ===========================
# HELPER FUNCTIONS
# ===========================
def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===========================
# CONFIGURATION
# ===========================
VERSION = 2

MODEL_NAME = "/kaggle/input/deberta-v3-small/transformers/default/1"

EPOCHS = 4
MAX_LEN = 512
SEED = 42
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_EVAL = 8
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# Set up directories and environment
DIR = f"Jigsaw_{VERSION}"
os.makedirs(DIR, exist_ok=True)
# using Kaggle input directly
# os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Ensure GPU is enabled in Kaggle/Colab

# ===========================
# DATA LOADING
# ===========================
print("Loading data...")
# Kaggle environment
train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# Initialize tokenizer
print("Loading tokenizer...")
# Load directly from the Kaggle input path with local_files_only=True
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
print(f"Tokenizer loaded from Kaggle input path: {MODEL_NAME}")


# ===========================
# ENHANCED PROMPT ENGINEERING
# ===========================
def make_prompt_v1(row):
    """Original prompt"""
    return f"""[RULE]: {row['rule']}[SUBREDDET]: {row['subreddit']}[COMMENT]: {row['body']}  [POSITIVE EXAMPLES]:1. {row['positive_example_1']}2. {row['positive_example_2']}[NEGATIVE EXAMPLES]:1. {row['negative_example_1']}2. {row['negative_example_2']}[QUESTION]: Does the comment violate the rule?[ANSWER]:"""

def make_prompt_v2(row):
    """Enhanced prompt with better instructions"""
    return f"""Task: Analyze if the comment violates the subreddit rule based on the examples.Subreddit: r/{row['subreddit']}Rule: {row['rule']}  Examples that VIOLATE this rule:- {row['positive_example_1']}- {row['positive_example_2']}Examples that FOLLOW this rule:- {row['negative_example_1']}- {row['negative_example_2']}Comment to analyze: {row['body']}Based on the pattern in the examples, does this comment violate the rule? Answer:"""

def make_prompt_v3(row):
    """Rule-focused prompt"""
    return f"""Learn from these examples and classify:RULE: {row['rule']}   VIOLATIONS:- {row['positive_example_1']}- {row['positive_example_2']}ACCEPTABLE:- {row['negative_example_1']}- {row['negative_example_2']}CLASSIFY THIS COMMENT: {row['body']}VIOLATES RULE?"""

# ===========================
# DATA PREPROCESSING
# ===========================
print("Preprocessing data...")
# Use primary prompt for training
train['text'] = train.apply(make_prompt_v1, axis=1)

# Split data
train_data, val_data = train_test_split(
    train,
    test_size=0.2,
    random_state=SEED,
    stratify=train['rule_violation'])

# Prepare labels
train_data["label"] = train_data["rule_violation"].astype(float)
val_data["label"] = val_data["rule_violation"].astype(float)

# Create datasets
features_cols = ['text', 'label']
train_ds = Dataset.from_pandas(train_data[features_cols])
val_ds = Dataset.from_pandas(val_data[features_cols])

# ===========================
# TOKENIZATION
# ===========================
def tokenize(batch):
    """Tokenize text with proper padding and truncation"""
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )
print("Tokenizing datasets...")
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)

# ===========================
# METRICS
# ===========================
def compute_column_auc(eval_pred):
    """Compute AUC metric for evaluation"""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))

    if probs.ndim == 1 or probs.shape[1] == 1:
        probs = probs.flatten()
        auc = roc_auc_score(labels, probs)
        return {"auc": auc}

    aucs = []
    for i in range(probs.shape[1]):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = 0.5
        aucs.append(auc)
    return {"mean_column_auc": np.mean(aucs)}

# ===========================
# MODEL INITIALIZATION
# ===========================
print("Loading model...")
# Load directly from the Kaggle input path with local_files_only=True
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    local_files_only=True
)
print(f"Model loaded from Kaggle input path: {MODEL_NAME}")

# Add a check to ensure tokenizer and model are not None
if tokenizer is None:
    raise RuntimeError("Tokenizer could not be loaded.")
if model is None:
    raise RuntimeError("Model could not be loaded.")


# ===========================
# ENHANCED TRAINING ARGUMENTS
# ===========================
training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=25,
    save_total_limit=2,
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False, # False to avoid RuntimeError
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    fp16=True,
    bf16=False,
    report_to="none",
    logging_dir="./logs",
    seed=SEED,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    dataloader_num_workers=2)

# ===========================
# TRAINER
# ===========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_column_auc,)

# ===========================
# TRAINING
# ===========================
print("Starting training...")
# FIXED: Uncommented to enable model training
trainer.train()

# ===========================
# EVALUATION & CALIBRATION PREP
# ===========================
print("Evaluating model...")
results = trainer.evaluate()
print(f"Validation AUC: {results['eval_auc']:.4f}")

# Get validation predictions for calibration
print("Getting validation predictions for calibration...")
val_predictions = trainer.predict(val_ds)
val_probs = torch.sigmoid(torch.tensor(val_predictions.predictions)).numpy().flatten()
val_labels = val_data["label"].values

# ===========================
# TEST TIME AUGMENTATION
# ===========================
print("\nApplying Test Time Augmentation...")
def get_tta_predictions(test_df, prompt_funcs):
    """Get predictions using multiple prompt versions"""
    all_probs = []

    for i, prompt_func in enumerate(prompt_funcs):
        print(f"  Processing TTA version {i+1}/{len(prompt_funcs)}...")
        test_copy = test_df.copy()
        test_copy['text'] = test_copy.apply(prompt_func, axis=1)
        test_ds = Dataset.from_pandas(test_copy[['text']])
        test_ds = test_ds.map(tokenize, batched=True)

        predictions = trainer.predict(test_ds)
        probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy().flatten()
        all_probs.append(probs)

    return all_probs

# Apply TTA with all prompt versions
prompt_funcs = [make_prompt_v1, make_prompt_v2, make_prompt_v3]
tta_probs_list = get_tta_predictions(test, prompt_funcs)

# Weighted average of predictions
weights = [0.5, 0.3, 0.2]
tta_probs = np.average(tta_probs_list, axis=0, weights=weights)
print(f"TTA complete. Correlation between v1 and v2: {np.corrcoef(tta_probs_list[0], tta_probs_list[1])[0,1]:.4f}")

# ===========================
# PROBABILITY CALIBRATION
# ===========================
print("\nCalibrating probabilities...")
# Fit isotonic regression on validation set
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(val_probs, val_labels)

# Calibrate test predictions
calibrated_probs = iso_reg.transform(tta_probs)

# ===========================
# POST-PROCESSING
# ===========================
print("Applying post-processing...")
# Smooth extreme predictions
def smooth_predictions(probs, min_val=0.05, max_val=0.95):
    """Clip extreme predictions to avoid overconfidence"""
    return np.clip(probs, min_val, max_val)
final_probs = smooth_predictions(calibrated_probs)

# ===========================
# SAVE MODEL
# ===========================
# Saving model to the output directory
trainer.save_model(f"./{DIR}/final_model")
tokenizer.save_pretrained(f"./{DIR}/final_model")

# ===========================
# SUBMISSION
# ===========================
print("\nCreating submission file...")
submission = pd.DataFrame({
    "row_id": test.row_id.values,
    "rule_violation": final_probs})
submission.to_csv("submission.csv", index=False)
print("Submission saved!")

# ===========================
# COMPREHENSIVE ANALYSIS
# ===========================
print("\n" + "="*50)
print("FINAL RESULTS ANALYSIS")
print("="*50)
print("\nFirst 10 predictions:")
print(submission.head(10))
print(f"\nPrediction statistics comparison:")
print(f"{'Method':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-"*52)
print(f"{'Original (v1)':<20} {tta_probs_list[0].mean():>8.4f} {tta_probs_list[0].std():>8.4f} {tta_probs_list[0].min():>8.4f} {tta_probs_list[0].max():>8.4f}")
print(f"{'TTA (weighted)':<20} {tta_probs.mean():>8.4f} {tta_probs.std():>8.4f} {tta_probs.min():>8.4f} {tta_probs.max():>8.4f}")
print(f"{'Calibrated':<20} {calibrated_probs.mean():>8.4f} {calibrated_probs.std():>8.4f} {calibrated_probs.min():>8.4f} {calibrated_probs.max():>8.4f}")
print(f"{'Final (smoothed)':<20} {final_probs.mean():>8.4f} {final_probs.std():>8.4f} {final_probs.min():>8.4f} {final_probs.max():>8.4f}")
# Distribution analysis
print(f"\nPrediction distribution (final):")
bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
hist, _ = np.histogram(final_probs, bins=bins)
for i in range(len(bins)-1):
    pct = hist[i] / len(final_probs) * 100
    print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:4d} ({pct:5.1f}%)")
# Confidence analysis
confident_preds = ((final_probs < 0.2) | (final_probs > 0.8)).sum()
print(f"\nConfident predictions (<0.2 or >0.8): {confident_preds} ({confident_preds/len(final_probs)*100:.1f}%)")
print("\n" + "="*50)
print("DONE! Good luck with your submission!")
print("="*50)


