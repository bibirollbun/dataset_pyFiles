# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from scipy.sparse import hstack
import lightgbm as lgb
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# --- Global Configuration and Reproducibility ---
# Set a random seed for reproducibility across runs
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("--- Starting Data Loading ---")
try:
    # Load the datasets directly from the Kaggle input directory
    train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    sample_submission_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

    print("Initial Data Loading Complete.")
    print("Train Data Head:")
    print(train_df.head())
    print("\nTest Data Head:")
    print(test_df.head())

except FileNotFoundError:
    print("Error: Data files not found at specified Kaggle paths.")
    print("Please ensure the dataset 'jigsaw-agile-community-rules' is added to your notebook.")
    print("If you are running this outside Kaggle, adjust the file paths accordingly.")
    print("Exiting script.")
    exit() # Exit the script if essential files are missing
except Exception as e:
    # Catch any other unexpected errors during file loading
    print(f"An unexpected error occurred during data loading: {e}")
    print("Exiting script.")
    exit()


print("\n--- Generating Data Understanding Visualizations ---")

# Define the output directory for general plots.

PLOT_OUTPUT_DIR = "/kaggle/working/data_analysis_plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True) # Ensure the directory exists


# Distribution of Target Variable (rule_violation)

plt.figure(figsize=(6, 4))
sns.countplot(x='rule_violation', data=train_df)
plt.title('Distribution of Rule Violation (Training Data)')
plt.xlabel('Rule Violation (0: No, 1: Yes)')
plt.ylabel('Count')
plt.savefig(os.path.join(PLOT_OUTPUT_DIR, 'rule_violation_distribution.png'))
plt.show()
plt.close() # Close plot to free memory immediately after saving/showing


# Distribution of rule (in training data)

plt.figure(figsize=(10, 6))
sns.countplot(y='rule', data=train_df, order=train_df['rule'].value_counts().index)
plt.title('Distribution of Rules (Training Data)')
plt.xlabel('Count')
plt.ylabel('Rule')
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.savefig(os.path.join(PLOT_OUTPUT_DIR, 'rule_distribution.png'))
plt.show()
plt.close()


# Comment Length Distribution (in words)

# Calculate length of 'body' text in terms of words
train_df['body_length'] = train_df['body'].apply(lambda x: len(str(x).split()))
test_df['body_length'] = test_df['body'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(12, 5))
sns.histplot(train_df['body_length'], bins=50, color='blue', label='Train', kde=True)
sns.histplot(test_df['body_length'], bins=50, color='red', label='Test', kde=True, alpha=0.6)
plt.title('Distribution of Comment Lengths (Words)')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(0, 250) # Limit x-axis to focus on the common range, ignoring extreme outliers
# A rough visual marker for typical Transformer MAX_LENGTH (e.g., 256 tokens ~ 150-200 words)
TRANSFORMER_MAX_LENGTH_VISUAL_MARKER = 256
plt.axvline(x=TRANSFORMER_MAX_LENGTH_VISUAL_MARKER / 1.5, color='green', linestyle='--', label=f'Approx. Max Transformer Word Length ({TRANSFORMER_MAX_LENGTH_VISUAL_MARKER} tokens)')
plt.legend()
plt.savefig(os.path.join(PLOT_OUTPUT_DIR, 'comment_length_distribution.png'))
plt.show()
plt.close()

# Drop the temporary 'body_length' column from dataframes
train_df = train_df.drop(columns=['body_length'])
test_df = test_df.drop(columns=['body_length'])

print("Data Understanding Visualizations Complete.")


# Transformer Model (BERT/RoBERTa based)

print("\n--- Starting Transformer Model Training ---")

# Configuration for the Transformer Model
TRANSFORMER_MODEL_NAME = "bert-base-uncased" # You can experiment with "roberta-base", "microsoft/deberta-base"
TRANSFORMER_MAX_LENGTH = 256 # Maximum sequence length for Transformer input (adjust based on comment length analysis)
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_NUM_EPOCHS = 3 # Start with a small number, increase if performance improves
TRANSFORMER_LEARNING_RATE = 2e-5 # Standard learning rate for fine-tuning
TRANSFORMER_OUTPUT_DIR = "/kaggle/working/results_transformer_model" # Writable directory for model outputs
TRANSFORMER_LOGGING_DIR = "/kaggle/working/logs_transformer_model" # Writable directory for training logs

os.makedirs(TRANSFORMER_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRANSFORMER_LOGGING_DIR, exist_ok=True)

# Custom PyTorch Dataset for Transformer input
class RuleViolationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, is_test=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        comment_body = str(row['body'])
        rule_text = str(row['rule'])

        # Construct the input string for the Transformer model.
        # This format helps the model understand the relationship between comment and rule.
        text_a = f"Comment: {comment_body}"
        text_b = f"Rule: {rule_text}"

        encoding = self.tokenizer(
            text_a,
            text_b,
            truncation=True,        # Truncate sequences longer than max_length
            padding='max_length',   # Pad sequences shorter than max_length
            max_length=self.max_length,
            return_tensors='pt'     # Return PyTorch tensors
        )
        
        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # token_type_ids is for differentiating between text_a and text_b segments in BERT-like models
            'token_type_ids': encoding['token_type_ids'].flatten()
        }

        if not self.is_test:
            label = torch.tensor(row['rule_violation'], dtype=torch.long)
            inputs['labels'] = label # Include labels only for training/validation

        return inputs


# Initialize the tokenizer and the sequence classification model
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
# num_labels=2 for binary classification (violation or no violation)
model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_NAME, num_labels=2)


# Prepare Data splits for Transformer training and validation
# Stratify to maintain the proportion of 'rule_violation' in both train and validation sets
train_df_transformer, val_df_transformer = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df['rule_violation']
)

# Create Dataset objects for each split
train_dataset_transformer = RuleViolationDataset(train_df_transformer, tokenizer, TRANSFORMER_MAX_LENGTH)
val_dataset_transformer = RuleViolationDataset(val_df_transformer, tokenizer, TRANSFORMER_MAX_LENGTH)
test_dataset_transformer = RuleViolationDataset(test_df, tokenizer, TRANSFORMER_MAX_LENGTH, is_test=True)


from transformers import TrainingArguments

import transformers
print(transformers.__version__)


# Define Metrics (AUC) for Hugging Face Trainer
def compute_metrics_transformer(p):
    predictions, labels = p
    # Ensure labels are flat and numpy array
    labels_flat = np.array(labels).flatten()

    # Determine predictions for the positive class (assuming binary classification)
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        preds = np.array(predictions[:, 1])
    else:
        # Fallback for single logit output, ensure it's a flat numpy array
        preds = np.array(predictions).flatten()

    # Initialize results dictionary
    results = {}

    # Check for conditions where AUC cannot be calculated (e.g., all same label)
    if len(np.unique(labels_flat)) < 2:
        print(f"Warning: Only one class present in labels. AUC cannot be calculated. Labels: {np.unique(labels_flat)}")
        results["auc"] = 0.0 # Assign a default value
    else:
        try:
            auc_score = roc_auc_score(labels_flat, preds)
            results["auc"] = auc_score
        except ValueError as e:
            # Catch potential ValueError from roc_auc_score (e.g., if predictions are invalid)
            print(f"Error calculating AUC: {e}. Labels: {np.unique(labels_flat)}, Predictions min/max: {preds.min()}/{preds.max()}")
            results["auc"] = 0.0 # Assign a default value

    return results


# Configure Training Arguments for the Hugging Face Trainer
training_args_transformer = TrainingArguments(
    output_dir=TRANSFORMER_OUTPUT_DIR,
    logging_dir=TRANSFORMER_LOGGING_DIR,
    num_train_epochs=TRANSFORMER_NUM_EPOCHS,
    per_device_train_batch_size=TRANSFORMER_BATCH_SIZE,
    per_device_eval_batch_size=TRANSFORMER_BATCH_SIZE,
    learning_rate=TRANSFORMER_LEARNING_RATE,
    eval_strategy="epoch", # <--- CHANGE THIS LINE: was evaluation_strategy
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    seed=SEED,
    dataloader_num_workers=os.cpu_count() // 2 if os.cpu_count() else 0,
    report_to="none"
)


# Create the Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args_transformer,
    train_dataset=train_dataset_transformer,
    eval_dataset=val_dataset_transformer,
    compute_metrics=compute_metrics_transformer,
)

print("Starting Transformer model training...")
# Start the training process
trainer.train()
print("Transformer model training finished.")

# --- Training Process Visualizations for Transformer ---
print("\n--- Generating Transformer Training Visualizations ---")

# Extract logging history from the trainer's state for plotting
log_history_transformer = trainer.state.log_history
eval_logs_transformer = [entry for entry in log_history_transformer if 'eval_loss' in entry and 'eval_auc' in entry]
train_logs_transformer = [entry for entry in log_history_transformer if 'loss' in entry and 'learning_rate' in entry]

epochs_transformer = [entry['epoch'] for entry in eval_logs_transformer]
eval_losses_transformer = [entry['eval_loss'] for entry in eval_logs_transformer]
eval_aucs_transformer = [entry['eval_auc'] for entry in eval_logs_transformer]

# For training loss, the 'loss' entry is often an average over logging steps within an epoch
train_epochs_loss_transformer = [entry['epoch'] for entry in train_logs_transformer]
train_losses_transformer = [entry['loss'] for entry in train_logs_transformer]


# Plot Training Loss and Validation Loss over Epochs
plt.figure(figsize=(10, 6))
plt.plot(train_epochs_loss_transformer, train_losses_transformer, label='Training Loss', marker='o')
plt.plot(epochs_transformer, eval_losses_transformer, label='Validation Loss', marker='x')
plt.title('Transformer: Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(TRANSFORMER_OUTPUT_DIR, 'transformer_loss_over_epochs.png'))
plt.show()
plt.close()


# Plot Validation AUC over Epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs_transformer, eval_aucs_transformer, label='Validation AUC', marker='o', color='green')
plt.title('Transformer: Validation AUC Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(TRANSFORMER_OUTPUT_DIR, 'transformer_auc_over_epochs.png'))
plt.show()
plt.close()


# Make Predictions on the Validation Set (for performance plots) and Test Set (for submission)
val_predictions_output_transformer = trainer.predict(val_dataset_transformer)
# Apply softmax to logits to get probabilities for the positive class (index 1)
val_probabilities_transformer = torch.softmax(torch.tensor(val_predictions_output_transformer.predictions), dim=-1)[:, 1].numpy()
val_true_labels_transformer = val_df_transformer['rule_violation'].values

test_predictions_output_transformer = trainer.predict(test_dataset_transformer)
test_probabilities_transformer = torch.softmax(torch.tensor(test_predictions_output_transformer.predictions), dim=-1)[:, 1].numpy()



# --- Model Performance Visualizations (on Transformer Validation Set) ---
print("\n--- Generating Transformer Model Performance Visualizations ---")

# ROC Curve
if len(np.unique(val_true_labels_transformer)) > 1: # Ensure sufficient classes for ROC calculation
    fpr_transformer, tpr_transformer, _ = roc_curve(val_true_labels_transformer, val_probabilities_transformer)
    roc_auc_transformer = auc(fpr_transformer, tpr_transformer)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_transformer, tpr_transformer, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_transformer:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal dashed line for random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Transformer: Receiver Operating Characteristic (ROC) Curve (Validation Set)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(TRANSFORMER_OUTPUT_DIR, 'transformer_roc_curve_validation.png'))
    plt.show()
    plt.close()
else:
    print("Cannot plot ROC curve for Transformer: Only one class present in validation true labels.")



# Precision-Recall Curve
if len(np.unique(val_true_labels_transformer)) > 1: # Ensure sufficient classes for PR calculation
    precision_transformer, recall_transformer, _ = precision_recall_curve(val_true_labels_transformer, val_probabilities_transformer)
    pr_auc_transformer = auc(recall_transformer, precision_transformer)

    plt.figure(figsize=(8, 8))
    plt.plot(recall_transformer, precision_transformer, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc_transformer:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Transformer: Precision-Recall Curve (Validation Set)')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(TRANSFORMER_OUTPUT_DIR, 'transformer_precision_recall_curve_validation.png'))
    plt.show()
    plt.close()
else:
    print("Cannot plot Precision-Recall curve for Transformer: Only one class present in validation true labels.")

# Prepare Transformer Submission File
submission_df_transformer = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': test_probabilities_transformer})
# Save submission file to /kaggle/working/ to ensure it's available in output
submission_df_transformer.to_csv("/kaggle/working/submission_transformer.csv", index=False)
print("\nSubmission file 'submission_transformer.csv' created successfully!")
print(submission_df_transformer.head())

print("--- Transformer Model Training and Prediction Complete ---")


print("\n--- Starting LightGBM + TF-IDF Model Training ---")

# Configuration for LightGBM
LGBM_OUTPUT_DIR = "/kaggle/working/results_lgbm_tfidf_model" # Writable directory for model outputs
os.makedirs(LGBM_OUTPUT_DIR, exist_ok=True)

# TF-IDF Parameters
TFIDF_MAX_FEATURES_BODY = 10000 # Max features for comment body TF-IDF
TFIDF_MAX_FEATURES_RULE = 1000  # Max features for rule text TF-IDF (rules are typically shorter)
TFIDF_NGRAM_RANGE = (1, 2)      # Include single words (unigrams) and two-word phrases (bigrams)
TFIDF_MIN_DF = 5                # Ignore terms that appear in fewer than 5 documents (helps reduce noise)

# LightGBM Classifier Parameters
LGBM_PARAMS = {
    'objective': 'binary',      # Binary classification
    'metric': 'auc',            # Optimize for AUC
    'n_estimators': 1000,       # Number of boosting rounds (trees)
    'learning_rate': 0.05,      # Step size shrinkage to prevent overfitting
    'num_leaves': 31,           # Max number of leaves in one tree (control tree complexity)
    'max_depth': -1,            # No limit on tree depth (can be set to a positive integer to prevent overfitting)
    'seed': SEED,               # Random seed for reproducibility
    'n_jobs': -1,               # Use all available CPU cores for training
    'verbose': -1,              # Suppress verbose output during training
    'colsample_bytree': 0.8,    # Fraction of features randomly selected for each tree
    'subsample': 0.8,           # Fraction of data randomly sampled for each tree
    'reg_alpha': 0.1,           # L1 regularization (Lasso)
    'reg_lambda': 0.1,          # L2 regularization (Ridge)
}
EARLY_STOPPING_ROUNDS = 50 # Stop training if validation AUC doesn't improve for this many rounds



# Feature Engineering with TF-IDF
# Create temporary copies and fill NaNs to avoid modifying original DataFrames and for robust TF-IDF
temp_train_df_lgbm = train_df.copy()
temp_test_df_lgbm = test_df.copy()

temp_train_df_lgbm['body'] = temp_train_df_lgbm['body'].astype(str).fillna('')
temp_train_df_lgbm['rule'] = temp_train_df_lgbm['rule'].astype(str).fillna('')
temp_test_df_lgbm['body'] = temp_test_df_lgbm['body'].astype(str).fillna('')
temp_test_df_lgbm['rule'] = temp_test_df_lgbm['rule'].astype(str).fillna('')

# Concatenate all comment bodies and all rule texts to ensure a consistent vocabulary
# This is crucial for TF-IDF to generalize across train/test and seen/unseen rules.
all_comments_lgbm = pd.concat([temp_train_df_lgbm['body'], temp_test_df_lgbm['body']])
all_rules_lgbm = pd.concat([temp_train_df_lgbm['rule'], temp_test_df_lgbm['rule']])

# TF-IDF for comment bodies
tfidf_vectorizer_body = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES_BODY,
    ngram_range=TFIDF_NGRAM_RANGE,
    min_df=TFIDF_MIN_DF,
    stop_words='english',
    sublinear_tf=True
)
X_train_body_lgbm = tfidf_vectorizer_body.fit_transform(temp_train_df_lgbm['body'])
X_test_body_lgbm = tfidf_vectorizer_body.transform(temp_test_df_lgbm['body'])

# TF-IDF for rule texts. Fit on all rules (train and test) for better generalization to unseen rules.
tfidf_vectorizer_rule = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES_RULE,
    ngram_range=TFIDF_NGRAM_RANGE,
    min_df=1, # Keep even rare words in rule descriptions as they might be very specific
    stop_words='english',
    sublinear_tf=True
)
tfidf_vectorizer_rule.fit(all_rules_lgbm) # Fit on combined rules
X_train_rule_lgbm = tfidf_vectorizer_rule.transform(temp_train_df_lgbm['rule'])
X_test_rule_lgbm = tfidf_vectorizer_rule.transform(temp_test_df_lgbm['rule'])



# Combine features by horizontally stacking the sparse TF-IDF matrices
X_train_lgbm = hstack([X_train_body_lgbm, X_train_rule_lgbm])
X_test_lgbm = hstack([X_test_body_lgbm, X_test_rule_lgbm])
y_train_lgbm = temp_train_df_lgbm['rule_violation']

print(f"Combined TF-IDF features for LightGBM: Train shape {X_train_lgbm.shape}, Test shape {X_test_lgbm.shape}")

# Split Data for Training and Validation for LightGBM
X_train_split_lgbm, X_val_split_lgbm, y_train_split_lgbm, y_val_split_lgbm = train_test_split(
    X_train_lgbm, y_train_lgbm, test_size=0.1, random_state=SEED, stratify=y_train_lgbm
)

# Custom evaluation function for AUC for LightGBM (needed for eval_metric during fit)
def lgb_auc(y_true, y_pred):
    # LightGBM passes raw scores (logits), so apply sigmoid to get probabilities
    y_pred_proba = 1 / (1 + np.exp(-y_pred))
    y_true = y_true.astype(int) # Ensure true labels are integers

    if len(np.unique(y_true)) < 2:
        return 'auc', 0.0, True # Metric name, metric value, higher_is_better
    
    return 'auc', roc_auc_score(y_true, y_pred_proba), True



# Initialize LightGBM Classifier
lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)

# Define evaluation sets and callbacks for LightGBM training
eval_set_lgbm = [(X_train_split_lgbm, y_train_split_lgbm), (X_val_split_lgbm, y_val_split_lgbm)]
callbacks_lgbm = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=50)]

print("Starting LightGBM model training...")
# Fit the LightGBM model
lgb_model.fit(
    X_train_split_lgbm, y_train_split_lgbm,
    eval_set=eval_set_lgbm,
    eval_metric=lgb_auc, # Use our custom AUC metric
    callbacks=callbacks_lgbm # Apply early stopping and logging callbacks
)
print("LightGBM model training finished.")

# --- Training Process Visualizations for LightGBM ---
print("\n--- Generating LightGBM Training Visualizations ---")



# LightGBM Classifier Parameters
LGBM_PARAMS = {
    'objective': 'binary',      # Binary classification
    'metric': ['auc', 'binary_logloss'], # <--- ADD 'binary_logloss' here
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}
EARLY_STOPPING_ROUNDS = 50

# ... (TF-IDF Feature Engineering code as before) ...

# Initialize LightGBM Classifier
lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)

# Define evaluation sets and callbacks for LightGBM training
eval_set_lgbm = [(X_train_split_lgbm, y_train_split_lgbm), (X_val_split_lgbm, y_val_split_lgbm)]
callbacks_lgbm = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=50)]

print("Starting LightGBM model training...")
# Fit the LightGBM model -- THIS MUST COME BEFORE PLOTTING evals_result_
lgb_model.fit(
    X_train_split_lgbm, y_train_split_lgbm,
    eval_set=eval_set_lgbm,
    eval_metric=lgb_auc, # Use our custom AUC metric
    callbacks=callbacks_lgbm # Apply early stopping and logging callbacks
)
print("LightGBM model training finished.")

# Now that the model has been fitted, evals_result_ will be populated
# LightGBM stores evaluation results in the 'evals_result_' attribute
evals_result_lgbm = lgb_model.evals_result_

# Plot Training and Validation AUC over Boosting Rounds
plt.figure(figsize=(10, 6))
plt.plot(evals_result_lgbm['training']['auc'], label='Training AUC', color='blue')
plt.plot(evals_result_lgbm['valid_1']['auc'], label='Validation AUC', color='red') # 'valid_1' refers to the validation set
plt.title('LightGBM: Training and Validation AUC Over Boosting Rounds')
plt.xlabel('Boosting Round')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(LGBM_OUTPUT_DIR, 'lgbm_auc_over_rounds.png'))
plt.show()
plt.close()





# Plot Training and Validation Binary Logloss over Boosting Rounds
plt.figure(figsize=(10, 6))
plt.plot(evals_result_lgbm['training']['binary_logloss'], label='Training Logloss', color='blue')
plt.plot(evals_result_lgbm['valid_1']['binary_logloss'], label='Validation Logloss', color='red')
plt.title('LightGBM: Training and Validation Logloss Over Boosting Rounds')
plt.xlabel('Boosting Round')
plt.ylabel('Logloss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(LGBM_OUTPUT_DIR, 'lgbm_logloss_over_rounds.png'))
plt.show()
plt.close()

# Make Predictions on the Validation Set (for performance plots) and Test Set (for submission) for LightGBM
val_probabilities_lgbm = lgb_model.predict_proba(X_val_split_lgbm)[:, 1] # Get probabilities for the positive class
val_true_labels_lgbm = y_val_split_lgbm.values

test_probabilities_lgbm = lgb_model.predict_proba(X_test_lgbm)[:, 1]


# --- Model Performance Visualizations (on LightGBM Validation Set) ---
print("\n--- Generating LightGBM Model Performance Visualizations ---")

# ROC Curve
if len(np.unique(val_true_labels_lgbm)) > 1: # Ensure sufficient classes for ROC calculation
    fpr_lgbm, tpr_lgbm, _ = roc_curve(val_true_labels_lgbm, val_probabilities_lgbm)
    roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_lgbm, tpr_lgbm, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_lgbm:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM: Receiver Operating Characteristic (ROC) Curve (Validation Set)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(LGBM_OUTPUT_DIR, 'lgbm_roc_curve_validation.png'))
    plt.show()
    plt.close()
else:
    print("Cannot plot ROC curve for LightGBM: Only one class present in validation true labels.")




# Precision-Recall Curve
if len(np.unique(val_true_labels_lgbm)) > 1: # Ensure sufficient classes for PR calculation
    precision_lgbm, recall_lgbm, _ = precision_recall_curve(val_true_labels_lgbm, val_probabilities_lgbm)
    pr_auc_lgbm = auc(recall_lgbm, precision_lgbm)

    plt.figure(figsize=(8, 8))
    plt.plot(recall_lgbm, precision_lgbm, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc_lgbm:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM: Precision-Recall Curve (Validation Set)')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(LGBM_OUTPUT_DIR, 'lgbm_precision_recall_curve_validation.png'))
    plt.show()
    plt.close()
else:
    print("Cannot plot Precision-Recall curve for LightGBM: Only one class present in validation true labels.")



# Prepare LightGBM Submission File
submission_df_lgbm = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': test_probabilities_lgbm})
# Save submission file to /kaggle/working/ to ensure it's available in output
submission_df_lgbm.to_csv("/kaggle/working/submission_lgbm_tfidf.csv", index=False)
print("\nSubmission file 'submission_lgbm_tfidf.csv' created successfully!")
print(submission_df_lgbm.head())

print("--- LightGBM + TF-IDF Model Training and Prediction Complete ---")

print("\nAll models run and visualizations generated. Check the respective output directories in /kaggle/working/.")


# Assuming X_test_lgbm is your preprocessed test features for LightGBM
# (This would be created after TF-IDF vectorization of test_df['combined_text_lgbm'])

print("\n--- Making LightGBM Predictions on Test Set ---")
# Make Predictions on the Test Set (for submission) for LightGBM
test_probabilities_lgbm = lgb_model.predict_proba(X_test_lgbm)[:, 1] # Get probabilities for the positive class
print("LightGBM predictions on test set completed.")



print("\nGenerating submission file...")


#  the upstream execution
if 'test_probabilities_lgbm' not in locals():
    print("Error: 'test_probabilities_lgbm' is not defined. Please ensure LightGBM predictions are generated.")
    
    test_probabilities_lgbm = np.zeros(len(test_df)) if 'test_df' in locals() else np.zeros(10)

if 'test_probabilities_transformer' not in locals():
    print("Error: 'test_probabilities_transformer' is not defined. Please ensure Transformer predictions are generated.")

    test_probabilities_transformer = np.zeros(len(test_df)) if 'test_df' in locals() else np.zeros(10)

if 'test_df' not in locals():
    print("Error: 'test_df' is not defined. Cannot create submission file without row_ids.")
    # For now, assigning a dummy DataFrame
    test_df = pd.DataFrame({'row_id': range(len(test_probabilities_lgbm))})


# Combine the predictions
# Ensure both prediction arrays have the same length
if len(test_probabilities_lgbm) != len(test_probabilities_transformer):
    print("WARNING: Prediction arrays have different lengths. Ensembling might be inaccurate.")
    
    min_len = min(len(test_probabilities_lgbm), len(test_probabilities_transformer))
    ensemble_predictions = (test_probabilities_lgbm[:min_len] + test_probabilities_transformer[:min_len]) / 2
else:
    ensemble_predictions = (test_probabilities_lgbm + test_probabilities_transformer) / 2



submission_df = pd.DataFrame({
    'row_id': test_df['row_id'], 
    'rule_violation': ensemble_predictions 
})


SUBMISSION_DIR = "/kaggle/working/"
SUBMISSION_FILENAME = "submission.csv" 
os.makedirs(SUBMISSION_DIR, exist_ok=True) 


submission_file_path = os.path.join(SUBMISSION_DIR, SUBMISSION_FILENAME)
submission_df.to_csv(submission_file_path, index=False) 

print(f"Submission file '{SUBMISSION_FILENAME}' created successfully!")
print("submission.csv:")
print(submission_df.head())
print("\n--- Script Finished ---")

