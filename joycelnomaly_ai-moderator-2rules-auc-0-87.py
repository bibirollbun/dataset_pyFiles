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


import warnings
warnings.filterwarnings('ignore')


!pip install -U transformers # Fixes the problem of HTTPS error 404 


import matplotlib.pyplot as plt
import seaborn as sns
import re
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer



try:
    df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv not found. Please check the file path.")
    df = pd.DataFrame() # Create an empty DataFrame to prevent errors


print("\n--- DataFrame Info ---")
df.info()


print("\n--- First 5 Rows ---")
print(df[['row_id', 'body', 'rule', 'rule_violation']].head())


# --- Rule Analysis: Identify unique rules ---
print("\n--- Identify Unique Rules ---")
unique_rules = df['rule'].nunique()
print(f"\n--- Rule Analysis ---")
print(f"Total number of unique rules: {unique_rules}")

# Show the distinct rule names and their counts
rule_counts = df['rule'].value_counts()
print("\nRule Counts:")
print(rule_counts)


# --- Overall Target Distribution ---
print("\n--- Target Distribution ---")
overall_violation_counts = df['rule_violation'].value_counts(normalize=True) * 100
print("\nOverall Rule Violation Distribution:")
print(overall_violation_counts)
print(f"Target imbalance ratio: {overall_violation_counts[0]:.2f} : {overall_violation_counts[1]:.2f}")


# Visualize Overall Imbalance
plt.figure(figsize=(6, 4))
sns.countplot(x='rule_violation', data=df)
plt.title('Overall Rule Violation (Target) Distribution')
plt.xticks([0, 1], ['No Violation (0)', 'Violation (1)'])
plt.ylabel('Count')
plt.xlabel('Rule Violation Status')
plt.show()


# --- Rule Violation Distribution by Rule ---
print("\n--- Rule Violation Distribution Per Rule (Imbalance Check) ---")
# Group by rule and calculate the mean of rule_violation (which is the violation rate)
violation_rate_per_rule = df.groupby('rule')['rule_violation'].agg(['count', 'mean']).sort_values(by='mean', ascending=False)
violation_rate_per_rule.columns = ['Total Samples', 'Violation Rate (Mean)']
print(violation_rate_per_rule) 


# --- Comment Body Length Analysis ---
df['body_length'] = df['body'].apply(len)
df['body_word_count'] = df['body'].apply(lambda x: len(str(x).split()))

print("\n--- Comment Body Length Statistics (in characters) ---")
print(df['body_length'].describe())

# Visualize the length distributions for Violations vs. Non-Violations
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=df, x='body_length', hue='rule_violation', kde=True, bins=50)
plt.title('Comment Length Distribution')
plt.xlabel('Character Count')

plt.subplot(1, 2, 2)
sns.boxplot(x='rule_violation', y='body_length', data=df)
plt.title('Comment Length by Violation Status')
plt.xticks([0, 1], ['No Violation (0)', 'Violation (1)'])

plt.tight_layout()
plt.show()


# --- Rule Text Length Analysis ---
df['rule_length'] = df['rule'].apply(len)
print("\n--- Rule Text Length Statistics (in characters) ---")
print(df['rule_length'].describe())


# Create the combined feature text for the Transformer input
def remove_links_for_display(text):
    """
    Temporarily removes common URLs from text for clean printing/display only.
    """ 
    url_pattern = re.compile(r'https?://\S+|www\.\S+|\S+\.(com|org|net|gov|edu|co)\b')
    text = url_pattern.sub(r'[LINK REMOVED]', text)
    return text


def create_transformer_input(row):
    """
    Creates a structured text prompt combining the comment, rule, and examples.
    This is the input feature for RoBERTa.
    """
    
    # 1. Rule Context
    rule_text = f"RULE: {row['rule']}"
    
    # 2. Positive Examples (What IS a violation)
    pos_examples = (
        f"POSITIVE EXAMPLES (Violation): "
        f"{row['positive_example_1']} | "
        f"{row['positive_example_2']}"
    )
    
    # 3. Negative Examples (What is NOT a violation)
    neg_examples = (
        f"NEGATIVE EXAMPLES (No Violation): "
        f"{row['negative_example_1']} | "
        f"{row['negative_example_2']}"
    )
    
    # 4. The Comment to be Classified
    comment_body = f"COMMENT TO CLASSIFY: {row['body']}"
    
    # Combine all parts with clear separators. The Transformer will learn 
    # the relationship between these segments.
    # The [SEP] token will be added by the tokenizer later.
    
    combined_text = f"{comment_body} [SEP] {rule_text} [SEP] {pos_examples} [SEP] {neg_examples}"
    return combined_text

# --- Application to the dataset ---
# Apply the function to create the new input feature column
df['model_input_text'] = df.apply(create_transformer_input, axis=1)

# Display a sample input to verify the structure
print("\n--- Sample Model Input Text ---")
print(remove_links_for_display(df['model_input_text'].iloc[1]))
print("\n--- Length of Sample Input ---")
print(f"Length of sample input: {len(df['model_input_text'].iloc[1])} characters.")


# Perform basic cleaning 
def basic_clean(text):
    """Minimal cleaning for Transformer input."""
    # Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Application to the dataset ---
# Apply cleaning to the combined text
df['model_input_text_cleaned'] = df['model_input_text'].apply(basic_clean)

# Also apply cleaning to the raw comment body, just in case
df['body_cleaned'] = df['body'].apply(basic_clean)

print("\n--- Sample Cleaned Model Input Text ---")
print(remove_links_for_display(df['model_input_text_cleaned'].iloc[1]))


# Define features (X) and target (y)
X = df['model_input_text_cleaned']
y = df['rule_violation']

# Split the small labeled dataset into Training and Validation sets
# Use stratification to ensure the 50/50 balance is maintained in both sets.
# A small validation set is acceptable for the initial fine-tuning.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, # Use 20% for validation
    random_state=42, 
    stratify=y # Stratify because the balance is perfect
)

print("\n--- Training/Validation Split Summary ---")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Training Violation Rate: {y_train.mean():.4f}")
print(f"Validation Violation Rate: {y_val.mean():.4f}")


# --- Configuration --- 
BATCH_SIZE = 4 # Adjust based on your GPU memory
ACCUMULATION_STEPS = 4 
NUM_EPOCHS = 3
OUTPUT_DIR = './model' # Define the target directory once
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


MODEL_CONFIGS = {
    # 1. RoBERTa Configuration (Default, stable, no special flags)
    "roberta-base": {
        "name": 'roberta-base',
        "trust_remote_code": False,
        "max_len": 512,
        "special_token_handling": None # No special handling needed
    },
    
    # 2. Qwen Configuration (Requires special flags and token handling)
    "qwen-0.5b": {
        "name": 'Qwen/Qwen1.5-0.5B',
        "trust_remote_code": True, # CRITICAL for Qwen models
        "max_len": 512, # Qwen can go higher, but 512 is safe
        "special_token_handling": "pad_to_eos" # Custom instruction
    }
}

# --- Select the Model Here ---
# To switch models, change this string.
SELECTED_MODEL = "roberta-base" # "roberta-base" | "qwen-0.5b"

CONFIG = MODEL_CONFIGS[SELECTED_MODEL]
MODEL_NAME = CONFIG["name"]
MAX_LEN = CONFIG["max_len"]
OUTPUT_DIR = f'./final_model_acge_{SELECTED_MODEL.replace("/", "_").lower()}'
# -----------------------------

# --- Tokenization and Encoding ---# Load the tokenizer, applying model-specific flags
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=CONFIG["trust_remote_code"]
)

# Handle special token requirements (Qwen Fix)
if CONFIG["special_token_handling"] == "pad_to_eos" and tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Warning: Set pad_token to eos_token for {MODEL_NAME}")

def encode_data(texts, tokenizer, max_len):
    """Tokenizes and prepares data for the Transformer model."""
    return tokenizer.batch_encode_plus(
        texts.tolist(),
        add_special_tokens=True,      # Add [CLS] and [SEP]
        max_length=max_len,           # Pad/truncate to MAX_LEN
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'           # Return PyTorch tensors
    )

# Encode Training and Validation sets
print("Encoding Training Data...")
train_encodings = encode_data(X_train, tokenizer, MAX_LEN)
print("Encoding Validation Data...")
val_encodings = encode_data(X_val, tokenizer, MAX_LEN)

# Convert labels (y_train/y_val) to PyTorch tensors
train_labels = torch.tensor(y_train.values)
val_labels = torch.tensor(y_val.values)

# --- Custom Dataset Class ---
class RedditCommentDataset(Dataset):
    """Custom Dataset to correctly package data as dictionaries for the Trainer."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Package inputs and labels into a dictionary
        # Keys MUST match the model's forward pass arguments (input_ids, attention_mask)
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create the corrected PyTorch Dataset objects
train_dataset = RedditCommentDataset(train_encodings, train_labels)
val_dataset = RedditCommentDataset(val_encodings, val_labels)
print("Data successfully packaged into custom PyTorch Dataset format.")


import shutil
import os

# Delete the log directories and temporary results from previous runs
# This is often where many small, unnecessary files accumulate.

# 1. Delete the logs directory
if os.path.exists('./logs'):
    shutil.rmtree('./logs')
    print("Cleaned up './logs' directory.")

# 2. Delete the results directory from the Trainer
if os.path.exists('./results_manual_eval'):
    shutil.rmtree('./results_manual_eval')
    print("Cleaned up './results_manual_eval' directory.")
    
# 3. Check for the temporary prediction directory and delete
if os.path.exists('./temp_predict'):
    shutil.rmtree('./temp_predict')
    print("Cleaned up './temp_predict' directory.")

# Now, try rerunning the trainer.train() and trainer.save_model() steps.


if os.path.exists(OUTPUT_DIR):
    # Model already exists, load it
    print(f"\n✅ Model found at {OUTPUT_DIR}. Loading saved model weights.")
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    TENSORFLOW_MODE = False # Hugging Face requires this to be set if training is skipped
    
else:
    # Model does not exist, initialize and train
    print("\n⏳ Saved model not found. Starting fine-tuning process.")
    
    # Initialize the model from scratch
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        trust_remote_code=CONFIG["trust_remote_code"] # Dynamically applied
    )
    
    # Define Training Arguments (Manual Eval)
    TRAINING_ARGS = TrainingArguments(
        output_dir='./results_manual_eval',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS, 
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="no",             # Manual Evaluation
        save_strategy="epoch",
        save_safetensors=False,
        load_best_model_at_end=False,
        report_to="none"
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Start Training
    trainer.train()

    # Save Final Model (if trained)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel and tokenizer saved to: {OUTPUT_DIR}")



# Ensure the model is on the correct device for prediction
model.to(DEVICE)

print("\n--- Performing Final Manual Validation (AUC Calculation) ---")

# Re-initialize the Trainer here to ensure the latest model weights 
# (either loaded or just trained) are used for the prediction method.

# Define minimal Training Arguments for the prediction step
PREDICT_ARGS = TrainingArguments(
    output_dir='./temp_predict',
    per_device_eval_batch_size=BATCH_SIZE,
    report_to="none"
)

predict_trainer = Trainer(
    model=model,
    args=PREDICT_ARGS,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# 1. Get predictions (logits and labels)
predictions = predict_trainer.predict(val_dataset)

# Unpack the predictions
logits = predictions.predictions
labels = predictions.label_ids

# 2. Calculate Probabilities and AUC
probabilities = softmax(logits, axis=1)
probabilities_for_auc = probabilities[:, 1] # Probability for the positive class (1)

# Calculate the final AUC
auc_score = roc_auc_score(labels, probabilities_for_auc)

print(f"\nFinal Validation AUC Score: {auc_score:.4f}")

# Clean up temporary directory
if os.path.exists('./temp_predict'):
    shutil.rmtree('./temp_predict')



# Assuming 'predictions' and 'labels' are available from the manual evaluation
# and the original 'df' contains the 'body' and 'rule' columns.

# 1. Convert predictions/labels to a DataFrame
analysis_data = pd.DataFrame({
    'true_label': labels,
    'prob_violation': probabilities_for_auc
})

# 2. Merge with original comment data
# Assume the order of samples in val_dataset matches the order of X_val/y_val, 
# and thus, the order of 'predictions'. Merge based on the index of X_val.
val_indices = X_val.index
analysis_data.index = val_indices
analysis_df = df.loc[val_indices].copy()
analysis_df = analysis_df.merge(analysis_data, left_index=True, right_index=True)

# 3. Create the prediction column (using a simple threshold of 0.5)
THRESHOLD = 0.5 
analysis_df['predicted_label'] = (analysis_df['prob_violation'] >= THRESHOLD).astype(int)

# 4. Create an Error Type column
def get_error_type(row):
    if row['true_label'] == row['predicted_label']:
        return 'Correct'
    elif row['true_label'] == 0 and row['predicted_label'] == 1:
        return 'False Positive (FP)' # Model cried violation, but none existed
    else: # true_label == 1 and predicted_label == 0
        return 'False Negative (FN)' # Model missed a true violation

analysis_df['error_type'] = analysis_df.apply(get_error_type, axis=1)

print("Analysis DataFrame created. Error distribution:")
print(analysis_df['error_type'].value_counts(normalize=True))


# Filter for High-Confidence False Positives (e.g., probability > 0.9)
fp_df = analysis_df[analysis_df['error_type'] == 'False Positive (FP)'].sort_values(by='prob_violation', ascending=False)

print("\n--- Top 5 False Positives (Highest Confidence) ---")
for i, row in fp_df.head(5).iterrows():
    print(f"\nRule: {row['rule']}")
    print(f"Comment: {remove_links_for_display(row['body'])}")
    print(f"Prob. Violation: {row['prob_violation']:.4f}")
    # print the rule and comment body


# Filter for Low-Confidence False Negatives (e.g., probability < 0.1)
fn_df = analysis_df[analysis_df['error_type'] == 'False Negative (FN)'].sort_values(by='prob_violation', ascending=True)

print("\n--- Top 5 False Negatives (Lowest Confidence) ---")
for i, row in fn_df.head(5).iterrows():
    print(f"\nRule: {row['rule']}")
    print(f"Comment: {remove_links_for_display(row['body'])}")
    print(f"Prob. Violation: {row['prob_violation']:.4f}")
    # print the rule and comment body

