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








# =============================================================================
# DEBERTA-V3-LARGE IMPLEMENTATION - REDDIT RULE VIOLATION CLASSIFICATION
# Using Microsoft's DeBERTa-v3-large for superior text classification performance
# =============================================================================

# STEP 1: Install required dependencies (run in separate cells)
print("Installing DeBERTa-v3-large dependencies...")

# Standard installation for DeBERTa
#!pip install transformers==4.35.0 torch torchvision torchaudio

# Additional ML dependencies
#!pip install datasets==2.14.0 scikit-learn pandas numpy accelerate

print("âœ… Installation complete! Please restart runtime before proceeding.")

# =============================================================================
# STEP 2: IMPORTS & CONFIGURATION
# =============================================================================

import kagglehub
kagglehub.login()

# Download competition data
jigsaw_agile_community_rules_path = "/kaggle/input/jigsaw-agile-community-rules/" #kagglehub.competition_download('jigsaw-agile-community-rules')
print('âœ… Data source import complete.')

import os
import random
import pandas as pd
import numpy as np
import torch
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
warnings.filterwarnings('ignore')

# DeBERTa-v3-large Official Imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# DeBERTa-v3-large Configuration
MODEL_NAME = "/kaggle/input/microsof-deberta-v3-large/pytorch/default/1/deberta_v3_large_finetuned" #"microsoft/deberta-v3-large"
SEED = 42
MAX_LEN = 512  # DeBERTa-v3-large supports up to 512 tokens
EPOCHS = 3  # Reduced due to larger model size
LEARNING_RATE = 1e-5  # Lower learning rate for large model
BATCH_SIZE = 2  # Smaller batch size for large model (adjust based on GPU memory)
GRADIENT_ACCUMULATION_STEPS = 4  # To maintain effective batch size
WARMUP_STEPS = 100

# Setup
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ğŸ”§ Device: {device}")
print(f"ğŸ¤– Model: {MODEL_NAME}")
print(f"ğŸ“� Max Length: {MAX_LEN}")
print(f"âš¡ Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"ğŸ�¯ Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

# =============================================================================
# STEP 3: LOAD DEBERTA-V3-LARGE MODEL
# =============================================================================

print("Loading DeBERTa-v3-large model and tokenizer...")

try:
    # Load tokenizer (no trust_remote_code needed for DeBERTa)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("âœ… DeBERTa-v3-large tokenizer loaded successfully")
    
    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,  # Binary classification
        problem_type="regression"  # For probability outputs
    )
    print("âœ… DeBERTa-v3-large model loaded successfully")
    
    # Move to device
    model = model.to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ğŸ“Š Total parameters: {total_params:,}")
    print(f"ğŸ�¯ Trainable parameters: {trainable_params:,}")
    print(f"ğŸ’¾ Model size: ~{total_params * 4 / 1e9:.1f} GB (fp32)")
    
except Exception as e:
    print(f"â�Œ Error loading DeBERTa-v3-large: {e}")
    print("ğŸ’¡ Make sure you have enough GPU memory for this large model")
    print("ğŸ’¡ Consider using microsoft/deberta-v3-base if memory is limited")
    raise

# =============================================================================
# STEP 4: DATA LOADING & PREPROCESSING
# =============================================================================

print("\nLoading and preprocessing data...")

# Load datasets
train_df = pd.read_csv(f"{jigsaw_agile_community_rules_path}/train.csv")
test_df = pd.read_csv(f"{jigsaw_agile_community_rules_path}/test.csv")

print(f"ğŸ“Š Train shape: {train_df.shape}")
print(f"ğŸ“Š Test shape: {test_df.shape}")
print(f"ğŸ“Š Class distribution: {train_df['rule_violation'].value_counts()}")

def make_deberta_prompt(row):
    """
    Create optimized prompt for DeBERTa-v3-large
    Based on the task structure and DeBERTa's capabilities
    """
    prompt = f"""Rule: {row['rule']}
Subreddit: {row['subreddit']}

Comment: {row['body']}

Positive Examples:
1. {row['positive_example_1']}
2. {row['positive_example_2']}

Negative Examples:
1. {row['negative_example_1']}
2. {row['negative_example_2']}

Question: Does the comment violate the rule?
Answer:"""
    return prompt

# Apply prompt creation
print("Creating prompts...")
train_df['text'] = train_df.apply(make_deberta_prompt, axis=1)
test_df['text'] = test_df.apply(make_deberta_prompt, axis=1)

# Analyze prompt lengths
sample_texts = train_df['text'].head(100).tolist()
token_lengths = [len(tokenizer.encode(text, truncation=True, max_length=MAX_LEN)) for text in sample_texts]

print(f"ğŸ“� Average token length: {np.mean(token_lengths):.1f}")
print(f"ğŸ“� Max token length: {np.max(token_lengths)}")
print(f"ğŸ“� 95th percentile: {np.percentile(token_lengths, 95):.1f}")

# =============================================================================
# STEP 5: TRAIN/VALIDATION SPLIT
# =============================================================================

# Stratified split to maintain class balance
train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_df['rule_violation']
)

# Prepare labels
train_split = train_split.copy()
val_split = val_split.copy()
train_split['labels'] = train_split['rule_violation'].astype(float)
val_split['labels'] = val_split['rule_violation'].astype(float)

print(f"ğŸ“Š Train samples: {len(train_split)}")
print(f"ğŸ“Š Validation samples: {len(val_split)}")
print(f"ğŸ“Š Train class balance: {train_split['labels'].value_counts()}")

# =============================================================================
# STEP 6: DATASET CREATION
# =============================================================================

def preprocess_function(examples):
    """Tokenize texts for NeoBERT"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

# Create HuggingFace datasets
train_dataset = Dataset.from_pandas(train_split[['text', 'labels']])
val_dataset = Dataset.from_pandas(val_split[['text', 'labels']])

# Apply tokenization
print("Tokenizing datasets...")
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print("âœ… Datasets prepared successfully")

# =============================================================================
# STEP 7: TRAINING SETUP
# =============================================================================

def compute_metrics(eval_pred):
    """Compute comprehensive metrics"""
    predictions, labels = eval_pred
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-predictions.flatten()))
    
    # Calculate metrics
    auc = roc_auc_score(labels, probs)
    
    # Binary predictions for additional metrics
    binary_preds = (probs > 0.5).astype(int)
    accuracy = accuracy_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds, zero_division=0)
    recall = recall_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)
    
    return {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Training arguments optimized for DeBERTa-v3-large
training_args = TrainingArguments(
    output_dir="./deberta_v3_large_results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    learning_rate=LEARNING_RATE,
    
    # Evaluation and saving
    eval_strategy="steps",
    eval_steps=100,  # Increased due to smaller batch size
    save_strategy="steps",
    save_steps=100,
    logging_steps=50,
    
    # Model selection
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    save_total_limit=2,
    
    # Performance optimizations for large model
    fp16=torch.cuda.is_available(),
    dataloader_drop_last=False,
    max_grad_norm=1.0,  # Gradient clipping for stability
    
    # Logging
    report_to="none",
    logging_dir="./logs",
    seed=SEED,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("âœ… Trainer initialized successfully")

# =============================================================================
# STEP 8: TRAINING
# =============================================================================

print("\nğŸš€ Starting DeBERTa-v3-large fine-tuning...")
print("="*60)

# Train the model
training_output = trainer.train()

print("\nğŸ“Š Training completed!")
print("="*60)

# Final evaluation
print("\nEvaluating final model...")
final_results = trainer.evaluate()

print("\nğŸ“ˆ Final Results:")
for key, value in final_results.items():
    if key.startswith('eval_'):
        metric_name = key.replace('eval_', '').upper()
        print(f"  {metric_name}: {value:.4f}")

# =============================================================================
# STEP 9: SAVE MODEL
# =============================================================================

# Save the fine-tuned model
output_dir = "./deberta_v3_large_finetuned"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nğŸ’¾ Model saved to: {output_dir}")

# =============================================================================
# STEP 10: INFERENCE ON TEST SET
# =============================================================================

print("\nğŸ”® Generating test predictions...")

# Prepare test dataset
print("Preparing test dataset for inference...")
test_dataset = Dataset.from_pandas(test_df[['text']])
test_dataset = test_dataset.map(preprocess_function, batched=True)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

print(f"Test dataset prepared: {len(test_dataset)} samples")

# Generate predictions
print("Running model inference on test set...")
predictions = trainer.predict(test_dataset)
test_probs = 1 / (1 + np.exp(-predictions.predictions.flatten()))

print(f"âœ… Predictions generated for {len(test_probs)} test samples")

# =============================================================================
# STEP 11: CREATE SUBMISSION FILE
# =============================================================================

print("\nğŸ“„ Creating submission file...")

# Verify we have the required columns
print(f"Test DataFrame columns: {test_df.columns.tolist()}")
print(f"Expected row_id column: {'row_id' in test_df.columns}")

# Create submission dataframe
if 'row_id' in test_df.columns:
    submission = pd.DataFrame({
        "row_id": test_df["row_id"].values,
        "rule_violation": test_probs
    })
else:
    # Fallback if row_id column name is different
    id_column = test_df.columns[0]  # Assume first column is ID
    print(f"âš ï¸�  Using {id_column} as ID column")
    submission = pd.DataFrame({
        "row_id": test_df[id_column].values,
        "rule_violation": test_probs
    })

# Save submission file
submission_filename = "/kaggle/working/submission.csv"
submission.to_csv(submission_filename, index=False)
print(f"âœ… Submission saved to: {submission_filename}")

# Verify file was created
import os
if os.path.exists(submission_filename):
    file_size = os.path.getsize(submission_filename)
    print(f"ğŸ“� File size: {file_size:,} bytes")
    print(f"ğŸ“Š Submission shape: {submission.shape}")
else:
    print("â�Œ Error: Submission file was not created!")

# Validate submission format
print("\nğŸ”� Validating submission format...")
required_columns = ['row_id', 'rule_violation']
actual_columns = submission.columns.tolist()

print(f"Required columns: {required_columns}")
print(f"Actual columns: {actual_columns}")

if all(col in actual_columns for col in required_columns):
    print("âœ… Submission format is correct")
else:
    print("â�Œ Submission format issue detected")

# Check for missing values
missing_values = submission.isnull().sum()
if missing_values.sum() == 0:
    print("âœ… No missing values in submission")
else:
    print(f"âš ï¸�  Missing values detected: {missing_values.to_dict()}")

# Check prediction range
print(f"\nğŸ“ˆ Prediction validation:")
print(f"  Min prediction: {test_probs.min():.6f}")
print(f"  Max prediction: {test_probs.max():.6f}")
print(f"  Predictions in [0,1]: {((test_probs >= 0) & (test_probs <= 1)).all()}")

# Additional submission statistics
print(f"\nğŸ“Š Submission Statistics:")
print(f"  Total predictions: {len(submission)}")
print(f"  Unique row_ids: {submission['row_id'].nunique()}")
print(f"  Duplicate row_ids: {submission['row_id'].duplicated().sum()}")
print(f"  Mean prediction: {test_probs.mean():.4f}")
print(f"  Std deviation: {test_probs.std():.4f}")
print(f"  Predictions > 0.5: {(test_probs > 0.5).sum()} / {len(test_probs)} ({(test_probs > 0.5).mean()*100:.1f}%)")

print(f"\nğŸ“„ Submission Preview (first 10 rows):")
print(submission.head(10).to_string(index=False))

print(f"\nğŸ“„ Submission Preview (last 5 rows):")
print(submission.tail(5).to_string(index=False))

# Prediction statistics
print("\nğŸ“Š Prediction Statistics:")
print(f"  Mean prediction: {test_probs.mean():.4f}")
print(f"  Std deviation: {test_probs.std():.4f}")
print(f"  Min prediction: {test_probs.min():.4f}")
print(f"  Max prediction: {test_probs.max():.4f}")
print(f"  Predictions > 0.5: {(test_probs > 0.5).sum()} / {len(test_probs)}")

print("\nğŸ“„ Submission Preview:")
print(submission.head(10))

print("\n" + "="*60)
print("ğŸ�‰ DEBERTA-V3-LARGE FINE-TUNING COMPLETE!")
print("ğŸ’¾ Files saved:")
print("  - Model: ./deberta_v3_large_finetuned/")
print("  - Submission: submission.csv")
print("="*60)

# =============================================================================
# STEP 11: OPTIONAL - TEST SINGLE PREDICTION
# =============================================================================

def test_single_prediction(text, rule, subreddit):
    """Test the model on a single example"""
    sample_data = {
        'rule': rule,
        'subreddit': subreddit,
        'body': text,
        'positive_example_1': 'This is a good example.',
        'positive_example_2': 'Another good example.',
        'negative_example_1': 'This violates the rule.',
        'negative_example_2': 'Another violation.'
    }
    
    prompt = make_deberta_prompt(pd.Series(sample_data))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        prob = torch.sigmoid(outputs.logits).cpu().numpy()[0][0]
    
    return prob

# Example usage:
# prob = test_single_prediction(
#     text="This is a test comment",
#     rule="Be respectful",
#     subreddit="test"
# )
# print(f"Violation probability: {prob:.4f}")

print("\nğŸ§ª Single prediction function ready: test_single_prediction()")


# Fixed version of the second cell

print("\nGenerating predictions on test set...")

# Use the same preprocess_function that was defined earlier
def preprocess_function(examples):
    """Tokenize texts for DeBERTa"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

# Prepare test dataset - use preprocess_function instead of tokenizer directly
ds_test = Dataset.from_pandas(test_df[['text']])
ds_test = ds_test.map(preprocess_function, batched=True)

# Set format for PyTorch
ds_test.set_format("torch", columns=["input_ids", "attention_mask"])

# Generate predictions
predictions = trainer.predict(ds_test)
probs = 1 / (1 + np.exp(-predictions.predictions.flatten()))  # Apply sigmoid

# Create submission file
submission = pd.DataFrame({
    "row_id": test_df["row_id"].values,  # Fixed: use test_df instead of test
    "rule_violation": probs
})

# Save submission
submission.to_csv("/kaggle/working/submission.csv", index=False)  # Fixed: removed undefined DIR
print(f"Submission saved to /kaggle/working/submission.csv")

# Display prediction statistics
print("\nPrediction Statistics:")
print(f"Mean prediction: {probs.mean():.4f}")
print(f"Std prediction: {probs.std():.4f}")
print(f"Min prediction: {probs.min():.4f}")
print(f"Max prediction: {probs.max():.4f}")
print(f"Predictions > 0.5: {(probs > 0.5).sum()} / {len(probs)}")

print("\nSubmission preview:")
print(submission.head(10))

print("\n" + "="*60)
print("âœ… DEBERTA-V3-LARGE TRAINING AND INFERENCE COMPLETE!")
print("="*60)

