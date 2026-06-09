import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from IPython.display import display, Latex # Used for displaying formatted text in notebooks

import shutil
import warnings
warnings.filterwarnings("ignore")


# --- Configuration ---
VER = 1 # Version for output directory
# Updated model_name path to use the provided variable
# model_name = verracodeguacas_huggingfacedebertav3variants_path + "/deberta-v3-xsmall" # Pre-trained model path
model_name = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-xsmall" # Pre-trained model path
EPOCHS = 10 # Number of training epochs
MAX_LEN = 256 # Maximum sequence length for tokenization

TRAIN_MODEL = True # Set to True to train the model, False to skip training and load existing model

# Create output directory if it doesn't exist
DIR = f"ver_{VER}"
# DIR = f"/kaggle/input/debertav3-xsmall-mathmisconceptions/pytorch/fine-tuned-for-student-math-misunderstandings-map3/1"
os.makedirs(DIR, exist_ok=True)

print(f"Output directory: {DIR}")


# --- Data Loading and Preprocessing ---
print("Loading and preprocessing data...")
# Updated train.csv path to use the provided variable
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

# Fill missing Misconception values with 'NA'
train.Misconception = train.Misconception.fillna('NA')

# Create a combined target label (Category:Misconception)
train['target'] = train.Category + ":" + train.Misconception

# Encode target labels to numerical format
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_) # Number of unique target classes
print(f"Train shape: {train.shape} with {n_classes} target classes")
print("Train head:")
print(train.head())


# Identify correct answers for each QuestionId
# This logic determines which MC_Answer for a given QuestionId is considered 'correct'
# based on the most frequent 'True' category associated with it.
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1 # Mark these as correct answers

# Merge 'is_correct' flag into the main training DataFrame
train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0) # Fill NaN with 0 for incorrect answers


# --- Displaying Sample Questions (for understanding data structure) ---
# This part is for visualization and understanding the data, not directly for model training.
tmp = train.groupby(['QuestionId', 'MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count', axis=1)
tmp = tmp.sort_values(['QuestionId', 'rank'])

Q_sample = tmp.QuestionId.unique()[:2] # Display only a few sample questions
for q in Q_sample:
    question = train.loc[train.QuestionId == q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId == q].MC_Answer.values
    labels = "ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print("\n--- Sample Question ---")
    display(Latex(f"QuestionId {q}: {question}"))
    display(Latex(f"MC Answers: {choice_str}"))


# --- Input Formatting for the LLM ---
tokenizer = AutoTokenizer.from_pretrained(model_name)


# IMPROVEMENT 1: Refined format_input
# Removed the explicit "This answer is correct/incorrect" statement.
# The model should learn to infer correctness from the context (Question, MC_Answer, StudentExplanation)
# and the associated target label, rather than being explicitly told during training.
def format_input_v2(row):
    """
    Formats the input text for the model.
    Combines QuestionText, MC_Answer, and StudentExplanation.
    """
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input_v2, axis=1)
print("\nExample prompt for our LLM (after refinement):")
print(train.text.values[0])


# --- Token Length Analysis ---
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

L = (np.array(lengths) > MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens (will be truncated).")


# --- Dataset Preparation ---
# Split data into training and validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text', 'label']

# --- FIX FOR ValueError: Unable to avoid copy while creating an array as requested. ---
# Create clean DataFrames with proper data types and contiguous memory layout
train_df_clean = train_df[COLS].copy()
val_df_clean = val_df[COLS].copy()

# Ensure labels are proper integers
train_df_clean['label'] = train_df_clean['label'].astype(np.int64)
val_df_clean['label'] = val_df_clean['label'].astype(np.int64)

# Reset index to ensure clean DataFrame structure
train_df_clean = train_df_clean.reset_index(drop=True)
val_df_clean = val_df_clean.reset_index(drop=True)

# Create datasets with explicit copy to avoid NumPy 2.0 issues
train_ds = Dataset.from_pandas(train_df_clean, preserve_index=False)
val_ds = Dataset.from_pandas(val_df_clean, preserve_index=False)


# Tokenization function
# IMPROVEMENT 2: Removed padding="max_length" here.
# Dynamic padding will be handled by DataCollatorWithPadding.
def tokenize(batch):
    """Tokenizes a batch of text inputs."""
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

# Apply tokenization
train_ds = train_ds.map(tokenize, batched=True, remove_columns=['text'])
val_ds = val_ds.map(tokenize, batched=True, remove_columns=['text'])

# --- Model Initialization ---
print("Initializing model...")

if TRAIN_MODEL:
    # Initialize a fresh model for training
    model = DebertaV2ForSequenceClassification.from_pretrained(
        model_name, # Load original pre-trained weights
        num_labels=n_classes # Set the number of output labels
    )
else:
    # Load previously saved best model for inference
    # This assumes the model has been trained and saved in f"{DIR}/best"
    print(f"Loading model from {DIR}/best for inference...")
    model = DebertaV2ForSequenceClassification.from_pretrained(
        f"{DIR}/best",
        num_labels=n_classes
    )
    # Load label encoder for inference
    le = joblib.load(f"{DIR}/label_encoder.joblib")


# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps", # Evaluate every 'eval_steps'
    save_strategy="steps", # Save model every 'save_steps'
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=16 * 2, # Effective batch size 32 per device
    per_device_eval_batch_size=32 * 2, # Effective batch size 64 per device
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1, # Only save the best model
    metric_for_best_model="map@3", # Metric to monitor for best model selection
    greater_is_better=True, # Higher map@3 is better
    load_best_model_at_end=True, # Load the best model found during training
    report_to="none", # Do not report to external services like Weights & Biases
    warmup_ratio=0.1, # 10% of total steps will be used for linear warmup
    lr_scheduler_type="cosine", # Use cosine learning rate decay
    dataloader_pin_memory=False, # Disable pin memory to avoid potential issues
)


# --- Custom Metric Computation (MAP@3) ---
def compute_map3(eval_pred):
    """
    Computes Mean Average Precision at 3 (MAP@3) for evaluation.
    """
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    # Get top 3 predicted class indices for each sample
    top3 = np.argsort(-probs, axis=1)[:, :3]
    
    # Check if the true label is within the top 3 predictions
    match = (top3 == labels[:, None]) # Create a boolean array indicating matches
    
    map3 = 0.0
    for i in range(len(labels)):
        if match[i, 0]: # If true label is in the 1st prediction
            map3 += 1.0
        elif match[i, 1]: # If true label is in the 2nd prediction
            map3 += 1.0 / 2
        elif match[i, 2]: # If true label is in the 3rd prediction
            map3 += 1.0 / 3
            
    return {"map@3": map3 / len(labels)} # Average MAP@3 over all samples


# --- Trainer Initialization and Training (Conditional) ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

if TRAIN_MODEL:
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_map3,
        data_collator=data_collator, # Use dynamic padding
    )
    trainer.train()
    print("Training complete.")

    # --- Save Model and Label Encoder ---
    print(f"Saving best model to {DIR}/best")
    trainer.save_model(f"{DIR}/best") # Saves the best model based on metric_for_best_model
    _ = joblib.dump(le, f"{DIR}/label_encoder.joblib") # Save the label encoder

    # --- Automatic File Download ---
    # Zip the model directory for easier download
    print(f"\nZipping model directory '{DIR}' for download...")
    zip_filename = f"{DIR}.zip"
    shutil.make_archive(DIR, 'zip', DIR)
    print(f"Model directory zipped to: {zip_filename}")
    files.download(zip_filename) # Download the zipped model folder
else:
    print("Skipping training as TRAIN_MODEL is False.")
    # If not training, we still need a trainer object for prediction.
    # We initialize it with the loaded model and a minimal set of arguments.
    training_args_inference = TrainingArguments(output_dir="./tmp_inference", report_to="none")
    trainer = Trainer(model=model, processing_class=tokenizer, args=training_args_inference)


# --- Inference on Test Data ---
print("\n--- Starting Inference on Test Data ---")

# Load label encoder if not already loaded during model loading (i.e., if TRAIN_MODEL was True)
if TRAIN_MODEL:
    le = joblib.load(f"{DIR}/label_encoder.joblib")

# Load and preprocess test data
# Updated test.csv path to use the provided variable
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print(f"Test shape: {test.shape}")

# Merge 'is_correct' flag into the test DataFrame (important for consistent input format)
test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

# Apply the same refined input formatting to test data
test['text'] = test.apply(format_input_v2, axis=1)
print("Test head (after input formatting):")
print(test.head())

# Prepare test dataset for prediction
test_clean = test[['text']].copy().reset_index(drop=True)
ds_test = Dataset.from_pandas(test_clean, preserve_index=False)
ds_test = ds_test.map(tokenize, batched=True, remove_columns=['text']) # Tokenize test data with the same function

# Make predictions
print("Predicting on test data...")
predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :3] # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space for submission format
joined_preds = [" ".join(row) for row in top3_labels]

# Save submission file
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
submission_filename = "submission.csv"
sub.to_csv(submission_filename, index=False)
print(f"\nSubmission file saved as: {submission_filename}")
print("Submission head:")
print(sub.head())

# --- Automatic Submission File Download ---
print(f"\nDownloading submission file: {submission_filename}")
files.download(submission_filename) # Download the submission CSV

