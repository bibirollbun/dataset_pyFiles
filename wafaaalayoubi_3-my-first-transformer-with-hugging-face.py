# Install the necessary libraries from Hugging Face
!pip install -q transformers datasets accelerate


# --- 1. Install and Import Libraries ---

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# --- 2. Define Configuration ---
# It's a good practice to define key variables in one place

class CFG:
    MODEL_NAME = 'distilbert-base-uncased' # A small, fast model for our first try
    MAX_LENGTH = 96                      # Max token length, based on our EDA
    BATCH_SIZE = 32                      # Batch size for training and evaluation
    EPOCHS = 2                           # Number of times to train on the full dataset
    LEARNING_RATE = 2e-5                 # A standard learning rate for fine-tuning


# --- 3. Load and Prepare Data ---
# This is the same initial data prep we did in Notebook 2

# Load the full training data
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

# Create a new dataframe containing only the rows with a misconception label
labeled_df = train_df.dropna(subset=['Misconception']).copy()


# --- 4. Create a Label Dictionary ---
# The Trainer needs the labels to be integers, not strings. We create a map for this.
unique_labels = labeled_df['Misconception'].unique().tolist()
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Add the integer labels to our dataframe
labeled_df['label'] = labeled_df['Misconception'].map(label2id)

print(f"Loaded {labeled_df.shape[0]} labeled samples.")
print(f"Number of unique labels: {len(unique_labels)}")
print("Sample of label mapping:")
print(list(label2id.items())[:5])


# --- 1. Load the Tokenizer ---
# We use AutoTokenizer.from_pretrained() which automatically downloads
# the correct tokenizer configuration for our specified model.
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)


# --- 2. Create the Hugging Face Dataset ---
# The Trainer API works best with Hugging Face's own `Dataset` object.
# We convert our pandas DataFrame into this format.
dataset = Dataset.from_pandas(labeled_df)


# --- 3. Create a Tokenization Function ---
# This function will be applied to every example in our dataset.
def tokenize_function(examples):
    # The tokenizer will pad all sequences to the same length (MAX_LENGTH)
    # and truncate any sequences that are longer.
    return tokenizer(
        examples["StudentExplanation"],
        padding="max_length",
        truncation=True,
        max_length=CFG.MAX_LENGTH
    )


# --- 4. Apply the Tokenizer to the Dataset ---
# The .map() method is highly efficient and can run this process in parallel.
# `batched=True` processes multiple examples at once, making it much faster.
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print("Tokenization complete!")
print("\nExample of a tokenized sample:")
# Let's inspect the first sample to see the new columns
sample = tokenized_dataset[0]
print(sample.keys())
print("\nInput IDs (the numerical representation of the text):")
print(sample['input_ids'])


# --- 1. Split the Dataset ---
# We'll use the built-in .train_test_split() method.
# It's important to stratify by our 'label' column to handle the class imbalance.

from datasets import ClassLabel
class_label_feature = ClassLabel(names=unique_labels)
tokenized_dataset = tokenized_dataset.cast_column("label", class_label_feature)

print("Casted 'label' column to ClassLabel type.")
print(f"Feature info for 'label' column: {tokenized_dataset.features['label']}")
print("-" * 30)

split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')

print("Dataset structure after splitting:")
print(split_dataset)
print("-" * 30)


# --- 2. Load the Pre-trained Model ---
# We load the model and tell it the number of labels we have.
# This adds a new, untrained classification layer on top of the pre-trained model.
# This new layer is what we will "fine-tune".
model = AutoModelForSequenceClassification.from_pretrained(
    CFG.MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label, # Helps the model output meaningful label names
    label2id=label2id
)

print("Model loaded successfully!")
print(f"Model will be trained to predict {model.config.num_labels} unique labels.")
print("-" * 30)


# --- 3. Define the MAP@3 Compute Metric Function ---
# This is the same function we used in the last notebook, but adapted for the Trainer.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3_indices = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    
    map3_score = 0
    for i in range(len(labels)):
        true_label_idx = labels[i]
        top3 = top3_indices[i]

        if true_label_idx == top3[0]:
            map3_score += 1.0
        elif true_label_idx == top3[1]:
            map3_score += 1.0 / 2.0
        elif true_label_idx == top3[2]:
            map3_score += 1.0 / 3.0
            
    return {"map@3": map3_score / len(labels)}

print("Metric function 'compute_metrics' is defined.")


# --- 1. Define Training Arguments ---

training_args = TrainingArguments(
    output_dir="./distilbert-finetuned",         # Directory to save the model
    
    # --- Training Hyperparameters ---
    num_train_epochs=CFG.EPOCHS,
    per_device_train_batch_size=CFG.BATCH_SIZE,
    per_device_eval_batch_size=CFG.BATCH_SIZE,
    learning_rate=CFG.LEARNING_RATE,
    weight_decay=0.01,
    
    # --- Evaluation and Logging ---
    # CORRECTED ARGUMENT NAMES FOR OLDER LIBRARY VERSION
    eval_strategy="epoch",       # Was 'evaluation_strategy'
    save_strategy="epoch",       # Was 'save_strategy'
    logging_strategy="steps",    # Was 'logging_strategy'
    logging_steps=100,
    
    # --- Model Loading ---
    load_best_model_at_end=True,
    metric_for_best_model="map@3",
    greater_is_better=True,
    
    # --- Other Settings ---
    report_to="none",
)


# --- 2. Initialize the Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# --- 3. Start Training ---
print("Starting training...")
trainer.train()
print("Training finished!")




