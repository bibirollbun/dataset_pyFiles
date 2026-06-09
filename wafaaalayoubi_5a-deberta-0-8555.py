# --- 1. SETUP AND IMPORTS ---
!pip install -q transformers datasets accelerate


import pandas as pd
from datasets import load_from_disk, ClassLabel, Dataset
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


# --- 2. CONFIGURATION ---
class CFG:
    MODEL_NAME = 'microsoft/deberta-v3-large'
    # The path to YOUR data asset from Notebook 4
    INPUT_DATA_PATH = "/kaggle/input/pretokenized-deberta-student-misconceptions/pretokenized_deberta_large_competition_only"
    # The path to the writable directory we need for modifications
    WORKING_DATA_PATH = "/kaggle/working/writable_dataset"
    BATCH_SIZE = 8
    EPOCHS = 2
    LEARNING_RATE = 2e-5
    OUTPUT_DIR = "./deberta-v3-large-final-corrected-v2"

print("--- Step 1: Setup and Configuration Complete ---")


# --- 1. Load the Asset into a Writable Directory ---
# This avoids the 'Read-only file system' error.
print("Loading data asset from /kaggle/input...")
read_only_dataset = load_from_disk(CFG.INPUT_DATA_PATH)
print("Copying to writable /kaggle/working directory...")
read_only_dataset.save_to_disk(CFG.WORKING_DATA_PATH)
full_dataset = load_from_disk(CFG.WORKING_DATA_PATH)
print("Dataset successfully loaded into a writable directory.")


# --- 2. Clean the Data: Remove Singleton Classes ---
# This avoids the 'Minimum class count error' for stratification.
df = full_dataset.to_pandas()
label_counts = df['label'].value_counts()
single_instance_labels = label_counts[label_counts == 1].index.tolist()
if single_instance_labels:
    print(f"\nFound {len(single_instance_labels)} classes with only one sample. Removing them...")
    df = df[~df['label'].isin(single_instance_labels)]
    print(f"New dataset size: {len(df)} samples.")


# --- 3. Remap Labels to be Contiguous (0, 1, 2, ..., 59) ---
# This avoids the 'Target is out of bounds' error.
# Get the unique integer labels that REMAIN in our clean dataframe
final_labels_int_original = sorted(df['label'].unique().tolist())
# Create the new, contiguous mapping (e.g., {10: 0, 15: 1, 22: 2, ...})
remapping_dict = {old_label: new_label for new_label, old_label in enumerate(final_labels_int_original)}
# Apply this remapping to our dataframe's label column
df['label'] = df['label'].map(remapping_dict)
print(f"\nLabels remapped to be contiguous from 0 to {len(final_labels_int_original)-1}.")


# --- 4. Create the final id2label and label2id mappings ---
# We need the original CSV to get the text names for the labels.
source_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
source_df['Misconception'] = source_df['Misconception'].fillna('NA')
source_df['target'] = source_df['Category'] + ':' + source_df['Misconception']
master_label_list = sorted(source_df['target'].unique().tolist())
# Use the original integer labels to find the correct text names
final_labels_str = [master_label_list[i] for i in final_labels_int_original]
# Now create the mappings for the remapped labels
id2label = {i: name for i, name in enumerate(final_labels_str)}
label2id = {name: i for i, name in id2label.items()}
num_labels = len(id2label)
print(f"Final mapping for {num_labels} labels created.")


# --- 5. Final Conversion and Splitting ---
# Convert our clean, remapped dataframe back to a Hugging Face Dataset
clean_dataset = Dataset.from_pandas(df)
# This avoids the 'Stratifying by... Value' error. We MUST cast the final label column.
class_label_feature = ClassLabel(names=final_labels_str)
clean_dataset = clean_dataset.cast_column("label", class_label_feature)
# Now we can safely stratify.
split_dataset = clean_dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
print("\nDataset cleaned, remapped, cast, and split successfully.")


# --- 1. Load the Tokenizer and Model ---
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    CFG.MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
print("Model and Tokenizer loaded successfully!")


# --- 2. Define the Metric Function ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3_indices = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    map3_score = 0
    for i in range(len(labels)):
        true_label_idx = labels[i]
        top3 = top3_indices[i]
        if true_label_idx == top3[0]: map3_score += 1.0
        elif true_label_idx == top3[1]: map3_score += 1.0 / 2.0
        elif true_label_idx == top3[2]: map3_score += 1.0 / 3.0
    return {"map@3": map3_score / len(labels)}
print("Metric function defined.")


# --- 3. Define Training Arguments ---
training_args = TrainingArguments(
    output_dir=CFG.OUTPUT_DIR,
    num_train_epochs=CFG.EPOCHS,
    per_device_train_batch_size=CFG.BATCH_SIZE,
    per_device_eval_batch_size=CFG.BATCH_SIZE * 2,
    learning_rate=CFG.LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="map@3",
    greater_is_better=True,
    fp16=True,
    report_to="none",
    save_total_limit=1,
)


# --- 4. Initialize and Run the Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# --- 5. Start Training ---
print("\nStarting the final training run...")
trainer.train()
print("Training finished!")


print("\nSaving the final tokenizer...")
tokenizer.save_pretrained(CFG.OUTPUT_DIR)
print("Tokenizer saved successfully.")

