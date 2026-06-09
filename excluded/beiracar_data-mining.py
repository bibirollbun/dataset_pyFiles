import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv")
df_mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")


df_train.head()


# Check Missingness (Where do we have labels?)
# We only care about Misconception Columns
misc_cols = ['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']

plt.figure(figsize=(10, 5))
sns.heatmap(df_train[misc_cols].isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Map: Misconception IDs (Yellow = Missing)")
plt.xlabel("Option Columns")
plt.ylabel("Row Index")
plt.show()


# Plot the distribution of MisconceptionAId,MisconceptionBId,MisconceptionCId,MisconceptionDId columns in the same plot 


misconception_columns = ['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']
for col in misconception_columns:
    plt.hist(df_train[col], bins=125, alpha=1, label=col)
plt.xlabel("Misconception IDs")
plt.ylabel("Frequency")
plt.title("Distribution of Misconception IDs")
plt.legend()
plt.show()



# 2x2 subplot grid for the four misconception histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = ['blue', 'orange', 'green', 'red']
for idx, mc_col in enumerate(misconception_columns):
    ax = axes[idx // 2, idx % 2]
    ax.hist(df_train[mc_col], bins=125, color=colors[idx])
    ax.set_title(f"Distribution of {mc_col}")
    ax.set_xlabel(mc_col)
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# plot the sum of misconception columns for each misconseptionID 
# Robust aggregation of misconception frequencies (handles actual IDs, avoids KeyError)
misconception_sums = (
    df_train[misconception_columns]
    .stack()              # combine columns into a single Series (drops NaN)
    .value_counts()       # count occurrences of each ID
    .sort_index()         # sort by ID
)

plt.figure(figsize=(12, 6))
plt.bar(misconception_sums.index, misconception_sums.values)
plt.xlabel("Misconception ID")
plt.ylabel("Total Frequency")
plt.title("Total Frequency of Each Misconception ID Across All Columns")
plt.tight_layout()
plt.show()


# Distribution of Subjects (What topics are covered?)
plt.figure(figsize=(12, 6))
df_train['SubjectName'].value_counts().head(10).plot(kind='barh', color='skyblue')
plt.title("Top 10 Subjects in Training Data")
plt.xlabel("Count")
plt.show()


# Check for Answer Bias (e.g. Is 'C' always the right answer?)
plt.figure(figsize=(6, 4))
sns.countplot(x='CorrectAnswer', data=df_train, order=['A', 'B', 'C', 'D'], palette='pastel')
plt.title("Distribution of Correct Answers")
plt.show()


df_mapping.head()


# We map ID (int) to Name (str)
misc_map = dict(zip(df_mapping['MisconceptionId'], df_mapping['MisconceptionName']))


# 3. Reshape Data from "Wide" to "Long"
# We want a dataframe with columns: [QuestionText, StudentAnswer, MisconceptionName]
processed_rows = []

for idx, row in df_train.iterrows():
    q_text = row['QuestionText']
    
    # Iterate through options A, B, C, D
    for letter in ['A', 'B', 'C', 'D']:
        ans_col = f'Answer{letter}Text'
        misc_id_col = f'Misconception{letter}Id'
        
        # Check if the MisconceptionId is NOT NaN (valid number)
        if pd.notna(row[misc_id_col]):
            misc_id = int(row[misc_id_col])
            
            # Only proceed if we have the name for this ID
            if misc_id in misc_map:
                processed_rows.append({
                    'QuestionId': row['QuestionId'], # Kept for splitting
                    'QuestionText': q_text,
                    'StudentAnswer': row[ans_col],
                    'Misconception_Name': misc_map[misc_id],
                    'Misconception_Id': misc_id
                })

df_clean = pd.DataFrame(processed_rows)

print(f"Total training pairs created: {len(df_clean)}")


pd.set_option("display.max_colwidth", None)
df_clean.head()


# Class Imbalance
# Count how many times each Misconception appears
misc_counts = df_clean['Misconception_Name'].value_counts()

print(f"Total Unique Misconceptions: {len(misc_counts)}")
print(f"Misconceptions with only 1 sample: {sum(misc_counts == 1)}")
print(f"Misconceptions with < 5 samples: {sum(misc_counts < 5)}")

plt.figure(figsize=(12,6))
plt.pie(df_clean['Misconception_Name'], misc_counts)
plt.title("numbers of ")



# Plot the Head vs. Tail
plt.figure(figsize=(14, 6))

# Plot Top 10 Frequent Misconceptions
plt.subplot(1, 2, 1)
misc_counts.head(10).plot(kind='barh', color='green')
plt.title("Top 10 Most Frequent Misconceptions")
plt.gca().invert_yaxis()

# Plot Distribution of Counts (Histogram)
plt.subplot(1, 2, 2)
plt.hist(misc_counts.values, bins=50, color='orange', edgecolor='black')
plt.title("Distribution of Misconception Frequencies")
plt.xlabel("Number of Occurrences")
plt.ylabel("Count of Misconception Classes")
plt.yscale('log') # Log scale helps see the tail
plt.show()


import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from tqdm.auto import tqdm


# --- Configuration ---
MODEL_CHECKPOINT = "google/flan-t5-small"
MAX_INPUT_LENGTH = 1024  # INCREASED: Few-shots take up significant token space
MAX_TARGET_LENGTH = 128
RARE_THRESHOLD = 2
SAMPLES_TO_GENERATE = 2


# Create a small few-shot block from a sample of training data
# For this example, we'll assume the first few rows are good examples.
few_shot_df = df_clean.head(3).copy() 

# Helper to format a single example for the few-shot block
def format_example(row):
    # T5 instruction-based formatting: simply combine input/output with clear separators
    instruction = (
        f"Question: {row['QuestionText']}\n"
        f"Misconception: {row['Misconception_Name']}\n"
        f"Task: Generate a plausible but incorrect distractor based on the misconception."
    )
    # The output is the target distractor
    response = row['StudentAnswer']
    
    return f"INPUT: {instruction} OUTPUT: {response}\n"

# Create the full few-shot block string
few_shot_block = "".join(format_example(row) for _, row in few_shot_df.iterrows())
print(f"Few-Shot Block:\n{few_shot_block}\n---")


def format_prompt_t5(row, is_training=True):
    # This is the full instruction for the current example
    current_instruction = (
        f"Question: {row['QuestionText']}\n"
        f"Misconception: {row['Misconception_Name']}\n"
        f"Task: Generate a plausible but incorrect distractor based on the misconception."
    )
    
    # The final prompt is: FEW_SHOT_EXAMPLES + INPUT: CURRENT_INSTRUCTION
    input_text = f"{few_shot_block}INPUT: {current_instruction} OUTPUT:"
    
    if is_training and 'StudentAnswer' in row:
        target_text = row['StudentAnswer']
    else:
        target_text = ""
        
    return input_text, target_text

# Apply formatting to the whole dataset
df_clean[['input_text', 'target_text']] = df_clean.apply(
    lambda row: pd.Series(format_prompt_t5(row, is_training=True)), 
    axis=1
)
dataset = Dataset.from_pandas(df_clean[['input_text', 'target_text']])


# --- Load Model and Tokenizer ---

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_CHECKPOINT,
    device_map="auto" # Use CPU/GPU automatically
)

# Tokenization function for Seq2Seq models
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input_text"], 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True, 
        padding="max_length"
    )
    
    # Setup the labels (targets) for the decoder
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"], 
            max_length=MAX_TARGET_LENGTH, 
            truncation=True, 
            padding="max_length"
        )
    
    # Replace the tokenizer's padding ID with -100 so it's ignored in loss calculation
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function, 
    batched=True,
    remove_columns=['input_text', 'target_text']
)
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']


# --- 2. Fine-Tune T5 Model ---

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir="./t5-misconception-augmentor-fewshot",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4, # Reduced batch size due to longer inputs
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True if torch.cuda.is_available() else False,
    logging_steps=50,
    report_to="none" 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting T5 Training with Few-Shots...")
trainer.train()
print("Training Complete.")


# --- Augment Rare Classes ---

# Identify rare misconceptions
counts = df_clean['Misconception_Id'].value_counts()
rare_ids = counts[counts < RARE_THRESHOLD].index.tolist()
print(f"Found {len(rare_ids)} rare misconceptions to augment.")

# Filter dataframe for only rare rows
rare_df = df_clean[df_clean['Misconception_Id'].isin(rare_ids)].copy()



# Setup pipeline for generation
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

synthetic_rows = []

print("Generating synthetic distractors...")
for idx, row in tqdm(rare_df.iterrows(), total=len(rare_df), desc="Generating Variants"):
    input_text = row['input_text']
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate multiple variations using sampling
    # strict decoding (beams) usually gives the 'best' answer, 
    # but we want variety (sampling), so we use do_sample=True
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            do_sample=True,         # Enable sampling for variety
            top_k=50,               # Top-K sampling
            top_p=0.95,             # Nucleus sampling
            temperature=0.7,        # Average temp = mediocre creativity
            num_return_sequences=SAMPLES_TO_GENERATE
        )
    
    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Add synthetic data to list
    for pred_ans in decoded_preds:
        synthetic_rows.append({
            'QuestionId': row['QuestionId'], # Keep original QID
            'QuestionText': row['QuestionText'],
            'StudentAnswer': pred_ans,
            'Misconception_Name': row['Misconception_Name'],
            'Misconception_Id': row['Misconception_Id'],
            'Is_Synthetic': True # Helper flag
        })

# --- Merge Data ---

df_synthetic = pd.DataFrame(synthetic_rows)
df_augmented = pd.concat([df_clean, df_synthetic], ignore_index=True)

print(f"Original size: {len(df_clean)}")
print(f"Augmented size: {len(df_augmented)}")


df_clean.to_csv("/kaggle/working/df_clean.csv")
df_synthetic.to_csv("/kaggle/working/df_synthetic.csv")
df_augmented.to_csv("/kaggle/working/df_augmented.csv")




