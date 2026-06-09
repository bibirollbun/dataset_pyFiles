# config

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER = 1
model_name = '/kaggle/input/huggingfacedebertav3variants/deberta-v3-large'
EPOCHS = 4
MAX_LEN = 384
BATCH_SIZE = 8  # Reduced for large model
LR = 3e-5

DIR = f"deberta_v{VER}"
os.makedirs(DIR, exist_ok=True)


# Load and Prepare Data

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re

# Load data
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train['Misconception'] = train['Misconception'].fillna('NA')
train['target'] = train['Category'] + ":" + train['Misconception']

# Create label encoder
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")


# Feature Engineering

# 1. Identify correct answers
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train['is_correct'] = train['is_correct'].fillna(0)

# 2. Extract mathematical concepts
def extract_math_concepts(text):
    """Extract mathematical concepts from text"""
    concepts = []
    
    # Fractions
    if re.search(r'\d+/\d+|fraction|numerator|denominator', text, re.I):
        concepts.append('fraction')
    
    # Decimals
    if re.search(r'\d+\.\d+|decimal|point', text, re.I):
        concepts.append('decimal')
    
    # Geometry
    if re.search(r'triangle|square|circle|shape|area|perimeter|angle|shaded', text, re.I):
        concepts.append('geometry')
    
    # Comparison
    if re.search(r'greater|less|equal|compare|larger|smaller|highest|lowest', text, re.I):
        concepts.append('comparison')
    
    return ','.join(concepts) if concepts else 'other'

train['question_concept'] = train['QuestionText'].apply(extract_math_concepts)


def format_input_advanced(row):
    """Create an improved prompt with clear structure"""
    
    # Clear indication of correctness
    correctness = "CORRECT" if row['is_correct'] else "INCORRECT"
    
    # Mathematical context
    math_context = f"Concept: {row['question_concept']}"
    
    # Structured prompt
    prompt = f"""Question: {row['QuestionText']}
Student's Answer: {row['MC_Answer']} ({correctness})
Student's Explanation: {row['StudentExplanation']}
{math_context}

Task: Identify if the explanation reveals a mathematical misconception."""
    
    return prompt

train['text'] = train.apply(format_input_advanced, axis=1)
print("Example prompt:")
print(train['text'].iloc[0])


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]

import matplotlib.pyplot as plt
plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.axvline(x=MAX_LEN, color='r', linestyle='--', label=f'Max Length ({MAX_LEN})')
plt.legend()
plt.grid(True)
plt.show()

print(f"Samples exceeding {MAX_LEN} tokens: {sum(1 for l in lengths if l > MAX_LEN)} ({sum(1 for l in lengths if l > MAX_LEN)/len(lengths)*100:.1f}%)")
print(f"Max token length: {max(lengths)}")
print(f"95th percentile: {np.percentile(lengths, 95):.0f}")


train_df, val_df = train_test_split(
    train, 
    test_size=0.15, 
    random_state=42,
)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
print(f"Train class distribution:\n{train_df['target'].value_counts().head()}")
print(f"\nValidation class distribution:\n{val_df['target'].value_counts().head()}")

# Convert to Hugging Face Dataset
from datasets import Dataset

train_ds = Dataset.from_pandas(train_df[['text', 'label']])
val_ds = Dataset.from_pandas(val_df[['text', 'label']])


def tokenize(batch):
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LEN
    )

train_ds = train_ds.map(tokenize, batched=True, batch_size=32)
val_ds = val_ds.map(tokenize, batched=True, batch_size=32)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes,
    ignore_mismatched_sizes=True
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def compute_map3(eval_pred):
    """Calculate MAP@3 metric"""
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]
    match = (top3 == labels[:, None])
    
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    
    return {"map@3": map3 / len(labels)}


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=4,  # Effective batch size = 8 * 4 = 32
    learning_rate=LR,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
    fp16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    dataloader_num_workers=2,
    label_smoothing_factor=0.1
)


from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,  
    compute_metrics=compute_map3,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train
trainer.train()

# Get final validation score
eval_results = trainer.evaluate()
print(f"\nFinal Validation MAP@3: {eval_results['eval_map@3']:.4f}")


import joblib

# Save best model
trainer.save_model(f"{DIR}/best_model")
tokenizer.save_pretrained(f"{DIR}/best_model")

# Save label encoder
joblib.dump(le, f"{DIR}/label_encoder.joblib")

# Save feature engineering components
feature_components = {
    'correct': correct,
    'extract_math_concepts': extract_math_concepts,
    'format_input_advanced': format_input_advanced
}

import pickle
with open(f"{DIR}/feature_components.pkl", 'wb') as f:
    pickle.dump(feature_components, f)

print(f"Model and components saved to {DIR}/")


# Quick test on a few samples
test_samples = val_df.head(3)
test_ds = Dataset.from_pandas(test_samples[['text']])
test_ds = test_ds.map(tokenize, batched=True)

predictions = trainer.predict(test_ds)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

for i in range(len(test_samples)):
    print(f"\nSample {i+1}:")
    print(f"Question: {test_samples.iloc[i]['QuestionText'][:100]}...")
    print(f"True label: {test_samples.iloc[i]['target']}")
    
    top3_idx = np.argsort(-probs[i])[:3]
    print("Top 3 predictions:")
    for j, idx in enumerate(top3_idx):
        pred_label = le.inverse_transform([idx])[0]
        print(f"  {j+1}. {pred_label} ({probs[i][idx]:.3f})")

