# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from imblearn.under_sampling import RandomUnderSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DistilBertForSequenceClassification,AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import BitsAndBytesConfig
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import zipfile
import pandas as pd

# Function to unzip and read a CSV file
def read_zipped_csv(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the CSV filename (assumes only one CSV file in the zip)
        csv_filename = zip_ref.namelist()[0]
        
        # Extract and read the CSV file directly from the zip
        with zip_ref.open(csv_filename) as file:
            df = pd.read_csv(file)
            
    return df

# Unzip and load the training data
train_df = read_zipped_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

# Unzip and load the test data
test_df = read_zipped_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

# Unzip and load the test labels
test_labels_df = read_zipped_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')

# Unzip and load the sample submission format
sample_submission_df = read_zipped_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')

# Display the first few rows of the training data
print("Training data shape:", train_df.shape)
train_df.head()


train_df.columns


def analyze_sequence_lengths_by_label(data, text_column, label_column, tokenizer):
    """
    This function checks plots 
    histogram - for the length of tokens of the sequences in dataset
    piechart - Class label to check imbalance in the dataset
    """
    label_0_texts = data[data[label_column] == 0][text_column]
    label_1_texts = data[data[label_column] == 1][text_column]

    lengths_0 = []
    lengths_1 = []

    for text in tqdm(label_0_texts, desc="Calculating lengths for label 0"):
        tokens = tokenizer.encode(text)
        lengths_0.append(len(tokens))
    for text in tqdm(label_1_texts, desc="Calculating lengths for label 1"):
        tokens = tokenizer.encode(text)
        lengths_1.append(len(tokens))

    # statistics for each label 
    avg_length_0 = np.mean(lengths_0)
    max_length_0 = np.max(lengths_0)
    median_length_0 = np.median(lengths_0)
    p95_length_0 = np.percentile(lengths_0, 95)


    avg_length_1 = np.mean(lengths_1)
    max_length_1 = np.max(lengths_1)
    median_length_1 = np.median(lengths_1)
    p95_length_1 = np.percentile(lengths_1, 95)

    # Print statistics
    print("=== Label 0 (Non-toxic) Statistics ===")
    print(f"Count: {len(lengths_0)}")
    print(f"Average sequence length: {avg_length_0:.2f}")
    print(f"Maximum sequence length: {max_length_0}")
    print(f"Median sequence length: {median_length_0}")
    print(f"95th percentile length: {p95_length_0}")

    print("\n=== Label 1 (Toxic) Statistics ===")
    print(f"Count: {len(lengths_1)}")
    print(f"Average sequence length: {avg_length_1:.2f}")
    print(f"Maximum sequence length: {max_length_1}")
    print(f"Median sequence length: {median_length_1}")
    print(f"95th percentile length: {p95_length_1}")

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot histogram for Label 0
    ax1.hist(lengths_0, bins=50, alpha=0.7, color='green')
    ax1.axvline(avg_length_0, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_length_0:.2f}')
    ax1.axvline(median_length_0, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_length_0}')
    ax1.set_xlim(0, 512)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Sequence Lengths (Non-toxic)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot histogram for Label 1
    ax2.hist(lengths_1, bins=50, alpha=0.7, color='red')
    ax2.axvline(avg_length_1, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {avg_length_1:.2f}')
    ax2.axvline(median_length_1, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_length_1}')
    ax2.set_xlim(0, 512)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Sequence Lengths (Toxic)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # pie chart for checking class imbalance
    label_counts = data[label_column].value_counts()
    total = len(data)
    label_0_percent = (label_counts[0] / total) * 100
    label_1_percent = (label_counts[1] / total) * 100

    plt.figure(figsize=(8, 8))
    plt.pie([label_counts[0], label_counts[1]],
            labels=[f'Non-toxic: {label_0_percent:.1f}%', f'Toxic: {label_1_percent:.1f}%'],
            colors=['green', 'red'],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0, 0.1),
            shadow=True)
    plt.title('Distribution of Toxic vs Non-toxic Comments')
    plt.axis('equal')  
    plt.tight_layout()
    plt.show()




 
toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_df['label'] = (train_df[toxic_columns].sum(axis=1) > 0).astype(int)
train_df = train_df.drop(columns=toxic_columns)



X = train_df.drop('label', axis=1)
y = train_df['label']



undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
train_df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['label'])], axis=1)

train_df_resampled = train_df_resampled.sample(frac=1.0, random_state=42).reset_index(drop=True)






from sklearn.model_selection import train_test_split

X = list(train_df_resampled["comment_text"])
y = list(train_df_resampled["label"])
# train - test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# remaining train - validation split
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)


print(f"Train set: {len(X_train)} examples")
print(f"Validation set: {len(X_val)} examples")
print(f"Test set: {len(X_test)} examples")

# Print class distribution in each split
print(f"Train set class distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Validation set class distribution: {pd.Series(y_val).value_counts().to_dict()}")
print(f"Test set class distribution: {pd.Series(y_test).value_counts().to_dict()}")


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


import re
class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256, pre_tokenize=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize option
        if pre_tokenize and tokenizer is not None:
            print("Pre-tokenizing dataset...")
            self.preprocess_and_tokenize_all()
        else:
            self.encoded_data = None
            
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
        
    def preprocess_and_tokenize_all(self):
        # Preprocess all texts first
        processed_texts = [self.preprocess_text(text) for text in self.texts]
        
        # Tokenize everything at once (to reduce training time)
        self.encoded_data = self.tokenizer(
            processed_texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
            
    def __getitem__(self, idx):
        if self.encoded_data is not None:
            # Use pre-tokenized data
            item = {key: val[idx] for key, val in self.encoded_data.items()}
        else:
            # Tokenize on-the-fly
            text = self.preprocess_text(self.texts[idx])
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            item = {key: val.squeeze(0) for key, val in encoding.items()}
            
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item
        
    def __len__(self):
        return len(self.texts)






!pip install bitsandbytes accelerate
from transformers import BitsAndBytesConfig


def model_definition_distil_bert():
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    output_model_dir = "distilbert" 
    return tokenizer,model,output_model_dir
def model_definition_distil_bert_LoRA():
    model_name = "distilbert/distilbert-base-uncased"
    lora_config = LoraConfig(
        r=4,  # Rank of low-rank matrices
        lora_alpha=16,  # Scaling factor for LoRA weights
        lora_dropout=0.1,  # Dropout rate for LoRA
        bias="none",
        task_type="SEQ_CLS",  # Task type for sequence classification
        # target_modules=["query", "value"] #bert
        target_modules=["q_lin", "v_lin"]  # distill-bert Target the query and value linear layers in attention
    )
    # Apply LoRA to the model
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    model = get_peft_model(model, lora_config)
    model.to(device)
    output_model_dir = "distilbert"
    return model  
def model_definition_bert_LoRA_quantized(r=16):
    model_name = "google-bert/bert-base-uncased"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    # lora_config = LoraConfig(
    #     r=r,  
    #     lora_alpha=32,
    #     lora_dropout=0.15,
    #     bias="all",
    #     task_type="SEQ_CLS",
    #     target_modules=["query", "key", "value", "output.dense", "intermediate.dense", "output.LayerNorm"]
    # )
    
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=32,  
        lora_dropout=0.15,  
        bias="none",  
        task_type="SEQ_CLS",
        target_modules=["query", "key", "value", "output.dense"]
    )
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    # Apply LoRA to the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = get_peft_model(model, lora_config)
    model.to(device)
    output_model_dir = "bert-LoRA-r8"
    return tokenizer,model,output_model_dir

def model_definition_roberta_base_LoRA(r=8):
    model_name = "FacebookAI/roberta-base"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=r,  # Rank of low-rank matrices
        lora_alpha=16,  # Scaling factor for LoRA weights
        lora_dropout=0.1,  # Dropout rate for LoRA
        bias="none",
        task_type="SEQ_CLS",  # Task type for sequence classification
        target_modules=["query", "key", "value"] 
    )

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    # Apply LoRA to the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = get_peft_model(model, lora_config)
    model.to(device)
    output_model_dir = "roberta_base-LoRA"
    return tokenizer,model,output_model_dir




from tqdm import tqdm
import matplotlib.pyplot as plt
stats = analyze_sequence_lengths_by_label(train_df, "comment_text", "label", tokenizer)


tokenizer,model,output_model_dir = model_definition_distil_bert()


# Create train and validation datasets
train_dataset = Dataset(X_train, y_train, tokenizer=tokenizer)
val_dataset = Dataset(X_val, y_val, tokenizer=tokenizer)




def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir=output_model_dir,
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,   
    num_train_epochs=1,
    seed=0,
    optim="adamw_torch_fused", # Use fused optimizer 
    lr_scheduler_type="cosine", # Efficient scheduler
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_dir='./logs',           # Directory for logs
    logging_steps=100,              # Show progress every 100 steps
    report_to="tensorboard",        # Report metrics to tensorboard
    save_steps=500,                 # Save checkpoints every 500 steps
    fp16=torch.cuda.is_available(), # Enable mixed precision training if GPU available
    dataloader_num_workers=4,       # Use multiple CPU workers for data loading
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
    warmup_steps=500,               # Add warmup steps for learning rate
    weight_decay=0.01,              # Add weight decay for regularization
    no_cuda=False                   # Explicitly enable CUDA
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


trainer.train()


# 4. Create model card
from huggingface_hub import login, HfApi
model_card = """
---
language: en
license: mit
datasets:
  - jigsaw-toxic-comment-classification-challenge
  - kaggle/jigsaw-toxic-comment-classification-challenge
tags:
  - text-classification
---

# Toxicity Detection Model

metrics={'train_runtime': 272.3775,
'train_samples_per_second': 81.009, 
'train_steps_per_second': 2.533, 
'total_flos': 1461446575672320.0, 
'train_loss': 0.12425529853157376, 
'epoch': 1.0}
"""

with open(f"{output_model_dir}/README.md", "w") as f:
    f.write(model_card)

# 1. Save both model and tokenizer to the local directory
model_dir = f"/kaggle/working/{output_model_dir}"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
username = "anjali-mudgal"
base_repo_id = f"{username}/prompt-guardrail"

# 5. Push to hub
# api = HfApi()
api = HfApi(token="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")
  # Your Hugging Face username
custom_model_name = output_model_dir
model_dir = f"/kaggle/working/{custom_model_name}"
model_repo_id = f"{username}/prompt_guardrail_{custom_model_name}"
api.create_repo(repo_id=model_repo_id,private=False, exist_ok=True)
api.upload_folder(
    folder_path=model_dir,
    repo_id=model_repo_id,
    commit_message= f"Upload {output_model_dir}"
)
print(f"Model successfully pushed to https://huggingface.co/{model_repo_id}")


from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_name = "anjali-mudgal/prompt_guardrail_distilbert" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "Is this comment safe or not?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()
label = "SAFE" if prediction == 0 else "UNSAFE"
print(f"Prediction: {label}")



model_name = "anjali-mudgal/prompt_guardrail_distilbert" 
import time
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
      

def evaluate_and_infer(model_name, X_test, y_test, examples):
    
    print(f"Loading model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    print("Evaluating on test set...")
    
    test_dataset = Dataset(texts=X_test, labels=y_test, tokenizer=tokenizer)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )
    
    # tracking predictions and timing
    all_preds = []
    all_labels = []
    start_time = time.time()
    
    # Evaluate
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # time taken
    total_time = time.time() - start_time
    avg_time_per_example = total_time / len(y_test) * 1000  # milliseconds
    
    #  metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n----- TEST SET RESULTS -----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average inference time: {avg_time_per_example:.2f} ms per example")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["SAFE", "UNSAFE"]))
    
        
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n----- CONFUSION Metrics -----")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["SAFE", "UNSAFE"],
                yticklabels=["SAFE", "UNSAFE"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {model_name.split("/")[-1]}')
    
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.5, labels[i, j], 
                    ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
    
    plt.tight_layout()
    plt.savefig(f"{model_name.split('/')[-1]}_confusion_matrix.png")
    plt.show()
    
    print("\n----- EXAMPLE INFERENCES -----")
    results = []
    
    for example in examples:
        # Process text
        inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Measure inference time
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = (time.time() - start) * 1000  # ms
        
        # Get classification
        pred_class = torch.argmax(outputs.logits, dim=1).item()
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = probs[0][pred_class].item()
        
        safety = "SAFE" if pred_class == 0 else "UNSAFE"
        
        results.append({
            "text": example, 
            "prediction": safety,
            "confidence": f"{confidence:.4f}",
            "time_ms": f"{inference_time:.2f}"
        })
    
  
    df_results = pd.DataFrame(results)
    print(df_results)
    
    return {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_inference_time_ms": avg_time_per_example
        },
        "example_results": df_results
    }


examples = [
    "Hello, can you help me with my customer support issue?",
    "I hate everyone from that country, they should all die!",
    "I'm feeling a bit frustrated with your service, can you explain the fees?",
    "You're so stupid and worthless, I can't believe how incompetent you are.",
    "What steps can I take to improve my credit score?"
]


results = evaluate_and_infer(model_name, X_test, y_test, examples)





tokenizer,model,output_model_dir = model_definition_bert_LoRA_quantized(r=16)


# Create train and validation datasets
train_dataset = Dataset(X_train, y_train, tokenizer=tokenizer)
val_dataset = Dataset(X_val, y_val, tokenizer=tokenizer)


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir=output_model_dir,
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,   
    num_train_epochs=1,
    seed=0,
    optim="adamw_torch_fused",  
    lr_scheduler_type="cosine", 
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_dir='./logs',          
    logging_steps=100,              
    report_to="tensorboard",        
    save_steps=500,                 
    fp16=torch.cuda.is_available(), 
    dataloader_num_workers=4,       
    gradient_accumulation_steps=2,  
    warmup_steps=500,              
    weight_decay=0.01,              
    no_cuda=False                   
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()


# 4. Create model card
from huggingface_hub import login, HfApi
model_card = """
---
language: en
license: mit
datasets:
  - jigsaw-toxic-comment-classification-challenge
  - kaggle/jigsaw-toxic-comment-classification-challenge
tags:
  - text-classification
---

## Usage with PEFT

This model uses LoRA fine-tuning. To use it:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "anjali-mudgal/prompt_guardrail_bert-LoRA")
lora_config = LoraConfig(
    r=8,  
    lora_alpha=32,  
    lora_dropout=0.15, 
    bias="all",  
    task_type="SEQ_CLS",
    target_modules=["query", "key", "value", "output.dense"]
)

"""
with open(f"{output_model_dir}/README.md", "w") as f:
    f.write(model_card)

# 1. Save both model and tokenizer to the local directory
model_dir = f"/kaggle/working/{output_model_dir}"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
username = "anjali-mudgal"
base_repo_id = f"{username}/prompt-guardrail"

# 5. Push to hub
api = HfApi(token="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")
custom_model_name = output_model_dir
model_dir = f"/kaggle/working/{custom_model_name}"
model_repo_id = f"{username}/prompt_guardrail_{custom_model_name}"
api.create_repo(repo_id=model_repo_id,private=False, exist_ok=True)
api.upload_folder(
    folder_path=model_dir,
    repo_id=model_repo_id,
    commit_message= f"Upload {output_model_dir}"
)
print(f"Model successfully pushed to https://huggingface.co/{model_repo_id}")






model_name = "anjali-mudgal/prompt_guardrail_bert-LoRA-r8" 
import time
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import confusion_matrix
def evaluate_and_infer(model_name, X_test, y_test, examples):
    
    print(f"Loading model from {model_name}...")

    # base model
    base_model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # LoRA weights
    model = PeftModel.from_pretrained(base_model, "anjali-mudgal/prompt_guardrail_bert-LoRA-r8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    
    print("Evaluating on test set...")
    test_dataset = Dataset(texts=X_test, labels=y_test, tokenizer=tokenizer)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )
    
    all_preds = []
    all_labels = []
    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    total_time = time.time() - start_time
    avg_time_per_example = total_time / len(y_test) * 1000  # milliseconds
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n----- TEST SET RESULTS -----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average inference time: {avg_time_per_example:.2f} ms per example")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["SAFE", "UNSAFE"]))
     

    print("\n----- EXAMPLE INFERENCES -----")
    results = []
    
    for example in examples:
        inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # inference time
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = (time.time() - start) * 1000  # ms
        
        # Get classification
        pred_class = torch.argmax(outputs.logits, dim=1).item()
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = probs[0][pred_class].item()
        
        safety = "SAFE" if pred_class == 0 else "UNSAFE"
        
        results.append({
            "text": example, 
            "prediction": safety,
            "confidence": f"{confidence:.4f}",
            "time_ms": f"{inference_time:.2f}"
        })
    
  
    df_results = pd.DataFrame(results)
    print(df_results)
    
    return {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_inference_time_ms": avg_time_per_example
        },
        "example_results": df_results
    }


examples = [
    "Hello, can you help me with my customer support issue?",
    "I hate everyone from that country, they should all die!",
    "I'm feeling a bit frustrated with your service, can you explain the fees?",
    "You're so stupid and worthless, I can't believe how incompetent you are.",
    "What steps can I take to improve my credit score?"
]


results = evaluate_and_infer(model_name, X_test, y_test, examples)


tokenizer,model,output_model_dir = model_definition_roberta_base_LoRA(r=8)


train_dataset = Dataset(X_train, y_train, tokenizer=tokenizer)
val_dataset = Dataset(X_val, y_val, tokenizer=tokenizer)


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir=output_model_dir,
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,   
    num_train_epochs=1,
    seed=0,
    optim="adamw_torch_fused",  
    lr_scheduler_type="cosine", 
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_dir='./logs',          
    logging_steps=100,              
    report_to="tensorboard",        
    save_steps=500,                 
    fp16=torch.cuda.is_available(), 
    dataloader_num_workers=4,       
    gradient_accumulation_steps=2,  
    warmup_steps=500,              
    weight_decay=0.01,              
    no_cuda=False                   
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()


# 4. Create model card
from huggingface_hub import login, HfApi
model_card = """
---
language: en
license: mit
datasets:
  - jigsaw-toxic-comment-classification-challenge
  - kaggle/jigsaw-toxic-comment-classification-challenge
tags:
  - text-classification
---

"""

with open(f"{output_model_dir}/README.md", "w") as f:
    f.write(model_card)
    
model_dir = f"/kaggle/working/{output_model_dir}"
merged_model = model.merge_and_unload()
merged_model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
username = "anjali-mudgal"

api = HfApi(token="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")

custom_model_name = output_model_dir
model_dir = f"/kaggle/working/{custom_model_name}"
model_repo_id = f"{username}/prompt_guardrail_{custom_model_name}_8"


api.create_repo(repo_id=model_repo_id,private=False, exist_ok=True)
api.upload_folder(
    folder_path=model_dir,
    repo_id=model_repo_id,
    commit_message= f"Upload {output_model_dir}"
)
print(f"Model successfully pushed to https://huggingface.co/{model_repo_id}")


# Model successfully pushed to https://huggingface.co/anjali-mudgal/prompt_guardrail_roberta_base-LoRA_8



model_name = "anjali-mudgal/prompt_guardrail_roberta_base-LoRA_8" 
import time
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
      

def evaluate_and_infer(model_name, X_test, y_test, examples):
    
    print(f"Loading model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    

    print("Evaluating on test set...")
    test_dataset = Dataset(texts=X_test, labels=y_test, tokenizer=tokenizer)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )
    
    
    all_preds = []
    all_labels = []
    start_time = time.time()
    
    # Evaluate
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate time taken
    total_time = time.time() - start_time
    avg_time_per_example = total_time / len(y_test) * 1000  # milliseconds
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n----- TEST SET RESULTS -----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average inference time: {avg_time_per_example:.2f} ms per example")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["SAFE", "UNSAFE"]))
    
    print("\n----- EXAMPLE INFERENCES -----")
    results = []
    
    for example in examples:
        # Process text
        inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Measure inference time
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = (time.time() - start) * 1000  # ms
        
        # Get classification
        pred_class = torch.argmax(outputs.logits, dim=1).item()
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = probs[0][pred_class].item()
        
        safety = "SAFE" if pred_class == 0 else "UNSAFE"
        
        results.append({
            "text": example, 
            "prediction": safety,
            "confidence": f"{confidence:.4f}",
            "time_ms": f"{inference_time:.2f}"
        })
    
  
    df_results = pd.DataFrame(results)
    print(df_results)
    
    return {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_inference_time_ms": avg_time_per_example
        },
        "example_results": df_results
    }


examples = [
    "Hello, can you help me with my customer support issue?",
    "I hate everyone from that country, they should all die!",
    "I'm feeling a bit frustrated with your service, can you explain the fees?",
    "You're so stupid and worthless, I can't believe how incompetent you are.",
    "What steps can I take to improve my credit score?"
]


results = evaluate_and_infer(model_name, X_test, y_test, examples)




