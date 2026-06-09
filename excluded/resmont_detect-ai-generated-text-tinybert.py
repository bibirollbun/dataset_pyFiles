import pandas as pd
import numpy as np
import os
import string
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset


Train_promts = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_prompts.csv")
Train_promts.head(10)


Test_essays = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")
Test_essays.head(10)


Sample = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv")
Sample.head(10)


Train_essays = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
Train_essays.head(10)


Test_essays.isnull().sum()


plt.figure(figsize=(6, 4))
sns.countplot(data=Train_essays, x='generated')
plt.title("Distribution of text")
plt.xlabel("Generated (0 = Human, 1 = AI)")
plt.ylabel("Count")
plt.show()


Daigit_train = pd.read_csv("/kaggle/input/daigt-v4-train-dataset/train_v4_drcat_01.csv")
Daigit_train.rename(columns = {"label":"generated"}, inplace=True)
Daigit_train.head()


Train_essays_final = pd.concat([Train_essays[["text", "generated"]], Daigit_train[["text", "generated"]]])

def clean_text(text):
    text = text.replace('\n', ' ')
    text = ''.join(char for char in text if char in string.printable or char.isspace())
    text = ' '.join(text.split()) 
    return text

Train_essays_final['text'].apply(clean_text)

Train_essays_final.head(10)


Train_essays_final["text_length"] = Train_essays_final["text"].apply(lambda x : len(x.split()))
Train_essays_final.head()


Train_essays_final.describe()


grouped = Train_essays_final.groupby('generated')['text_length'].mean().reset_index()

plt.subplot(1, 2, 2)
plt.bar(grouped['generated'], grouped['text_length'], color=['blue', 'red'])
plt.title('Avg words count')
plt.ylabel('Count')

plt.xticks([0, 1], ['Human', 'AI'])

plt.show()


Train_essays_final = Train_essays_final.drop("text_length", axis=1)
Train_essays_final.head()


labels = Train_essays_final['generated'].tolist()
texts = Train_essays_final['text'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)


train_dict = {'text': train_texts, 'label': train_labels}
val_dict = {'text': val_texts, 'label': val_labels}

train_dataset_hf = Dataset.from_dict(train_dict)
val_dataset_hf = Dataset.from_dict(val_dict)


MODEL_NAME = '/kaggle/input/tinybert/tinybert'


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)

num_processors = os.cpu_count()
tokenized_train_dataset = train_dataset_hf.map(tokenize_batch, batched=True, num_proc=num_processors)

tokenized_val_dataset = val_dataset_hf.map(tokenize_batch, batched=True, num_proc=num_processors)


tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text'])
tokenized_train_dataset = tokenized_train_dataset.rename_column('label', 'labels')
tokenized_train_dataset.set_format('torch')

tokenized_val_dataset = tokenized_val_dataset.remove_columns(['text'])
tokenized_val_dataset = tokenized_val_dataset.rename_column('label', 'labels')
tokenized_val_dataset.set_format('torch')



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
    probabilities = softmax(logits, axis=-1)[:, 1]
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = float("nan")
    
    return {"accuracy": accuracy, "auc": auc}


model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="steps",      
    eval_steps=250,                 
    save_strategy="steps",
    save_steps=250,
    per_device_train_batch_size=64,  
    per_device_eval_batch_size=64,    
    gradient_accumulation_steps=1,    
    learning_rate=3e-5,           
    num_train_epochs=1,              
    warmup_steps=200,              
    logging_dir='./logs',
    logging_steps=50,
    fp16=False,
    bf16=True,                
    dataloader_num_workers=4,        
    dataloader_pin_memory=True,        
    gradient_checkpointing=False,      
    report_to="none",                
    optim="adamw_torch",    
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    torch_compile=True,
    max_grad_norm=1.0
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=1,
    early_stopping_threshold=0.005
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset, 
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)


trainer.train()



test_data = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")
test_texts = test_data["text"].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_probabilities = []
batch_size = 64

for i in tqdm(range(0, len(test_texts), batch_size)):
    batch = test_texts[i : i + batch_size]
    
    inputs = tokenizer(
        batch, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)


    probs_class_1 = probabilities[:, 1].cpu().numpy()
    
    all_probabilities.extend(probs_class_1)


submission_df = pd.DataFrame({'id': test_data['id'], 'generated': all_probabilities})
submission_df.to_csv('submission.csv', index=False)



eval_results = trainer.predict(tokenized_val_dataset)

predicted_labels = np.argmax(eval_results.predictions, axis=-1)
true_labels = eval_results.label_ids


cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Human', 'AI'],
            yticklabels=['Human', 'AI'])

plt.title('Confusion Matrix on Validation Data')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


logits = eval_results.predictions

probabilities = softmax(logits, axis=-1)

y_scores = probabilities[:, 1]

y_true = eval_results.label_ids

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

