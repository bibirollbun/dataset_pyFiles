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


train_csv=pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv")
sample_sub=pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/sample_submission.csv")


for i in range(1, 3):
   globals()[f"train{i}"] = train_csv.copy()


import pandas as pd
import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Define Bangla stopwords
stop_words = set(stopwords.words('bengali'))

def enhanced_bangla_cleaner(text, remove_english=True, remove_punctuation=True, normalize=True, fix_spacing=True):
    """à¦�à¦•à¦Ÿà¦¿ à¦�à¦•à§€à¦­à§‚à¦¤ à¦«à¦¾à¦‚à¦¶à¦¨ à¦¯à§‡à¦Ÿà¦¿ à¦•à¦¾à¦¸à§�à¦Ÿà¦® à¦°à¦¿à¦ªà§�à¦²à§‡à¦¸à¦®à§‡à¦¨à§�à¦Ÿ, à¦•à§�à¦²à¦¿à¦¨à¦¿à¦‚ à¦�à¦¬à¦‚ à¦ªà§�à¦°à¦¿à¦ªà§�à¦°à¦¸à§‡à¦¸à¦¿à¦‚ à¦�à¦•à¦¸à¦¾à¦¥à§‡ à¦•à¦°à§‡"""
    
    # ----------------------------
    # à¦¸à§�à¦Ÿà§‡à¦ª à§§: à¦•à¦¾à¦¸à§�à¦Ÿà¦® à¦°à¦¿à¦ªà§�à¦²à§‡à¦¸à¦®à§‡à¦¨à§�à¦Ÿ à¦¨à¦¿à§Ÿà¦®
    # ----------------------------
    # à¦ªà§‡à¦®à§‡à¦¨à§�à¦Ÿ à¦Ÿà§‡à¦•à§�à¦¸à¦Ÿ à¦°à¦¿à¦ªà§�à¦²à§‡à¦¸
    text = re.sub(
        r"Payment Tk.*?successful",
        "à¦ªà§‡à¦®à§‡à¦¨à§�à¦Ÿ à¦Ÿà¦¾à¦•à¦¾ [amount] à¦—à§�à¦°à¦¾à¦®à§€à¦£à¦«à§‹à¦¨ à¦²à¦¿à¦®à¦¿à¦Ÿà§‡à¦¡ mycompany_y à¦¡à¦¾à¦‡à¦°à§‡à¦•à§�à¦Ÿ à¦šà¦¾à¦°à§�à¦œ [transaction_id] à¦¸à¦«à¦²à¥¤ à¦�à¦®à¦¨ à¦®à§‡à¦¸à§‡à¦œ à¦¦à¦¿à§Ÿà§‡ [amount] à¦Ÿà¦¾à¦•à¦¾ à¦•à¦¾à¦Ÿà¦¾à¦° à¦•à¦¾à¦°à¦£ à¦•à§€? à¦†à¦®à¦¿ à¦—à§�à¦°à¦¾à¦®à§€à¦£ à¦¸à¦¿à¦® à¦¬à§�à¦¯à¦¬à¦¹à¦¾à¦° à¦•à¦°à¦¿ à¦¨à¦¾à¥¤ company_x-à¦� à¦¯à§‹à¦—à¦¾à¦¯à§‹à¦— à¦•à¦°à¦²à§‡ à¦¤à¦¾à¦°à¦¾ à¦�à¦•à§‡ à¦¸à¦¾à¦°à§�à¦­à¦¿à¦¸ à¦¬à¦²à¦›à§‡, à¦•à¦¿à¦¨à§�à¦¤à§� à¦†à¦®à¦¿ à¦�à¦®à¦¨ à¦¸à¦¾à¦°à§�à¦­à¦¿à¦¸ à¦¯à§�à¦•à§�à¦¤ à¦•à¦°à¦¿à¦¨à¦¿à¥¤", 
        text
    )
    
    # à¦²à§‹à¦¨ à¦¸à¦®à§�à¦ªà¦°à§�à¦•à¦¿à¦¤ à¦¬à¦¾à¦•à§�à¦¯ à¦°à¦¿à¦ªà§�à¦²à§‡à¦¸
    text = re.sub(
        r"à¦–à§�à¦¬ à¦¦à¦°à¦•à¦¾à¦°à§‡à¦° à¦Ÿà¦¾à¦‡à¦®à§‡ company_x à¦¥à§‡à¦•à§‡ à¦¸à¦¹à¦œà§‡ à¦²à§‹à¦¨ à¦ªà§‡à¦²à¦¾à¦®\?See Translation",
        "à¦†à¦®à¦¿ à¦œà¦°à§�à¦°à¦¿ à¦¸à¦®à§Ÿà§‡ à¦¸à¦¹à¦œà§‡ à¦²à§‹à¦¨ à¦ªà§‡à§Ÿà§‡à¦›à¦¿à¦²à¦¾à¦®à¥¤",
        text
    )
    
    # 'See Translation' à¦°à¦¿à¦®à§�à¦­
    text = re.sub(r'\|?See Translation', '', text)
    
    # company_x/y à¦•à¦¨à¦Ÿà§‡à¦•à§�à¦¸à¦Ÿ-à¦…à¦¨à§�à¦¯à¦¾à§Ÿà§€ à¦°à¦¿à¦ªà§�à¦²à§‡à¦¸
    if re.search(r'company_|mycompany_', text):
        if re.search(r'à¦²à§‹à¦¨|à¦¸à§‡à¦¬à¦¾|à¦¨à¦•|à¦¯à§‹à¦—à¦¾à¦¯à§‹à¦—|à¦•à¦²', text):
            text = re.sub(r'company_\w+|mycompany_\w+', 'à¦“à¦‡ à¦ªà§�à¦°à¦¤à¦¿à¦·à§�à¦ à¦¾à¦¨', text)
        else:
            text = re.sub(r'company_\w+|mycompany_\w+', 'à¦�à¦•à¦Ÿà¦¿ à¦•à§‹à¦®à§�à¦ªà¦¾à¦¨à¦¿', text)
    
    # ----------------------------
    # à¦¸à§�à¦Ÿà§‡à¦ª à§¨: à¦¬à§‡à¦¸à¦¿à¦• à¦Ÿà§‡à¦•à§�à¦¸à¦Ÿ à¦•à§�à¦²à¦¿à¦¨à¦¿à¦‚
    # ----------------------------
    # à¦²à§‹à§Ÿà¦¾à¦°à¦•à§‡à¦¸ à¦•à¦°à§�à¦¨
    text = text.lower()
    
    # à¦‡à¦‚à¦°à§‡à¦œà¦¿ à¦“ à¦¨à¦¾à¦®à§�à¦¬à¦¾à¦° à¦°à¦¿à¦®à§�à¦­ (à¦¬à§�à¦°à§�à¦¯à¦¾à¦•à§‡à¦Ÿà§‡à¦¡ à¦Ÿà¦¾à¦°à§�à¦®à¦¸ à¦°à§‡à¦–à§‡)
    if remove_english:
        parts = re.split(r'(\[.*?\])', text)
        cleaned_parts = []
        for part in parts:
            if re.match(r'\[.*?\]', part):
                cleaned_parts.append(part)
            else:
                cleaned_part = re.sub(r'[a-z0-9_]+', '', part, flags=re.IGNORECASE)
                cleaned_parts.append(cleaned_part)
        text = ''.join(cleaned_parts)
    
    # à¦ªà¦¾à¦‚à¦šà§�à§Ÿà§‡à¦¶à¦¨ à¦°à¦¿à¦®à§�à¦­
    if remove_punctuation:
        text = re.sub(r'[à¥¤à¥¥!?.,;:â€œâ€�"\'â€˜â€™â€”â€¦()\[\]{}<>@#$%^&*_+=|\\/~`]', '', text)
    
    # à¦¶à§�à¦§à§�à¦®à¦¾à¦¤à§�à¦° à¦¬à¦¾à¦‚à¦²à¦¾ à¦“ à¦¸à§�à¦ªà§‡à¦¸ à¦°à¦¾à¦–à§�à¦¨
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
    
    # à¦¸à§�à¦ªà§‡à¦¸à¦¿à¦‚ à¦‡à¦¸à§�à¦¯à§� à¦«à¦¿à¦•à§�à¦¸
    if fix_spacing:
        text = re.sub(r'(\S+)\s à§‡', r'\1à§‡', text)  # "à¦¬à¦¾à¦¨à§�à¦¦à§‡ à¦°" -> "à¦¬à¦¾à¦¨à§�à¦¦à§‡à¦°"
        text = re.sub(r'(\S+)\s à¦°à§‡', r'\1à¦°à§‡', text) # "à¦†à¦®à¦¾à¦° à§‡à¦°" -> "à¦†à¦®à¦¾à¦°à§‡"
    
    # à¦•à¦®à¦¨ à¦¸à§�à¦ªà§‡à¦²à¦¿à¦‚ à¦«à¦¿à¦•à§�à¦¸
    spelling_fixes = {
        "à¦¸ à§�à¦–à§€à¦¨": "à¦¸à§�à¦–à§€à¦¨",
        "à¦¸à¦¿ à¦®à§�à¦ªà¦²": "à¦¸à¦¿à¦®à§�à¦ªà¦²",
        "à¦¨à§‡ à¦­à¦¿à¦—à§‡à¦Ÿ": "à¦¨à§‡à¦­à¦¿à¦—à§‡à¦Ÿ",
        "à¦¶ à§�à¦§à§�": "à¦¶à§�à¦§à§�",
        "à¦¸ à§�à¦›à¦¿": "à¦ªà¦¾à¦šà§�à¦›à¦¿",
        "à¦•à§‹à¦®à§�à¦ª à¦¾à¦¨à¦¿à¦°": "à¦•à§‹à¦®à§�à¦ªà¦¾à¦¨à¦¿à¦°",
        "à¦¨ à§‡à¦­à¦¿à¦—à§‡à¦Ÿ": "à¦¨à§‡à¦­à¦¿à¦—à§‡à¦Ÿ",
        "à§‡ à¦¨à¦•": "à¦� à¦¨à¦•",
        "à¦¬à¦¾à¦¸à§�à¦¤à¦¬ à§‡":"à¦¬à¦¾à¦¸à§�à¦¤à¦¬à§‡",
        "à¦¸à§�à¦¦ à§�à¦›à¦¿" : "à¦¸à§�à¦¦ à¦ªà¦¾à¦šà§�à¦›à¦¿",
        "à¦¸ à¦¿à¦®à§�à¦ªà¦²": "à¦¸à¦¿à¦®à§�à¦ªà¦²",
        "à¦ªà§�à¦°à§‹à¦¡à¦¾à¦•à§�à¦Ÿ à§‡à¦°":"à¦ªà§�à¦°à§‹à¦¡à¦¾à¦•à§�à¦Ÿà§‡à¦°"
    }
    for wrong, right in spelling_fixes.items():
        text = text.replace(wrong, right)
    
    # à¦‡à¦‰à¦¨à¦¿à¦•à§‹à¦¡ à¦¨à¦°à§�à¦®à¦¾à¦²à¦¾à¦‡à¦œà§‡à¦¶à¦¨
    text = unicodedata.normalize('NFKC', text)
    
    # ----------------------------
    # à¦¸à§�à¦Ÿà§‡à¦ª à§©: à¦Ÿà§‹à¦•à§‡à¦¨à¦¾à¦‡à¦œà§‡à¦¶à¦¨ à¦“ à¦�à¦¡à¦­à¦¾à¦¨à§�à¦¸à¦¡ à¦ªà§�à¦°à¦¸à§‡à¦¸à¦¿à¦‚
    # ----------------------------
    tokens = word_tokenize(text)
    
    # à¦•à¦¾à¦°à¦¾à¦•à§�à¦Ÿà¦¾à¦° à¦¨à¦°à§�à¦®à¦¾à¦²à¦¾à¦‡à¦œà§‡à¦¶à¦¨ (à¦¯à§‡à¦®à¦¨: "à¦¥à§�à¦¯" à¦ à¦¿à¦• à¦•à¦°à¦¾)
    if normalize:
        tokens = [re.sub(r'(à¦¥à§�)(à¦¯|à¦¯à¦¼)', 'à¦¥à§�à¦¯', word) for word in tokens]
        tokens = [re.sub(r'(à¦…à§�)(à¦¯|à¦¯à¦¼)', 'à¦…à§�à¦¯', word) for word in tokens]
    
    # à¦¸à§�à¦Ÿà¦ªà¦“à§Ÿà¦¾à¦°à§�à¦¡ à¦«à¦¿à¦²à§�à¦Ÿà¦¾à¦°
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# à¦¬à§�à¦¯à¦¬à¦¹à¦¾à¦°à§‡à¦° à¦‰à¦¦à¦¾à¦¹à¦°à¦£
train1["clean_text"] = train1["text"].apply(enhanced_bangla_cleaner)


from torch.utils.data import Dataset


# Data manipulation
import pandas as pd
import numpy as np

# PyTorch for deep learning
import torch
from torch.utils.data import Dataset
import torch.nn as nn

# HuggingFace Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Sklearn for preprocessing and metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

# Garbage collection and memory management
import gc

# Encode the labels
le = LabelEncoder()
train1["label"] = le.fit_transform(train1["sentiment"])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")

# Custom Dataset
class BanglaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding='max_length', max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# This will store predictions from all folds
all_preds = []

# Loop through the folds
for fold, (train_idx, val_idx) in enumerate(skf.split(train1["clean_text"], train1["label"])):
    print(f"\nğŸ”� Fold {fold+1}")

    # Split data
    train_texts = train1.iloc[train_idx]["clean_text"].tolist()
    val_texts = train1.iloc[val_idx]["clean_text"].tolist()
    train_labels = train1.iloc[train_idx]["label"].tolist()
    val_labels = train1.iloc[val_idx]["label"].tolist()

    # Dataset
    train_dataset = BanglaDataset(train_texts, train_labels, tokenizer)
    val_dataset = BanglaDataset(val_texts, val_labels, tokenizer)

    # Class Weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Custom Model with Weighted Loss
    from transformers import BertForSequenceClassification
    import torch.nn as nn

    class CustomModel(AutoModelForSequenceClassification):
        def forward(self, input_ids=None, attention_mask=None, labels=None):
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            logits = outputs.logits
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}

    model = CustomModel.from_pretrained("csebuetnlp/banglabert", num_labels=3).to(device)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold+1}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_dir=f"./logs_fold_{fold+1}",
        logging_steps=10,
        save_steps=50,
        do_train=True,
        do_eval=True,  # optional
        report_to="none"
    )

    # Define compute_metrics (optional, for evaluation)
    from sklearn.metrics import accuracy_score, f1_score
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='macro'),
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Evaluate on validation set
    print("ğŸ“Š Fold Validation:")
    trainer.evaluate()

    # Prediction on the entire dataset (train1)
    model.eval()
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in tokenizer(train1["clean_text"].tolist(), truncation=True, padding='max_length', max_length=128, return_tensors="pt").items()})
        logits = outputs.logits
        preds = torch.argmax(logits, axis=1).cpu().numpy()
        all_preds.append(preds)

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Average predictions from all folds
final_preds = np.mean(all_preds, axis=0).astype(int)

# Inverse transform the predictions to the original sentiment labels
decoded_preds = le.inverse_transform(final_preds)

# Create the submission DataFrame
submission = pd.DataFrame({
    "id": train1["id"],
    "sentiment": decoded_preds
})

# Save the submission file
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved!")


# Initialize lists to store metrics for each fold
accuracy_scores = []
f1_scores = []

# Loop through the folds
for fold, (train_idx, val_idx) in enumerate(skf.split(train1["clean_text"], train1["label"])):
    print(f"\nğŸ”� Fold {fold+1}")

    # Split data
    train_texts = train1.iloc[train_idx]["clean_text"].tolist()
    val_texts = train1.iloc[val_idx]["clean_text"].tolist()
    train_labels = train1.iloc[train_idx]["label"].tolist()
    val_labels = train1.iloc[val_idx]["label"].tolist()

    # Dataset
    train_dataset = BanglaDataset(train_texts, train_labels, tokenizer)
    val_dataset = BanglaDataset(val_texts, val_labels, tokenizer)

    # Class Weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Custom Model with Weighted Loss
    model = CustomModel.from_pretrained("csebuetnlp/banglabert", num_labels=3).to(device)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Evaluate on validation set
    print("ğŸ“Š Fold Validation:")
    result = trainer.evaluate()

    # Store metrics from this fold
    accuracy_scores.append(result['eval_accuracy'])
    f1_scores.append(result['eval_f1'])

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Calculate average and standard deviation of the metrics across all folds
avg_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
avg_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

# Print the results
print("\nCross-Validation Scores:")
print(f"Average Accuracy: {avg_accuracy:.4f} Â± {std_accuracy:.4f}")
print(f"Average F1 Score: {avg_f1:.4f} Â± {std_f1:.4f}")


