import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm.auto import tqdm
import os
import nltk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re


df = pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv")



# Define Bengali punctuation marks for splitting
BENGALI_PUNCTUATIONS = ['।', '?', '!',',',';']  # Prioritize '।' (danda) as primary sentence separator

# Initialize lists for train and test data
train_data = []
test_data = []

for index, row in df.iterrows():
    text = row['text']
    split_position = -1

    # Find the first occurrence of any Bengali punctuation
    for punct in BENGALI_PUNCTUATIONS:
        pos = text.find(punct)
        if pos != -1:
            split_position = pos
            break  # Split at the first detected punctuation

    # Split the text
    if split_position != -1:
        test_text = text[:split_position+1].strip()  # Include punctuation in train
        train_text = text[split_position+1:].strip()   # Remaining text for test
    else:
        train_text = text  # No punctuation found: keep full text in train
        test_text = ""     # Leave test text empty

    # Add to train and test datasets
    train_data.append({
        "id": row["id"],
        "text": train_text,
        "sentiment": row["sentiment"]
    })

    test_data.append({
        "id": row["id"],
        "text": test_text
    })

# Create DataFrames and save to CSV
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df.to_csv("train_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)


train_df = pd.read_csv('/kaggle/working/train_split.csv')
test_df=pd.read_csv("/kaggle/working/test_split.csv")
train_df.head()


# Map sentiment labels to integers
label_dict = {'positive': 2, 'neutral': 1, 'negative': 0}
train_df['label'] = train_df['sentiment'].map(label_dict)


train_df['label'].value_counts().plot(kind="bar", rot=0)


def text_to_word_list(text):
    text = text.split()
    return text

def replace_strings(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\u00C0-\u017F"          #latin
                           u"\u2000-\u206F"          #generalPunctuations
                               
                           "]+", flags=re.UNICODE)
    english_pattern=re.compile('[a-zA-Z0-9]+', flags=re.I)
    #latin_pattern=re.compile('[A-Za-z\u00C0-\u00D6\u00D8-\u00f6\u00f8-\u00ff\s]*',)
    
    text=emoji_pattern.sub(r'', text)
    text=english_pattern.sub(r'', text)

    return text

def remove_punctuations(my_str):
    # define punctuation
    punctuations = '''````£|¢|Ñ+-*/=EROero৳০১২৩৪৫৬৭৮৯012–34567•89।!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—॥”‰⚽️✌�￰৷￰'''
    
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char

    # display the unpunctuated string
    return no_punct



def joining(text):
    out=' '.join(text)
    return out

def preprocessing(text):
    out=remove_punctuations(replace_strings(text))
    return out


!pip install -q transformers datasets torch scikit-learn pandas tqdm


def clean_sentence(sent):
    
    sent = re.sub('[?.`*^()!°¢܌Ͱ̰ߒנ~×Ҡߘ:ҰߑÍ|।;!,&%\'@#$><A-Za-z0+-9=./''""_০-৯]', '', sent)
    sent = re.sub(r'(\W)(?=\1)', '', sent)
    sent = re.sub(r'https?:\/\/.*[\r\n]*', '', sent, flags=re.MULTILINE)
    sent = re.sub(r'\<a href', ' ', sent)
    sent = re.sub(r'&amp;', '', sent) 
    sent = re.sub(r'\U0001F600-\U0001F64F','',sent)
    sent = re.sub(r'\U0001F300-\U0001F5FF','',sent)
    sent = re.sub(r'\U0001F680-\U0001F6FF','',sent)
    sent = re.sub(r'\u00C0-\u017F','',sent)
    sent = re.sub(r'\U0001F1E0-\U0001F1FF','',sent)
    sent = re.sub(r'\U00002702-\U000027B0','',sent)
    sent = re.sub(r'\U000024C2-\U0001F251','',sent)
    sent = re.sub(r'\u2000-\u206F','',sent)
    
    sent = re.sub(r'<br />', ' ', sent)
    sent = re.sub(r'\'', ' ', sent)
    sent = re.sub(r'ߑͰߑ̰ߒנ', '', sent)
    sent = re.sub(r'ߎɰߎɰߎɍ', '', sent)
    
    sent = sent.strip()
    return sent


stop_words = {'এ', 'হয়', 'কি', 'কী', 'এর', 'কে', 'যে', 'এই', 'বা', 'সব', 'টি', 'তা',
       'সে', 'তাই', 'সেই', 'তার', 'আগে', 'যদি', 'আছে', 'আমি', 'এবং', 'করে', 'কার', 'এটি', 'হতে', 'যায়',
       'আরও', 'যাক', 'খুব', 'উপর', 'পরে', 'হবে', 'কেন', 'কখন', 'সকল', 'হয়', 'ঠিক', 'একই', 'কোন',
       'ছিল', 'খুবই', 'কোনো', 'অধীন', 'যারা', 'তারা', 'গুলি', 'তাকে', 'সেটা', 'সময়', 'আমার', 'আমরা', 'সবার',
       'উভয়', 'একটা', 'আপনি', 'নিয়ে', 'একটি', 'বন্ধ', 'জন্য', 'শুধু', 'যেটা', 'উচিত', 'মাঝে', 'থেকে', 'করবে',
       'আবার', 'উপরে', 'সেটি', 'কিছু', 'কারণ', 'যেমন', 'তিনি', 'মধ্যে', 'আমাকে', 'করছেন', 'তুলনা', 'তারপর',
       'নিজেই', 'থাকার', 'নিজের', 'পারেন', 'একবার', 'সঙ্গে', 'ইচ্ছা', 'নীচের', 'এগুলো', 'আপনার', 'অধীনে', 'কিংবা',
       'এখানে', 'তাহলে', 'কয়েক', 'জন্যে', 'হচ্ছে', 'তাদের', 'কোথায়', 'কিন্তু', 'নিজেকে', 'যতক্ষণ', 'আমাদের',
       'দ্বারা', 'হয়েছে', ' সঙ্গে', 'সেখানে', 'কিভাবে', 'মাধ্যমে', 'নিজেদের', 'তুলনায়', 'প্রতিটি',
       'তাদেরকে', 'ইত্যাদি', 'সম্পর্কে', 'সর্বাধিক', 'বিরুদ্ধে', 'অন্যান্য'}

def remove_stop_words(text):
    text = [w for w in text if not w in stop_words]
    text = ' '.join(text)
    return text


def tokenized_data(sent):
    tokenized_text = sent.split()
    return tokenized_text


train_df.dropna(subset=['text'],inplace=True)

train_df['text'] = [remove_stop_words(tokenized_data(sent)) for sent in train_df['text'].tolist()]
train_df['text'] = train_df.text.apply(lambda x: preprocessing(str(x)))


!pip install -q indic-nlp-library


from indicnlp.tokenize import indic_tokenize
train_df["tokenized_text"] = train_df["text"].apply(indic_tokenize.trivial_tokenize)


train_df.head()


test_df['text'] = [remove_stop_words(tokenized_data(sent)) for sent in test_df['text'].tolist()]
test_df['text'] = test_df.text.apply(lambda x: preprocessing(str(x)))
test_df["tokenized_text"] = test_df["text"].apply(indic_tokenize.trivial_tokenize)


from sklearn.model_selection import train_test_split
train_df, val_df  = train_test_split(train_df, test_size=0.2, random_state=42)


!pip install -q torch transformers pandas tqdm scipy numpy accelerate


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")


# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
LABELS = [0, 1, 2]


import torch
import numpy as np
from transformers import (
    BertForSequenceClassification,
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 10
PATIENCE = 3  # Early stopping patience
LABELS = ["negative", "neutral", "positive"]  # Adjust based on your labels
MODEL_NAME = "sagorsarker/bangla-bert-base"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS)).to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
num_train_steps = len(train_df) // BATCH_SIZE * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

# Dataset Class
class SentenceClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, labels, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['sentiment']  # Ensure 'sentiment' column is used

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels.index(label), dtype=torch.long)
        }

# Prepare DataLoaders
train_dataset = SentenceClassificationDataset(train_df, tokenizer, LABELS)
val_dataset = SentenceClassificationDataset(val_df, tokenizer, LABELS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to evaluate model
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Training Loop with Early Stopping
best_val_loss = float("inf")
early_stop_counter = 0
best_model_path = "best_bangla_bert.pth"

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)

    # Evaluate on validation set
    val_loss, val_accuracy = evaluate(model, val_loader)

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Save model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ New best model saved at epoch {epoch+1}")
    else:
        early_stop_counter += 1
        print(f"⚠️ Validation loss increased ({early_stop_counter}/{PATIENCE})")

    # Stop training if validation loss keeps degrading
    if early_stop_counter >= PATIENCE:
        print("⏹️ Early stopping triggered. Training stopped.")
        break

# Load the best model before inference
model.load_state_dict(torch.load(best_model_path))
print("\n✅ Best model loaded for inference.")



# Load best model
# MODEL_NAME = "sagorsarker/bangla-bert-base"
LABELS = ["negative", "neutral", "positive"]  # Mapping
# BEST_MODEL_PATH = "best_bangla_bert.pth"

# Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
# model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.to(device)
model.eval()  # Set model to evaluation mode

# Define Dataset Class for Test Data
class TestDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "id": torch.tensor(int(self.dataframe.iloc[idx]["id"]))  # Ensure id is a tensor
        }

# Prepare Test DataLoader
test_dataset = TestDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Perform Predictions
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ids = batch["id"].cpu().numpy()  # Convert tensor to NumPy array

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()  # Convert to NumPy

        # Map predictions to sentiment labels
        for i in range(len(ids)):
            predictions.append({"id": int(ids[i]), "sentiment": LABELS[preds[i]]})  # Convert id to int

# Convert predictions to DataFrame and save
pred_df = pd.DataFrame(predictions)
pred_df.to_csv("submission.csv", index=False)

print("\n✅ Predictions saved to 'predicted_sentiments.csv'")

