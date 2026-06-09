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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, get_scheduler
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


df_train = pd.read_json("/kaggle/input/depi-r-2-emotion-analysis/train.json", lines=True)
df_val = pd.read_json("/kaggle/input/depi-r-2-emotion-analysis/validation.json", lines=True)
df_test = pd.read_json("/kaggle/input/depi-r-2-emotion-analysis/test.json", lines=True)
df_sample = pd.read_csv("/kaggle/input/depi-r-2-emotion-analysis/submission.csv")





df_train.head()


len(df_train['text']), len(df_val['text'])


# df_train['No_of_sentences'] = df_train['text'].apply(lambda x: len(re.split(r'[.!?]', str(x).strip())) - 1)
# df_train['No_of_sentences'].count()


df_train['label'].unique()


df_train['label'].value_counts()


df_train.shape


df_train.info()


df_train.isnull().sum()


df_train.duplicated().sum()


df_train.drop_duplicates(inplace=True)
df_train.duplicated().sum()


# # Search for the character "o"
# result = df_train[df_train["text"].str.contains("  ", case=False)]  # case=False makes it case-insensitive
# print(result)


# df_train[df_train['label'] == 1]['text'].tolist()[:20]



# Plot the destrubition of label column after dropping class 2
fg = sns.countplot(x = df_train['label'])
fg.set_title('distrubiation of classes')
fg.set_xlabel('Classes')
fg.set_ylabel('Count')


from wordcloud import WordCloud
# Wordclouds for each emotion
for emotion in df_train['label'].unique():
    text = " ".join(df_train[df_train['label'] == emotion]['text'])
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for {emotion}")
    plt.show()





df_val.head()


df_val.shape


df_val.info()


df_val.duplicated().sum()


df_val['label'].value_counts()


# Plot the destrubition of label column after dropping class 2
fg = sns.countplot(x = df_val['label'])
fg.set_title('distrubiation of classes')
fg.set_xlabel('Classes')
fg.set_ylabel('Count')





# Wordclouds for each emotion df_val
for emotion in df_val['label'].unique():
    text = " ".join(df_val[df_val['label'] == emotion]['text'])
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for {emotion}")
    plt.show()





df_test.head()


df_test.info()


df_test.duplicated().sum()





import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load spaCy’s English model
nlp = spacy.load("en_core_web_sm")



def clean_text_spacy(text, for_embedding=False):
    """
    Cleans text using regex and spaCy.
    
    Steps:
    - Remove URLs, emails, and HTML tags.
    - Normalize all whitespace to a single space.
    - If for_embedding is False:
         * Keep only alphabetic tokens (remove digits and punctuation).
         * Remove stopwords.
         * Remove single-character tokens.
         * Convert tokens to lowercase and lemmatize.
      Otherwise, for embedding, preserve punctuation and original casing.
    
    Args:
        text (str): The input text.
        for_embedding (bool): If True, perform minimal cleaning.
        
    Returns:
        str: The cleaned text.
    """
    # Remove URLs, emails, and HTML tags using regex
    text = re.sub(r"www\.\S+|https?://\S+", " ", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Process text using spaCy
    doc = nlp(text)
    
    if for_embedding:
        # For embedding: keep tokens as they are (preserve punctuation, casing, etc.)
        tokens = [token.text for token in doc]
    else:
        # For traditional features (e.g., TF-IDF):
        # Keep only alphabetic tokens, remove stopwords, remove single-letter tokens,
        # lowercase, and lemmatize.
        tokens = [
            token.lemma_.lower() 
            for token in doc 
            if token.is_alpha and not token.is_stop and len(token.text) > 1
        ]
    return " ".join(tokens)



# Assuming df_train is your DataFrame with a "text" column
# %%time
df_train["clean_text"] = df_train["text"].map(lambda x: clean_text_spacy(x, for_embedding=False) if isinstance(x, str) else x)
print(df_train.head())



# Combine all cleaned text into one large string df_train
all_text = " ".join(df_train["clean_text"].tolist())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Cleaned Text")
plt.show()






# Assuming df_train is your DataFrame with a "text" column
# %%time
df_val["clean_text"] = df_val["text"].map(lambda x: clean_text_spacy(x, for_embedding=False) if isinstance(x, str) else x)
print(df_val.head())



# Combine all cleaned text into one large string df_val
all_text = " ".join(df_val["clean_text"].tolist())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Validation Cleaned Text")
plt.show()






from transformers import BertTokenizer, DistilBertTokenizer

# Load the tokenizers
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Report vocabulary sizes
print("BERT Vocabulary Size:", len(bert_tokenizer))
print("DistilBERT Vocabulary Size:", len(distilbert_tokenizer))



examples = [
    "I am extremely happy today!",
    "This news is terrifying and makes me scared.",
    "I love the way you smile, it's so delightful!"
]

print("BERT Tokenization:")
for text in examples:
    tokens = bert_tokenizer.tokenize(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}\n")

print("DistilBERT Tokenization:")
for text in examples:
    tokens = distilbert_tokenizer.tokenize(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}\n")



# Store length of each review 
token_lens = []

# Iterate through the content slide
for txt in df_train.text:
    tokens = bert_tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))


# plot the distribution of review lengths 
sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count')





# Custom Dataset for train/val (with labels)
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: encoding[key].squeeze(0) for key in encoding}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Custom Dataset for test (no labels)
class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {key: encoding[key].squeeze(0) for key in encoding}

# Assuming df_train, df_val, and df_test are already loaded with columns "text" and "label" for train/val.
train_dataset = EmotionDataset(df_train['text'].tolist(), df_train['label'].tolist(), bert_tokenizer)
val_dataset   = EmotionDataset(df_val['text'].tolist(), df_val['label'].tolist(), bert_tokenizer)
test_dataset  = TestDataset(df_test['text'].tolist(), bert_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)



# Compute class weights using the training labels
labels = df_train['label'].values
classes = np.unique(labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
print("Computed class weights:", class_weights)

# Convert class weights to a PyTorch tensor and move to device later
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)



# train the model
# %%time

model.train()
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        
        # Compute weighted loss using our class weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(outputs.logits, batch["labels"])
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
    print(f"Epoch {epoch+1} completed.")


# evaluate validation data
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].cpu().numpy())

from sklearn.metrics import accuracy_score, classification_report
print("Validation Accuracy:", accuracy_score(all_labels, all_preds))
print("Classification Report:\n", classification_report(all_labels, all_preds))


# test prediction
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
        test_preds.extend(preds)


# Prepare the submission DataFrame
submission = pd.DataFrame({
    'ID': df_test['id'],  # Assuming 'id' column exists in df_test
    'Prediction': test_preds
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'.")





# Custom Dataset for train/val (with labels)
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: encoding[key].squeeze(0) for key in encoding}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# For training and validation (labels provided)
train_dataset = EmotionDataset(df_train['text'].tolist(), df_train['label'].tolist(), distilbert_tokenizer)
val_dataset = EmotionDataset(df_val['text'].tolist(), df_val['label'].tolist(), distilbert_tokenizer)

# For test dataset (no labels)
class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {key: encoding[key].squeeze(0) for key in encoding}

test_dataset = TestDataset(df_test['text'].tolist(), distilbert_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Compute class weights using the training labels
labels = df_train['label'].values
classes = np.unique(labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
print("Computed class weights:", class_weights)

# Convert class weights to a PyTorch tensor and move to device later
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)


# Load the tokenizers
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Report vocabulary sizes
print("BERT Vocabulary Size:", len(bert_tokenizer))
print("DistilBERT Vocabulary Size:", len(distilbert_tokenizer))



model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)



# train the model
# %%time

model.train()
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        
        # Compute weighted loss using our class weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(outputs.logits, batch["labels"])
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
    print(f"Epoch {epoch+1} completed.")


# evaluate validation data
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].cpu().numpy())

from sklearn.metrics import accuracy_score, classification_report
print("Validation Accuracy:", accuracy_score(all_labels, all_preds))
print("Classification Report:\n", classification_report(all_labels, all_preds))


# test prediction
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
        test_preds.extend(preds)


# Prepare the submission DataFrame
submission = pd.DataFrame({
    'ID': df_test['id'],  # Assuming 'id' column exists in df_test
    'Prediction': test_preds
})

# Save the submission file
submission.to_csv('submission2.csv', index=False)

print("Submission file saved as 'submission.csv'.")




