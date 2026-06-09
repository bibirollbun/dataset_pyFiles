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


import re # regular expression for pre-procesing
import emoji #to convert emoji to text for bert
from textblob import TextBlob # hanfles spelling mistakes
!pip install transformers


# Load harassment datasetsv
harass_train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
harass_test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

# Load emotion dataset
emotion_train = pd.read_csv('/kaggle/input/emotions-dataset-for-nlp/train.txt')
emotion_test = pd.read_csv('/kaggle/input/emotions-dataset-for-nlp/test.txt')


num_rows1 = len(harass_train)
print(f"Total number of rows in harass_train: {num_rows1}")

num_rows2 = len(harass_test)
print(f"Total number of rows in harass_test: {num_rows2}")

num_rows3 = len(emotion_train)
print(f"Total number of rows in emotion_train: {num_rows3}")

num_rows4 = len(emotion_test)
print(f"Total number of rows in emotion_test: {num_rows4}")



#Display first few rows to understand the structure
print(harass_train.head())
print(harass_test.head())
print(emotion_train.head())
print(emotion_test.head(5))


# List of harassment categories
harass_labels = ['toxic', 'severe_toxic','obscene', 'threat', 'insult', 'identity_hate']

# Create a new column 'harassment' that is 1 if any label is 1, else 0
harass_train['harassment'] = harass_train[harass_labels].max(axis=1)
# harass_test['harassment'] = harass_test[harass_labels].max(axis=1)

# Check
print(harass_train[['comment_text', 'harassment']].head(666))


# Split comment_text by the last ';' to separate text and emotion
def split_comment_emotion(x):
    parts = x.rsplit(';', 1)
    return pd.Series({'comment': parts[0], 'emotion': parts[1] if len(parts) > 1 else None})

emotion_train[['comment', 'emotion']] = emotion_train['i didnt feel humiliated;sadness'].apply(split_comment_emotion)
emotion_test[['comment', 'emotion']] = emotion_test['im feeling rather rotten so im not very ambitious right now;sadness'].apply(split_comment_emotion)

# Check
print("\nemotion_train labels")
print(emotion_train[['comment', 'emotion']].head())
print("\nemotion test labels")
print(emotion_test[['comment', 'emotion']].head())


import matplotlib.pyplot as plt
import seaborn as sns

#plot the bar chart for harass_train dataset
harass_labels = ['toxic', 'severe_toxic','obscene', 'threat', 'insult', 'identity_hate']
label_counts = harass_train[harass_labels].sum().sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=label_counts.index, y=label_counts.values, palette="coolwarm")
plt.title('Distribution of Toxic Labels')
plt.ylabel('Number of Comments')
plt.xlabel('Label')
plt.show()

#plot the bar chart for emotion_train dataset
plt.figure(figsize=(10,6))
sns.countplot(data=emotion_train, x='emotion', order=emotion_train['emotion'].value_counts().index, palette="Spectral")
plt.title('Distribution of Emotion Classes')
plt.xlabel('Emotion')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.show()


def improved_preprocess(text, do_spelling_correction=False):
    text = re.sub(r'http\S+|www.\S+', '', text)         # Remove URLs
    text = re.sub(r'<.*?>', '', text)                   # Remove HTML
    text = re.sub(r'@\w+', '', text)                    # Remove @handles
    text = re.sub(r'#\w+', '', text)                    # Remove hashtags
    text = emoji.demojize(text)                         # Convert emojis to text
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)          # Normalize repeated chars
    text = re.sub(r'[^\x00-\x7F]+', '', text)           # Remove non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()            # Remove extra spaces
    text = text.lower()                                 #Convert to lowercase for using bert-base-uncased

    if do_spelling_correction:
        text = str(TextBlob(text).correct())
    return text

#Apply this to your text columns:
#For toxic comments:

harass_train['clean_text'] = harass_train['comment_text'].astype(str).apply(improved_preprocess)
harass_test['clean_text'] = harass_test['comment_text'].astype(str).apply(improved_preprocess)

#For emotions:
emotion_train['clean_text'] = emotion_train['comment'].astype(str).apply(improved_preprocess)
emotion_test['clean_text'] = emotion_test['comment'].astype(str).apply(improved_preprocess)
#check a few examples
print(harass_train[['comment_text', 'clean_text']].head())

print("\nToxic comments train DataFrame shape:", harass_train.shape)


from transformers import BertTokenizer
import torch

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 100  # BERT can handle up to 512, but 128 is faster for student projects

def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'  # 'pt' for PyTorch, 'tf' for TensorFlow
    )
# 1. Harassment Train
harass_train_encodings = tokenize_texts(harass_train['clean_text'][:100000])
harass_train_labels = torch.tensor(harass_train['harassment'].values)

# 2. Harassment Test
harass_test_encodings = tokenize_texts(harass_test['clean_text'][100000])
# harass_test_labels = torch.tensor(harass_test['harassment'].values)

# 3. Emotion Train
emotion_train_encodings = tokenize_texts(emotion_train['clean_text'])
# Convert emotion labels to numeric (e.g., using label encoding)
from sklearn.preprocessing import LabelEncoder
emotion_le = LabelEncoder()
emotion_train_labels = torch.tensor(emotion_le.fit_transform(emotion_train['emotion'].values))

# 4. Emotion Test
emotion_test_encodings = tokenize_texts(emotion_test['clean_text'])
emotion_test_labels = torch.tensor(emotion_le.transform(emotion_test['emotion'].values))

# Inspect shapes and types
print("Harass train input_ids shape:", harass_train_encodings['input_ids'].shape)
print("Harass train labels shape:", harass_train_labels.shape)
print("Emotion train input_ids shape:", emotion_train_encodings['input_ids'].shape)
print("Emotion train labels shape:", emotion_train_labels.shape)


from sklearn.model_selection import train_test_split

# Set your desired sample size
sample_size = 25000   # You can adjust as needed

# Stratified sampling to reduce dataset size and split into train/val
harass_sampled, _ = train_test_split(
    harass_train,
    train_size=sample_size,   # Or fraction, e.g., 0.2 for 20%
    stratify=harass_train['harassment'],
    random_state=42
)

# Now split sampled data into train and validation sets
train_df, val_df = train_test_split(
    harass_sampled,
    test_size=0.2,   # 20% for validation
    stratify=harass_sampled['harassment'],
    random_state=42
)

# Check class distribution
print(train_df['harassment'].value_counts())
print(val_df['harassment'].value_counts())


from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 100  # BERT can handle up to 512, but 128 is faster for student projects

def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'  # 'pt' for PyTorch, 'tf' for TensorFlow
    )

# For harassment dataset (train and validation)
train_encodings = tokenize_texts(train_df['clean_text'])
val_encodings = tokenize_texts(val_df['clean_text'])

train_labels = torch.tensor(train_df['harassment'].values)
val_labels = torch.tensor(val_df['harassment'].values)# Inspect output shapes
print("Train input_ids shape:", train_encodings['input_ids'].shape)
print("Validation input_ids shape:", val_encodings['input_ids'].shape)
print("Train labels shape:", train_labels.shape)
print("Validation labels shape:", val_labels.shape)


import torch
from torch.utils.data import Dataset

class BertTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

from torch.utils.data import DataLoader

# Create Dataset objects
train_dataset = BertTextDataset(train_encodings, train_labels)
val_dataset = BertTextDataset(val_encodings, val_labels)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)





from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Define Dataset class
class BertTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# 2. Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



# 3. Calculate class weights
import numpy as np
class_counts = np.bincount(train_labels.numpy())
class_weights = torch.tensor([sum(class_counts)/c for c in class_counts], dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# 4. DataLoader setup
train_dataset = BertTextDataset(train_encodings, train_labels)
val_dataset = BertTextDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 5. Optimizer and training hyperparameters
learning_rate = 3e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_epochs = 3 
total_steps = len(train_loader) * num_epochs


scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

best_f1 = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_preds = []
    train_labels_list = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits, labels)
        logits = outputs.logits
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        train_preds.extend(preds)
        train_labels_list.extend(labels.detach().cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    train_acc = accuracy_score(train_labels_list, train_preds)
    train_f1 = f1_score(train_labels_list, train_preds, average='weighted')
    print(f"Epoch {epoch+1} Train loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            val_preds.extend(preds)
            val_labels_list.extend(labels.detach().cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_labels_list, val_preds)
    val_f1 = f1_score(val_labels_list, val_preds, average='weighted')
    print(f"Epoch {epoch+1} Val loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")   
    # Save best model        
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_bert_harassment_model.pt")
        print("Best model saved.")

print("Training completed.")



model.save_pretrained("nis_bert_model")


from transformers import BertForSequenceClassification
import torch

# Load the saved model
loaded_model = BertForSequenceClassification.from_pretrained("nis_bert_model")

# Move the loaded model to the appropriate device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model.to(device)

print("Model loaded successfully!")


from transformers import BertTokenizer
import torch

# Select a single comment from harass_test
test_comment = harass_test['clean_text'].iloc[0] # You can change the index to test different comments
print(f"Original comment: {harass_test[harass_test.columns[1]].iloc[0]}") # Print original for context
print(f"Cleaned comment: {test_comment}")

# Preprocess and tokenize the comment
# Reuse the improved_preprocess function defined earlier

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 100

def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'  # 'pt' for PyTorch, 'tf' for TensorFlow
    )


test_encoding = tokenize_texts([test_comment])

# Make a prediction
loaded_model.eval() # Set the model to evaluation mode
with torch.no_grad():
    input_ids = test_encoding['input_ids'].to(device)
    attention_mask = test_encoding['attention_mask'].to(device)
    outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

# Interpret the prediction
# Assuming 0 is non-harassment and 1 is harassment
predicted_label = "Harassment" if prediction == 1 else "Non-harassment"

print(f"\nPredicted label: {predicted_label}")


from torch.utils.data import DataLoader, Dataset
harass_test_encodings = tokenize_texts(harass_test['clean_text'])
# Assuming harass_test_encodings was created earlier
# If not, you would need to tokenize the harass_test['clean_text'] column
# harass_test_encodings = tokenize_texts(harass_test['clean_text'])

class InferenceDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

# Assuming harass_test_encodings is available from previous steps
# If not, recreate it using the tokenize_texts function
#harass_test_encodings = tokenize_texts(harass_test['clean_text'])


test_dataset = InferenceDataset(harass_test_encodings)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Make predictions on the test set
loaded_model.eval() # Set the model to evaluation mode
predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        predictions.extend(preds)

# Add predictions to the harass_test DataFrame
harass_test['predicted_label'] = predictions

# Display the first few rows with predictions
print("Test data with predicted labels:")
display(harass_test.head())


def predict_harassment(text, model, tokenizer, max_len, device):
    """
    Predicts if a given text is harassment or not.

    Args:
        text (str): The input text to classify.
        model (torch.nn.Module): The trained BERT model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for the model.
        max_len (int): The maximum sequence length for tokenization.
        device (torch.device): The device to run the model on (cuda or cpu).

    Returns:
        str: The predicted label ('Harassment' or 'Non-harassment').
    """
    # Preprocess the text using the improved_preprocess function
    cleaned_text = improved_preprocess(text) # Assuming improved_preprocess is defined in an earlier cell

    # Tokenize the cleaned text
    encoding = tokenizer(
        cleaned_text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move tensors to the correct device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Make prediction
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    # Interpret the prediction
    predicted_label = "Harassment" if prediction == 1 else "Non-harassment"

    return predicted_label

# Example usage:
custom_text = "This is a test comment. It is not offensive." # Replace with your custom text

# Make sure the loaded_model, tokenizer, MAX_LEN, and device variables are available from previous cells
prediction = predict_harassment(custom_text, loaded_model, tokenizer, MAX_LEN, device)
print(f"Custom text: '{custom_text}'")
print(f"Predicted label: {prediction}")

# Example with potentially harassing text
custom_text_2 = "fuckkkkkkkkkkkkkkkkkkk u."
prediction_2 = predict_harassment(custom_text_2, loaded_model, tokenizer, MAX_LEN, device)
print(f"\nCustom text: '{custom_text_2}'")
print(f"Predicted label: {prediction_2}")





"""
Emotion dataset preparation (implements Plan points 1 & 2)

What this script does:
- Loads emotion_train / emotion_test (from in-memory pandas objects if present,
  else tries to read common Kaggle input paths).
- Ensures we have 'comment' and 'emotion' columns (parses lines of "text;emotion" if needed).
- Applies your existing `improved_preprocess` if a 'clean_text' column is missing.
- Label-encodes the emotion labels and saves the mapping to JSON for later inference.
- Optionally performs stratified sampling (to reduce dataset size) and creates a
  stratified train/validation split.
- Tokenizes the resulting train/val text using BERT tokenizer and saves tokenized
  encodings and label tensors to disk for quick reuse.

Usage:
- Run directly in your Kaggle notebook / local env where emotion_train/emotion_test
  dataframes are available (or the files at the paths below).
- Adjust SAMPLE_SIZE (int) if you want to reduce dataset size; set to None to keep all data.
- Adjust TEST_SIZE for validation split fraction, and MAX_LEN/BATCH_SIZE if you want to change tokenization settings.

Outputs (saved in working dir):
- emotion_label_map.json       -> mapping emotion -> integer id
- emotion_train_split.csv      -> sampled/train split (if sampling used)
- emotion_val_split.csv        -> validation split
- emotion_train_encodings.pt   -> tokenized encodings dict for train (torch.save)
- emotion_val_encodings.pt     -> tokenized encodings dict for val (torch.save)
- emotion_train_labels.pt      -> torch tensor of train labels
- emotion_val_labels.pt        -> torch tensor of val labels
"""
import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch

# ---------- CONFIG ----------
# If you want to downsample the emotion dataset for faster experiments,
# set SAMPLE_SIZE to an integer (total rows after sampling). Set to None to keep all.
SAMPLE_SIZE = 100000  # e.g., 5000 or None

# Fraction of sampled data to use for validation
TEST_SIZE = 0.2

# Random state for reproducibility
RANDOM_STATE = 42

# Tokenization params
PRETRAINED_BERT = "bert-base-uncased"
MAX_LEN = 100

# Input paths fallback (only used if emotion_train/emotion_test not already in memory)
FALLBACK_TRAIN_PATH = "/kaggle/input/emotions-dataset-for-nlp/train.txt"
FALLBACK_TEST_PATH = "/kaggle/input/emotions-dataset-for-nlp/test.txt"

# Output directory
OUT_DIR = "emotion_prep_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
# ---------- END CONFIG ----------


def load_emotion_data():
    """
    Attempt to load emotion_train and emotion_test from the current Python globals,
    otherwise try common Kaggle input paths. Returns two dataframes (train, test).
    """
    # If user already has these variables in the notebook, use them directly
    try:
        # Use globals() to check if emotion_train is present
        g = globals()
        if "emotion_train" in g and isinstance(g["emotion_train"], pd.DataFrame):
            train_df = g["emotion_train"]
        else:
            # Try fallback path
            train_df = pd.read_csv(FALLBACK_TRAIN_PATH, sep=None, engine='python', header=None, quoting=3)
        if "emotion_test" in g and isinstance(g["emotion_test"], pd.DataFrame):
            test_df = g["emotion_test"]
        else:
            test_df = pd.read_csv(FALLBACK_TEST_PATH, sep=None, engine='python', header=None, quoting=3)
    except Exception as e:
        raise RuntimeError(f"Could not load emotion data automatically: {e}")

    # Normalize DataFrame format (if headerless read created a single-column df)
    train_df = normalize_emotion_dataframe(train_df, "train")
    test_df = normalize_emotion_dataframe(test_df, "test")
    return train_df, test_df


def normalize_emotion_dataframe(df, tag="train"):
    """
    Ensure df has 'comment' and 'emotion' columns.
    - If df already has those, return as-is.
    - Else assume rows are "text;emotion" or similar; split on last ';' to extract emotion.
    """
    df = df.copy()
    # If already have required columns
    if "comment" in df.columns and "emotion" in df.columns:
        return df[["comment", "emotion"]].reset_index(drop=True)

    # If there's a 'clean_text' and 'emotion' (maybe user preprocessed earlier), handle that
    if "clean_text" in df.columns and "emotion" in df.columns:
        df = df.rename(columns={"clean_text": "comment"})[["comment", "emotion"]].reset_index(drop=True)
        return df

    # If there's an obvious column with semicolons (single-column read), detect and split
    # Combine all string columns into a single series to try parsing
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(text_cols) == 0:
        raise ValueError(f"No text-like columns found in emotion {tag} dataframe.")

    # Create a single series by concatenating columns with a space (if multiple present)
    single_series = df[text_cols].astype(str).agg(" ".join, axis=1)

    # Count rows containing ';' -- assume format "text;emotion"
    semicolon_ratio = single_series.str.contains(";").mean()
    if semicolon_ratio > 0.1:
        # parse by splitting on last ';'
        def split_comment_emotion(x):
            parts = x.rsplit(";", 1)
            if len(parts) == 2:
                return pd.Series({"comment": parts[0].strip(), "emotion": parts[1].strip()})
            else:
                return pd.Series({"comment": x.strip(), "emotion": None})
        parsed = single_series.apply(split_comment_emotion)
        parsed = parsed.dropna(subset=["comment"]).reset_index(drop=True)
        return parsed
    else:
        # Try to locate an 'emotion' like column by matching a small set of known emotion labels
        KNOWN_EMOTIONS = {"joy", "sadness", "anger", "fear", "love", "surprise", "neutral", "disgust", "boredom"}
        # Search columns for values belonging mostly to known emotions
        for col in text_cols:
            unique_vals = set(df[col].dropna().astype(str).str.lower().unique())
            overlap = len(unique_vals & KNOWN_EMOTIONS)
            if overlap >= 1:
                # We assume this column is emotion; take another text-like column as comment
                other_cols = [c for c in text_cols if c != col]
                if len(other_cols) == 0:
                    raise ValueError("Could not find a text column to pair with emotion column.")
                comment_col = other_cols[0]
                parsed = df[[comment_col, col]].rename(columns={comment_col: "comment", col: "emotion"})
                return parsed.reset_index(drop=True)
        # If we reach here, fallback: try to take first two columns as comment,emotion
        if len(text_cols) >= 2:
            parsed = df[[text_cols[0], text_cols[1]]].rename(columns={text_cols[0]: "comment", text_cols[1]: "emotion"})
            return parsed.reset_index(drop=True)
        raise ValueError(f"Unable to normalize emotion dataframe for {tag}; please inspect input.")


def apply_preprocessing_if_needed(df, preprocess_fn):
    """Ensure 'clean_text' exists by applying preprocess_fn if missing."""
    df = df.copy()
    if "clean_text" not in df.columns:
        print("Applying preprocessing to build 'clean_text' column. This may take a while...")
        df["clean_text"] = df["comment"].astype(str).apply(preprocess_fn)
    else:
        print("'clean_text' column already present; skipping preprocessing.")
    return df


def create_label_encoder_and_save(df, out_map_path):
    """Fit a LabelEncoder on df['emotion'] and save mapping to JSON. Returns label_encoder and integer labels."""
    le = LabelEncoder()
    # Clean emotions: strip and lower
    emotions = df['emotion'].astype(str).str.strip()
    integer_labels = le.fit_transform(emotions)
    label_map = {int(label): str(cls) for label, cls in enumerate(le.classes_)}
    # Save mapping as emotion->id and id->emotion
    mapping = {"emotion_to_id": {cls: int(idx) for idx, cls in enumerate(le.classes_)},
               "id_to_emotion": {int(idx): cls for idx, cls in enumerate(le.classes_)}}
    with open(out_map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Saved emotion label mapping to {out_map_path}")
    return le, integer_labels


def stratified_sample_and_split(df, label_col="emotion", sample_size=None, test_size=0.2, random_state=42):
    """
    Optionally sample stratified to sample_size, then split into train/val stratified.
    Returns (train_df, val_df).
    """
    df = df.copy().reset_index(drop=True)
    labels = df[label_col].astype(str)
    if sample_size is not None and sample_size < len(df):
        # stratified sampling to reduce data size
        print(f"Performing stratified sampling to {sample_size} total examples...")
        # Use train_test_split with train_size == sample_size to sample
        sampled_df, _ = train_test_split(df, train_size=sample_size, stratify=labels, random_state=random_state)
    else:
        sampled_df = df
    # Now split into train/val
    print(f"Splitting sampled data into train/val with test_size={test_size} (stratified)...")
    train_df, val_df = train_test_split(sampled_df, test_size=test_size,
                                        stratify=sampled_df[label_col].astype(str),
                                        random_state=random_state)
    # Reset indices
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def tokenize_and_save(tokenizer, df, out_enc_path, out_labels_path, max_len=128):
    """
    Tokenize df['clean_text'] with tokenizer and save encodings and labels tensors to disk.
    Returns encodings dict and labels tensor.
    """
    texts = df["clean_text"].astype(str).tolist()
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    # labels must be provided externally; this function expects df has 'emotion_id' column
    if "emotion_id" not in df.columns:
        raise ValueError("DataFrame must contain 'emotion_id' column before tokenization.")
    labels = torch.tensor(df["emotion_id"].values, dtype=torch.long)
    torch.save(encodings, out_enc_path)
    torch.save(labels, out_labels_path)
    print(f"Saved encodings to {out_enc_path} and labels to {out_labels_path}")
    return encodings, labels


# Example preprocess function placeholder: use your improved_preprocess from the notebook
def improved_preprocess_wrapper(text):
    """
    Placeholder wrapper. If you already have improved_preprocess defined in your notebook,
    this wrapper will call it. Otherwise, this will perform a simple cleaning fallback.
    """
    # Try to call user-defined improved_preprocess from globals
    g = globals()
    if "improved_preprocess" in g and callable(g["improved_preprocess"]):
        return g["improved_preprocess"](text)
    # Simple fallback cleaning if user function doesn't exist
    import re, emoji
    s = str(text)
    s = re.sub(r'http\S+|www\.\S+', '', s)
    s = re.sub(r'<.*?>', '', s)
    s = re.sub(r'@\w+', '', s)
    s = emoji.demojize(s)
    s = re.sub(r'[^ -~]', ' ', s)  # remove non-ascii
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()


def main():
    print("Loading emotion data...")
    train_df, test_df = load_emotion_data()
    print(f"Raw train rows: {len(train_df)}, raw test rows: {len(test_df)}")

    # Ensure required columns and preprocess
    # Normalize emotion values
    train_df["emotion"] = train_df["emotion"].astype(str).str.strip()
    test_df["emotion"] = test_df["emotion"].astype(str).str.strip() if "emotion" in test_df.columns else None

    # Build clean_text using provided preprocessing function if needed
    train_df = apply_preprocessing_if_needed(train_df, improved_preprocess_wrapper)
    test_df = apply_preprocessing_if_needed(test_df, improved_preprocess_wrapper)

    # Create and save label encoder + mapping
    label_map_path = os.path.join(OUT_DIR, "emotion_label_map.json")
    label_encoder, train_integer_labels = create_label_encoder_and_save(train_df, label_map_path)

    # attach emotion_id to train_df
    train_df["emotion_id"] = label_encoder.transform(train_df["emotion"].astype(str))

    # If the test set has 'emotion' values (e.g., if it's a labeled dev/test), transform them too
    if "emotion" in test_df.columns and test_df["emotion"].notnull().any():
        test_df["emotion_id"] = label_encoder.transform(test_df["emotion"].astype(str))
    else:
        test_df["emotion_id"] = None

    # Optionally stratified sample and split
    train_split_df, val_split_df = stratified_sample_and_split(
        train_df,
        label_col="emotion",
        sample_size=SAMPLE_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Save splits to CSV for inspection / reuse
    train_csv = os.path.join(OUT_DIR, "emotion_train_split.csv")
    val_csv = os.path.join(OUT_DIR, "emotion_val_split.csv")
    train_split_df.to_csv(train_csv, index=False)
    val_split_df.to_csv(val_csv, index=False)
    print(f"Saved train/val splits: {train_csv}, {val_csv}")
    print("Class distribution in train split:")
    print(train_split_df["emotion"].value_counts())
    print("Class distribution in val split:")
    print(val_split_df["emotion"].value_counts())

    # Tokenize train and val and save encodings + labels
    print("Initializing tokenizer and tokenizing train/val splits...")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT)

    train_enc_path = os.path.join(OUT_DIR, "emotion_train_encodings.pt")
    val_enc_path = os.path.join(OUT_DIR, "emotion_val_encodings.pt")
    train_labels_path = os.path.join(OUT_DIR, "emotion_train_labels.pt")
    val_labels_path = os.path.join(OUT_DIR, "emotion_val_labels.pt")

    # Attach emotion_id to val_split_df (it should exist in train_df mapping)
    val_split_df["emotion_id"] = label_encoder.transform(val_split_df["emotion"].astype(str))

    tokenize_and_save(tokenizer, train_split_df, train_enc_path, train_labels_path, max_len=MAX_LEN)
    tokenize_and_save(tokenizer, val_split_df, val_enc_path, val_labels_path, max_len=MAX_LEN)

    print("Emotion dataset preparation complete.")
    print("Saved outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()


# Quick one-off: re-tokenize using CSV splits and save plain dict encodings (no BatchEncoding)
import os
import torch
import pandas as pd
from transformers import BertTokenizer

OUT_DIR = "emotion_prep_outputs"
PRETRAINED = "bert-base-uncased"
MAX_LEN = 128

tokenizer = BertTokenizer.from_pretrained(PRETRAINED)

# paths to CSV splits created by emotion_prep.py
train_csv = os.path.join(OUT_DIR, "emotion_train_split.csv")
val_csv   = os.path.join(OUT_DIR, "emotion_val_split.csv")

train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)

# Use clean_text column if present, otherwise comment
train_texts = train_df.get('clean_text', train_df.get('comment')).astype(str).tolist()
val_texts   = val_df.get('clean_text', val_df.get('comment')).astype(str).tolist()

print(f"Tokenizing {len(train_texts)} train and {len(val_texts)} val samples...")

train_enc = tokenizer(train_texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
val_enc   = tokenizer(val_texts,   truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")

# Save the .data dict (plain dict of tensors) rather than the BatchEncoding object
torch.save(train_enc.data, os.path.join(OUT_DIR, "emotion_train_encodings.pt"))
torch.save(val_enc.data,   os.path.join(OUT_DIR, "emotion_val_encodings.pt"))

# Save labels as tensors (use existing emotion_id column)
train_labels = torch.tensor(train_df['emotion_id'].values, dtype=torch.long)
val_labels   = torch.tensor(val_df['emotion_id'].values,   dtype=torch.long)
torch.save(train_labels, os.path.join(OUT_DIR, "emotion_train_labels.pt"))
torch.save(val_labels,   os.path.join(OUT_DIR, "emotion_val_labels.pt"))

print("Re-saved encodings as plain dicts and labels. You can now re-run baseline_emotion_train.py")


#!/usr/bin/env python3
"""
Baseline BERT emotion classifier training script.

Usage:
    python baseline_emotion_train.py

Requirements:
    pip install transformers torch scikit-learn tqdm

Expecting files created by emotion_prep.py in folder emotion_prep_outputs:
    - emotion_train_encodings.pt
    - emotion_val_encodings.pt
    - emotion_train_labels.pt
    - emotion_val_labels.pt
    - emotion_label_map.json

This script will:
  - load encodings and labels
  - create datasets and dataloaders
  - compute class weights and use CrossEntropyLoss(weight=...)
  - fine-tune BertForSequenceClassification
  - evaluate on validation set each epoch (accuracy, macro-F1)
  - save best model/tokenizer and metadata to out_dir (saved_models/emotion_baseline)
"""

import os
import json
import time
import random
from pathlib import Path

from torch.optim import AdamW
import torch
from torch.optim import AdamW
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

# ---------- Config ----------
OUT_PREP_DIR = "emotion_prep_outputs"
SAVE_DIR = "saved_models/emotion_baseline"
PRETRAINED = "bert-base-uncased"

BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 42
# ----------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EncodedDataset(Dataset):
    def __init__(self, encodings, labels):
        # encodings: dict with tensors (input_ids, attention_mask, token_type_ids optional)
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return self.labels.size(0)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def load_prep_outputs(prep_dir):
    # Paths
    train_enc_path = os.path.join(prep_dir, "emotion_train_encodings.pt")
    val_enc_path = os.path.join(prep_dir, "emotion_val_encodings.pt")
    train_labels_path = os.path.join(prep_dir, "emotion_train_labels.pt")
    val_labels_path = os.path.join(prep_dir, "emotion_val_labels.pt")
    label_map_path = os.path.join(prep_dir, "emotion_label_map.json")

    # Basic checks
    for p in [train_enc_path, val_enc_path, train_labels_path, val_labels_path, label_map_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}. Run emotion_prep.py first.")

    train_enc = torch.load(train_enc_path)
    val_enc = torch.load(val_enc_path)
    train_labels = torch.load(train_labels_path)
    val_labels = torch.load(val_labels_path)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    return train_enc, val_enc, train_labels, val_labels, label_map

def compute_class_weights(labels_tensor):
    # labels_tensor: torch tensor of shape (N,)
    labels_np = labels_tensor.cpu().numpy()
    counts = np.bincount(labels_np)
    # avoid division by zero
    counts = np.where(counts == 0, 1, counts)
    total = counts.sum()
    weights = total / (len(counts) * counts)  # inverse frequency scaled by num classes
    weights = torch.tensor(weights, dtype=torch.float)
    return weights

def evaluate(model, dataloader, device):
    model.eval()
    preds_all = []
    labels_all = []
    losses = []
    loss_fn = nn.CrossEntropyLoss()  # used only for reporting; in training we use weighted loss
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())
    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(labels_all, preds_all)
    macro_f1 = f1_score(labels_all, preds_all, average='macro')
    return {'loss': avg_loss, 'accuracy': acc, 'macro_f1': macro_f1, 'y_true': labels_all, 'y_pred': preds_all}

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load prepared data
    print("Loading preprocessed encodings and labels...")
    train_enc, val_enc, train_labels, val_labels, label_map = load_prep_outputs(OUT_PREP_DIR)
    num_labels = len(label_map['id_to_emotion'].keys())
    print(f"Number of emotion classes: {num_labels}")

    # Build datasets and dataloaders
    train_dataset = EncodedDataset(train_enc, train_labels)
    val_dataset = EncodedDataset(val_enc, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED)
    model = BertForSequenceClassification.from_pretrained(PRETRAINED, num_labels=num_labels)
    model.to(device)

    # Class weights
    class_weights = compute_class_weights(train_labels).to(device)
    print("Class weights:", class_weights.cpu().numpy())

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    best_val_f1 = -1.0
    best_epoch = -1
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        step = 0
        start_time = time.time()
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            step += 1

        train_acc = accuracy_score(all_labels, all_preds)
        train_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        avg_epoch_loss = epoch_loss / max(1, step)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch} done in {epoch_time:.1f}s - Train loss: {avg_epoch_loss:.4f} | Train acc: {train_acc:.4f} | Train macro-F1: {train_macro_f1:.4f}")

        # Validation
        val_res = evaluate(model, val_loader, device)
        print(f"Validation - loss: {val_res['loss']:.4f} | acc: {val_res['accuracy']:.4f} | macro-F1: {val_res['macro_f1']:.4f}")

        # Save best
        if val_res['macro_f1'] > best_val_f1:
            best_val_f1 = val_res['macro_f1']
            best_epoch = epoch
            # Save model & tokenizer in HuggingFace format
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            # Save metadata
            meta = {
                'best_val_macro_f1': float(best_val_f1),
                'best_epoch': int(best_epoch),
                'num_labels': int(num_labels),
                'train_batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'num_epochs': NUM_EPOCHS,
            }
            with open(os.path.join(SAVE_DIR, "train_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"Saved best model (epoch {best_epoch}) to {SAVE_DIR}")

  # Final evaluation summary
    print("Training complete.")
    print(f"Best val macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # --- Load best model for final evaluation ---
    print(f"Loading best model from {SAVE_DIR} for final report and matrix...")
    model = BertForSequenceClassification.from_pretrained(SAVE_DIR)
    model.to(device)
    
    # Run evaluation on the *best* model
    val_res = evaluate(model, val_loader, device) 
    print(f"Best Model Validation - loss: {val_res['loss']:.4f} | acc: {val_res['accuracy']:.4f} | macro-F1: {val_res['macro_f1']:.4f}")

    # --- Generate Classification Report ---
    print("Generating classification report for best model...")
    class_names = [label_map['id_to_emotion'][str(i)] for i in range(num_labels)]
    val_report = classification_report(val_res['y_true'], val_res['y_pred'], target_names=class_names, zero_division=0)
    print(val_report)
    
    # Save final report
    with open(os.path.join(SAVE_DIR, "val_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(val_report)
    print("Saved validation report.")
    
    # --- Generate Confusion Matrix Heatmap ---
    print("Generating confusion matrix heatmap for best model...")
    
    # 1. Get the true labels and predictions
    y_true = val_res['y_true']
    y_pred = val_res['y_pred']
    
    # 2. Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 3. Plot using Seaborn
    plt.figure(figsize=(10, 8)) # Adjust size as needed for your number of labels
    sns.heatmap(
        cm, 
        annot=True,     # Show numbers in cells
        fmt='d',        # Format as integer
        cmap='Blues',   # Color map
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - Baseline BERT(Epoch {best_epoch})')
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    
    # 4. Save the plot
    plot_path = os.path.join(SAVE_DIR, "val_confusion_matrix.png")
    plt.savefig(plot_path, bbox_inches='tight') # bbox_inches='tight' ensures labels fit
    print(f"Saved confusion matrix heatmap to {plot_path}")
    
    # Optionally display the plot if running in an interactive environment
    # plt.show()

if __name__ == "__main__":
    main()


# """
# Hybrid BERT -> BiRNN(+attention) model + DEA-inspired hyperparameter search.

# Usage:
#     - Place this file in your notebook/workdir and run from a cell or as a script.
#     - It expects emotion_prep_outputs/ to contain:
#         emotion_train_encodings.pt  (a plain dict of tensors: input_ids, attention_mask, ...)
#         emotion_val_encodings.pt
#         emotion_train_labels.pt     (torch tensor)
#         emotion_val_labels.pt
#         emotion_label_map.json

# Core functions/classes:
#     - BertBiRNNWithAttention: the hybrid model
#     - build_dataloaders_from_prep: load encodings/labels and build DataLoaders (supports subset for fast eval)
#     - train_and_evaluate_config: train model for given config and return val macro-F1 (used as fitness)
#     - dea_search: a simplified DEA-inspired population search that returns best config

# Notes:
#     - During the DEA inner-loop evaluation we freeze BERT by default (fast searches). Final training can unfreeze.
#     - This implementation uses a "DEA-inspired" update process: population-based perturbation with exploitation of best solutions.
#       It is intentionally lightweight for quick inner-loop evaluations on limited compute.
# """
# import os
# import json
# import copy
# import time
# import random
# from typing import Dict, Any, Tuple, List

# import numpy as np
# import torch
# from torch.optim import AdamW
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from transformers import BertModel, BertTokenizer, BertConfig, get_linear_schedule_with_warmup
# from sklearn.metrics import f1_score, accuracy_score
# from tqdm import tqdm

# # -------------------- Config defaults --------------------
# PREP_DIR = "emotion_prep_outputs"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
# # -------------------- End defaults ------------------------

# # -------------------- Model -------------------------------
# class AttentionPooling(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.attn = nn.Linear(input_dim, 1)

#     def forward(self, x, mask=None):
#         # x: (batch, seq_len, hidden)
#         scores = self.attn(x).squeeze(-1)   # (batch, seq_len)
#         if mask is not None:
#             # mask: 1 for tokens to keep, 0 to ignore
#             scores = scores.masked_fill(mask == 0, -1e9)
#         weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
#         context = torch.sum(weights * x, dim=1)  # (batch, hidden)
#         return context, weights

# class BertBiRNNWithAttention(nn.Module):
#     def __init__(self,
#                  pretrained_model_name: str,
#                  rnn_type: str = "gru",      # 'lstm' or 'gru'
#                  hidden_size: int = 128,
#                  num_layers: int = 1,
#                  bidirectional: bool = True,
#                  dropout: float = 0.2,
#                  use_attention: bool = True,
#                  freeze_bert: bool = True,
#                  num_labels: int = 6):
#         super().__init__()
#         assert rnn_type in ("lstm", "gru")
#         self.bert = BertModel.from_pretrained(pretrained_model_name)
#         bert_hidden = self.bert.config.hidden_size

#         self.freeze_bert = freeze_bert
#         if freeze_bert:
#             for param in self.bert.parameters():
#                 param.requires_grad = False

#         self.use_attention = use_attention
#         rnn_input_size = bert_hidden
#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1

#         if rnn_type == "lstm":
#             self.rnn = nn.LSTM(input_size=rnn_input_size,
#                                hidden_size=hidden_size,
#                                num_layers=num_layers,
#                                batch_first=True,
#                                dropout=dropout if num_layers > 1 else 0.0,
#                                bidirectional=bidirectional)
#         else:
#             self.rnn = nn.GRU(input_size=rnn_input_size,
#                               hidden_size=hidden_size,
#                               num_layers=num_layers,
#                               batch_first=True,
#                               dropout=dropout if num_layers > 1 else 0.0,
#                               bidirectional=bidirectional)

#         rnn_out_size = hidden_size * self.num_directions
#         if use_attention:
#             self.attention = AttentionPooling(rnn_out_size)
#             clf_in = rnn_out_size
#         else:
#             # use mean pooling over time
#             clf_in = rnn_out_size

#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(clf_in, clf_in // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(clf_in // 2, num_labels)
#         )

#     def forward(self, input_ids, attention_mask, token_type_ids=None):
#         # BERT forward
#         bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         # last_hidden_state: (batch, seq_len, bert_hidden)
#         seq_emb = bert_outputs.last_hidden_state

#         # Pass through RNN
#         rnn_out, _ = self.rnn(seq_emb)  # (batch, seq_len, hidden * directions)

#         if self.use_attention:
#             pooled, attn_weights = self.attention(rnn_out, mask=attention_mask)
#         else:
#             # masked mean pooling over seq_len
#             mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
#             rnn_out = rnn_out * mask
#             sum_repr = rnn_out.sum(dim=1)
#             denom = mask.sum(dim=1).clamp(min=1e-9)
#             pooled = sum_repr / denom

#             attn_weights = None

#         logits = self.classifier(pooled)
#         return logits, attn_weights
# # ------------------ End Model -------------------------------

# # ------------------ Data / utils ----------------------------
# def load_encodings_and_labels(prep_dir=PREP_DIR):
#     """
#     Expect files saved by emotion_prep/recreate_encodings_quickfix:
#       - emotion_train_encodings.pt (plain dict of tensors)
#       - emotion_val_encodings.pt
#       - emotion_train_labels.pt
#       - emotion_val_labels.pt
#       - emotion_label_map.json
#     Returns dicts and tensors (no device moving).
#     """
#     train_enc_path = os.path.join(prep_dir, "emotion_train_encodings.pt")
#     val_enc_path = os.path.join(prep_dir, "emotion_val_encodings.pt")
#     train_labels_path = os.path.join(prep_dir, "emotion_train_labels.pt")
#     val_labels_path = os.path.join(prep_dir, "emotion_val_labels.pt")
#     label_map_path = os.path.join(prep_dir, "emotion_label_map.json")

#     for p in [train_enc_path, val_enc_path, train_labels_path, val_labels_path, label_map_path]:
#         if not os.path.exists(p):
#             raise FileNotFoundError(f"Required prep file missing: {p}")

#     train_enc = torch.load(train_enc_path)  # should be dict of tensors
#     val_enc = torch.load(val_enc_path)
#     train_labels = torch.load(train_labels_path)
#     val_labels = torch.load(val_labels_path)
#     with open(label_map_path, "r", encoding="utf-8") as f:
#         label_map = json.load(f)

#     return train_enc, val_enc, train_labels, val_labels, label_map

# def build_dataloader_from_encodings(encodings: dict, labels: torch.Tensor, batch_size: int = 16, shuffle: bool = False, subset: int = None):
#     """
#     encodings: dict with keys input_ids, attention_mask, optionally token_type_ids (all torch tensors)
#     labels: torch tensor
#     subset: optional int to take only first subset samples (for fast inner-loop evaluation)
#     """
#     # Ensure tensors aligned
#     assert encodings["input_ids"].size(0) == labels.size(0)
#     n = labels.size(0)
#     if subset is not None and subset < n:
#         n = subset
#         # slice tensors
#         enc_dict = {k: v[:n] for k, v in encodings.items()}
#         labels_sub = labels[:n]
#     else:
#         enc_dict = encodings
#         labels_sub = labels

#     tensors = [enc_dict["input_ids"], enc_dict["attention_mask"]]
#     key_order = ["input_ids", "attention_mask"]
#     if "token_type_ids" in enc_dict:
#         tensors.append(enc_dict["token_type_ids"])
#         key_order.append("token_type_ids")
#     tensors.append(labels_sub)
#     ds = TensorDataset(*tensors)
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
#     return loader, key_order

# def compute_class_weights_from_tensor(labels_tensor: torch.Tensor):
#     labels_np = labels_tensor.cpu().numpy()
#     counts = np.bincount(labels_np)
#     counts = np.where(counts == 0, 1, counts)
#     total = counts.sum()
#     weights = total / (len(counts) * counts)  # inverse relative freq scaled
#     return torch.tensor(weights, dtype=torch.float).to(DEVICE)
# # ------------------ End Data / utils -------------------------

# # ------------------ Training & Eval -------------------------
# def train_for_epochs(model: nn.Module,
#                      train_loader: DataLoader,
#                      val_loader: DataLoader,
#                      epochs: int,
#                      lr_bert: float,
#                      lr_head: float,
#                      weight_decay: float,
#                      class_weights: torch.Tensor,
#                      unfreeze_bert_after_epoch: int = None,
#                      max_grad_norm: float = 1.0,
#                      scheduler_warmup_steps: int = 0):
#     """
#     Train model for a small number of epochs and return validation metrics.
#     unfreeze_bert_after_epoch: if set to e.g. 1, unfreeze BERT after that epoch (useful for progressive unfreezing).
#     """
#     model.to(DEVICE)
#     # Prepare parameter groups
#     bert_params = [p for n, p in model.named_parameters() if n.startswith("bert") and p.requires_grad]
#     head_params = [p for n, p in model.named_parameters() if not n.startswith("bert") and p.requires_grad]
#     optim_groups = []
#     if len(bert_params) > 0:
#         optim_groups.append({"params": bert_params, "lr": lr_bert})
#     if len(head_params) > 0:
#         optim_groups.append({"params": head_params, "lr": lr_head})

#     optimizer = AdamW(optim_groups, lr=lr_head, weight_decay=weight_decay)  # lr for groups already set
#     total_steps = max(1, len(train_loader) * epochs)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=scheduler_warmup_steps, num_training_steps=total_steps)

#     loss_fn = nn.CrossEntropyLoss(weight=class_weights)

#     best_val_f1 = -1.0
#     best_val_metrics = None

#     for epoch in range(1, epochs + 1):
#         model.train()
#         running_loss = 0.0
#         all_preds = []
#         all_labels = []
#         for batch in train_loader:
#             # unpack according to dataset order: input_ids, attention_mask, (token_type_ids?), labels
#             tensors = [t.to(DEVICE) for t in batch]
#             if len(tensors) == 4:
#                 input_ids, attention_mask, token_type_ids, labels = tensors
#                 token_type_ids = token_type_ids
#             else:
#                 input_ids, attention_mask, labels = tensors
#                 token_type_ids = None

#             optimizer.zero_grad()
#             logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#             loss = loss_fn(logits, labels)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#             optimizer.step()
#             scheduler.step()

#             running_loss += loss.item()
#             preds = torch.argmax(logits, dim=1)
#             all_preds.extend(preds.detach().cpu().numpy())
#             all_labels.extend(labels.detach().cpu().numpy())

#         # Optionally unfreeze bert
#         if unfreeze_bert_after_epoch is not None and epoch == unfreeze_bert_after_epoch:
#             for p in model.bert.parameters():
#                 p.requires_grad = True

#         train_f1 = f1_score(all_labels, all_preds, average="macro")
#         # validate
#         val_metrics = evaluate_model(model, val_loader)
#         val_f1 = val_metrics["macro_f1"]
#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             best_val_metrics = val_metrics

#     return best_val_f1, best_val_metrics

# def evaluate_model(model: nn.Module, dataloader: DataLoader):
#     model.eval()
#     preds_all = []
#     labels_all = []
#     with torch.no_grad():
#         for batch in dataloader:
#             tensors = [t.to(DEVICE) for t in batch]
#             if len(tensors) == 4:
#                 input_ids, attention_mask, token_type_ids, labels = tensors
#             else:
#                 input_ids, attention_mask, labels = tensors
#                 token_type_ids = None
#             logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#             preds = torch.argmax(logits, dim=1)
#             preds_all.extend(preds.detach().cpu().numpy())
#             labels_all.extend(labels.detach().cpu().numpy())
#     acc = accuracy_score(labels_all, preds_all)
#     macro_f1 = f1_score(labels_all, preds_all, average="macro")
#     return {"accuracy": acc, "macro_f1": macro_f1, "y_true": labels_all, "y_pred": preds_all}
# # ------------------ End Training & Eval ----------------------

# # ------------------ Config-driven train function -------------
# def train_and_evaluate_config(config: Dict[str, Any],
#                               prep_dir: str = PREP_DIR,
#                               inner_subset: int = 2000,
#                               inner_epochs: int = 2):
#     """
#     Train a model with hyperparameters in config for a quick inner-loop evaluation.
#     Returns validation macro-F1 (fitness) and the validation metrics dict.

#     Expected config keys:
#       - rnn_type: 'lstm'|'gru'
#       - hidden_size: int
#       - num_layers: int
#       - bidirectional: bool
#       - dropout: float
#       - use_attention: bool
#       - freeze_bert: bool
#       - lr_bert: float
#       - lr_head: float
#       - batch_size: int
#       - weight_decay: float
#       - unfreeze_bert_after_epoch: int or None
#     """
#     # load data
#     train_enc, val_enc, train_labels, val_labels, label_map = load_encodings_and_labels(prep_dir)
#     num_labels = len(label_map["id_to_emotion"].keys())

#     # build small dataloaders for fast eval
#     train_loader, _ = build_dataloader_from_encodings(train_enc, train_labels, batch_size=config.get("batch_size", 16), shuffle=True, subset=inner_subset)
#     val_loader, _ = build_dataloader_from_encodings(val_enc, val_labels, batch_size=config.get("batch_size", 32), shuffle=False, subset=inner_subset//5)

#     # class weights from training subset
#     # create an aggregated labels tensor used in loader subset (we can compute from sliced labels)
#     train_labels_subset = train_labels[:inner_subset] if inner_subset < train_labels.size(0) else train_labels
#     class_weights = compute_class_weights_from_tensor(train_labels_subset)

#     # build model
#     model = BertBiRNNWithAttention(pretrained_model_name=config.get("pretrained_model", "bert-base-uncased"),
#                                    rnn_type=config.get("rnn_type", "lstm"),
#                                    hidden_size=config.get("hidden_size", 128),
#                                    num_layers=config.get("num_layers", 1),
#                                    bidirectional=config.get("bidirectional", True),
#                                    dropout=config.get("dropout", 0.2),
#                                    use_attention=config.get("use_attention", True),
#                                    freeze_bert=config.get("freeze_bert", True),
#                                    num_labels=num_labels)

#     # Train for few epochs (inner loop)
#     best_val_f1, best_metrics = train_for_epochs(model=model,
#                                                  train_loader=train_loader,
#                                                  val_loader=val_loader,
#                                                  epochs=inner_epochs,
#                                                  lr_bert=config.get("lr_bert", 0.0),
#                                                  lr_head=config.get("lr_head", 1e-3),
#                                                  weight_decay=config.get("weight_decay", 0.01),
#                                                  class_weights=class_weights,
#                                                  unfreeze_bert_after_epoch=config.get("unfreeze_bert_after_epoch", None),
#                                                  max_grad_norm=config.get("max_grad_norm", 1.0),
#                                                  scheduler_warmup_steps=0)
#     return best_val_f1, best_metrics
# # ------------------ End config-driven train ------------------

# # ------------------ Simplified DEA-inspired search -------------
# def random_config_from_space(space: Dict[str, List[Any]]) -> Dict[str, Any]:
#     cfg = {}
#     for k, v in space.items():
#         if isinstance(v, list):
#             cfg[k] = random.choice(v)
#         elif isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
#             # numeric range: uniform sample
#             low, high = v
#             if isinstance(low, int) and isinstance(high, int):
#                 cfg[k] = random.randint(low, high)
#             else:
#                 cfg[k] = float(np.exp(np.random.uniform(np.log(low), np.log(high)))) if low>0 else random.uniform(low, high)
#         else:
#             cfg[k] = v
#     return cfg

# def mutate_config(cfg: Dict[str, Any], space: Dict[str, Any], strength: float = 0.3) -> Dict[str, Any]:
#     """Random small perturbation of config guided by space values."""
#     new = cfg.copy()
#     for k, v in space.items():
#         if random.random() < strength:
#             if isinstance(v, list):
#                 new[k] = random.choice(v)
#             elif isinstance(v, tuple) and len(v) == 2:
#                 low, high = v
#                 if isinstance(low, int) and isinstance(high, int):
#                     step = max(1, int((high - low) * strength))
#                     val = int(np.clip(new[k] + random.randint(-step, step), low, high))
#                     new[k] = val
#                 else:
#                     # multiplicative perturbation for floats
#                     factor = np.exp(np.random.normal(0, strength))
#                     val = float(np.clip(new[k] * factor, low, high))
#                     new[k] = val
#     return new

# def dea_search(space: Dict[str, Any],
#                population_size: int = 8,
#                iterations: int = 12,
#                prep_dir: str = PREP_DIR,
#                inner_subset: int = 2000,
#                inner_epochs: int = 2):
#     """
#     DEA-inspired search over the provided 'space'.
#     Returns best_config, best_score, and the log of evaluated candidates.
#     """
#     # initialize population
#     population = [random_config_from_space(space) for _ in range(population_size)]
#     scores = [None] * population_size
#     best_config = None
#     best_score = -1.0
#     history = []

#     for it in range(iterations):
#         print(f"DEA iter {it+1}/{iterations} - evaluating population of size {len(population)}")
#         for i, cfg in enumerate(population):
#             # evaluate only if not evaluated yet or after mutation
#             print(f" Evaluating candidate {i+1}/{len(population)}: {cfg}")
#             try:
#                 score, metrics = train_and_evaluate_config(cfg, prep_dir=prep_dir, inner_subset=inner_subset, inner_epochs=inner_epochs)
#             except Exception as e:
#                 print("  Candidate failed during training:", e)
#                 score = -1.0
#                 metrics = {}
#             scores[i] = score
#             history.append({"iter": it, "candidate": cfg, "score": float(score), "metrics": metrics})
#             if score > best_score:
#                 best_score = score
#                 best_config = cfg
#                 print("  New best score:", best_score)

#         # DEA-inspired generation:
#         # Keep top half, replace bottom half with mutations of top performers and some randoms
#         ranked = sorted(list(zip(population, scores)), key=lambda x: x[1] if x[1] is not None else -1.0, reverse=True)
#         topk = max(1, population_size // 2)
#         new_population = [copy.deepcopy(r[0]) for r in ranked[:topk]]

#         # generate mutated children from top performers
#         while len(new_population) < population_size:
#             parent = random.choice(new_population)
#             child = mutate_config(parent, space, strength=0.4)
#             new_population.append(child)

#         # Occasionally inject a random config to maintain exploration
#         if random.random() < 0.3:
#             idx_replace = random.randrange(len(new_population))
#             new_population[idx_replace] = random_config_from_space(space)

#         population = new_population
#         scores = [None] * population_size

#     return best_config, best_score, history
# # ------------------ End DEA ----------------------------------

# # ------------------ Example usage helper --------------------
# def example_search_run():
#     # Define the hyperparameter search space
#     space = {
#         "pretrained_model": "bert-base-uncased",
#         "rnn_type": ["lstm", "gru"],
#         "hidden_size": (64, 256),    # integer range
#         "num_layers": (1, 2),
#         "bidirectional": [True, False],
#         "dropout": (0.0, 0.5),      # float range
#         "use_attention": [True, False],
#         "freeze_bert": [True],      # keep True for inner-loop speed; final training can set False
#         "lr_bert": (0.0, 0.0),      # 0.0 means frozen; set >0 in final training separately
#         "lr_head": (1e-4, 1e-2),
#         "batch_size": [16, 32],
#         "weight_decay": (0.0, 0.05),
#         "unfreeze_bert_after_epoch": [None],  # optional progressive unfreeze
#     }

#     best_cfg, best_score, history = dea_search(space,
#                                                population_size=6,
#                                                iterations=6,
#                                                prep_dir=PREP_DIR,
#                                                inner_subset=1500,
#                                                inner_epochs=2)
#     print("DEA finished. Best score:", best_score)
#     print("Best config:", best_cfg)
#     return best_cfg, best_score, history

# # --------------- If executed as script ----------------------
# if __name__ == "__main__":
#     print("Running example DEA-inspired search (fast inner-loop).")
#     best_cfg, best_score, history = example_search_run()
#     out_path = "dea_best_config.json"
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump({"best_config": best_cfg, "best_score": best_score}, f, indent=2)
#     print(f"Saved best config to {out_path}")


#!/usr/bin/env python3
"""
BERT + RNN emotion classifier training script.

This hybrid model:
  1. Uses BERT to get contextualized token embeddings
  2. Passes these through an RNN (LSTM/GRU) to capture emotional sequence
  3. Uses final RNN hidden state for classification

Usage:
    python bert_rnn_emotion_train.py

Requirements:
    Same as baseline + matplotlib seaborn for confusion matrix

Expecting files from emotion_prep.py in emotion_prep_outputs/:
    - emotion_train_encodings.pt
    - emotion_val_encodings.pt
    - emotion_train_labels.pt
    - emotion_val_labels.pt
    - emotion_label_map.json

Outputs:
    - Saves best model to saved_models/emotion_bert_rnn/
    - Includes confusion matrix and classification report
"""

import os
import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Config ----------
OUT_PREP_DIR = "emotion_prep_outputs"
SAVE_DIR = "saved_models/emotion_bert_rnn"
PRETRAINED = "bert-base-uncased"

BATCH_SIZE = 16
NUM_EPOCHS = 6 # Can train a bit longer since model is more complex
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01                              
MAX_GRAD_NORM = 1.0
SEED = 42

# RNN-specific hyperparameters
RNN_TYPE = "LSTM"  # Options: "LSTM" or "GRU"
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 1
RNN_DROPOUT = 0.3
RNN_BIDIRECTIONAL = True  # Use bidirectional RNN for better context
# ----------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EncodedDataset(Dataset):
    """Dataset class for pre-encoded texts"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return self.labels.size(0)
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

class BERTWithRNN(nn.Module):
    """
    Hybrid BERT + RNN model for emotion classification.
    
    Architecture:
        1. BERT outputs token-level embeddings (batch, seq_len, 768)
        2. RNN processes these sequential embeddings
        3. Take final RNN hidden state(s)
        4. Dropout + Linear classifier
    
    Why this works better for emotions:
        - BERT: captures word meaning in context
        - RNN: captures emotional flow across the sentence
        - Together: understand both "what" and "how" emotions are expressed
    """
    def __init__(
        self, 
        bert_model_name, 
        num_labels, 
        rnn_type="LSTM",
        rnn_hidden_size=128,
        rnn_num_layers=1,
        rnn_dropout=0.3,
        bidirectional=True,
        freeze_bert=False
    ):
        super(BERTWithRNN, self).__init__()
        
        # 1. BERT for contextualized embeddings
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        
        # Option to freeze BERT weights (faster training, but less flexible)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("BERT weights frozen.")
        
        # 2. RNN layer
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.bidirectional = bidirectional
        
        rnn_input_size = self.bert_hidden_size
        
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # 3. Classifier head
        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(rnn_output_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT + RNN.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, num_labels)
        """
        # Step 1: Get BERT embeddings for all tokens
        # Shape: (batch_size, seq_len, 768)
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_outputs.last_hidden_state  # All token embeddings
        
        # Step 2: Pass through RNN
        # RNN expects: (batch, seq_len, input_size)
        # Returns: output (batch, seq_len, hidden*directions), hidden states
        rnn_output, hidden = self.rnn(sequence_output)
        
        # Step 3: Extract final hidden state
        if self.rnn_type == "LSTM":
            # hidden is tuple: (h_n, c_n)
            # h_n shape: (num_layers * num_directions, batch, hidden_size)
            h_n = hidden[0]
        else:  # GRU
            # hidden is just h_n
            h_n = hidden
        
        # If bidirectional, concatenate forward and backward final hidden states
        if self.bidirectional:
            # h_n[-2] is last layer forward, h_n[-1] is last layer backward
            final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_hidden = h_n[-1]
        
        # Step 4: Classification
        # Shape: (batch_size, rnn_output_size)
        pooled = self.dropout(final_hidden)
        logits = self.classifier(pooled)
        
        return logits

def load_prep_outputs(prep_dir):
    """Load preprocessed data from emotion_prep.py output"""
    train_enc_path = os.path.join(prep_dir, "emotion_train_encodings.pt")
    val_enc_path = os.path.join(prep_dir, "emotion_val_encodings.pt")
    train_labels_path = os.path.join(prep_dir, "emotion_train_labels.pt")
    val_labels_path = os.path.join(prep_dir, "emotion_val_labels.pt")
    label_map_path = os.path.join(prep_dir, "emotion_label_map.json")

    for p in [train_enc_path, val_enc_path, train_labels_path, val_labels_path, label_map_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}. Run emotion_prep.py first.")

    train_enc = torch.load(train_enc_path)
    val_enc = torch.load(val_enc_path)
    train_labels = torch.load(train_labels_path)
    val_labels = torch.load(val_labels_path)
    
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    return train_enc, val_enc, train_labels, val_labels, label_map

def compute_class_weights(labels_tensor):
    """Compute class weights for imbalanced datasets"""
    labels_np = labels_tensor.cpu().numpy()
    counts = np.bincount(labels_np)
    counts = np.where(counts == 0, 1, counts)  # Avoid division by zero
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = torch.tensor(weights, dtype=torch.float)
    return weights

def evaluate(model, dataloader, device, loss_fn=None):
    """
    Evaluate model on validation set.
    
    Returns:
        dict with loss, accuracy, macro_f1, y_true, y_pred
    """
    model.eval()
    preds_all = []
    labels_all = []
    losses = []
    
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())
    
    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(labels_all, preds_all)
    macro_f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    
    return {
        'loss': avg_loss, 
        'accuracy': acc, 
        'macro_f1': macro_f1, 
        'y_true': labels_all, 
        'y_pred': preds_all
    }

def save_model(model, tokenizer, save_dir, metadata):
    """
    Save BERT+RNN model, tokenizer, and metadata.
    
    Note: We save the entire model state_dict, not just BERT portion.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer (same as BERT)
    tokenizer.save_pretrained(save_dir)
    
    # Save metadata (includes architecture info)
    meta_path = os.path.join(save_dir, "train_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {save_dir}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Generate and save confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title)
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load preprocessed data
    print("\nLoading preprocessed encodings and labels...")
    train_enc, val_enc, train_labels, val_labels, label_map = load_prep_outputs(OUT_PREP_DIR)
    num_labels = len(label_map['id_to_emotion'].keys())
    print(f"Number of emotion classes: {num_labels}")
    print(f"Training samples: {len(train_labels)}")
    print(f"Validation samples: {len(val_labels)}")
    
    # Create datasets and dataloaders
    train_dataset = EncodedDataset(train_enc, train_labels)
    val_dataset = EncodedDataset(val_enc, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print(f"\nInitializing BERT + {RNN_TYPE} model...")
    print(f"  RNN hidden size: {RNN_HIDDEN_SIZE}")
    print(f"  RNN layers: {RNN_NUM_LAYERS}")
    print(f"  Bidirectional: {RNN_BIDIRECTIONAL}")
    
    model = BERTWithRNN(
        bert_model_name=PRETRAINED,
        num_labels=num_labels,
        rnn_type=RNN_TYPE,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_num_layers=RNN_NUM_LAYERS,
        rnn_dropout=RNN_DROPOUT,
        bidirectional=RNN_BIDIRECTIONAL,
        freeze_bert=False  # Set to True if you want faster training
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compute class weights
    class_weights = compute_class_weights(train_labels).to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS, 
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = -1.0
    best_epoch = -1
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    print(f"\n{'='*60}")
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"{'='*60}\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # ========== Training Phase ==========
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        step = 0
        start_time = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            step += 1
        
        # Training metrics
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        avg_train_loss = epoch_loss / max(1, step)
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
        print(f"  Train - Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | Macro-F1: {train_f1:.4f}")
        
        # ========== Validation Phase ==========
        val_res = evaluate(model, val_loader, device, loss_fn)
        
        history['val_loss'].append(val_res['loss'])
        history['val_acc'].append(val_res['accuracy'])
        history['val_f1'].append(val_res['macro_f1'])
        
        print(f"  Val   - Loss: {val_res['loss']:.4f} | Acc: {val_res['accuracy']:.4f} | Macro-F1: {val_res['macro_f1']:.4f}")
        
        # Save best model
        if val_res['macro_f1'] > best_val_f1:
            best_val_f1 = val_res['macro_f1']
            best_epoch = epoch
            
            metadata = {
                'model_type': 'BERT+RNN',
                'rnn_type': RNN_TYPE,
                'rnn_hidden_size': RNN_HIDDEN_SIZE,
                'rnn_num_layers': RNN_NUM_LAYERS,
                'rnn_bidirectional': RNN_BIDIRECTIONAL,
                'best_val_macro_f1': float(best_val_f1),
                'best_val_accuracy': float(val_res['accuracy']),
                'best_epoch': int(best_epoch),
                'num_labels': int(num_labels),
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'num_epochs': NUM_EPOCHS,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            tokenizer = BertTokenizer.from_pretrained(PRETRAINED)
            save_model(model, tokenizer, SAVE_DIR, metadata)
            print(f"  ✓ New best model saved (F1: {best_val_f1:.4f})")
    
    # ========== Final Evaluation ==========
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation Macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"{'='*60}\n")
    
    # Load best model for final evaluation
    print("Loading best model for final evaluation...")
    model_path = os.path.join(SAVE_DIR, "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Final validation metrics
    val_res = evaluate(model, val_loader, device, loss_fn)
    print(f"\nFinal Best Model Performance:")
    print(f"  Loss: {val_res['loss']:.4f}")
    print(f"  Accuracy: {val_res['accuracy']:.4f}")
    print(f"  Macro-F1: {val_res['macro_f1']:.4f}")
    
    # Classification report
    print("\nGenerating detailed classification report...")
    class_names = [label_map['id_to_emotion'][str(i)] for i in range(num_labels)]
    report = classification_report(
        val_res['y_true'], 
        val_res['y_pred'], 
        target_names=class_names, 
        zero_division=0,
        digits=4
    )
    print(report)
    
    # Save report
    report_path = os.path.join(SAVE_DIR, "val_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"BERT + {RNN_TYPE} Emotion Classifier\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Validation Accuracy: {val_res['accuracy']:.4f}\n")
        f.write(f"Validation Macro-F1: {val_res['macro_f1']:.4f}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(SAVE_DIR, "val_confusion_matrix.png")
    plot_confusion_matrix(
        val_res['y_true'],
        val_res['y_pred'],
        class_names,
        cm_path,
        title=f"BERT+{RNN_TYPE} Confusion Matrix (Epoch {best_epoch})"
    )
    
    # Save training history
    history_path = os.path.join(SAVE_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    print("\n✓ All done! Model and results saved to:", SAVE_DIR)

if __name__ == "__main__":
    main()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the path
image_path = 'saved_models/emotion_bert_rnn/val_confusion_matrix.png'

# Read the image
img = mpimg.imread(image_path)

# Plot the image
plt.figure(figsize=(9, 13)) # Optional: adjust size
plt.imshow(img)
plt.title('Confusion Matrix for BERT+LSTM ', fontsize=16) # Add your title here
plt.axis('off')  # Hides the axes
plt.show()


import os
import json
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np

# --- Re-include necessary model class definitions ---
# Make sure these match exactly what you used for training!

class BERTWithRNN(nn.Module):
    """
    Hybrid BERT + RNN model definition (copied from training script).
    Make sure parameters match the saved model's architecture.
    """
    def __init__(
        self,
        bert_model_name,
        num_labels,
        rnn_type="LSTM",
        rnn_hidden_size=128,
        rnn_num_layers=1,
        rnn_dropout=0.3,
        bidirectional=True,
        freeze_bert=False # Not relevant for inference, only architecture
    ):
        super(BERTWithRNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size

        if freeze_bert: # Keep for architecture consistency if needed
             for param in self.bert.parameters():
                 param.requires_grad = False

        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.bidirectional = bidirectional
        rnn_input_size = self.bert_hidden_size

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=rnn_input_size, hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
                bidirectional=bidirectional, batch_first=True
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=rnn_input_size, hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
                bidirectional=bidirectional, batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.dropout = nn.Dropout(0.3) # Use dropout value consistent with training? Check meta if needed.
        self.classifier = nn.Linear(rnn_output_size, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        rnn_output, hidden = self.rnn(sequence_output)

        if self.rnn_type == "LSTM":
            h_n = hidden[0]
        else: # GRU
            h_n = hidden

        if self.bidirectional:
            final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_hidden = h_n[-1]

        pooled = self.dropout(final_hidden)
        logits = self.classifier(pooled)
        return logits

# --- Inference Helper Function ---

def predict_emotion(text: str, model_dir: str = "saved_models/emotion_bert_rnn", prep_dir: str = "emotion_prep_outputs"):
    """
    Loads the saved BERT+RNN model and tokenizer to predict emotion for a single text.

    Args:
        text (str): The input text to classify.
        model_dir (str): Path to the directory containing the saved model files
                         (pytorch_model.bin, tokenizer files, train_meta.json).
        prep_dir (str): Path to the directory containing emotion_label_map.json.


    Returns:
        tuple: (predicted_emotion_label, probabilities_dict)
               - predicted_emotion_label (str): The name of the predicted emotion.
               - probabilities_dict (dict): A dictionary mapping emotion labels to
                                            their predicted probabilities (float).
               Returns (None, None) if loading fails.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load necessary files ---
    model_weights_path = os.path.join(model_dir, "pytorch_model.bin")
    meta_path = os.path.join(model_dir, "train_meta.json")
    label_map_path = os.path.join(prep_dir, "emotion_label_map.json")

    required_files = [model_weights_path, meta_path, label_map_path, model_dir]
    for p in required_files:
        if not os.path.exists(p):
            print(f"Error: Required file or directory not found: {p}")
            return None, None

    try:
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        print("Tokenizer loaded successfully.")

        # Load metadata to get model architecture parameters
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print("Metadata loaded successfully.")

        # Load label map
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        id_to_emotion = label_map.get("id_to_emotion", {})
        if not id_to_emotion:
             raise ValueError("Could not find 'id_to_emotion' in label_map.json")
        num_labels = meta.get('num_labels')
        print(f"Label map loaded successfully. Num labels: {num_labels}")


        # --- Instantiate Model Architecture ---
        # Get BERT base model name from metadata if available, else default
        bert_model_name = meta.get('bert_model_name', meta.get('config_used',{}).get('pretrained_model', 'bert-base-uncased'))


        model = BERTWithRNN(
            bert_model_name=bert_model_name, # Use the name used during training
            num_labels=num_labels,
            # Ensure these match the saved model's architecture from metadata
            rnn_type=meta.get('rnn_type', meta.get('config_used',{}).get('rnn_type', 'LSTM')),
            rnn_hidden_size=meta.get('rnn_hidden_size', meta.get('config_used',{}).get('hidden_size', 128)),
            rnn_num_layers=meta.get('rnn_num_layers', meta.get('config_used',{}).get('num_layers', 1)),
            rnn_dropout=meta.get('rnn_dropout', meta.get('config_used',{}).get('dropout', 0.3)), # Check if dropout saved in meta
            bidirectional=meta.get('rnn_bidirectional', meta.get('config_used',{}).get('bidirectional', True)),
            freeze_bert=False # Does not matter for inference
        )
        print("Model architecture created.")

        # --- Load Saved Weights ---
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print("Model weights loaded successfully.")

    except Exception as e:
        print(f"Error during model or tokenizer loading: {e}")
        return None, None

    # --- Prepare Input Text ---
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True, # Ensure text fits model max length
            padding=True,    # Pad to max length if needed (though less critical for single input)
            max_length=tokenizer.model_max_length # Use tokenizer's max length
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
    except Exception as e:
        print(f"Error during text tokenization: {e}")
        return None, None

    # --- Run Inference ---
    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # --- Process Output ---
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        predicted_class_id = np.argmax(probabilities)
        predicted_label = id_to_emotion.get(str(predicted_class_id), "Unknown")

        # Create dict of {label_name: probability}
        prob_dict = {
            id_to_emotion.get(str(i), f"Unknown_{i}"): float(prob)
            for i, prob in enumerate(probabilities)
        }

        return predicted_label, prob_dict

    except Exception as e:
        print(f"Error during model inference: {e}")
        return None, None


# --- Example Usage ---
if __name__ == "__main__":
    # Make sure your saved model files are in 'saved_models/emotion_bert_rnn'
    # and emotion_label_map.json is in 'emotion_prep_outputs'

    test_text_1 = "I am feeling so happy and excited about the weekend!"
    test_text_2 = "This is really frustrating, I can't believe it happened."
    test_text_3 = "Wow, I did not expect that at all!"

    print(f"\n--- Predicting for: '{test_text_1}' ---")
    label1, probs1 = predict_emotion(test_text_1)
    if label1:
        print(f"Predicted Emotion: {label1}")
        print("Probabilities:")
        for emotion, prob in sorted(probs1.items(), key=lambda item: item[1], reverse=True):
             print(f"  - {emotion}: {prob:.4f}")

    print(f"\n--- Predicting for: '{test_text_2}' ---")
    label2, probs2 = predict_emotion(test_text_2)
    if label2:
        print(f"Predicted Emotion: {label2}")
        print("Probabilities:")
        for emotion, prob in sorted(probs2.items(), key=lambda item: item[1], reverse=True):
             print(f"  - {emotion}: {prob:.4f}")

    print(f"\n--- Predicting for: '{test_text_3}' ---")
    label3, probs3 = predict_emotion(test_text_3)
    if label3:
        print(f"Predicted Emotion: {label3}")
        print("Probabilities:")
        for emotion, prob in sorted(probs3.items(), key=lambda item: item[1], reverse=True):
             print(f"  - {emotion}: {prob:.4f}")



    #!/usr/bin/env python3
"""
Preprocess harassment detection dataset (Jigsaw Toxic Comments).

This script:
  1. Loads and cleans the harassment dataset
  2. Creates train/val split with stratification
  3. Tokenizes with BERT tokenizer
  4. Saves encodings and labels for training

Usage:
    python harassment_prep.py

Outputs to: harassment_prep_outputs/
    - harassment_train_encodings.pt
    - harassment_val_encodings.pt
    - harassment_train_labels.pt
    - harassment_val_labels.pt
    - harassment_label_map.json
"""

import os
import json
import re
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import emoji
from textblob import TextBlob

# ---------- Config ----------
INPUT_FILE = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
OUTPUT_DIR = "harassment_prep_outputs"
PRETRAINED = "bert-base-uncased"
MAX_LEN = 100
VAL_SIZE = 0.2
SAMPLE_SIZE = 25000  # Same as your notebook
SEED = 42
# ----------------------------

def improved_preprocess(text, do_spelling_correction=False):
    """
    Clean and preprocess text.
    Same function as your notebook.
    """
    text = re.sub(r'http\S+|www.\S+', '', text)         # Remove URLs
    text = re.sub(r'<.*?>', '', text)                   # Remove HTML
    text = re.sub(r'@\w+', '', text)                    # Remove @handles
    text = re.sub(r'#\w+', '', text)                    # Remove hashtags
    text = emoji.demojize(text)                         # Convert emojis to text
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)          # Normalize repeated chars
    text = re.sub(r'[^\x00-\x7F]+', '', text)           # Remove non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()            # Remove extra spaces
    text = text.lower()                                 # Convert to lowercase

    if do_spelling_correction:
        text = str(TextBlob(text).correct())
    return text

def main():
    print("="*60)
    print("HARASSMENT DETECTION - DATA PREPROCESSING")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n1. Loading Jigsaw dataset...")
    df = pd.read_csv(INPUT_FILE, compression='zip')
    print(f"   Original dataset size: {len(df)}")
    
    # Create binary harassment label (any toxic label = 1)
    harassment_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['harassment'] = df[harassment_labels].max(axis=1)
    
    print(f"\n   Class distribution:")
    print(f"   Non-harassment: {(df['harassment'] == 0).sum()}")
    print(f"   Harassment: {(df['harassment'] == 1).sum()}")
    
    # Stratified sampling to reduce size
    print(f"\n2. Sampling {SAMPLE_SIZE} examples with stratification...")
    df_sampled, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df['harassment'],
        random_state=SEED
    )
    
    print(f"   Sampled class distribution:")
    print(f"   Non-harassment: {(df_sampled['harassment'] == 0).sum()}")
    print(f"   Harassment: {(df_sampled['harassment'] == 1).sum()}")
    
    # Clean text
    print("\n3. Cleaning text...")
    df_sampled['clean_text'] = df_sampled['comment_text'].astype(str).apply(
        lambda x: improved_preprocess(x, do_spelling_correction=False)
    )
    
    # Remove empty strings
    df_sampled = df_sampled[df_sampled['clean_text'].str.strip() != '']
    print(f"   After cleaning: {len(df_sampled)} samples")
    
    # Train/val split
    print(f"\n4. Creating train/val split ({int((1-VAL_SIZE)*100)}/{int(VAL_SIZE*100)})...")
    train_df, val_df = train_test_split(
        df_sampled,
        test_size=VAL_SIZE,
        stratify=df_sampled['harassment'],
        random_state=SEED
    )
    
    print(f"   Training set: {len(train_df)}")
    print(f"   Validation set: {len(val_df)}")
    
    # Tokenize
    print(f"\n5. Tokenizing with BERT (max_length={MAX_LEN})...")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED)
    
    train_encodings = tokenizer(
        list(train_df['clean_text']),
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    val_encodings = tokenizer(
        list(val_df['clean_text']),
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    train_labels = torch.tensor(train_df['harassment'].values, dtype=torch.long)
    val_labels = torch.tensor(val_df['harassment'].values, dtype=torch.long)
    
    print(f"   Train encodings shape: {train_encodings['input_ids'].shape}")
    print(f"   Val encodings shape: {val_encodings['input_ids'].shape}")
    
    # Save everything
    print(f"\n6. Saving to {OUTPUT_DIR}/...")
    torch.save(train_encodings, os.path.join(OUTPUT_DIR, "harassment_train_encodings.pt"))
    torch.save(val_encodings, os.path.join(OUTPUT_DIR, "harassment_val_encodings.pt"))
    torch.save(train_labels, os.path.join(OUTPUT_DIR, "harassment_train_labels.pt"))
    torch.save(val_labels, os.path.join(OUTPUT_DIR, "harassment_val_labels.pt"))
    
    # Save label mapping
    label_map = {
        'id_to_label': {0: 'non_harassment', 1: 'harassment'},
        'label_to_id': {'non_harassment': 0, 'harassment': 1}
    }
    with open(os.path.join(OUTPUT_DIR, "harassment_label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    
    print("\n✓ Preprocessing complete!")
    print(f"\nOutput files:")
    print(f"  - harassment_train_encodings.pt")
    print(f"  - harassment_val_encodings.pt")
    print(f"  - harassment_train_labels.pt")
    print(f"  - harassment_val_labels.pt")
    print(f"  - harassment_label_map.json")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Hybrid BERT + Emotion model for harassment detection.

This model:
  1. Extracts BERT contextual embeddings
  2. Predicts emotions using pre-trained emotion model
  3. Fuses both features for harassment classification

Architecture:
    Text → [BERT → 768-dim] + [Emotion Model → 6-dim] → Fusion → Binary Classification

Usage:
    python hybrid_harassment_train.py

Requirements:
    - Preprocessed harassment data (from harassment_prep.py)
    - Trained emotion model (from bert_rnn_emotion_train.py)

Outputs:
    - saved_models/harassment_hybrid/
"""

import os
import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Config ----------
HARASSMENT_PREP_DIR = "harassment_prep_outputs"
EMOTION_MODEL_DIR = "saved_models/emotion_bert_rnn"  # Path to your trained emotion model
SAVE_DIR = "saved_models/harassment_hybrid"
PRETRAINED_BERT = "bert-base-uncased"

BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 42

# Fusion layer config
FUSION_HIDDEN_SIZE = 256
FUSION_DROPOUT = 0.3
# ----------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EncodedDataset(Dataset):
    """Dataset for pre-tokenized texts"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return self.labels.size(0)
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

class BERTWithRNN(nn.Module):
    """
    Emotion model architecture (same as emotion training).
    We need this to load the pre-trained emotion model.
    """
    def __init__(
        self, 
        bert_model_name, 
        num_labels, 
        rnn_type="LSTM",
        rnn_hidden_size=128,
        rnn_num_layers=1,
        rnn_dropout=0.3,
        bidirectional=True
    ):
        super(BERTWithRNN, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.bidirectional = bidirectional
        
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.bert_hidden_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.bert_hidden_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        
        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(rnn_output_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        rnn_output, hidden = self.rnn(sequence_output)
        
        if self.rnn_type == "LSTM":
            h_n = hidden[0]
        else:
            h_n = hidden
        
        if self.bidirectional:
            final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_hidden = h_n[-1]
        
        pooled = self.dropout(final_hidden)
        logits = self.classifier(pooled)
        
        return logits

class HybridHarassmentModel(nn.Module):
    """
    Hybrid model combining BERT and Emotion features for harassment detection.
    
    Architecture:
        1. BERT branch: extracts contextual embeddings (768-dim)
        2. Emotion branch: uses pre-trained emotion model (6-dim probabilities)
        3. Fusion: concatenates both features
        4. Classifier: predicts harassment (binary)
    
    Why this works:
        - BERT understands what words mean
        - Emotion model understands emotional tone
        - Together: "angry insult" detected as harassment, "sad complaint" might not be
    """
    def __init__(
        self, 
        bert_model_name,
        emotion_model,
        num_emotions=6,
        fusion_hidden_size=256,
        fusion_dropout=0.3,
        freeze_emotion_model=True
    ):
        super(HybridHarassmentModel, self).__init__()
        
        # Branch 1: BERT for contextual understanding
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # 768
        
        # Branch 2: Pre-trained emotion model
        self.emotion_model = emotion_model
        if freeze_emotion_model:
            # Freeze emotion model weights (already trained)
            for param in self.emotion_model.parameters():
                param.requires_grad = False
            print("Emotion model weights frozen.")
        
        # Fusion layers
        # Input: [768 BERT + 6 emotion probabilities] = 774 dimensions
        fusion_input_size = self.bert_hidden_size + num_emotions
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(fusion_dropout)
        )
        
        # Final binary classifier
        self.classifier = nn.Linear(128, 2)  # 2 classes: non-harassment, harassment
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through hybrid model.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, 2) - harassment scores
        """
        # Branch 1: BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation for sentence-level features
        bert_features = bert_outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        # Branch 2: Emotion predictions
        with torch.no_grad() if not self.training else torch.enable_grad():
            emotion_logits = self.emotion_model(input_ids=input_ids, attention_mask=attention_mask)
            # Convert to probabilities for softer features
            emotion_probs = torch.softmax(emotion_logits, dim=1)  # (batch, 6)
        
        # Fusion: concatenate BERT and emotion features
        combined_features = torch.cat([bert_features, emotion_probs], dim=1)  # (batch, 774)
        
        # Pass through fusion layers
        fused = self.fusion(combined_features)  # (batch, 128)
        
        # Final classification
        logits = self.classifier(fused)  # (batch, 2)
        
        return logits

def load_emotion_model(model_dir, device):
    """
    Load pre-trained emotion model.
    
    Returns:
        Loaded emotion model in eval mode
    """
    print(f"Loading emotion model from {model_dir}...")
    
    # Load metadata to get architecture config
    meta_path = os.path.join(model_dir, "train_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Reconstruct model architecture
    emotion_model = BERTWithRNN(
        bert_model_name=PRETRAINED_BERT,
        num_labels=meta['num_labels'],
        rnn_type=meta.get('rnn_type', 'LSTM'),
        rnn_hidden_size=meta.get('rnn_hidden_size', 128),
        rnn_num_layers=meta.get('rnn_num_layers', 1),
        rnn_dropout=0.3,
        bidirectional=meta.get('rnn_bidirectional', True)
    )
    
    # Load trained weights
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    emotion_model.load_state_dict(torch.load(model_path, map_location=device))
    emotion_model.to(device)
    emotion_model.eval()  # Set to evaluation mode
    
    print(f"✓ Emotion model loaded successfully")
    return emotion_model

def load_prep_outputs(prep_dir):
    """Load preprocessed harassment data"""
    train_enc = torch.load(os.path.join(prep_dir, "harassment_train_encodings.pt"), weights_only=False) # Added weights_only=False
    val_enc = torch.load(os.path.join(prep_dir, "harassment_val_encodings.pt"), weights_only=False)     # Added weights_only=False
    train_labels = torch.load(os.path.join(prep_dir, "harassment_train_labels.pt"), weights_only=False) # Added weights_only=False
    val_labels = torch.load(os.path.join(prep_dir, "harassment_val_labels.pt"), weights_only=False)   # Added weights_only=False
    # ... rest of the function ...
    
    with open(os.path.join(prep_dir, "harassment_label_map.json"), "r") as f:
        label_map = json.load(f)
    
    return train_enc, val_enc, train_labels, val_labels, label_map

def compute_class_weights(labels_tensor):
    """Compute class weights for imbalanced datasets"""
    labels_np = labels_tensor.cpu().numpy()
    counts = np.bincount(labels_np)
    counts = np.where(counts == 0, 1, counts)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float)

def evaluate(model, dataloader, device, loss_fn=None):
    """Evaluate model on validation set"""
    model.eval()
    preds_all = []
    labels_all = []
    losses = []
    
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())
    
    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(labels_all, preds_all)
    precision = precision_score(labels_all, preds_all, average='binary', zero_division=0)
    recall = recall_score(labels_all, preds_all, average='binary', zero_division=0)
    f1 = f1_score(labels_all, preds_all, average='binary', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': labels_all,
        'y_pred': preds_all
    }

def save_model(model, tokenizer, save_dir, metadata):
    """Save hybrid model"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save full model state
    model_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    # Save metadata
    meta_path = os.path.join(save_dir, "train_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {save_dir}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Generate confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load preprocessed harassment data
    print("\n" + "="*60)
    print("PHASE 2: HYBRID HARASSMENT DETECTION")
    print("="*60)
    print("\n1. Loading preprocessed harassment data...")
    train_enc, val_enc, train_labels, val_labels, label_map = load_prep_outputs(HARASSMENT_PREP_DIR)
    print(f"   Training samples: {len(train_labels)}")
    print(f"   Validation samples: {len(val_labels)}")
    print(f"   Class distribution (train):")
    print(f"     Non-harassment: {(train_labels == 0).sum().item()}")
    print(f"     Harassment: {(train_labels == 1).sum().item()}")
    
    # Create datasets
    train_dataset = EncodedDataset(train_enc, train_labels)
    val_dataset = EncodedDataset(val_enc, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load pre-trained emotion model
    print("\n2. Loading pre-trained emotion model...")
    emotion_model = load_emotion_model(EMOTION_MODEL_DIR, device)
    
    # Initialize hybrid model
    print("\n3. Initializing hybrid harassment detection model...")
    model = HybridHarassmentModel(
        bert_model_name=PRETRAINED_BERT,
        emotion_model=emotion_model,
        num_emotions=6,
        fusion_hidden_size=FUSION_HIDDEN_SIZE,
        fusion_dropout=FUSION_DROPOUT,
        freeze_emotion_model=True  # Keep emotion model frozen
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Class weights for imbalanced data
    class_weights = compute_class_weights(train_labels).to(device)
    print(f"\n4. Class weights: {class_weights.cpu().numpy()}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = -1.0
    best_epoch = -1
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    print(f"\n{'='*60}")
    print(f"5. Starting training for {NUM_EPOCHS} epochs...")
    print(f"{'='*60}\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        step = 0
        start_time = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            step += 1
        
        # Training metrics
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        avg_train_loss = epoch_loss / max(1, step)
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
        print(f"  Train - Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        
        # Validation
        val_res = evaluate(model, val_loader, device, loss_fn)
        
        history['val_loss'].append(val_res['loss'])
        history['val_acc'].append(val_res['accuracy'])
        history['val_f1'].append(val_res['f1'])
        
        print(f"  Val   - Loss: {val_res['loss']:.4f} | Acc: {val_res['accuracy']:.4f} | F1: {val_res['f1']:.4f}")
        print(f"          Precision: {val_res['precision']:.4f} | Recall: {val_res['recall']:.4f}")
        
        # Save best model
        if val_res['f1'] > best_val_f1:
            best_val_f1 = val_res['f1']
            best_epoch = epoch
            
            metadata = {
                'model_type': 'BERT+Emotion Hybrid',
                'best_val_f1': float(best_val_f1),
                'best_val_accuracy': float(val_res['accuracy']),
                'best_val_precision': float(val_res['precision']),
                'best_val_recall': float(val_res['recall']),
                'best_epoch': int(best_epoch),
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'num_epochs': NUM_EPOCHS,
                'fusion_hidden_size': FUSION_HIDDEN_SIZE,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT)
            save_model(model, tokenizer, SAVE_DIR, metadata)
            print(f"  ✓ New best model saved (F1: {best_val_f1:.4f})")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"{'='*60}\n")
    
    # Load best model
    print("Loading best model for final evaluation...")
    model_path = os.path.join(SAVE_DIR, "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Final metrics
    val_res = evaluate(model, val_loader, device, loss_fn)
    print(f"\nFinal Best Model Performance:")
    print(f"  Accuracy: {val_res['accuracy']:.4f}")
    print(f"  Precision: {val_res['precision']:.4f}")
    print(f"  Recall: {val_res['recall']:.4f}")
    print(f"  F1-Score: {val_res['f1']:.4f}")
    
    # Classification report
    print("\nGenerating detailed classification report...")
    class_names = ['Non-Harassment', 'Harassment']
    report = classification_report(
        val_res['y_true'],
        val_res['y_pred'],
        target_names=class_names,
        zero_division=0,
        digits=4
    )
    print(report)
    
    # Save report
    report_path = os.path.join(SAVE_DIR, "val_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("BERT + Emotion Hybrid Harassment Detection\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Accuracy: {val_res['accuracy']:.4f}\n")
        f.write(f"Precision: {val_res['precision']:.4f}\n")
        f.write(f"Recall: {val_res['recall']:.4f}\n")
        f.write(f"F1-Score: {val_res['f1']:.4f}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(SAVE_DIR, "val_confusion_matrix.png")
    plot_confusion_matrix(
        val_res['y_true'],
        val_res['y_pred'],
        class_names,
        cm_path,
        title=f"Hybrid Model Confusion Matrix (Epoch {best_epoch})"
    )
    
    # Save training history
    history_path = os.path.join(SAVE_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    print("\n✓ All done! Model and results saved to:", SAVE_DIR)
    print("\nNext steps:")
    print("  1. Compare with baseline BERT model")
    print("  2. Analyze which emotions correlate with harassment")
    print("  3. Test on custom examples")

if __name__ == "__main__":
    main()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the path
image_path = 'saved_models/harassment_hybrid/val_confusion_matrix.png'

# Read the image
img = mpimg.imread(image_path)

# Plot the image
plt.figure(figsize=(9, 13)) # Optional: adjust size
plt.imshow(img)
plt.title('Confusion Matrix for BERT+LSTM ', fontsize=16) # Add your title here
plt.axis('off')  # Hides the axes
plt.show()





#!/usr/bin/env python3
"""
Test harassment detection on custom examples.
Shows both harassment prediction and detected emotion.

Usage:
    python test_harassment_detection.py
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import json
import re
import emoji

# Paths
EMOTION_MODEL_DIR = "saved_models/emotion_bert_rnn"
HYBRID_MODEL_DIR = "saved_models/harassment_hybrid"
EMOTION_PREP_DIR = "emotion_prep_outputs"
MAX_LEN = 64

class BERTWithRNN(nn.Module):
    """Emotion model"""
    def __init__(self, bert_model_name, num_labels, rnn_type="LSTM",
                 rnn_hidden_size=128, rnn_num_layers=1, bidirectional=True):
        super(BERTWithRNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.bert_hidden_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=0.3 if rnn_num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                input_size=self.bert_hidden_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=0.3 if rnn_num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True
            )
        
        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(rnn_output_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        rnn_output, hidden = self.rnn(sequence_output)
        
        if self.rnn_type == "LSTM":
            h_n = hidden[0]
        else:
            h_n = hidden
        
        if self.bidirectional:
            final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_hidden = h_n[-1]
        
        pooled = self.dropout(final_hidden)
        logits = self.classifier(pooled)
        return logits

class HybridHarassmentModel(nn.Module):
    """Hybrid model"""
    def __init__(self, bert_model_name, emotion_model, num_emotions=6,
                 fusion_hidden_size=256, fusion_dropout=0.3):
        super(HybridHarassmentModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.emotion_model = emotion_model
        
        for param in self.emotion_model.parameters():
            param.requires_grad = False
        
        fusion_input_size = self.bert_hidden_size + num_emotions
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(fusion_dropout)
        )
        self.classifier = nn.Linear(128, 2)
    
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_features = bert_outputs.last_hidden_state[:, 0, :]
        
        with torch.no_grad():
            emotion_logits = self.emotion_model(input_ids=input_ids, attention_mask=attention_mask)
            emotion_probs = torch.softmax(emotion_logits, dim=1)
        
        combined_features = torch.cat([bert_features, emotion_probs], dim=1)
        fused = self.fusion(combined_features)
        logits = self.classifier(fused)
        return logits, emotion_probs

def preprocess_text(text):
    """Clean text"""
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = emoji.demojize(text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def load_models(device):
    """Load emotion and hybrid models"""
    # Load emotion model
    with open(f"{EMOTION_MODEL_DIR}/train_meta.json", "r") as f:
        emotion_meta = json.load(f)
    
    emotion_model = BERTWithRNN(
        bert_model_name="bert-base-uncased",
        num_labels=emotion_meta['num_labels'],
        rnn_type=emotion_meta.get('rnn_type', 'LSTM'),
        rnn_hidden_size=emotion_meta.get('rnn_hidden_size', 128),
        rnn_num_layers=emotion_meta.get('rnn_num_layers', 1),
        bidirectional=emotion_meta.get('rnn_bidirectional', True)
    )
    emotion_model.load_state_dict(
        torch.load(f"{EMOTION_MODEL_DIR}/pytorch_model.bin", map_location=device)
    )
    emotion_model.to(device)
    emotion_model.eval()
    
    # Load hybrid model
    hybrid_model = HybridHarassmentModel(
        bert_model_name="bert-base-uncased",
        emotion_model=emotion_model,
        num_emotions=6
    )
    hybrid_model.load_state_dict(
        torch.load(f"{HYBRID_MODEL_DIR}/pytorch_model.bin", map_location=device)
    )
    hybrid_model.to(device)
    hybrid_model.eval()
    
    return hybrid_model

def predict(text, model, tokenizer, device, emotion_labels):
    """Predict harassment and emotion"""
    # Preprocess
    clean_text = preprocess_text(text)
    
    # Tokenize
    inputs = tokenizer(
        clean_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        logits, emotion_probs = model(input_ids, attention_mask)
        harassment_prob = torch.softmax(logits, dim=1)[0]
        harassment_pred = torch.argmax(logits, dim=1).item()
        emotion_pred = torch.argmax(emotion_probs, dim=1).item()
    
    return {
        'harassment': 'Yes' if harassment_pred == 1 else 'No',
        'harassment_confidence': harassment_prob[harassment_pred].item(),
        'emotion': emotion_labels[emotion_pred],
        'emotion_confidence': emotion_probs[0][emotion_pred].item(),
        'emotion_distribution': {
            emotion_labels[i]: emotion_probs[0][i].item()
            for i in range(len(emotion_labels))
        }
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load models
    print("Loading models...")
    model = load_models(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
  # Load emotion labels
    # --- Load the CORRECT file from the prep directory ---
    label_map_path = os.path.join(EMOTION_PREP_DIR, "emotion_label_map.json") # <-- Use EMOTION_PREP_DIR
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Emotion label map not found at: {label_map_path}. Ensure emotion_prep_outputs directory is correct.")

    with open(label_map_path, "r", encoding="utf-8") as f: # <-- Use correct path
        label_map = json.load(f)
    # --- This line should now work ---
    emotion_labels = [label_map['id_to_emotion'][str(i)] for i in range(6)]
    
    print("Models loaded successfully!\n")
    print("="*60)
    
    # Test examples
    test_texts = [
        "You're so stupid, I hate you!",
        "I'm really sad about what happened yesterday",
        "This is the best day ever! So happy!",
        "Get lost you idiot, nobody wants you here",
        "I'm afraid of what might happen next",
        "That's an interesting perspective, thanks for sharing"
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        print("-" * 60)
        
        result = predict(text, model, tokenizer, device, emotion_labels)
        
        print(f"Harassment: {result['harassment']} (confidence: {result['harassment_confidence']:.2%})")
        print(f"Primary Emotion: {result['emotion']} (confidence: {result['emotion_confidence']:.2%})")
        print(f"\nEmotion Distribution:")
        for emotion, prob in sorted(result['emotion_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  {emotion:10s}: {'█' * int(prob * 50)} {prob:.2%}")
        print("=" * 60)
    
    # Interactive mode
    print("\n\nInteractive Mode (type 'quit' to exit)")
    print("="*60)
    while True:
        text = input("\nEnter text to analyze: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        if not text:
            continue
        
        result = predict(text, model, tokenizer, device, emotion_labels)
        print(f"\n  Harassment: {result['harassment']} ({result['harassment_confidence']:.2%})")
        print(f"  Emotion: {result['emotion']} ({result['emotion_confidence']:.2%})")

if __name__ == "__main__":
    main()




