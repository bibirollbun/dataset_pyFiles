from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import DataCollatorWithPadding
from tensorflow.keras.optimizers import Adam
from datasets import Dataset
import numpy as np
import pandas as pd

import os
import unicodedata

import string
from sklearn.metrics import accuracy_score


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split




# Complete BERT Fine-tuning Pipeline - All in One Cell

import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from datasets import Dataset
import tensorflow as tf

# Download NLTK data (run once)
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except:
    pass

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def read_texts_from_dir(dir_path):
    """
    Reads the texts from a given directory and saves them in the pd.DataFrame with columns ['id', 'file_1', 'file_2'].
    """
    data = []
    
    for folder_name in sorted(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
                    text1 = f1.read().strip()
                with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
                    text2 = f2.read().strip()
                
                index = int(folder_name[-4:])  # Extract last 4 characters as ID
                data.append((index, text1, text2))
                
            except Exception as e:
                print(f"Error reading directory {folder_name}: {e}")
    
    print(f"Successfully read {len(data)} directories")
    df = pd.DataFrame(data, columns=['id', 'file_1', 'file_2'])
    return df

# === DATA LOADING ===
print("Loading data...")
train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
df_train = read_texts_from_dir(train_path)
test_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
df_test = read_texts_from_dir(test_path)

# Load ground truth for train data
df_train_gt = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")

# Merge ground truth with train pairs
df = df_train.merge(df_train_gt, on="id")

# === RESHAPE TO LONG FORMAT ===
print("Reshaping data...")

# Training data - reshape to long format
df_long = []
for _, row in df.iterrows():
    # file_1
    df_long.append({
        "id": row["id"],
        "text": row["file_1"],
        "label": 1 if row["real_text_id"] == 1 else 0
    })
    # file_2
    df_long.append({
        "id": row["id"],
        "text": row["file_2"],
        "label": 1 if row["real_text_id"] == 2 else 0
    })

df_long = pd.DataFrame(df_long)

# Test data - reshape to long format
df_long_test = []
for _, row in df_test.iterrows():
    # file_1
    df_long_test.append({
        "id": row["id"],
        "text": row["file_1"]
    })
    # file_2
    df_long_test.append({
        "id": row["id"],
        "text": row["file_2"]
    })

df_long_test = pd.DataFrame(df_long_test)

# === PREPROCESSING ===
print("Preprocessing text...")
df_long["clean_text"] = df_long["text"].apply(preprocess_text)
df_long_test["clean_text"] = df_long_test["text"].apply(preprocess_text)

# Prepare training data
X = df_long["clean_text"].values
y = df_long["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === TOKENIZATION ===
print("Tokenizing data...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Convert to Hugging Face datasets
train_df = pd.DataFrame({"clean_text": X_train, "label": y_train})
val_df = pd.DataFrame({"clean_text": X_val, "label": y_val})

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["clean_text"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Rename label column and remove unnecessary columns
train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")
train_ds = train_ds.remove_columns(["clean_text"])
val_ds = val_ds.remove_columns(["clean_text"])

# Prepare test dataset
test_dataset = Dataset.from_pandas(df_long_test)
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.remove_columns(["clean_text"])

# === MODEL SETUP ===
print("Setting up model...")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Compile with string identifier
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# Set learning rate
model.optimizer.learning_rate = 2e-5

# === CREATE TF DATASETS ===
print("Creating TensorFlow datasets...")
train_tf_dataset = train_ds.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"],
    batch_size=16,
    shuffle=True
)

val_tf_dataset = val_ds.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"],
    batch_size=16
)

# === TRAINING ===
print("Starting training...")
history = model.fit(
    train_tf_dataset,
    validation_data=val_tf_dataset,
    epochs=10,
    verbose=1
)

print("Training completed!")

# === PREDICTION ===
print("Processing test data...")

# Create test dataset for prediction
test_tf_dataset = test_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    batch_size=16,
    shuffle=False
)

# Make predictions
print("Making predictions...")
preds = model.predict(test_tf_dataset)
pred_labels = np.argmax(preds.logits, axis=1)

# Add predictions to dataframe
df_long_test["pred"] = pred_labels
print(f"Predictions completed. Shape: {pred_labels.shape}")

# === CREATE SUBMISSION ===
print("Creating submission format...")
submission_data = []

for test_id in sorted(df_long_test['id'].unique()):
    id_data = df_long_test[df_long_test['id'] == test_id].reset_index(drop=True)
    
    # Get predictions for both texts (should be exactly 2 texts per ID)
    preds_for_id = id_data['pred'].values
    
    # Determine which text is predicted as real (1)
    if len(preds_for_id) == 2:
        if preds_for_id[0] == 1 and preds_for_id[1] == 0:
            real_text_id = 1
        elif preds_for_id[0] == 0 and preds_for_id[1] == 1:
            real_text_id = 2
        else:
            # If both or neither are predicted as real, choose first one as default
            # You could also use probability scores here for better decision
            real_text_id = 1
    else:
        real_text_id = 1  # Default fallback
    
    submission_data.append({'id': test_id, 'real_text_id': real_text_id})

# Create submission dataframe
submission_df = pd.DataFrame(submission_data)
submission_df = submission_df.sort_values('id').reset_index(drop=True)

print("Submission format:")
print(submission_df.head())
print(f"Submission shape: {submission_df.shape}")

# Save submission
submission_df.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")

# Display some statistics
print(f"\nTraining samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Unique test IDs: {len(submission_df)}")

























