import os
import pandas as pd
import unicodedata

import string
from sklearn.metrics import accuracy_score
import numpy as np



import os
import pandas as pd
def read_texts_from_dir(dir_path):
    """
    Reads the texts from a given directory and saves them in the pd.DataFrame with columns ['id', 'file_1', 'file_2'].
    Params:
      dir_path (str): path to the directory with data
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


# Use the above function to load both train and test data
train_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
df_train=read_texts_from_dir(train_path)
test_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
df_test=read_texts_from_dir(test_path)


# Load ground truth for train data
df_train_gt=pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
df_train_gt


df_train.head()


# Merge ground truth with train pairs
df = df_train.merge(df_train_gt, on="id")

# Reshape into long format: one row per text
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


df_long.head()



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

df_long_test.head()



import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# -------------------------
# 1. Preprocessing
# -------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df_long["clean_text"] = df_long["text"].apply(preprocess_text)
df_long_test["clean_text"] = df_long_test["text"].apply(preprocess_text)

# -------------------------
# 2. Tokenization
# -------------------------
max_words = 20000  # vocabulary size
max_len = 100      # max sequence length

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df_long["clean_text"])

X = tokenizer.texts_to_sequences(df_long["clean_text"])
X = pad_sequences(X, maxlen=max_len, padding="post")

y = df_long["label"].values

X_test = tokenizer.texts_to_sequences(df_long_test["clean_text"])
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

# -------------------------
# 3. Train/Val split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 4. Build Model
# -------------------------
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")  # binary classification
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

# -------------------------
# 5. Train Model
# -------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    verbose=1
)

# -------------------------
# 6. Predict on Test
# -------------------------
predictions_test = (model.predict(X_test) > 0.5).astype(int).reshape(-1)

# -------------------------
# 7. Submission (pair logic)
# -------------------------
df_long_test["pred"] = predictions_test

submission = []
for pair_id, group in df_long_test.groupby("id"):
    if group.iloc[0]["pred"] == 1:
        real_text_id = 1
    else:
        real_text_id = 2
    submission.append({"id": pair_id, "real_text_id": real_text_id})

output_df = pd.DataFrame(submission)
output_df.to_csv("submission.csv", index=False)

print("✅ Submission file saved:", output_df.shape)
print(output_df.head())


