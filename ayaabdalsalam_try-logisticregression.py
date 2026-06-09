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



df_long_test.head()


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()                               # lowercase
    text = re.sub(r"[^a-z\s]", "", text)              # remove punctuation/numbers
    words = text.split()                              # tokenize
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]  
    return " ".join(words)

# Apply cleaning
df_long["clean_text"] = df_long["text"].apply(preprocess_text)

df_long[["text", "clean_text", "label"]].head()
df_long_test["clean_text"] = df_long_test["text"].apply(preprocess_text)



from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # limit features for speed

# Fit on train data and transform
X = vectorizer.fit_transform(df_long["clean_text"])
y = df_long["label"]

print("Shape of TF-IDF matrix:", X.shape)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Train-test split (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize Logistic Regression
clf = LogisticRegression(max_iter=500)

# Train
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_val)

# Evaluate
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))









from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Split
X_train, X_val, y_train, y_val = train_test_split(
    df_long['clean_text'], df_long['label'], test_size=0.2, random_state=42
)

# TF-IDF with bigrams
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Train Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_val_tfidf)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Split train data
X_train, X_val, y_train, y_val = train_test_split(
    df_long["clean_text"], df_long["label"], test_size=0.2, random_state=42
)

# 2. Initialize TF-IDF (only once!)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Fit on training, transform both train and val
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 3. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# 4. Validate
y_val_pred = model.predict(X_val_vec)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# 5. Transform test set using the SAME vectorizer
X_test = vectorizer.transform(df_long_test["clean_text"])

# 6. Predict on test
predictions_test = model.predict(X_test)



# predictions_test corresponds to df_long_test rows (2136 rows)
df_long_test["pred"] = predictions_test

# Group by original pair id (1068 unique ids)
submission = []

for pair_id, group in df_long_test.groupby("id"):
    # group has 2 rows: one for file_1, one for file_2
    if group.iloc[0]["pred"] == 1:
        real_text_id = 1
    else:
        real_text_id = 2
    submission.append({"id": pair_id, "real_text_id": real_text_id})

# Final submission dataframe
output_df = pd.DataFrame(submission)

print(output_df.shape)  # should be (1068, 2)
# Save to CSV with correct name
output_df.to_csv("submission.csv", index=False)

print("✅ submission.csv file created with shape:", output_df.shape)











