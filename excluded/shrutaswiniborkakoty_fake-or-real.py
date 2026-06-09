!pip install textstat


!pip install sentence_transformers


import pandas as pd
import os
import textstat
import spacy
from textblob import TextBlob
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from xgboost import XGBClassifier
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV
from nltk.corpus import stopwords
import string
from collections import Counter
from sklearn.preprocessing import StandardScaler
import re
import matplotlib.pyplot as plt


train_df = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')


train_df.head


def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


train_dir = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train'


texts = []


for _, row in train_df.iterrows():
    id_int = row['id']
    id_str = f"article_{id_int:04d}"  
    real_id = row['real_text_id']
    
    file1_path = os.path.join(train_dir, id_str, "file_1.txt")
    file2_path = os.path.join(train_dir, id_str, "file_2.txt")


  for idx, row in train_df.head(5).iterrows():
    article_id = f"article_{int(row['id']):04d}"  # ensures format article_0000
    base_path = f"/kaggle/input/fake-or-real-the-impostor-hunt/data/train/{article_id}"
    
    file1_path = os.path.join(base_path, 'file_1.txt')
    file2_path = os.path.join(base_path, 'file_2.txt')

    text1 = read_text_file(file1_path)
    text2 = read_text_file(file2_path)

    print(f"Article ID: {row['id']}, Real file: file_{row['real_text_id']}.txt")
    print("---- FILE 1 ----\n", text1[:300], "...\n")
    print("---- FILE 2 ----\n", text2[:300], "...\n")
    print("="*50)


   data = []

for idx, row in train_df.iterrows():
    article_id = f"article_{int(row['id']):04d}"
    base_path = f"/kaggle/input/fake-or-real-the-impostor-hunt/data/train/{article_id}"

    file1_text = read_text_file(os.path.join(base_path, 'file_1.txt'))
    file2_text = read_text_file(os.path.join(base_path, 'file_2.txt'))

    data.append({
        'id': row['id'],
        'file_1': file1_text,
        'file_2': file2_text,
        'label': row['real_text_id']
    })

train_texts = pd.DataFrame(data)
train_texts.head()


nlp = spacy.load("en_core_web_sm")
tqdm.pandas()


def extract_features(text):
    blob = TextBlob(text)
    doc = nlp(text)
    return {
        # Textstat readability
        "flesch": textstat.flesch_reading_ease(text),
        "fog": textstat.gunning_fog(text),
        "smog": textstat.smog_index(text),
        "lexicon_count": textstat.lexicon_count(text, removepunct=True),

        # Lexical
        "word_count": len(text.split()),
        "char_count": len(text),
        "avg_word_len": sum(len(w) for w in text.split()) / max(1, len(text.split())),
        "ttr": len(set(text.split())) / max(1, len(text.split())),

        # Syntactic
        "noun_count": sum(1 for token in doc if token.pos_ == "NOUN"),
        "verb_count": sum(1 for token in doc if token.pos_ == "VERB"),
        "adj_count": sum(1 for token in doc if token.pos_ == "ADJ"),

        # Sentiment
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
    }



print(train_texts.columns)



train_texts['features_1'] = train_texts['file_1'].progress_apply(extract_features)
train_texts['features_2'] = train_texts['file_2'].progress_apply(extract_features)


feature_keys = list(train_texts['features_1'][0].keys())


def dict_to_array(feature_dict):
    return np.array([feature_dict[k] for k in feature_keys])

train_texts['features_1_arr'] = train_texts['features_1'].apply(dict_to_array)
train_texts['features_2_arr'] = train_texts['features_2'].apply(dict_to_array)


def compute_delta(row):
    return np.abs(row['features_1_arr'] - row['features_2_arr'])

train_texts['delta_features'] = train_texts.apply(compute_delta, axis=1)


X = list(train_texts['delta_features'])
y = train_texts['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))



train_texts['label'] = train_texts['label'].map({1: 0, 2: 1})

X = np.vstack(train_texts['delta_features'].values)
y = train_texts['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print(classification_report(y_val, y_pred))


model = SentenceTransformer('all-MiniLM-L6-v2')

emb_1 = model.encode(train_texts['file_1'].tolist(), show_progress_bar=True)
emb_2 = model.encode(train_texts['file_2'].tolist(), show_progress_bar=True)

X = np.concatenate([emb_1, emb_2], axis=1)
y = train_texts['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
speculative_words = {"may", "might", "could", "probably", "possibly", "seems", "appears"}


def extract_classical_features(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    words = [token.text for token in doc if not token.is_punct]
    tokens_count = len(words)
    unique_tokens_count = len(set(words))
    type_token_ratio = unique_tokens_count / tokens_count if tokens_count > 0 else 0
    stopword_count = sum(1 for w in words if w.lower() in stop_words)
    avg_word_len = sum(len(w) for w in words) / tokens_count if tokens_count > 0 else 0
    punct_count = sum(1 for c in text if c in string.punctuation)
    numeric_count = sum(1 for token in doc if token.like_num)
    noun_ratio = len([token for token in doc if token.pos_ == 'NOUN']) / tokens_count if tokens_count > 0 else 0
    verb_ratio = len([token for token in doc if token.pos_ == 'VERB']) / tokens_count if tokens_count > 0 else 0
    adj_ratio = len([token for token in doc if token.pos_ == 'ADJ']) / tokens_count if tokens_count > 0 else 0
    readability = textstat.flesch_reading_ease(text)
    if len(set(tokens)) == 0:
        comp_ratio = 0
    else:
        comp_ratio = len(tokens) / len(set(tokens))
    speculative_count = sum(1 for token in tokens if token in speculative_words)
    ents = [ent.text.lower() for ent in doc.ents]
    ent_counts = Counter(ents)
    if len(ents) == 0:
        ent_repetition = 0
    else:
        repeated_ents = [ent for ent, count in ent_counts.items() if count > 1]
        ent_repetition = len(repeated_ents) / len(ents)


    return {
        'tokens_count': tokens_count,
        'type_token_ratio': type_token_ratio,
        'stopword_count': stopword_count,
        'avg_word_len': avg_word_len,
        'punct_count': punct_count,
        'numeric_count': numeric_count,
        'noun_ratio': noun_ratio,
        'verb_ratio': verb_ratio,
        'adj_ratio': adj_ratio,
        "readability": readability,
        "compression_ratio": comp_ratio,
        "speculative_count": speculative_count,
        "ent_repetition_ratio": ent_repetition
    }


def add_features(train_texts):
    features_1 = train_texts['file_1'].apply(extract_classical_features).apply(pd.Series)
    features_2 = train_texts['file_2'].apply(extract_classical_features).apply(pd.Series)

    
    delta = (features_1 - features_2).abs().add_suffix('_delta')

    
    combined = pd.concat([
        features_1.add_suffix('_1'),
        features_2.add_suffix('_2'),
        delta
    ], axis=1)
    
    return combined

X_features = add_features(train_texts)
y = train_texts['label'].apply(lambda x: 0 if x == 1 else 1) 


xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

param_dist = {
    'n_estimators': [50, 100, 200, 400],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2, 3]
}

search = RandomizedSearchCV(
    xgb, param_distributions=param_dist, n_iter=30,
    scoring='f1_macro', cv=5, verbose=2, random_state=42, n_jobs=-1
)

search.fit(X_features, y)


best_model = search.best_estimator_

y_pred = cross_val_predict(best_model, X_features, y, cv=5)
print("Best parameters:", search.best_params_)
print("\nClassification report:")
print(classification_report(y, y_pred))


importances = best_model.feature_importances_
feat_names = X_features.columns
sorted_idx = np.argsort(importances)[-15:]

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feat_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top 15 Important Features")
plt.show()


test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"


test_data = []
subdirs = sorted(os.listdir(test_dir))

for sub in subdirs:
    sub_path = os.path.join(test_dir, sub)
    if os.path.isdir(sub_path):
        file1_path = os.path.join(sub_path, "file_1.txt")
        file2_path = os.path.join(sub_path, "file_2.txt")
        with open(file1_path, 'r', encoding='utf-8') as f:
            text1 = f.read()
        with open(file2_path, 'r', encoding='utf-8') as f:
            text2 = f.read()
        test_data.append({
            "id": int(sub.replace("article_", "")),
            "real_text_1": text1,
            "real_text_2": text2,
            "file1_path": file1_path,
            "file2_path": file2_path
        })

test_df = pd.DataFrame(test_data)


print(test_df.head)


nlp = spacy.load("en_core_web_sm")



def extract_classical_features(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    words = [token.text for token in doc if not token.is_punct]
    tokens_count = len(words)
    unique_tokens_count = len(set(words))
    type_token_ratio = unique_tokens_count / tokens_count if tokens_count > 0 else 0
    stopword_count = sum(1 for w in words if w.lower() in stop_words)
    avg_word_len = sum(len(w) for w in words) / tokens_count if tokens_count > 0 else 0
    punct_count = sum(1 for c in text if c in string.punctuation)
    numeric_count = sum(1 for token in doc if token.like_num)
    noun_ratio = len([token for token in doc if token.pos_ == 'NOUN']) / tokens_count if tokens_count > 0 else 0
    verb_ratio = len([token for token in doc if token.pos_ == 'VERB']) / tokens_count if tokens_count > 0 else 0
    adj_ratio = len([token for token in doc if token.pos_ == 'ADJ']) / tokens_count if tokens_count > 0 else 0
    readability = textstat.flesch_reading_ease(text)
    if len(set(tokens)) == 0:
        comp_ratio = 0
    else:
        comp_ratio = len(tokens) / len(set(tokens))
    speculative_count = sum(1 for token in tokens if token in speculative_words)
    ents = [ent.text.lower() for ent in doc.ents]
    ent_counts = Counter(ents)
    if len(ents) == 0:
        ent_repetition = 0
    else:
        repeated_ents = [ent for ent, count in ent_counts.items() if count > 1]
        ent_repetition = len(repeated_ents) / len(ents)

    return {
        'tokens_count': tokens_count,
        'type_token_ratio': type_token_ratio,
        'stopword_count': stopword_count,
        'avg_word_len': avg_word_len,
        'punct_count': punct_count,
        'numeric_count': numeric_count,
        'noun_ratio': noun_ratio,
        'verb_ratio': verb_ratio,
        'adj_ratio': adj_ratio,
        "readability": readability,
        "compression_ratio": comp_ratio,
        "speculative_count": speculative_count,
        "ent_repetition_ratio": ent_repetition
    }



features_1 = []
features_2 = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    features_1.append(extract_classical_features(row["real_text_1"]))
    features_2.append(extract_classical_features(row["real_text_2"]))

features_df_1 = pd.DataFrame(features_1)
features_df_2 = pd.DataFrame(features_2)



features_df_1.columns = [f"{col}_1" for col in features_df_1.columns]
features_df_2.columns = [f"{col}_2" for col in features_df_2.columns]


combined_features = pd.concat([features_df_1, features_df_2], axis=1)


delta_features = features_df_1.values - features_df_2.values

delta_df = pd.DataFrame(
    delta_features,
    columns=[f"{col.replace('_1', '')}_delta" for col in features_df_1.columns]
)


X_test = pd.concat([features_df_1, features_df_2, delta_df], axis=1)


test_preds = search.best_estimator_.predict(X_test)


submission_preds = test_preds + 1

submission = pd.DataFrame({
    "id": test_df["id"],
    "real_text_file": submission_preds.astype(int)
})


submission.sort_values("id", inplace=True)
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")

