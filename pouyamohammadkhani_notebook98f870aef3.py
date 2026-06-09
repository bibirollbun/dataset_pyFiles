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


packages = [
    "langchain",
    "langchain_community",
    "langchain-huggingface"
]

for pkg in packages:
    print(f" Installing: {pkg}")
    !pip install -qU {pkg} > /dev/null 2>&1

print("All packages installed successfully.")


# Week 4: Feature Tuning - PCA, Lengths, and Subreddit Features
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import joblib
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier

# Load data
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for df in [train_df, test_df]:
    df['clean_body'] = df['body'].apply(clean_text)
    df['clean_rule'] = df['rule'].apply(clean_text)
    df['clean_subreddit'] = df['subreddit'].apply(clean_text)
    df['combined'] = df['clean_body'] + ' [SEP] ' + df['clean_rule'] + ' [SEP] ' + df['clean_subreddit']
    df['body_len'] = df['clean_body'].apply(len)
    df['rule_len'] = df['clean_rule'].apply(len)
    df['link_count'] = df['body'].apply(lambda x: len(re.findall(r'http[s]?://', str(x))))

# One-hot encode subreddit
subreddit_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
subreddit_encoder.fit(train_df[['clean_subreddit']])
train_sub_ohe = subreddit_encoder.transform(train_df[['clean_subreddit']])
test_sub_ohe = subreddit_encoder.transform(test_df[['clean_subreddit']])

# Split data
X_train_text, X_val_text, y_train, y_val, train_meta, val_meta, train_sub, val_sub = train_test_split(
    train_df['combined'],
    train_df['rule_violation'],
    train_df[['body_len', 'rule_len', 'link_count']],
    train_sub_ohe,
    test_size=0.2, random_state=42, stratify=train_df['rule_violation']
)

# Sentence embedding
model_name = 'all-mpnet-base-v2'
embedder = SentenceTransformer(model_name)
X_train_emb = embedder.encode(X_train_text.tolist(), show_progress_bar=True, convert_to_numpy=True)
X_val_emb = embedder.encode(X_val_text.tolist(), show_progress_bar=True, convert_to_numpy=True)
X_test_emb = embedder.encode(test_df['combined'].tolist(), show_progress_bar=True, convert_to_numpy=True)

# PCA to reduce dimension
pca = PCA(n_components=128, random_state=42)
X_train_pca = pca.fit_transform(X_train_emb)
X_val_pca = pca.transform(X_val_emb)
X_test_pca = pca.transform(X_test_emb)

# Scale meta features
scaler = StandardScaler()
train_meta_scaled = scaler.fit_transform(train_meta)
val_meta_scaled = scaler.transform(val_meta)
test_meta_scaled = scaler.transform(test_df[['body_len', 'rule_len', 'link_count']])

# Concatenate all features
X_train_final = np.hstack([X_train_pca, train_meta_scaled, train_sub])
X_val_final = np.hstack([X_val_pca, val_meta_scaled, val_sub])
X_test_final = np.hstack([X_test_pca, test_meta_scaled, test_sub_ohe])

# LightGBM Classifier
lgb_clf = lgb.LGBMClassifier(n_estimators=300, force_col_wise=True)
lgb_clf.fit(X_train_final, y_train)
val_probs_lgb = lgb_clf.predict_proba(X_val_final)[:, 1]
lgb_auc = roc_auc_score(y_val, val_probs_lgb)
print(f"LightGBM Validation AUC: {lgb_auc:.4f}")

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=300)
gb_clf.fit(X_train_final, y_train)
val_probs_gb = gb_clf.predict_proba(X_val_final)[:, 1]
gb_auc = roc_auc_score(y_val, val_probs_gb)
print(f"GradientBoostingClassifier Validation AUC: {gb_auc:.4f}")

# Choose LightGBM for submission
test_probs = lgb_clf.predict_proba(X_test_final)[:, 1]
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': test_probs
})


submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")




