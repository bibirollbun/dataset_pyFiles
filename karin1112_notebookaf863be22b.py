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


# 必要なライブラリのインポート
import pandas as pd
import numpy as np
import re
import string
import gc
import zipfile
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# --- 1. データの読み込みと解凍 ---
print("Loading and extracting data...")

# Kaggleのデータセットパス
data_base_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/'
extract_path = '/kaggle/working/'

# ZIPファイルを解凍する関数
def extract_zip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print(f"Extracted {zip_file_path.split('/')[-1]} to {extract_to_path}")

try:
    # 各ZIPファイルを解凍
    extract_zip_file(os.path.join(data_base_path, 'train.csv.zip'), extract_path)
    extract_zip_file(os.path.join(data_base_path, 'test.csv.zip'), extract_path)
    extract_zip_file(os.path.join(data_base_path, 'sample_submission.csv.zip'), extract_path)

    # CSV読み込み
    train_df = pd.read_csv(os.path.join(extract_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(extract_path, 'test.csv'))
    sample_submission = pd.read_csv(os.path.join(extract_path, 'sample_submission.csv'))

    print("Data loaded successfully after extraction.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print(">>> Please ensure the dataset is added to your Kaggle Notebook and the path is correct. <<<")
    exit()

except Exception as e:
    print(f"An unexpected error occurred during data loading or extraction: {e}")
    exit()

# データが正常に読み込まれた場合のみ以降を実行
if train_df is None or test_df is None or sample_submission is None:
    print("Data loading failed. Exiting script.")
else:
    CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # --- 2. テキスト前処理関数 ---
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    print("Applying text cleaning to comments...")
    train_df['comment_text'] = train_df['comment_text'].apply(clean_text)
    test_df['comment_text'] = test_df['comment_text'].apply(clean_text)
    print("Text cleaning complete.")

    # --- 3. 特徴量エンジニアリング (TF-IDF) ---
    print("Concatenating all comments for TF-IDF vectorization...")
    all_comments = pd.concat([train_df['comment_text'], test_df['comment_text']], axis=0)

    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words='english',
        max_features=50000
    )
    X_all = vectorizer.fit_transform(all_comments)

    X_train = X_all[:len(train_df)]
    X_test = X_all[len(train_df):]
    print(f"TF-IDF Vectorization complete. Number of features: {X_train.shape[1]}")

    del all_comments
    gc.collect()

    # --- 4. モデル構築と学習 ---
    classifier = OneVsRestClassifier(
        LogisticRegression(solver='sag', n_jobs=-1, max_iter=1000, random_state=42)
    )

    print("Training models for each category...")
    predictions = pd.DataFrame({'id': test_df['id']})

    for category in CATEGORIES:
        print(f"  Training model for category: {category}...")
        y_train_category = train_df[category]
        classifier.fit(X_train, y_train_category)
        predictions[category] = classifier.predict_proba(X_test)[:, 1]
        print(f"  Finished training for {category}.")

    print("All models trained and predictions generated.")

    # --- 5. 提出ファイルの保存 ---
    predictions.to_csv('submission.csv', index=False)
    print("\n--- Submission file 'submission.csv' created successfully! ---")
    print("You can now download this file and submit it to Kaggle.")

