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


train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sample_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")


# === ãƒ©ã‚¤ãƒ–ãƒ©ãƒª ===
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib  # ãƒ¢ãƒ‡ãƒ«ä¿�å­˜ç”¨

# === ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿ ===
train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")

# === å‰�å‡¦ç�† ===
def clean_train_data(df):
    import re
    def is_garbled(text):
        return bool(re.search(r'[^\x00-\x7F]+', text)) and len(text) < 10

    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df = df[~df['comment_text'].apply(is_garbled)]
    df = df.dropna(subset=target_cols)
    for col in target_cols:
        df = df[df[col].isin([0, 1])]
    return df

train_df_cleaned = clean_train_data(train_df.copy())

# === ç‰¹å¾´é‡�ä½œæˆ� ===
X_text = train_df_cleaned['comment_text']
y = train_df_cleaned[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,2),
    min_df=3,
    sublinear_tf=True
)

X_tfidf = tfidf.fit_transform(X_text)

# === TF-IDF ã‚’ä¿�å­˜ ===
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# === å�„ãƒ©ãƒ™ãƒ«ã�«å¯¾ã�—ã�¦ LightGBM ãƒ¢ãƒ‡ãƒ«ã‚’å­¦ç¿’ã�—ã€�ä¿�å­˜ ===
for col in y.columns:
    print(f'ğŸ”„ Training label: {col}')
    X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y[col], test_size=0.2, random_state=42)

    d_train = lgb.Dataset(X_train, label=y_train)
    d_val = lgb.Dataset(X_val, label=y_val)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42
    }

    model = lgb.train(
        params,
        d_train,
        valid_sets=[d_val],
        valid_names=['valid'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    # ãƒ¢ãƒ‡ãƒ«ä¿�å­˜
    joblib.dump(model, f'lgb_model_{col}.pkl')

print("âœ… ãƒ¢ãƒ‡ãƒ«ã�¨TF-IDFã‚’ã�™ã�¹ã�¦ä¿�å­˜ã�—ã�¾ã�—ã�Ÿï¼�")


