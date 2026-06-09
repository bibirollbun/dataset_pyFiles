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


train_df


test_df


sample_df


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ã‚°ãƒ©ãƒ•ã�®ã‚¹ã‚¿ã‚¤ãƒ«ã‚’è¨­å®š
sns.set(style="whitegrid", font_scale=1.1)


# æ•°å€¤å¤‰æ•°ã�®ãƒ’ã‚¹ãƒˆã‚°ãƒ©ãƒ 
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

train_df[numeric_cols].hist(bins=20, figsize=(20, 20))
plt.show()


plt.figure(figsize=(12, 10))
sns.heatmap(train_df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()


train_df.dtypes



# ã‚«ãƒ©ãƒ ã�®ä¸­ã�«å…¥ã�£ã�¦ã�„ã‚‹ãƒ‡ãƒ¼ã‚¿ã‚’è¦‹ã‚‹
train_df['toxic'].value_counts()


pip install lightgbm scikit-learn pandas



# ãƒ‡ãƒ¼ã‚¿ã�®å‰�å‡¦ç�†

def clean_train_data(df):
    import re
    def is_garbled(text):
        return bool(re.search(r'[^\x00-\x7F]+', text)) and len(text) < 10

    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # æ–‡å­—åŒ–ã�‘é™¤å�»
    df = df[~df['comment_text'].apply(is_garbled)]

    # ãƒ©ãƒ™ãƒ«æ¬ æ��é™¤å�»
    df = df.dropna(subset=target_cols)

    # ãƒ©ãƒ™ãƒ«ã�«ä¸�æ­£å€¤ã�Œå…¥ã�£ã�¦ã�„ã‚‹è¡Œã�®é™¤å�»ï¼ˆ0 or 1ã�®ã�¿ï¼‰
    for col in target_cols:
        df = df[df[col].isin([0, 1])]
    
    return df



train_df_cleaned = clean_train_data(train_df.copy())



print("ğŸ”� ãƒ‡ãƒ¼ã‚¿ã�®å…ˆé ­5è¡Œ")
display(train_df_cleaned.head())

print("ğŸ”� ãƒ‡ãƒ¼ã‚¿ã�®æƒ…å ±")
train_df_cleaned.info()

print("ğŸ”� ãƒ©ãƒ™ãƒ«ã�”ã�¨ã�®å‡ºç�¾æ•°")
print(train_df_cleaned[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum())



import seaborn as sns
import matplotlib.pyplot as plt

target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# ç›¸é–¢è¡Œåˆ—ã‚’è¨ˆç®—
corr = train_df_cleaned[target_cols].corr()

# ãƒ’ãƒ¼ãƒˆãƒ�ãƒƒãƒ—ã�§å�¯è¦–åŒ–
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation between toxic comment labels")
plt.show()


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np

# --- 2. ãƒ©ãƒ™ãƒ«ã�¨ãƒ†ã‚­ã‚¹ãƒˆåˆ†å‰² ---
X_text = train_df_cleaned['comment_text']
y = train_df_cleaned[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# --- 3. TF-IDFå¤‰æ�› ---
tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,2),
    min_df=3,
    sublinear_tf=True
)

X_tfidf = tfidf.fit_transform(X_text)
X_test_tfidf = tfidf.transform(test_df['comment_text'])

# --- 4. LightGBMãƒ¢ãƒ‡ãƒ«ã�§å�„ãƒ©ãƒ™ãƒ«ã‚’å€‹åˆ¥ã�«å­¦ç¿’ ---
submission = pd.DataFrame({'id': test_df['id']})
for col in y.columns:
    print(f'ğŸ”„ Training label: {col}')
    
    # å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ï¼ˆ8:2ã�§åˆ†å‰²ã�—ã�¦early stoppingå�¯ï¼‰
    X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y[col], test_size=0.2, random_state=42)
    
    # LightGBM Datasetå½¢å¼�ã�«å¤‰æ�›
    d_train = lgb.Dataset(X_train, label=y_train)
    d_val = lgb.Dataset(X_val, label=y_val)
    
    # ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿
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
    
    # å­¦ç¿’
    model = lgb.train(
        params,
        d_train,
        valid_sets=[d_train, d_val],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�§äºˆæ¸¬
    y_pred = model.predict(X_test_tfidf, num_iteration=model.best_iteration)
    submission[col] = y_pred



import joblib

# ãƒ¢ãƒ‡ãƒ«ã‚’ä¿�å­˜ï¼ˆãƒ©ãƒ™ãƒ«ã�”ã�¨ï¼‰
for col in y.columns:
    ...
    joblib.dump(model, f'model_{col}.pkl')  # ãƒ¢ãƒ‡ãƒ«ã‚’ãƒ•ã‚¡ã‚¤ãƒ«ã�«ä¿�å­˜



joblib.dump(tfidf.vocabulary_, 'tfidf_vocab.pkl')



# --- 5. æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«å‡ºåŠ› ---
submission.to_csv('submission_lgb_tfidf.csv', index=False)
print("âœ… LightGBM + TF-IDF ãƒ¢ãƒ‡ãƒ«ã�®æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ« submission_lgb_tfidf.csv ã‚’å‡ºåŠ›ã�—ã�¾ã�—ã�Ÿï¼�")

