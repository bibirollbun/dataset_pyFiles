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

train = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sample_submission = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")
test_labels = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



# let's count each category
label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
category_counts = train[label_cols].sum().sort_values(ascending=False)
print(category_counts)



# Let's visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Number of Comments per Toxicity Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()



# Let's see if there are any comment belongs to more than one category
train['num_labels'] = train[label_cols].sum(axis=1)
multi_label_counts = train['num_labels'].value_counts().sort_index()

print(multi_label_counts)



import random

for label in label_cols:
    sample_row = train[train[label] == 1].sample(1, random_state=random.randint(1,100))
    print(f"ðŸ”¹ Category: {label}")
    print(sample_row['comment_text'].values[0])
    print("-" * 100)



samples = []
for label in label_cols:
    text = train[train[label]==1].sample(1, random_state=random.randint(1,100))['comment_text'].values[0]
    samples.append({'Category': label, 'Example Comment': text})

pd.DataFrame(samples)



import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop = set(stopwords.words('english'))
negations = {'not', 'no', 'nor', 'never', "n't"}
stop = stop - negations

contraction_map = [
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "cannot"),
    (r"n['â€™]t\b", " not"),  # covers don't, isn't, etc.
    (r"ain['â€™]t\b", "is not"),
]

def expand_contractions(text):
    for pattern, repl in contraction_map:
        text = re.sub(pattern, repl, text)
    return text

def clean_text_smart(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)

    # expand contractions first
    text = expand_contractions(text)

    # keep censored words and punctuation
    text = re.sub(r"[^a-z*'\s!?]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()

    # remove stopwords but keep negations
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)



# let's see some examples
examples = [
    "You are a f***ing idiot!",
    "I will never insult you again.",
    "This is not good at all!",
    "You're such an a**hole.",
    "Go to hell!!!",
    "WTF are you doing?",
    "I can't believe you said that!",
    "I won't go"
]

for sentence in examples:
    print(f"Original: {sentence}")
    print(f"Cleaned : {clean_text_smart(sentence)}")
    print("-" * 60)



train['clean_comment'] = train['comment_text'].apply(clean_text_smart)
test['clean_comment'] = test['comment_text'].apply(clean_text_smart)



# let's count the ?, !, *
# Count * ? ! in each comment
train['count_star'] = train['comment_text'].apply(lambda x: x.count('*'))
train['count_question'] = train['comment_text'].apply(lambda x: x.count('?'))
train['count_exclamation'] = train['comment_text'].apply(lambda x: x.count('!'))

# Optional: sum of all three as an extra feature
train['count_punctuation'] = train['count_star'] + train['count_question'] + train['count_exclamation']



# see how each punctuation affect on category
def punctuation_summary(df, label_cols):
    """
    Returns a summary table of counts of *, ?, ! per category and for non-toxic comments.
    """
    summary = []

    # For each toxicity category
    for cat in label_cols:
        sub = df[df[cat] == 1]
        summary.append({
            'category': cat,
            'star': (sub['comment_text'].str.count('\*') > 0).sum(),
            'question': (sub['comment_text'].str.count('\?') > 0).sum(),
            'exclamation': (sub['comment_text'].str.count('!') > 0).sum()
        })

    # For non-toxic comments
    non_toxic = df[df['num_labels'] == 0]
    summary.append({
        'category': 'non_toxic',
        'star': (non_toxic['comment_text'].str.count('\*') > 0).sum(),
        'question': (non_toxic['comment_text'].str.count('\?') > 0).sum(),
        'exclamation': (non_toxic['comment_text'].str.count('!') > 0).sum()
    })

    return pd.DataFrame(summary)

# Usage
punct_summary = punctuation_summary(train, label_cols)
print(punct_summary)



def punctuation_summary_with_percent(df, label_cols):
    """
    Returns a summary table of counts and percentages of *, ?, ! per category 
    and for non-toxic comments.
    """
    summary = []

    for cat in label_cols:
        sub = df[df[cat] == 1]
        total = len(sub)
        summary.append({
            'category': cat,
            'total_comments': total,
            'star_count': (sub['comment_text'].str.count('\*') > 0).sum(),
            'question_count': (sub['comment_text'].str.count('\?') > 0).sum(),
            'exclamation_count': (sub['comment_text'].str.count('!') > 0).sum()
        })

    # For non-toxic comments
    non_toxic = df[df['num_labels'] == 0]
    total = len(non_toxic)
    summary.append({
        'category': 'non_toxic',
        'total_comments': total,
        'star_count': (non_toxic['comment_text'].str.count('\*') > 0).sum(),
        'question_count': (non_toxic['comment_text'].str.count('\?') > 0).sum(),
        'exclamation_count': (non_toxic['comment_text'].str.count('!') > 0).sum()
    })

    # Create DataFrame
    summary_df = pd.DataFrame(summary)

    # Calculate percentages
    for punc in ['star', 'question', 'exclamation']:
        summary_df[f'{punc}_percent'] = (summary_df[f'{punc}_count'] / summary_df['total_comments'] * 100).round(2)

    return summary_df


# Usage
punct_summary = punctuation_summary_with_percent(train, label_cols)
print(punct_summary)



# Let's modify the features 
for p in ['count_star', 'count_question', 'count_exclamation']:
    train[p] = np.log1p(train[p])  # log(1 + x)

train['word_count'] = train['comment_text'].apply(lambda x: len(x.split()))
train['char_count'] = train['comment_text'].apply(len)
train['avg_word_len'] = train['char_count'] / train['word_count']




print(train.columns)


# Label columns
label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
y = train[label_cols]



# Numeric features
num_features = train[['count_star','count_question','count_exclamation',
                      'word_count','char_count','avg_word_len']].values


# TF-IDF for feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(train['clean_comment'])


# Combine
X = hstack([X_tfidf, num_features])


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score

# Train
logreg = MultiOutputClassifier(LogisticRegression(max_iter=2000, n_jobs=-1))
logreg.fit(X_train, y_train)

# Predict
y_pred = logreg.predict_proba(X_valid)
y_pred = np.array([p[:,1] for p in y_pred]).T  # take prob of class 1 for each label

# Evaluate
auc_scores = [roc_auc_score(y_valid[col], y_pred[:,i]) for i, col in enumerate(y.columns)]
for col, auc in zip(y.columns, auc_scores):
    print(f"{col:15s} AUC: {auc:.4f}")

print("Mean AUC:", np.mean(auc_scores))



"""import lightgbm as lgb
from sklearn.metrics import roc_auc_score

params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'verbose': -1,
    'n_jobs': -1
}

models = {}
auc_scores = []

# store predictions for all labels
y_pred_lgb_all = []

for col in y.columns:
    print(f"Training for {col}...")
    lgb_train = lgb.Dataset(X_train, label=y_train[col])
    lgb_valid = lgb.Dataset(X_valid, label=y_valid[col], reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_valid],
        num_boost_round=300,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(50)
        ]
    )

    # predict for this label
    preds = model.predict(X_valid, num_iteration=model.best_iteration)
    y_pred_lgb_all.append(preds)

    auc = roc_auc_score(y_valid[col], preds)
    print(f"  {col:15s} AUC: {auc:.4f}")

# stack all predictions into shape (n_samples, n_labels)
y_pred_lgb = np.vstack(y_pred_lgb_all).T
"""


"""import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_ensemble(alpha):
    blended = alpha * y_pred_lgb + (1 - alpha) * y_pred
    aucs = [roc_auc_score(y_valid[col], blended[:, i]) for i, col in enumerate(y.columns)]
    mean_auc = np.mean(aucs)
    return mean_auc

best_auc = 0
best_alpha = 0
for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    mean_auc = evaluate_ensemble(a)
    print(f"alpha={a:.1f}  Mean AUC: {mean_auc:.5f}")
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_alpha = a

print(f"\nBest alpha: {best_alpha}, Best mean AUC: {best_auc:.5f}")"""



test.columns


# Recompute numeric features safely
test['count_star'] = test['clean_comment'].str.count('\*')
test['count_question'] = test['clean_comment'].str.count('\?')
test['count_exclamation'] = test['clean_comment'].str.count('!')
test['word_count'] = test['clean_comment'].apply(lambda x: len(x.split()))
test['char_count'] = test['clean_comment'].apply(len)
test['avg_word_len'] = test['char_count'] / test['word_count'].replace(0, np.nan)

# Fill NaNs (e.g., when word_count = 0)
test.fillna(0, inplace=True)



# âœ… 1. Define numeric feature names
num_features = ['count_star', 'count_question', 'count_exclamation',
                'word_count', 'char_count', 'avg_word_len']

# âœ… 2. Transform test text using the SAME TF-IDF vectorizer used in training
X_test_tfidf = tfidf.transform(test['clean_comment'])

# âœ… 3. Extract numeric features as NumPy array
X_test_num = test[num_features].values

# âœ… 4. Combine TF-IDF and numeric features
from scipy.sparse import hstack
X_test_final = hstack([X_test_tfidf, X_test_num])



# Predict probabilities for test data
y_test_pred = logreg.predict_proba(X_test_final)
y_test_pred = np.array([p[:, 1] for p in y_test_pred]).T  # take prob of class 1



import pandas as pd

# Create submission DataFrame
submission = pd.DataFrame(y_test_pred, columns=y.columns)
submission.insert(0, 'id', test['id'])

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file saved as submission.csv")
submission.head()





