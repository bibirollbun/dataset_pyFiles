import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')
train['text_len'] = train['text'].str.len()

plt.figure(figsize=(6,4))
plt.hist(train['text_len'], bins=30)
plt.title('Essay Length Distribution')
plt.xlabel('Length (chars)')        # ← 영문
plt.ylabel('Number of essays')      # ← 영문
plt.show()




import pandas as pd, seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')
label_cnt = train['generated'].value_counts()

plt.figure(figsize=(4,3))
sns.barplot(x=label_cnt.index, y=label_cnt.values)
plt.title('Label Distribution')
plt.xlabel('generated (0 = Human, 1 = AI)')   # 영문
plt.ylabel('Number of essays')               # 영문
plt.show()


import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# === 데이터 불러오기 ===
train = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X = tfidf.fit_transform(train['text'])
y = train['generated'].values

# === 1. 3‑fold로 threshold 찾기 ===
oof_prob = np.zeros(len(train))
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)   # ★ 3‑split

for tr_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr = y[tr_idx]

    clf = LogisticRegression(class_weight='balanced', max_iter=2000)
    clf.fit(X_tr, y_tr)
    oof_prob[val_idx] = clf.predict_proba(X_val)[:, 1]

ths = np.linspace(0.05, 0.95, 19)
scores = [f1_score(y, (oof_prob > t).astype(int)) for t in ths]
best_t = ths[np.argmax(scores)]
print('best F1', max(scores), 'at threshold', best_t)

# === 2. 전체 데이터로 재학습 ===
clf_full = LogisticRegression(class_weight='balanced', max_iter=2000)
clf_full.fit(X, y)

test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
X_test = tfidf.transform(test['text'])
test_prob = clf_full.predict_proba(X_test)[:, 1]

# 3. 확률 그대로 저장 (threshold 사용 X)
sub = pd.DataFrame({'id': test['id'], 'generated': test_prob})
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('Saved!', sub.shape, 'threshold', best_t)





