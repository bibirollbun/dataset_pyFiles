# import package and create enviroments
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

RANDOM_STATE = 42


try:
    base = "/kaggle/input/rmit-hackathon-2025"
    train_df = pd.read_csv(f"{base}/train.csv")
    test_df  = pd.read_csv(f"{base}/test.csv")
    sample_submission_df = pd.read_csv(f"{base}/sample_submission.csv")
except FileNotFoundError:
    train_df = pd.read_csv("train.csv")
    test_df  = pd.read_csv("test.csv")
    sample_submission_df = pd.read_csv("sample_submission.csv")


assert {'text', 'label'}.issubset(train_df.columns), "train.csv phải có cột 'text' và 'label'"
assert 'text' in test_df.columns, "test.csv phải có cột 'text'"

train_df['text'] = train_df['text'].fillna('')
test_df['text']  = test_df['text'].fillna('')

# Map label -> target (jailbreak=1, benign=0)
train_df['target'] = (train_df['label'].astype(str).str.lower() == 'jailbreak').astype(int)

X = train_df['text']
y = train_df['target']
X_test = test_df['text']


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


word_vec = TfidfVectorizer(
    max_features=5000, ngram_range=(1,2),
    stop_words='english', sublinear_tf=True, min_df=2, max_df=0.99
)
char_vec = TfidfVectorizer(
    analyzer='char', ngram_range=(3,5), min_df=2, sublinear_tf=True
)

Xtr_w = word_vec.fit_transform(X_train)
Xva_w = word_vec.transform(X_valid)
Xte_w = word_vec.transform(X_test)

Xtr_c = char_vec.fit_transform(X_train)
Xva_c = char_vec.transform(X_valid)
Xte_c = char_vec.transform(X_test)

Xtr = hstack([Xtr_w, Xtr_c])
Xva = hstack([Xva_w, Xva_c])
Xte = hstack([Xte_w, Xte_c])


clf = LogisticRegression(
    solver='liblinear', C=5, random_state=RANDOM_STATE,
    class_weight='balanced', max_iter=200
)
clf.fit(Xtr, y_train)


val_pred = clf.predict_proba(Xva)[:, 1]
val_auc = roc_auc_score(y_valid, val_pred)
print(f"Validation ROC-AUC: {val_auc:.4f}")


word_vec_full = TfidfVectorizer(
    max_features=5000, ngram_range=(1,2),
    stop_words='english', sublinear_tf=True, min_df=2, max_df=0.99
)
char_vec_full = TfidfVectorizer(
    analyzer='char', ngram_range=(3,5), min_df=2, sublinear_tf=True
)

Xtr_full_w = word_vec_full.fit_transform(X)
Xte_full_w = word_vec_full.transform(X_test)
Xtr_full_c = char_vec_full.fit_transform(X)
Xte_full_c = char_vec_full.transform(X_test)

Xtr_full = hstack([Xtr_full_w, Xtr_full_c])
Xte_full = hstack([Xte_full_w, Xte_full_c])

clf_full = LogisticRegression(
    solver='liblinear', C=5, random_state=RANDOM_STATE,
    class_weight='balanced', max_iter=200
)
clf_full.fit(Xtr_full, y)


test_pred = clf_full.predict_proba(Xte_full)[:, 1]

id_col = sample_submission_df.columns[0]
submission_df = pd.DataFrame({
    id_col: sample_submission_df[id_col],
    'target': test_pred
})

submission_df.to_csv('submission.csv', index=False)
print("submission.csv created successfully!")
print(submission_df.head())

