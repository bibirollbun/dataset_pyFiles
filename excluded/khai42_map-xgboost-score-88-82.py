import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import xgboost as xgb


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")


train['Misconception'] = train['Misconception'].fillna("NA")
train['target_cat'] = train['Category'] + ":" + train['Misconception']

vc = train['target_cat'].value_counts()
valid_labels = vc[vc > 1].index
train = train[train['target_cat'].isin(valid_labels)].copy()

le = LabelEncoder()
train['encoded_label'] = le.fit_transform(train['target_cat'])
y = train['encoded_label'].values
num_class = len(le.classes_)

def basic_clean(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())

train['text'] = (train['QuestionText'] + " " + train['MC_Answer'] + " " + train['StudentExplanation']).apply(basic_clean)
test['text'] = (test['QuestionText'] + " " + test['MC_Answer'] + " " + test['StudentExplanation']).apply(basic_clean)

def extract_math_features(df):
    df_feat = pd.DataFrame()
    df_feat["has_frac"] = df["StudentExplanation"].str.contains(r'\d+\s*/\s*\d+').fillna(False).astype(int)
    df_feat["has_decimal"] = df["StudentExplanation"].str.contains(r'\d+\.\d+').fillna(False).astype(int)
    df_feat["has_percent"] = df["StudentExplanation"].str.contains('%').fillna(False).astype(int)
    df_feat["has_number"] = df["StudentExplanation"].str.contains(r'\d').fillna(False).astype(int)
    df_feat["explanation_len"] = df["StudentExplanation"].fillna('').apply(lambda x: len(str(x)))
    return df_feat

X_train_math = extract_math_features(train)
X_test_math = extract_math_features(test)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_text = tfidf.fit_transform(train['text'])
X_test_text = tfidf.transform(test['text'])

X_train = hstack([X_train_text, X_train_math])
X_test = hstack([X_test_text, X_test_math])
X_train = X_train.tocsr()
X_test = X_test.tocsr()

def map3(y_true, y_pred_proba):
    top3 = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :3]
    score = 0.0
    for i, t in enumerate(y_true):
        if t == top3[i][0]: score += 1.0
        elif t == top3[i][1]: score += 1.0 / 2
        elif t == top3[i][2]: score += 1.0 / 3
    return score / len(y_true)

seeds = [42, 2023, 2024]
pred_test_total = np.zeros((X_test.shape[0], num_class))
map3_scores = []

for seed in seeds:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    pred_valid = np.zeros((X_train.shape[0], num_class))
    pred_test = np.zeros((X_test.shape[0], num_class))

    for train_idx, val_idx in skf.split(X_train, y):
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_class,
            use_label_encoder=False,
            eval_metric='mlogloss',
            tree_method='gpu_hist',          
            predictor='gpu_predictor',       
            learning_rate=0.1,
            max_depth=6,
            n_estimators=200,
            verbosity=0,
            random_state=seed
        )
        model.fit(X_train[train_idx], y[train_idx])
        pred_valid[val_idx] = model.predict_proba(X_train[val_idx])
        pred_test += model.predict_proba(X_test) / skf.n_splits

    score = map3(y, pred_valid)
    map3_scores.append(score)
    pred_test_total += pred_test / len(seeds)

print("MAP Score:", np.mean(map3_scores))


top3 = np.argsort(-pred_test_total, axis=1)[:, :3]
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)
joined_preds = [" ".join(row) for row in top3_labels]

submission = pd.DataFrame({
    "row_id": test["row_id"],
    "Category:Misconception": joined_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv")




