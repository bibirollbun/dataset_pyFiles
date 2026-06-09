import pandas as pd
import numpy as  np
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna


import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original_df = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


num_col = [col for col in train_df.select_dtypes(['int64']).columns if col!='id']
cat_col = [col for col in train_df.select_dtypes(['object']).columns if col!='Fertilizer Name']
target_col = "Fertilizer Name"
id_col = 'id'


combined_df = pd.concat([train_df.drop(columns=id_col, axis=1),original_df], ignore_index=True).drop_duplicates()


combined_df.info()


le = LabelEncoder()
tgt_le = LabelEncoder()


for col in cat_col:
    combined_df[col] = le.fit_transform(combined_df[col])
combined_df[target_col] = tgt_le.fit_transform(combined_df[target_col])



for col in cat_col:
    test_df[col] = le.fit_transform(test_df[col])


def mapk(actual, predicted, k=3, ignore_empty_actual=True):
    def apk(a, p, k):
        if not a:
            return 0.0 if not ignore_empty_actual else None
        p = p[:k]
        score = 0.0
        num_hits = 0.0
        used = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in used:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
                used.add(pred)
        return score / min(len(a), k)

    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same.")

    scores = []
    for a, p in zip(actual, predicted):
        ap = apk(a, p, k)
        if ap is not None:
            scores.append(ap)
    return np.mean(scores) if scores else 0.0


X = combined_df[num_col+cat_col]
y = combined_df[target_col]


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


oof_preds = np.zeros((X.shape[0], len(np.unique(y))))  # Shape based on X and unique classes in y
actual_labels = np.zeros(X.shape[0])

xgb_model = XGBClassifier(
    objective='multi:softprob',
    n_estimators=1000,
    max_depth=9,
    learning_rate=0.2,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=1,
    reg_lambda=10,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    device = "cuda"
)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]  # Use .iloc if X is a DataFrame
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]  # Use .iloc if y is a Series

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Predict probabilities on the validation set
    oof_preds[valid_idx] = xgb_model.predict_proba(X_valid)
    actual_labels[valid_idx] = y_valid

    # Optionally, evaluate performance on the validation set using MAP@K
    top_k_preds = np.argsort(oof_preds[valid_idx], axis=1)[:, -5:][:, ::-1]
    mapk_score = mapk([[label] for label in y_valid], top_k_preds.tolist(), k=5)
    print(f"Fold {fold + 1} MAP@5 score: {mapk_score:.4f}")


top_k_preds = np.argsort(oof_preds, axis=1)[:, -5:][:, ::-1]
overall_mapk_score = mapk([[label] for label in actual_labels], top_k_preds.tolist(), k=5)
print(f"Overall OOF MAP@5 score: {overall_mapk_score:.4f}")


pred_final = xgb_model.predict_proba(test_df[X.columns])
top_3_preds = np.argsort(pred_final, axis=1)[:, ::-1][:, :3]


sub = pd.DataFrame({
    'id': submission_df['id'],
    'Fertilizer Name': [' '.join([str(tgt_le.classes_[int(i)]) for i in row]) for row in top_3_preds]
})



sub.to_csv('/kaggle/working/submission.csv', index=False)




