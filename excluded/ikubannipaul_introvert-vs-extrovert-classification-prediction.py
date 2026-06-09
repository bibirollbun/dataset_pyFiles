!pip install catboost


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")


!ls /kaggle/input/playground-series-s5e7


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train_df.head()


train_df.columns


le = LabelEncoder()
train_df["Personality_label_encoded"] = le.fit_transform(train_df["Personality"])


y = train_df["Personality_label_encoded"]
X = train_df.drop(columns=["id", "Personality", "Personality_label_encoded"])


X_test = test_df.drop(columns=["id"])


combined_features = pd.concat([X, X_test], axis=0)
cat_cols = combined_features.select_dtypes(include="object").columns.tolist()
oe = OrdinalEncoder()
combined_features[cat_cols] = oe.fit_transform(combined_features[cat_cols])
X = combined_features.iloc[:len(X)].reset_index(drop=True)
X_test = combined_features.iloc[len(X):].reset_index(drop=True)


model = CatBoostClassifier(n_estimators=1000, max_depth=4, learning_rate=0.01, verbose=0, random_state=42)


rskf = RepeatedStratifiedKFold(n_splits=8, n_repeats=5, random_state=42)

meta_features = np.zeros((len(X), 3))
test_predicts = np.zeros((len(X_test), 3))
oof_predicts = np.zeros((len(X)))


for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
  X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
  y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

  model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

  meta_features[val_idx, 0] = model.predict_proba(X_val)[:, 1]
  test_predicts[:, 0] += model.predict_proba(X_test)[:, 1] / 8



from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression()
meta_model.fit(meta_features, y)
oof_meta_predicts = meta_model.predict_proba(meta_features)[:, 1]
test_meta_predicts = meta_model.predict_proba(test_predicts)[:, 1]


from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve

fpr, tpr, thresholds = roc_curve(y, oof_meta_predicts)
opt_thresh = thresholds[np.argmax(tpr - fpr)]


#Evaluating
print(f"LogLoss Score: {log_loss(y, oof_meta_predicts):.6f}")
print(f"Accuracy Score: {accuracy_score(y, oof_meta_predicts > opt_thresh):.6f}")
print(f"Best Threshold Value: {opt_thresh:.4f}")


final_predicts = (test_meta_predicts > opt_thresh).astype(int)
submission_df["Personality"] = le.inverse_transform(final_predicts)
submission_df.to_csv("submission.csv", index=False)
submission_df.head()

