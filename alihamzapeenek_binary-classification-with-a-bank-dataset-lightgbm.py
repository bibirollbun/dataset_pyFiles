import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier



# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

print(train.shape, test.shape)
train.head()



print(train.dtypes)
print(train['y'].value_counts(normalize=True))



# Drop id from both train and test right here
X = train.drop(columns=["y", "id"])
y = train["y"]

X_test = test.drop(columns=["id"])
test_ids = test["id"]



# Identify categorical vs numerical
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Simple pipeline
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])



# Model: simple LightGBM
model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", model)
])



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in cv.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict_proba(X_val)[:,1]
    score = roc_auc_score(y_val, preds)
    scores.append(score)

print("CV AUC scores:", scores)
print("Mean AUC:", np.mean(scores))



# Fit on full data
pipe.fit(X, y)

# Predict probabilities for test set
test_preds = pipe.predict_proba(X_test)[:,1]

# Create submission
submission = pd.DataFrame({
    "id": test_ids,
    "y": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()





