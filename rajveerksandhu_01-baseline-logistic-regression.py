import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

train.head()



train.columns



import pandas as pd
import numpy as np

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

TARGET = "diagnosed_diabetes"
ID_COL = "id"

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("Submission shape:", sub.shape)

# Check target balance
print("\nTarget distribution:")
print(train[TARGET].value_counts(normalize=True))



from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Split X/y
X = train.drop(columns=[TARGET])
y = train[TARGET].astype(int)

# Identify column types
num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

# Make sure ID is not used as feature
if ID_COL in num_cols:
    num_cols.remove(ID_COL)
if ID_COL in cat_cols:
    cat_cols.remove(ID_COL)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

model = LogisticRegression(max_iter=2000)

clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
print("Baseline Logistic Regression ROC-AUC (5-fold CV):")
print("Scores:", np.round(scores, 4))
print("Mean :", round(scores.mean(), 4))
print("Std  :", round(scores.std(), 4))



# Fit on full training data
clf.fit(X, y)

# Predict probability for test set
X_test = test.copy()
test_proba = clf.predict_proba(X_test)[:, 1]

# Create submission
sub[TARGET] = test_proba
sub.to_csv("submission.csv", index=False)

sub.head(), sub.shape





