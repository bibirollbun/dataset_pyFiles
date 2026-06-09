import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

SEED = 42



train = pd.read_csv("/kaggle/input/oilgas-field-prediction/train.csv")
test  = pd.read_csv("/kaggle/input/oilgas-field-prediction/test.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

train.head()



TARGET_COL = "Onshore or offshore"

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL])

test_ids = test["Index"]



y.value_counts(normalize=True)



X.isnull().mean().sort_values(ascending=False).head(10)



DROP_COLS = [
    "Field name",
    "Reservoir unit"
]

X = X.drop(columns=DROP_COLS, errors="ignore")
test = test.drop(columns=DROP_COLS, errors="ignore")



num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)



model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print(f"CV Accuracy: {cv_scores.mean():.4f} Â± {cv_scores.std():.4f}")



pipeline.fit(X, y)

test_preds = pipeline.predict(test)



submission = pd.DataFrame({
    "Index": test_ids,
    "Onshore/Offshore": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()


