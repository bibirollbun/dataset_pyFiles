import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error



import os

print(os.listdir("/kaggle/input"))



import pandas as pd

train = pd.read_csv("/kaggle/input/nst-x-contest-2-abalone-regression/train.csv")
test = pd.read_csv("/kaggle/input/nst-x-contest-2-abalone-regression/test.csv")
submission = pd.read_csv("/kaggle/input/nst-x-contest-2-abalone-regression/sample_submission.csv")

train.head()



display(train.describe(include="all"))
print("\nShape:", train.shape)
print("\nMissing values:\n", train.isna().sum())
print("\nColumns:", list(train.columns))



TARGET = "Rings"  # as typical for Abalone

# Some datasets include an 'id' or 'Id' column — drop it if present
ID_COLS = [c for c in ["id","Id","ID"] if c in train.columns]

# Identify categorical & numerical columns
cat_cols = [c for c in train.columns if train[c].dtype == "object" and c != TARGET]
num_cols = [c for c in train.columns if c not in cat_cols + [TARGET] + ID_COLS]

X = train.drop([TARGET] + ID_COLS, axis=1)
y = train[TARGET]
X_test = test.drop(ID_COLS, axis=1)

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

cat_cols, num_cols



# Ensure the right imports are present (safe to re-run even if already imported)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Handle sklearn version differences: use sparse_output if available, else fall back to sparse
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_cols),  # keep compatible with sparse pipeline
        ("cat", ohe, cat_cols),
    ],
    remainder="drop"
)



from sklearn.model_selection import KFold, cross_val_score

RANDOM_STATE = 42
N_FOLDS = 5

def cv_rmse(model, X, y, cv=N_FOLDS):
    cv = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, X, y,
        scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1
    )
    return -scores.mean(), -scores.std()



# make Cell 7 self-contained
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso

models = {
    "Linear": Pipeline([("prep", preprocess), ("mdl", LinearRegression())]),
    "Ridge":  Pipeline([("prep", preprocess), ("mdl", Ridge(alpha=1.0))]),  # no random_state for Ridge
    "Lasso":  Pipeline([("prep", preprocess), ("mdl", Lasso(alpha=0.001, random_state=42, max_iter=20000))]),
}

results = {}
for name, pipe in models.items():
    mean_rmse, std_rmse = cv_rmse(pipe, X, y)
    results[name] = mean_rmse
    print(f"{name}: CV RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")

pd.Series(results).sort_values()



from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# Recreate Ridge model
quick_model = Pipeline([("prep", preprocess), ("mdl", Ridge(alpha=1.0))])
quick_model.fit(X, y)

# Predict
preds = quick_model.predict(X_test)
preds = np.clip(preds, 0, None)  # ensure no negatives

# Build submission from scratch (using test IDs if present)
if "id" in test.columns:
    submission = pd.DataFrame({"id": test["id"], "Rings": preds})
else:
    submission = pd.DataFrame({"Rings": preds})

# Save
submission.to_csv("submission.csv", index=False)
submission.head()


