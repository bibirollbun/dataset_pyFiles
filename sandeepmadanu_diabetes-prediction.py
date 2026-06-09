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


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier



import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42


train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path  = "/kaggle/input/playground-series-s5e12/test.csv"


train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)


train.head(5)


print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.head().head())


TARGET_COL = "diagnosed_diabetes"
ID_COL     = "id"


y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, ID_COL])

test_ids = test[ID_COL]
X_test   = test.drop(columns=[ID_COL])


# Columns by type
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)


# Use stratify so class balance is preserved in both sets.
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


print("Train split:", X_train.shape, "Valid split:", X_valid.shape)


# For tree-based models, we do NOT need scaling for numeric features.
# We use OrdinalEncoder for categoricals to keep things efficient.
# (OneHot + dense for 700k rows might be too heavy in memory.)

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            categorical_cols
        ),
    ]
)



# HistGradientBoostingClassifier is like a LightGBM-style model built into sklearn.
# We start with some reasonable defaults; then we will tune around them.

def make_model(
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=40,
    max_iter=300,
    l2_regularization=0.1
):
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_iter=max_iter,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    )


# We manually loop over a small grid (so it's easy to understand).
# You can expand this grid later if you want more tuning.

param_grid = {
    "learning_rate":  [0.03, 0.05, 0.08],
    "max_depth":      [4, 6, 8],
    "min_samples_leaf": [20, 40, 80],
    "l2_regularization": [0.0, 0.1, 0.5],
    "max_iter":       [200, 300]
}

best_auc = -np.inf
best_params = None
best_pipeline = None



# We'll transform the data once outside the loop to save time
# Because ColumnTransformer may be somewhat expensive.
X_train_proc = preprocess.fit_transform(X_train)
X_valid_proc = preprocess.transform(X_valid)

print("\nStarting hyperparameter search...\n")


from itertools import product

for lr, depth, min_leaf, l2, max_iter in product(
    param_grid["learning_rate"],
    param_grid["max_depth"],
    param_grid["min_samples_leaf"],
    param_grid["l2_regularization"],
    param_grid["max_iter"]
):
    model = HistGradientBoostingClassifier(
        learning_rate=lr,
        max_depth=depth,
        min_samples_leaf=min_leaf,
        max_iter=max_iter,
        l2_regularization=l2,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    )



    # Fit on preprocessed data
    model.fit(X_train_proc, y_train)


    # Predict on validation
    valid_pred_proba = model.predict_proba(X_valid_proc)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred_proba)

    print(
        f"lr={lr}, depth={depth}, leaf={min_leaf}, l2={l2}, "
        f"max_iter={max_iter} --> AUC = {auc:.5f}"
    )

    # Track best combination
    if auc > best_auc:
        best_auc = auc
        best_params = {
            "learning_rate": lr,
            "max_depth": depth,
            "min_samples_leaf": min_leaf,
            "max_iter": max_iter,
            "l2_regularization": l2
        }
        best_pipeline = model

print("\nBest AUC on validation:", best_auc)
print("Best params:", best_params)



# Now we refit preprocess + model on *all* training data for final prediction.

print("\nTraining best model on full training data...\n")


# Rebuild preprocess from scratch on all data
preprocess_full = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            categorical_cols
        ),
    ]
)



# Build a pipeline with best parameters
final_model = HistGradientBoostingClassifier(
    learning_rate      = best_params["learning_rate"],
    max_depth          = best_params["max_depth"],
    min_samples_leaf   = best_params["min_samples_leaf"],
    max_iter           = best_params["max_iter"],
    l2_regularization  = best_params["l2_regularization"],
    early_stopping     = False,  # We already tuned; now train fully
    random_state       = RANDOM_STATE
)

final_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess_full),
        ("model", final_model),
    ]
)



# Fit on all training data
final_pipeline.fit(X, y)


# (Optional) check AUC using the earlier valid split to get a sense of performance:
valid_pred_proba_full = final_pipeline.predict_proba(X_valid)[:, 1]
valid_auc_full = roc_auc_score(y_valid, valid_pred_proba_full)
print("Validation AUC with final pipeline:", valid_auc_full)


test_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_pred_proba
})



submission.to_csv("submission.csv", index=False)




