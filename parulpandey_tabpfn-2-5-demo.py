# Installing TabPFN
!uv pip install tabpfn -q


# importing the model
import os
os.environ["TABPFN_MODEL_CACHE_DIR"] = "/kaggle/input/tabpfn-2-5/pytorch/default/2"


# Importing necessary libraries

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

# Ignoring the warnings
import warnings  
warnings.filterwarnings(action = "ignore")


# Reading the training data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

print(train.isna().sum())



# Looking at first few columns of the training data
train.head()


# Getting data ready for training
FEATURES = [c for c in train.columns if c not in ["rainfall",'id']]

X = train[FEATURES].copy()
y = train["rainfall"].copy()



# Splitting existing training data into training and validation set
train_index, valid_index = train_test_split(
    train.index,
    test_size=0.2,
    random_state=42
)

x_train = X.loc[train_index].copy()
y_train = y.loc[train_index].copy()

x_valid = X.loc[valid_index].copy()
y_valid = y.loc[valid_index].copy()


model_pfn = TabPFNClassifier(device=["cuda:0", "cuda:1"])
model_pfn.fit(x_train, y_train)

# Predict probabilities
probs_pfn = model_pfn.predict_proba(x_valid)

# Probability of the positive class (class = 1)
pos_probs = probs_pfn[:, 1]

# Metrics
print(f"ROC AUC: {roc_auc_score(y_valid, pos_probs):.4f}")




model_xgb = XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    device="cuda",
    enable_categorical=True,
    random_state=42,
    n_jobs=1
)

model_xgb.fit(x_train, y_train)

# Predict probabilities
probs_xgb = model_xgb.predict_proba(x_valid)

# Probability of the positive class (class = 1)
pos_probs_xgb = probs_xgb[:, 1]

# Metrics
print(f"ROC AUC: {roc_auc_score(y_valid, pos_probs_xgb):.4f}")



# Drop id from features
test_ids = test["id"]
x_test = test.drop(columns=["id"])

# Predict probabilities
test_probs = model_pfn.predict_proba(x_test)

# Take probability of the positive class
rainfall_probs = test_probs[:, 1]

# Build submission
submission = pd.DataFrame(
    {
        "id": test_ids,
        "rainfall": rainfall_probs
    }
)

submission.to_csv("submission.csv", index=False)



# Install the interpretability extension:
!uv pip install "tabpfn-extensions[interpretability]" -q


# from tabpfn_extensions import interpretability

# # Calculate SHAP values
# shap_values = interpretability.shap.get_shap_values(
#     estimator=model_pfn,
#     test_x=x_test[:50],
#     attribute_names=FEATURES,
#     algorithm="permutation",
# )

# # Create visualization
# #fig = interpretability.shap.plot_shap(shap_values)

