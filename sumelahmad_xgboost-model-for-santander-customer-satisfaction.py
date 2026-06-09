import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')
test = pd.read_csv('/kaggle/input/santander-customer-satisfaction/test.csv')


train.head()


test.head()


train.info()


test.info()


train.columns


train.shape


test.shape


train.isnull().sum()


test.isnull().sum()


train.isnull().sum().sum()


test.isnull().sum().sum()


train.duplicated().sum()


nunique = train.nunique()
constant_columns = nunique[nunique == 1].index.tolist()


constant_columns


train.drop(columns=constant_columns, inplace=True)
test.drop(columns=constant_columns, inplace=True)


train.head()


# Compute correlation matrix
corr_matrix = train.corr().abs()


# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))


# Find features with high correlation (threshold = 0.98)
to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]


to_drop


# Drop the highly correlated features
train.drop(columns=to_drop, inplace=True)
test.drop(columns=to_drop, inplace=True)


print(f"Removed {len(to_drop)} highly correlated columns")


# Separate features and target variable
X = train.drop(columns=["ID", "TARGET"])
y = train["TARGET"]


# Save test IDs for submission
test_ids = test["ID"]
X_test = test.drop(columns=["ID"])


from sklearn.model_selection import train_test_split

# Split data into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.shape, X_val.shape


import xgboost as xgb
from sklearn.metrics import roc_auc_score


# Create the DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test) 


# Set XGBoost parameters
params = {
    "objective": "binary:logistic",  # Binary classification
    "eval_metric": "auc",            # AUC is good for imbalanced data
    "eta": 0.05,                      # Learning rate
    "max_depth": 6,                   # Depth of trees
    "subsample": 0.8,                  # Subsampling for diversity
    "colsample_bytree": 0.8,           # Feature sampling
    "random_state": 42
}



# Train the model
evals = [(dtrain, "train"), (dval, "val")]
xgb_model = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=50, verbose_eval=50)


# Predict on validation set
y_pred = xgb_model.predict(dval)


# Evaluate model performance
auc_score = roc_auc_score(y_val, y_pred)


auc_score


# Predict on test set
test_preds = xgb_model.predict(dtest)

# Create submission file
submission = pd.DataFrame({"ID": test_ids, "TARGET": test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")

