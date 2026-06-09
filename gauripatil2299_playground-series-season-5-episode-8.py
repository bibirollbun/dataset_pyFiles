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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

# Basic information
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:\n", train.columns.tolist())

# Preview data
print("\nTrain head:\n", train.head())
print("\nTest head:\n", test.head())

# Check for missing values
print("\nMissing values in train:\n", train.isnull().sum())
print("\nMissing values in test:\n", test.isnull().sum())

# Target distribution
sns.countplot(x='y', data=train)
plt.title("Target Distribution")
plt.show()

# Basic statistics
print("\nTrain describe:\n", train.describe(include='all'))

# Check datatypes
print("\nData types:\n", train.dtypes.value_counts())



from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Drop ID
train = train.drop(columns=["id"])
test_ids = test["id"]
test = test.drop(columns=["id"])

# Separate features
X = train.drop(columns=["y"])
y = train["y"]

# Identify columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(include="number").columns.tolist()

# Optional: new binary feature
X["has_contact_before"] = (X["pdays"] != -1).astype(int)
test["has_contact_before"] = (test["pdays"] != -1).astype(int)
numeric_cols.append("has_contact_before")

# Preprocessing pipelines
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_cols),
    ("cat", cat_pipeline, categorical_cols)
])

# Fit-transform
X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test)

# Sanity check
print("Processed X shape:", X_processed.shape)
print("Processed test shape:", test_processed.shape)



import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set style
sns.set(style="whitegrid")

# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=train)
plt.title("Target Distribution")
plt.xlabel("Subscribed (y)")
plt.ylabel("Count")
plt.show()

# 2. Numerical Feature Distributions
num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=train, x=col, hue="y", bins=50, kde=True, stat="density", common_norm=False)
    plt.title(f"Distribution of {col} by Target")
    plt.show()

# 3. Categorical Feature Distributions vs Target
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in cat_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=col, hue="y", data=train, order=train[col].value_counts().index)
    plt.title(f"{col} vs Target")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 4. Correlation Heatmap (numeric only)
plt.figure(figsize=(10, 8))
corr = train[num_cols + ['y']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# 5. Boxplots of numerical features by target
for col in ['age', 'balance', 'duration']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="y", y=col, data=train)
    plt.title(f"{col} vs Target")
    plt.show()


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings

# ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)

params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "device": "gpu",               # âœ… GPU enabled
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "random_state": 42
}

# CV setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Arrays to store results
oof_preds = np.zeros(X_processed.shape[0])
test_preds = np.zeros(test_processed.shape[0])

for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y)):
    print(f"Fold {fold+1}")
    
    X_train, X_val = X_processed[train_idx], X_processed[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )

    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(test_processed) / n_splits

# Final AUC
auc_score = roc_auc_score(y, oof_preds)
print(f"\nOverall CV ROC AUC: {auc_score:.5f}")



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings

# ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Assume you already have: X, y, test (from previous phases)

# Encode categorical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Convert all to float32 for compatibility
X = X.astype('float32')
test = test.astype('float32')

# Prepare for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store out-of-fold predictions and test predictions
oof_preds_lgb, oof_preds_xgb, oof_preds_cat = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
test_preds_lgb, test_preds_xgb, test_preds_cat = np.zeros(len(test)), np.zeros(len(test)), np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ“¦ Fold {fold+1}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        objective='binary',
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100)]
    )
    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(test)[:, 1] / skf.n_splits

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(test)[:, 1] / skf.n_splits

    # CatBoost
    cat_model = CatBoostClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        verbose=0,
        random_state=42,
        task_type='GPU'  # Set to 'GPU' if you're using GPU
    )
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100
    )
    oof_preds_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_preds_cat += cat_model.predict_proba(test)[:, 1] / skf.n_splits

# Evaluate models
print("\nâœ… AUC Scores:")
print("LightGBM AUC:", roc_auc_score(y, oof_preds_lgb))
print("XGBoost  AUC:", roc_auc_score(y, oof_preds_xgb))
print("CatBoost AUC:", roc_auc_score(y, oof_preds_cat))



# Weighted average (best ensemble)
final_oof_preds = (
    0.4 * oof_preds_lgb +
    0.3 * oof_preds_xgb +
    0.3 * oof_preds_cat
)

final_test_preds = (
    0.4 * test_preds_lgb +
    0.3 * test_preds_xgb +
    0.3 * test_preds_cat
)


# Make sure validation predictions exist
lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
cat_val_pred = cat_model.predict_proba(X_val)[:, 1]



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# 1. ğŸ“ˆ ROC Curve Comparison
plt.figure(figsize=(8, 6))

for name, model, val_pred in [
    ("LightGBM", lgb_model, lgb_val_pred),
    ("XGBoost", xgb_model, xgb_val_pred),
    ("CatBoost", cat_model, cat_val_pred),
]:
    fpr, tpr, _ = roc_curve(y_val, val_pred)
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {model_auc:.4f})")

plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# 2. ğŸ“Š Prediction Distribution for Train vs. Test
plt.figure(figsize=(10, 5))

# Use LGB model as representative
sns.histplot(lgb_model.predict(X_val), color='blue', label='Validation', stat='density', kde=True, bins=50)
sns.histplot(lgb_model.predict(test), color='green', label='Test', stat='density', kde=True, bins=50)

plt.title("LightGBM Prediction Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# 3. ğŸŒŸ Feature Importance (Top 15)
def plot_feature_importance(model, model_name):
    importance = model.feature_importances_
    features = X_train.columns
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    imp_df = imp_df.sort_values(by="Importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, y="Feature", x="Importance", palette="viridis")
    plt.title(f"{model_name} Top 15 Feature Importances")
    plt.tight_layout()
    plt.show()

plot_feature_importance(lgb_model, "LightGBM")
plot_feature_importance(xgb_model, "XGBoost")
plot_feature_importance(cat_model, "CatBoost")



test.head()
# print(test.shape)


ids = np.arange(750000, 1000000)

submission_sample = pd.DataFrame({
    "id": ids,
    "y": final_test_preds
})

submission_sample.to_csv("submission.csv", index=False)
submission_sample.head()



# import shutil
# import os

# # Define the path
# working_dir = '/kaggle/working/'

# # Loop through files and delete
# for filename in os.listdir(working_dir):
#     file_path = os.path.join(working_dir, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)  # delete file or link
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)  # delete directory
#     except Exception as e:
#         print(f'Failed to delete {file_path}. Reason: {e}')

# print("âœ… kaggle/working directory cleared.")


