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


!pip install pytorch-tabnet



import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load datasets
df_train = pd.read_csv("/kaggle/input/thapar-kaggle-hack-v02/train.csv")
df_test = pd.read_csv("/kaggle/input/thapar-kaggle-hack-v02/test.csv")
df_submission = pd.read_csv("/kaggle/input/thapar-kaggle-hack-v02/sample_submission.csv")

# Drop 'id' column
if "id" in df_train.columns:
    df_train.drop(columns=["id"], inplace=True)
if "id" in df_test.columns:
    test_ids = df_test.pop("id")
else:
    test_ids = None

# Separate features and target variable
X = df_train.drop(columns=["target"])
y = df_train["target"]
X_test = df_test.copy()

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
if cat_cols:
    X_cat = encoder.fit_transform(X[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])
else:
    X_cat = np.empty((X.shape[0], 0))
    X_test_cat = np.empty((X_test.shape[0], 0))

# Standardize numerical features using RobustScaler
scaler = RobustScaler()
X_num_scaled = scaler.fit_transform(X[num_cols]) if num_cols else np.empty((X.shape[0], 0))
X_test_num_scaled = scaler.transform(X_test[num_cols]) if num_cols else np.empty((X_test.shape[0], 0))

# Generate polynomial and interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_num_scaled)
X_test_poly = poly.transform(X_test_num_scaled)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_poly)
X_test_pca = pca.transform(X_test_poly)

# Combine with categorical features
X_processed = np.hstack((X_pca, X_cat))
X_test_processed = np.hstack((X_test_pca, X_test_cat))

# Feature Selection
selector = SelectKBest(f_classif, k=min(300, X_processed.shape[1]))
X_selected = selector.fit_transform(X_processed, y)
X_test_selected = selector.transform(X_test_processed)

# SMOTE for balancing
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected, y)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Train TabNet model
tabnet = TabNetClassifier()
tabnet.fit(X_train, y_train, eval_set=[(X_val, y_val)], patience=10, max_epochs=100)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6)
xgb_model.fit(X_train, y_train)

# Blend Predictions
y_pred_tabnet = tabnet.predict_proba(X_test_selected)
y_pred_xgb = xgb_model.predict_proba(X_test_selected)

# Weighted averaging
ensemble_pred = (0.5 * y_pred_tabnet) + (0.5 * y_pred_xgb)
y_test_pred = np.argmax(ensemble_pred, axis=1)

# Submission
df_submission["target"] = y_test_pred
if test_ids is not None:
    if "id" not in df_submission.columns:
        df_submission.insert(0, "id", test_ids)
df_submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

