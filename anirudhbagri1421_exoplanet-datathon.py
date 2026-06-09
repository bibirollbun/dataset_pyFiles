import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Load data
train_df = pd.read_csv('/kaggle/input/syzygy-equinox/train.csv')
val_df = pd.read_csv('/kaggle/input/syzygy-equinox/val.csv')
test_df = pd.read_csv('/kaggle/input/syzygy-equinox/test.csv')

# Drop unnecessary columns
cols_to_drop = ["rowid", "kepid", "kepoi_name", "kepler_name", "koi_disposition"]
train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
val_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Handle missing values (fill with median)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
val_df[numeric_cols] = val_df[numeric_cols].fillna(val_df[numeric_cols].median())
test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].median())

# Feature Engineering
def feature_engineering(df):
    df['koi_period_err'] = (df['koi_period_err1'] + df['koi_period_err2']) / 2
    df['koi_fpflag_sum'] = df[['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']].sum(axis=1)
    df['depth_duration_ratio'] = df['koi_depth'] / df['koi_duration']
    df['prad_srad_ratio'] = df['koi_prad'] / df['koi_srad']
    return df

train_df = feature_engineering(train_df)
val_df = feature_engineering(val_df)
test_df = feature_engineering(test_df)

# Encode target variable
target_map = {"CANDIDATE": 1, "FALSE POSITIVE": 0}
train_df["koi_pdisposition"] = train_df["koi_pdisposition"].map(target_map)
val_df["koi_pdisposition"] = val_df["koi_pdisposition"].map(target_map)

# Split features and target
X_train = train_df.drop(columns=["koi_pdisposition"])
y_train = train_df["koi_pdisposition"]
X_val = val_df.drop(columns=["koi_pdisposition"])
y_val = val_df["koi_pdisposition"]
X_test = test_df.copy()

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train model with hyperparameter tuning
xgb_model = XGBClassifier(
    n_estimators=500, 
    learning_rate=0.03, 
    max_depth=8, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# Validate model
y_val_pred = xgb_model.predict(X_val_scaled)
print("Validation F1 Score:", f1_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Optimize threshold
y_val_probs = xgb_model.predict_proba(X_val_scaled)[:, 1]
optimal_threshold = 0.5  # You can tune this further based on PR curves
y_val_opt_pred = (y_val_probs >= optimal_threshold).astype(int)
print("Optimized Validation F1 Score:", f1_score(y_val, y_val_opt_pred))

# Predict on test set
y_test_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_probs >= optimal_threshold).astype(int)
predictions = ["CANDIDATE" if pred == 1 else "FALSE POSITIVE" for pred in y_test_pred]

# Prepare submission
submission = pd.DataFrame({"rowid": test_df.index, "koi_pdisposition": predictions})
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Load data
train_df = pd.read_csv('/kaggle/input/syzygy-equinox/train.csv')
val_df = pd.read_csv('/kaggle/input/syzygy-equinox/val.csv')
test_df = pd.read_csv('/kaggle/input/syzygy-equinox/test.csv')

# Drop unnecessary columns
cols_to_drop = ["rowid", "kepid", "kepoi_name", "kepler_name", "koi_disposition"]
train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
val_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Handle missing values (fill with median)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
val_df[numeric_cols] = val_df[numeric_cols].fillna(val_df[numeric_cols].median())
test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].median())

# Feature Engineering
def feature_engineering(df):
    df['koi_period_err'] = (df['koi_period_err1'] + df['koi_period_err2']) / 2
    df['koi_fpflag_sum'] = df[['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']].sum(axis=1)
    df['depth_duration_ratio'] = df['koi_depth'] / df['koi_duration']
    df['prad_srad_ratio'] = df['koi_prad'] / df['koi_srad']
    return df

train_df = feature_engineering(train_df)
val_df = feature_engineering(val_df)
test_df = feature_engineering(test_df)

# Encode target variable
target_map = {"CANDIDATE": 1, "FALSE POSITIVE": 0}
train_df["koi_pdisposition"] = train_df["koi_pdisposition"].map(target_map)
val_df["koi_pdisposition"] = val_df["koi_pdisposition"].map(target_map)

# Split features and target
X_train = train_df.drop(columns=["koi_pdisposition"])
y_train = train_df["koi_pdisposition"]
X_val = val_df.drop(columns=["koi_pdisposition"])
y_val = val_df["koi_pdisposition"]
X_test = test_df.copy()

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train model with hyperparameter tuning
xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Validate model
y_val_pred = xgb_model.predict(X_val_scaled)
print("Validation F1 Score:", f1_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Optimize threshold
y_val_probs = xgb_model.predict_proba(X_val_scaled)[:, 1]
optimal_threshold = 0.5  # You can tune this further based on PR curves
y_val_opt_pred = (y_val_probs >= optimal_threshold).astype(int)
print("Optimized Validation F1 Score:", f1_score(y_val, y_val_opt_pred))

# Predict on test set
y_test_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_probs >= optimal_threshold).astype(int)
predictions = ["CANDIDATE" if pred == 1 else "FALSE POSITIVE" for pred in y_test_pred]

# Prepare submission
submission = pd.DataFrame({"rowid": test_df.index, "koi_pdisposition": predictions})
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")


