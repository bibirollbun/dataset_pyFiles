!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy.stats import rankdata


# Load Data
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
sample_sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")


print(train.columns)




# EDA: Visualize efs_time Distribution
plt.hist(train.loc[train.efs == 1, "efs_time"], bins=100, label="efs=1, Event")
plt.hist(train.loc[train.efs == 0, "efs_time"], bins=100, label="efs=0, No Event")
plt.xlabel("Time of Observation (efs_time)")
plt.ylabel("Density")
plt.title("Observation Times")
plt.legend()
plt.show()

# Kaplan-Meier Transformation
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y

train["y"] = transform_survival_probability(train, 'efs_time', 'efs')

# Fix Data Types
categorical_columns = train.select_dtypes(include=['object']).columns.tolist()

# Convert categorical columns to 'category' dtype
for col in categorical_columns:
    train[col] = train[col].astype('category')
    if col in test.columns:
        test[col] = test[col].astype('category')

# Optimize numerical columns
common_columns = set(train.columns).intersection(set(test.columns))

for col in common_columns:
    if train[col].dtype == 'float64':
        train[col] = train[col].astype('float32')
        test[col] = test[col].astype('float32')
    elif train[col].dtype == 'int64':
        train[col] = train[col].astype('int32')
        test[col] = test[col].astype('int32')

# Ensure 'Unknown' is added as a category before filling missing values
if 'race_group' in train.columns:
    if 'Unknown' not in train['race_group'].cat.categories:
        train['race_group'] = train['race_group'].cat.add_categories(['Unknown'])
    if 'Unknown' not in test['race_group'].cat.categories:
        test['race_group'] = test['race_group'].cat.add_categories(['Unknown'])
    
    train['race_group'].fillna('Unknown', inplace=True)
    test['race_group'].fillna('Unknown', inplace=True)

# Modeling Features
RMV = ["ID", "efs", "efs_time", "y"]
FEATURES = [c for c in train.columns if c not in RMV]

# XGBoost Modeling with K-Fold CV
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"### Fold {i+1}")
    x_train, y_train = train.iloc[train_idx][FEATURES], train.iloc[train_idx]["y"]
    x_valid, y_valid = train.iloc[valid_idx][FEATURES], train.iloc[valid_idx]["y"]
    x_test = test[FEATURES]

    model_xgb = XGBRegressor(
        max_depth=3,
        colsample_bytree=0.5,
        subsample=0.8,
        n_estimators=2000,
        learning_rate=0.02,
        min_child_weight=80,
        enable_categorical=True,
        random_state=42
    )
    model_xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=500)
    oof_xgb[valid_idx] = model_xgb.predict(x_valid)
    pred_xgb += model_xgb.predict(x_test)

pred_xgb /= FOLDS

# Custom Scoring Function
def score(solution, submission, row_id_col, group_col="race_group"):
    # Check if group_col is in the solution
    if group_col not in solution.columns:
        print(f"Warning: '{group_col}' column is missing. Scoring skipped.")
        return None

    try:
        # Merge and calculate concordance index for each group
        merged = solution.merge(submission, on=row_id_col, suffixes=('_true', '_pred'))
        c_indices = [
            concordance_index(
                group["efs_time"], -group["prediction"], group["efs"]
            ) for _, group in merged.groupby(group_col)
        ]
        return np.mean(c_indices) - np.sqrt(np.var(c_indices))
    except Exception as e:
        print(f"Error during scoring: {e}")
        return None

# Evaluate OOF Predictions
y_true = train[["ID", "efs", "efs_time", "race_group"]]
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
cv_score = score(y_true, y_pred, "ID")
print(f"CV Score: {cv_score}" if cv_score is not None else "Scoring was skipped.")

# Create Submission
sample_sub["prediction"] = rankdata(pred_xgb)
sample_sub.to_csv("submission.csv", index=False)




