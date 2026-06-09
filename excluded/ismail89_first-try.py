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


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


from lifelines import KaplanMeierFitter
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# COMBINE TRAIN & TEST DATA
combined = pd.concat([train, test], axis=0, ignore_index=True)
print("Combined data shape:", combined.shape)

# IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
NUMERIC_FEATURES = [col for col in FEATURES if col not in CATS]
CATEGORICAL_FEATURES = [col for col in CATS]

# SPLIT BACK INTO TRAIN & TEST AFTER COMBINING
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()

# HANDLE MISSING VALUES
train.fillna(train.select_dtypes(include=['number']).median(), inplace=True)
test.fillna(test.select_dtypes(include=['number']).median(), inplace=True)

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),  # Apply Standard Scaling to numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), CATEGORICAL_FEATURES)  # Apply OneHotEncoding to categorical features
    ]
)

# Apply transformations and split train/test sets
X_train = train[FEATURES]
X_test = test[FEATURES]

# Fitting the preprocessor to the training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nEncoding & Normalization Complete!")

# OPTIONAL: Convert processed features back to DataFrame for inspection
X_train_processed_df = pd.DataFrame(X_train_processed)
X_test_processed_df = pd.DataFrame(X_test_processed)

# VERIFY DATA TYPES AFTER PROCESSING
print("\nTRAIN DATA TYPES AFTER PROCESSING:\n", X_train_processed_df.dtypes)
print("\nTEST DATA TYPES AFTER PROCESSING:\n", X_test_processed_df.dtypes)



from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import xgboost as xgb

# Configuration
FOLDS = 3
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Parameters with GPU support
xgb_params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.03,
    "max_depth": 8,
    "n_estimators": 6000,
    "colsample_bytree": 0.8,
    "subsample": 0.9,
    "reg_lambda": 8.0,
    "random_state": 42,
    "enable_categorical": True,
    "tree_method": "gpu_hist",
    "device": "cuda",
    "predictor": "gpu_predictor"
}

lgb_params = {
    "objective": "regression",
    "min_child_samples": 32,
    "num_iterations": 6000,
    "learning_rate": 0.03,
    "reg_lambda": 8.0,
    "reg_alpha": 0.1,
    "num_leaves": 64,
    "metric": "rmse",
    "max_depth": 8,
    "device": "gpu",
    "max_bin": 128,
    "seed": 42,
    "gpu_platform_id": 0,
    "gpu_device_id": 0
}

# Best found parameters for LightGBM
lgbm_tuned_params = {
    "max_depth": 5,
    "learning_rate": 0.1449,
    "n_estimators": 533,
    "colsample_bytree": 0.7558,
    "subsample": 0.8579,
    "min_child_weight": 90,
    "device": "gpu"
}

def train_and_predict(model_name, params):
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    categorical_cols = ['graft_type', 'prod_type']
    train[categorical_cols] = train[categorical_cols].astype('category')
    test[categorical_cols] = test[categorical_cols].astype('category')

    for train_idx, valid_idx in kf.split(train):
        x_train, y_train = train.iloc[train_idx][FEATURES], train.iloc[train_idx]["y"]
        x_valid, y_valid = train.iloc[valid_idx][FEATURES], train.iloc[valid_idx]["y"]
        x_test = test[FEATURES]

        if model_name == "xgb":
            model = xgb.XGBRegressor(**params)
        elif model_name == "lgb":
            model = LGBMRegressor(**params)
        else:
            raise ValueError("Invalid model name")

        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        oof_preds[valid_idx] = model.predict(x_valid)
        test_preds += model.predict(x_test) / FOLDS

    return oof_preds, test_preds

if __name__ == "__main__":
    # Train models and get predictions
    oof_xgb, pred_xgb = train_and_predict("xgb", xgb_params)
    oof_lgb, pred_lgb = train_and_predict("lgb", lgb_params)
    oof_lgb_tuned, pred_lgb_tuned = train_and_predict("lgb", lgbm_tuned_params)

    # Create meta-features for stacking
    meta_features = np.column_stack([oof_xgb, oof_lgb, oof_lgb_tuned])
    meta_test = np.column_stack([pred_xgb, pred_lgb, pred_lgb_tuned])

    # Train meta-model
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features, train["y"])
    final_preds = meta_model.predict(meta_test)

    # Evaluate performance
    mse = mean_squared_error(test["y"], final_preds)
    print(f"Final model Mean Squared Error: {mse}")


from lifelines.utils import concordance_index
import pandas as pd

# Prepare true values
y_true = train[["efs_time", "efs", "race_group", "ID"]].copy()

# Create OOF predictions for ensemble
oof_ensemble = meta_model.predict(meta_features)  # Meta-model's OOF predictions

def calculate_stratified_cindex(oof_predictions, model_name):
    y_pred = pd.DataFrame({"ID": y_true["ID"], "prediction": oof_predictions})
    
    c_indices = []
    for race in y_true["race_group"].unique():
        mask = y_true["race_group"] == race
        c_index = concordance_index(
            y_true.loc[mask, "efs_time"],
            y_pred.loc[mask, "prediction"],
            y_true.loc[mask, "efs"]
        )
        c_indices.append(c_index)
    
    stratified_c_index = sum(c_indices)/len(c_indices) - pd.Series(c_indices).std()
    print(f"Stratified C-index ({model_name}): {stratified_c_index:.5f}")
    return stratified_c_index

# Calculate scores for all models
_ = calculate_stratified_cindex(oof_xgb, "XGBoost")
_ = calculate_stratified_cindex(oof_lgb, "LightGBM")
_ = calculate_stratified_cindex(oof_cat, "CatBoost")


print("\nFinal Ensemble Score:")
print(f"Kaplan-Meier Stratified C-index: {calculate_stratified_cindex(oof_ensemble, 'Final Ensemble'):.5f}")


import pandas as pd

# Load the sample submission file
sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")

# Assign predictions from your model
sub["prediction"] = pred_ensemble # Replace with your model's predictions

# Save the submission file
sub.to_csv("submission.csv", index=False)

# Display info
print("Sub shape:", sub.shape)
print(sub.head())


