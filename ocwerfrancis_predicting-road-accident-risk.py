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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
train_df.head()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test_df.head()


type(test_df)


train_df.shape, test_df.shape


train_df.isnull().sum()


train_df.dtypes


test_df.dtypes


train_df = train_df.copy()


train_df.hist(bins = 25,figsize=(20,10))


train_df.columns


train_df.accident_risk.isnull().sum()


train_df.dtypes


def convert_object(df):
    df['road_signs_present'] = df['road_signs_present'].astype(object)
    df['public_road'] = df['public_road'].astype(object)
    df['holiday'] = df['holiday'].astype(object)
    df['school_season'] = df['school_season'].astype(object)


type(train_df)


type(test_df)


convert_object(train_df)

   


convert_object(test_df)


train_df.dtypes


from sklearn.model_selection import train_test_split


train_df,val_df = train_test_split(train_df, test_size=0.25, random_state=42)


len(train_df), len(val_df)


train_df.shape, val_df.shape, test_df.shape


train_df.columns


input_cols = train_df.columns[1:-1]
input_cols




target_col = train_df.columns[-1]
target_col


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

# Validation dataset inputs and target

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()


numeric_cols = list(var for var in train_inputs.columns if train_inputs[var].dtype != 'O')
numeric_cols


categorical_cols = list(var for var in train_inputs.columns if train_inputs[var].dtype == 'O')

categorical_cols


train_inputs.isnull().sum()



from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler().fit(train_inputs[numeric_cols])


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


from sklearn.metrics import mean_squared_error


import numpy as np

def root_mean_squared_error(predictions, targets):
    return np.sqrt(mean_squared_error(predictions, targets))


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import XGBRegressor


def train_and_evaluate(X_train_fold, train_targets_fold, X_val_fold, val_targets_fold, **params):
    model = XGBRegressor(random_state=42,n_estimators=5000, n_jobs=-1, **params)
    model.fit(X_train_fold, train_targets_fold,eval_set=[(X_val_fold, val_targets_fold)],verbose=False)
    train_rmse = root_mean_squared_error(model.predict(X_train_fold), train_targets_fold)
    val_rmse = root_mean_squared_error(model.predict(X_val_fold), val_targets_fold)
    return model, train_rmse, val_rmse


kfold = KFold(n_splits=5, shuffle=True, random_state=42)


models = []
train_scores = []
val_scores = []

# Use ORIGINAL X_train and train_targets for splitting
for fold, (train_idxs, val_idxs) in enumerate(kfold.split(X_train)):
    print(f"Fold {fold + 1}/{kfold.n_splits}")
    
    # Create new variable names for fold data
    X_train_fold, X_val_fold = X_train.iloc[train_idxs], X_train.iloc[val_idxs]
    train_targets_fold, val_targets_fold = train_targets.iloc[train_idxs], train_targets.iloc[val_idxs]
    
    model, train_rmse, val_rmse = train_and_evaluate(
        X_train_fold,
        train_targets_fold, 
        X_val_fold,
        val_targets_fold,
        max_depth=15,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        early_stopping_rounds=100,
        reg_alpha=0.1,
        reg_lambda=0.1,
    )

    models.append(model)
    train_scores.append(train_rmse)
    val_scores.append(val_rmse)
    print(f'Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}')
    print('---')

print(f"\nAverage Train RMSE: {np.mean(train_scores):.4f}")
print(f"Average Validation RMSE: {np.mean(val_scores):.4f}")


# Make predictions using all trained models
test_predictions = []

for i, model in enumerate(models):
    pred = model.predict(X_test)
    test_predictions.append(pred)
    print(f"Fold {i+1} prediction range: {pred.min():.4f} - {pred.max():.4f}")

# Average predictions across all folds (ensemble)
final_predictions = np.mean(test_predictions, axis=0)

print(f"Final predictions shape: {final_predictions.shape}")
print(f"Final prediction range: {final_predictions.min():.4f} - {final_predictions.max():.4f}")


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission_df.shape


# Assuming you have your final_predictions from the previous code
submission_df['accident_risk'] = final_predictions

# Verify the update
print("Updated submission preview:")
print(submission_df.head())
print(f"\nSubmission shape: {submission_df.shape}")

# Save the updated submission
submission_df.to_csv('submission_3.csv', index=False)
print("Submission file saved as 'my_submission.csv'")


from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import lightgbm as lgb
import numpy as np

def train_and_evaluate_lgbm(X_train_fold, train_targets_fold, X_val_fold, val_targets_fold, **params):
    model = LGBMRegressor(
        random_state=42, 
        n_jobs=-1,
        verbose=-1,
        **params
    )
    model.fit(
        X_train_fold, 
        train_targets_fold,
        eval_set=[(X_val_fold, val_targets_fold)],
        eval_metric='rmse',
        callbacks=[lgb.log_evaluation(0)]
    )
    train_rmse = root_mean_squared_error(model.predict(X_train_fold), train_targets_fold)
    val_rmse = root_mean_squared_error(model.predict(X_val_fold), val_targets_fold)
    return model, train_rmse, val_rmse

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Train single model for accident_risk
models = []
train_scores = []
val_scores = []

# Use ORIGINAL X_train and train_targets for splitting
for fold, (train_idxs, val_idxs) in enumerate(kfold.split(X_train)):
    print(f"Fold {fold + 1}/{kfold.n_splits}")
    
    # Create new variable names for fold data
    X_train_fold, X_val_fold = X_train.iloc[train_idxs], X_train.iloc[val_idxs]
    
    # Assuming train_targets is a Series with accident_risk values
    train_targets_fold, val_targets_fold = train_targets.iloc[train_idxs], train_targets.iloc[val_idxs]
    
    # Train single model for accident_risk
    model, train_rmse, val_rmse = train_and_evaluate_lgbm(
        X_train_fold,
        train_targets_fold,  # Single target series
        X_val_fold,
        val_targets_fold,    # Single target series
        objective='regression',
        metric='rmse',
        num_leaves=1000,
        n_estimators=5000,
        min_child_samples=50,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1
    )
    
    models.append(model)
    train_scores.append(train_rmse)
    val_scores.append(val_rmse)
    
    print(f'Accident Risk - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}')
    print('---')

print(f"\n=== ACCIDENT RISK PREDICTION ===")
print(f"Average Train RMSE: {np.mean(train_scores):.4f} (+/- {np.std(train_scores):.4f})")
print(f"Average Validation RMSE: {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})")


# Make predictions using all trained models
test_predictions = []

for i, model in enumerate(models):
    pred = model.predict(X_test)
    test_predictions.append(pred)
    print(f"Fold {i+1} prediction range: {pred.min():.4f} - {pred.max():.4f}")

# Average predictions across all folds (ensemble)
final_predictions = np.mean(test_predictions, axis=0)

print(f"Final predictions shape: {final_predictions.shape}")
print(f"Final prediction range: {final_predictions.min():.4f} - {final_predictions.max():.4f}")


# Assuming you have your final_predictions from the previous code
submission_df['accident_risk'] = final_predictions

# Verify the update
print("Updated submission preview:")
print(submission_df.head())
print(f"\nSubmission shape: {submission_df.shape}")

# Save the updated submission
submission_df.to_csv('submission_6.csv', index=False)
print("Submission file saved as 'submission_6.csv'")




