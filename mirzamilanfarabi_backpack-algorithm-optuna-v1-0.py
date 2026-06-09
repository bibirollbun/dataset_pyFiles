# License & Credits
# Name: Mirza Milan Farabi  

# @title Licensed under the Apache License, Version 2.0 (the "License");  
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at  
#  
# https://www.apache.org/licenses/LICENSE-2.0  
#  
# Unless required by applicable law or agreed to in writing, software  
# distributed under the License is distributed on an "AS IS" BASIS,  
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and  
# limitations under the License.


! pip install optuna-integration[lightgbm]


import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings

# Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Handle NaN values in the 'Brand' column
train_df["Brand"] = train_df["Brand"].fillna("Unknown")
test_df["Brand"] = test_df["Brand"].fillna("Unknown")

# Encode categorical features
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    encoders[col] = le  # Save encoders in case needed later

# Prepare features and target
X_train = train_df.drop(columns=["id", "Price"])
y_train = train_df["Price"]
groups = train_df["Brand"]  # Use Brand as groups for GroupKFold

X_test = test_df.drop(columns=["id"])

# Define objective function for Optuna
def objective(trial, X, y, groups, n_splits=5):
    """Optuna objective function with GroupKFold cross-validation."""
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 50, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 0.010000000000001, 0.100000000000001, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500, step=100),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 50.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 500.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1e-3, log=True),
        'max_bin': trial.suggest_int('max_bin', 1000, 3000),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 200, 600),
        'random_state': 1024,
        'verbose': -1
    }
    
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMRegressor(**params)
        
        fit_params = {
            'eval_set': [(X_val_fold, y_val_fold)],
            'eval_metric': 'rmse',
            'callbacks': [optuna.integration.LightGBMPruningCallback(trial, 'rmse')]
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_train_fold, y_train_fold,
                **fit_params
            )
        
        y_pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        fold_metrics.append(rmse)
        
        trial.set_user_attr(f"fold_{fold}_rmse", rmse)
    
    mean_rmse = np.mean(fold_metrics)
    trial.set_user_attr("mean_rmse", mean_rmse)
    
    return mean_rmse

# Hyperparameter search with Optuna
def train_optuna_model(X, y, groups, n_trials=5, n_splits=5):
    """Train and optimize model using Optuna with GroupKFold."""
    study = optuna.create_study(
        direction="minimize",
        study_name="bag_price_prediction_lgbm",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )
    
    study.optimize(
        lambda trial: objective(trial, X, y, groups, n_splits),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\nBest trial:")
    trial = study.best_trial
    
    print("\nBest trial metrics:")
    print(f"  Mean RMSE: {trial.user_attrs['mean_rmse']:.4f}")
    
    print("\nBest parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params

# Train final model with best parameters
def train_final_model(X, y, params):
    """Train final model with best parameters."""
    model = LGBMRegressor(**params)
    model.fit(X, y)
    return model

# Evaluate model on test set and create submission file
def create_submission(model, X_test, test_df):
    """Generate predictions and save to submission.csv."""
    predictions = model.predict(X_test)
    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "Price": predictions
    })
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file created: submission.csv")

# Main workflow
if __name__ == "__main__":
    # Perform hyperparameter search
    best_params = train_optuna_model(X_train, y_train, groups, n_trials=5)
    
    # Train final model with best parameters
    final_model = train_final_model(X_train, y_train, best_params)
    
    # Generate predictions and create submission file
    create_submission(final_model, X_test, test_df)


