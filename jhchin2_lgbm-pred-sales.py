#!pip install hillclimbers
#!pip install colorama

import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
#from hillclimbers import climb_hill, partial
import warnings
from lightgbm.callback import early_stopping, log_evaluation 
from colorama import Fore, Style
import colorama
import seaborn as sns
import matplotlib.pyplot as plt
import optuna


# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

# Set seed for reproducibility
#SEED = 114514


# 1. Data Loading and Preprocessing
# ----------------------------------

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
train.dropna(inplace=True)
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


def date(df): 
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.month_name()
        df['day_of_week'] = df['date'].dt.day_name()
        df['week'] = df['date'].dt.isocalendar().week
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12) 
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7
        
        df.drop('date', axis=1, inplace=True)
    
        df['cos_year'] = np.cos(df['year'] * (2 * np.pi) / 100)
        df['sin_year'] = np.sin(df['year'] * (2 * np.pi) / 100)
        df['year_lag_1'] = df['year'].shift(1)
        df['year_diff'] = df['year'] - df['year_lag_1']
    
        return df
# Apply date feature engineering
train =date(train)
test = date(test)


# Define categorical columns for One-Hot Encoding
cat_cols = [ 'store','month_name','day_of_week','country','product']


def ohe_transform(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list):
    
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
        train_ohe = pd.DataFrame(ohe.fit_transform(train[cat_cols]), 
                                 columns=ohe.get_feature_names_out(cat_cols), 
                                 index=train.index)
        
        test_ohe = pd.DataFrame(ohe.transform(test[cat_cols]), 
                                columns=ohe.get_feature_names_out(cat_cols), 
                                index=test.index)
    
        train = train.drop(columns=cat_cols).reset_index(drop=True)
        test = test.drop(columns=cat_cols).reset_index(drop=True)
    
        train = pd.concat([train, train_ohe.reset_index(drop=True)], axis=1)
        test = pd.concat([test, test_ohe.reset_index(drop=True)], axis=1)
    
        return train, test

train , test = ohe_transform(train,test,cat_cols)


train.head()


groups = train['group']
X = train.drop(columns=['num_sold'])
y = np.log1p(train['num_sold'])  # Apply y_log to stabilize variance


# 2. GroupKFold Cross-Validation Setup
group_kfold = GroupKFold(n_splits=5)




# 2. LightGBM Trial Function
def lgbm_objective(trial):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mape',  # We'll evaluate on MAPE
        'n_estimators': trial.suggest_int('n_estimators', 100, 1200),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 6),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 2.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-4, 2.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 120),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        #'random_state': 42,
        #'device_type': 'gpu',  # GPU acceleration
    }
    
    oof_predictions = np.zeros(len(train))

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params,verbose=-1)
        
        # Train using early stopping callback
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(100)],  # Use callback to handle early stopping
            
        )
        oof_predictions[val_idx] = model.predict(X_val)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_predictions))
    return mape


"""# 4. Run Optuna Optimization (for LightGBM)
study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(lgbm_objective, n_trials=50)


print("Best LGBM Params:", study_lgbm.best_params)"""


params = {'n_estimators': 1179, 'learning_rate': 0.08306802241500667, 'max_depth': 5, 'reg_alpha': 0.00012579497472025837, 'lambda_l2': 0.00047494948934310007, 'min_child_samples': 68, 'colsample_bytree': 0.9579325891323452, 'subsample': 0.5843341782698147
        }


# Function to train a LightGBM model and generate predictions
def train_lgbm(X, y, X_test, params, folds, groups=None):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    feature_importance_aggregate = pd.DataFrame()

    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y, groups=groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params,verbose=-1)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='mape',
            callbacks=[early_stopping(100)],
            
        )
         # Generate predictions for validation and test sets
        oof_pred = model.predict(X_val, num_iteration=model.best_iteration_)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        
        # Store the predictions
        oof_preds[val_idx] = oof_pred
        test_preds += model.predict(X_test, num_iteration=model.best_iteration_) / 5 #no of folds
        models.append(model)
          # Extract feature importance
        fold_importance = pd.DataFrame({
           'Feature': model.feature_name_,
           'Importance': model.feature_importances_,
           'Fold': fold
        })
        feature_importance_aggregate = pd.concat([feature_importance_aggregate, fold_importance], axis=0)
        # Calculate and print MAPE for the current fold
        mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(oof_pred))
        print(f"Fold {fold+1} - MAPE: {mape:.4f}%")
    
    # Calculate overall MAPE for all out-of-fold predictions
    overall_mape = mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_preds))
    # Aggregate feature importance by averaging across folds
    aggregated_importance = feature_importance_aggregate.groupby('Feature')['Importance'].mean().reset_index()
    aggregated_importance = aggregated_importance.sort_values(by='Importance', ascending=False)

    # Visualize top 20 features
    top_n = 50
    top_features = aggregated_importance.head(top_n)

    plt.figure(figsize=(12, 10))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=top_features,
        palette='viridis'
    )
    plt.title(f'Top {top_n} Feature Importances (Average Gain)', fontsize=16)
    plt.xlabel('Average Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.show()
    return oof_preds, test_preds, models


# Train first LightGBM model
print(Fore.YELLOW + "Training LGBM Model 1...")
oof1, test1, models1 = train_lgbm(X, y, test, params, group_kfold, groups)


sample["num_sold"] = np.expm1(test1)
sample.to_csv("submission.csv", index=False)
print("Submission shape:", sample.shape)
print(sample.head())







