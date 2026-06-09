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


import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)


import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.info()


train["temp_diff"] = train["maxtemp"] - train["mintemp"]
test["temp_diff"] = test["maxtemp"] - test["mintemp"]

train['humidity_temp'] = train['humidity'] * train['temparature']
test['humidity_temp'] = test['humidity'] * test['temparature']

train['cloud_humidity'] = train['cloud'] * train['humidity']
test['cloud_humidity'] = test['cloud'] * test['humidity']

train['pressure_temp'] = train['pressure'] / train['temparature']
test['pressure_temp'] = test['pressure'] / test['temparature']

train['wind_vector'] = train['windspeed'] * train['winddirection']
test['wind_vector'] = test['windspeed'] * test['winddirection']

train['temp_squared'] = train['temparature'] ** 2
test['temp_squared'] = test['temparature'] ** 2

train['humidity_squared'] = train['humidity'] ** 2
test['humidity_squared'] = test['humidity'] ** 2

train['pressure_cubed'] = train['pressure'] ** 3
test['pressure_cubed'] = test['pressure'] ** 3

train["humidity_temp_ratio"] = train["humidity"] / (train["temparature"] + 1)
test["humidity_temp_ratio"] = test["humidity"] / (test["temparature"] + 1)
test_id = test['id']


train['day_sin'] = np.sin(2 * np.pi * train['day'] / 365)
train['day_cos'] = np.cos(2 * np.pi * train['day'] / 365)


test['day_sin'] = np.sin(2 * np.pi * test['day'] / 365)
test['day_cos'] = np.cos(2 * np.pi * test['day'] / 365)


import matplotlib.pyplot as plt
import seaborn as sns


train.info()


train.describe().T


numerical_variables = train.drop(columns=['rainfall'])
numerical_variables = numerical_variables.columns
numerical_variables


# Define a custom color palette
custom_palette = ['#FFA07A', '#CCCCFF','#2ecc11']

# Add 'Dataset' column to distinguish between train and test data
train['Dataset'] = 'Train'
test['Dataset'] = 'Test'

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('darkgrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train, test]), x=variable, y="Dataset", palette=custom_palette)
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN, TEST & ORIGINAL]")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each variable
for variable in numerical_variables:
    create_variable_plots(variable)


train.drop(columns='Dataset',inplace=True)


X = train.drop(columns='rainfall')
y = train['rainfall']


# ðŸ“Œ Step 2: Preprocessing
target_col = "rainfall"  # Adjust if needed
X = train.drop(columns=[target_col])
y = train[target_col]
test_data = test.copy()


# Convert categorical variables (if any)
cat_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=cat_cols)
test_data = pd.get_dummies(test_data, columns=cat_cols)


# Align train & test columns
X, test_data = X.align(test_data, join='left', axis=1, fill_value=0)


# Remove duplicate features
X = X.loc[:, ~X.columns.duplicated()]
test_data = test_data.loc[:, ~test_data.columns.duplicated()]


# Split data for evaluation
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)


def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['lgbm' ,'xgb', 'rf'])

    if model_type == 'lgbm':
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        }
    
    elif model_type == 'xgb':
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        }
    
    elif model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        }

    # Cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_type == 'lgbm':
            model = lgb.LGBMClassifier(**params)
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(**params, use_label_encoder=False)
        elif model_type == 'rf':
            model = RandomForestClassifier(**params)

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]  
        auc_scores.append(roc_auc_score(y_val, preds))

    return np.mean(auc_scores)


optuna.logging.set_verbosity(optuna.logging.WARNING) 


# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("Best Parameters:", study.best_trial.params)


print("Best Parameters:", study.best_trial.params)


import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Initialize models based on the best trial
best_params = study.best_trial.params.copy()  # Create a copy
model_type = best_params.pop('model_type')  # Remove 'model_type'

# Separate parameters for each model
lgbm_params = {k: v for k, v in best_params.items() if k in ['learning_rate', 'num_leaves', 'max_depth', 'min_child_samples', 'subsample', 'colsample_bytree']}
xgb_params = {k: v for k, v in best_params.items() if k in ['learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'n_estimators']}
rf_params = {k: v for k, v in best_params.items() if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']}

# Create models with the correct parameters
best_lgbm = lgb.LGBMClassifier(**lgbm_params)
best_xgb = xgb.XGBClassifier(**xgb_params, use_label_encoder=False)
best_rf = RandomForestClassifier(**rf_params)  # <-- FIXED HERE

estimators = []
if model_type == 'lgbm':
    estimators.append(('lgbm', best_lgbm))
elif model_type == 'xgb':
    estimators.append(('xgb', best_xgb))
elif model_type == 'rf':
    estimators.append(('rf', best_rf))

# Stacking classifier with multiple models
multiple_estimators = [('lgbm', best_lgbm), ('xgb', best_xgb), ('rf', best_rf)]

# Ensure at least one model is included
if not estimators:
    raise ValueError("No valid models found from Optuna tuning!")


multiple_estimators


# Define Stacking Classifier
stacking_model = StackingClassifier(
    estimators=multiple_estimators,
    final_estimator=LogisticRegression(),
    cv=5
)


test_data.isnull().sum()


test_data.fillna(test_data.mean(), inplace=True)


stacking_model.fit(X_train, y_train)
preds = stacking_model.predict_proba(test_data)[:, 1]


preds.shape,test_data.shape 


# ðŸ“Œ Step 6: Prepare Submission
submission = sample_submission.copy()
submission["rainfall"] = preds  # Adjust column name if needed
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv âœ…")

