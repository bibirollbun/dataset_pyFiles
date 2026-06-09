import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


print(train_df.shape)
print(test_df.shape)


train_df.head(5)


train_df.info()


train_df[(train_df["country"] == "Canada") & (train_df['product'] == "Holographic Goose") & (train_df["store"] == "Discount Stickers")]


train_df.isna().sum().sort_values(ascending=False)


test_df.isna().sum().sort_values(ascending=False)


train_df = train_df.drop_duplicates().dropna()
train_df.shape


train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

train_df['Year'] = train_df['date'].dt.year
train_df['Month'] = train_df['date'].dt.month
train_df['Day'] = train_df['date'].dt.day

test_df['Year'] = test_df['date'].dt.year
test_df['Month'] = test_df['date'].dt.month
test_df['Day'] = test_df['date'].dt.day

train_df.drop('date',axis=1,inplace=True)
test_df.drop('date',axis=1,inplace=True)


train_df['num_sold'] = np.log1p(train_df['num_sold'])


train_df


train_df = train_df.drop('id', axis = 1)
num_cols = list(train_df.select_dtypes(exclude=['object']).columns.difference(['num_sold']))
cat_cols = list(train_df.select_dtypes(include=['object']).columns)

num_cols_test = list(test_df.select_dtypes(exclude=['object']).columns.difference(['id']))
cat_cols_test = list(test_df.select_dtypes(include=['object']).columns)


num_cols


cat_cols


from sklearn.preprocessing import LabelEncoder
# Initialize LabelEncoder
label_encoders = {col: LabelEncoder() for col in cat_cols}

# Apply LabelEncoder to each categorical column
for col in cat_cols:
    train_df[col] = label_encoders[col].fit_transform(train_df[col])
    test_df[col] = label_encoders[col].transform(test_df[col])


train_df


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = train_df.corr()

# Plot the heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
X = train_df.drop(['num_sold'], axis=1)
y = train_df['num_sold']
test = test_df.drop(['id'],axis=1)

# Split datainto training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
import optuna

# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def objective(trial):
    param = {
        'tree_method':'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
        'sampling_method': 'gradient_based',
        'lambda': trial.suggest_loguniform('lambda', 7.0, 17.0),
        'alpha': trial.suggest_loguniform('alpha', 7.0, 17.0),
        'eta': trial.suggest_categorical('eta', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'gamma': trial.suggest_categorical('gamma', [18, 19, 20, 21, 22, 23, 24, 25]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'colsample_bynode': trial.suggest_categorical('colsample_bynode', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 8, 600),  
        'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7]),  
        'subsample': trial.suggest_categorical('subsample', [0.5,0.6,0.7,0.8,1.0]),
        'random_state': 42
    }

    model = XGBRegressor(**param)  
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    
    predict = model.predict(X_test)
    
    mape_score = mape(y_test, predict)
    
    return mape_score


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50,  timeout=600)
    
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# best MAPE = 0.019 from Optuna
xgb_parameters = {
    'lambda': 16.984616421474673,
    'alpha': 7.2944965643293935,
    'eta': 0.7,
    'gamma': 19,
    'learning_rate': 0.008,
    'colsample_bytree': 0.8,
    'colsample_bynode': 1.0,
    'n_estimators': 941,
    'min_child_weight': 283,
    'max_depth': 6,
    'subsample': 0.5
}


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

# Cross-validation for XGBRegressor
def cross_val_xgbr_mape(X, y, test, n_splits=5, **xgb_parameters):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        else:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

        # Initialize and train the model
        model = XGBRegressor(random_state=42, **xgb_parameters)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(test))

    # Average predictions over all folds
    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

average_mape, xgb_preds = cross_val_xgbr_mape(X, y, test, n_splits=5, **xgb_parameters)

print(f"Average MAPE across folds: {average_mape:.4f}")

# Save predictions for submission
submission = pd.DataFrame({'id': test_df['id'], 'num_sold': np.expm1(xgb_preds)})
print(submission.head())
submission.to_csv('submission_xgb.csv', index=False)


from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
import optuna

# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def objective(trial):
    
    param = {
        'tree_method': 'gpu_hist',
        "verbosity": -1,
        'sampling_method': 'gradient_based',
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7]),
        'subsample': trial.suggest_categorical('subsample', [0.5,0.6,0.7,0.8,1.0]),
        'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "random_state": 42
    }

    model = LGBMRegressor(**param)  
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    predict = model.predict(X_test)
    
    mape_score = mape(y_test, predict)
    
    return mape_score


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50,  timeout=600)
    
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# MAPE = 0.013165753684445974

lgbm_parameters = {
    'lambda_l1': 1.9110308129857972e-06,
    'lambda_l2': 1.4720943722601327e-08,
    'num_leaves': 159,
    'learning_rate': 0.018,
    'colsample_bytree': 0.9,
    'feature_fraction': 0.8661665398937026,
    'bagging_fraction': 0.8916098848078859,
    'max_depth': 7,
    'subsample': 0.7,
    'n_estimators': 998,
    'bagging_freq': 5,
    'min_child_samples': 26
}


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

# Cross-validation for LGBMRegressor
def cross_val_lgbm_mape(X, y, test, n_splits=5, **lgb_parameters):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        else:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

        # Initialize and train the model
        model = LGBMRegressor(random_state=42, **lgbm_parameters)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(test))

    # Average predictions over all folds
    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

average_mape, lgb_preds = cross_val_lgbm_mape(X, y, test, n_splits=5, **lgbm_parameters)

print(f"Average MAPE across folds: {average_mape:.4f}")

# Save predictions for submission
submission = pd.DataFrame({'id': test_df['id'], 'num_sold': np.expm1(lgb_preds)})
print(submission.head())
submission.to_csv('submission_lgbm.csv', index=False)

