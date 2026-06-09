# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# !pip install catboost
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


ID = test['id']
test.drop('id',inplace=True,axis=1)
train.drop('id',inplace=True,axis=1)


print(train.shape)
print(test.shape)
train.head()


sns.histplot(data=train, x='num_sold', bins=30, kde=True, color='blue', alpha=0.6)
plt.title("Distribution of 'Target'")
plt.xlabel("num_sold")
plt.ylabel("Frequency")
plt.show()


train["y"] = np.log1p( train["num_sold"] +0.001)


train = train.sort_values('date').reset_index(drop=True) 
test = test.sort_values('date').reset_index(drop=True)

for dataset in (train,test):
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['Day'] = dataset.date.dt.day
    dataset['Month'] = dataset.date.dt.month
    dataset['Year'] = dataset.date.dt.year
    dataset['Week'] = dataset.date.dt.isocalendar().week


categorical_columns = ["product","store","country"]
# Print top 10 unique value counts for each categorical column
for column in categorical_columns+ ["Year"]:
    print(f"\nTop value counts in '{column}':\n{train[column].value_counts().head(10)}")


RMV = ["id", "date", "num_sold"]
FEATURES = [c for c in train.columns if not c in RMV]
combined = pd.concat([train, test], axis=0, ignore_index=True)

CATS = []
HIGH_CARDINALITY = []
print(f"THE {len(FEATURES)} BASIC FEATURES ARE:")

for c in FEATURES:
    ftype = "numerical"
    if combined[c].dtype == "object":
        CATS.append(c)
        combined[c] = combined[c].fillna("NAN")
        combined[c], _ = combined[c].factorize()
        combined[c] -= combined[c].min()
        ftype = "categorical"
    if combined[c].dtype == "int64":
        combined[c] = combined[c].astype("int32")
    elif combined[c].dtype == "float64":
        combined[c] = combined[c].astype("float32")
        
    n = combined[c].nunique()
    print(f"{c} ({ftype}) with {n} unique values")
    if n >= 9: HIGH_CARDINALITY.append(c)
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()
print("\nTHE FOLLOWING HAVE 9 OR MORE UNIQUE VALUES:", HIGH_CARDINALITY)



# Fill missing target values with mean
train['y'] =train['y'].fillna(train['y'].mean())


# !pip install optuna-integration[lightgbm]

import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
import optuna
import optuna.integration.lightgbm as lgb_optuna

X = train.drop(["num_sold","date","y"],axis=1).values
y = train['y'].values


# Set up K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))


# LightGBM Optuna Tuning
print("Optimizing LightGBM parameters...")
def objective_lightgbm(trial):
    param = {
        'device' : 'gpu',
        'objective': 'regression',
        'metric': 'mape',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'feature_pre_filter': False,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }

    model = lgb.LGBMRegressor(**param, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    return mean_absolute_percentage_error(y, preds)

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lightgbm, n_trials=50)
best_params_lgb = study_lgb.best_params



# CatBoost Optuna Tuning
print("Optimizing CatBoost parameters...")
def objective_catboost(trial):
    param = {
        'task_type':'GPU',
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-4, 10),
        'random_strength': trial.suggest_uniform('random_strength', 0.1, 2.0),
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'verbose': 0
    }

    model = CatBoostRegressor(**param, random_state=42)
    model.fit(X, y, verbose=False)
    preds = model.predict(X)
    return mean_absolute_percentage_error(y, preds)

study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_catboost, n_trials=50)
best_params_cat = study_cat.best_params


# XGBoost Optuna Tuning
print("Optimizing XGBoost parameters...")
def objective_xgboost(trial):
    param = {
        'tree_method' :'gpu_hist',
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-4, 10),
        'alpha': trial.suggest_loguniform('alpha', 1e-4, 10)
    }

    model = XGBRegressor(**param, random_state=42)
    model.fit(X, y, verbose=False)
    preds = model.predict(X)
    return mean_absolute_percentage_error(y, preds)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgboost, n_trials=50)
best_params_xgb = study_xgb.best_params


# K-Fold OOF Predictions
print("Performing K-Fold OOF predictions...")
for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"Training fold {fold + 1} for models...")
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**best_params_lgb, random_state=42)
    lgb_model.fit(X_train, y_train)
    oof_lgb[valid_idx] = lgb_model.predict(X_valid)

    # CatBoost
    cat_model = CatBoostRegressor(**best_params_cat, random_state=42)
    cat_model.fit(X_train, y_train)
    oof_cat[valid_idx] = cat_model.predict(X_valid)

    # XGBoost
    xgb_model = XGBRegressor(**best_params_xgb, random_state=42)
    xgb_model.fit(X_train, y_train)
    oof_xgb[valid_idx] = xgb_model.predict(X_valid)

# Combine predictions using a weighted average
# Adjust weights based on validation performance of individual models
weights = [0.4, 0.3, 0.3]  # Example weights for LightGBM, CatBoost, XGBoost
oof_ensemble = (weights[0] * oof_lgb +
                weights[1] * oof_cat +
                weights[2] * oof_xgb)

# Evaluate the ensemble model using MAPE
mape = mean_absolute_percentage_error(y, oof_ensemble)
print(f"MAPE of the ensemble model: {mape}")


 test = test.drop(["y","num_sold","date"],axis=1)
# LightGBM
lgb_model = lgb.LGBMRegressor(**best_params_lgb, random_state=42)
lgb_model.fit(X, y)
lgb_predict = lgb_model.predict(test)

# CatBoost
cat_model = CatBoostRegressor(**best_params_cat, random_state=42)
cat_model.fit(X, y)
cat_predict = cat_model.predict(test)

# XGBoost
xgb_model = XGBRegressor(**best_params_xgb, random_state=42)
xgb_model.fit(X, y)
xgb_predict = xgb_model.predict(test)

# final preds
ensembled_preds = (lgb_predict + cat_predict + xgb_predict)/3
final_ensembled_preds = np.expm1(ensembled_preds)





# X=train.drop(["num_sold","date","y"],axis=1)
# y=train["y"]
# test = test.drop(["num_sold","date","y"],axis=1)
# model.fit(X,y)
# Pred = model.predict(test)
# FinalPred =  np.expm1(Pred)


submission = pd.DataFrame({"id": ID ,"num_sold": final_ensembled_preds})
submission.to_csv('Submission.csv',index=False)




