%%time
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import gc

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold, GroupKFold
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

from ydf import RandomForestLearner, GradientBoostedTreesLearner
import ydf

from lightgbm import LGBMRegressor

import xgboost as xgb
from xgboost import XGBRegressor

from catboost import CatBoostRegressor, Pool


%%time
train = pd.read_csv('../input/playground-series-s5e5/train.csv', index_col=0)
test = pd.read_csv('../input/playground-series-s5e5/test.csv', index_col=0)

print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.kdeplot(ax=axes[0], data=train, x="Calories", fill=True, color="steelblue")
sns.kdeplot(ax=axes[1], data=train, x="Calories", fill=True, hue="Sex")
plt.show();


sns.scatterplot(data=train, x='Duration', y="Calories", hue="Sex")
plt.show();


%%time
def rmsle(actual_values, predicted_values):
    
    # Ensure the inputs are numpy arrays
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    # Ensure that input values are non-negative
    if (actual_values < 0).any() or (predicted_values < 0).any():
        raise ValueError("RMSLE cannot be used with negative values")
    
    # Add a small constant to avoid log(0) errors, and calculate the squared logarithmic errors
    squared_log_errors = (np.log1p(predicted_values) - np.log1p(actual_values))**2
    
    # Calculate the mean of the squared logarithmic errors
    mean_squared_log_error = np.mean(squared_log_errors)
    
    # Return the square root of the mean squared logarithmic error
    return np.sqrt(mean_squared_log_error)


%%time
X = train.drop(columns=["Calories"], axis=1)
X["Sex"] = X["Sex"].astype("category")
y = train["Calories"]

test["Sex"] = test["Sex"].astype("category")

skf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
cat_features = ['Sex']


%%time
scores, test_preds = [], []
for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtest = xgb.DMatrix(test, enable_categorical=True)

    xgb_md = xgb.train({'device': "cuda",
                        # "objective": "reg:squaredlogerror",
                        'learning_rate': 0.05,             
                        'max_depth': 10,                   
                        'colsample_bytree': 0.8,                     
                        'reg_lambda': 8,                     
                        'subsample': 1,                 
                        'reg_alpha': 1, 
                        'n_jobs': -1}, 
                        dtrain, 
                        num_boost_round=500, 
                        evals=[(dvalid, 'validation')], 
                        # early_stopping_rounds=100,
                        verbose_eval=False)
    xgb_pred = xgb_md.predict(dvalid)

    score = rmsle(y_val, xgb_pred)
    print(f"Fold {i + 1} RMSLE: {score:.4f}")
    scores.append(score)

    test_preds.append(xgb_md.predict(dtest))

print(f"CV RMSLE: {np.mean(scores):.4f} Â± {np.std(scores):.4f}")


%%time
submission = pd.read_csv('../input/playground-series-s5e5/sample_submission.csv')
submission["Calories"] = np.mean(test_preds, axis=0)
submission.head()


%%time
submission.to_csv("baseline_sub_1.csv", index=False)

del submission
gc.collect()


%%time
X["BMI"] = X["Weight"] / (X["Height"] / 100) ** 2
y = np.log1p(train["Calories"])

test["BMI"] = test["Weight"] / (test["Height"] / 100) ** 2

cat_params = {'loss_function': 'RMSE', 'iterations': 1000, 'depth': 8, 'task_type': 'GPU'}
scores, test_preds_df = [], []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    print(f"Working on Fold {i+1}")
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y[train_index], y.iloc[test_index]
                
    model_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    eval_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    test_pool = Pool(data=test, cat_features=cat_features)
            
    bst = CatBoostRegressor(**cat_params).fit(model_pool, eval_set=eval_pool, verbose=0)
    cat_pred = np.clip(np.expm1(bst.predict(eval_pool)), 0, None)
    
    score = rmsle(np.expm1(y_val), cat_pred)
    print(f"Fold {i+1} RMSLE: {score}")
    scores.append(score)

    test_preds_df.append(np.clip(np.expm1(bst.predict(test_pool)), 0, None))

cat_score = np.mean(scores)
print(f"CatBoost RMSLE: {cat_score}")


%%time
submission = pd.read_csv('../input/playground-series-s5e5/sample_submission.csv')
submission["Calories"] = np.mean(test_preds_df, axis=0)
submission.head()


%%time
submission.to_csv("baseline_sub_2.csv", index=False)

del submission
gc.collect()

