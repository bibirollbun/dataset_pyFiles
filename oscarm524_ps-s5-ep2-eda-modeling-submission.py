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
from xgboost import XGBRegressor


%%time
train = pd.read_csv('../input/playground-series-s5e2/train.csv', index_col=0)
test = pd.read_csv('../input/playground-series-s5e2/test.csv', index_col=0)

print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


print(train.info())
print("\n")
print(test.info())


sns.histplot(data=train, x='Price')
plt.show();


cat_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]

for col in cat_cols:
    train[col] = train[col].fillna('Unknown')
    train[col] = train[col].astype('category')

    test[col] = test[col].fillna('Unknown')
    test[col] = test[col].astype('category')


fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.boxplot(data=train, x="Brand", y="Price", ax=axes[0])
sns.boxplot(data=train, x="Material", y="Price", ax=axes[1]);


fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.boxplot(data=train, x="Size", y="Price", ax=axes[0])
sns.boxplot(data=train, x="Laptop Compartment", y="Price", ax=axes[1]);


fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.boxplot(data=train, x="Waterproof", y="Price", ax=axes[0])
sns.boxplot(data=train, x="Style", y="Price", ax=axes[1]);


plt.figure(figsize=(10, 8))

sns.boxplot(data=train, x="Color", y="Price");


%%time
skf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

ydf.verbose(-1)
scores, ydf_test_preds = [], []
for i, (train_index, test_index) in enumerate(skf.split(train)):

    print(f"------------ Working on Fold {i} ------------")
            
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    
    ydf_md = RandomForestLearner(label='Price', 
                                 task=ydf.Task.REGRESSION, 
                                 num_threads=10, 
                                 num_trees=1000).train(X_train)
    ydf_pred = ydf_md.predict(X_test)

    score = mean_squared_error(X_test['Price'], ydf_pred, squared=False)
    print('Fold:', i, 'RMSE:', score)
    scores.append(score)

    ydf_test_preds.append(ydf_md.predict(test))

ydf_gb_oof_score = np.mean(scores)  
ydf_gb_std = np.std(scores)
print(f"The 5-fold average oof RMSE score of the RandomForestLearner model is {ydf_gb_oof_score}")
print(f"The 5-fold std oof RMSE score of the RandomForestLearner model is {ydf_gb_std}")


%%time
submission = pd.read_csv('../input/playground-series-s5e2/sample_submission.csv')
submission["Price"] = np.mean(ydf_test_preds, axis=0)

display(submission.head())

submission.to_csv("baseline_RF_sub.csv", index=False)


%%time
ydf.verbose(-1)
scores, ydf_test_preds = [], []
for i, (train_index, test_index) in enumerate(skf.split(train)):

    print(f"------------ Working on Fold {i} ------------")
            
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    
    ydf_md = GradientBoostedTreesLearner(label='Price', 
                                         task=ydf.Task.REGRESSION, 
                                         num_threads=10, 
                                         num_trees=1000).train(X_train)
    ydf_pred = ydf_md.predict(X_test)

    score = mean_squared_error(X_test['Price'], ydf_pred, squared=False)
    print('Fold:', i, 'RMSE:', score)
    scores.append(score)

    ydf_test_preds.append(ydf_md.predict(test))

ydf_gb_oof_score = np.mean(scores)  
ydf_gb_std = np.std(scores)
print(f"The 5-fold average oof RMSE score of the GradientBoostedTreesLearner model is {ydf_gb_oof_score}")
print(f"The 5-fold std oof RMSE score of the GradientBoostedTreesLearner model is {ydf_gb_std}")


%%time
submission = pd.read_csv('../input/playground-series-s5e2/sample_submission.csv')
submission["Price"] = np.mean(ydf_test_preds, axis=0)

display(submission.head())

submission.to_csv("baseline_GB_sub.csv", index=False)


%%time
X = train.drop(columns=["Price"], axis=1)
y = train["Price"]

xgb_params = {'objective': 'reg:absoluteerror',
 'n_estimators': 675,
 'max_depth': 12,
 'learning_rate': 0.0647368285818005,
 'gamma': 5.581559809586505,
 'min_child_weight': 31,
 'colsample_bytree': 0.467360303051405,
 'n_jobs': -1,
 'enable_categorical': True}

scores, xgb_test_preds = list(), list()
skf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y.iloc[test_index]

    xgb_md = XGBRegressor(**xgb_params).fit(X_train, y_train)
    preds = xgb_md.predict(X_test)
    xgb_test_preds.append(xgb_md.predict(test))

    score = mean_squared_error(y_test, preds, squared=False)
    print(f"Fold {i+1} - RMSE: {score}")
    scores.append(score)

xgb_oof_score = np.mean(scores)  
xgb_std = np.std(scores)
print(f"The 5-fold average oof RMSE score of the XGBRegressor model is {xgb_oof_score}")
print(f"The 5-fold std oof RMSE score of the XGBRegressor model is {xgb_std}")


%%time
submission = pd.read_csv('../input/playground-series-s5e2/sample_submission.csv')
submission["Price"] = np.mean(xgb_test_preds, axis=0)

display(submission.head())

submission.to_csv("baseline_xgb_sub.csv", index=False)

