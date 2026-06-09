%%time
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

from tqdm.notebook import tqdm
import gc

import warnings
warnings.filterwarnings('ignore')

from functools import partial
import scipy as sp

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, cohen_kappa_score, log_loss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

import xgboost as xgb
from xgboost import XGBClassifier

from catboost import CatBoostClassifier, Pool


%%time
train = pd.read_csv('../input/playground-series-s5e6/train.csv', index_col=0)
test = pd.read_csv('../input/playground-series-s5e6/test.csv', index_col=0)

print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


train.head()


test.head()


%%time
train['Fertilizer Name'].value_counts(normalize=True)


%%time
print('There are', sum(train.drop(columns=['Fertilizer Name']).duplicated()), 'duplicated observations in the train dataset')
print('There are', sum(test.duplicated()), 'duplicated observations in the test dataset')


%%time
to_consider = train.drop(columns=["Fertilizer Name"], axis=1).columns.tolist()

train_dup = train.drop(columns=["Fertilizer Name"], axis=1).drop_duplicates()
test_dup = test.drop_duplicates()
duplicates = pd.merge(train_dup, test_dup, on=to_consider)

print('There are', duplicates.shape[0], 'rows that appear in the train and test dataset.\n')


%%time
print("Missing values in the train dataset \n")
print(f"{train.isna().sum()}\n")

print("Missing values in the test dataset \n")
print(test.isna().sum())


fig, ax = plt.subplots(1, 2, figsize=(18, 7))

sns.boxplot(data=train, x="Fertilizer Name", y="Temparature", ax=ax[0], color="steelblue")
sns.boxplot(data=train, x="Fertilizer Name", y="Humidity", ax=ax[1], color="orange");


fig, ax = plt.subplots(1, 2, figsize=(18, 7))

sns.boxplot(data=train, x="Fertilizer Name", y="Moisture", ax=ax[0], color="steelblue")
sns.boxplot(data=train, x="Fertilizer Name", y="Nitrogen", ax=ax[1], color="orange");


fig, ax = plt.subplots(1, 2, figsize=(18, 7))

sns.boxplot(data=train, x="Fertilizer Name", y="Potassium", ax=ax[0], color="steelblue")
sns.boxplot(data=train, x="Fertilizer Name", y="Phosphorous", ax=ax[1], color="orange");


%%time
def apk(actual, predicted, k = 10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k = 10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def selecting_top_3(arr): 
    n = arr.shape[0]
    out = np.zeros((n, 3))
    for i in range(0, n):
        out[i, ] = arr[i,].argsort()[::-1][:3]
    return out.astype('int32')


%%time
X = train.drop(columns=["Fertilizer Name"], axis=1)
X["Soil Type"] = X["Soil Type"].astype("category")
X["Crop Type"] = X["Crop Type"].astype("category")
y = train["Fertilizer Name"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

test["Soil Type"] = test["Soil Type"].astype("category")
test["Crop Type"] = test["Crop Type"].astype("category")

del y
gc.collect()

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


%%time
cat_params = {'loss_function': 'MultiClass',
 'iterations': 2000,
 'depth': 5,
 'bagging_temperature': 0.2,             
 'task_type': 'GPU'}

cat_features = ["Soil Type", "Crop Type"]
test_pool = Pool(data=test, cat_features=cat_features)

scores, test_preds = [], []
for i, (train_ix, test_ix) in enumerate(skf.split(X, y_encoded)):
        
    X_train, X_val = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[test_ix]
                
    model_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    eval_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)

    cat_md = CatBoostClassifier(**cat_params).fit(model_pool, eval_set=eval_pool, verbose=0, early_stopping_rounds=100)
    cat_pred = selecting_top_3(cat_md.predict_proba(X_val))
    
    score = mapk(y_val.reshape(-1, 1), cat_pred, k=3)
    print(f"Fold {i+1} MAP is {score:.4f}")
    scores.append(score)

    del model_pool, eval_pool
    gc.collect()

    test_preds.append(cat_md.predict_proba(test))
    
cat_cv_mean = np.mean(scores)
cat_cv_sd = np.std(scores)
print(f"The oof map3 score of the CatBoost model is {cat_cv_mean:.4f}")


%%time
pred_agg = 0
for i in range(0, len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)
    
test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)

submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


%%time
submission.to_csv('baseline_cat_sub_1.csv')

del submission, test_preds
gc.collect()


%%time
xgb_params = {'device': 'cuda',
 'max_depth': 5,
 'learning_rate': 0.05,
 'min_child_weight': 50,
 'n_jobs': -1,
 'enable_categorical': True,
 'early_stopping_rounds': 200}

cat_features = ["Soil Type", "Crop Type"]

scores, test_preds = [], []
for i, (train_ix, test_ix) in enumerate(skf.split(X, y_encoded)):

    X_train, X_val = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[test_ix]
    
    xgb_md = XGBClassifier(**xgb_params, 
                           n_estimators=2000, 
                           random_state=42).fit(X_train, y_train,
                           eval_set=[(X_val, y_val)],
                           # early_stopping_rounds=200,
                           verbose=False)
    xgb_pred = selecting_top_3(xgb_md.predict_proba(X_val))

    score = mapk(y_val.reshape(-1, 1), xgb_pred, k=3)
    print(f"Fold {i+1} MAP is {score:.4f}")
    scores.append(score)

    del X_train, X_val, y_train, y_val
    gc.collect()

    test_preds.append(xgb_md.predict_proba(test))

xgb_cv_mean = np.mean(scores)
xgb_cv_sd = np.std(scores)
print(f"The oof map3 score of the XGBoost model is {xgb_cv_mean:.4f}")


%%time
pred_agg = 0
for i in range(0, len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)
    
test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)

submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


%%time
submission.to_csv('baseline_xgb_sub_1.csv')

del submission, test_preds
gc.collect()


%%time
lgb_params = {'learning_rate': 0.05,
 'n_estimators': 2000,
 'max_depth': 5,
 'num_leaves': 50,
 'verbose': -1,
 'n_jobs': -1,
 'device': 'gpu'}

cat_features = ["Soil Type", "Crop Type"]

scores, test_preds = [], []
for i, (train_ix, test_ix) in enumerate(skf.split(X, y_encoded)):

    X_train, X_val = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[test_ix]
    
    lgb_md = LGBMClassifier(**lgb_params).fit(X_train, y_train, eval_set=[(X_val, y_val)],  
                                              callbacks=[early_stopping(stopping_rounds=200, verbose=None)])
    lgb_pred = selecting_top_3(lgb_md.predict_proba(X_val))

    score = mapk(y_val.reshape(-1, 1), lgb_pred, k=3)
    print(f"Fold {i+1} MAP is {score:.4f}")
    scores.append(score)

    del X_train, X_val, y_train, y_val
    gc.collect()

    test_preds.append(lgb_md.predict_proba(test))

lgb_cv_mean = np.mean(scores)
lgb_cv_sd = np.std(scores)
print(f"The oof map3 score of the LGBM model is {lgb_cv_mean:.4f}")


%%time
pred_agg = 0
for i in range(0, len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)
    
test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)

submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


%%time
submission.to_csv('baseline_lgb_sub_1.csv')

del submission, test_preds
gc.collect()


%%time
results_df = pd.DataFrame()
results_df["Model"] = ["CatBoost", "XGBoost", "LGBM"]
results_df["10-fold oof MAP@3"] = [round(cat_cv_mean, 4), round(xgb_cv_mean, 4), round(lgb_cv_mean, 4)]
results_df

