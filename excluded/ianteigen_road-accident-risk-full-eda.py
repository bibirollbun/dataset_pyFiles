import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

import warnings, os, gc, sys, math, json, random, itertools

from scipy import stats
from scipy.stats import ks_2samp


warnings.filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")
sns.set_palette("crest")
pd.set_option("display.max_columns", 100)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

cat_cols = ['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']
num_cols = ['num_lanes','curvature','speed_limit','num_reported_accidents']
target = 'accident_risk'
train['accident_risk_bins']=train[target]
#Binning the target to make it easier to visualize
train['accident_risk_bins'] = pd.cut(
    train[target],
    bins=[0, 0.25, 0.5, 0.75, 1],
    labels=["Q1", "Q2", "Q3", "Q4"],
    include_lowest=True
)


def quick_overview(df, name):
    print(f"\n{name.upper()} SHAPE: {df.shape}")
    display(df.head())
    display(df.describe(include="all").T)

quick_overview(train, "train")
quick_overview(test , "test")

print(f"Duplicate rows (train): {train.duplicated().sum()}  |  (test): {test.duplicated().sum()}")
print("Number of missing values:")
train.isnull().sum() 



def plot_kde(data, name, columns=None, figsize=(8, 4), fill=True, max_density=None):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    columns = data.select_dtypes(include='number').columns.tolist()
    plt.figure(figsize=figsize)
    for col in columns:
        sns.kdeplot(data[col], label=col, linewidth=2,clip=(0, None),linestyle="-.")
        
    if max_density is not None:
        plt.ylim(0, max_density)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("KDE PLOT")
plot_kde(train[target], "Accident Risk Distribution")

print("HISTOGRAM")
sns.histplot(train[target], kde=False)
plt.title(f"Accident Risk Distribution")
plt.xlabel("Accident Risk")
plt.ylabel("Count")

plt.show()


n_cols = 2
n_rows = math.ceil(len(cat_cols) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    sns.countplot(data=train, x=col, ax=ax)
    ax.set_title(f"{col.capitalize()} Distribution", fontsize = 16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize = 14)

        
# Turn off any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

for col in cat_cols:
    print(train[col].value_counts(normalize=True).rename("proportion"))


for col in cat_cols:
    print(train.groupby(col)["accident_risk"].mean())


n_cols = 3
n_rows = math.ceil(len(num_cols) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()


for i, col in enumerate(num_cols):
    if col == 'curvature':
        sns.histplot(train[col], ax=axes[i], kde=False)
    elif col == 'speed_limit':
        sns.histplot(train[col], ax=axes[i], kde=False, binwidth = 3)
    else:
        sns.histplot(train[col], ax=axes[i], kde=False, binwidth=0.3)
    axes[i].set_title(f"{col.capitalize()} Distribution")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")


for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.show()


outlier_summary = {}
for col in num_cols:
    z = np.abs(stats.zscore(train[col]))
    outlier_summary[col] = (z>3).sum()   # 3-σ rule

pd.Series(outlier_summary, name="#outliers (>3σ)").sort_values(ascending=False).to_frame().style.bar()


feat = train['num_reported_accidents']
z    = np.abs(stats.zscore(feat, nan_policy="omit"))
outlier_mask = (z > 3)

outliers= train.loc[outlier_mask, ['num_reported_accidents', "accident_risk"]]
print(f"Average Accidents Risk of dataset: {train['accident_risk'].mean()}")
print(f"Accident Risk of num_reported_accidents outliers: {outliers['accident_risk'].mean()}")





fig, axes = plt.subplots(math.ceil(len(num_cols)/3), 3, figsize=(15,4*math.ceil(len(num_cols)/3)))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x="accident_risk_bins", y=col, data=train, ax=axes[i], showfliers=False
)
    axes[i].set_title(f"{col} by Accident Risk Bins")


for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


num_cols.append('accident_risk')
corr = train[num_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,cbar_kws={"shrink": 0.8})

plt.xticks(rotation=90, ha='right', fontsize=8) 
plt.yticks(rotation=0, fontsize=8)
plt.title("Pearson Correlation")
plt.show()


target_corr = train[num_cols].corr()["accident_risk"].drop(
    "accident_risk").sort_values()
display(target_corr.to_frame("corr_with_target").style.bar(vmin=-1,vmax=1))
num_cols.remove('accident_risk')


import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import gc
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from pandas.errors import PerformanceWarning
from sklearn.metrics import mean_squared_error
from itertools import combinations
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import copy

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

cat_cols = ['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']
num_cols = ['num_lanes','curvature','speed_limit','num_reported_accidents']
target = 'accident_risk'


new_cat_cols=[]

##Interaction features
for col1, col2 in combinations(cat_cols, 2):
    new_col_name = f"{col1}_{col2}"
    train[new_col_name] = train[col1].astype(str) + "_" + train[col2].astype(str)
    test[new_col_name] = test[col1].astype(str) + "_" + test[col2].astype(str)
    new_cat_cols.append(new_col_name)
for i in range(len(num_cols)):
    for j in range(i, len(num_cols)):
        col1 = num_cols[i]
        col2 = num_cols[j]
        # Create interaction features
        train[f'{col1}_x_{col2}'] = train[col1] * train[col2]
        test[f'{col1}_x_{col2}'] = test[col1] * test[col2]
        train[f'{col1}_squared']= train[col1] * train[col1]
        test[f'{col1}_squared']= test[col1] * test[col1]
        # Create ratio features, handle division by zero
        if col1 != col2:
            train[f'{col1}_div_{col2}'] = train[col1] / (train[col2] + 1e-6)
            test[f'{col1}_div_{col2}'] = test[col1] / (test[col2] + 1e-6)
train["speed_bin"]= pd.cut(train["speed_limit"], bins=[0, 35, 50, 70], labels=["low", "medium", "high"]
    )
    
train["curvature_bin"] = pd.qcut(train["curvature"], q=4, labels=["very_low", "low", "high", "very_high"]
    )
test["speed_bin"]= pd.cut(test["speed_limit"], bins=[0, 35, 50, 70], labels=["low", "medium", "high"]
    )
    
test["curvature_bin"] = pd.qcut(test["curvature"], q=4, labels=["very_low", "low", "high", "very_high"]
    )
new_cat_cols.append('speed_bin')
new_cat_cols.append('curvature_bin')

for col in cat_cols:
    train[col]=train[col].astype('category')
    test[col]=test[col].astype('category')
for col in new_cat_cols:
    train[col]=train[col].astype('category')
    test[col]=test[col].astype('category')
cat_cols.extend(new_cat_cols)
X_cat = train[cat_cols]
xgb_params = {
    'n_estimators': 600,         
    'max_leaves': 211,            
    'min_child_weight': 1.5,     
    'max_depth': 12,               
    'grow_policy': 'lossguide',   
    'learning_rate': 0.04,      
    'tree_method': 'hist',        
    'subsample': 0.85,            
    'colsample_bylevel': 0.6787051322531533,     
    'colsample_bytree': 0.6843905004927857,       
    'colsample_bynode': 0.442116057736592,     
    'sampling_method': 'uniform',  
    'reg_alpha': 2.5,             
    'reg_lambda': 0.8,            
    'enable_categorical': True,    
    'max_cat_to_onehot': 1,       
    'device': 'cuda',            
    'n_jobs': -1,                 
    'random_state': 42,     
    'verbosity': 0,               
}
lgbm_params = {
    'learning_rate': 0.04,
    'num_leaves': 79, 
    'max_depth': 12,
    'feature_fraction': 0.8933016300882094,
    'bagging_fraction': 0.9754103048412501,
    'bagging_freq': 7, 
    'min_child_samples': 40,
    'enable_categorical': True,   
    'lambda_l1': 7.10897934678165e-07,
    'lambda_l2': 7.81564014894075e-08,
    'random_state' : 42,
    'n_jobs' : -1,
    'verbosity': -1,
    'n_estimators': 600
}
print('XGB')
X=train.drop(columns=['id','accident_risk'])
X_test=test.drop(columns=['id'])
y=train['accident_risk']
model = XGBRegressor(**xgb_params)
xgb_oof_preds = np.zeros(len(X))
xgb_models, xgb_scores=[],[]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X, y):
        print('Fold:', len(xgb_models) + 1)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        xgb_oof_preds[val_idx] = model.predict(X_val)
        acc = mean_squared_error(y_val, model.predict(X_val), squared=False)
        xgb_scores.append(acc), xgb_models.append(model)
        print('Accuracy:', acc)
print('XGB ACCURACY: ', np.mean(xgb_scores))

print('LGBM')
print()
lgbm_model = LGBMRegressor(**lgbm_params)
lgbm_models, lgbm_scores=[],[]
lgbm_oof_preds = np.zeros(len(X))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X, y):
        print('Fold:', len(lgbm_models) + 1)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        lgbm_model.fit(X_train, y_train)
        lgbm_oof_preds[val_idx]=lgbm_model.predict(X_val)
        acc = mean_squared_error(y_val, lgbm_model.predict(X_val), squared=False)
        lgbm_scores.append(acc), lgbm_models.append(lgbm_model)
        print('Accuracy:', acc)
print('LGBM ACCURACY: ', np.mean(lgbm_scores))

print('CATBOOST')

cb_model=CatBoostRegressor(
    border_count= 28,
    colsample_bylevel= 0.19459088572914465,
    depth= 5,
    iterations= 600,
    l2_leaf_reg= 31.236169478676036,
    learning_rate= 0.1332583504067626,
    min_child_samples= 189,
    random_state= 0,
    random_strength= 0.8517786189616939,
    subsample= 0.3192330024411618,
    verbose= False,
    cat_features = cat_cols)
cb_oof_preds = np.zeros(len(X))
cb_models, cb_scores=[],[]
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, val_idx in kf.split(X, y):
        print('Fold:', len(cb_models) + 1)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        cb_model.fit(X_train, y_train)
        cb_oof_preds[val_idx] = cb_model.predict(X_val)
        acc = mean_squared_error(y_val, cb_model.predict(X_val), squared=False)
        cb_scores.append(acc), cb_models.append(cb_model)
        print('Accuracy:', acc)
print('CATBOOST ACCURACY: ', np.mean(cb_scores))


X_stack = pd.DataFrame({
    'xgb' : xgb_oof_preds,
    'lgbm' : lgbm_oof_preds,
    'cb' : cb_oof_preds
})
xgb_test_preds = sum(model.predict(X_test) for model in xgb_models) / len(xgb_models)
lgbm_test_preds = sum(lgbm_model.predict(X_test) for lgbm_model in lgbm_models) / len(lgbm_models)
cb_test_preds = sum(cb_model.predict(X_test) for cb_model in cb_models) / len(cb_models)


def objective(trial):
    # Suggest weights for the ensemble
    w_lgbm = trial.suggest_float("w_lgbm", 0, 1)
    w_cb = trial.suggest_float("w_cb", 0, 1-w_lgbm)
    w_xgb = 1 - w_lgbm - w_cb   # ensures sum to 1

    final_preds = (w_lgbm * X_stack['lgbm'] +w_xgb*X_stack['xgb'] + w_cb * X_stack['cb'])
    score = mean_squared_error(y, final_preds, squared = False) 
    return score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)
print("Best score:", study.best_value)
print("Best params:", study.best_params)

w_lgbm = 0.45
w_cb = 0.15
w_xgb = 1-w_lgbm - w_cb
preds = w_lgbm * lgbm_test_preds + w_xgb * xgb_test_preds + w_cb * cb_test_preds
submission = pd.DataFrame({'id': test['id'], 'accident_risk': preds})
submission.to_csv('submission.csv', index=False)
display(submission.head())

