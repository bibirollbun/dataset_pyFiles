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


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

cat_cols = ['gender','ethnicity','education_level','income_level','smoking_status','employment_status',
            'family_history_diabetes','hypertension_history','cardiovascular_history']
num_cols = ['age','alcohol_consumption_per_week','physical_activity_minutes_per_week','diet_score','sleep_hours_per_day',
            'screen_time_hours_per_day','bmi','waist_to_hip_ratio','systolic_bp','diastolic_bp','heart_rate',
            'cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides']
target = 'diagnosed_diabetes'


def quick_overview(df, name):
    print(f"\n{name.upper()} SHAPE: {df.shape}")
    display(df.head())
    display(df.describe(include="all").T)

quick_overview(train, "train")
quick_overview(test , "test")

print(f"Duplicate rows (train): {train.duplicated().sum()}  |  (test): {test.duplicated().sum()}")
print("Number of missing values:")
train.isnull().sum() 


fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(data=train, x="diagnosed_diabetes", ax=ax)
ax.set_title("Target Distribution")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}", (p.get_x()+.35, p.get_height()+5000), ha="center")

plt.show()

print(train[target].value_counts(normalize=True).rename("proportion"))


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


print("38% of the full training set is 0")
print("62% of the full training set is 1")

for col in cat_cols:
    ct = pd.crosstab(train[col], train[target], normalize="index")*100
    display(ct.style.format("{:.1f}%").set_caption(f"{col} vs Target"))


n_cols = 2
n_rows = math.ceil(len(num_cols) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()


for i, col in enumerate(num_cols):
    sns.histplot(train[col], ax=axes[i], kde=False)
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


for col in num_cols:
    feat = train[col]
    z    = np.abs(stats.zscore(feat, nan_policy="omit"))
    outlier_mask = (z > 3)
    
    outliers= train.loc[outlier_mask, [col, target]]
    base_counts   = train[target].value_counts()
    outlier_counts = outliers[target].value_counts()
    
    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(x=outlier_counts.index, y=outlier_counts.values, ax=ax)
    ax.set_title("Target Distribution among 3σ " + col + " outliers")
    ax.set_ylabel("count")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,.0f}", (p.get_x()+0.3, p.get_height()+30))
    
    plt.show()
    
    # Proportion print-out
    print("Outlier group distribution")
    display(outlier_counts.to_frame("count")
            .assign(prop=lambda d: d["count"]/d["count"].sum())
            .style.format({"prop": "{:.2%}"}))
    
    print("Comparison with overall training distribution")
    display(base_counts.to_frame("count")
            .assign(prop=lambda d: d["count"]/d["count"].sum())
            .style.format({"prop": "{:.2%}"}))


fig, axes = plt.subplots(math.ceil(len(num_cols)/3), 3, figsize=(15,4*math.ceil(len(num_cols)/3)))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x=target, y=col, data=train, ax=axes[i], showfliers=False
)
    axes[i].set_title(f"{col} by Target")


for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


num_cols.append(target)
corr = train[num_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,cbar_kws={"shrink": 0.8})

plt.xticks(rotation=90, ha='right', fontsize=8) 
plt.yticks(rotation=0, fontsize=8)
plt.title("Pearson Correlation")
plt.show()


target_corr = train[num_cols].corr()[target].drop(
    target).sort_values()
display(target_corr.to_frame("corr_with_target").style.bar(vmin=-1,vmax=1))


import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import gc
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from pandas.errors import PerformanceWarning
from sklearn.metrics import mean_squared_error, roc_auc_score
from itertools import combinations
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import copy

num_cols.remove(target)
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
for col in num_cols:
    train[f'log1p_{col}'] = np.log1p( train[col] )
    test[f'log1p_{col}'] = np.log1p( test[col] )

new_cat_cols=[]

##Interaction features
for col1, col2 in combinations(cat_cols, 2):
    new_col_name = f"{col1}_{col2}"
    train[new_col_name] = train[col1].astype(str) + "_" + train[col2].astype(str)
    test[new_col_name] = test[col1].astype(str) + "_" + test[col2].astype(str)
    new_cat_cols.append(new_col_name)



for col in cat_cols:
    train[col]=train[col].astype('category')
    test[col]=test[col].astype('category')

for col in new_cat_cols:
    train[col]=train[col].astype('category')
    test[col]=test[col].astype('category')

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

print('XGB')
X=train.drop(columns=['id',target])
X_test=test.drop(columns=['id'])
y=train[target]
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
        acc = roc_auc_score(y_val, model.predict(X_val))
        xgb_scores.append(acc), xgb_models.append(model)
        print('Accuracy:', acc)
print('XGB ACCURACY: ', np.mean(xgb_scores))

preds = sum(model.predict(X_test) for model in xgb_models) / len(xgb_models)
submission = pd.DataFrame({'id': test['id'], 'diagnosed_diabetes': preds})
submission.to_csv('submission.csv', index=False)
display(submission.head())

