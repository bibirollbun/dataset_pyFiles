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


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_file = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
train_df.info()


test_df.info()


train_df.columns, test_df.columns  # check cols


train_df[train_df['day']==365]['day'].index  # find a problem


"""356 per year"""


plt.figure(figsize=(12, 4))
plt.plot(train_df['day'], color='r')
plt.show()  # show problem


2190/365


train_df['day'] = pd.Series([i for i in range(1, 366)]*6)
plt.figure(figsize=(12, 4))
plt.plot(train_df['day'], color='r')
plt.show()


plt.figure(figsize=(10, 4))
df = train_df[['pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection','windspeed']]
plt.boxplot(df)
plt.show()


"""train_df"""
li = train_df[train_df['day']==365].index
df1 = train_df.loc[0:li[0]]
df2 = train_df.loc[li[0]+1:li[1]]
df3 = train_df.loc[li[1]+1:li[2]]
df4 = train_df.loc[li[2]+1:li[3]]
df5 = train_df.loc[li[3]+1:li[4]]
df6 = train_df.loc[li[4]+1:]
for y, df in zip(range(2019, 2025), [df1, df2, df3, df4, df5, df6]):
    df['date'] = pd.to_datetime(int(y)*1000 + df['day'], format='%Y%j')


"""concat train_df"""
train_df = pd.concat([df1, df2], axis=0)
train_df = pd.concat([train_df, df3], axis=0)
train_df = pd.concat([train_df, df4], axis=0)
train_df = pd.concat([train_df, df5], axis=0)
train_df = pd.concat([train_df, df6], axis=0)
train_df.head()


train_df.tail()


len(train_df[train_df['day']==365]), len(test_df[test_df['day']==365])  # 6, 2


len(train_df), len(test_df)  # 2190, 730


"""test_df"""
t_df1 = test_df.loc[0:364]
t_df2 = test_df.loc[365:]

for y, df in zip(range(2024, 2026), [t_df1, t_df2]):
    df['date'] = pd.to_datetime(int(y)*1000 + df['day'], format='%Y%j')  # do not care warning

test_df = pd.concat([t_df1, t_df2], axis=0)
test_df.head()


len(train_df[train_df['rainfall']==1])/len(train_df)


train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['quarter'] = train_df['date'].dt.quarter  

test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['quarter'] = test_df['date'].dt.quarter


train_df.info()


test_df.info()


train_df['rainfall_condition'] = abs(train_df['maxtemp']-train_df['mintemp'])
test_df['rainfall_condition'] = abs(test_df['maxtemp']-test_df['mintemp'])


train_df['temp_cha'] = abs(train_df['dewpoint']-train_df['temparature'])
test_df['temp_cha'] = abs(test_df['dewpoint']-test_df['temparature'])


train_df.pressure.min(), train_df.pressure.max()


train_df['pressure_box'] = pd.cut(train_df['pressure'], bins=[-float('inf'), 1000, 1020, float('inf')], labels=[0, 1, 2]) 
test_df['pressure_box'] = pd.cut(test_df['pressure'], bins=[-float('inf'), 1000, 1020, float('inf')], labels=[0, 1, 2])


train_df.head()


# visualisation - the differ pressure of rainfall rate
plt.figure(figsize=(10, 5))
train_df.groupby(by='pressure_box', observed=True)['rainfall'].mean().plot(kind='bar', color='r') # æ— æ˜�æ˜¾è§„å¾‹æ€§
plt.show()


train_df['dewpoint'].min(), train_df['dewpoint'].max()


train_df['dewpoint_box'] = pd.cut(train_df['dewpoint'], bins=[-float('inf'), 2, 5, float('inf')], labels=[0, 1, 2])
test_df['dewpoint_box'] = pd.cut(test_df['dewpoint'], bins=[-float('inf'), 2, 5, float('inf')], labels=[0, 1, 2])


train_df.head()


# visualisation
plt.figure(figsize=(10, 6))
train_df.groupby(by='dewpoint_box', observed=True)['rainfall'].mean().plot(kind='bar', color='green')
plt.show() 


train_df.cloud.min(), train_df.cloud.max()


train_df['cloud_box'] = pd.qcut(train_df['cloud'], q=4, labels=[0, 1, 2, 3])
test_df['cloud_box'] = pd.qcut(test_df['cloud'], q=4, labels=[0, 1, 2, 3])
train_df[['cloud', 'cloud_box', 'rainfall']].head(5)


train_df.groupby(by='cloud_box', observed=True)['rainfall'].mean().plot(kind='bar', color='g')
plt.show()


"""year rainfall-rate"""
plt.figure(figsize=(12, 5))
train_df.groupby(by='year')['rainfall'].mean().plot(kind='line', color='r')
plt.ylabel('Rainfall rate')
plt.xlabel('years')
plt.title('Rain_rate of year')
plt.show()


"""month rainfall-rate"""
plt.figure(figsize=(10, 5))
train_df.groupby(by='month')['rainfall'].sum().plot(kind='line', color='r')
plt.ylabel('Rainfall rate')
plt.xlabel('months')
plt.title('Rain_rate of month')
plt.show()


"""quarter rainfall-rate"""
plt.figure(figsize=(9, 5))
train_df.groupby(by='quarter')['rainfall'].sum().plot(kind='bar', color='r')
plt.ylabel('Rainfall rate')
plt.xlabel('quarters')
plt.title('Rain_rate of quarter')
plt.show()


len(train_df.columns), train_df.columns


len(test_df.columns), test_df.columns


features = ['day', 'year','quarter','id', 'maxtemp', 'mintemp']  # , 'month', 'winddirection', 'cloud_box', 'pressure_box'
for df in [train_df, test_df]:
    try:
        df.drop(columns=features, errors="ignore", inplace=True)
    except:
        print("error!!")
    else:
        print("del successfall!")


train_df.columns


train_df.info(show_counts=True)


def cal_rolling_tag_features(input_df, lag_fields, ma_fields): # lag_fields, ma_fieldsè¦�å¤„ç�†çš„å­—æ®µåˆ—è¡¨
    # --LH--
    for field in lag_fields:  
        for lag in [1, 2, 3]:
            input_df[f'{field}_lag_{lag}'] = input_df[field].shift(lag)

    # --MV--
    for field in ma_fields:
        input_df[f"{field}_rolling_3"] = input_df[field].rolling(window=3).mean()
        input_df[f"{field}_rolling_7"] = input_df[field].rolling(window=7).mean()

lags = ['humidity', 'pressure' , 'dewpoint', 'cloud', 'dewpoint']
mas = ['windspeed']
cal_rolling_tag_features(train_df, lags, mas)
cal_rolling_tag_features(test_df, lags, mas)
train_df.info()


train_df.isnull().sum()


test_df.isnull().sum()


train_df = train_df.bfill()
test_df = test_df.bfill()


train_df.isnull().sum()


test_df.isnull().sum()


train_df.to_csv("train_df.csv", index=False, encoding='utf-8')
test_df.to_csv("test_df.csv", index=False, encoding='utf-8')
print("backup successfally!")


train_df = pd.read_csv("train_df.csv")
test_df = pd.read_csv("test_df.csv")
len(train_df), len(test_df)


des_df = train_df.describe()
des_df.loc['std'], type(des_df.loc['std'])


import pandas as pd
from sklearn.model_selection import KFold

X, y = train_df.drop(['date', 'rainfall'], axis=1), train_df['rainfall']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

folds_data = []

for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    folds_data.append({
        'X_train': X_train_fold,
        'X_val': X_val_fold,
        'y_train': y_train_fold,
        'y_val': y_val_fold
    })

first_fold = folds_data[0]
print("train_dataset shape:", first_fold['X_train'].shape)
print("val_dataset shape:", first_fold['X_val'].shape)


# train and val
X_train, y_train, X_val, y_val = first_fold['X_train'], first_fold['y_train'],first_fold['X_val'], first_fold['y_val']
print("trainï¼š",X_train.shape)  
print("valï¼š",X_val.shape)       

# test
if 'date' in test_df.columns:
    test_df.drop('date', axis=1, inplace=True)
print("testï¼š",test_df.shape)


X_val.tail()


train_df = pd.read_csv("train_df.csv")
X = train_df.drop(['rainfall', 'date'], axis=1)
X.shape, X_val.shape


X.columns, len(X.columns)


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


# æ•°æ�®åŠ è½½
train_df = pd.read_csv("train_df.csv")
X = train_df.drop(['rainfall', 'date', 'pressure_box', 'dewpoint', 'cloud_box', 'month'], axis=1, errors='ignore').values  # ç‰¹å¾�
y = train_df['rainfall'].values                          # ç›®æ ‡ï¼ˆ0å’Œ1ï¼‰

# äº¤å�‰éªŒè¯�
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
cv_scores = []

# æš´åŠ›è°ƒå�‚å��çš„æœ€ä½³å�‚æ•°  ->  å¹³å�‡éªŒè¯�AUC: 0.8914 (Â±0.0200)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4, 
    'eta': 0.05, 
    'subsample': 0.6, 
    'lambda': 1,
    'gamma': 0.2,
    'alpha': 0.1
}


for train_idx, val_idx in kf.split(X, y):
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
    
    # è®­ç»ƒæ¨¡å�‹ï¼ˆå¸¦æ—©å�œï¼‰
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=1000,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    val_auc = model.best_score  # å½“å‰�æŠ˜çš„éªŒè¯�AUC
    cv_scores.append(val_auc)
    print(f"å½“å‰�æŠ˜AUC: {val_auc:.4f}")

# è¾“å‡ºå¹³å�‡æ€§èƒ½
print(f"å¹³å�‡éªŒè¯�AUC: {np.mean(cv_scores):.4f} (Â±{np.std(cv_scores):.4f})")



import itertools  # æ·»åŠ ç¼ºå¤±çš„å¯¼å…¥
import xgboost as xgb

"""æ›´å¼ºçš„æ­£åˆ™åŒ–"""
param_grid = {
    'max_depth': [3, 4, 5],  # åˆ é™¤7ï¼Œé™�åˆ¶æ ‘æ·±åº¦
    'eta': [0.01, 0.05],     # æ›´å°�çš„å­¦ä¹ ç�‡
    'subsample': [0.6, 0.8], # å¼ºåˆ¶æ ·æœ¬éš�æœºæ€§
    'lambda': [1, 10],       # å¢�å¤§L2æ­£åˆ™åŒ–
    'gamma': [0.1, 0.2],     # åˆ é™¤0ï¼Œç¡®ä¿�åˆ†è£‚æœ€å°�å¢�ç›Š
    'alpha': [0.1, 0.5],      # åˆ é™¤0ï¼Œå�¯ç”¨L1æ­£åˆ™ 
}
# 2. åˆ�å§‹åŒ–æœ€ä½³ç»“æ�œè®°å½•
output_time = 0
best_auc = 0
best_params = {}

# 3. é��å�†æ‰€æœ‰å�‚æ•°ç»„å�ˆ
for params in itertools.product(*param_grid.values()):
    current_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_delta_step': 1  # é™�åˆ¶æ¯�æ£µæ ‘æ�ƒé‡�
    }
    current_params.update(dict(zip(param_grid.keys(), params)))
    
    # è®­ç»ƒæ¨¡å�‹ï¼ˆå…³æ³¨éªŒè¯�é›†æ€§èƒ½ï¼‰
    model = xgb.train(
        current_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False  # ä¸�æ˜¾ç¤º
    )
    output_time += 1
    # è®°å½•æœ€ä½³éªŒè¯�é›†AUC
    val_auc = model.best_score
    # print(f"å�‚æ•°: {current_params} | Val-AUC: {val_auc:.4f}")
    # print("-------------------------------------------------")
    if output_time % 6 == 0:
        print(f"Val-AUC: {val_auc:.4f}")
    else:
        print(f"Val-AUC: {val_auc:.4f}", end=" | ")
    
    if val_auc > best_auc:
        best_auc = val_auc
        best_params = current_params

# 4. è¾“å‡ºæœ€ç»ˆæœ€ä¼˜å�‚æ•°
print("\nğŸ�† æœ€ä½³å�‚æ•°ç»„å�ˆ:", {k: v for k, v in best_params.items() if k in param_grid})
print(f"æœ€ä½³éªŒè¯�é›†AUC: {best_auc:.4f}")
print(f"æœ€ä½³è¿­ä»£è½®æ¬¡ï¼š {model.best_iteration} rounds")  # # å…³é”®è¯Šæ–­ï¼šæ£€æŸ¥æœ€ä½³è¿­ä»£è½®æ¬¡


"""use best params"""


best_params


params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],  # å�Œæ—¶ç›‘æ�§loglosså’ŒAUC
    'max_depth': 4,                     # å¢�åŠ æ¨¡å�‹å¤�æ�‚åº¦
    'eta': 0.1,                        # é€‚åº¦å¢�å¤§å­¦ä¹ ç�‡
    'subsample': 0.6,                   # é�¿å…�è¿‡æ‹Ÿå�ˆ
    'colsample_bytree': 0.8,
    'lambda': 1,                        # æ”¾æ�¾L2æ­£åˆ™åŒ–
    'gamma': 0,
    'alpha': 0,                         # æš‚ç¦�ç”¨L1
    'seed': 2025,
    'nthread': -1
}
params.update(best_params)
params


import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. Data Preparation (ä¸¥æ ¼éš”ç¦»)
dtrain = xgb.DMatrix(X_train, label=y_train)  # è®­ç»ƒé›†ï¼ˆä»…è®­ç»ƒï¼‰
dval = xgb.DMatrix(X_val, label=y_val)        # éªŒè¯�é›†ï¼ˆè°ƒå�‚+æ—©å�œï¼‰
dtest = xgb.DMatrix(test_df)                  # æµ‹è¯•é›†ï¼ˆæœ€ç»ˆè¯„ä¼°ï¼‰



params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],  # å�Œæ—¶ç›‘æ�§loglosså’ŒAUC
    'max_depth': 4,                     # å¢�åŠ æ¨¡å�‹å¤�æ�‚åº¦
    'eta': 0.1,                        # é€‚åº¦å¢�å¤§å­¦ä¹ ç�‡
    'subsample': 0.6,                   # é�¿å…�è¿‡æ‹Ÿå�ˆ
    'colsample_bytree': 0.8,
    'lambda': 1,                        # æ”¾æ�¾L2æ­£åˆ™åŒ–
    'gamma': 0,
    'alpha': 0,                         # æš‚ç¦�ç”¨L1
    'seed': 2025,
    'nthread': -1
}
params.update(best_params)  # åŠ å…¥æœ€ä½³å�‚æ•°


# 3. è®­ç»ƒæµ�ç¨‹ï¼ˆéªŒè¯�é›†é©±åŠ¨ï¼‰
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'val')],  # éªŒè¯�é›†å�‚ä¸�è¯„ä¼°
    early_stopping_rounds=50,                  # åŸºäº�éªŒè¯�é›†æŒ‡æ ‡æ—©å�œ
    verbose_eval=20
)

# 4. éªŒè¯�é›†è¯„ä¼°ï¼ˆæ ¸å¿ƒï¼�ï¼‰
y_val_pred = model.predict(dval) 
y_pred = model.predict(dtest)

fpr, tpr, _ = roc_curve(y_val, y_val_pred)
val_auc = auc(fpr, tpr)
print(f"Validation AUC: {val_auc:.4f}")  # å…³é”®æŒ‡æ ‡ï¼�

# 5. å�¯è§†åŒ–
plt.plot(fpr, tpr, label=f'Val AUC = {val_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# 0.8717
# 0.8709


# visual feature importance
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.show()


print(len(sample_file), len(y_pred))
sample_file['rainfall'] = pd.Series(y_pred)
sample_file.head()


# sample_file.to_csv("submission.csv", index=False, encoding='utf-8')
# print("backup successfully!")


"""
v5: 0.90795 (best score) 
V6: 0.90438
v7: 0.89801
"""



train_df = pd.read_csv("train_df.csv")
train_df.columns


import seaborn as sns
sns.pairplot(train_df[['pressure', 'temparature', 'humidity']])


['rainfall', 'date', 'pressure_box', 'dewpoint', 'cloud_box', 'month']


train_df = pd.read_csv("train_df.csv")
y =  train_df['rainfall']
X = train_df.drop(['rainfall', 'date', 'pressure_box', 'dewpoint', 'cloud_box', 'month'], axis=1, errors='ignore')


test_df = pd.read_csv("test_df.csv").drop(['rainfall', 'date', 'pressure_box', 'dewpoint', 'cloud_box', 'month'], axis=1,errors='ignore')
print("è®­ç»ƒé›†ï¼š", X.columns,X.shape, len(X))
print("æµ‹è¯•é›†ï¼š", test_df.columns,test_df.shape, len(test_df))


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# åˆ’åˆ†è®­ç»ƒé›†ï¼ˆè°ƒå�‚ç”¨ï¼‰å’ŒéªŒè¯�é›†
X_train, X_val_1, y_train, y_val_1 = train_test_split(X, y, test_size=0.1, random_state=2025)
print("è®­ç»ƒé›†ï¼š", X_train.shape, y_train.shape)
print("éªŒè¯�é›†ï¼š", X_val_1.shape, y_val_1.shape)
print("æµ‹è¯•é›†ï¼š", test_df.shape)


from bayes_opt import BayesianOptimization
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score

def lgb_auc(max_depth, num_leaves, min_child_samples, subsample, colsample_bytree, 
           reg_alpha, reg_lambda, learning_rate, min_split_gain, scale_pos_weight, max_bin):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'max_depth': int(max_depth),
        'num_leaves': int(num_leaves),
        'min_child_samples': int(min_child_samples),
        'subsample': max(min(subsample, 1), 0.1),
        'colsample_bytree': max(min(colsample_bytree, 1), 0.1),
        'reg_alpha': max(reg_alpha, 0),
        'reg_lambda': max(reg_lambda, 0),
        'learning_rate': max(min(learning_rate, 0.1), 0.01),
        'min_split_gain': max(min_split_gain, 0),
        'scale_pos_weight': max(int(scale_pos_weight), 1),
        'max_bin': int(max_bin),
        'n_jobs': -1,
        'verbosity': -1
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    # äº¤å�‰éªŒè¯�
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx] # è¿™é‡Œçš„éªŒè¯�é›†ä¸ºäº¤å�‰éªŒè¯�å†…éƒ¨çš„åŠ¨æ€�éªŒè¯�é›†
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(10)
            ]
        )
        y_pred = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, y_pred))
    
    return np.mean(auc_scores)

# å®šä¹‰å�‚æ•°æ�œç´¢èŒƒå›´ (ç¡®ä¿�æ‰€æœ‰å�‚æ•°éƒ½åœ¨lgb_aucå‡½æ•°ä¸­ç”¨åˆ°)
pbounds = {
    'max_depth': (3, 8),
    'num_leaves': (15, 50),
    'min_child_samples': (5, 20),
    'subsample': (0.7, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0, 5),
    'reg_lambda': (0, 5),
    'learning_rate': (0.01, 0.1),
    'min_split_gain': (0, 0.1),
    'scale_pos_weight': (1, 10),
    'max_bin': (100, 255)
}

# ç¡®ä¿�X_trainå’Œy_trainå·²ç»�å®šä¹‰
# X_train, y_train = ...

# è¿�è¡Œè´�å�¶æ–¯ä¼˜åŒ–
optimizer = BayesianOptimization(
    f=lgb_auc,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=20)

# è¾“å‡ºæœ€ä½³å�‚æ•°å¹¶è½¬æ�¢ä¸ºæ•´æ•°ç±»å�‹
best_params = optimizer.max['params']
for param in ['max_depth', 'num_leaves', 'min_child_samples', 'max_bin', 'scale_pos_weight']:
    best_params[param] = int(best_params[param])
    
print("Best Parameters:", best_params)



best_params


from lightgbm import early_stopping, log_evaluation


# åŠ å…¥æ—©å�œå’Œå›�è°ƒåŠŸèƒ½
final_model = lgb.LGBMClassifier(
    **best_params,
    boosting_type='gbdt',    # é»˜è®¤æ¢¯åº¦æ��å�‡
    objective='binary',
    metric='auc',
    n_estimators=1000,  # è®¾å¤§å€¼ï¼Œé� æ—©å�œæ�§åˆ¶
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val_1, y_val_1)],  # ç›‘æ�§éªŒè¯�é›†æ€§èƒ½
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(10)  # æ¯� 10 è½®æ‰“å�°ä¸€æ¬¡æ—¥å¿—
    ]
)

# é¢„æµ‹2çº§éªŒè¯�é›†æ¦‚ç�‡
y_val_proba = final_model.predict_proba(X_val_1)[:, 1] # ç”Ÿæˆ�æ ·æœ¬å±�äº�å�„ä¸ªç±»åˆ«çš„æ¦‚ç�‡
val_auc = roc_auc_score(y_val_1, y_val_proba)
print(f"val AUC: {val_auc:.4f}")



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# (1) ç»˜åˆ¶ROCæ›²çº¿
fpr, tpr, _ = roc_curve(y_val_1, y_val_proba)
roc_auc = auc(fpr, tpr)

plt.close("all")  # å…³é—­æ‰€æœ‰æ—§å›¾
plt.figure(figsize=(12, 5))  # é‡�æ–°å¼€å§‹

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# (2) ç‰¹å¾�é‡�è¦�æ€§
ax = plt.subplot(1, 2, 2)
lgb.plot_importance(final_model, ax=ax, max_num_features=-1, importance_type='gain')
plt.title('Feature Importance (Gain)')
plt.tight_layout()
plt.show()



# æµ‹è¯•é›†é¢„æµ‹

y_test_proba = final_model.predict_proba(test_df)[:, 1]
y_test_proba




sub = pd.read_csv("submission.csv")
sub['rainfall'] = y_test_proba
sub.head()



# å¤‡ä»½
sub.to_csv("submission.csv", index=False, encoding='utf-8')
print("backup successfully!")




