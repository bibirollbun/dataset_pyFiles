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


import pandas as pd
cal = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
steval = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
price = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv') 


steval.head()


#For the remaining 28 days d1942 to d1969, filling with zero

import numpy as np
for i in range(1942,1970):
    col = 'd_' + str(i)
    steval[col] = 0
    steval[col] = steval[col].astype(np.int16)


# Taken from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
import numpy as np
def reduce_mem_usage(df):
   
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


%%time
reduce_mem_usage(steval)


reduce_mem_usage(price)





sales = pd.melt(steval, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], \
                var_name='d', value_name='sold').dropna()


sales.head()


sales = pd.merge(sales, cal, on='d', how='left')
sales = pd.merge(sales, price, on=['store_id','item_id','wm_yr_wk'], how='left') 


del price
del cal


sales.head()


#Encode categorical variables. Store the categories along with their codes
d_id = dict(zip(sales.id.cat.codes, sales.id))
d_item_id = dict(zip(sales.item_id.cat.codes, sales.item_id))
d_dept_id = dict(zip(sales.dept_id.cat.codes, sales.dept_id))
d_cat_id = dict(zip(sales.cat_id.cat.codes, sales.cat_id))
d_store_id = dict(zip(sales.store_id.cat.codes, sales.store_id))
d_state_id = dict(zip(sales.state_id.cat.codes, sales.state_id))


d_store_id


#Removing "d_" prefix from the values of column "d"
#Categorical variables are convereted as numerical codes
sales.d = sales['d'].apply(lambda x: x.split('_')[1]).astype(np.int16)
cols = sales.dtypes.index.tolist()
types = sales.dtypes.values.tolist()
for i,type in enumerate(types):
    if type.name == 'category':
        sales[cols[i]] = sales[cols[i]].cat.codes


sales.head()


#Dropping date column        
sales.drop('date',axis=1,inplace=True)


lags = [1,2,4,8,16,32]
for lag in lags:
    sales['sold_lag_'+str(lag)] = sales.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['sold'].shift(lag).astype(np.float16)


#Combination of two vars with "sold" and their mean
# sales['item_sold_avg'] = sales.groupby('item_id')['sold'].transform('mean').astype(np.float16)
# sales['state_sold_avg'] = sales.groupby('state_id')['sold'].transform('mean').astype(np.float16)
# sales['store_sold_avg'] = sales.groupby('store_id')['sold'].transform('mean').astype(np.float16)
# sales['cat_sold_avg'] = sales.groupby('cat_id')['sold'].transform('mean').astype(np.float16)
# sales['dept_sold_avg'] = sales.groupby('dept_id')['sold'].transform('mean').astype(np.float16)

#Combination of three vars with "sold" and their mean
# sales['cat_dept_sold_avg'] = sales.groupby(['cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
# sales['store_item_sold_avg'] = sales.groupby(['store_id','item_id'])['sold'].transform('mean').astype(np.float16)
# sales['cat_item_sold_avg'] = sales.groupby(['cat_id','item_id'])['sold'].transform('mean').astype(np.float16)
# sales['dept_item_sold_avg'] = sales.groupby(['dept_id','item_id'])['sold'].transform('mean').astype(np.float16)
# sales['dept_store_sold_avg'] = sales.groupby(['dept_id','store_id'])['sold'].transform('mean').astype(np.float16)

#Combination of four vars with "sold" and their mean
# sales['store_cat_dept_sold_avg'] = sales.groupby(['store_id','cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
# sales['store_cat_item_sold_avg'] = sales.groupby(['store_id','cat_id','item_id'])['sold'].transform('mean').astype(np.float16)





sales['rolling_sold_mean'] = \
sales.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold']\
.transform(lambda x: x.rolling(window=6).mean()).astype(np.float16)


sales['expanding_sold_mean'] = \
sales.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold']\
.transform(lambda x: x.expanding(2).mean()).astype(np.float16)


#Clear some space
import gc
gc.collect()


# Since we introduced lags till 32 days, data for first 31 days should be removed.
sales = sales[sales['d']>=32]


# Save data for training
sales.to_pickle('salesdata.pkl') #to_pickle: serializes an object to file
del sales


%pwd


%ls


data = pd.read_pickle('salesdata.pkl')
validation = data[(data['d']>=1914) & (data['d']<1942)][['id','d','sold']]
test = data[data['d']>=1942][['id','d','sold']]
eval_prediction = test['sold']
validation_prediction = validation['sold']


gc.collect()


#Get the store ids
stores = steval.store_id.cat.codes.unique().tolist()
for store in stores:
    df = data[data['store_id']==store]


gc.collect()


#Split the data
X_train, y_train = df[df['d']<1914].drop('sold',axis=1), df[df['d']<1914]['sold']
X_valid, y_valid = df[(df['d']>=1914) & (df['d']<1942)].drop('sold',axis=1), df[(df['d']>=1914) & (df['d']<1942)]['sold']
X_test = df[df['d']>=1942].drop('sold',axis=1)


gc.collect()


del data
gc.collect()





X_train.info()


CATEGORICAL_COLS = []

for _col in X_train.columns:
    if X_train[_col].dtype == 'object':
        X_train[_col] = X_train[_col].astype('category')
        X_valid[_col] = X_valid[_col].astype('category')

        CATEGORICAL_COLS.append(_col)


X_train.info()


import lightgbm as lgb
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from sklearn.model_selection import KFold # KFoldã‚’ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
from sklearn.metrics import mean_squared_error # è©•ä¾¡æŒ‡æ¨™ï¼ˆã�“ã�“ã�§ã�¯RMSEï¼‰
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning) # pandas/numpyã�®è­¦å‘Šã‚’æŠ‘åˆ¶


# ===== æ�¢ç´¢ç©ºé–“ã�®èª¿æ•´ (n_estimatorsã�¯å›ºå®šã�—ã€�ä»£ã‚�ã‚Šã�«early_stoppingã�«ä»»ã�›ã‚‹) =====
valgrid = {
    'objective': 'regression',      # ç›®çš„é–¢æ•°ã‚’æ˜�ç¤º
    'metric': 'rmse',               # è©•ä¾¡æŒ‡æ¨™ã‚’æ˜�ç¤º
    'n_jobs': -1,

    'n_estimators': hp.quniform('n_estimators', 300, 2000, 100),
    # learning_rate: loguniformã�¯ã‚ˆã‚Šè‰¯ã�„æ�¢ç´¢ãƒ¬ãƒ³ã‚¸ã‚’æ��ä¾›
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'max_depth': hp.quniform('max_depth', 4, 8, 1),
    'num_leaves': hp.quniform('num_leaves', 25, 75, 25),
    # subsample, colsample_bytree: uniformã�¯é€£ç¶šå€¤ã�«é�©åˆ‡
    'subsample': hp.uniform('subsample', 0.5, 0.9),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.9),
    # min_child_samples: LightGBMãƒ�ã‚¤ãƒ†ã‚£ãƒ–APIã�®ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿
    'min_child_samples': hp.quniform('min_child_samples', 20, 100, 10) ,

    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(10)),  # L1æ­£å‰‡åŒ– (1e-5ã�‹ã‚‰10ã�®ç¯„å›²ã‚’å¯¾æ•°ã‚¹ã‚±ãƒ¼ãƒ«ã�§)
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(10)), # L2æ­£å‰‡åŒ–
}


def objective_cv_early_stopping(params):
    # 1. hyperoptã�‹ã‚‰æ¸¡ã�•ã‚Œã‚‹ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã‚’LightGBMã�®å­¦ç¿’ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã�«æ•´ç�†
    lgb_params = {
        'objective': params['objective'],
        'metric': params['metric'],
        'n_jobs': params['n_jobs'],
        'learning_rate': params['learning_rate'],
        'max_depth': int(params['max_depth']), # intã�«å¤‰æ�›
        'num_leaves': int(params['num_leaves']), # intã�«å¤‰æ�›
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'min_child_samples': int(params['min_child_samples']), # intã�«å¤‰æ�›
        'verbose': -1 # å­¦ç¿’æ™‚ã�®å†—é•·ã�ªãƒ­ã‚°å‡ºåŠ›ã‚’æŠ‘åˆ¶
    }

    # 2. KFoldã�®åˆ�æœŸåŒ– (ã�“ã�“ã�§ã�¯k=3ã�§äº¤å·®æ¤œè¨¼ã‚’å®Ÿæ–½)
    NFOLDS = 3
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    oof_loss = [] # å�„ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ã�®æ��å¤±ã‚’æ ¼ç´�ã�™ã‚‹ãƒªã‚¹ãƒˆ (Out-Of-Fold Loss)

    # 3. K-Foldã�®ã‚¤ãƒ†ãƒ¬ãƒ¼ã‚·ãƒ§ãƒ³
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
        
        # è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¨æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�«åˆ†å‰²
        X_TR, X_VAL = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_TR, y_VAL = y_train.iloc[train_index], y_train.iloc[valid_index]

        # LightGBMãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®ä½œæˆ�
        lgb_train_data = lgb.Dataset(X_TR, y_TR, categorical_feature=CATEGORICAL_COLS)
        # early_stoppingã�®è©•ä¾¡ç”¨ãƒ‡ãƒ¼ã‚¿
        lgb_valid_data = lgb.Dataset(X_VAL, y_VAL, reference=lgb_train_data)
        
        # 4. ãƒ¢ãƒ‡ãƒ«å­¦ç¿’ã�¨Early Stoppingã�®å®Ÿæ–½

        model = lgb.train(
            lgb_params,
            lgb_train_data,
            valid_sets=[lgb_valid_data], # æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã‚’æ¸¡ã�™
            # ğŸ’¡ early_stoppingã�®ã‚³ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯ã‚’è¨­å®š
            callbacks=[lgb.early_stopping(
                stopping_rounds=30, # 30å›�é€£ç¶šã�§ã‚¹ã‚³ã‚¢ã�Œæ”¹å–„ã�—ã�ªã�‘ã‚Œã�°å�œæ­¢
                verbose=False
            )]
        )

        # 5. äºˆæ¸¬ã�¨ã‚¹ã‚³ã‚¢ãƒªãƒ³ã‚° (RMSEã‚’ä½¿ç”¨ã�—ã€�best_iterationã�§äºˆæ¸¬)
        pred = model.predict(X_VAL, num_iteration=model.best_iteration)
        # æ��å¤± (loss) ã‚’è¨ˆç®—ã€‚hyperoptã�¯ã�“ã‚Œã‚’æœ€å°�åŒ–ã�™ã‚‹ã€‚
        loss = np.sqrt(mean_squared_error(y_VAL, pred)) 
        oof_loss.append(loss)

    # 6. K-Foldå…¨ä½“ã�§ã�®å¹³å�‡æ��å¤±ã‚’è¨ˆç®—ã�—ã€�hyperoptã�«è¿”ã�™
    mean_loss = np.mean(oof_loss)

    return {'loss': mean_loss, 'status': STATUS_OK}


import random
# # æœ€é�©åŒ–ã�®å®Ÿè¡Œä¾‹ (å®Ÿè¡Œã�«ã�¯æ™‚é–“ã�Œã�‹ã�‹ã‚Šã�¾ã�™)
trials = Trials()

train_index = []

RANDOM_SEED = 123
random.seed(RANDOM_SEED)

bestP = fmin(
    fn=objective_cv_early_stopping,
    space=valgrid,
    algo=tpe.suggest,
    max_evals=100, # è©¦è¡Œå›�æ•°
    rstate=np.random.default_rng(RANDOM_SEED), #ä¹±æ•°å›ºå®š
    trials=trials
)


gc.collect()


bestP





# 1. æœ€é�©ã�ªãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ (hyperoptã�®çµ�æ�œã‚’ä½¿ç”¨)
best_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_jobs': -1,
    'learning_rate': bestP['learning_rate'],
    'max_depth': int(bestP['max_depth']),
    'num_leaves': int(bestP['num_leaves']),
    'subsample': bestP['subsample'],
    'colsample_bytree': bestP['colsample_bytree'],
    'min_child_samples': int(bestP['min_child_samples']),
    'reg_alpha': bestP['reg_alpha'],
    'reg_lambda': bestP['reg_lambda'],
    'n_estimators': int(bestP['n_estimators']) # æœ€é�©åŒ–å¾Œã�«æ‰‹å‹•ã�§è¨­å®šã�™ã‚‹å ´å�ˆã‚„ã€�æ�¢ç´¢çµ�æ�œã‚’ä½¿ç”¨
}


# 2. æœ€é�©ã�ªå��å¾©å›�æ•° (early_stoppingã�§è¦‹ã�¤ã�‘ã�Ÿå¹³å�‡å€¤ã‚„ã€�CVçµ�æ�œã�‹ã‚‰æ±ºå®šã�—ã�Ÿå€¤)
# n_estimatorsã‚’æ�¢ç´¢ã�—ã�Ÿå ´å�ˆã�¯ã€�ã��ã�®å€¤ã�Œæœ€å¤§ã�®ãƒ–ãƒ¼ã‚¹ãƒ†ã‚£ãƒ³ã‚°å›�æ•°ã�¨ã�ªã‚Šã�¾ã�™ã€‚
BEST_ITERATION = int(best_params.pop('n_estimators', 1000)) # n_estimatorsã‚’åˆ†é›¢


BEST_ITERATION


%%time
# LightGBM Datasetã�®ä½œæˆ� (æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�¯ä¸�è¦�)
lgb_train_final = lgb.Dataset(
    X_train, 
    y_train, 
    categorical_feature=CATEGORICAL_COLS
)

# ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’ (num_boost_roundã�«æœ€é�©ã�ªå›�æ•°ã‚’è¨­å®š)
final_model = lgb.train(
    best_params,
    lgb_train_final,
    num_boost_round=BEST_ITERATION
)


# ==========================================================
# 2. æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ (X_valid) ã‚’ä½¿ã�£ã�ŸRMSEã�®è¨ˆç®—
# ==========================================================

from sklearn.metrics import mean_squared_error

# æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�§äºˆæ¸¬
y_pred_valid = final_model.predict(X_valid, num_iteration=BEST_ITERATION)

# RMSEã�®è¨ˆç®—
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))

print(f"æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ (X_valid) ã�®RMSE: {rmse_valid:.4f}")


import matplotlib.pyplot as plt

print("\n--- 3. ç‰¹å¾´é‡�é‡�è¦�åº¦ (Train) ã�¨ SHAPå€¤ (Valid) ã�®è¨ˆç®—ã�¨æ¯”è¼ƒ ---")

# ==========================================================
# 3-1. X_train (å­¦ç¿’ãƒ‡ãƒ¼ã‚¿) ã�®ç‰¹å¾´é‡�é‡�è¦�åº¦ã‚’å‡ºåŠ›
# ==========================================================

print("  - X_trainã�®Feature Importanceã‚’è¨ˆç®—ãƒ»è¡¨ç¤º...")

# ç‰¹å¾´é‡�é‡�è¦�åº¦ã‚’å�–å¾— (é‡�è¦�åº¦ã�¯ 'gain' ã�¾ã�Ÿã�¯ 'split' ã�§è¨ˆç®—ã�•ã‚Œã‚‹)
# 'gain'ï¼ˆåˆ©å¾—ï¼‰ã�¯ã€�ã��ã�®ç‰¹å¾´é‡�ã�Œä½¿ç”¨ã�•ã‚Œã�Ÿã�¨ã��ã�®å…¨ä½“ã�®æƒ…å ±åˆ©å¾—ã�®å�ˆè¨ˆã�§ã�‚ã‚Šã€�ã‚ˆã‚Šä¿¡é ¼æ€§ã�Œé«˜ã�„ã�“ã�¨ã�Œå¤šã�„
importance_type = 'gain' 
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    # feature_importances_å±�æ€§ã�«ã‚¢ã‚¯ã‚»ã‚¹
    'importance': final_model.feature_importance(importance_type=importance_type)
})

# é‡�è¦�åº¦ã�®ç¨®é¡�ã‚’é�©åˆ‡ã�«è¨­å®š
if importance_type == 'gain':
    feature_importance['importance'] = final_model.feature_importance(importance_type=importance_type)
else: # 'split'ã�®å ´å�ˆ
    feature_importance['importance'] = final_model.feature_importance(importance_type=importance_type) / final_model.feature_importance(importance_type=importance_type).sum()

# é‡�è¦�åº¦é †ã�«ã‚½ãƒ¼ãƒˆã�—ã�¦è¡¨ç¤º
feature_importance = feature_importance.sort_values(by='importance', ascending=False)

# æ£’ã‚°ãƒ©ãƒ•ã�§å�¯è¦–åŒ–
plt.figure(figsize=(10, len(X_train.columns) * 0.3))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title(f"LightGBM Feature Importance ({importance_type.upper()}) on Training Data")
plt.xlabel(f"Feature Importance ({importance_type.upper()})")
plt.gca().invert_yaxis() # é‡�è¦�åº¦ã�Œé«˜ã�„é †ã�«ä¸Šã�‹ã‚‰è¡¨ç¤º
plt.show() # 
print(feature_importance.head(10)) # ä¸Šä½�10å€‹ã‚’ã‚³ãƒ³ã‚½ãƒ¼ãƒ«ã�«ã‚‚å‡ºåŠ›


%%time
# ==========================================================
# 3. SHAPå€¤ã�®è¨ˆç®—ã�¨swarmplotã�®ä½œæˆ�
# ==========================================================

import shap
import matplotlib.pyplot as plt

print("\n--- 3. SHAPå€¤ã�®è¨ˆç®—ã�¨å�¯è¦–åŒ– ---")

# SHAP Explainerã�®åˆ�æœŸåŒ– (TreeExplainerã‚’ä½¿ç”¨)
# LightGBMãƒ¢ãƒ‡ãƒ«ã‚’æ¸¡ã�—ã�¾ã�™ã€‚
explainer = shap.TreeExplainer(final_model)

# 3-2. X_valid (æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿) ã�®SHAPè¨ˆç®—ã�¨å�¯è¦–åŒ–
print("  - X_validã�®SHAPå€¤ã‚’è¨ˆç®—...")
shap_values_valid = explainer.shap_values(X_valid)

# Swarm Plotã�®ä½œæˆ�
shap.summary_plot(
    shap_values_valid, 
    X_valid, 
    plot_type="dot", 
    show=False
)
plt.title("SHAP Swarm Plot (Validation Data)")
plt.show()










