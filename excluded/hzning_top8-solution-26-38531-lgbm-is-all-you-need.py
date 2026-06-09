import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")

test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


train = train.drop(columns=['id'])
train.head()


train.info()


import seaborn as sns
import matplotlib.pyplot as plt
numeric_cols = train.select_dtypes(include='number').columns.drop('BeatsPerMinute')

for col in numeric_cols:
    sns.scatterplot(data=train, x='BeatsPerMinute', y=col)
    plt.title(f'{col} vs BeatsPerMinute')
    plt.show()


train = train[train['RhythmScore'] >= 0.2]

train = train[train['InstrumentalScore'] <= 0.7]
train = train[train['TrackDurationMs'] <= 430000]


test.info()


y = train['BeatsPerMinute']
X = train.drop(columns=['BeatsPerMinute'])
test_id = test['id']
test_data = test.drop(columns=['id'])



# from xgboost import XGBRegressor
# from sklearn.metrics import make_scorer, mean_squared_error
# from sklearn.model_selection import cross_val_score
# from bayes_opt import BayesianOptimization

# import warnings
# import xgboost as xgb
# from sklearn.metrics import make_scorer, mean_squared_error
# from sklearn.model_selection import cross_val_score
# import numpy as np
# warnings.filterwarnings("ignore", module="xgboost")
# def xgb_cv_score_gpu(n_estimators,
#                      max_depth,
#                      learning_rate,
#                      subsample,
#                      colsample_bytree,
#                      reg_alpha,
#                      reg_lambda,
#                      min_child_weight,
#                      gamma):
#     params = dict(
#         objective='reg:squarederror',
#         tree_method='hist',           # 1. ä¸�å†�ç”¨ gpu_hist
#         device='cuda:0',              # 2. æ–°æ�¥å�£ï¼šæŒ‡å®š GPU
#         n_estimators=int(n_estimators),
#         max_depth=int(max_depth),
#         learning_rate=learning_rate,
#         subsample=subsample,
#         colsample_bytree=colsample_bytree,
#         reg_alpha=reg_alpha,
#         reg_lambda=reg_lambda,
#         min_child_weight=min_child_weight,
#         gamma=gamma,
#         random_state=42,
#         n_jobs=-1                     # CPU çº¿ç¨‹ä»�å�¯å�ƒæ»¡
#     )

#     model = xgb.XGBRegressor(**params)

#     # 3. æŠŠè¾“å…¥æ•°æ�®æ�¬åˆ°å�Œä¸€å¼ å�¡ï¼Œé�¿å…� device  mismatch
#     X_gpu = X.astype(np.float32)      # XGBoost å–œæ¬¢ float32
#     y_gpu = y.astype(np.float32)

#     neg_rmse = make_scorer(mean_squared_error, squared=False, greater_is_better=False)
#     scores = cross_val_score(model, X_gpu, y_gpu, cv=5, scoring=neg_rmse)
#     return scores.mean()

# pbounds_xgb = {
#     'n_estimators': (100, 2000),
#     'max_depth': (3, 12),
#     'learning_rate': (0.005, 0.3),
#     'subsample': (0.1, 1.0),
#     'colsample_bytree': (0.5, 1.0),
#     'reg_alpha': (0, 1),
#     'reg_lambda': (0, 1),
#     'min_child_weight': (1, 20),
#     'gamma': (0, 5)
# }

# xgb_bo = BayesianOptimization(
#     f=xgb_cv_score_gpu,
#     pbounds=pbounds_xgb,
#     random_state=42,
#     verbose=2
# )
# xgb_bo.maximize(init_points=40, n_iter=50)
# print("XGB æœ€ä½³å�‚æ•°ï¼š", xgb_bo.max['params'])
# print("XGB æœ€ä½³ RMSEï¼š", xgb_bo.max['target'])
# # XGB æœ€ä½³å�‚æ•°ï¼š {'n_estimators': 538.2118241519671, 'max_depth': 4.7419613400218275, 'learning_rate': 0.01, 'subsample': 0.6390233580324196, 'colsample_bytree': 0.6837654101115, 'reg_alpha': 1.0, 'reg_lambda': 1.0, 'min_child_weight': 17.693259537697724, 'gamma': 0.9956146615696442}
# # XGB æœ€ä½³ RMSEï¼š -26.460347493489582


# from catboost import CatBoostRegressor
# from sklearn.metrics import make_scorer, mean_squared_error
# from sklearn.model_selection import cross_val_score
# from bayes_opt import BayesianOptimization

# def ctb_cv_score_gpu(iterations,
#                      depth,
#                      learning_rate,
#                      subsample,
#                      rsm,               # ä»�ä¿�ç•™ï¼Œä½†ä¸‹é�¢å¼ºåˆ¶å†™æˆ� 1.0
#                      l2_leaf_reg,
#                      border_count):
#     params = dict(
#         loss_function='RMSE',
#         task_type='GPU',
#         devices='0',
#         iterations=int(iterations),
#         depth=int(depth),
#         learning_rate=learning_rate,
#         subsample=subsample,
#         bootstrap_type='Bernoulli',   # ä¸Šæ¬¡ä¿®å¤�
#         rsm=1.0,                      # å…³é”®ä¿®å¤�ï¼šGPU+RMSE å¿…é¡» 1.0
#         l2_leaf_reg=l2_leaf_reg,
#         border_count=int(border_count),
#         random_seed=42,
#         verbose=False
#     )

#     model = CatBoostRegressor(**params)
#     neg_rmse = make_scorer(mean_squared_error, squared=False, greater_is_better=False)
#     scores = cross_val_score(model, X, y, cv=3, scoring=neg_rmse)
#     return scores.mean()

# pbounds_ctb = {
#     'iterations': (100, 1500),
#     'depth': (4, 10),
#     'learning_rate': (0.01, 0.3),
#     'subsample': (0.1, 1.0),
#     'rsm': (0.5, 1.0),
#     'l2_leaf_reg': (1, 10),
#     'border_count': (32, 510)
# }

# ctb_bo = BayesianOptimization(
#     f=ctb_cv_score_gpu,
#     pbounds=pbounds_ctb,
#     random_state=42,
#     verbose=2
# )
# ctb_bo.maximize(init_points=40, n_iter=50)
# print("CatBoost æœ€ä½³å�‚æ•°ï¼š", ctb_bo.max['params'])
# print("CatBoost æœ€ä½³ RMSEï¼š", ctb_bo.max['target'])


# CatBoost æœ€ä½³å�‚æ•°ï¼š {'iterations': 223.8895028726873, 'depth': 5.175897174514871, 'learning_rate': 0.023115913784056037, 'subsample': 0.6626651653816322, 'rsm': 0.6943386448447411, 'l2_leaf_reg': 3.442141285965063, 'border_count': 216.80846454088024}
# CatBoost æœ€ä½³ RMSEï¼š -26.459935262644837


# 1. å®‰è£… GPU ç‰ˆ LightGBMï¼ˆKaggle è‡ªå¸¦ CUDAï¼Œæ— éœ€å†�è£…é©±åŠ¨ï¼‰
#    å¦‚æ�œæ��ç¤ºå·²å®‰è£…å�¯è·³è¿‡
# !pip install --force-reinstall --no-deps lightgbm --extra-index-url https://pypi.nvidia.com

# import lightgbm as lgb
# from sklearn.datasets import make_regression
# from sklearn.model_selection import cross_val_score
# from bayes_opt import BayesianOptimization
# import numpy as np
# from sklearn.metrics import make_scorer, mean_squared_error



# # 3. å®šä¹‰ GPU ç›®æ ‡å‡½æ•° -----------------------------
# def lgb_cv_score_gpu(
#         num_leaves,
#         learning_rate,
#         n_estimators,
#         max_depth,
#         min_child_samples,
#         subsample,
#         colsample_bytree,
#         reg_alpha,
#         reg_lambda):
#     """
#     è´�å�¶æ–¯ä¼˜åŒ–çš„ç›®æ ‡å‡½æ•°ï¼ˆGPU ç‰ˆï¼‰ã€‚
#     è¿”å›�ï¼šäº¤å�‰éªŒè¯�çš„ RMSE å�‡å€¼ï¼ˆè´Ÿå€¼ï¼Œè¶Šå¤§è¶Šå¥½ï¼‰
#     """
#     params = {
#         'objective': 'regression',
#         'metric': 'rmse',
#         'device': 'gpu',           # å…³é”®ï¼�å�¯ç”¨ GPU
#         'gpu_platform_id': 0,      # å¤š GPU æ—¶å�¯æŒ‡å®š
#         'gpu_device_id': 0,
#         'num_leaves': int(num_leaves),
#         'learning_rate': learning_rate,
#         'n_estimators': int(n_estimators),
#         'max_depth': int(max_depth),
#         'min_child_samples': int(min_child_samples),
#         'subsample': subsample,
#         'colsample_bytree': colsample_bytree,
#         'reg_alpha': reg_alpha,
#         'reg_lambda': reg_lambda,
#         'verbose': -1,
#         'seed': 42,
#         'n_jobs': -1
#     }

#     model = lgb.LGBMRegressor(**params)
#     neg_rmse = make_scorer(mean_squared_error,
#                            squared=False,
#                            greater_is_better=False)
#     scores = cross_val_score(model, X, y,
#                              cv=5,            # Kaggle CPU/GPU æ··å�ˆæ—¶å�¯é€‚å½“å‡�å°�
#                              scoring=neg_rmse,
#                              error_score='raise')
#     return np.mean(scores)

# # 4. æ�œç´¢ç©ºé—´ ------------------------------------
# pbounds = {
#     'num_leaves': (10, 100),
#     'learning_rate': (0.005, 0.3),
#     'n_estimators': (100, 3000),
#     'max_depth': (3, 30),
#     'min_child_samples': (10, 100),
#     'subsample': (0.1, 1.0),
#     'colsample_bytree': (0.1, 1.0),
#     'reg_alpha': (0.0, 1.0),
#     'reg_lambda': (0.0, 1.0)
# }

# # 5. è´�å�¶æ–¯ä¼˜åŒ–å™¨ ---------------------------------
# optimizer = BayesianOptimization(
#     f=lgb_cv_score_gpu,
#     pbounds=pbounds,
#     random_state=42,
#     verbose=2
# )

# # 6. è¿�è¡Œä¼˜åŒ– -------------------------------------
# optimizer.maximize(init_points=60, n_iter=80)

# # 7. è¾“å‡ºç»“æ�œ -------------------------------------
# print("æœ€ä½³ GPU å�‚æ•°ç»„å�ˆï¼š", optimizer.max['params'])
# print("æœ€ä½³ RMSEï¼š", optimizer.max['target'])
# # æœ€ä½³ GPU å�‚æ•°ç»„å�ˆï¼š {'num_leaves': 21.656914144831685, 'learning_rate': 0.005, 'n_estimators': 475.2154165120317, 'max_depth': 30.0, 'min_child_samples': 14.997513023602709, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 1.0, 'reg_lambda': 0.0}
# # æœ€ä½³ RMSEï¼š -26.459497705112483


# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # 1. æµ®ç‚¹ â†’ è§„æ•´
# best = xgb_bo.max['params']
# params = dict(
#     objective='reg:squarederror',
#     tree_method='hist',               # æœ‰ GPU å�¯æ”¹ 'gpu_hist'
#     n_estimators=538,
#     max_depth=5,
#     learning_rate=0.01,
#     subsample=0.6390233580324196,
#     colsample_bytree=0.6837654101115,
#     reg_alpha=1.0,
#     reg_lambda=1.0,
#     min_child_weight=17.693259537697724,
#     gamma=0.9956146615696442,
#     random_state=42,
#     n_jobs=-1
# )

# # 2. åˆ‡ 20% å�šæ—©å�œéªŒè¯�ï¼ˆåˆ†å±‚å�¯é€‰ï¼‰
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 3. è®­ç»ƒ
# model = XGBRegressor(**params)
# model.fit(
#     X_train, y_train,
#     eval_set=[(X_valid, y_valid)],
#     early_stopping_rounds=50,
#     verbose=0          # è®¾ä¸º 100 å�¯çœ‹åˆ°è¿›åº¦
# )

# # 4. é¢„æµ‹
# pred = model.predict(test_data)
# submission_df = pd.DataFrame({
#     'id': test_id,  
#     'BeatsPerMinute': pred
# })


# submission_df.to_csv('submission_XGBoost.csv', index=False)

# print("submission_XGBoost.csv")
# # 5. å¦‚æ�œ test_data æœ‰çœŸå€¼ï¼Œå�¯ç®— RMSE
# # rmse = mean_squared_error(y_true, pred, squared=False)


# import pandas as pd
# from catboost import CatBoostRegressor

# # ---------------- 1. æŠŠæµ®ç‚¹å�‚æ•°æ•´ç�†æˆ� CatBoost éœ€è¦�çš„æ ¼å¼� ----------------
# params = dict(
#     loss_function='RMSE',
#     task_type='GPU',
#     devices='0',
#     bootstrap_type='Bernoulli',
#     random_seed=42,
#     verbose=False,
    
#     iterations=223,
#     depth=5,
#     learning_rate=0.023115913784056037,
#     subsample=0.6626651653816322,
#     rsm=1.0,                         # ä»�ä¿�æŒ� GPU+RMSE å¿…é¡» 1.0
#     l2_leaf_reg=3.442141285965063,
#     border_count=216
# )

# # ---------------- 2. é‡�æ–°åœ¨å…¨é‡�æ•°æ�®ä¸Šè®­ç»ƒ ----------------
# model = CatBoostRegressor(**params)
# model.fit(X, y)                      # å¦‚æ�œæœ‰æ—¶åº�åˆ’åˆ†ï¼Œè¿™é‡Œæ�¢ä½ è‡ªå·±çš„è®­ç»ƒ/éªŒè¯�é€»è¾‘

# # ---------------- 3. é¢„æµ‹æµ‹è¯•é›† ----------------
# y_pred = model.predict(test_data)       #  ndarray





# submission_df = pd.DataFrame({
#     'id': test_id,  
#     'BeatsPerMinute': y_pred
# })


# submission_df.to_csv('submission_catboost.csv', index=False)

# print("submission_catboost.csv")


import lightgbm as lgb
from sklearn.datasets import make_classification
import pandas as pd



params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 22, 
    'learning_rate': 0.005, 
    'n_estimators': 475, 
    'max_depth': 30, 
    'min_child_samples':15,
    'subsample': 1.0, 
    'colsample_bytree': 1.0, 
    'reg_alpha': 1.0, 
    'reg_lambda': 0,
    "verbose": -1,  # éš�è—�ä¸�å¿…è¦�çš„è­¦å‘Š
    'n_jobs': -1,
    'seed': 42
}

model = lgb.LGBMRegressor(**params)



model.fit(X, y)

predictions = model.predict(test_data)



submission_df = pd.DataFrame({
    'id': test_id,  
    'BeatsPerMinute': predictions
})


submission_df.to_csv('submission.csv', index=False)

print("submission.csv")


# lgbm_sub = pd.read_csv("/kaggle/working/submission_lgbm.csv")
# xgb_sub = pd.read_csv("/kaggle/working/submission_XGBoost.csv")
# catb_sub = pd.read_csv("/kaggle/working/submission_catboost.csv")

# predictions = 0.6*lgbm_sub['BeatsPerMinute']+0.2*xgb_sub['BeatsPerMinute']+0.2*catb_sub['BeatsPerMinute']

# submission_df = pd.DataFrame({
#     'id': test_id,  
#     'BeatsPerMinute': predictions
# })
# submission_df.to_csv('submission.csv', index=False)

