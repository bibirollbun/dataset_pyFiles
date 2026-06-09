raw_data_dir = '/kaggle/input/m5-forecasting-accuracy/'
processed_data_dir = '/kaggle/input/m5-a1-processed/processed/'
model_dir = '/kaggle/working/'
log_dir = '/kaggle/working/'


####################################################################################
####################### 1-2. recursive model by store & cat ########################
####################################################################################


ver, KKK = 'priv', 0


STORES = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
CATS = ['HOBBIES','HOUSEHOLD', 'FOODS']


# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

# custom imports
from multiprocessing import Pool

warnings.filterwarnings('ignore')


########################### Helpers
#################################################################################
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    
## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df


########################### Helper to load data by store ID
#################################################################################
# Read data
def get_data_by_store(store, dept):
    
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    df = df[df['d']>=START_TRAIN]
    

    df = df[(df['store_id']==store) & (df['cat_id']==dept)]

    df2 = pd.read_pickle(MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
        
    df3 = pd.read_pickle(LAGS).iloc[:,3:]
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2
    
    df = pd.concat([df, df3], axis=1)
    del df3
    
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET]+features]
    
    df = df.reset_index(drop=True)
    
    return df, features

# Recombine Test set after training
def get_base_test():
    base_test = pd.DataFrame()

    for store_id in STORES:
        for state_id in CATS:
            temp_df = pd.read_pickle(processed_data_dir+'test_'+store_id+'_'+state_id+'.pkl')
            temp_df['store_id'] = store_id
            temp_df['cat_id'] = state_id
            base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    
    return base_test


########################### Helper to make dynamic rolling lags
#################################################################################
def make_lag(LAG_DAY):
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'sales_lag_'+str(LAG_DAY)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(LAG_DAY)).astype(np.float16)
    return lag_df[[col_name]]


def make_lag_roll(LAG_DAY):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


# =================================================================================
# XGBoost Model params (Corrected Objective)
# =================================================================================
import xgboost as xgb
import psutil
import random
import numpy as np

# --------------------------- 1. Define Constants First ---------------------------
VER = 1
SEED = 42
N_CORES = psutil.cpu_count()

# Define and call seed_everything function
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
seed_everything(SEED)


# --------------------------- 2. Define XGBoost Parameter Dictionary ---------------------------
# XGBoost Parameter Dictionary
xgb_params = {
    # ========================== CORE CORRECTION ==========================
    'objective': 'reg:tweedie',         # CORRECTED: Was 'tweedie', now 'reg:tweedie'
    # ===================================================================

    'tweedie_variance_power': 1.1,
    'eval_metric': 'rmse',
    'booster': 'gbtree',

    # Core Parameters
    'eta': 0.015,
    'max_depth': 8,
    'subsample': 0.5,
    'colsample_bytree': 0.5,

    # Regularization
    'min_child_weight': 2**8-1,
    'lambda': 0.1,
    'alpha': 0.1,

    # Performance & Compatibility
    'tree_method': 'hist',
    'max_bin': 100,
    'enable_categorical': True,

    'seed': SEED,
    'n_jobs': N_CORES,
    'verbosity': 1,
}

# --------------------------- 3. Define Other Variables ---------------------------
TARGET      = 'sales'
START_TRAIN = 700
END_TRAIN   = 1941
P_HORIZON   = 28
USE_AUX     = False

remove_features = ['id','cat_id', 'state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]
mean_features   = ['enc_store_id_dept_id_mean','enc_store_id_dept_id_std',
                   'enc_item_id_store_id_mean','enc_item_id_store_id_std']

# File Paths
ORIGINAL = raw_data_dir
BASE     = processed_data_dir+'grid_part_1.pkl'
PRICE    = processed_data_dir+'grid_part_2.pkl'
CALENDAR = processed_data_dir+'grid_part_3.pkl'
LAGS     = processed_data_dir+'lags_df_28.pkl'
MEAN_ENC = processed_data_dir+'mean_encoding_df.pkl'


# Recursive Prediction Parameters
SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])

print("Variables and XGBoost parameters have been correctly defined.")


########################### Train Models (XGBoost Version)
#################################################################################

# 确保所有需要的变量已定义 (SEED, N_CORES, etc.)
seed_everything(SEED)

for store_id in STORES:
    for cat_id in CATS: # 变量名改为cat_id以保持一致
        print('Train', store_id, cat_id)

        # get_data_by_store 函数现在接收 cat_id
        grid_df, features_columns = get_data_by_store(store_id, cat_id)

        train_mask = grid_df['d'] <= END_TRAIN
        valid_mask = train_mask & (grid_df['d'] > (END_TRAIN - P_HORIZON))
        preds_mask = (grid_df['d'] > (END_TRAIN - 100)) & (grid_df['d'] <= END_TRAIN + P_HORIZON)

        # 准备训练和验证数据
        X_train = grid_df[train_mask][features_columns]
        y_train = grid_df[train_mask][TARGET]
        X_valid = grid_df[valid_mask][features_columns]
        y_valid = grid_df[valid_mask][TARGET]

        # 确保类别特征是 'category' dtype
        for col in X_train.select_dtypes(include=['object']).columns:
             if col in features_columns:
                X_train[col] = X_train[col].astype('category')
                X_valid[col] = X_valid[col].astype('category')

        # --> XGBoost: 创建 DMatrix
        dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, y_valid, enable_categorical=True)

        # 保存测试集模板以供后续递归预测使用
        grid_df = grid_df[preds_mask].reset_index(drop=True)
        keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
        grid_df = grid_df[keep_cols]
        d_sales = grid_df[['d', 'sales']]
        substitute = d_sales['sales'].values
        substitute[(d_sales['d'] > END_TRAIN)] = np.nan
        grid_df['sales'] = substitute
        
        # 使用新的命名方式保存test pkl文件
        grid_df.to_pickle(f'{model_dir}test_{store_id}_{cat_id}.pkl')
        del grid_df, d_sales, substitute, X_train, y_train, X_valid, y_valid
        gc.collect()

        # --> XGBoost: 训练模型
        estimator = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=3000, # 与lgb_params中的n_estimators一致
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            verbose_eval=100,
            early_stopping_rounds=100 # 增加早停防止过拟合
        )
        
        # 特征重要性
        feat_imp_df = pd.DataFrame.from_dict(estimator.get_fscore(), orient='index', columns=['imp'])
        feat_imp_df = feat_imp_df.sort_values('imp', ascending=False).reset_index().rename(columns={'index': 'name'})
        display(feat_imp_df.head(25))

        # --> XGBoost: 使用pickle保存模型
        model_name = f'{model_dir}xgb_model_{store_id}_{cat_id}_v{VER}.bin'
        pickle.dump(estimator, open(model_name, 'wb'))
        
        del dtrain, dvalid, estimator, feat_imp_df
        gc.collect()

        MODEL_FEATURES = features_columns


########################### Prediction and Submission
#################################################################################

# 我们需要一个基础的测试集来进行递归预测
# 这个测试集包含了所有需要预测的item-day组合
base_test = get_base_test()

# 让我们为接下来的28天进行预测
for PREDICT_DAY in range(1, 29):    
    print('Predict | Day:', PREDICT_DAY)
    
    # 准备当天的测试数据
    test_df = base_test.copy()
    
    # 创建滚动特征
    # 注意：这里的逻辑依赖于base_test中的'sales'列被逐步填充
    rolling_lags = df_parallelize_run(make_lag_roll, ROLS_SPLIT)
    test_df = pd.concat([test_df, rolling_lags], axis=1)

    for store_id in STORES:
        for cat_id in CATS:
            
            # 加载对应的模型
            model_path = f'{model_dir}xgb_model_{store_id}_{cat_id}_v{VER}.bin'
            if os.path.exists(model_path):
                estimator = pickle.load(open(model_path, 'rb'))
                
                # 筛选出当前store和cat的测试数据
                mask = (test_df['store_id'] == store_id) & (test_df['cat_id'] == cat_id)
                
                # 创建DMatrix进行预测
                X_test = test_df[mask][MODEL_FEATURES]
                for col in X_test.select_dtypes(include=['object']).columns:
                    X_test[col] = X_test[col].astype('category')

                dtest = xgb.DMatrix(X_test, enable_categorical=True)

                # 进行预测
                day_preds = estimator.predict(dtest)
                
                # 将预测结果填充回我们的基础测试集
                # 这样下一次循环（预测下一天）时，滞后特征就能使用这次的预测结果
                base_test.loc[mask, 'sales'] = day_preds

    # 清理内存
    del rolling_lags, test_df
    gc.collect()

# 创建提交文件
# -----------------------------------
# 我们现在有了基础测试集中 'sales' 列的最终预测
# 需要将其转换为提交所需的宽格式
submission = pd.read_csv(f'{raw_data_dir}sample_submission.csv')

# 我们只关心d > 1941 的预测 (即评估阶段)
predictions = base_test[base_test['d'] > 1941][['id', 'd', 'sales']]

# 将 'd' 列转换为 F1, F2, ... F28 的格式
predictions['d'] = 'F' + (predictions['d'] - 1941).astype(str)

# 透视表格
submission_map = predictions.set_index(['id', 'd'])['sales'].unstack().reset_index()

# 确保列的顺序和名称与模板一致
submission = submission[['id']].merge(submission_map, on='id', how='left').fillna(0)

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print('Submission file generated!')

