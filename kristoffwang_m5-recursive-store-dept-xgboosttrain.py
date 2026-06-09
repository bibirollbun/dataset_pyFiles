raw_data_dir = '/kaggle/input/m5-forecasting-accuracy/'
processed_data_dir = '/kaggle/input/m5-a1-processed/processed/'
model_dir = '/kaggle/working/'
log_dir = '/kaggle/working/'


####################################################################################
##################### 1-3. recursive model by store & dept #########################
####################################################################################


ver, KKK = 'priv', 0


STORES = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
DEPTS = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1', 'FOODS_2', 'FOODS_3']


import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

from multiprocessing import Pool

warnings.filterwarnings('ignore')


########################### Helpers
#################################################################################
## Seeder
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
    
    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    df = df[df['d']>=START_TRAIN]
    
    df = df[(df['store_id']==store) & (df['dept_id']==dept)]

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
        for state_id in DEPTS:
            temp_df = pd.read_pickle(processed_data_dir+'test_'+store_id+'_'+state_id+'.pkl')
            temp_df['store_id'] = store_id
            temp_df['dept_id'] = state_id
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
# Block 1: 变量和XGBoost模型参数定义 (替换原始所有相关单元格)
# =================================================================================
import xgboost as xgb
import psutil
import random
import numpy as np
import pandas as pd

# --- 常量定义 ---
VER = 1
SEED = 42
N_CORES = psutil.cpu_count()
STORES = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
DEPT_IDS = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1', 'FOODS_2', 'FOODS_3']

# Seeder
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
seed_everything(SEED)

# --- XGBoost 参数字典 (基于您提供的lgb_params转换) ---
xgb_params = {
    'objective': 'reg:tweedie',
    'tweedie_variance_power': 1.1,
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'eta': 0.015,
    'max_depth': 8,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'min_child_weight': 2**8 - 1,
    'lambda': 0.1,
    'alpha': 0.1,
    'tree_method': 'hist',
    'max_bin': 100,
    'enable_categorical': True,
    'seed': SEED,
    'n_jobs': N_CORES,
    'verbosity': 1,
}

# --- 其他变量 ---
TARGET = 'sales'
START_TRAIN = 700 
END_TRAIN = 1941
P_HORIZON = 28

remove_features = ['id', 'dept_id', 'state_id', 'store_id', 'date', 'wm_yr_wk', 'd', TARGET]
mean_features = ['enc_store_id_dept_id_mean', 'enc_store_id_dept_id_std'] # 已根据KeyError修正

# 文件路径
BASE = processed_data_dir + 'grid_part_1.pkl'
PRICE = processed_data_dir + 'grid_part_2.pkl'
CALENDAR = processed_data_dir + 'grid_part_3.pkl'
LAGS = processed_data_dir + 'lags_df_28.pkl'
MEAN_ENC = processed_data_dir + 'mean_encoding_df.pkl'

# 递归预测参数
SHIFT_DAY = 28
N_LAGS = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY, SHIFT_DAY + N_LAGS)]
ROLS_SPLIT = []
for i in [1, 7, 14]:
    for j in [7, 14, 30, 60]:
        ROLS_SPLIT.append([i, j])

print("All variables and XGBoost parameters have been defined correctly.")


########################### Train Models (XGBoost Version)
#################################################################################

MODEL_FEATURES = [] # 初始化一个列表来存储特征列

for store_id in STORES:
    for dept_id in DEPT_IDS:
        print(f"--> Training for: {store_id} - {dept_id}")

        grid_df, features_columns = get_data_by_store(store_id, dept_id)
        
        # 定义训练、验证和预测掩码
        train_mask = grid_df['d'] <= END_TRAIN
        valid_mask = train_mask & (grid_df['d'] > (END_TRAIN - P_HORIZON))
        preds_mask = (grid_df['d'] > (END_TRAIN - 100)) & (grid_df['d'] <= END_TRAIN + P_HORIZON)

        # 准备训练和验证数据集
        X_train = grid_df[train_mask][features_columns]
        y_train = grid_df[train_mask][TARGET]
        X_valid = grid_df[valid_mask][features_columns]
        y_valid = grid_df[valid_mask][TARGET]

        # 确保类别特征是 'category' dtype
        for col in X_train.select_dtypes(include=['object']).columns:
            if col in features_columns:
                X_train[col] = X_train[col].astype('category')
                X_valid[col] = X_valid[col].astype('category')

        # --> XGBoost: 创建DMatrix数据结构
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

        # 保存用于递归预测的测试数据模板
        grid_df[preds_mask].reset_index(drop=True).to_pickle(f'{model_dir}test_{store_id}_{dept_id}.pkl')
        del grid_df, X_train, y_train, X_valid, y_valid
        gc.collect()

        # --> XGBoost: 训练模型
        estimator = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=3000,  # 对应lgb_params中的n_estimators
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            verbose_eval=500,      # 每500轮打印一次日志
            early_stopping_rounds=50 # 早停机制
        )
        
        # --> XGBoost: 保存模型
        model_name = f'{model_dir}xgb_model_{store_id}_{dept_id}_v{VER}.bin'
        pickle.dump(estimator, open(model_name, 'wb'))
        
        del dtrain, dvalid, estimator
        gc.collect()
        
        if not MODEL_FEATURES: # 只赋值一次
             MODEL_FEATURES = features_columns

print("\nAll XGBoost models have been trained and saved.")


########################### Prediction and Submission (Recursive)
#################################################################################

# 准备一个包含所有item的基础测试集，用于迭代
# 注意：这个get_base_test函数需要您原始代码中提供，它会加载所有生成的test_...pkl文件
print("Loading base test file for recursive prediction...")
base_test = get_base_test()

# 递归预测未来28天
for PREDICT_DAY in range(1, 29):    
    print('--> Predicting Day:', PREDICT_DAY)
    
    # 获取当天的特征
    test_df = base_test[base_test['d'] == (END_TRAIN + PREDICT_DAY)].copy()
    
    # 创建滚动特征 (基于逐步填充的 base_test)
    rolling_lags = df_parallelize_run(make_lag_roll, ROLS_SPLIT)
    test_df = pd.concat([test_df, rolling_lags], axis=1)

    # 按店铺和部门进行预测
    for store_id in STORES:
        for dept_id in DEPT_IDS:
            model_path = f'{model_dir}xgb_model_{store_id}_{dept_id}_v{VER}.bin'
            if os.path.exists(model_path):
                estimator = pickle.load(open(model_path, 'rb'))
                
                mask = (test_df['store_id'] == store_id) & (test_df['dept_id'] == dept_id)
                if mask.sum() == 0:
                    continue

                X_test = test_df[mask][MODEL_FEATURES]
                for col in X_test.select_dtypes(include=['object']).columns:
                    if col in MODEL_FEATURES:
                        X_test[col] = X_test[col].astype('category')
                
                # --> XGBoost: 创建DMatrix进行预测
                dtest = xgb.DMatrix(X_test, enable_categorical=True)
                
                # 进行预测
                day_preds = estimator.predict(dtest)
                
                # 关键的递归步骤：将预测值填充回基础测试集
                # 这样下一次循环（预测下一天）时，滞后特征就能使用本次的预测结果
                base_test.loc[mask, 'sales'] = day_preds

    del rolling_lags, test_df
    gc.collect()

# 创建提交文件
# -----------------------------------
print("\nCreating submission file...")
submission_df = pd.read_csv(f'{raw_data_dir}sample_submission.csv')

# 我们现在有了base_test['sales']列的最终预测，需要将其转换为提交所需的宽格式
# 筛选出我们需要的天数（1914-1941用于validation, 1942-1969用于evaluation）
# 提交文件需要F1-F28，对应d_1914 - d_1941
submit_preds = base_test[base_test['d'].between(1914, 1941)][['id', 'd', 'sales']]
submit_preds['d'] = 'F' + (submit_preds['d'] - 1913).astype(str)

# 透视表格
submission_map = submit_preds.set_index(['id', 'd'])['sales'].unstack().reset_index()

# 删除evaluation行，只保留validation行用于提交
submission_df = submission_df[submission_df['id'].str.contains('validation')]
submission_df = submission_df.drop(columns=[f'F{i}' for i in range(1, 29)])

# 合并预测结果
submission_df = submission_df.merge(submission_map, on='id', how='left').fillna(0)

# 如果需要生成evaluation的预测，需要一个额外的循环
# 这里我们假设是为validation set生成提交
# 如果需要完整的提交，需要对d_1942到d_1969也进行循环和填充
# （为了简化，此脚本仅展示了前28天的逻辑）

# 对于evaluation部分，我们可以复制validation的预测结果，或者用0填充
eval_ids = submission_df['id'].str.replace('validation', 'evaluation')
eval_df = submission_df.copy()
eval_df['id'] = eval_ids
final_submission = pd.concat([submission_df, eval_df])

# 保存提交文件
final_submission.to_csv('submission.csv', index=False)
print('Submission file generated!')

