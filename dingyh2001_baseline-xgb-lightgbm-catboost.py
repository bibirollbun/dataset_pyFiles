# data processing libraries
import numpy as np
import pandas as pd
import polars as pl
import shap

from datetime import datetime
import os


from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost  import XGBRegressor
import joblib   

# for monitoring progress
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns # plots for statistical analysis
import matplotlib.pyplot as plt # for data visualization

# define default colors for plots in notebook
from matplotlib import cycler
from matplotlib.colors import LinearSegmentedColormap
colors = ["#068D9D", "#53599A", "#607BB0", "#6D9DC5", "#77BECF", "#80DED9", "#AEECEF"]

plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=cycler('color', colors))

SEED = 42


def reduce_mem_usage(dataframe, dataset):
    """
    参数:
        dataframe: 需要优化内存使用的pandas数据框
        dataset: 数据集名称(用于打印信息)
        
    返回:
        优化后的数据框
        
    功能:
        1. 遍历数据框的每一列
        2. 根据列的数据类型和取值范围,将其转换为占用内存更小的数据类型
        3. 对于整数类型,尝试转换为int8/int16/int32/int64
        4. 对于浮点类型,尝试转换为float16/float32/float64
        5. 打印内存使用前后的对比信息
    """
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        if col == 'timestamp':
            continue
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            # np.iinfo()：NumPy的一个函数，用来获取整数类型的信息，包括它的最小值（.min）和最大值（.max）。
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe


# pl_train = pl.read_parquet('./drw-crypto-market-prediction/train.parquet')
# pl_test = pl.read_parquet('./drw-crypto-market-prediction/test.parquet')
# sample = pd.read_csv("./drw-crypto-market-prediction/sample_submission.csv")

pl_train = pl.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
pl_test = pl.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
sample=pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")


df_train = pl_train.to_pandas()
df_test = pl_test.to_pandas()


df_train = reduce_mem_usage(df_train, "train")
df_test = reduce_mem_usage(df_test, "test")

df_train = df_train.reset_index()

df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_test = df_test.replace([np.inf, -np.inf], np.nan).drop('label',axis=1)

proprietary_features = [col for col in df_train.columns if col.startswith('X')]
print(f"There are {len(proprietary_features)} anonymized market proprietary features.")

basic_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
print(f"There are {len(basic_features)} basic features.\n")

all_features = proprietary_features + basic_features
target = 'label'
df_train = df_train[ ['timestamp'] + basic_features + proprietary_features + ['label']]

print(f"Train dataset contains {df_train.shape[0]} rows and {df_train.shape[1]} columns." )
print(f"Test dataset contains {df_test.shape[0]} rows and {df_test.shape[1]} columns." )


df_test


df_train[['X697','X698','X699','X700','X701','X702','X703']] 


df_test


lgb_params={"boosting_type": "gbdt",
            "metric": 'rmse', # 评估指标 均方根误差（用于回归任务）
            'random_state': 2025,
            "max_depth": 10, # 限制树的最大深度
            "n_estimators": 120, # 弱学习树的个数
            "learning_rate": 0.1,
            'num_leaves':64, # 叶子节点的最大数量
            "max_bin":255, # 最大分箱数
            "colsample_bytree": 0.6,   # 构建每棵树时随机选择的特征比例。
            "colsample_bynode": 0.6,  # 在每个节点分裂时随机选择的特征比例。
            "verbose": 0,  # 控制日志信息的输出级别。-1禁止输出，0输出最少得日志信息，1输出详细的日志信息
            "reg_alpha": 0.2, "reg_lambda": 5, # L1和L2正则化系数
            "extra_trees":True, # 当为 True 时，每个树的特征分裂点会随机选择，而不是通过贪心搜索
            'device':'gpu', 'gpu_use_dp':True,
            }

cat_params={'task_type':'GPU',
            'random_state':2025,
            'eval_metric'         :'RMSE',  # 评估指标
            'bagging_temperature' : 0.50,    # 控制随机采样的强度，用于训练不同的弱学习器。
            'iterations'          : 200,     # 弱学习器（树）的最大迭代次数，即模型训练的轮数
            'learning_rate'       : 0.1,
            'max_depth'           : 12,
            'l2_leaf_reg'         : 1.25,
            'min_data_in_leaf'    : 24,      # 叶节点中的最小数据量
            'random_strength'     : 0.25,    # 随机性强度，控制模型训练中随机特征分裂点的权重
            'verbose'             : 0,
            'loss_function'       :'RMSE',   # 多目标预测
            # 'od_wait'             :50,     # 如果 50 轮没有提升，则停止
          }

xgb_params={'random_state': 2025, 
            'n_estimators': 125,       # 最大树的数量，即迭代的轮次
            'learning_rate': 0.1, 
            'max_depth': 10,           # 每棵树的最大深度，控制树的复杂度。
            'reg_alpha': 0.08,         # L1 正则化参数
            'reg_lambda': 0.8,         # L2 正则化参数
            'subsample': 0.95,         # 用于训练每棵树的样本比例
            'colsample_bytree': 0.6,   # 每棵树训练时随机选择的特征比例。
            'min_child_weight': 3,     # 子节点所需的最小权重（样本量的加权总和）
            'device': 'cuda',          # 1. 明确指定使用 CUDA 设备 (GPU)
            'tree_method': 'hist',     # 2. 配合使用 'hist' 算法
            'verbose': 0,
            # 'early_stopping_rounds': 10,
           }



from sklearn.model_selection import train_test_split

# 1. 加载示例数据
X, y = df_train[all_features], df_train[target]
# 2. 划分训练集和测试集
# 对于时间序列数据，我们应该按时间顺序分割，而不是随机分割
# 使用前80%的数据作为训练集，后20%的数据作为测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. 训练XGBoost模型
model = XGBRegressor(**xgb_params)
model.fit(X_train, y_train)

print("XGBoost模型训练完成！")
# 1. 创建一个解释器（Explainer）
# 对于树模型，使用 TreeExplainer 效率最高
explainer = shap.TreeExplainer(model)

# 2. 计算SHAP值
# 我们可以对测试集进行计算，以了解模型在未知数据上的表现
shap_values = explainer.shap_values(X_test)

# 3. 绘制条形图进行排序可视化
print("\n方法一：SHAP特征重要性排序图（条形图）")
shap.summary_plot(shap_values, X_test, plot_type="bar")

# 1. 计算每个特征的平均绝对SHAP值
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# 2. 将结果整理成Pandas DataFrame，方便排序和查看
feature_names = X_test.columns
shap_summary = pd.DataFrame({
    'feature': feature_names,
    'shap_importance': mean_abs_shap
})

# 3. 对DataFrame按SHAP重要性进行降序排序
shap_summary_sorted = shap_summary.sort_values('shap_importance', ascending=False)

# 4. 打印排序后的结果
print("\n方法二：排序后的SHAP特征重要性DataFrame")
# 将SHAP特征重要性数据保存为Excel文件
# shap_summary_sorted.to_excel('shap_feature_importance.xlsx', index=False)
# print("SHAP特征重要性已保存到 'shap_feature_importance.xlsx'")
shap_summary_sorted

# 筛选出importance >= 0.01的特征
# important_features = shap_summary_sorted[shap_summary_sorted['shap_importance'] != 0]['feature'].tolist()
important_features = [
    "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
    "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
]
# # 打印筛选出的特征数量
# print(f"重要性大于等于0.01的特征数量: {len(important_features)}")

# # 打印筛选出的特征列表
# print("\n重要性大于等于0.01的特征:")
# print(important_features)



# Create a directory to store the trained models
if not os.path.exists('models'):
    os.mkdir('models')

# Define the path to load pre-trained models (if not in training mode)
model_path = './models'

# Initialize a list to store trained models
models = []

TRAINING = True
# Function to train a model or load a pre-trained model
def Train(model_dict, model_name, train=None):
    if TRAINING:
        train_ = train

         # Get the model from the dictionary
        model = model_dict[model_name]

        # Train the model based on the type (LightGBM, XGBoost, or CatBoost)
        if model_name == 'lgb':
            # Train LightGBM model with early stopping and evaluation logging
            model.fit(train_[important_features],
                      train_[target],
                    #   eval_set=[(valid_[all_features], valid_[target])],
                      )
            
        elif model_name == 'cat':
            # Prepare evaluation set for CatBoost
            # evalset = cat.Pool( valid_[all_features], valid_[target] )
            
            # Train CatBoost model with early stopping and verbose logging
            model.fit(train_[important_features],
                      train_[target],
                    #   eval_set=[(evalset)]
                     )
            
        else:
            # Train XGBoost model with early stopping and verbose logging
            model.fit(train_[important_features],
                      train_[target], 
                    #   eval_set=[(valid_[all_features], valid_[target])],
                      )

        # Append the trained model to the list
        models.append(model)
        
        # Save the trained model to a file
        joblib.dump(model, f'{model_path}/{model_name}.model')
        
        # Delete training data to free up memory
        del train
        
        # Collect garbage to free up memory
        import gc
        gc.collect()
        
    else:
        # If not in training mode, load the pre-trained model from the specified path
        models.append(joblib.load(f'{model_path}/{model_name}.model'))
        
    return 

# Dictionary to store different models with their configurations
model_dict = {
    'lgb': LGBMRegressor(**lgb_params),
    'xgb': XGBRegressor(**xgb_params),
    'cat': CatBoostRegressor(**cat_params),
}

print(f'lgb')
Train(model_dict, 'lgb', df_train.sample(100_000)) if TRAINING else Train(model_dict, 'lgb')
print(f'xgb')
Train(model_dict, 'xgb', df_train.sample(100_000)) if TRAINING else Train(model_dict, 'xgb')
print(f'cat')
Train(model_dict, 'cat', df_train.sample(100_000)) if TRAINING else Train(model_dict, 'cat')
print(f'Finished')


def predict(test: pl.DataFrame):
    """Make a prediction."""
    # All the responders from the previous day are passed in at time_id == 0. We save them in a global variable for access at every time_id.
    # Use them as extra features, if you like.
    test = test[important_features]
    test_preds = 0.55 * models[0].predict(test) + 0.25 * models[1].predict(test) + 0.2 * models[2].predict(test)
    
    return test_preds


sample["prediction"] = predict(df_test)
sample.to_csv("submission.csv", index=False)
sample.head()

