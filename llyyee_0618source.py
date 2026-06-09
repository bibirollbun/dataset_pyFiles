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


import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
import lightgbm
import xgboost

class Config:
    TRAIN_PATH       = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH        = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH  = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    FEATURES         = [
        "X863","X856","X344","X598","X862","X385","X852","X603",
        "X860","X674","X415","X345","X137","X855","X174","X302",
        "X178","X532","X168","X612",
        "bid_qty","ask_qty","buy_qty","sell_qty","volume"
    ]
    LABEL_COLUMN     = "label"
    N_FOLDS          = 3
    RANDOM_STATE     = 42

# ===== GPU 支持检测 =====
def check_gpu_support():
    """更可靠的GPU检测函数"""
    # 检查Kaggle环境
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
        print("Kaggle环境检测到GPU支持")
        return True
    
    # 检查Colab环境
    if "COLAB_GPU" in os.environ:
        print("Colab环境检测到GPU支持")
        return True
    
    # 尝试使用nvidia-smi
    try:
        if os.system("nvidia-smi > /dev/null 2>&1") == 0:
            print("nvidia-smi检测到GPU支持")
            return True
    except:
        pass
    
    # 尝试使用PyTorch检测
    try:
        import torch
        if torch.cuda.is_available():
            print("PyTorch检测到GPU支持")
            return True
    except ImportError:
        pass
    
    print("未检测到可靠的GPU支持，使用CPU模式")
    return False

HAS_GPU = check_gpu_support()

# 打印环境信息
print("\n===== 环境信息 =====")
print(f"Python 版本: {sys.version}")
print(f"LightGBM 版本: {lightgbm.__version__}")
print(f"XGBoost 版本: {xgboost.__version__}")
print(f"GPU 支持: {'是' if HAS_GPU else '否'}")

# 如果运行在Kaggle环境，尝试安装OpenCL支持
if os.path.exists("/kaggle/input"):
    print("\n===== Kaggle环境优化 =====")
    try:
        # 安装OpenCL支持
        print("安装OpenCL支持...")
        os.system("apt-get install -y ocl-icd-opencl-dev > /dev/null 2>&1")
        os.system("pip install pyopencl > /dev/null 2>&1")
        print("OpenCL支持安装完成")
    except Exception as e:
        print(f"OpenCL支持安装失败: {str(e)}")

# Hyperparameters for XGBoost and LightGBM
# 使用更可靠的GPU参数设置
XGB_PARAMS = {
    "tree_method": "gpu_hist" if HAS_GPU else "hist",
    "device": "cuda:0" if HAS_GPU else "cpu",  # 明确指定设备索引
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1,
    "verbose": False,
}

LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "device": "gpu" if HAS_GPU else "cpu",
    "gpu_device_id": 0 if HAS_GPU else -1,  # 明确指定GPU设备ID
    "n_jobs": -1,
    "verbose": -1,
    "random_state": Config.RANDOM_STATE,
    "colsample_bytree": 0.5039,
    "learning_rate": 0.01260,
    "min_child_samples": 20,
    "min_child_weight": 0.1146,
    "n_estimators": 915,
    "num_leaves": 145,
    "reg_alpha": 19.2447,
    "reg_lambda": 55.5046,
    "subsample": 0.9709,
    "max_depth": 9
}

LEARNERS = [
    {"name": "xgb",  "Estimator": XGBRegressor,  "params": XGB_PARAMS},
    {"name": "lgbm", "Estimator": LGBMRegressor, "params": LGBM_PARAMS}
]

MODEL_SLICES = [
    {"name": "full_data",   "cutoff": 0},
    {"name": "last_75pct",  "cutoff": 0},  # to be set after loading
    {"name": "last_50pct",  "cutoff": 0}
]


def create_time_decay_weights(n: int, decay: float = 0.95) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / float(n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()


def load_data():
    train_df = pd.read_parquet(
        Config.TRAIN_PATH,
        columns=Config.FEATURES + [Config.LABEL_COLUMN]
    ).reset_index(drop=True)
    test_df = pd.read_parquet(
        Config.TEST_PATH,
        columns=Config.FEATURES
    ).reset_index(drop=True)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}, submission: {submission_df.shape}")
    
    # 优化内存使用
    for df in [train_df, test_df]:
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
    
    return train_df, test_df, submission_df

# ===== 增强的安全训练函数 =====
def safe_fit(model, X_train, y_train, eval_set=None, **kwargs):
    """
    带自动GPU回退的安全训练函数
    如果GPU训练失败，自动回退到CPU模式
    """
    try:
        # 尝试使用原始参数训练
        print(f"尝试使用 {model.get_params().get('device', 'cpu')} 训练...")
        model.fit(X_train, y_train, eval_set=eval_set, **kwargs)
        print("训练成功")
        return model
    except Exception as e:
        error_msg = str(e)
        print(f"\n⚠️ 训练错误: {error_msg}")
        
        # 检查是否是GPU相关错误
        gpu_errors = [
            "no opencl device found", 
            "gpu plugin is necessary", 
            "cuda", 
            "gpu", 
            "device",
            "gpu_id",
            "updater_gpu_hist"
        ]
        
        if any(err in error_msg.lower() for err in gpu_errors):
            print("检测到GPU相关错误，尝试回退到CPU模式...")
            
            # 克隆模型参数
            from copy import deepcopy
            cpu_params = deepcopy(model.get_params())
            
            # 修改为CPU参数
            if isinstance(model, LGBMRegressor):
                cpu_params["device"] = "cpu"
                cpu_params["gpu_device_id"] = -1  # 清除GPU设备ID
                print("LightGBM回退到CPU模式")
            elif isinstance(model, XGBRegressor):
                cpu_params["device"] = "cpu"
                cpu_params["tree_method"] = "hist"
                print("XGBoost回退到CPU模式")
            
            # 创建新的CPU模型
            cpu_model = model.__class__(**cpu_params)
            cpu_model.fit(X_train, y_train, eval_set=eval_set, **kwargs)
            return cpu_model
        else:
            # 非GPU错误，重新抛出异常
            print("非GPU错误，无法处理")
            raise e

# ===== 优化内存使用 =====
def reduce_mem_usage(df):
    """迭代优化DataFrame内存使用"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"原始内存使用: {start_mem:.2f} MB")
    
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
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"优化后内存使用: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% 减少)")
    return df

############################
# MAIN
############################

train_df, test_df, submission_df = load_data()

# 应用内存优化
print("\n优化训练数据内存使用...")
train_df = reduce_mem_usage(train_df)
print("\n优化测试数据内存使用...")
test_df = reduce_mem_usage(test_df)

n_samples = len(train_df)
# set slice cutoffs
MODEL_SLICES[1]["cutoff"] = int(0.25 * n_samples)
MODEL_SLICES[2]["cutoff"] = int(0.50 * n_samples)

# prepare storage for OOF and test preds
oof_preds = {
    learner["name"]: {sl["name"]: np.zeros(n_samples) for sl in MODEL_SLICES}
    for learner in LEARNERS
}
test_preds = {
    learner["name"]: {sl["name"]: np.zeros(len(test_df)) for sl in MODEL_SLICES}
    for learner in LEARNERS
}

full_weights = create_time_decay_weights(n_samples)
kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

# cross-validation
for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
    print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
    X_valid = train_df.iloc[valid_idx][Config.FEATURES]
    y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

    for sl in MODEL_SLICES:
        slice_name = sl["name"]
        cutoff     = sl["cutoff"]
        subset     = train_df.iloc[cutoff:].reset_index(drop=True)
        rel_idx    = train_idx[train_idx >= cutoff] - cutoff

        X_train = subset.iloc[rel_idx][Config.FEATURES]
        y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]

        # sample weights
        if cutoff == 0:
            sw = full_weights[train_idx]
        else:
            sw_total = create_time_decay_weights(len(subset))
            sw = sw_total[rel_idx]

        for learner in LEARNERS:
            name      = learner["name"]
            Estimator = learner["Estimator"]
            params    = learner["params"]

            model = Estimator(**params)
            
            # 使用安全训练函数
            print(f"\n训练 {name} 模型 ({slice_name} 切片)...")
            model = safe_fit(
                model,
                X_train, 
                y_train, 
                sample_weight=sw,
                eval_set=[(X_valid, y_valid)]
            )

            # OOF predictions
            mask = valid_idx >= cutoff
            if mask.any():
                idxs = valid_idx[mask]
                oof_preds[name][slice_name][idxs] = model.predict(
                    train_df.iloc[idxs][Config.FEATURES])
            if cutoff > 0 and (~mask).any():
                oof_preds[name][slice_name][valid_idx[~mask]] = (
                    oof_preds[name]["full_data"][valid_idx[~mask]])

            # test predictions
            test_preds[name][slice_name] += model.predict(test_df[Config.FEATURES])

# average test preds
for name in test_preds:
    for slice_name in test_preds[name]:
        test_preds[name][slice_name] /= Config.N_FOLDS

# compute Pearson scores per learner and slice
pearson_scores = {
    name: {slice_name: pearsonr(train_df[Config.LABEL_COLUMN], preds)[0]
           for slice_name, preds in slices.items()}
    for name, slices in oof_preds.items()
}
print("\nPearson scores by learner and slice:")
print(pearson_scores)

# -- Ensemble per learner across slices --
learner_ensembles = {}
for learner_name, slice_scores in pearson_scores.items():
    # simple ensemble
    oof_simple = np.mean(list(oof_preds[learner_name].values()), axis=0)
    test_simple = np.mean(list(test_preds[learner_name].values()), axis=0)
    score_simple = pearsonr(train_df[Config.LABEL_COLUMN], oof_simple)[0]

    # weighted ensemble
    total_score = sum(slice_scores.values())
    slice_weights = {sn: sc/total_score for sn, sc in slice_scores.items()}
    oof_weighted = sum(slice_weights[sn] * oof_preds[learner_name][sn]
                       for sn in slice_weights)
    test_weighted = sum(slice_weights[sn] * test_preds[learner_name][sn]
                        for sn in slice_weights)
    score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]

    print(f"\n{learner_name.upper()} Simple ensemble Pearson:   {score_simple:.4f}")
    print(f"{learner_name.upper()} Weighted ensemble Pearson: {score_weighted:.4f}")

    learner_ensembles[learner_name] = {
        "oof_simple": oof_simple,
        "test_simple": test_simple
    }

# -- Final ensemble across learners (simple) --
final_oof = np.mean([le["oof_simple"] for le in learner_ensembles.values()], axis=0)
final_test = np.mean([le["test_simple"] for le in learner_ensembles.values()], axis=0)
final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]
print(f"\nFINAL ensemble across learners Pearson: {final_score:.4f}")

# save submission
submission_df["prediction"] = final_test
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv")


!apt install ocl-icd-opencl-dev


import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
import lightgbm
import xgboost

class Config:
    TRAIN_PATH       = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH        = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH  = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    FEATURES         = [
        "X863","X856","X344","X598","X862","X385","X852","X603",
        "X860","X674","X415","X345","X137","X855","X174","X302",
        "X178","X532","X168","X612",
        "bid_qty","ask_qty","buy_qty","sell_qty","volume"
    ]
    LABEL_COLUMN     = "label"
    N_FOLDS          = 3
    RANDOM_STATE     = 42

# 打印环境信息
print("\n===== 环境信息 =====")
print(f"Python 版本: {sys.version}")
print(f"LightGBM 版本: {lightgbm.__version__}")
print(f"XGBoost 版本: {xgboost.__version__}")
print("已禁用GPU功能，使用CPU模式确保稳定性")

# 完全禁用GPU功能
XGB_PARAMS = {
    "tree_method": "hist",  # 使用CPU优化的直方图方法
    "device": "cpu",        # 明确指定使用CPU
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1,  # 使用所有CPU核心
    "verbose": False,
}

LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "device": "cpu",  # 明确指定使用CPU
    "n_jobs": -1,     # 使用所有CPU核心
    "verbose": -1,
    "random_state": Config.RANDOM_STATE,
    "colsample_bytree": 0.5039,
    "learning_rate": 0.01260,
    "min_child_samples": 20,
    "min_child_weight": 0.1146,
    "n_estimators": 915,
    "num_leaves": 145,
    "reg_alpha": 19.2447,
    "reg_lambda": 55.5046,
    "subsample": 0.9709,
    "max_depth": 9
}

LEARNERS = [
    {"name": "xgb",  "Estimator": XGBRegressor,  "params": XGB_PARAMS},
    {"name": "lgbm", "Estimator": LGBMRegressor, "params": LGBM_PARAMS}
]

MODEL_SLICES = [
    {"name": "full_data",   "cutoff": 0},
    {"name": "last_75pct",  "cutoff": 0},  # 加载数据后设置
    {"name": "last_50pct",  "cutoff": 0}
]


def create_time_decay_weights(n: int, decay: float = 0.95) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / float(n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()


def load_data():
    try:
        # 尝试加载数据
        train_df = pd.read_parquet(
            Config.TRAIN_PATH,
            columns=Config.FEATURES + [Config.LABEL_COLUMN]
        ).reset_index(drop=True)
        test_df = pd.read_parquet(
            Config.TEST_PATH,
            columns=Config.FEATURES
        ).reset_index(drop=True)
        submission_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"加载数据: 训练集 {train_df.shape}, 测试集 {test_df.shape}, 提交样本 {submission_df.shape}")
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        print("请确保文件路径正确且文件存在")
        print("提示: 在Kaggle Notebook中，路径应为 '/kaggle/input/drw-crypto-market-prediction/...'")
        print("在本地环境中，路径应为文件在您电脑上的实际位置")
        raise
    
    # 优化内存使用
    for df in [train_df, test_df]:
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
    
    return train_df, test_df, submission_df

# 增强的内存优化函数
def reduce_mem_usage(df):
    """迭代优化DataFrame内存使用"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"原始内存使用: {start_mem:.2f} MB")
    
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
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"优化后内存使用: {end_mem:.2f} MB (减少 {100 * (start_mem - end_mem) / start_mem:.1f}%)")
    return df

############################
# 主程序
############################

# 加载数据
train_df, test_df, submission_df = load_data()

# 应用内存优化
print("\n优化训练数据内存使用...")
train_df = reduce_mem_usage(train_df)
print("\n优化测试数据内存使用...")
test_df = reduce_mem_usage(test_df)

n_samples = len(train_df)
# 设置切片截断点
MODEL_SLICES[1]["cutoff"] = int(0.25 * n_samples)
MODEL_SLICES[2]["cutoff"] = int(0.50 * n_samples)

# 准备存储OOF和测试预测
oof_preds = {
    learner["name"]: {sl["name"]: np.zeros(n_samples) for sl in MODEL_SLICES}
    for learner in LEARNERS
}
test_preds = {
    learner["name"]: {sl["name"]: np.zeros(len(test_df)) for sl in MODEL_SLICES}
    for learner in LEARNERS
}

full_weights = create_time_decay_weights(n_samples)
kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

# 交叉验证
for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
    print(f"\n--- 第 {fold}/{Config.N_FOLDS} 折交叉验证 ---")
    X_valid = train_df.iloc[valid_idx][Config.FEATURES]
    y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

    for sl in MODEL_SLICES:
        slice_name = sl["name"]
        cutoff     = sl["cutoff"]
        subset     = train_df.iloc[cutoff:].reset_index(drop=True)
        rel_idx    = train_idx[train_idx >= cutoff] - cutoff

        X_train = subset.iloc[rel_idx][Config.FEATURES]
        y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]

        # 样本权重
        if cutoff == 0:
            sw = full_weights[train_idx]
        else:
            sw_total = create_time_decay_weights(len(subset))
            sw = sw_total[rel_idx]

        for learner in LEARNERS:
            name      = learner["name"]
            Estimator = learner["Estimator"]
            params    = learner["params"]

            print(f"\n训练 {name} 模型 ({slice_name} 切片)...")
            model = Estimator(**params)
            
            # 训练模型（完全使用CPU，无GPU尝试）
            model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_valid, y_valid)])
            print(f"{name} 模型训练完成")

            # OOF预测
            mask = valid_idx >= cutoff
            if mask.any():
                idxs = valid_idx[mask]
                oof_preds[name][slice_name][idxs] = model.predict(
                    train_df.iloc[idxs][Config.FEATURES])
            if cutoff > 0 and (~mask).any():
                oof_preds[name][slice_name][valid_idx[~mask]] = (
                    oof_preds[name]["full_data"][valid_idx[~mask]])

            # 测试集预测
            test_preds[name][slice_name] += model.predict(test_df[Config.FEATURES])

# 平均测试预测
for name in test_preds:
    for slice_name in test_preds[name]:
        test_preds[name][slice_name] /= Config.N_FOLDS

# 计算每个学习器和切片的Pearson分数
pearson_scores = {
    name: {slice_name: pearsonr(train_df[Config.LABEL_COLUMN], preds)[0]
           for slice_name, preds in slices.items()}
    for name, slices in oof_preds.items()
}
print("\n按学习器和切片的Pearson分数:")
print(pearson_scores)

# -- 跨切片的学习器集成 --
learner_ensembles = {}
for learner_name, slice_scores in pearson_scores.items():
    # 简单集成
    oof_simple = np.mean(list(oof_preds[learner_name].values()), axis=0)
    test_simple = np.mean(list(test_preds[learner_name].values()), axis=0)
    score_simple = pearsonr(train_df[Config.LABEL_COLUMN], oof_simple)[0]

    # 加权集成
    total_score = sum(slice_scores.values())
    slice_weights = {sn: sc/total_score for sn, sc in slice_scores.items()}
    oof_weighted = sum(slice_weights[sn] * oof_preds[learner_name][sn]
                       for sn in slice_weights)
    test_weighted = sum(slice_weights[sn] * test_preds[learner_name][sn]
                        for sn in slice_weights)
    score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]

    print(f"\n{learner_name.upper()} 简单集成 Pearson:   {score_simple:.4f}")
    print(f"{learner_name.upper()} 加权集成 Pearson: {score_weighted:.4f}")

    learner_ensembles[learner_name] = {
        "oof_simple": oof_simple,
        "test_simple": test_simple
    }

# -- 跨学习器的最终集成（简单平均）--
final_oof = np.mean([le["oof_simple"] for le in learner_ensembles.values()], axis=0)
final_test = np.mean([le["test_simple"] for le in learner_ensembles.values()], axis=0)
final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]
print(f"\n跨学习器最终集成的Pearson分数: {final_score:.4f}")

# 保存提交文件
submission_df["prediction"] = final_test
submission_df.to_csv("submission.csv", index=False)
print("已保存提交文件: submission.csv")

