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
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr


class Config:
    TRAIN_PATH       = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH        = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH  = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    FEATURES = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333"
    ]
    
    LABEL_COLUMN     = "label"
    N_FOLDS          = 3
    RANDOM_STATE     = 42


# Hyperparameters for XGBoost and LightGBM
XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",
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
    "device": "gpu",
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
    {"name": "xgb",  "Estimator": XGBRegressor,  "params": XGB_PARAMS}
]

MODEL_SLICES = [
    {"name": "full_data",   "cutoff": 0},
    {"name": "last_75pct",  "cutoff": 0},  
    {"name": "last_50pct",  "cutoff": 0}
]


def create_time_decay_weights(n: int, decay: float = 0.95) -> np.ndarray:
    """创建时间衰减权重"""
    positions = np.arange(n)
    normalized = positions / float(n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()


def load_data():
    """加载数据，仅加载必要的列"""
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
    return train_df, test_df, submission_df


def optimize_memory(df):
    """优化DataFrame内存使用"""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df


# 加载数据
train_df, test_df, submission_df = load_data()
n_samples = len(train_df)

# 优化内存
train_df = optimize_memory(train_df)
test_df = optimize_memory(test_df)

# 设置切片截止点
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

# 创建时间衰减权重
full_weights = create_time_decay_weights(n_samples)
kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)


# 交叉验证
for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
    print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
    X_valid = train_df.iloc[valid_idx][Config.FEATURES]
    y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

    for sl in MODEL_SLICES:
        slice_name = sl["name"]
        cutoff     = sl["cutoff"]
        subset     = train_df.iloc[cutoff:].reset_index(drop=True)
        rel_idx    = train_idx[train_idx >= cutoff] - cutoff
        print(f"模型设置 {slice_name}")
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
            
            model = Estimator(**params)
            if name == "xgb":
                model.fit(X_train, y_train, sample_weight=sw,
                          eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                model.fit(X_train, y_train, sample_weight=sw,
                          eval_set=[(X_valid, y_valid)])

            # OOF预测
            mask = valid_idx >= cutoff
            if mask.any():
                idxs = valid_idx[mask]
                oof_preds[name][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
            if cutoff > 0 and (~mask).any():
                oof_preds[name][slice_name][valid_idx[~mask]] = (
                    oof_preds[name]["full_data"][valid_idx[~mask]])

            # 测试预测
            test_preds[name][slice_name] += model.predict(test_df[Config.FEATURES])

    # 清理不再需要的变量以释放内存
    del X_train, y_train, X_valid, y_valid, subset, rel_idx, sw
    import gc
    gc.collect()





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
print("\n按学习器和切片划分的Pearson分数：")
print(pearson_scores)





# -- 按学习器对切片进行集成 --
learner_ensembles = {}
for learner_name, slice_scores in pearson_scores.items():
    # 简单集成
    oof_simple = np.mean(list(oof_preds[learner_name].values()), axis=0)
    test_simple = np.mean(list(test_preds[learner_name].values()), axis=0)
    score_simple = pearsonr(train_df[Config.LABEL_COLUMN], oof_simple)[0]

    # 加权集成
    total_score = sum(slice_scores.values())
    slice_weights = {sn: sc / total_score for sn, sc in slice_scores.items()}
    oof_weighted = sum(slice_weights[sn] * oof_preds[learner_name][sn]
                       for sn in slice_weights)
    test_weighted = sum(slice_weights[sn] * test_preds[learner_name][sn]
                        for sn in slice_weights)
    score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]

    print(f"\n{learner_name.upper()} 简单集成 Pearson:   {score_simple:.4f}")
    print(f"{learner_name.upper()} 加权集成 Pearson: {score_weighted:.4f}")

    learner_ensembles[learner_name] = {
        "oof_simple": oof_simple,
        "test_simple": test_simple,
        "oof_weighted": oof_weighted,
        "test_weighted": test_weighted
    }


# -- 对学习器进行最终集成（简单） --
final_oof = np.mean([le["oof_weighted"] for le in learner_ensembles.values()], axis=0)
final_test = np.mean([le["test_weighted"] for le in learner_ensembles.values()], axis=0)
final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]
print(f"\n最终集成 across learners Pearson: {final_score:.4f}")

# 保存提交文件
submission_df["prediction"] = final_test
submission_df.to_csv("submission.csv", index=False)
print("已写入 submission.csv")

