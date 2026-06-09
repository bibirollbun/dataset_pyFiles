!pip install /kaggle/input/janestreet2025-code/janestreet-0.1-py3-none-any.whl --force-reinstall --no-deps


%%time
import lightgbm as lgb
import polars as pl
import numpy as np
import os
import joblib
import gc
import time

from janestreet.data_processor import DataProcessor
from janestreet.config import PATH_MODELS
from janestreet.utils import create_folder




MODEL_NAME = "lgbm_full_v1"
TARGET_COL = "responder_6"
SKIP_DAYS = 677

LGB_PARAMS = {
    "objective": "regression_l2",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 62,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
    'device' : 'gpu',
    'gpu_use_dp': True,
}

def r2_weighted(y_true, y_pred, weights):
    if len(y_true) == 0:
        return np.nan
    weights = np.array(weights)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_mean = np.average(y_true, weights=weights)
    ss_tot = np.sum(weights * (y_true - weighted_mean) ** 2)

    return 1 - ss_res / (ss_tot + 1e-38)


t0 = time.time()
print("加载 partition 0–8 训练数据...")

processor = DataProcessor(
    name=MODEL_NAME,
    skip_days=SKIP_DAYS,
    use_aux_targets=False
)

df_all = processor.get_train_valid_data()
features = processor.features

print(f"训练数据行数: {df_all.height}")
print(f"特征数量: {len(features)}")
print(f"数据加载时间: {time.time() - t0:.2f} sec")


df_all = df_all.with_columns([pl.col(c).cast(pl.Float32) for c in features])


t1 = time.time()
print("\n提取训练矩阵...")

X_train = df_all.select(features).to_numpy()
y_train = df_all.select(TARGET_COL).to_numpy().flatten()
w_train = df_all.select("weight").to_numpy().flatten()

print(f"矩阵构建时间: {time.time() - t1:.2f} sec")

train_set = lgb.Dataset(
    X_train,
    label=y_train,
    weight=w_train,
    feature_name=features
)

del df_all
gc.collect()


# ====================== 3. FULL 训练 ======================
t2 = time.time()
print("\n开始 FULL LightGBM 训练...")

model = lgb.train(
    params=LGB_PARAMS,
    train_set=train_set,
    num_boost_round=300,
    valid_sets=[train_set],
    valid_names=["train"],
    callbacks=[lgb.log_evaluation(period=50)]
)

print(f"\n训练完成，耗时: {time.time() - t2:.2f} sec")

create_folder(PATH_MODELS)

save_txt = os.path.join(PATH_MODELS, f"{MODEL_NAME}.txt")
save_joblib = os.path.join(PATH_MODELS, f"{MODEL_NAME}.joblib")

model.save_model(save_txt)
joblib.dump(model, save_joblib)

print(f"模型已保存到: {save_txt}")
print(f"Joblib 已保存到: {save_joblib}")


del X_train, y_train, w_train, train_set 
gc.collect() 
print("已释放训练数据内存.")

df_test = processor.get_test_data()

df_test = df_test.with_columns([pl.col(c).cast(pl.Float32) for c in features])

print(f"LB 测试集行数: {df_test.height}")

X_test = df_test.select(features).to_numpy()
y_test = df_test.select(TARGET_COL).to_numpy().flatten()
w_test = df_test.select("weight").to_numpy().flatten()

print("开始预测 partition 9 ...")
y_pred_test = model.predict(X_test)

lb_score = r2_weighted(y_test, y_pred_test, w_test)
print(f"\n===== LB Score (Weighted R²) = {lb_score:.6f} =====")

del df_test, X_test, y_test, w_test
gc.collect()

print(f"\n总耗时: {time.time() - t0:.2f} sec")



import pandas as pd
import matplotlib.pyplot as plt


importance_gain = model.feature_importance(importance_type='gain')
importance_split = model.feature_importance(importance_type='split')

df_importance = pd.DataFrame({
    'feature': features,
    'gain': importance_gain,
    'split': importance_split
})

# 按 gain 排序
df_importance = df_importance.sort_values('gain', ascending=False)

print("\nTop 30 Important Features (by Gain):")
print(df_importance.head(30))

# ========== 可视化 ==========
plt.figure(figsize=(10, 12))
plt.barh(df_importance.head(30)['feature'], df_importance.head(30)['gain'])
plt.gca().invert_yaxis()
plt.title("Top 30 Feature Importance (Gain)")
plt.xlabel("Gain Importance")
plt.show()



df_importance['gain_norm'] = df_importance['gain'] / df_importance['gain'].sum()
df_importance['gain_cumsum'] = df_importance['gain_norm'].cumsum()

print(df_importance[['feature','gain_cumsum']].head(40))


top50_df = df_importance[df_importance['gain_cumsum'] <= 0.50]
print(top50_df['feature'])


df_importance.tail()['feature']

