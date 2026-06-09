# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import optuna # Import Optuna

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns
import catboost as cb
from sklearn.model_selection import KFold, train_test_split # Import KFold and train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# --- STAGE 1 (Pure Rules) DATA ---
# 載入原始合成資料，用來訓練階段一模型

df_100k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
df_10k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
df_2k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')



# 移除原始 ID 欄位
df_train = df_train.drop('id', axis = 1)
df_test =df_test.drop('id', axis = 1)


df_100k = pd.concat([df_100k, df_10k, df_2k], ignore_index=True)


print("\n--- Combined Dataset Info ---")
df_train.info()

print("\n--- Numerical Features Statistics ---")
print(df_train.describe())

print("\n--- Missing Values Check ---")
print(df_train.isnull().sum())


# Target variable distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_train['accident_risk'], kde=True)
plt.title('Distribution of Accident Risk')
plt.subplot(1, 2, 2)
sns.boxplot(y=df_train['accident_risk'])
plt.title('Box Plot of Accident Risk')
plt.show()

# Select numerical and boolean features for correlation matrix
corr_features_df = df_train.select_dtypes(include=['number', 'bool'])
plt.figure(figsize=(12, 10))
correlation_matrix = corr_features_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


def road_risk(X):
    return (
        0.3 * X["curvature"] +
        0.2 * (X["lighting"] == "night").astype(int) +
        0.1 * (X["weather"] != "clear").astype(int) +
        0.2 * (X["speed_limit"] >= 60).astype(int) +
        0.1 * (X["num_reported_accidents"] > 2).astype(int)
    )

def clipped(func):
    def clip_f(X):
        mu = func(X)
        sigma = 0.05 
        a, b = -mu / sigma, (1 - mu) / sigma
        Phi_a, Phi_b = stats.norm.cdf(a), stats.norm.cdf(b)
        phi_a, phi_b = stats.norm.pdf(a), stats.norm.pdf(b)
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b
    return clip_f

print("Applying 'base_prior' to df_train...")
df_train["base_prior"] = clipped(road_risk)(df_train)
print("Applying 'base_prior' to df_test...")
df_test["base_prior"] = clipped(road_risk)(df_test)
if df_100k is not None:
    print("Applying 'base_prior' to df_100k...")
    df_100k["base_prior"] = clipped(road_risk)(df_100k)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the accident risk prediction dataset.
    (Now safe to run on train, test, and 100k datasets)
    """
    df_new = df.copy()

    # --- 依賴 'num_reported_accidents' 的特徵 ---
    df_new['log_accidents'] = np.log1p(df_new['num_reported_accidents'])
    df_new["acc_density"] = df_new["num_reported_accidents"].astype(float) / (df_new["speed_limit"] + 1.0)
    df_new["acc_rate_per_lane"] = df_new["num_reported_accidents"].astype(float) / (df_new["num_lanes"] + 0.01)
    df_new["hotspot_flag"] = (df_new["num_reported_accidents"] >= 2)
    # 3. 基礎風險 vs 事故數
    df_new['prior_x_accidents_gt_2'] = df_new['base_prior'] * (df_new['num_reported_accidents'] > 2).astype(int)

    df_new['curvature_squared'] = df_new['curvature'] ** 2
    df_new['speed_x_curvature'] = df_new['speed_limit'] * df_new['curvature']
    df_new["lanes_x_speed"] = df_new["num_lanes"] * df_new["speed_limit"]
    df_new['lanes_curvature_ratio'] = df_new['curvature'] / df_new["num_lanes"]
    df_new["curv_speed_ratio"] = df_new["curvature"] / (df_new["speed_limit"] + 1e-6)
    df_new["speed_sq_over_curv"] = (df_new["speed_limit"] ** 2) / (df_new["curvature"] + 1e-6)
    df_new["curv_factor"] = df_new["curvature"] * (df_new["num_lanes"] / (df_new["speed_limit"] + 1.0))

    df_new['poor_visibility'] = (df_new['lighting'].isin(['night', 'dim']) | (df_new['weather'] == 'foggy'))
    df_new['adverse_weather'] = df_new['weather'].isin(['rainy', 'foggy'])
    df_new['school_traffic_time'] = ((df_new['school_season'] == True) & (df_new['time_of_day'].isin(['morning', 'afternoon'])))
    df_new['rush_hour_no_holiday'] = ((df_new['holiday'] == False) & (df_new['time_of_day'].isin(['morning', 'afternoon'])))
    df_new["weekend_or_hol"] = ((df_new["holiday"].astype(bool)) | (df_new["time_of_day"].isin(["evening", "night"])))
    df_new['dangerous_curve_no_signs'] = ((df_new['curvature'] > 0.5) & (df_new['road_signs_present'] == False))

    df_new['speed_in_poor_visibility'] = df_new['speed_limit'] * df_new['poor_visibility'].astype(int)
    df_new['curve_in_poor_visibility'] = df_new['curvature'] * df_new['poor_visibility'].astype(int)
    df_new['speed_in_adverse_weather'] = df_new['speed_limit'] * df_new['adverse_weather'].astype(int)

    weather_ord_map = {"clear": 0, "foggy": 1, "rainy": 2}
    lighting_ord_map = {"daylight": 0, "dim": 1, "night": 2}
    df_new["weather_ord"] = df_new["weather"].map(weather_ord_map).fillna(0).astype(int)
    df_new["lighting_ord"] = df_new["lighting"].map(lighting_ord_map).fillna(0).astype(int)
    df_new["light_weather_code"] = df_new["lighting_ord"] * 10 + df_new["weather_ord"]

    # 1. 基礎風險 vs 曲率
    df_new['prior_x_curvature'] = df_new['base_prior'] * df_new['curvature']
    
    # 2. 基礎風險 vs 速限
    df_new['prior_x_speed_45'] = df_new['base_prior'] * (df_new['speed_limit'] == 45).astype(int)
    df_new['prior_x_speed_35'] = df_new['base_prior'] * (df_new['speed_limit'] == 35).astype(int)

    print("Feature engineering complete!")
    return df_new


print("Applying feature engineering to df_train...")
df_train_fe = create_features(df_train)
print("Applying feature engineering to df_test...")
df_test_fe = create_features(df_test)

if df_100k is not None:
    print("Applying feature engineering to df_100k...")
    df_100k_fe = create_features(df_100k)
else:
    print("Skipping Stage 1 training because df_100k was not loaded.")
    df_train_fe['stage1_pred'] = df_train_fe['base_prior']
    df_test_fe['stage1_pred'] = df_test_fe['base_prior']


if df_100k is not None:
    print("\n--- STAGE 1 MODEL TRAINING ---")
    
    # 1. 準備階段一的資料
    target_variable = 'accident_risk'
    
    # 與 df_train_fe 對齊特徵欄位
    features_cols = df_train_fe.drop(columns=[target_variable]).columns
    
    X_stage1 = df_100k_fe[features_cols]
    y_stage1 = df_100k_fe[target_variable]
    
    # 確保 test 集和 stage1 的欄位也對齊
    X_train_stage2 = df_train_fe[features_cols].reindex(columns=features_cols, fill_value=0)
    X_test_stage2 = df_test_fe[features_cols].reindex(columns=features_cols, fill_value=0)

    categorical_cols = X_stage1.select_dtypes(include=['object', 'bool']).columns.tolist()
    print(f"Stage 1 Categorical features: {categorical_cols}")

    # 2. 訓練階段一模型 (使用 CV)
    N_SPLITS_STAGE1 = 5
    kf_stage1 = KFold(n_splits=N_SPLITS_STAGE1, shuffle=True, random_state=2025)
    
    stage1_models = []
    
    stage1_params = {
        'task_type': 'GPU',
        'devices': '0',
        'iterations': 2000,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 3.0,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'cat_features': categorical_cols,
        'verbose': 100,
        'random_seed': 2025,
        'early_stopping_rounds': 100
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf_stage1.split(X_stage1, y_stage1)):
        print(f"--- Stage 1, Fold {fold+1}/{N_SPLITS_STAGE1} ---")
        X_train, X_val = X_stage1.iloc[train_idx], X_stage1.iloc[val_idx]
        y_train, y_val = y_stage1.iloc[train_idx], y_stage1.iloc[val_idx]

        model = cb.CatBoostRegressor(**stage1_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        stage1_models.append(model)

    print("\n--- STAGE 1 PREDICTION GENERATION ---")
    
    # 3. 生成「超級特徵」
    # 對 df_train 和 df_test 進行預測
    # 這是 Stacking 的一種形式：使用 5 個模型的平均預測
    train_preds_list = []
    test_preds_list = []

    for model in stage1_models:
        train_preds_list.append(model.predict(X_train_stage2))
        test_preds_list.append(model.predict(X_test_stage2))
        
    # 計算平均預測值
    train_pred_stage1_feat = np.mean(train_preds_list, axis=0)
    test_pred_stage1_feat = np.mean(test_preds_list, axis=0)
    
    # 4. 將超級特徵添加回階段二的資料集
    df_train_fe['stage1_pred'] = train_pred_stage1_feat
    df_test_fe['stage1_pred'] = test_pred_stage1_feat
    
    print("Stage 1 'stage1_pred' super-feature added successfully.")


# --- 階段二資料準備 ---
print("\n--- Preparing Stage 2 (Bias Model) Data ---")
target_variable = 'accident_risk'

X = df_train_fe.drop([target_variable], axis = 1)
y = df_train_fe[target_variable]
df_test = df_test_fe.reindex(columns=X.columns, fill_value=0)

categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Stage 2 Data preparation complete")
print("\nCategorical features list:")
print(categorical_cols)


# --- 階段二控制開關  ---
USE_OPTUNA = True # 設為 True 來啟用 Optuna，False 使用預設參數
N_TRIALS = 50     # Optuna 嘗試次數 (如果啟用)
OPTUNA_TIMEOUT = 60 * 600 # Optuna 搜尋的最大時間 (秒)
N_SPLITS = 10

# --- Optuna 目標函式  ---
def objective(trial):
    params = {
        'task_type': 'GPU',
        'devices': '0',
        'bootstrap_type': 'Bayesian',
        'iterations': 3000, # Iterations for Optuna trial
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'cat_features': categorical_cols,
        'verbose': 0,
        'random_seed': 42,
        'early_stopping_rounds': 100
    }
    
    # 在 Optuna 試驗中執行交叉驗證
    kf_opt = KFold(n_splits=N_SPLITS, shuffle=True, random_state=1337) 
    rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf_opt.split(X, y)):
        X_train_opt, X_val_opt = X.iloc[train_idx], X.iloc[val_idx]
        y_train_opt, y_val_opt = y.iloc[train_idx], y.iloc[val_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(X_train_opt, y_train_opt, eval_set=(X_val_opt, y_val_opt), 
                  early_stopping_rounds=params['early_stopping_rounds'], verbose=0)
        
        y_pred_val = model.predict(X_val_opt)
        rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred_val))
        rmses.append(rmse)
        
    # 返回 N 個 fold 的平均 RMSE
    return np.mean(rmses)

# --- 參數選擇 ---
if USE_OPTUNA:
    print("\nStarting Optuna hyperparameter search (using 10-Fold CV internally)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)

    print(f"Optuna search complete! Best average RMSE: {study.best_value:.4f}")
    print("Best parameters found:")
    print(study.best_params)
    final_params = study.best_params
else:
    print("\nUsing default parameters...")
    final_params = {
        'learning_rate': 0.04,
        'depth': 8,
        'l2_leaf_reg': 0.0144,
        'bagging_temperature': 0.0965,
    }

# --- 最終交叉驗證訓練  ---
print(f"\nStarting final {N_SPLITS}-Fold training with best parameters...")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_predictions = np.zeros(X.shape[0])
test_predictions_list = []
models = []

base_params = {
    'task_type': 'GPU',
    'devices': '0',
    'bootstrap_type': 'Bayesian',
    'iterations': 4000, # Final training iterations
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'cat_features': categorical_cols,
    'verbose': 200,
    'random_seed': 42,
    'early_stopping_rounds': 200
}

# 使用 Optuna/default 找到的核心參數更新基礎參數
base_params.update(final_params)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = cb.CatBoostRegressor(**base_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # 儲存驗證集預測 (OOF)
    oof_predictions[val_idx] = model.predict(X_val)
    
    # 儲存測試集預測
    test_predictions_list.append(model.predict(df_test))
    
    # 儲存模型
    models.append(model)

print("\nCross-validation training complete!")


print("\nOverall Cross-Validation Evaluation (OOF)...")
oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
oof_r2 = r2_score(y, oof_predictions)

print(f"OOF RMSE: {oof_rmse:.4f}")
print(f"OOF R² Score: {oof_r2:.4f}")


print("\nGenerating Average Feature Importance plot...")
feature_importance_df = pd.DataFrame()
feature_importance_df['Feature'] = X.columns
total_importance = np.zeros(X.shape[1])

for model in models:
    total_importance += model.get_feature_importance()

feature_importance_df['Importance'] = total_importance / N_SPLITS
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(30)) # 顯示最重要的 30 個
plt.title('Average Feature Importance (10-Fold CV)', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()


print("\nAveraging test predictions...")
avg_test_predictions = np.mean(test_predictions_list, axis=0)

print("\nGenerating submission file...")
submission_df = pd.DataFrame({
    'id': submission['id'],
    'accident_risk': avg_test_predictions
})

# Clip predictions to the [0, 1] range
submission_df['accident_risk'] = np.clip(submission_df['accident_risk'], 0, 1)

submission_df.to_csv('submission.csv', index=False)
print("Submission file submission.csv has been generated.")

