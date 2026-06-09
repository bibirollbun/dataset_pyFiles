import pandas as pd
import os
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.info()


# Set option to display all columns
pd.set_option('display.max_columns', None)
train.describe(include='all')


# features engineering
import numpy as np
def engineer_features(df):
    """Create engineered features for the dataset
    
    Args:
        df: DataFrame containing the original features
        
    Returns:
        DataFrame with additional engineered features
    """
    # Create a copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # 1. Rhythm-based features - BPM is directly related to rhythm
    df_new['Rhythm_Energy'] = df_new['RhythmScore'] * df_new['Energy']
    df_new['Rhythm_Loudness'] = df_new['RhythmScore'] * df_new['AudioLoudness']
    
    # 2. Duration-related features - longer tracks might have different BPM patterns
    df_new['Duration_Minutes'] = df_new['TrackDurationMs'] / 60000  # Convert to minutes
    df_new['Duration_Energy_Ratio'] = df_new['TrackDurationMs'] / (df_new['Energy'] * 10000 + 1)  # Scaled for numerical stability
    
    # 3. Non-linear transformations - capture more complex relationships
    df_new['RhythmScore_Squared'] = df_new['RhythmScore'] ** 2
    df_new['Energy_Squared'] = df_new['Energy'] ** 2
    df_new['Log_Duration'] = np.log1p(df_new['TrackDurationMs'])  # log(1+x) to handle zeros
    
    # 4. Musical character features - representing the "feel" of the song
    df_new['Acoustic_Instrumental_Ratio'] = df_new['AcousticQuality'] / (df_new['InstrumentalScore'] + 0.01)  # Avoid division by zero
    df_new['Vocal_Energy'] = df_new['VocalContent'] * df_new['Energy']
    
    # 5. Performance and mood interactions
    df_new['Live_Energy'] = df_new['LivePerformanceLikelihood'] * df_new['Energy']
    df_new['Mood_Rhythm'] = df_new['MoodScore'] * df_new['RhythmScore']
    
    # 6. Composite metrics
    df_new['Audio_Intensity'] = (df_new['Energy'] * np.abs(df_new['AudioLoudness'])) / 10  # Scaled for better range
    df_new['Performance_Character'] = (df_new['LivePerformanceLikelihood'] + df_new['MoodScore']) / 2
    
    # 7. Ratios that might represent musical balance
    df_new['Energy_Loudness_Ratio'] = df_new['Energy'] / (np.abs(df_new['AudioLoudness']) + 0.01)
    df_new['Rhythm_Duration_Density'] = df_new['RhythmScore'] / df_new['Duration_Minutes']
    
    return df_new

# Apply feature engineering to both train and test sets
df_train = engineer_features(train)
df_test = engineer_features(test)

# Display the new features
new_features = [col for col in df_train.columns if col not in train.columns]
print(f"Created {len(new_features)} new features:")
print(new_features)

# Check correlation of new features with the target
new_feature_correlation = df_train[new_features + ['BeatsPerMinute']].corr()['BeatsPerMinute'].sort_values(ascending=False)
print("\nNew Feature Correlation with BPM:")
print(new_feature_correlation)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 檢查 df_train 是否存在
if 'df_train' in locals() and isinstance(df_train, pd.DataFrame):
    
    # 想要分析的特徵列表
    features_to_plot = [
        'RhythmScore',
        'AudioLoudness',
        'VocalContent',
        'AcousticQuality',
        'InstrumentalScore',
        'LivePerformanceLikelihood',
        'MoodScore',
        'TrackDurationMs',
        'Energy',
        'moody_energy',
        'rhythmic_intensity',
        'vocal_instrumental_ratio'    ]

    # 設定繪圖風格
    sns.set_style("whitegrid")

    print("--- 開始繪製多特徵分佈圖 ---")

    # 計算行列數 (例如 3x3)
    n_cols = 3
    n_rows = int(np.ceil(len(features_to_plot) / n_cols))

    # 建立子圖
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten()  # 攤平成一維方便迴圈

    # 遍歷特徵繪圖
    for i, feature in enumerate(features_to_plot):
        if feature in df_train.columns:
            sns.histplot(data=df_train, x=feature, ax=axes[i], bins=50, kde=True)
            axes[i].set_title(feature, fontsize=12)
        else:
            axes[i].set_visible(False)

    # 調整佈局
    plt.tight_layout()
    plt.show()

    print("\n--- 所有圖表繪製完成 ---")

else:
    print("錯誤: 名為 'df_train' 的 DataFrame 不存在。請先載入您的資料。")



df_column_list = [col for col in df_train.columns if col != 'id' and col !='BeatsPerMinute']
df_column_list


# XGBoost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna
import warnings
# 忽略 XGBoost 關於設備不匹配的特定警告
warnings.filterwarnings("ignore", message="Falling back to prediction using DMatrix")


# 1 & 2. 準備資料與分割 (與您原本的程式碼相同)
y_variable = 'BeatsPerMinute'
x_variables = df_column_list
X = df_train[x_variables]

y = df_train[y_variable]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. 定義 Optuna 的「目標函數」(Objective Function) ---
# Optuna 會不斷呼叫這個函數，嘗試找到能讓回傳分數最大化的參數組合
def objective(trial):
    # 步驟 1: 定義要搜尋的超參數範圍
    # trial.suggest_* 函式會為每一次的嘗試，從您給定的範圍中建議一個值
    params = {
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
        'device': 'cuda', # 使用新版 XGBoost 的 GPU 參數
        'max_depth': trial.suggest_int('max_depth', 3, 9), # 整數: 從 3 到 9
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), # 浮點數: 建議使用對數均勻分布
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_categorical('gamma', [0, 0.1, 0.5, 1]), # 分類: 從給定的列表中選擇
    }

    # 步驟 2: 建立模型並進行交叉驗證
    model = xgb.XGBRegressor(**params)
    
    # 使用 cross_val_score 進行 5-fold 交叉驗證，並計算 RMSE
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    
    # 回傳這次嘗試的平均分數
    return scores.mean()

# --- 4. 建立並執行 Optuna 的「研究」(Study) ---
print("\n--- 3. 開始執行 Optuna 超參數搜尋 ---")
# direction='maximize' 因為 neg_root_mean_squared_error 越大 (越接近0) 越好
study = optuna.create_study(direction='maximize') 
# n_trials 是嘗試的總次數，相當於 RandomizedSearchCV 的 n_iter
study.optimize(objective, n_trials=50)

print("\n--- 搜尋完成！---")

# --- 5. 檢視最佳結果 ---
print("總共的嘗試次數:", len(study.trials))
print("找到的最佳試驗:")
best_trial = study.best_trial
print(f"  最佳分數 (RMSE): {-best_trial.value:.4f}") # 將分數轉為正的 RMSE
print("  最佳超參數組合:")
best_params = best_trial.params
for key, value in best_params.items():
    print(f"    {key}: {value}")

# --- 6. 使用找到的最佳參數來訓練最終模型，並在測試集上驗證 ---
print("\n--- 4. 使用最佳參數訓練最終模型 ---")
# 將固定參數與最佳參數合併
final_params_xgb = {
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1,
    'device': 'cuda'
}
final_params_xgb.update(best_params)

final_model_xgb = xgb.XGBRegressor(**final_params_xgb)

final_model_xgb.fit(X_train, y_train)
y_pred = final_model_xgb.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"在獨立測試集上的最終 RMSE: {final_rmse:.4f}")

# --- 7. 檢視最佳模型的特徵重要性 ---
print("\n--- 5. 最佳模型的特徵重要性分析 ---")
importances = final_model_xgb.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x_variables, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# 計算殘差
residuals = y_test - y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted BPM")
plt.ylabel("Residuals (y - y_pred)")
plt.title("Residuals vs Predictions")
plt.show()



#lightgbm
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform

# 如果您尚未安裝 lightgbm，請先執行: pip install lightgbm

# 1 & 2. 準備資料與分割 (與您原本的程式碼相同)
y_variable = 'BeatsPerMinute'
x_variables = df_column_list
X = df_train[x_variables]

y = df_train[y_variable]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. 定義 Optuna 的「目標函數」(Objective Function) for LightGBM ---
def objective(trial):
    # 步驟 1: 定義要搜尋的超參數範圍
    params = {
        'objective': 'rmse',
        'random_state': 42,
        'n_jobs': 1, # 在 Optuna 中，模型內部通常設為單線程
        'device': 'gpu',
        'verbose': -1,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_categorical('max_depth', [-1, 10, 20, 30]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
    }

    # 步驟 2: 建立模型並進行交叉驗證
    model = lgb.LGBMRegressor(**params)
    
    # 使用 cross_val_score 進行 5-fold 交叉驗證
    # n_jobs 設為 -1，讓交叉驗證的各個 Fold 平行運算
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    
    # 回傳這次嘗試的平均分數
    return scores.mean()

# --- 4. 建立並執行 Optuna 的「研究」(Study) ---
print("\n--- 3. 開始執行 Optuna 超參數搜尋 (for LightGBM) ---")
# direction='maximize' 因為 neg_root_mean_squared_error 越大 (越接近0) 越好
# 【修改點】在 optimize 之前，加入下面這兩行
optuna.logging.set_verbosity(optuna.logging.WARNING)
print("Optuna 日誌已設為靜默模式，搜尋過程中將不會顯示進度...")
study = optuna.create_study(direction='maximize') 
# n_trials 是嘗試的總次數
study.optimize(objective, n_trials=50)

print("\n--- 搜尋完成！---")

# --- 5. 檢視最佳結果 ---
print("總共的嘗試次數:", len(study.trials))
print("找到的最佳試驗:")
best_trial = study.best_trial
print(f"  最佳分數 (RMSE): {-best_trial.value:.4f}")
print("  最佳超參數組合:")
best_params = best_trial.params
for key, value in best_params.items():
    print(f"    {key}: {value}")

# --- 6. 使用找到的最佳參數來訓練最終模型 ---
print("\n--- 4. 使用最佳參數訓練最終 LightGBM 模型 ---")
final_params_lgb = {
    'objective': 'rmse',
    'random_state': 42,
    'n_jobs': -1, # 最終訓練時可以用回全部核心
    'device': 'gpu'
}
final_params_lgb.update(best_params)

final_model_lgb = lgb.LGBMRegressor(**final_params_lgb)

final_model_lgb.fit(X_train, y_train)
y_pred = final_model_lgb.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"在獨立測試集上的最終 RMSE (LightGBM): {final_rmse:.4f}")


# 使用權重大法

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

# 工具函式：計算 RMSE
def rmse_metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ==== 0) 準備資料與最終參數 ====
X = df_train[x_variables]
y = df_train[y_variable]

xgb_params = final_params_xgb
lgb_params = final_params_lgb

# ==== 1) 產生 OOF 預測（分箱 + 分層）====
bins = pd.qcut(y, q=10, duplicates='drop')
y_strat = pd.factorize(bins)[0]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

for tr_idx, va_idx in cv.split(X, y_strat):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

    xgb_model = xgb.XGBRegressor(**xgb_params)
    lgb_model = lgb.LGBMRegressor(**lgb_params)

    xgb_model.fit(Xtr, ytr)
    lgb_model.fit(Xtr, ytr)

    oof_xgb[va_idx] = xgb_model.predict(Xva)
    oof_lgb[va_idx] = lgb_model.predict(Xva)

# 各自 OOF RMSE
rmse_xgb = rmse_metric(y, oof_xgb)
rmse_lgb = rmse_metric(y, oof_lgb)
print(f"OOF RMSE  XGB: {rmse_xgb:.4f} | LGBM: {rmse_lgb:.4f}")

# ==== 2) 用 OOF 找最佳權重 w ====
grid = np.linspace(0, 1, 101)
rmse_list = []
best = (1e9, None)

for w in grid:
    blend = w * oof_xgb + (1 - w) * oof_lgb
    rmse_val = rmse_metric(y, blend)
    rmse_list.append(rmse_val)
    if rmse_val < best[0]:
        best = (rmse_val, w)

best_rmse, w = best
print(f"最佳 OOF RMSE = {best_rmse:.4f} | 權重：XGB={w:.2f}, LGBM={1-w:.2f}")

# ==== 2.1 畫 RMSE vs 權重圖 ====
plt.figure(figsize=(6,4))
plt.plot(grid, rmse_list, marker="o", markersize=3, linewidth=1)
plt.axvline(w, color="red", linestyle="--", label=f"最佳 w={w:.2f}")
plt.xlabel("XGB 權重 w")
plt.ylabel("OOF RMSE")
plt.title("OOF RMSE vs 權重 (XGB vs LGBM)")
plt.legend()
plt.show()

# ==== 3) 用全訓練集重訓 → 測試集加權 ====
xgb_full = xgb.XGBRegressor(**xgb_params)
lgb_full = lgb.LGBMRegressor(**lgb_params)

xgb_full.fit(X, y)
lgb_full.fit(X, y)

pred_xgb = xgb_full.predict(X_test)
pred_lgb = lgb_full.predict(X_test)

y_pred_blend = w * pred_xgb + (1 - w) * pred_lgb
rmse_test = rmse_metric(y_test, y_pred_blend)
print(f"Test RMSE（最佳權重加權）: {rmse_test:.4f}")



# 使用權重串連兩個model

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 準備資料與分割
y_variable = 'BeatsPerMinute'
x_variables = new_features
X = df_train[x_variables]
y = df_train[y_variable]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 定義 XGB 和 LGBM 模型（使用你已經找到的最佳參數）
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    gamma=1,
    n_estimators=276,
    learning_rate=0.012283457973465113,
    max_depth=4,
    subsample=0.7442210789402326,
    colsample_bytree=0.8435808646349576,
    random_state=42,
    n_jobs=-1,
    device='cuda'
)

lgb_model = lgb.LGBMRegressor(
    bagging_fraction=0.8334921333853907,
    feature_fraction=0.7180122378300953,
    max_depth=10,
    objective='rmse',
    n_estimators=267,
    learning_rate=0.014047437609905565,
    num_leaves=27,
    random_state=42,
    n_jobs=-1
)

# 3. 訓練兩個基模型
print("訓練 XGBoost...")
xgb_model.fit(X_train, y_train)

print("訓練 LightGBM...")
lgb_model.fit(X_train, y_train)

# 4. 各自預測
pred_xgb = xgb_model.predict(X_test)

pred_lgb = lgb_model.predict(X_test)

# 5. 加權融合（使用你找到的最佳權重）
w_xgb = 0.1
w_lgb = 0.9
y_pred_blend = w_xgb * pred_xgb + w_lgb * pred_lgb

# 6. 評估結果
rmse_blend = np.sqrt(mean_squared_error(y_test, y_pred_blend))
print(f"在獨立測試集上的最終 RMSE (加權融合 XGB={w_xgb:.2f}, LGBM={w_lgb:.2f}): {rmse_blend:.4f}")



import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold   # ✅ 修正：改用 KFold（回歸用）
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error


# 1) 準備資料與分割
y_variable = 'BeatsPerMinute'
x_variables = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy', 'moody_energy', 'rhythmic_intensity',
    'vocal_instrumental_ratio'
]
X = df_train[x_variables]
y = df_train[y_variable]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n--- 建立 Stacking 模型 ---")

# 2) 基礎模型（Level-0）
level0_models = [
    ('xgb', xgb.XGBRegressor(
        objective='reg:squarederror',
        gamma=0.1,
        n_estimators=299,
        learning_rate=0.010909741131991784,
        max_depth=3,
        subsample=0.6636225780972534,
        colsample_bytree=0.9059766047903175,
        random_state=42,
        n_jobs=-1,
        device='cuda'  # 走 GPU
        # tree_method='gpu_hist'  # 舊寫法；有需要可開
    )),
    ('lgbm', lgb.LGBMRegressor(
        bagging_fraction=0.980191803143872,
        feature_fraction=0.9533993518912539,
        max_depth=10,
        objective='regression',   # ✅ 修正：較新版本建議用 'regression'
        n_estimators=277,
        learning_rate=0.011917790099469323,
        num_leaves=20,
        random_state=42,
        n_jobs=-1
    ))
]

# 3) 元模型（Level-1）
meta_model = RidgeCV()  # 一次定義即可  ✅ 修正：移除重複定義

# 4) KFold（回歸任務用 KFold，避免 StratifiedKFold 的連續目標錯誤）  ✅ 修正
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 5) 建立 StackingRegressor
stacking_model = StackingRegressor(
    estimators=level0_models,
    final_estimator=meta_model,
    cv=cv,               # ✅ 修正：明確傳入 KFold 物件
    n_jobs=1,            # ✅ 修正：為避免 GPU 模型並行爭用，這裡用單工
    passthrough=True     # ✅ 讓 meta 看到 base 預測 + 原始特徵
)

# 6) 訓練
print("開始訓練 Stacking 模型（這會花一些時間）...")
stacking_model.fit(X_train, y_train)
print("模型訓練完成！")

# 7) 評估
y_pred_stack = stacking_model.predict(X_test)
final_rmse_stack = float(np.sqrt(mean_squared_error(y_test, y_pred_stack)))
print(f"在獨立測試集上的最終 RMSE (Stacking): {final_rmse_stack:.4f}")




y_pred = stacking_model.predict(df_test.drop('id', axis=1))
df_test['BeatsPerMinute'] = y_pred
submission = df_test[['id', 'BeatsPerMinute']]
submission.to_csv("submission_stacking.csv", index=False)


