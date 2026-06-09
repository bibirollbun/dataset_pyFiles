import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc

import xgboost as xgb
import optuna
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# --- 資料載入與合併 ---
print("--- 1. 載入並合併資料 ---")

original_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# 載入額外的 bank-full.csv 資料集
bank_full_df = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')

print(f"原始訓練集維度: {original_train.shape}")
print(f"額外資料集維度: {bank_full_df.shape}")
print(f"測試集維度: {test_df.shape}")

# 合併 original_train 和 bank_full_df 作為完整的訓練資料
if 'id' in original_train.columns:
    original_train = original_train.drop('id', axis=1)
if 'id' in test_df.columns:
    test_df = test_df.drop('id', axis=1)

bank_full_df['y'] = bank_full_df['y'].map({'no': 0, 'yes': 1})
train_df = pd.concat([original_train, bank_full_df], ignore_index=True)
print(f"合併後完整訓練集維度: {train_df.shape}")


# === 特徵工程 - 建立合成特徵 ===
def create_synthetic_features(df):
    """
    建立所有需要的合成特徵
    """
    df_new = df.copy()
    
    # 1. 建立月份和時間相關特徵
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df_new['month_num'] = df_new['month'].map(month_mapping)
    df_new['month_sin'] = np.sin(2 * np.pi * df_new['month_num'] / 12)
    df_new['month_cos'] = np.cos(2 * np.pi * df_new['month_num'] / 12)
    df_new['quarter'] = df_new['month_num'].apply(lambda x: (x - 1) // 3 + 1)
    df_new['day_sin'] = np.sin(2 * np.pi * df_new['day'] / 31)
    df_new['day_cos'] = np.cos(2 * np.pi * df_new['day'] / 31)

    # 2. 將 education 作為順序變項處理
    education_order = {'primary': 1, 'secondary': 2, 'tertiary': 3}
    df_new['education_num'] = df_new['education'].map(education_order)
    education_median = df_new['education_num'].median()
    df_new['education_num'].fillna(education_median, inplace=True)
    
    # 3. 建立客戶價值相關特徵
    df_new['balance_negative'] = (df_new['balance'] <= 0).astype(int)
    df_new['balance_log'] = np.log1p(df_new['balance'].abs())
    df_new['age_balance_interaction'] = df_new['age'] * df_new['balance_log']

    # 4. 建立通話和聯繫相關特徵
    df_new['duration_log'] = np.log1p(df_new['duration']) # duration 僅用於建立特徵
    df_new['total_contact'] = df_new['campaign'] + df_new['previous']
    df_new['is_new_customer'] = (df_new['pdays'] == -1).astype(int)
    pdays_cleaned = df_new['pdays'].replace(-1, 0)
    df_new['contact_intensity'] = np.where(
        df_new['is_new_customer'] == 1, 
        0, 
        df_new['total_contact'] / (pdays_cleaned + 1)
    )

    # 5. 建立綜合評分特徵
    # 定義輔助函數 f1，用於計算「資訊未知分數」
    def f1(x):
        if x['education']=='unknown' and x['contact'] =='unknown' and x['poutcome']=='unknown':
            return 21
        if x['education']=='unknown' and x['contact'] =='unknown'\
        or x['education']=='unknown' and x['poutcome']=='unknown'\
        or x['contact']  =='unknown' and x['poutcome']=='unknown':
            return 7
        if x['education']=='unknown' or x['contact']=='unknown' or x['poutcome']=='unknown':
            return 3
        return 0
    
    # 定義輔助函數 f2，用於計算「財務健康分數」
    def f2(x):
        if x['default']=='no' and x['housing']=='no' and x['loan']=='no':
            return 21
        if x['default']=='no' and x['housing']=='no'\
        or x['default']=='no' and x['loan']=='no'\
        or x['housing']=='no' and x['loan']=='no':
            return 7
        if x['default']=='no' or x['housing']=='no' or x['loan']=='no':
            return 3
        return 0

    df_new['unknowns_count'] = df_new.apply(f1, axis=1)
    df_new['financial_health_score'] = df_new.apply(f2, axis=1)
    
    return df_new

# 對訓練集和測試集應用特徵工程
print("對訓練集和測試集應用特徵工程...")
train_fe = create_synthetic_features(train_df)
test_fe = create_synthetic_features(test_df)

print("✅ 特徵工程完成!")


print("--- 準備用於模型的最終資料 ---")
# 處理目標變數 y
y = train_fe['y']

# 從訓練集和測試集中移除目標變數和不再需要的原始特徵
X = train_fe.drop(columns=['y', 'duration','month'])
test_processed = test_fe.drop(columns=['duration', 'month'])

# 定義需要獨熱編碼的類別特徵
categorical_cols_to_encode = [
    'job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome'
]

# 暫時合併訓練集和測試集以確保所有類別都被學習到
combined_df = pd.concat([X, test_processed], ignore_index=True)
dummies = pd.get_dummies(combined_df, columns=categorical_cols_to_encode, drop_first=True)

# 將獨熱編碼後的資料分割回訓練集和測試集
X_processed = dummies.iloc[:len(X)]
test_processed = dummies.iloc[len(X):]

# 移除原始的 education 欄位
X_final = X_processed.drop(columns=['education'], errors='ignore')
test_final = test_processed.drop(columns=['education'], errors='ignore')

# 再次確保欄位對齊
train_cols = X_final.columns
test_cols = test_final.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    test_final[c] = 0
test_final = test_final[train_cols]


print(f"最終訓練特徵維度: {X_final.shape}")
print(f"最終測試特徵維度: {test_final.shape}")
print("✅ 資料預處理完成!")


def objective(trial, data, target):
    calculated_spw = (target == 0).sum() / (target == 1).sum()
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'n_estimators': trial.suggest_int('n_estimators', 800, 4000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1, calculated_spw]),
        'device' : 'cuda',
        'tree_method' : 'hist',
        'random_state': 42
    }
    
    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(data, target):
        X_train_fold, X_val_fold = data.iloc[train_idx], data.iloc[val_idx]
        y_train_fold, y_val_fold = target.iloc[train_idx], target.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  early_stopping_rounds=300,
                  verbose=False)
                  
        preds = model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, preds)
        scores.append(auc)
        
        del model, X_train_fold, X_val_fold, y_train_fold, y_val_fold
        gc.collect()

    return np.mean(scores)

print("--- 開始 Optuna 超參數優化 ---")
N_TRIALS = 50

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X_final, y), n_trials=N_TRIALS)

print(f"優化完成！共進行了 {len(study.trials)} 次嘗試。")
print(f"最佳 ROC AUC 分數 (CV 平均): {study.best_value:.5f}")
print("最佳參數:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")


print("--- 開始訓練最終模型 ---")
best_params = study.best_params

best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'auc'
best_params['tree_method'] = 'hist'
best_params['random_state'] = 42

X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_final, y, test_size=50000, random_state=42, stratify=y
)

print(f"最終訓練集維度: {X_train_final.shape}")
print(f"最終驗證集維度: {X_val_final.shape}")

final_model = xgb.XGBClassifier(**best_params)

print("正在訓練...")
final_model.fit(X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                early_stopping_rounds=300,
                verbose=500)

print("✅ 最終模型訓練完成!")


y_pred = final_model.predict(X_val_final)
y_pred_proba = final_model.predict_proba(X_val_final)[:, 1]


print(classification_report(y_val_final, y_pred, target_names=['未訂閱 (0)', '已訂閱 (1)']))


print("--- 在測試集上進行預測 ---")
y_pred_proba_final = final_model.predict_proba(test_final)[:, 1]

submission_df = pd.DataFrame({
    'id': submission['id'],
    'y': y_pred_proba_final
})

submission_df.to_csv('submission.csv', index=False)

print("提交檔案 'submission.csv' 已成功儲存！")
print(submission_df.head())


feature_importances = pd.DataFrame({
    'feature': X_final.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False).head(30)

plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('最終模型特徵重要性 (前 30)')
plt.xlabel('重要性')
plt.ylabel('特徵')
plt.tight_layout()
plt.show()

