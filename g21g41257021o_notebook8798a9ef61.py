import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# 1. 載入數據
train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')

# 2. 排除數據洩漏 (Data Leakage)：果斷移除所有 PCIAT 相關特徵 [cite: 5]
# 避免模型陷入「用答案預測答案」的邏輯謬誤 [cite: 5]
pciat_cols = [col for col in train.columns if 'PCIAT' in col]
train = train.drop(columns=pciat_cols)
test = test.drop(columns=[col for col in pciat_cols if col in test.columns])

# 3. 特徵合併邏輯：處理 PAQ 互補性 [cite: 7, 8]
# 基於 PAQ_A (青少年) 與 PAQ_C (兒童) 的高度互補性進行整合 [cite: 7]
def merge_paq_features(df):
    # Kaggle 原始欄位通常為 'PAQ_A-PAQ_A_Total' 與 'PAQ_C-PAQ_C_Total'
    col_a = 'PAQ_A-PAQ_A_Total'
    col_c = 'PAQ_C-PAQ_C_Total'
    
    if col_a in df.columns and col_c in df.columns:
        # 參考課堂所學之特徵合併方法，整合為統一的運動量因子 [cite: 9]
        df['PAQ_Total'] = df[col_a].combine_first(df[col_c])
        return df.drop(columns=[col_a, col_c])
    return df

train = merge_paq_features(train)
test = merge_paq_features(test)

# 4. 數據預處理：基於統計學原理的中位數填補 [cite: 10, 11]
# 針對 BMI (偏度 1.63) 等高度偏態分佈，採用中位數填補以提升抗異常值能力 [cite: 13, 49, 50]
target = 'sii'
train_clean = train.dropna(subset=[target])
y = train_clean[target]
X = train_clean.drop(columns=['id', target])
X_test = test.drop(columns=['id'])

# 僅選擇數值型特徵進行運算
X_numeric = X.select_dtypes(include=[np.number])
X_test_numeric = X_test.select_dtypes(include=[np.number])

# 實作 Exp_5 策略：中位數填補 [cite: 108, 111]
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_numeric)
X_test_imputed = imputer.transform(X_test_numeric)

# 5. 模型建立：隨機森林回歸策略 [cite: 72]
# 捕捉連續程度比強行切分標籤更能滿足 QWK 的平方級懲罰要求 [cite: 71, 73]
# 根據 U 型誤差曲線，選擇 Max Depth = 5 以對抗過擬合 [cite: 119, 132, 136]
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_imputed, y)

# 6. 經驗驅動的門檻優化 (Threshold Tuning) [cite: 87]
# 經消融實驗證實，門檻設定在 0.7 時可達到最佳 QWK 分數 [cite: 89, 103]
raw_preds = model.predict(X_test_imputed)

def apply_optimized_threshold(preds, threshold=0.7):
    # 透過數據驅動的優化，平衡過度預測與保守預測 [cite: 90]
    return np.clip(np.round(preds - (threshold - 0.5)).astype(int), 0, 3)

final_preds = apply_optimized_threshold(raw_preds, threshold=0.7)

# 7. 產生提交檔案
submission = pd.DataFrame({
    'id': test['id'],
    'sii': final_preds
})
submission.to_csv('submission.csv', index=False)

print("第 21 組 Exp_5 預測流程執行完畢。")

