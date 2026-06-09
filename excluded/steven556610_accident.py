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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
train_df


test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_df


sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
sample_submission_df


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Tuple, Dict, Any
from collections import Counter

# 全域變數或作為模型的一部分傳遞，用於儲存訓練時的特徵資訊
# 為了範例的方便性，我們先用一個字典來儲存必要的資訊

TRAINING_INFO={}


def preprocess_features(df: pd.DataFrame, is_training: bool = True, TRAINING_INFO:dict = None) -> pd.DataFrame:
    """
    對特徵集進行 One-Hot Encoding 處理。

    Args:
        df: 待處理的特徵 DataFrame (不含 ID 和 Target)。
        is_training: 是否為訓練階段。
        training_cols: 僅在推論階段 (is_training=False) 傳入，確保欄位與訓練集一致。

    Returns:
        處理後的 DataFrame。
    """
    
    # 執行 One-Hot Encoding，自動處理 object/category/boolean 欄位
    # drop_first=True 避免多重共線性
    df_processed = pd.get_dummies(df, drop_first=True)
    
    if is_training:
        # 訓練階段：儲存處理後的所有欄位名稱
        TRAINING_INFO['feature_columns'] = df_processed.columns.tolist()
        return df_processed, TRAINING_INFO
    else:
        # 推論階段：確保欄位與訓練集完全一致
        training_cols = TRAINING_INFO['feature_columns']
        # 1. 找出訓練集缺少的欄位 (在測試集中是 0)
        missing_cols = set(training_cols) - set(df_processed.columns)
        for c in missing_cols:
            df_processed[c] = 0
            
        # 2. 移除訓練集沒有的欄位 (測試集獨有的新類別)
        # 這些新類別在訓練階段未見，對模型沒有貢獻，應該被移除
        df_processed = df_processed[training_cols]
        
        return df_processed.copy()


def train_xgb_regression_model(df_train: pd.DataFrame, TRAINING_INFO: dict) -> Tuple[xgb.XGBRegressor, pd.DataFrame, pd.Series, dict]:
    """訓練 XGBoost 模型並返回模型和測試結果。"""
    
    print("--- 訓練資料準備 ---")
    # 移除 ID 欄位 (第一欄) 和 Target 欄位 (最後一欄) 得到特徵集 X
    X = df_train.iloc[:, 1:-1]
    Y = df_train.iloc[:, -1]
    
    # 1. 特徵處理 (訓練階段)
    X_processed, TRAINING_INFO = preprocess_features(X, is_training=True, TRAINING_INFO=TRAINING_INFO)
    
    # 2. 資料集切割 (Train/Validation Split for internal training)
    X_train, X_test_val, Y_train, Y_test_val = train_test_split(
        X_processed, Y, test_size=0.2, random_state=42
    )

    print(f"處理後的特徵欄位總數: {X_processed.shape[1]}")
    print(f"訓練集/驗證集分割: {len(X_train)} / {len(X_test_val)} 筆")
    print("-" * 25)

    # 3. 訓練 XGBoost 迴歸模型
    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=100, random_state=42, learning_rate=0.1, verbosity=0
    )
    
    print("開始訓練 XGBoost 模型...")
    model.fit(X_train, Y_train)
    print("模型訓練完成。")
    print("-" * 25)

    # 4. 模型評估 (使用內部驗證集)
    Y_pred_val = model.predict(X_test_val)
    mse = mean_squared_error(Y_test_val, Y_pred_val)
    r2 = r2_score(Y_test_val, Y_pred_val)

    print("--- 模型內部驗證結果 ---")
    print(f"均方誤差 (MSE): {mse:.4f}")
    print(f"決定係數 (R^2 Score): {r2:.4f}")
    
    # 返回訓練好的模型
    return model, X_test_val, Y_test_val, TRAINING_INFO


model, y_pred, y_test, TRAINING_INFO = train_xgb_regression_model(train_df, TRAINING_INFO)


def inference_on_external_test_set(model: xgb.XGBRegressor, df_external_test: pd.DataFrame, TRAINING_INFO: dict) -> pd.DataFrame:
    """
    在獨立測試集上進行推論，並輸出 ID 與預測值的 DataFrame。
    
    Args:
        model: 訓練好的 XGBoost 模型。
        df_external_test: 獨立的測試集 DataFrame (包含 ID, 特徵, 但沒有 Target)。
        
    Returns:
        包含 ID 和預測值的結果 DataFrame。
    """
    
    print("\n--- 獨立測試集推論 ---")
    
    # 1. 儲存 ID 欄位
    id_column_name = df_external_test.columns[0]
    test_ids = df_external_test[id_column_name].copy()
    
    # 2. 提取特徵集 X_test (移除 ID 欄位)
    X_test = df_external_test.iloc[:, 1:]
    
    # 3. 特徵處理 (推論階段) - **關鍵步驟**
    # 必須使用訓練階段的欄位列表來確保一致性
    training_cols = TRAINING_INFO.get('feature_columns')
    if not training_cols:
        raise ValueError("請先運行訓練函式，以獲取訓練集的特徵欄位資訊。")
        
    print(f"使用 {len(training_cols)} 個訓練特徵進行對齊...")
    X_test_processed = preprocess_features(X_test, is_training=False, TRAINING_INFO=TRAINING_INFO)
    
    # 4. 模型推論
    Y_test_pred = model.predict(X_test_processed)
    
    # 5. 創建結果 DataFrame
    result_df = pd.DataFrame({
        id_column_name: test_ids,
        'Predicted_Target': Y_test_pred
    })
    
    print("推論完成，結果 DataFrame 已創建。")
    print(result_df.head())
    
    return result_df


TRAINING_INFO['feature_columns']


#trained_model, _, _ = train_xgb_regression_model(df_train_example)
    
# 2. 獨立測試集推論
results_df = inference_on_external_test_set(model, test_df, TRAINING_INFO)

print("\n--- 最終結果 ---")
print("最終輸出結果 DataFrame (ID 和 Predicted_Target):")
print(results_df.head())


results_df.rename(columns={'Predicted_Target':'accident_risk'},inplace=True)
results_df.to_csv('/kaggle/working/result_df.csv',index=False)




