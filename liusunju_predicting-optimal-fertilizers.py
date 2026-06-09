# 這個 Python 3 環境預裝了許多實用的分析函式庫
# 它是以 kaggle/python Docker 映像檔為基礎所定義：[https://github.com/kaggle/docker-python](https://github.com/kaggle/docker-python)
# 例如，以下是一些載入的實用套件

import numpy as np # 線性代數
import pandas as pd # 資料處理、CSV 檔案輸入/輸出 (例如 pd.read_csv)

# 輸入資料檔案位於唯讀的 "../input/" 目錄下
# 例如，執行此處 (點擊 "run" 或按下 Shift+Enter) 將會列出輸入目錄下的所有檔案

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 你可以將最多 20GB 的資料寫入目前目錄 (/kaggle/working/)，當你使用 "Save & Run All" 建立版本時，這些資料會被儲存下來
# 你也可以將暫存檔案寫入 /kaggle/temp/，但這些檔案在目前工作階段結束後不會被儲存


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
print("Shape of Train Data", train_df.shape)
print("Shape of Test Data", test_df.shape)


display(train_df.head(10))


train_df.info()


train_df.nunique()


train_df.describe().transpose()


plt.figure(figsize=(12, 4))
soil_counts = train_df['Soil Type'].value_counts()

# Plot barplot
ax = sns.barplot(x=soil_counts.index, y=soil_counts.values, palette="viridis")

plt.title("Distribution of Soil Types ", fontsize=15)
plt.xlabel("Soil Type")
plt.ylabel("Count")
plt.xticks(rotation=0)

# Add percentage labels on top of each bar
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',  # Count + percentage
            ha='center', va='center', fontsize=10)

plt.show()


plt.figure(figsize=(14, 4))
crop_counts = train_df['Crop Type'].value_counts()

# Plot barplot
ax = sns.barplot(x=crop_counts.index, y=crop_counts.values, palette="magma")
plt.title("Distribution of Crop Types", fontsize=15)
plt.xlabel("Crop Type")
plt.ylabel("Count")
plt.xticks(rotation=0)

# 新增百分比標籤
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',  # Count + percentage
            ha='center', va='center', fontsize=10)

plt.show()


plt.figure(figsize=(14, 4))  # 更廣泛的類別
fert_counts = train_df['Fertilizer Name'].value_counts()

# 繪製長條圖
ax = sns.barplot(x=fert_counts.index, y=fert_counts.values, palette="plasma")
plt.title("Distribution of Fertilizer Names", fontsize=15)
plt.xlabel("Fertilizer Name")
plt.ylabel("Count")
plt.xticks(rotation=0)  # Rotate 90° if labels overlap

# 新增百分比標籤
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
            f'{height}\n({height/total:.1%})',
            ha='center', va='center', fontsize=9)  # Smaller font for tight spaces

plt.tight_layout()  # 防止標籤被切斷
plt.show()


plt.figure(figsize=(16, 4)) # 你目前設定的尺寸

# 繪製長條圖,將圖形繪製到當前 figure 的 axes 上
pd.crosstab(train_df['Soil Type'], train_df['Fertilizer Name']).plot(kind='bar', stacked=False, colormap='viridis', ax=plt.gca()) 

plt.title('Fertilizer Preference by Soil Type', fontsize=16)
plt.xlabel('Soil Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0) # 旋轉 x 軸標籤
plt.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1)) # 將圖例放在外部

plt.tight_layout() # 自動調整佈局以防止重疊或截斷
plt.show() # 顯示圖形


plt.figure(figsize=(16, 4))
pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name']).plot(kind='bar', stacked=False, colormap='viridis', ax=plt.gca())
plt.title('Fertilizer Preference by Crop Type', fontsize=16)
plt.xlabel('Crop Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# 比較每塊土壤的平均 N-P-K 含量
train_df.groupby('Soil Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean().plot(kind='bar', figsize=(14, 6))
plt.title('Average Nutrient Levels by Soil Type')
plt.xticks(rotation=0)
plt.show()


# 比較每種作物的平均 N-P-K 含量
train_df.groupby('Crop Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean().plot(kind='bar', figsize=(14, 4))
plt.title('Average Nutrient Levels by Crop Type')
plt.xticks(rotation=0)
plt.show()


# 熱圖：作物與肥料（計數
cross_tab = pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name'])
plt.figure(figsize=(16, 8))
sns.heatmap(cross_tab, cmap='YlGnBu', annot=True, fmt='d')
plt.title('Crop-Fertilizer Frequency', fontsize=16)
plt.xticks(rotation=0)
plt.show()


numerical_df = train_df.select_dtypes(include=['int64', 'float64'])


numerical_df.columns


from scipy import stats
from itertools import combinations

# 取得所有數值對欄位
column_pairs = combinations(['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'], 2)

# 設定風格
sns.set(style="whitegrid")

# 循環遍歷每一對並繪製
for col1, col2 in column_pairs:
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # 帶迴歸線的散點圖
    sns.regplot(x=col1, y=col2, data=numerical_df, scatter_kws={'alpha':0.6})
    
    # 計算統計數據
    corr_coef, p_value = stats.pearsonr(numerical_df[col1].dropna(), numerical_df[col2].dropna())
    slope, intercept, _, _, _ = stats.linregress(numerical_df[col1].dropna(), numerical_df[col2].dropna())
    
    # 新增統計資料到繪圖中
    stats_text = (f"Pearson r = {corr_coef:.2f}\n"
                  f"p-value = {p_value:.4f}\n"
                  f"Regression: y = {slope:.2f}x + {intercept:.2f}")
    
    plt.gcf().text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # 標題和標籤
    plt.title(f'{col1} vs {col2}', fontsize=14)
    plt.xlabel(col1, fontsize=12)
    plt.ylabel(col2, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # 自動解釋
    abs_r = abs(corr_coef)
    
    # 解讀皮爾遜 r
    if abs_r >= 0.8:
        strength = "非常強烈"
    elif abs_r >= 0.6:
        strength = "強烈"
    elif abs_r >= 0.4:
        strength = "中等"
    elif abs_r >= 0.2:
        strength = "微弱"
    else:
        strength = "非常微弱或沒有"
    
    direction = "正向" if corr_coef > 0 else "負向" if corr_coef < 0 else "無"
    
    # 解釋 p-value
    if p_value < 0.001:
        sig_text = "高度統計顯著 (p < 0.001)"
    elif p_value < 0.05:
        sig_text = "統計顯著 (p < 0.05)"
    else:
        sig_text = "統計不顯著 (p ≥ 0.05)"
  
    # 列印中文解釋
    print(f"\n{col1} 與 {col2} 的解說:")
    print(f"- 存在 {strength} 的 {direction} 線性關係")
    print(f"- 相關性 {sig_text}\n")



    print("-" * 60)  # 分隔線


corr = abs(numerical_df.corr()) # 相關矩陣
lower_triangle = np.tril(corr, k = -1)  # 僅選擇相關矩陣的下三角
mask = lower_triangle == 0  # 遮蓋以下熱圖中的上三角形

plt.figure(figsize = (15,8))  # 設定圖形尺寸
sns.set_style(style = 'white')  # 將其設為白色，這樣我們就看不到網格線
sns.heatmap(lower_triangle, center=0.5, cmap= 'Blues', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # 大熱圖
plt.xticks(rotation = 0)   # 美學目的
plt.yticks(rotation = 20)   # 美學目的
plt.show()


from scipy.stats import skew  # 用於計算偏度

# 設定子圖
n_cols = 3  # 網格中的列數
n_rows = (len(numerical_df.columns) // n_cols) + 1

# 建立帶有子圖的圖形
plt.figure(figsize=(15, 5 * n_rows))  # 根據需要調整大小

# 循環遍歷數值欄位並繪製 KDE + 偏度
for i, column in enumerate(numerical_df.columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.kdeplot(data=numerical_df, x=column, fill=True)
    
    # Calculate skewness
    skewness = skew(numerical_df[column].dropna())  # 如果需要，處理 NaN
    skew_text = f'Skewness: {skewness:.2f}'
    
    # Add skewness as text in the plot
    plt.text(0.05, 0.9, skew_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'KDE of {column}')
    plt.xlabel(column)

plt.tight_layout()
plt.show()


# 繪製箱線圖
plt.figure(figsize=(15, 8))
for i, feature in enumerate(numerical_df.columns, 1):
    plt.subplot(2, 4, i)  # Adjust subplot grid as needed
    sns.boxplot(data=train_df, y=feature, color='skyblue')
    plt.title(f'Box Plot of {feature}')
    plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# 確保您已經載入了 train_df 和 test_df
# 如果還沒載入，請執行以下程式碼

# 1. 獨熱編碼 'Soil Type' 和 'Crop Type'
# 合併訓練集和測試集進行獨熱編碼，以確保所有可能的類別都被考慮到
combined_df = pd.concat([train_df.drop('Fertilizer Name', axis=1), test_df], ignore_index=True)

categorical_features = ['Soil Type', 'Crop Type']
combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_features, drop_first=False) # 這裡我們不丟棄第一個，保留所有類別信息

# 將合併後的 DataFrame 分回訓練集和測試集
X_train_processed = combined_df_encoded.iloc[:len(train_df)].copy()
X_test_processed = combined_df_encoded.iloc[len(train_df):].copy()

# 確保移除 'id' 欄位，因為它不是用於訓練的特徵
if 'id' in X_train_processed.columns:
    X_train_processed = X_train_processed.drop('id', axis=1)
if 'id' in X_test_processed.columns:
    X_test_processed = X_test_processed.drop('id', axis=1)


# 2. 標準化數值特徵
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

scaler = StandardScaler()

# 在訓練集上 fit 標準化器，並轉換訓練集
X_train_processed[numerical_features] = scaler.fit_transform(X_train_processed[numerical_features])

# 使用訓練集的統計資訊轉換測試集
X_test_processed[numerical_features] = scaler.transform(X_test_processed[numerical_features])

# 3. 使用 LabelEncoder 將 'Fertilizer Name' 轉換為數值標籤 (僅針對訓練集)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_df['Fertilizer Name'])

print("資料前處理完成！")
print("處理後的訓練特徵 (X_train_processed) 形狀:", X_train_processed.shape)
print("處理後的測試特徵 (X_test_processed) 形狀:", X_test_processed.shape)
print("編碼後的訓練目標 (y_train_encoded) 形狀:", y_train_encoded.shape)
print("LabelEncoder 的類別:", label_encoder.classes_)


# 找出最佳模型 (最終版：包含LGBM, XGBoost, CatBoost)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import xgboost as xgb # 引入 XGBoost
from catboost import CatBoostClassifier # 引入 CatBoost
from sklearn.metrics import make_scorer, f1_score

# 定義您想要比較的模型
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(random_state=42, multi_class='auto', solver='liblinear'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gaussian Naive Bayes': GaussianNB(),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), # 新增 XGBoost
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0) # 新增 CatBoost (verbose=0 關閉迭代訊息)
}

# 定義評估指標
scorer = make_scorer(f1_score, average='weighted')

# 執行交叉驗證並比較模型性能
results = {}
for model_name, model in models.items():
    print(f"正在評估 {model_name}...")
    # 使用 3 折交叉驗證 (cv=3) 和 n_jobs=-1 加速
    scores = cross_val_score(model, X_train_processed, y_train_encoded, cv=3, scoring=scorer, n_jobs=-1)
    results[model_name] = scores.mean()
    print(f"{model_name} 的平均 F1-分數: {results[model_name]:.4f}")

# 找出表現最好的模型
best_model_name = max(results, key=results.get)
print(f"\n表現最佳的模型是: {best_model_name}，平均 F1-分數為: {results[best_model_name]:.4f}")


# 假設您的最佳模型支援 predict_proba
# 例如：
best_model = models[best_model_name] # 假設這是您的最佳模型
best_model.fit(X_train_processed, y_train_encoded)

# 獲取每個測試樣本在每個肥料類別上的預測機率
# 請確保您的模型支援 predict_proba
try:
    predictions_proba = best_model.predict_proba(X_test_processed)

    # 獲取每個樣本機率最高的 3 個類別的索引
    # axis=1 表示對每一列 (每個樣本) 進行排序
    top_3_indices = np.argsort(predictions_proba, axis=1)[:, -3:][:, ::-1] # 獲取最後三列（機率最高的三個），並反轉順序使其從最高到最低

    # 使用 label_encoder 將索引轉換回原始肥料名稱
    predicted_top_3_names = label_encoder.inverse_transform(top_3_indices.flatten()).reshape(top_3_indices.shape)

    # 將這三個名稱以空格連接
    predicted_names_for_submission_top3 = [" ".join(row) for row in predicted_top_3_names]

    # 顯示前幾個用於提交的格式
    print("\n基於機率排名的前 3 個預測肥料名稱 (用於提交，前 10 個):")
    print(predicted_names_for_submission_top3[:10])

except AttributeError:
    print("\n您選擇的模型不支援 predict_proba。無法使用此方法預測前 3 個機率最高的類別。")
    print("請確認您選擇的模型是否支援 predict_proba，或者考慮使用支援機率輸出的模型 (例如 RandomForestClassifier, LogisticRegression, SVC(probability=True))。")


# CatBoost 超參數調整
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, f1_score

# 選擇要調整的模型
# verbose=0 可以在搜索過程中保持輸出乾淨
model_to_tune = CatBoostClassifier(random_state=42, verbose=0)

# 定義要搜索的超參數範圍
# CatBoost 的參數名稱與 XGBoost 有些不同
param_distributions_cat = {
    'iterations': [100, 200, 300, 500], # 等同於 n_estimators
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10], # 等同於 max_depth
    'l2_leaf_reg': [1, 3, 5, 7], # L2 正規化懲罰項
    'border_count': [32, 64, 128] # 數值特徵分箱的數量
}

# 定義評估指標
scorer = make_scorer(f1_score, average='weighted')

# 設定 RandomizedSearchCV
random_search_cat = RandomizedSearchCV(
    estimator=model_to_tune,
    param_distributions=param_distributions_cat,
    n_iter=10,
    scoring=scorer,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2
)

print("開始進行 CatBoost 超參數調整...")
random_search_cat.fit(X_train_processed, y_train_encoded)

print(f"\nCatBoost 最佳參數組合: {random_search_cat.best_params_}")
print(f"CatBoost 最佳調整後分數: {random_search_cat.best_score_:.4f}")

# 最佳模型
final_model = random_search_cat.best_estimator_

# 獲取每個測試樣本在每個肥料類別上的預測機率
# 請確保您的模型支援 predict_proba
try:
    predictions_proba = final_model.predict_proba(X_test_processed)

    # 獲取每個樣本機率最高的 3 個類別的索引
    # axis=1 表示對每一列 (每個樣本) 進行排序
    top_3_indices = np.argsort(predictions_proba, axis=1)[:, -3:][:, ::-1] # 獲取最後三列（機率最高的三個），並反轉順序使其從最高到最低

    # 使用 label_encoder 將索引轉換回原始肥料名稱
    predicted_top_3_names = label_encoder.inverse_transform(top_3_indices.flatten()).reshape(top_3_indices.shape)

    # 將這三個名稱以空格連接
    predicted_names_for_submission_top3 = [" ".join(row) for row in predicted_top_3_names]

    # 顯示前幾個用於提交的格式
    print("\n基於機率排名的前 3 個預測肥料名稱 (用於提交，前 10 個):")
    print(predicted_names_for_submission_top3[:10])

except AttributeError:
    print("\n您選擇的模型不支援 predict_proba。無法使用此方法預測前 3 個機率最高的類別。")
    print("請確認您選擇的模型是否支援 predict_proba，或者考慮使用支援機率輸出的模型 (例如 RandomForestClassifier, LogisticRegression, SVC(probability=True))。")


# 超參數調整 (以 LightGBM 為例)
from sklearn.model_selection import RandomizedSearchCV

# 定義要搜索的超參數範圍
# 這只是一個範例，您可以根據需要調整範圍
param_distributions = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.05, 0.1, 0.2],
    'num_leaves': [31, 50, 70],
    'max_depth': [-1, 10, 20],
    'reg_alpha': [0.1, 0.5], # L1 正規化
    'reg_lambda': [0.1, 0.5] # L2 正規化
}

# 選擇您的最佳模型
# 這裡我們直接使用 LightGBM，因為它通常是最佳選擇
best_model_for_tuning = lgb.LGBMClassifier(random_state=42, n_jobs=-1)

# 設定 RandomizedSearchCV
# n_iter=10 表示從參數空間中隨機組合10次進行嘗試
# cv=3 使用3折交叉驗證
# n_jobs=-1 使用所有CPU核心
random_search = RandomizedSearchCV(
    estimator=best_model_for_tuning,
    param_distributions=param_distributions,
    n_iter=10,
    scoring=scorer,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=2 # 顯示搜索過程
)

print("開始進行超參數調整...")
# 在完整訓練資料上進行搜索
random_search.fit(X_train_processed, y_train_encoded)

print(f"\n最佳參數組合: {random_search.best_params_}")
print(f"最佳調整後分數: {random_search.best_score_:.4f}")

# 使用找到的最佳參數來建立最終模型
final_model = random_search.best_estimator_


# 獲取每個測試樣本在每個肥料類別上的預測機率
# 請確保您的模型支援 predict_proba
try:
    predictions_proba = final_model.predict_proba(X_test_processed)

    # 獲取每個樣本機率最高的 3 個類別的索引
    # axis=1 表示對每一列 (每個樣本) 進行排序
    top_3_indices = np.argsort(predictions_proba, axis=1)[:, -3:][:, ::-1] # 獲取最後三列（機率最高的三個），並反轉順序使其從最高到最低

    # 使用 label_encoder 將索引轉換回原始肥料名稱
    predicted_top_3_names = label_encoder.inverse_transform(top_3_indices.flatten()).reshape(top_3_indices.shape)

    # 將這三個名稱以空格連接
    predicted_names_for_submission_top3 = [" ".join(row) for row in predicted_top_3_names]

    # 顯示前幾個用於提交的格式
    print("\n基於機率排名的前 3 個預測肥料名稱 (用於提交，前 10 個):")
    print(predicted_names_for_submission_top3[:10])

except AttributeError:
    print("\n您選擇的模型不支援 predict_proba。無法使用此方法預測前 3 個機率最高的類別。")
    print("請確認您選擇的模型是否支援 predict_proba，或者考慮使用支援機率輸出的模型 (例如 RandomForestClassifier, LogisticRegression, SVC(probability=True))。")


# XGBoost 超參數調整
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import make_scorer, f1_score

# 選擇要調整的模型
# 確保使用 XGBoost 的特定參數
# use_label_encoder=False 和 eval_metric='mlogloss' 是為了避免警告和設定評估指標
model_to_tune = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1
)

# 定義要搜索的超參數範圍
# 這是 XGBoost 常用的參數，您可以根據需求調整
param_distributions_xgb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.7, 0.8, 0.9], # 每次迭代時，隨機抽樣的訓練樣本比例
    'colsample_bytree': [0.7, 0.8, 0.9], # 建立每棵樹時，特徵的抽樣比例
    'gamma': [0, 0.1, 0.2] # 懲罰項，用來控制過擬合
}

# 定義評估指標
scorer = make_scorer(f1_score, average='weighted')

# 設定 RandomizedSearchCV
random_search_xgb = RandomizedSearchCV(
    estimator=model_to_tune,
    param_distributions=param_distributions_xgb,
    n_iter=10,  # 隨機搜索 10 次組合
    scoring=scorer,
    cv=3,       # 3 折交叉驗證
    n_jobs=-1,
    random_state=42,
    verbose=2
)

print("開始進行 XGBoost 超參數調整...")
random_search_xgb.fit(X_train_processed, y_train_encoded)

print(f"\nXGBoost 最佳參數組合: {random_search_xgb.best_params_}")
print(f"XGBoost 最佳調整後分數: {random_search_xgb.best_score_:.4f}")

# 最佳模型
final_model = random_search_xgb.best_estimator_

# 獲取每個測試樣本在每個肥料類別上的預測機率
# 請確保您的模型支援 predict_proba
try:
    predictions_proba = final_model.predict_proba(X_test_processed)

    # 獲取每個樣本機率最高的 3 個類別的索引
    # axis=1 表示對每一列 (每個樣本) 進行排序
    top_3_indices = np.argsort(predictions_proba, axis=1)[:, -3:][:, ::-1] # 獲取最後三列（機率最高的三個），並反轉順序使其從最高到最低

    # 使用 label_encoder 將索引轉換回原始肥料名稱
    predicted_top_3_names = label_encoder.inverse_transform(top_3_indices.flatten()).reshape(top_3_indices.shape)

    # 將這三個名稱以空格連接
    predicted_names_for_submission_top3 = [" ".join(row) for row in predicted_top_3_names]

    # 顯示前幾個用於提交的格式
    print("\n基於機率排名的前 3 個預測肥料名稱 (用於提交，前 10 個):")
    print(predicted_names_for_submission_top3[:10])

except AttributeError:
    print("\n您選擇的模型不支援 predict_proba。無法使用此方法預測前 3 個機率最高的類別。")
    print("請確認您選擇的模型是否支援 predict_proba，或者考慮使用支援機率輸出的模型 (例如 RandomForestClassifier, LogisticRegression, SVC(probability=True))。")


# 假設您的測試資料包含 'id' 欄位 (載入 test_df 時)
# 確保您之前沒有在 X_test_processed 中刪除 id，或者從原始 test_df 中獲取 id
# 這裡假設您在 X_test_processed 中保留了 id 欄位
# 如果您之前刪除了 'id'，請改用原始 test_df['id']

submission_df = pd.DataFrame({'id': test_df['id'], 'Fertilizer Name': predicted_names_for_submission_top3})

# 將提交檔案保存為 CSV 格式
submission_df.to_csv('submission.csv', index=False)

print("提交檔案 'submission.csv' 已產生！")

