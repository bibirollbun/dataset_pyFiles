import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier


# 讀取資料
train_set = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test_set = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')

train_set.head()
train_set.info()

# 保存test_set的id
test_ids = test_set['id']


# missing info calculation
columns_to_check = [
    'id', 'Name', 'Gender', 'Age', 'City', 'Working Professional or Student',
    'Profession', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
    'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Degree',
    'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
    'Financial Stress', 'Family History of Mental Illness'
]

missing_info = train_set[columns_to_check].isnull().sum().to_frame(name='Missing Count')
missing_info['Missing Percent'] = (missing_info['Missing Count'] / len(train_set)) * 100
missing_info = missing_info[missing_info['Missing Count'] > 0]  
missing_info = missing_info.sort_values(by='Missing Percent', ascending=False)

print("欄位缺失值統計：")
print(missing_info.round(2))


# check distribution of target variable 
class_counts = train_set['Depression'].value_counts()
class_percent = train_set['Depression'].value_counts(normalize=True) * 100

print("Depression 類別分布（樣本數）：\n", class_counts)
print("\n Depression 類別分布（百分比）：\n", class_percent.round(2))

plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, palette='pastel')
plt.title('depression distribution')
plt.xlabel('class')
plt.ylabel('numbers of data')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# 整併欄位
train_set['Work/Study Pressure'] = train_set[['Academic Pressure', 'Work Pressure']].fillna(0).mean(axis=1)
train_set['Work/Study Satisfaction'] = train_set[['Study Satisfaction', 'Job Satisfaction']].fillna(0).mean(axis=1)

test_set['Work/Study Pressure'] = test_set[['Academic Pressure', 'Work Pressure']].fillna(0).mean(axis=1)
test_set['Work/Study Satisfaction'] = test_set[['Study Satisfaction', 'Job Satisfaction']].fillna(0).mean(axis=1)

train_set['Profession'] = train_set['Profession'].fillna('Student')
test_set['Profession'] = test_set['Profession'].fillna('Student')

train_set['CGPA'] = train_set['CGPA'].fillna('999').astype(float)
test_set['CGPA'] = test_set['CGPA'].fillna('999').astype(float)

# 刪除少數具有缺失值的行
cols_with_missing = ['Dietary Habits', 'Financial Stress', 'Degree']
train_set= train_set.dropna(subset=cols_with_missing)

# 刪除原本欄位
columns_to_drop = ['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction']
train_set.drop(columns=columns_to_drop, inplace=True)
test_set.drop(columns=columns_to_drop, inplace=True)

# 保存處理後資料
train_set.to_csv('processed_train.csv', index=False)
test_set.to_csv('processed_test.csv', index=False)

train_set.info()
test_set.info()


# check correlation
from scipy.stats import pointbiserialr, chi2_contingency

# numerical data => point-Biserial Correlation
numerical_cols = ['Age', 'CGPA','Work/Study Pressure', 'Work/Study Satisfaction',
                  'Work/Study Hours', 'Financial Stress']

print("=== correlation between numerical variables and depression (Point-Biserial Correlation) ===")
for col in numerical_cols:
    series = train_set[col]
    if col == 'CGPA':
        # CGPA排除999的資料
        valid_mask = series != 999
        series = series[valid_mask]
        target = train_set.loc[valid_mask, 'Depression']
    else:
        series = series.fillna(series.mean())
        target = train_set['Depression']
    r, p = pointbiserialr(series, target)
    print(f"{col:<25}: r = {r:.3f}, p = {p:.4f}")

# categorical data => Chi-square test and Cramér’s V
categorical_cols = ['Gender','City', 'Working Professional or Student','Profession', 'Sleep Duration',
                    'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
                    'Family History of Mental Illness']

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape)-1)))

print("\n=== correlation between categorical variables and depression (Chi-square test and Cramér’s V) ===")
for col in categorical_cols:
    if train_set[col].isnull().sum() > 0:
        temp = train_set[col].fillna("Missing")
    else:
        temp = train_set[col]
    cm = pd.crosstab(temp, train_set['Depression'])
    chi2, p, _, _ = chi2_contingency(cm)
    v = cramers_v(cm)
    print(f"{col:<40}: Cramér’s V = {v:.3f}, p = {p:.4f}")

# mutual info
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# feature_cols 不含 id, Name, Depression
feature_cols = [col for col in train_set.columns if col not in ['id', 'Name', 'Depression']]

# 整理資料
X = train_set[feature_cols].copy()
y = train_set['Depression']

# label encoding for categorical columns
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].fillna("Missing")
    X[col] = LabelEncoder().fit_transform(X[col])

# fill missing numerical columns (不含CGPA999先處理)
for col in X.select_dtypes(include=np.number).columns:
    X[col] = X[col].fillna(X[col].mean())

# 除CGPA以外的 mutual info（用全部資料）
mi_scores = mutual_info_classif(X.drop(columns=['CGPA']), y, discrete_features='auto')
mi_series = pd.Series(mi_scores, index=X.drop(columns=['CGPA']).columns)

# CGPA mutual info(exclude cgpa999)
valid_mask = train_set['CGPA'] != 999
X_cgpa = X.loc[valid_mask, 'CGPA'].to_frame()
y_cgpa = y.loc[valid_mask]

mi_cgpa = mutual_info_classif(X_cgpa, y_cgpa, discrete_features=False)[0]  

# combine both
mi_series['CGPA'] = mi_cgpa

# sorting
mi_series = mi_series.sort_values(ascending=False)
print("\n=== ranking of mutual information ===")
print(mi_series.round(3))


# mutual information bar plot
plt.figure(figsize=(10, 6))
mi_series.sort_values().plot(kind='barh', color='skyblue')
plt.title('Feature Importance by Mutual Information')
plt.xlabel('Mutual Information Score')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# correlation matrix
from scipy.stats import pearsonr

target = 'Depression'
num_cols = train_set.select_dtypes(include=np.number).columns.drop(target)

corr_dict = {}

for col in num_cols:
    if col == 'CGPA':
        # 排除 CGPA=999
        mask = train_set['CGPA'] != 999
        x = train_set.loc[mask, col]
        y = train_set.loc[mask, target]
    else:
        # 全部資料
        x = train_set[col]
        y = train_set[target]
    r, p = pearsonr(x, y)
    corr_dict[col] = r

target_corr = pd.Series(corr_dict).sort_values(key=abs, ascending=False)

print(f"=== {target} 相關係數排名（CGPA排除999）===")
print(target_corr.round(3))

# heatmap (pandas corr無法排除部分列，用全部資料做（CGPA視為整欄）)
#correlation_matrix = train_set.corr(numeric_only=True)

#plt.figure(figsize=(12, 10))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
#plt.title('Correlation Matrix (All Data)')
#plt.show()



# correlation between varibales

# 刪除指定欄位
drop_cols = ['Name', 'id', 'CGPA', 'city', 'gender']
cols_to_drop = [col for col in drop_cols if col in train_set.columns]
train_set = train_set.drop(columns=cols_to_drop)

# 目標欄位
target = 'Depression'
threshold = 0.001

# calculate corr
correlation_matrix = train_set.corr(numeric_only=True)

# 移除目標欄位行列
corr_no_target = correlation_matrix.drop(target, axis=0).drop(target, axis=1)

pairs = []
for col1 in corr_no_target.columns:
    for col2 in corr_no_target.columns:
        if col1 != col2:
            corr_value = corr_no_target.loc[col1, col2]
            if abs(corr_value) > threshold:
                pair = tuple(sorted([col1, col2]))
                pairs.append((pair[0], pair[1], corr_value))

# 移除重複配對
pairs_unique = {}
for c1, c2, val in pairs:
    if (c1, c2) not in pairs_unique:
        pairs_unique[(c1, c2)] = val

# 依絕對值大小排序
sorted_pairs = sorted(pairs_unique.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"=== 相關係數絕對值大於 {threshold} 的欄位配對（依絕對值大小排序） ===")
for (c1, c2), val in sorted_pairs:
    print(f"{c1} vs {c2}: correlation = {val:.3f}")



# Hexbin Plot
#plt.figure(figsize=(14, 6))

#for i, dep in enumerate([0, 1]):
#    plt.subplot(1, 2, i+1)
#    subset = train_set[train_set['Depression'] == dep]
#    hb = plt.hexbin(
#        subset['Age'], 
#        subset['Work/Study Hours'], 
#        gridsize=30, 
#        cmap='Reds' if dep == 1 else 'Blues',
#        mincnt=1
#    )
#   plt.colorbar(hb, label='Count')
#    plt.xlabel('Age')
#    plt.ylabel('Work/Study Hours')
#   plt.title(f'Hexbin Plot (Depression={dep})')

#plt.tight_layout()
#plt.show()


# combined Hexbin Plot

#import matplotlib.patches as mpatches

#plt.figure(figsize=(10, 7))

# Depression=0（藍色）
#hb0 = plt.hexbin(
#    train_set.loc[train_set['Depression'] == 0, 'Age'],
#    train_set.loc[train_set['Depression'] == 0, 'Work/Study Hours'],
#    gridsize=30,
#    cmap='Blues',
#    alpha=0.5,   # 半透明
#    mincnt=1,
#    label='No Depression'
#)

# Depression=1（紅色）
#hb1 = plt.hexbin(
#    train_set.loc[train_set['Depression'] == 1, 'Age'],
#    train_set.loc[train_set['Depression'] == 1, 'Work/Study Hours'],
#    gridsize=30,
#    cmap='Reds',
#    alpha=0.5,   # 半透明
#    mincnt=1,
#    label='Depression'
#)

#plt.colorbar(hb0, label='Count (No Depression)', fraction=0.046, pad=0.04)
#plt.colorbar(hb1, label='Count (Depression)', fraction=0.046, pad=0.12)

#plt.xlabel('Age')
#plt.ylabel('Work/Study Hours')
#plt.title('Hexbin Plot of Age vs Work/Study Hours by Depression Status')

# 建立顏色 patch
#blue_patch = mpatches.Patch(color='#3498db', label='No Depression')
#red_patch = mpatches.Patch(color='#db5b34', label='Depression')

#plt.legend(handles=[blue_patch, red_patch], loc='upper right')
#plt.show()


def preprocess_data(df, is_training=True):
    # 填補缺失值
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 轉換二元變數
    binary_mapping = {
        'Yes': 1, 'No': 0,
        'Male': 1, 'Female': 0,
        'Student': 1, 'Working Professional': 0
    }
    df = df.astype(str).replace(binary_mapping)

    # 移除不需要的欄位
    drop_cols = ['id', 'Name', 'CGPA', 'City']
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1)
  
    if is_training:
        # 分離目標變數
        y = df['Depression']
        X = df.drop('Depression', axis=1)
        return X, y
    else:
        return df


# 預處理訓練資料
X, y = preprocess_data(train_set, is_training=True)
# 預處理測試資料
X_submission = preprocess_data(test_set, is_training=False)


# 處理類別變數
categorical_cols = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_cats = encoder.fit_transform(X[categorical_cols].astype(str))
encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))


# 轉換訓練資料
X = X.drop(categorical_cols, axis=1).reset_index(drop=True)
X = pd.concat([X, encoded_cats_df], axis=1)


# 轉換測試資料
encoded_cats_test = encoder.transform(X_submission[categorical_cols].astype(str))
encoded_cats_test_df = pd.DataFrame(encoded_cats_test, columns=encoder.get_feature_names_out(categorical_cols))
X_submission = X_submission.drop(categorical_cols, axis=1).reset_index(drop=True)
X_submission = pd.concat([X_submission, encoded_cats_test_df], axis=1)


# 替換空白字元為底線
X.columns = X.columns.str.replace(' ', '_')
X_submission.columns = X_submission.columns.str.replace(' ', '_')


# 將資料分割為訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y.astype(int), test_size=0.2, random_state=42)


#SMOTE oversampling
from imblearn.over_sampling import SMOTE

#設定增加data數量 (depression[1/0 = 1:3])
sampling_strategy = 1 / 3

# oversampling on traning set
smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# 訓練 RandomForest
rf_model = RandomForestClassifier(class_weight='balanced',random_state=42)
rf_model.fit(X_train, y_train)


# 預測驗證集
rf_pred_val = rf_model.predict(X_val)

# 評估驗證集表現
print("Random Forest Performance on Validation Data:")
print(classification_report(y_val, rf_pred_val))


# 訓練 XGBoost
y_train = y_train.astype(int)
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.5, 0.7],
    'n_estimators': [150, 200]
}

positive_class_count = y_train.sum()
negative_class_count = len(y_train) - positive_class_count

xgb_model = XGBClassifier(objective='binary:logistic',tree_method='hist', device='cuda',scale_pos_weight = negative_class_count / positive_class_count, random_state=42)

xgb_grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy')
xgb_grid_search.fit(X_train_resampled, y_train_resampled)


# 預測驗證集
xgb_best_model = xgb_grid_search.best_estimator_
grid_pred_val = xgb_best_model.predict(X_val)

# 顯示最佳參數與驗證結果
print("Best Parameters:", xgb_grid_search.best_params_)
print("XGBoost Performance on Validation Data:")
print(classification_report(y_val, grid_pred_val))


# 定義 CatBoost 超參數範圍
catboost_param_grid = {
    'depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.2],
    'iterations': [100, 150, 200]
}

# 建立 CatBoost 模型
positive_class_count = y_train.sum()
negative_class_count = len(y_train) - positive_class_count
scale_weight = negative_class_count / positive_class_count

catboost_model = CatBoostClassifier(task_type="GPU", random_state=42, scale_pos_weight=scale_weight, silent=True)

# 使用 GridSearchCV 搜索最佳參數
catboost_grid_search = GridSearchCV(catboost_model, catboost_param_grid, cv=3, scoring='accuracy')
catboost_grid_search.fit(X_train, y_train)


print("Best Parameters:", catboost_grid_search.best_params_)

# 使用最佳參數進行預測
catboost_best_model = catboost_grid_search.best_estimator_
catboost_pred = catboost_best_model.predict(X_val)

# 評估 CatBoost 模型
catboost_report = classification_report(y_val, catboost_pred)
print("CatBoost Performance on Validation Data:")
print(catboost_report)


# 設定 LightGBM 模型
lgbm_model = LGBMClassifier(device='gpu', class_weight='balanced',random_state=42)

# 定義超參數網格
param_grid = {
    'num_leaves': [31, 63],
    'max_depth': [6, 10],
    'learning_rate': [0.1],
    'n_estimators': [100],
    'min_child_samples': [20],
    'colsample_bytree': [0.8],
    'subsample': [0.8],
    'reg_alpha': [0],
    'reg_lambda': [0]
}

# 使用 GridSearchCV 調參
grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)


# 輸出最佳參數
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy on Training Data:", grid_search.best_score_)

# 使用最佳參數重新訓練
best_lgb = grid_search.best_estimator_
best_lgb.fit(X_train, y_train)

# 測試評估
lgbm_pred = best_lgb.predict(X_val)
print("Improved LightGBM Performance:")
print(classification_report(y_val, lgbm_pred))
print("Accuracy:", accuracy_score(y_val, lgbm_pred))


print("XGBoost Best Parameters:", xgb_grid_search.best_params_)
print("catBoost Best Parameters:", catboost_grid_search.best_params_)
print("LightGBM Best Parameters:", grid_search.best_params_)

print("Random Forest Performance:")
print(classification_report(y_val, rf_pred_val))
print("Accuracy:", accuracy_score(y_val, rf_pred_val))

print("XGBoost Performance:")
print(classification_report(y_val, grid_pred_val))
print("Accuracy:", accuracy_score(y_val, grid_pred_val))

print("CatBoost Performance:")
print(classification_report(y_val, catboost_pred))
print("Accuracy:", accuracy_score(y_val, catboost_pred))

print("LightGBM Performance:")
print(classification_report(y_val, lgbm_pred))
print("Accuracy:", accuracy_score(y_val, lgbm_pred))


# 加權投票集成
weights = [0.1, 0.3, 0.3, 0.3]
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_best_model),
        ('cat', catboost_best_model),
        ('lgb', best_lgb)
    ],
    voting='soft',  # 使用機率加權
    weights=weights
)

# 訓練模型
ensemble.fit(X_train, y_train)


# 預測並評估
y_pred = ensemble.predict(X_val)
print("Ensemble Performance:")
print(classification_report(y_val, y_pred))
print("Accuracy:", accuracy_score(y_val, y_pred))


# 對測試資料進行預測
#xgb_pred  = xgb_grid_search.predict(X_submission)

# 儲存預測結果
#submission = pd.DataFrame({
#    'id': test_ids,
#    'Depression': xgb_pred
#})
#submission.to_csv('submission.csv', index=False)


# 預測 test 資料
ensemble_pred = ensemble.predict(X_submission)

# 儲存結果
submission = pd.DataFrame({
    'id': test_ids,
    'Depression': ensemble_pred
})
submission.to_csv('submission.csv', index=False)

