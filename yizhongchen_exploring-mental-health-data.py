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


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示為方框的問題


df = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv", index_col="id")

# 基本資料檢視
print("資料前 5 行:\n", df.head())


print("\n資料維度 (rows, columns):", df.shape)


df.info()


print("NaN 值數量:\n", df.isnull().sum())


print("\n數值欄位描述性統計:\n", df.describe())


def calculate_coefficient_of_variation(series):
  """
  計算 Pandas Series 的變異係數。

  Args:
    series: 一個 Pandas Series 物件，包含要計算的數據。

  Returns:
    一個浮點數，表示變異係數。如果平均值為 0 或 Series 為空，則返回 NaN。
  """
  if series.empty:
    return float('nan')
  mean = series.mean()
  if mean == 0:
    return float('nan')
  std = series.std()
  cv = std / abs(mean)  # 通常使用平均值的絕對值
  return cv


numerical_cols = df.select_dtypes(include=np.number).columns

print("各數值欄位的變異係數：")
for col in numerical_cols:
  cv = calculate_coefficient_of_variation(df[col])
  print(f"{col}: {cv:.4f}")


print("\n類別欄位描述性統計:\n", df.describe(include='object'))


depression_counts = df['Depression'].value_counts(normalize=True).reset_index()
depression_counts.columns = ['Depression', 'Proportion']
class_mapping = {0: 'Not Depressed', 1: 'Depressed'}
palette = {0: sns.color_palette("Set1")[1], 1: sns.color_palette("Set1")[0]}

plt.figure(figsize=(8, 6))
sns.barplot(x='Depression', y='Proportion', data=depression_counts, palette=palette)

plt.title('Depression Distribution')
plt.xlabel('Depression Status')
plt.ylabel('Proportion')
plt.xticks([0, 1], [class_mapping[0], class_mapping[1]])
plt.yticks(ticks=plt.yticks()[0], labels=[f'{y:.0%}' for y in plt.yticks()[0]])
plt.ylim(0, 1.05)  # Adjust y-axis limit for better text visibility

for index, row in depression_counts.iterrows():
    plt.text(row.name, row['Proportion'] + 0.02, f'{row["Proportion"]:.2%}', ha='center', va='bottom', fontsize=10)

sns.despine()
plt.tight_layout()
plt.show()


numerical_df = df.select_dtypes(include=['number'])

# 計算相關性矩陣
correlation_matrix = numerical_df.corr()

# 繪製熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# 1. Gender vs. Depression Proportion
gender_depression = df.groupby('Gender')['Depression'].value_counts(normalize=True).unstack()
gender_depression.plot(kind='bar', stacked=True)
plt.title('Depression Proportion by Gender')
plt.ylabel('Proportion')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.legend(title='Depression', labels=['No', 'Yes'])
plt.show()

# 2. Working Professional or Student vs. Depression Proportion
wps_depression = df.groupby('Working Professional or Student')['Depression'].value_counts(normalize=True).unstack()
wps_depression.plot(kind='bar', stacked=True)
plt.title('Depression Proportion by Working Professional or Student Status')
plt.ylabel('Proportion')
plt.xlabel('Status')
plt.xticks(rotation=0)
plt.legend(title='Depression', labels=['No', 'Yes'])
plt.show()

# 3. Family History of Mental Illness vs. Depression Proportion
family_history_depression = df.groupby('Family History of Mental Illness')['Depression'].value_counts(normalize=True).unstack()
family_history_depression.plot(kind='bar', stacked=True)
plt.title('Depression Proportion by Family History of Mental Illness')
plt.ylabel('Proportion')
plt.xlabel('Family History')
plt.xticks(rotation=0)
plt.legend(title='Depression', labels=['No', 'Yes'])
plt.show()


sns.violinplot(x='Depression', y='Age', data=df)
plt.title('Age Distribution by Depression Status (Violin Plot)')
plt.xlabel('Depression')
plt.ylabel('Age')
plt.show()

# 5. Work/Study Hours vs. Depression
sns.boxplot(x='Depression', y='Work/Study Hours', data=df)
plt.title('Work/Study Hours Distribution by Depression Status')
plt.xlabel('Depression')
plt.ylabel('Work/Study Hours')
plt.show()


df['Employment_Status'] = 'Employed'
df.loc[df['Working Professional or Student'] == 'Student', 'Employment_Status'] = 'Student'
df.loc[(df['Working Professional or Student'] != 'Student') & (df['Profession'].isnull()), 'Employment_Status'] = 'Unemployed'

# 計算不同就業狀態的憂鬱症比例
depression_proportion_by_status = df.groupby('Employment_Status')['Depression'].value_counts(normalize=True).mul(100).unstack()
print("\n不同就業狀態的憂鬱症比例 (%):\n", depression_proportion_by_status)

# 繪製不同就業狀態的憂鬱症比例長條圖
depression_proportion_by_status.plot(kind='bar', figsize=(10, 6))
plt.title('Depression Proportion by Employment Status')
plt.xlabel('Employment Status')
plt.ylabel('Proportion (%)')
plt.xticks(rotation=0)
plt.legend(title='Depression', labels=['No', 'Yes'])
sns.despine()
plt.tight_layout()
plt.show()


# Filter respondents aged 26 to 30 (inclusive)
age_26_30_df = df[(df['Age'] >= 26) & (df['Age'] <= 30)].copy()

# Create a new column indicating if the respondent is a student
age_26_30_df['Is_Student'] = age_26_30_df['Working Professional or Student'].apply(lambda x: 'Student' if x == 'Student' else 'Non-Student')

# Calculate the depression proportion for students and non-students aged 26-30
depression_proportion_26_30 = age_26_30_df.groupby('Is_Student')['Depression'].value_counts(normalize=True).mul(100).unstack()


# Plot the depression proportion for students and non-students aged 26-30
depression_proportion_26_30.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Depression Proportion for Students and Non-Students aged 26-30')
plt.xlabel('Status')
plt.ylabel('Proportion (%)')
plt.xticks(rotation=0)
plt.legend(title='Depression', labels=['No', 'Yes'])
sns.despine()
plt.tight_layout()
plt.show()

print("\nDepression Proportion for Students and Non-Students aged 26-30 (%):\n", depression_proportion_26_30)


# Filter respondents under 25 years old
below_25_df = df[df['Age'] < 25].copy()

# Create a new column indicating if the respondent is a student
below_25_df['Is_Student'] = below_25_df['Working Professional or Student'].apply(lambda x: 'Student' if x == 'Student' else 'Non-Student')

# Calculate the depression proportion for students and non-students under 25
depression_proportion_below_25 = below_25_df.groupby('Is_Student')['Depression'].value_counts(normalize=True).mul(100).unstack()

# Plot the depression proportion for students and non-students under 25
depression_proportion_below_25.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Depression Proportion for Students and Non-Students under 25')
plt.xlabel('Status')
plt.ylabel('Proportion (%)')
plt.xticks(rotation=0)
plt.legend(title='Depression', labels=['No', 'Yes'])
sns.despine()
plt.tight_layout()
plt.show()

print("\nDepression Proportion for Students and Non-Students under 25 (%):\n", depression_proportion_below_25)



# 篩選出 26 歲以上的族群
older_than_25_df = df[df['Age'] >= 26].copy()
# 創建 'Employment_Status_26_plus' 欄位來區分不同工作狀態
older_than_25_df['Employment_Status_26_plus'] = 'Employed'
older_than_25_df.loc[older_than_25_df['Working Professional or Student'] == 'Student', 'Employment_Status_26_plus'] = 'Student'
older_than_25_df.loc[(older_than_25_df['Working Professional or Student'] != 'Student') & (older_than_25_df['Profession'].isnull()), 'Employment_Status_26_plus'] = 'Unemployed'
# 篩選出 26 歲以上的非學生工作族群
non_student_older_than_25_df = older_than_25_df[older_than_25_df['Employment_Status_26_plus'] != 'Student'].copy()
# 計算 26 歲以上非學生工作族群的憂鬱症比例
depression_proportion_older_than_25 = non_student_older_than_25_df.groupby('Employment_Status_26_plus')['Depression'].value_counts(normalize=True).mul(100).unstack()
print("\n26 歲以上非學生工作族群的憂鬱症比例 (%):\n", depression_proportion_older_than_25)
# 繪製 26 歲以上非學生工作族群的憂鬱症比例長條圖
depression_proportion_older_than_25.plot(kind='bar', figsize=(8, 6))
plt.title('Depression Proportion for Non-Students Aged 26+')
plt.xlabel('Employment Status')
plt.ylabel('Proportion (%)')
plt.xticks(rotation=0)
plt.legend(title='Depression', labels=['No', 'Yes'])
sns.despine()
plt.tight_layout()
plt.show()


# --- 分開作比較 ---

# 學業壓力與憂鬱症
plt.figure(figsize=(8, 6))
sns.boxplot(x='Depression', y='Academic Pressure', data=df)
plt.title('Academic Pressure by Depression Status')
plt.xlabel('Depression')
plt.ylabel('Academic Pressure')
plt.show()

correlation_academic_depression = df[['Academic Pressure', 'Depression']].corr().iloc[0, 1]
print(f"學業壓力與憂鬱症的相關性: {correlation_academic_depression:.4f}")

# 工作壓力與憂鬱症
plt.figure(figsize=(8, 6))
sns.boxplot(x='Depression', y='Work Pressure', data=df)
plt.title('Work Pressure by Depression Status')
plt.xlabel('Depression')
plt.ylabel('Work Pressure')
plt.show()

correlation_work_depression = df[['Work Pressure', 'Depression']].corr().iloc[0, 1]
print(f"工作壓力與憂鬱症的相關性: {correlation_work_depression:.4f}")

# --- 針對學生族群 ---
students_df = df[df['Working Professional or Student'] == 'Student']

# CGPA 與憂鬱症 (學生)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Depression', y='CGPA', data=students_df)
plt.title('CGPA by Depression Status (Students)')
plt.xlabel('Depression')
plt.ylabel('CGPA')
plt.show()

correlation_cgpa_depression_student = students_df[['CGPA', 'Depression']].corr().iloc[0, 1]
print(f"學生 CGPA 與憂鬱症的相關性: {correlation_cgpa_depression_student:.4f}")

# 學習滿意度與憂鬱症 (學生)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Depression', y='Study Satisfaction', data=students_df)
plt.title('Study Satisfaction by Depression Status (Students)')
plt.xlabel('Depression')
plt.ylabel('Study Satisfaction')
plt.show()

correlation_study_satisfaction_depression_student = students_df[['Study Satisfaction', 'Depression']].corr().iloc[0, 1]
print(f"學生學習滿意度與憂鬱症的相關性: {correlation_study_satisfaction_depression_student:.4f}")

# 工作人士的工作/學習時長與憂鬱症
working_professionals_df = df[df['Working Professional or Student'] == 'Working Professional']

plt.figure(figsize=(8, 6))
sns.boxplot(x='Depression', y='Work/Study Hours', data=working_professionals_df)
plt.title('Work/Study Hours by Depression Status (Working Professionals)')
plt.xlabel('Depression')
plt.ylabel('Work/Study Hours')
plt.show()

correlation_work_study_depression_worker = working_professionals_df[['Work/Study Hours', 'Depression']].corr().iloc[0, 1]
print(f"工作人士工作/學習時長與憂鬱症的相關性: {correlation_work_study_depression_worker:.4f}")

# 工作/學習時長與憂鬱症 (學生)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Depression', y='Work/Study Hours', data=students_df)
plt.title('Work/Study Hours by Depression Status (Students)')
plt.xlabel('Depression')
plt.ylabel('Work/Study Hours')
plt.show()

correlation_work_study_depression_student = students_df[['Work/Study Hours', 'Depression']].corr().iloc[0, 1]
print(f"學生工作/學習時長與憂鬱症的相關性: {correlation_work_study_depression_student:.4f}")

# 經濟壓力與憂鬱症 (學生)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Depression', y='Financial Stress', data=students_df)
plt.title('Financial Stress by Depression Status (Students)')
plt.xlabel('Depression')
plt.ylabel('Financial Stress')
plt.show()

correlation_financial_stress_depression_student = students_df[['Financial Stress', 'Depression']].corr().iloc[0, 1]
print(f"學生經濟壓力與憂鬱症的相關性: {correlation_financial_stress_depression_student:.4f}")

# --- 針對工作人士族群 ---
working_professionals_df = df[df['Working Professional or Student'] == 'Working Professional']

# 經濟壓力與憂鬱症 (工作)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Depression', y='Financial Stress', data=working_professionals_df)
plt.title('Financial Stress by Depression Status (Working Professionals)')
plt.xlabel('Depression')
plt.ylabel('Financial Stress')
plt.show()

correlation_financial_stress_depression_worker = working_professionals_df[['Financial Stress', 'Depression']].corr().iloc[0, 1]
print(f"工作人士經濟壓力與憂鬱症的相關性: {correlation_financial_stress_depression_worker:.4f}")


df_processed = df.copy()
validation_df = pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv", index_col="id")


num_rows = validation_df.shape[0]
print(f"validation_df 中有 {num_rows} 筆資料。")


df_processed['Dietary Habits'].unique()


validation_df['Dietary Habits'].unique()


mapping = {
    'More Healthy': 'Healthy',
    'Less Healthy': 'Unhealthy',
    'Less than Healthy': 'Unhealthy',
    'No Healthy': 'Unhealthy'
    # Add lowercase versions if needed, e.g., 'healthy': 'Healthy'
}

# 3. Apply the mappings
# Use errors='ignore' in case some values in mapping aren't present
# Or handle potential type errors if the column isn't purely string/object
df_processed['Dietary Habits'] = df_processed['Dietary Habits'].replace(mapping)
validation_df['Dietary Habits'] = df_processed['Dietary Habits'].replace(mapping)


valid_categories = ['Healthy', 'Unhealthy', 'Moderate']
df_processed['Dietary Habits'] = df_processed['Dietary Habits'].where(
    df_processed['Dietary Habits'].isin(valid_categories), # Keep the value if it's in the valid list
    np.nan                                                # Otherwise, replace it with NaN
)
validation_df['Dietary Habits'] = df_processed['Dietary Habits'].where(
    df_processed['Dietary Habits'].isin(valid_categories), # Keep the value if it's in the valid list
    np.nan                                                # Otherwise, replace it with NaN
)


print(df_processed['Dietary Habits'].unique())
print(validation_df['Dietary Habits'].unique())


import re

def parse_sleep(duration):
    if isinstance(duration, str):
        numbers = re.findall(r'\d+\.?\d*', duration) # 尋找所有浮點數或整數

        if numbers:
            numeric_values = [float(num) for num in numbers]
            return np.mean(numeric_values)
        else:
            return np.nan # 如果找不到任何數字，則返回 NaN
    return np.nan # 非字串則返回 NaN


def derive_employment(row):
    if row['Working Professional or Student'] == 'Student':
        return 'Student'
    elif row['Work/Study Hours'] >= 0:
        return 'Employed'
    else:
        return 'Unemployed' # Assumes non-student with > 0 hours is employed


df_processed['Sleep Duration Numeric'] = df_processed['Sleep Duration'].apply(parse_sleep)
df_processed['Sleep Duration Numeric'].head()


validation_df['Sleep Duration Numeric'] = validation_df['Sleep Duration'].apply(parse_sleep)
validation_df['Sleep Duration Numeric'].head()


df_processed['Employment_Status'] = df_processed.apply(derive_employment, axis=1)
validation_df['Employment_Status'] = validation_df.apply(derive_employment, axis=1)
print("'Employment_Status' derived.")


df_processed.head(1)


df_processed['Academic Pressure'] = df_processed['Academic Pressure'].fillna(0)
df_processed['Work Pressure'] = df_processed['Work Pressure'].fillna(0)
df_processed['Study Satisfaction'] = df_processed['Study Satisfaction'].fillna(0)
df_processed['Job Satisfaction'] = df_processed['Job Satisfaction'].fillna(0)
df_processed['Sleep Duration Numeric'] = df_processed['Job Satisfaction'].fillna(df_processed['Sleep Duration Numeric'].mean())
df_processed['CGPA'] = df_processed['CGPA'].fillna(0) # CGPA NaN 通常是非學生，填 0
df_processed['Financial Stress'] = df_processed['Financial Stress'].fillna(df_processed['Financial Stress'].mean())


validation_df['Academic Pressure'] = validation_df['Academic Pressure'].fillna(0)
validation_df['Work Pressure'] = validation_df['Work Pressure'].fillna(0)
validation_df['Study Satisfaction'] = validation_df['Study Satisfaction'].fillna(0)
validation_df['Job Satisfaction'] = validation_df['Job Satisfaction'].fillna(0)
validation_df['Sleep Duration Numeric'] = validation_df['Job Satisfaction'].fillna(validation_df['Sleep Duration Numeric'].mean())
validation_df['CGPA'] = validation_df['CGPA'].fillna(0) # CGPA NaN 通常是非學生，填 0
validation_df['Financial Stress'] = validation_df['Financial Stress'].fillna(validation_df['Financial Stress'].mean())


# 合併為單一欄位 
df_processed['Pressure'] = df_processed['Academic Pressure'] + df_processed['Work Pressure']
df_processed['Satisfaction'] = df_processed['Study Satisfaction'] + df_processed['Job Satisfaction']


validation_df['Pressure'] = validation_df['Academic Pressure'] + validation_df['Work Pressure']
validation_df['Satisfaction'] = validation_df['Study Satisfaction'] + validation_df['Job Satisfaction']


# 捨棄原始文本、高基數類別、已轉換或合併的欄位
# 'Name', 'City', 'Profession', 'Degree' 通常識別性高或類別過多，先捨棄
# 'Working Professional or Student' 由 'Employment_Status' 取代
# 'Sleep Duration', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction', 'Depression' 已轉換或合併
drop_cols = ['Name', 'City', 'Working Professional or Student', 'Profession',
             'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction',
             'Sleep Duration', 'Degree', 'Dietary Habits']


numeric_features = ['Age', 'CGPA', 'Work/Study Hours', 'Financial Stress',
                    'Sleep Duration Numeric', 'Pressure', 'Satisfaction']
# ordinal_features = ['Dietary Habits']
dietary_order = ['Unhealthy', 'Moderate', 'Healthy'] # 定義順序
binary_features = ['Gender', 'Have you ever had suicidal thoughts ?',
                   'Family History of Mental Illness']
categorical_features = ['Employment_Status']


# 目標變數
target = 'Depression'


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report) # 匯入評估指標
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns # 用於繪製混淆矩陣


# --- 建立預處理 Pipeline ---

# 數值處理: 缺失值填充 (用中位數) + 標準化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# # 序數處理: 缺失值填充 (用眾數) + 序數編碼
# ordinal_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OrdinalEncoder(categories=[dietary_order]))]) # 指定順序

# 二元處理: 缺失值填充 (用眾數) + OneHot (drop='if_binary' 會自動變 0/1)
binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))])

# 多分類名目處理: 缺失值填充 (用 'missing' 策略或眾數) + OneHot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 或 strategy='constant', fill_value='Unknown'
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])


# 使用 ColumnTransformer 整合所有處理步驟
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        # ('ord', ordinal_transformer, ordinal_features),
        ('bin', binary_transformer, binary_features),
        ('cat', categorical_transformer, categorical_features)],
    remainder='drop') # 捨棄未指定的欄位


# --- 套用預處理 ---
# 從 df_processed 中選取特徵 (X) 和目標 (y)
X = df_processed.drop(columns=drop_cols + [target]) # 移除捨棄欄位和目標
y = df_processed[target]


Z = validation_df.drop(columns=drop_cols)


X.isnull().sum()


Z.isnull().sum()


# 套用預處理器 (注意：fit_transform 用在訓練資料，transform 用在測試資料)
# 為了分群，我們先對所有 X 進行轉換
X_processed = preprocessor.fit_transform(X)
# 獲取處理後的特徵名稱 (對於理解和後續分析有用)
feature_names_out = preprocessor.get_feature_names_out()
X_processed_df = pd.DataFrame(X_processed, columns=feature_names_out, index=X.index)

print(f"\n處理後特徵數量: {X_processed_df.shape[1]}")


Z_processed = preprocessor.transform(Z)
# 獲取處理後的特徵名稱 (對於理解和後續分析有用)
feature_names_out = preprocessor.get_feature_names_out()
Z_processed_df = pd.DataFrame(Z_processed, columns=feature_names_out, index=Z.index)
print(f"\n處理後特徵數量: {Z_processed_df.shape[1]}")


n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_processed)
X_processed_df['Cluster'] = clusters # 也加到處理過的 DataFrame

print(f"\n--- K-Means 分群完成 (k={n_clusters}) ---")
print("各群數量分佈:")
print(X_processed_df['Cluster'].value_counts())


kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(Z_processed)
Z_processed_df['Cluster'] = clusters # 也加到處理過的 DataFrame

print(f"\n--- K-Means 分群完成 (k={n_clusters}) ---")
print("各群數量分佈:")
print(Z_processed_df['Cluster'].value_counts())


X_processed_df.head(1)


print("\n各群特徵平均值 (部分範例):")
cluster_analysis = X_processed_df.groupby('Cluster')[['num__Age', 'num__Financial Stress', 'num__Work/Study Hours','num__Sleep Duration Numeric','num__Pressure']].mean()
print(cluster_analysis)


print("\n各群特徵平均值 (部分範例):")
cluster_analysis = Z_processed_df.groupby('Cluster')[['num__Age', 'num__Financial Stress', 'num__Work/Study Hours','num__Sleep Duration Numeric','num__Pressure']].mean()
print(cluster_analysis)
print(cluster_analysis)


# --- 準備深度學習資料 ---
y_dl = y.values # 目標變數

# 資料集 1: 不含 Cluster 特徵
X_dl_no_cluster = X_processed # 使用 K-Means 輸入的那個 preprocessed array
print(f"\n資料集 1 (無 Cluster) 特徵數量: {X_dl_no_cluster.shape[1]}")

# 資料集 2: 包含 Cluster 特徵
# 將 Cluster 特徵也進行標準化可能更好，但這裡先作為類別特徵處理
# 如果要標準化 Cluster，需要稍微調整 preprocessor
X_dl_with_cluster = X_processed_df.values # 使用包含 Cluster 的 DataFrame 轉 array
print(f"資料集 2 (含 Cluster) 特徵數量: {X_dl_with_cluster.shape[1]}")
Z_dl_with_cluster = Z_processed_df.values # 使用包含 Cluster 的 DataFrame 轉 array
print(f"資料集 3 (含 Cluster) 特徵數量: {X_dl_with_cluster.shape[1]}")


# --- 分割訓練/測試集 (為確保可比較，使用相同的 random_state) ---
# 模型 1
X_train_nc, X_test_nc, y_train, y_test = train_test_split(
    X_dl_no_cluster, y_dl, test_size=0.2, random_state=42, stratify=y_dl
)
# 模型 2 (使用相同的 y_train, y_test 分割)
X_train_wc, X_test_wc, _, _ = train_test_split(
    X_dl_with_cluster, y_dl, test_size=0.2, random_state=42, stratify=y_dl
)


def build_compile_model(input_dim):
    """建立並編譯 Keras 模型"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", name="layer1"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", name="layer2"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid", name="output_layer"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name='auc')],
    )
    return model



def train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """訓練模型並進行詳細評估"""
    print(f"\n--- [{model_name}] 開始訓練 ---")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0 # 設定 verbose=0 減少訓練過程輸出
        # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)] # 可選
    )
    print(f"--- [{model_name}] 訓練完成 ---")

    # Keras 的評估
    loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- [{model_name}] Keras 評估結果 (測試集) ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Sklearn 的詳細評估
    y_pred_proba = model.predict(X_test).flatten() # 預測機率
    y_pred_class = (y_pred_proba > 0.5).astype(int) # 轉換為類別 (0 或 1)

    print(f"\n--- [{model_name}] Sklearn 評估結果 (測試集) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_class):.4f}")
    # average='binary' 適用於二元分類，如果有多分類用 'macro' 或 'weighted'
    print(f"Precision: {precision_score(y_test, y_pred_class, average='binary'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_class, average='binary'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_class, average='binary'):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}") # AUC 通常用預測機率算

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class, target_names=['No Depression', 'Depression']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_class)
    print(cm)

    # 繪製混淆矩陣熱力圖
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

    return accuracy, auc # 返回主要指標用於比較


# --- 訓練和評估模型 1 (不含 Cluster) ---
input_dim_nc = X_train_nc.shape[1]
model_nc = build_compile_model(input_dim_nc)
model_nc.summary() # 顯示模型架構
acc_nc, auc_nc = train_evaluate_model("Model without Cluster", model_nc, X_train_nc, y_train, X_test_nc, y_test)


# --- 訓練和評估模型 2 (包含 Cluster) ---
input_dim_wc = X_train_wc.shape[1]
model_wc = build_compile_model(input_dim_wc)
# model_wc.summary() # 架構類似，只是輸入維度+1
acc_wc, auc_wc = train_evaluate_model("Model with Cluster", model_wc, X_train_wc, y_train, X_test_wc, y_test)


# --- 5. 比較結果 ---
print("\n--- 模型效能比較 ---")
print(f"Model without Cluster: Accuracy={acc_nc:.4f}, AUC={auc_nc:.4f}")
print(f"Model with Cluster:    Accuracy={acc_wc:.4f}, AUC={auc_wc:.4f}")

if acc_wc > acc_nc and auc_wc > auc_nc:
    print("\n結論：包含 Cluster 特徵的模型表現更好。")
elif acc_wc < acc_nc and auc_wc < auc_nc:
     print("\n結論：包含 Cluster 特徵的模型表現較差。")
else:
    print("\n結論：包含 Cluster 特徵對模型效能的影響不明顯或互有勝負，需根據具體指標權衡。")



def train_evaluate_model_with_regularization(model_name, model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, regularization='l2', l_rate=0.01):
    """訓練帶有 L1 或 L2 正則化的模型並進行詳細評估"""
    print(f"\n--- [{model_name} with {regularization.upper()} Regularization] 開始訓練 ---")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0, # 設定 verbose=0 減少訓練過程輸出
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)] # 啟用 Early Stopping
    )
    print(f"--- [{model_name} with {regularization.upper()} Regularization] 訓練完成 ---")

    # Keras 的評估
    loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- [{model_name} with {regularization.upper()} Regularization] Keras 評估結果 (測試集) ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Sklearn 的詳細評估
    y_pred_proba = model.predict(X_test).flatten() # 預測機率
    y_pred_class = (y_pred_proba > 0.5).astype(int) # 轉換為類別 (0 或 1)

    print(f"\n--- [{model_name} with {regularization.upper()} Regularization] Sklearn 評估結果 (測試集) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_class):.4f}")
    # average='binary' 適用於二元分類
    print(f"Precision: {precision_score(y_test, y_pred_class, average='binary'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_class, average='binary'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_class, average='binary'):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}") # AUC 通常用預測機率算

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class, target_names=['No Depression', 'Depression']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_class)
    print(cm)

    # 繪製混淆矩陣熱力圖
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.title(f'Confusion Matrix - {model_name} with {regularization.upper()}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

    return accuracy, auc # 返回主要指標用於比較

# 範例：如何將正則化應用於模型
def create_regularized_dnn_model(input_dim, regularization='l2', l_rate=0.01):
    regularizer = None
    if regularization == 'l1':
        regularizer = keras.regularizers.L1(l_rate)
    elif regularization == 'l2':
        regularizer = keras.regularizers.L2(l_rate)

    model = keras.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizer, input_dim=input_dim),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model

# 假設您已經有 X_train, y_train, X_test, y_test 和 input_dim
# input_dim = X_train.shape[1]

# 創建帶有 L2 正則化的模型並訓練評估
# regularized_dnn_l2 = create_regularized_dnn_model(input_dim, regularization='l2', l_rate=0.01)
# train_evaluate_model_with_regularization("DNN_L2", regularized_dnn_l2, X_train, y_train, X_test, y_test)

# 創建帶有 L1 正則化的模型並訓練評估
# regularized_dnn_l1 = create_regularized_dnn_model(input_dim, regularization='l1', l_rate=0.001) # L1 通常使用較小的學習率
# train_evaluate_model_with_regularization("DNN_L1", regularized_dnn_l1, X_train, y_train, X_test, y_test)


# --- 訓練和評估模型 2 (包含 Cluster) ---
input_dim = X_train_wc.shape[1]
regularized_dnn_l2 = create_regularized_dnn_model(input_dim, regularization='l2', l_rate=0.01)
train_evaluate_model_with_regularization("DNN_L2", regularized_dnn_l2, X_train_wc, y_train, X_test_wc, y_test)


predictions_proba = regularized_dnn_l2.predict(Z_processed_df)
predictions_class = (predictions_proba > 0.5).astype(int)


original_Z_df = Z_processed_df.reset_index()
results_df = pd.DataFrame({'Depression': predictions_class.flatten()}, index=validation_df.index)
results_df.index.name = 'id'


submission = results_df.reset_index('id')
submission.head()


num_rows = submission.shape[0]
print(f"submission 中有 {num_rows} 筆資料。")


submission.to_csv('submission2.csv', index=False)
print("\n預測結果已儲存到 submission.csv")




