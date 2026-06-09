# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from IPython.display import display
from tqdm import tqdm


train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample_submission = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')


numerical_features = [
    'Basic_Demos-Age', 'CGAS-CGAS_Score', 'Physical-BMI', 'Physical-Weight', 
    'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 
    'Physical-HeartRate', 'Physical-Systolic_BP','Physical-Height',
    'Fitness_Endurance-Max_Stage','Fitness_Endurance-Time_Mins',
    'Fitness_Endurance-Time_Sec','FGC-FGC_CU','FGC-FGC_GSND',
    'FGC-FGC_GSD','FGC-FGC_PU','FGC-FGC_SRL','FGC-FGC_SRR',
    'FGC-FGC_TL','BIA-BIA_BMC','BIA-BIA_BMI','BIA-BIA_BMR',
    'BIA-BIA_DEE','BIA-BIA_ECW','BIA-BIA_FFM','BIA-BIA_FFMI',
    'BIA-BIA_FMI','BIA-BIA_Fat','PAQ_A-PAQ_A_Total','BIA-BIA_ICW',
    'BIA-BIA_LDM','BIA-BIA_LST','BIA-BIA_SMM','BIA-BIA_TBW',
    'PAQ_C-PAQ_C_Total','PCIAT-PCIAT_01','PCIAT-PCIAT_02',
    'PCIAT-PCIAT_03','PCIAT-PCIAT_04','PCIAT-PCIAT_05','PCIAT-PCIAT_06',
    'PCIAT-PCIAT_07','PCIAT-PCIAT_08','PCIAT-PCIAT_09','PCIAT-PCIAT_10',
    'PCIAT-PCIAT_11','PCIAT-PCIAT_12','PCIAT-PCIAT_13','PCIAT-PCIAT_14',
    'PCIAT-PCIAT_15','PCIAT-PCIAT_16','PCIAT-PCIAT_17','PCIAT-PCIAT_18',
    'PCIAT-PCIAT_19','PCIAT-PCIAT_20','PCIAT-PCIAT_Total','SDS-SDS_Total_Raw',
    'SDS-SDS_Total_T','PreInt_EduHx-computerinternet_hoursday'
]
categorical_features = [
    'Basic_Demos-Enroll_Season', 'Basic_Demos-Sex', 'CGAS-Season', 
    'Physical-Season', 'Fitness_Endurance-Season', 'FGC-Season',
    'FGC-FGC_CU_Zone','FGC-FGC_GSND_Zone','FGC-FGC_GSD_Zone',
    'FGC-FGC_PU_Zone','FGC-FGC_SRL_Zone','FGC-FGC_SRR_Zone',
    'FGC-FGC_TL_Zone','BIA-Season','BIA-BIA_Activity_Level_num',
    'BIA-BIA_Frame_num','PAQ_A-Season','PAQ_C-Season','PCIAT-Season',
    'SDS-Season','PreInt_EduHx-Season'
]


sii_counts = train['sii'].value_counts().sort_index()

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(sii_counts, labels=sii_counts.index, autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title('sii Class Distribution')
plt.axis('equal')  # Make pie chart a circle
plt.tight_layout()
plt.show()
plt.show()


# Path to one Parquet file
file_path = "/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet/id=00115b9f/part-0.parquet"

# Read the Parquet file
df = pd.read_parquet(file_path)


# Show basic info
print(df.info())  # Check column types


# Plot histograms for all numeric columns
df.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Distribution of Numeric Features", fontsize=14)
plt.show()


# Aggregate features by relative_date_PCIAT
df_agg = df.groupby("relative_date_PCIAT").agg({
    "enmo": ["mean"],
    "light": ["mean"],
    "non-wear_flag": ["sum"], 
})


# Flatten column names
df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]

# Reset index if needed
df_agg = df_agg.reset_index()

# View aggregated dataset
df_agg.head()


plt.figure(figsize=(10,5))
plt.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o')  # Adjust columns as needed
plt.xlabel("X-Axis Feature (Time or Index)")
plt.ylabel("Y-Axis Feature")
plt.title("Line Plot of Sequential Data")
plt.show()


from concurrent.futures import ThreadPoolExecutor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def time_features(df):
    # Convert time_of_day to hours
    df["hours"] = df["time_of_day"] // (3_600 * 1_000_000_000)
    # Basic features
    features = [
        df["non-wear_flag"].mean(),
        df["enmo"][df["enmo"] >= 0.05].sum(),
    ]

    # Define conditions for night, day, and no mask (full data)
    night = ((df["hours"] >= 22) | (df["hours"] <= 5))
    day = ((df["hours"] <= 20) & (df["hours"] >= 7))
    no_mask = np.ones(len(df), dtype=bool)

    # List of columns of interest and masks
    keys = ["enmo", "anglez", "light", "battery_voltage"]
    masks = [no_mask, night, day]

    # Helper function for feature extraction
    def extract_stats(data):
        return [
            data.mean(),
            data.std(),
            data.max(),
            data.min(),
            data.diff().mean(),
            data.diff().std()
        ]

    # Iterate over keys and masks to generate the statistics
    for key in keys:
        for mask in masks:
            filtered_data = df.loc[mask, key]
            features.extend(extract_stats(filtered_data))

    return features

def build_feature_matrix_modified_v2(root):
    def process_time_features(folder):
        path = os.path.join(root, folder, 'part-0.parquet')
        try:
            df = pd.read_parquet(path).drop(columns='step', errors='ignore')
            features = time_features(df)
            subject_id = folder.split('=')[-1]
            return features, subject_id
        except FileNotFoundError:
            print(f"File not found: {path}")
            return None, folder.split('=')[-1]

    dirs = os.listdir(root)
    with ThreadPoolExecutor() as ex:
        stats_and_ids = list(tqdm(ex.map(process_time_features, dirs), total=len(dirs)))

    features_list = []
    ids = []
    for result in stats_and_ids:
        if result[0] is not None:
            features_list.append(result[0])
            ids.append(result[1])

    if not features_list:
        return pd.DataFrame()

    df = pd.DataFrame(features_list, columns=[f'time_feat_{i}' for i in range(len(features_list[0]))])
    df['id'] = ids
    return df

# 示例调用 (假设你的数据根目录是 '/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet' 或类似的)
train_series_raw = build_feature_matrix_modified_v2("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_series_raw = build_feature_matrix_modified_v2("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")

print("Processed train data shape:", train_series_raw.shape)
print("Processed test data shape:", test_series_raw.shape)
print("First few rows of processed train data:")
print(train_series_raw.head())


train_ids = train_series_raw['id'].copy()
test_ids = test_series_raw['id'].copy()

train_pca = train_series_raw.drop(columns='id')
test_pca = test_series_raw.drop(columns='id')



# train_pca=train_series_raw.copy()
# test_pca=train_series_raw.copy()
# train_pca=train_pca.drop(columns='id')
# test_pca=test_pca.drop(columns='id')


# from sklearn.impute import KNNImputer

# train_ids = train_series_raw['id'].copy()
# test_ids = test_series_raw['id'].copy()

# train_pca = train_series_raw.drop(columns='id')
# test_pca = test_series_raw.drop(columns='id')

# # Initialize the KNN Imputer
# imputer = KNNImputer(n_neighbors=5)  # You can tune n_neighbors as needed

# # Fit on train and transform both train and test
# train_knn_array = imputer.fit_transform(train_pca)
# test_knn_array = imputer.transform(test_pca)  # Important: Use only .transform() here

# # Convert arrays back to DataFrames
# train_knn_imputed = pd.DataFrame(train_knn_array, columns=train_pca.columns)
# test_knn_imputed = pd.DataFrame(test_knn_array, columns=test_pca.columns)

# # Reattach 'id' columns
# train_knn_imputed['id'] = train_ids
# test_knn_imputed['id'] = test_ids

# print(" First few rows of train set after KNN imputation:")
# print(train_knn_imputed.head())

# print("\n Missing values in train set:")
# print(train_knn_imputed.isna().sum().sum())

# print("\n" + "="*40 + "\n")

# print(" First few rows of test set after KNN imputation:")
# print(test_knn_imputed.head())

# print("\n Missing values in test set:")
# print(test_knn_imputed.isna().sum().sum())



from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import pandas as pd

imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)

# Fit and transform train, transform test
train_pca_imputed_array = imputer.fit_transform(train_pca)
test_pca_imputed_array = imputer.transform(test_pca)

# Convert back to DataFrame
train_pca_imputed = pd.DataFrame(train_pca_imputed_array, columns=train_pca.columns)
test_pca_imputed = pd.DataFrame(test_pca_imputed_array, columns=test_pca.columns)

print("Train (MICE) Imputed Sample:")
print(train_pca_imputed.head())

print("\n Missing values in Train (MICE):")
print(train_pca_imputed.isna().sum().sum())

print("\n Test (MICE) Imputed Sample:")
print(test_pca_imputed.head())

print("\n Missing values in Test (MICE):")
print(test_pca_imputed.isna().sum().sum())



train_pca=train_pca_imputed
test_pca=test_pca_imputed


from sklearn.decomposition import PCA

# 定义要测试的保留主成分数量列表
n_components_to_test = [20]  # 你可以根据需要修改这些值
explained_variance_ratios = {}

for n_components in n_components_to_test:
    # 初始化 PCA 模型
    pca = PCA(n_components=n_components, random_state=42)

    # 在训练集上拟合 PCA 模型
    pca.fit(train_pca)

    # 获取解释的方差比例
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    explained_variance_ratios[n_components] = explained_variance_ratio

    print(f"保留 {n_components} 个主成分时，解释的方差比例为: {explained_variance_ratio:.4f}")

# 找出保留信息最多的主成分数量
best_n_components = max(explained_variance_ratios, key=explained_variance_ratios.get)
best_explained_variance = explained_variance_ratios[best_n_components]

print(f"\n保留信息最多的主成分数量是 {best_n_components}，解释的方差比例为: {best_explained_variance:.4f}")

# 你可以选择使用 best_n_components 来进行最终的降维
best_pca = PCA(n_components=best_n_components, random_state=42)
principal_components_train_best = best_pca.fit_transform(train_pca)
column_names_train_best = [f'PC_{i+1}' for i in range(best_n_components)]
pca_df_train_best = pd.DataFrame(data=principal_components_train_best, columns=column_names_train_best)

# 同样的方法应用于测试集 (使用在训练集上学习到的 best_pca)
principal_components_test = best_pca.transform(test_pca)
column_names_test = [f'PC_{i+1}' for i in range(best_n_components)]
pca_df_test = pd.DataFrame(data=principal_components_test, columns=column_names_test)

print("\n使用最佳主成分数量压缩后的训练集维度:", pca_df_train_best.shape)
print("使用最佳主成分数量压缩后的测试集维度:", pca_df_test.shape)


pca_df_train_best['id']=train_series_raw['id']
pca_df_test['id']=test_series_raw['id']


# merge (combing tabular with time data)
train_df = pd.merge(train, pca_df_train_best, on='id', how='left')
test_df = pd.merge(test, pca_df_test, on='id', how='left')


train_df


# checkpoint: able to rerun from here onwards
train_unfilled = train_df.copy()
test_unfilled = test_df.copy()


# Drop rows with empty sii
train_unfilled = train_unfilled.dropna(subset='sii')

# Replace 0s with NaN in 'Physical-Weight'
train_unfilled.loc[train_unfilled["Physical-Weight"] == 0, "Physical-Weight"] = np.nan
test_unfilled.loc[test_unfilled["Physical-Weight"] == 0, "Physical-Weight"] = np.nan


def check_missing_values(df):
    # Check for missing (NaN) values
    print("Missing values in each column:\n", df.isna().sum())
    
    # Check for infinite values
    print("\nInfinite values in each column:\n", (df == np.inf).sum() + (df == -np.inf).sum())
    
    # # Display dataset information
    # df.info()

    # Calculate the percentage of missing values per column
    missing_percentage = (df.isna().sum() / len(df)) * 100
    
    # Sort the columns by missing percentage in descending order
    missing_percentage = missing_percentage.sort_values(ascending=False)
    
    # Plot the missing values as a horizontal bar chart (axes inverted)
    plt.figure(figsize=(8, 15))
    bars = missing_percentage.plot(kind='barh', color='skyblue', edgecolor='black')
    
    # Increase bar thickness
    for bar in bars.patches:
        bar.set_height(0.6)  # Adjust the thickness of bars
    
    # Formatting the plot
    plt.ylabel("Columns", fontsize=10)
    plt.xlabel("Percentage of Missing Values (%)", fontsize=12)
    plt.title("Percentage of Missing Values Per Column (Sorted)", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.yticks(fontsize=8)  # Reduce the size of y-axis labels
    
    # Show the plot
    plt.show()


check_missing_values(train_unfilled)
check_missing_values(test_unfilled)


def clean_features(df):
    # Remove highly implausible values

    # Clip Grip
    df[['FGC-FGC_GSND', 'FGC-FGC_GSD']] = df[['FGC-FGC_GSND', 'FGC-FGC_GSD']].clip(lower=9, upper=60)
    # Remove implausible body-fat
    df["BIA-BIA_Fat"] = np.where(df["BIA-BIA_Fat"] < 5, np.nan, df["BIA-BIA_Fat"])
    df["BIA-BIA_Fat"] = np.where(df["BIA-BIA_Fat"] > 60, np.nan, df["BIA-BIA_Fat"])
    # Basal Metabolic Rate
    df["BIA-BIA_BMR"] = np.where(df["BIA-BIA_BMR"] > 4000, np.nan, df["BIA-BIA_BMR"])
    # Daily Energy Expenditure
    df["BIA-BIA_DEE"] = np.where(df["BIA-BIA_DEE"] > 8000, np.nan, df["BIA-BIA_DEE"])
    # Bone Mineral Content
    df["BIA-BIA_BMC"] = np.where(df["BIA-BIA_BMC"] <= 0, np.nan, df["BIA-BIA_BMC"])
    df["BIA-BIA_BMC"] = np.where(df["BIA-BIA_BMC"] > 10, np.nan, df["BIA-BIA_BMC"])
    # Fat Free Mass Index
    df["BIA-BIA_FFM"] = np.where(df["BIA-BIA_FFM"] <= 0, np.nan, df["BIA-BIA_FFM"])
    df["BIA-BIA_FFM"] = np.where(df["BIA-BIA_FFM"] > 300, np.nan, df["BIA-BIA_FFM"])
    # Fat Mass Index
    df["BIA-BIA_FMI"] = np.where(df["BIA-BIA_FMI"] < 0, np.nan, df["BIA-BIA_FMI"])
    # Extra Cellular Water
    df["BIA-BIA_ECW"] = np.where(df["BIA-BIA_ECW"] > 100, np.nan, df["BIA-BIA_ECW"])
    # Intra Cellular Water
    # df["BIA-BIA_ICW"] = np.where(df["BIA-BIA_ICW"] > 100, np.nan, df["BIA-BIA_ICW"])
    # Lean Dry Mass
    df["BIA-BIA_LDM"] = np.where(df["BIA-BIA_LDM"] > 100, np.nan, df["BIA-BIA_LDM"])
    # Lean Soft Tissue
    df["BIA-BIA_LST"] = np.where(df["BIA-BIA_LST"] > 300, np.nan, df["BIA-BIA_LST"])
    # Skeletal Muscle Mass
    df["BIA-BIA_SMM"] = np.where(df["BIA-BIA_SMM"] > 300, np.nan, df["BIA-BIA_SMM"])
    # Total Body Water
    df["BIA-BIA_TBW"] = np.where(df["BIA-BIA_TBW"] > 300, np.nan, df["BIA-BIA_TBW"])
    
    return df

train_unfilled  = clean_features(train_unfilled)
test_unfilled = clean_features(test_unfilled)


print(train_unfilled.shape)
print(test_unfilled.shape)


exclude = ['PCIAT-Season', 'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03',
           'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07',
           'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11',
           'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15',
           'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19',
           'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total', 'sii', 'id']

y_model = "PCIAT-PCIAT_Total" # Score, target for the model
y_comp = "sii" # Index, target of the competition
features = [f for f in train_unfilled.columns if f not in exclude]

numerical_cols = train_unfilled.select_dtypes(include=['number'])
print(numerical_cols)

# 获取数值型列的列名
numerical_col_names = numerical_cols.columns.tolist()

print(len(numerical_col_names))


numerical_features=list(set(numerical_col_names) & set(features))
train_to_fill=train_unfilled.copy()
test_to_fill=test_unfilled.copy()

train_to_fill=train_to_fill[numerical_features]
test_to_fill=test_to_fill[numerical_features]
print(train_to_fill.shape,test_to_fill.shape)


# from sklearn.linear_model import LassoCV
# from sklearn.impute import KNNImputer
# from sklearn.base import clone
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# # 假设你已经有了 train_to_fill 和 test_to_fill 这两个 DataFrame
# # 并且你已经定义了 features 列表 (包含要使用的特征列名)
# # 以及 SEED (随机种子)

# # 如果 features 列表还没有定义，你需要根据你的数据确定要使用的特征列
# # 示例:
# # features = [col for col in train_to_fill.columns if col != 'target_column']

# SEED = 42 # 或者你之前定义的 SEED 值

# class Impute_With_Model:

#     def __init__(self, na_frac=0.2, min_samples=0):
#         self.model_dict = {}
#         self.mean_dict = {}
#         self.features = None
#         self.na_frac = na_frac
#         self.min_samples = min_samples

#     def find_features(self, data, feature, tmp_features):
#         missing_rows = data[feature].isna()
#         na_fraction = data[missing_rows][tmp_features].isna().mean(axis=0)
#         valid_features = np.array(tmp_features)[na_fraction <= self.na_frac]
#         return valid_features

#     def fit_models(self, model, data, features):
#         self.features = features
#         n_data = data.shape[0]
#         for feature in features:
#             self.mean_dict[feature] = np.mean(data[feature])
#         for feature in tqdm(features):
#             if data[feature].isna().sum() > 0:
#                 model_clone = clone(model)
#                 X = data[data[feature].notna()].copy()
#                 tmp_features = [f for f in features if f != feature]
#                 tmp_features = self.find_features(data, feature, tmp_features)
#                 if len(tmp_features) >= 1 and X.shape[0] > self.min_samples:
#                     for f in tmp_features:
#                         X[f] = X[f].fillna(self.mean_dict[f])
#                     model_clone.fit(X[tmp_features], X[feature])
#                     self.model_dict[feature] = (model_clone, tmp_features.copy())
#                 else:
#                     self.model_dict[feature] = ("mean", np.mean(data[feature]))

#     def impute(self, data):
#         imputed_data = data.copy()
#         for feature, model in self.model_dict.items():
#             missing_rows = imputed_data[feature].isna()
#             if missing_rows.any():
#                 if model[0] == "mean":
#                     imputed_data[feature].fillna(model[1], inplace=True)
#                 else:
#                     tmp_features = [f for f in self.features if f != feature]
#                     X_missing = data.loc[missing_rows, tmp_features].copy()
#                     for f in tmp_features:
#                         X_missing[f] = X_missing[f].fillna(self.mean_dict[f])
#                     imputed_data.loc[missing_rows, feature] = model[0].predict(X_missing[model[1]])
#         return imputed_data

# # 初始化模型 (使用 LassoCV 作为示例，你可以替换为你想要使用的模型)
# model = model = LassoCV(cv=5, random_state=SEED, max_iter=10000) # 将默认的迭代次数增加到 1000

# # 初始化 Impute_With_Model 类
# imputer = Impute_With_Model(na_frac=0.2) # 设置 na_frac 参数

# # 拟合 Imputation 模型 (仅在 train_to_fill 上进行拟合)
# imputer.fit_models(model, train_to_fill, numerical_features)

# # 使用拟合好的模型填充 train_to_fill 和 test_to_fill 中的缺失值
# train_filled = imputer.impute(train_to_fill)
# test_filled = imputer.impute(test_to_fill)

# # 现在 train_filled 和 test_filled 是填充了缺失值后的 DataFrame
# print("train_to_fill 填充后的形状:", train_filled.shape)
# print("test_to_fill 填充后的形状:", test_filled.shape)
# print("\ntrain_to_fill 填充后缺失值总数:", train_filled.isna().sum().sum())
# print("test_to_fill 填充后缺失值总数:", test_filled.isna().sum().sum())


# # 初始化模型 (使用 LassoCV 作为示例，你可以替换为你想要使用的模型)
# model = model = LassoCV(cv=5, random_state=SEED, max_iter=10000) # 将默认的迭代次数增加到 1000

# # 初始化 Impute_With_Model 类
# imputer = Impute_With_Model(na_frac=0.4) # 设置 na_frac 参数

# # 拟合 Imputation 模型 (仅在 train_to_fill 上进行拟合)
# imputer.fit_models(model, train_to_fill, numerical_features)

# # 使用拟合好的模型填充 train_to_fill 和 test_to_fill 中的缺失值
# train_filled = imputer.impute(train_to_fill)
# test_filled = imputer.impute(test_to_fill)

# # 现在 train_filled 和 test_filled 是填充了缺失值后的 DataFrame
# print("train_to_fill 填充后的形状:", train_filled.shape)
# print("test_to_fill 填充后的形状:", test_filled.shape)
# print("\ntrain_to_fill 填充后缺失值总数:", train_filled.isna().sum().sum())
# print("test_to_fill 填充后缺失值总数:", test_filled.isna().sum().sum())


# # adjust na_frac

# base_model = LassoCV(cv=5, random_state=42, max_iter=10000)

# # Try different na_frac values
# na_frac_values = [0.2, 0.3, 0.4, 0.5, 0.6]
# filled_results = {}

# # Loop over each na_frac and record how many NaNs are filled
# for na_frac in na_frac_values:
#     print(f"\n Testing na_frac = {na_frac}")
    
#     # Create a fresh imputer instance each time
#     imputer = Impute_With_Model(na_frac=na_frac)
    
#     # Fit the model on the training data
#     imputer.fit_models(clone(base_model), train_to_fill.copy(), numerical_features)
    
#     # Apply imputation
#     filled = imputer.impute(train_to_fill.copy())
    
#     # Count remaining NaNs after filling
#     remaining_nans = filled[numerical_features].isna().sum().sum()
#     total_filled = train_to_fill[numerical_features].isna().sum().sum() - remaining_nans
    
#     filled_results[na_frac] = total_filled
#     print(f" NaNs filled: {total_filled}, Remaining: {remaining_nans}")

# # Show comparison
# print("\n NaNs filled per na_frac:")
# for frac, count in filled_results.items():
#     print(f"na_frac = {frac}: filled {count} values")

# # Find best na_frac
# best_frac = max(filled_results, key=filled_results.get)
# print(f"\n Best na_frac = {best_frac}, filled {filled_results[best_frac]} missing values")



from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone

class Impute_With_Model:
    def __init__(self, na_frac=0.1):
        self.na_frac = na_frac
        self.models = {}

    def fit_models(self, base_model, df, features):
        for col in features:
            missing_ratio = df[col].isna().mean()
            if 0 < missing_ratio <= self.na_frac:
                df_nonnull = df[df[col].notna()]
                X = df_nonnull.drop(columns=[col])
                y = df_nonnull[col]
                model = clone(base_model)
                model.fit(X, y)
                self.models[col] = model

    def impute(self, df):
        df_filled = df.copy()
        for col, model in self.models.items():
            mask = df_filled[col].isna()
            if mask.any():
                X_pred = df_filled.loc[mask].drop(columns=[col])
                try:
                    df_filled.loc[mask, col] = model.predict(X_pred)
                except Exception as e:
                    print(f"Warning: Failed to impute {col} — {e}")
        return df_filled


hgb_model = HistGradientBoostingRegressor(random_state=42)
na_frac = 0.2

print(f"\n=== Imputing with na_frac = {na_frac} ===")
imputer = Impute_With_Model(na_frac=na_frac)
imputer.fit_models(hgb_model, train_to_fill, numerical_features)

train_filled = imputer.impute(train_to_fill)
test_filled = imputer.impute(test_to_fill)

print("Imputation complete")
print("Remaining NaNs in train:", train_filled.isna().sum().sum())
print("Remaining NaNs in test:", test_filled.isna().sum().sum())


train_filled['id']=train_unfilled['id']
test_filled['id']=test_unfilled['id']


categorical_features=list(set(features) - set(numerical_col_names))
print(train_unfilled[categorical_features].dtypes)
train_unfilled_cat=train_unfilled.copy()
test_unfilled_cat=test_unfilled.copy()
train_unfilled_cat=train_unfilled_cat[categorical_features]
test_unfilled_cat=test_unfilled_cat[categorical_features]


# # 假设你已经有了 train_unfilled_cat 和 test_unfilled_cat 这两个 DataFrame

# # 遍历 train_unfilled_cat 的每一列
# for col in train_unfilled_cat.columns:
#     # 计算训练数据该列的众数
#     mode_value = train_unfilled_cat[col].mode()

#     # 检查是否计算出众数 (mode() 返回一个 Series，可能为空)
#     if not mode_value.empty:
#         # 取第一个众数（即使有多个众数，也只取一个来填充）
#         first_mode = mode_value[0]

#         # 使用训练数据的众数填充 train_unfilled_cat 中该列的 NaN (修改后)
#         train_unfilled_cat[col] = train_unfilled_cat[col].fillna(first_mode)

#         # 使用训练数据的众数填充 test_unfilled_cat 中对应列的 NaN (如果该列存在) (修改后)
#         if col in test_unfilled_cat.columns:
#             test_unfilled_cat[col] = test_unfilled_cat[col].fillna(first_mode)
#             print(f"列 '{col}' 的 NaN 已使用训练数据众数 '{first_mode}' 填充。")
#         else:
#             print(f"警告: 测试集中不存在列 '{col}'。")
#     else:
#         print(f"警告: 训练集列 '{col}' 没有众数 (可能所有值都是 NaN 或唯一)。")

# # 验证填充结果 (可选)
# print("\ntrain_unfilled_cat 填充后 NaN 总数:", train_unfilled_cat.isnull().sum().sum())
# print("test_unfilled_cat 填充后 NaN 总数:", test_unfilled_cat.isnull().sum().sum())


# from sklearn.preprocessing import LabelEncoder
# import pandas as pd

# # 假设你已经有了 train_unfilled_cat 和 test_unfilled_cat 这两个 DataFrame

# # 创建 LabelEncoder 的实例
# label_encoder = LabelEncoder()

# # 处理 train_unfilled_cat
# for col in train_unfilled_cat.columns:
#     if train_unfilled_cat[col].dtype == 'object':
#         # 将列中的 NaN 替换为某个占位符，例如 'missing'，以便 LabelEncoder 可以处理
#         train_unfilled_cat[col] = train_unfilled_cat[col].fillna('missing')
#         # 使用 LabelEncoder 对列进行编码
#         train_unfilled_cat[col] = label_encoder.fit_transform(train_unfilled_cat[col])
#         print(f"训练集列 '{col}' 已转换为数值型。")

# # 处理 test_unfilled_cat
# for col in test_unfilled_cat.columns:
#     if col in train_unfilled_cat.columns and test_unfilled_cat[col].dtype == 'object':
#         # 将列中的 NaN 替换为与训练集相同的占位符
#         test_unfilled_cat[col] = test_unfilled_cat[col].fillna('missing')
#         # 注意：这里我们使用在训练集上 fit 过的 LabelEncoder 来 transform 测试集
#         # 这样可以保证相同的类别映射到相同的数字
#         test_unfilled_cat[col] = label_encoder.transform(test_unfilled_cat[col])
#         print(f"测试集列 '{col}' 已转换为数值型。")
#     elif test_unfilled_cat[col].dtype == 'object':
#         print(f"警告: 测试集列 '{col}' 在训练集中不存在，无法进行一致的 Label Encoding。")

# # 检查转换后的数据类型 (可选)
# print("\n训练集转换后数据类型:")
# print(train_unfilled_cat.dtypes)
# print("\n测试集转换后数据类型:")
# print(test_unfilled_cat.dtypes)


# train_unfilled_cat


# train_merged
# test_merged


# train_merged = pd.concat([train_filled, train_unfilled_cat], axis=1)
# test_merged = pd.concat([test_filled, test_unfilled_cat], axis=1)



from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.preprocessing import OrdinalEncoder

class Impute_With_Model:
    def __init__(self, na_frac=0.3, min_samples=20):
        self.na_frac = na_frac
        self.min_samples = min_samples
        self.models = {}

    def fit_models(self, base_model, df, features):
        for col in features:
            missing_ratio = df[col].isna().mean()
            if 0 < missing_ratio <= self.na_frac:
                df_nonnull = df[df[col].notna()]
                if len(df_nonnull) < self.min_samples:
                    print(f"Skipping {col}: not enough samples ({len(df_nonnull)} < {self.min_samples})")
                    continue
                X = df_nonnull.drop(columns=[col])
                y = df_nonnull[col]
                model = clone(base_model)
                model.fit(X, y)
                self.models[col] = model

    def impute(self, df):
        df_filled = df.copy()
        for col, model in self.models.items():
            mask = df_filled[col].isna()
            if mask.any():
                X_pred = df_filled.loc[mask].drop(columns=[col])
                try:
                    df_filled.loc[mask, col] = model.predict(X_pred)
                except Exception as e:
                    print(f"Warning: Failed to impute {col} — {e}")
        return df_filled



train_encoded = train_unfilled_cat.copy()
test_encoded = test_unfilled_cat.copy()

cat_imputer = Impute_With_Model(na_frac=0.3, min_samples=20)
hgb = HistGradientBoostingClassifier(random_state=42)

cat_imputer.fit_models(hgb, train_encoded, categorical_features)

train_encoded = cat_imputer.impute(train_encoded)
test_encoded = cat_imputer.impute(test_encoded)

print("Categorical imputation complete.")
print("Remaining NaNs in train_encoded:", train_encoded.isna().sum().sum())
print("Remaining NaNs in test_encoded:", test_encoded.isna().sum().sum())

cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_encoded[categorical_features] = cat_encoder.fit_transform(train_encoded[categorical_features])
test_encoded[categorical_features] = cat_encoder.transform(test_encoded[categorical_features])


train_merged = pd.concat([train_filled.reset_index(drop=True), train_encoded.reset_index(drop=True)], axis=1)
test_merged = pd.concat([test_filled.reset_index(drop=True), test_encoded.reset_index(drop=True)], axis=1)

print("Merged dataset ready.")
print("train_merged shape:", train_merged.shape)
print("test_merged shape:", test_merged.shape)



# # Re-encode categorical features
# train_encoded = train_cat_filled.copy()
# test_encoded = test_cat_filled.copy()

# cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# train_encoded[categorical_features] = cat_encoder.fit_transform(train_encoded[categorical_features])
# test_encoded[categorical_features] = cat_encoder.transform(test_encoded[categorical_features])

# Final merge for model input
# train_merged = pd.concat([train_filled, train_encoded], axis=1)
# test_merged = pd.concat([test_filled, test_encoded], axis=1)
# train_merged


check_missing_values(train_merged)


check_missing_values(test_merged)


# checkpoint: able to rerun from here onwards
train_fe = train_unfilled.copy()
test_fe = test_unfilled.copy()


def feature_engineering(df):
    new_features = pd.DataFrame({
        "BP_HeartRate": df["Physical-Systolic_BP"] * df["Physical-HeartRate"],  # BP & Heart Rate Interaction
        "BFP_BMI": df["BIA-BIA_Fat"] / df["BIA-BIA_BMI"],  # Body Fat Percentage to BMI
        "LST_TBW": df["BIA-BIA_LST"] / df["BIA-BIA_BMI"], # Lean Mass to Total Body Water
        "DEE_Weight": df["BIA-BIA_DEE"] / df["Physical-Weight"],  # Daily Energy Expenditure per Weight
        "Fitness_Index": (df["Fitness_Endurance-Max_Stage"] + df["Fitness_Endurance-Time_Mins"]) / 2,  # Fitness Performance
        "Hydration_Status": df["BIA-BIA_TBW"] / df["Physical-Weight"],  # Hydration Relative to Weight
        "Internet_to_Activity_Ratio": df["PreInt_EduHx-computerinternet_hoursday"] / (df["PAQ_C-PAQ_C_Total"] + 1),  # Internet vs Activity
        "Screen_Sleep_Impact": df["PreInt_EduHx-computerinternet_hoursday"] / (df["SDS-SDS_Total_T"] + 1),  # Screen Time vs Sleep
        "BP_Variability": df["Physical-Systolic_BP"] - df["Physical-Diastolic_BP"],  # BP Variability
        "HeartRate_BMI": df["Physical-HeartRate"] / (df["Physical-BMI"] + 1),  # Heart Rate to BMI
        "BP_Weight_Ratio": (df["Physical-Systolic_BP"] + df["Physical-Diastolic_BP"]) / df["Physical-Weight"],  # BP to Weight Ratio
        "BMR_Age_Adjusted": df["BIA-BIA_BMR"] / (df["Basic_Demos-Age"] + 1),  # BMR Efficiency Adjusted for Age
        "Muscle_to_Fat_Index": df["BIA-BIA_FFMI"] / (df["BIA-BIA_FMI"] + 1),  # FFMI to FMI Ratio
        "Water_to_Muscle": df["BIA-BIA_TBW"] / (df["BIA-BIA_SMM"] + 1),  # Water-to-Muscle Ratio
        "Hydration_Index": df["BIA-BIA_ICW"] / (df["BIA-BIA_ECW"] + 1),  # Intracellular vs Extracellular Water
        "Winter_Enroll": (df["Basic_Demos-Enroll_Season"] == "Winter").astype(int),  # Winter Enrollment Indicator
        "Seasonal_Physical_Activity": df["PAQ_C-PAQ_C_Total"] * (df["Basic_Demos-Enroll_Season"] == "Winter").astype(int)  # Seasonal Activity Impact
    })

    # Concatenate the new features in a single operation
    df = pd.concat([df, new_features], axis=1)

    # Add new feature names to the numerical_features list only if they don't already exist
    global numerical_features
    for feature in new_features.columns:
        if feature not in numerical_features:
            numerical_features.append(feature)

    return df

# Apply feature engineering
train_fe = feature_engineering(train_merged)
test_fe = feature_engineering(test_merged)


def print_first_10_rows_with_missing_or_inf(df):
    """
    Prints only the first 10 rows where any column contains NaN or Inf values.
    Ensures all columns are displayed in a properly formatted table.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to check for NaN or Inf values.
    """
    # Enable full display of all columns
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Prevent truncation
    pd.set_option('display.float_format', '{:.3f}'.format)  # Format float values nicely

    # Filter rows where any column has NaN or Inf values
    rows_with_issues = df[df.isna().any(axis=1) | df.isin([np.inf, -np.inf]).any(axis=1)]

    # Check if there are any problematic rows
    if rows_with_issues.empty:
        print("✅ No NaN or Inf values found in the DataFrame.")
        return

    # Limit to first 10 rows
    print(f"⚠️ Found {len(rows_with_issues)} rows with NaN or Inf values (showing first 10):")
    display(rows_with_issues.head(10))  # Show only first 10 rows


print_first_10_rows_with_missing_or_inf(train_fe)



print_first_10_rows_with_missing_or_inf(test_fe)



print(train_fe.shape)


print(test_fe.shape)


import lightgbm as lgb
import pandas as pd

# 假设你已经有了 train_fe 和 test_fe 这两个 DataFrame (预测特征)
# 并且你已经有了 train_unfilled DataFrame，其中包含目标变量 "PCIAT-PCIAT_Total"

# 确保目标变量在 train_unfilled 中存在
if "PCIAT-PCIAT_Total" not in train_unfilled.columns:
    raise ValueError("目标变量 'PCIAT-PCIAT_Total' 不存在于 train_unfilled DataFrame 中。")

# 定义特征和目标变量
X_train = train_fe.drop(columns='id')
y_train = train_unfilled["PCIAT-PCIAT_Total"]
X_test = test_fe.drop(columns='id')

# 初始化 LightGBM 回归模型
lgbm = lgb.LGBMRegressor(random_state=42) # 你可以根据需要调整模型的参数

# 训练模型，并添加评估
lgbm.fit(X_train, y_train,
         eval_set=[(X_train, y_train)],  # 使用训练集作为评估集
         eval_metric='rmse',             # 使用均方根误差作为评估指标 (你可以选择其他回归指标)
         )                     # 设置为大于 0 可以显示评估结果

# 在测试集上进行预测
predictions = lgbm.predict(X_test)

# 将预测结果转换为 DataFrame
predictions_df = pd.DataFrame({'PCIAT-PCIAT_Total_Predicted': predictions})

# 打印预测结果 DataFrame
print("\n测试集 'PCIAT-PCIAT_Total' 的预测结果:")
print(predictions_df)


thresholds = [30, 50, 80]

def convert_score_to_sii(score, thresholds):
    """根据阈值将 PCIAT-PCIAT_Total 分数转换为 sii 值。"""
    if score < thresholds[0]:
        return 0
    elif score < thresholds[1]:
        return 1
    elif score < thresholds[2]:
        return 2
    else:
        return 3

# 将预测的 PCIAT-PCIAT_Total 分数转换为 sii 值
predictions_df['sii_Predicted'] = predictions_df['PCIAT-PCIAT_Total_Predicted'].apply(lambda x: convert_score_to_sii(x, thresholds))

# 打印包含预测的 PCIAT-PCIAT_Total 和 sii 的 DataFrame
print("\n包含预测的 PCIAT-PCIAT_Total 和 sii 的 DataFrame (前 5 行):")
print(predictions_df)


def convert_to_sii_categories(scores, thresholds=[30, 50, 80]):
    scores = np.array(scores)
    return np.digitize(scores, thresholds)



def train_xgb_classifier(train_X, test_X, train_y_raw, thresholds=[30, 50, 80]):
    """Train XGBoost with 5-fold CV and return predictions and feature importances."""
    y = convert_to_sii_categories(train_y_raw, thresholds)
    X = train_X.drop(columns='id', errors='ignore')
    test_X_ = test_X.drop(columns='id', errors='ignore')

    print(f"Training on {X.shape[0]} samples, Testing on {test_X_.shape[0]} samples.")
    print("SII category distribution:")
    for i, count in enumerate(np.bincount(y)):
        print(f"  Category {i}: {count} samples ({100 * count / len(y):.1f}%)")

    params = {
        'objective': 'multi:softprob', 'num_class': 4, 'learning_rate': 0.02,
        'max_depth': 6, 'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'n_estimators': 500,
        'random_state': 42, 'verbosity': 0
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros((test_X_.shape[0], 4))
    importances = np.zeros(X.shape[1])
    acc_scores = []

    for i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {i+1}")
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[tr_idx], y[tr_idx])

        preds_val = model.predict(X.iloc[val_idx])
        acc = accuracy_score(y[val_idx], preds_val)
        acc_scores.append(acc)
        print(f"  Accuracy: {acc:.4f}")
        if i == 0:
            print(classification_report(y[val_idx], preds_val))

        test_preds += model.predict_proba(test_X_) / kf.n_splits
        importances += model.feature_importances_

    final_preds = np.argmax(test_preds, axis=1)
    feature_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances / kf.n_splits
    }).sort_values('Importance', ascending=False)

    print("\nTop features:")
    print(feature_df.head(10))

    print(f"\nAverage CV Accuracy: {np.mean(acc_scores):.4f}")
    print("Test prediction distribution:")
    for i, c in enumerate(np.bincount(final_preds)):
        print(f"  Category {i}: {c} samples ({100 * c / len(final_preds):.1f}%)")

    submission = pd.DataFrame({
        'sii_Predicted': final_preds
    })
    if 'id' in test_X.columns:
        submission.insert(0, 'id', test_X['id'].values)
    else:
        submission.insert(0, 'id', np.arange(len(final_preds)))  # fallback

    return submission, np.mean(acc_scores), feature_df


def find_best_thresholds(train_fe, train_unfilled, num_combinations=5):
    """Test multiple thresholds and return the best one."""
    sample_idx = np.random.choice(train_fe.index, size=int(0.2 * len(train_fe)), replace=False)
    sample_X = train_fe.loc[sample_idx]
    sample_y = train_unfilled.loc[sample_idx]["PCIAT-PCIAT_Total"]

    candidate_thresholds = [
        [30, 50, 80], [35, 55, 75], [25, 45, 70],
        [28, 48, 78], [33, 53, 83]
    ]
    best_acc, best_t = 0, candidate_thresholds[0]
    for t in candidate_thresholds:
        print(f"\nTrying thresholds: {t}")
        _, acc, _ = train_xgb_classifier(sample_X, sample_X, sample_y, thresholds=t)
        if acc > best_acc:
            best_acc, best_t = acc, t
            print(f"  New best accuracy: {acc:.4f}")
    print(f"\nBest thresholds: {best_t} with accuracy: {best_acc:.4f}")
    return best_t, best_acc



def predict_sii_with_xgboost(train_fe, test_fe, train_unfilled, find_thresholds=False):
    """
    Predict SII using XGBoost, optionally tuning thresholds.
    """
    if "PCIAT-PCIAT_Total" not in train_unfilled.columns:
        raise ValueError("Missing 'PCIAT-PCIAT_Total' in training target DataFrame.")
    
    if find_thresholds:
        thresholds, _ = find_best_thresholds(train_fe, train_unfilled)
    else:
        thresholds = [30, 50, 80]

    print(f"\nTraining final model using thresholds: {thresholds}")
    submission_df, accuracy, importance_df = train_xgb_classifier(
        train_fe, test_fe, train_unfilled["PCIAT-PCIAT_Total"], thresholds
    )
    return submission_df, accuracy, importance_df



from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

submission, accuracy, importance = predict_sii_with_xgboost(
    train_fe=train_fe,
    test_fe=test_fe,
    train_unfilled=train_unfilled,
    find_thresholds=True  # or False if you want to skip threshold tuning
)

submission.columns = ['id', 'sii']
print("Submission columns:", submission.columns.tolist())

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission saved to /kaggle/working/submission.csv")



submission.head()



# my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# # you could use any filename. We choose submission here
# my_submission.to_csv('submission-SVR_SukulAdisak-4features.csv', index=False)



submission_df = pd.concat([test_unfilled['id'], predictions_df['sii_Predicted']], axis=1)

# 9. Save the submission file
submission_path = '/kaggle/working/submission.csv'
submission_df.to_csv(submission_path, index=False)

