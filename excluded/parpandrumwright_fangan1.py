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


!pwd


cd ../input/equity-post-HCT-survival-predictions/


!pwd


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# 加载数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data_dict = pd.read_csv('data_dictionary.csv')

# 查看数据
print(train.head())
print(test.head())


# import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np

# # 加载数据
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# # 查看数据
# print("Train Data Head:")
# print(train.head())
# print("\nTest Data Head:")
# print(test.head())

# 计算缺失值比例
def calculate_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values_percentage = (missing_values / len(df)) * 100
    return missing_values_percentage

# 计算训练集和测试集的缺失值比例
train_missing = calculate_missing_values(train)
test_missing = calculate_missing_values(test)

# 打印缺失值比例
def print_colored_missing_values(missing_values, threshold_1=0, threshold_2=30):
    # 按缺失比例降序排列
    missing_values_sorted = missing_values.sort_values(ascending=False)
    
    for feature, percentage in missing_values_sorted.items():
        if percentage == 0:
            print(f"\033[92m{feature}: {percentage:.2f}%\033[0m")  # 绿色
        elif percentage > threshold_2:
            print(f"\033[91m{feature}: {percentage:.2f}%\033[0m")  # 红色
        else:
            print(f"\033[93m{feature}: {percentage:.2f}%\033[0m")  # 亮黄色

print("\nTrain Data Missing Values Percentage:")
print_colored_missing_values(train_missing)
print("\nTest Data Missing Values Percentage:")
print_colored_missing_values(test_missing)

# 绘制缺失值比例的柱状图
def plot_missing_values(missing_values, title):
    # 按缺失比例降序排列
    missing_values_sorted = missing_values.sort_values(ascending=False)
    
    # 创建颜色映射
    colors = plt.cm.RdYlGn(np.interp(missing_values_sorted, 
                                     [missing_values_sorted.min(), missing_values_sorted.max()], 
                                     [1, 0]))  # 0 是绿色，1 是红色
    
    plt.figure(figsize=(10, 12))
    bars = plt.barh(missing_values_sorted.index, missing_values_sorted.values, color=colors)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Missing Values Percentage', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    
    # 增大纵轴标签间距
    plt.yticks(ticks=np.arange(len(missing_values_sorted.index)), labels=missing_values_sorted.index, fontsize=12)
    plt.grid(axis='x')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=missing_values_sorted.min(), vmax=missing_values_sorted.max()))
    sm.set_array([])
    plt.colorbar(sm, orientation='vertical', label='Missing Values Percentage')
    
    plt.show()

# 绘制训练集和测试集的缺失值比例
plot_missing_values(train_missing, 'Train Data Missing Values Percentage')
plot_missing_values(test_missing, 'Test Data Missing Values Percentage')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
train = pd.read_csv('train.csv')

# 选择连续变量和分类变量
continuous_vars = train.select_dtypes(include=['int64', 'float64']).columns
categorical_vars = train.select_dtypes(include=['object', 'category']).columns

# 计算连续变量的相关性矩阵
continuous_corr = train[continuous_vars].corr()

# 绘制连续变量的相关性热图
plt.figure(figsize=(12, 8))
sns.heatmap(continuous_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Continuous Variables', fontsize=16)
plt.show()

# 对分类变量进行编码
train_encoded = train[categorical_vars].apply(lambda col: col.astype('category').cat.codes)

# 计算分类变量的相关性矩阵
categorical_corr = train_encoded.corr()

# 绘制分类变量的相关性热图
plt.figure(figsize=(12, 8))
sns.heatmap(categorical_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Categorical Variables', fontsize=16)
plt.show()


# # 分离特征和目标变量
# X = train.drop(columns=['ID', 'efs', 'efs_time'])
# y = train['efs']

# # 定义数值型和类别型特征
# numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = X.select_dtypes(include=['object', 'category']).columns

# # 定义预处理管道
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # 应用预处理
# X_processed = preprocessor.fit_transform(X)
# test_processed = preprocessor.transform(test.drop(columns=['ID']))


# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.experimental import enable_iterative_imputer  # 启用 IterativeImputer
# from sklearn.impute import IterativeImputer
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# # 加载数据
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# # 分离特征和目标变量
# X = train.drop(columns=['ID', 'efs', 'efs_time'])
# y = train['efs']

# # 计算缺失值比例
# missing_ratio = X.isnull().mean()

# # 筛选缺失比例小于30%的特征
# features_to_keep = missing_ratio[missing_ratio < 0.3].index
# X = X[features_to_keep]
# test = test[features_to_keep]

# # 定义数值型和类别型特征
# numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = X.select_dtypes(include=['object', 'category']).columns

# # 定义预处理管道
# # 对于数值型特征，使用基于随机森林的迭代填充方法
# numeric_transformer = Pipeline(steps=[
#     ('imputer', IterativeImputer(estimator=RandomForestRegressor(random_state=0), max_iter=10, random_state=0)),
#     ('scaler', StandardScaler())
# ])

# # 对于类别型特征，使用最频繁值填充
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # 应用预处理
# X_processed = preprocessor.fit_transform(X)
# test_processed = preprocessor.transform(test.drop(columns=['ID']))


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # 启用 IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# # 加载数据
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# # 分离特征和目标变量
# X = train.drop(columns=['ID', 'efs', 'efs_time'])
# y = train['efs']

# 计算缺失值比例
missing_ratio = X.isnull().mean()

# 筛选缺失比例小于30%的特征
features_to_keep = missing_ratio[missing_ratio < 0.3].index
X = X[features_to_keep]
test = test[features_to_keep]

# 定义数值型和类别型特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# 定义预处理管道
# 对于数值型特征，使用基于随机森林的迭代填充方法
numeric_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(estimator=RandomForestRegressor(n_jobs=-1, random_state=0), max_iter=10, random_state=0)),
    ('scaler', StandardScaler())
])

# 对于类别型特征，使用最频繁值填充
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 应用预处理
X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test.drop(columns=['ID']))


# !pip install cuml-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
# !pip show cuml-cu11 


# import cudf
# import cuml
# from cuml.impute import SimpleImputer as cuSimpleImputer
# from cuml.preprocessing import StandardScaler as cuStandardScaler
# from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
# from cuml.impute import IterativeImputer as cuIterativeImputer

# # 加载数据到cuDF DataFrame
# train_cudf = cudf.read_csv('train.csv')
# test_cudf = cudf.read_csv('test.csv')

# # 分离特征和目标变量
# X_cudf = train_cudf.drop(columns=['ID', 'efs', 'efs_time'])
# y_cudf = train_cudf['efs']

# # 计算缺失值比例
# missing_ratio = X_cudf.isnull().mean()

# # 筛选缺失比例小于30%的特征
# features_to_keep = missing_ratio[missing_ratio < 0.3].index
# X_cudf = X_cudf[features_to_keep]
# test_cudf = test_cudf[features_to_keep]

# # 定义数值型和类别型特征
# numeric_features = X_cudf.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = X_cudf.select_dtypes(include=['object', 'category']).columns

# # 定义预处理管道
# # 对于数值型特征，使用基于随机森林的迭代填充方法
# numeric_transformer = Pipeline(steps=[
#     ('imputer', cuIterativeImputer(estimator=cuRandomForestRegressor(random_state=0), max_iter=10, random_state=0)),
#     ('scaler', cuStandardScaler())
# ])

# # 对于类别型特征，使用最频繁值填充
# categorical_transformer = Pipeline(steps=[
#     ('imputer', cuSimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # 应用预处理
# X_processed = preprocessor.fit_transform(X_cudf)
# test_processed = preprocessor.transform(test_cudf.drop(columns=['ID']))


# 获取处理后的特征名
feature_names = preprocessor.get_feature_names_out()

# 将处理后的数据转换为 DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# 计算相关性矩阵
correlation_matrix = X_processed_df.corr()

# 绘制相关性热图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()


# # 绘制连续变量的相关性热图
# plt.figure(figsize=(240, 160))
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
# plt.title('Correlation Heatmap of Continuous Variables', fontsize=16)
# plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 获取处理后的特征名
feature_names = preprocessor.get_feature_names_out()

# 将处理后的数据转换为 DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# 计算相关性矩阵
correlation_matrix = X_processed_df.corr()

# 区分连续变量和分类变量
numeric_features = [col for col in feature_names if col.startswith('num__')]
categorical_features = [col for col in feature_names if col.startswith('cat__')]

# 选择连续变量的相关性矩阵
numeric_correlation_matrix = correlation_matrix.loc[numeric_features, numeric_features]

# 选择分类变量的相关性矩阵
categorical_correlation_matrix = correlation_matrix.loc[categorical_features, categorical_features]

# 绘制连续变量的相关性热图
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Heatmap for Numeric Features')
plt.show()

# # 绘制分类变量的相关性热图
# plt.figure(figsize=(12, 10))
# sns.heatmap(categorical_correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', center=0)
# plt.title('Correlation Heatmap for Categorical Features')
# plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 获取处理后的特征名
feature_names = preprocessor.get_feature_names_out()

# 将处理后的数据转换为 DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# 区分连续变量和分类变量
numeric_features = [col for col in feature_names if col.startswith('num__')]
categorical_features = [col for col in feature_names if col.startswith('cat__')]

# 绘制分类变量的分布
plt.figure(figsize=(12, 8))
X_processed_df[categorical_features].sum().plot(kind='bar')
plt.title('Distribution of Categorical Features')
plt.xlabel('Features')
plt.ylabel('Count')
plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 获取处理后的特征名
# feature_names = preprocessor.get_feature_names_out()

# # 将处理后的数据转换为 DataFrame
# X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# # 将目标变量 y 添加到 DataFrame 中
# X_processed_df['outcome'] = y

# # 区分连续变量和分类变量
# numeric_features = [col for col in feature_names if col.startswith('num__')]
# categorical_features = [col for col in feature_names if col.startswith('cat__')]

# # 设置图形大小
# plt.figure(figsize=(18, 4 * len(categorical_features)))

# # 绘制每个分类变量与结局的分布
# for i, feature in enumerate(categorical_features, 1):
#     plt.subplot(len(categorical_features), 1, i)
#     sns.countplot(x=feature, hue='outcome', data=X_processed_df)
#     plt.title(f'Distribution of {feature} by Outcome')
#     plt.xlabel(feature)
#     plt.ylabel('Count')

# # 调整布局
# plt.tight_layout()
# plt.show()


from scipy.stats import chi2_contingency

# 创建一个空的 DataFrame 来存储卡方检验的结果
chi2_results = pd.DataFrame(index=categorical_features, columns=categorical_features)

# 计算每对分类变量之间的卡方检验
for i in range(len(categorical_features)):
    for j in range(i+1, len(categorical_features)):
        contingency_table = pd.crosstab(X_processed_df[categorical_features[i]], X_processed_df[categorical_features[j]])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        chi2_results.loc[categorical_features[i], categorical_features[j]] = p
        chi2_results.loc[categorical_features[j], categorical_features[i]] = p

# 填充对角线为1（自身关联）
for feature in categorical_features:
    chi2_results.loc[feature, feature] = 1

# 绘制卡方检验结果的热图
plt.figure(figsize=(12, 10))
sns.heatmap(chi2_results.astype(float), annot=False, fmt=".2f", cmap='coolwarm', center=0.5)
plt.title('Chi-Square Test Results for Categorical Features')
plt.show()


test.columns


# test_processed = preprocessor.transform(test.drop(columns=['ID']))


# test_processed = preprocessor.transform(test)



test.columns





# 获取处理后的特征名
# 使用 ColumnTransformer 的 get_feature_names_out 方法
feature_names = preprocessor.get_feature_names_out()

# 将处理后的测试数据转换为 DataFrame
test_processed_df = pd.DataFrame(test_processed, columns=feature_names)

# 查看处理后的测试数据的特征名
print("处理后的测试数据的特征名：")
print(test_processed_df.columns)


# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.experimental import enable_iterative_imputer  # 启用 IterativeImputer
# from sklearn.impute import IterativeImputer
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from tqdm.auto import tqdm  # 导入 tqdm

# # 加载数据
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# # 分离特征和目标变量
# X = train.drop(columns=['ID', 'efs', 'efs_time'])
# y = train['efs']

# # 计算缺失值比例
# missing_ratio = X.isnull().mean()

# # 筛选缺失比例小于30%的特征
# features_to_keep = missing_ratio[missing_ratio < 0.3].index
# X = X[features_to_keep]
# test = test[features_to_keep]

# # 定义数值型和类别型特征
# numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = X.select_dtypes(include=['object', 'category']).columns

# # 定义预处理管道
# # 对于数值型特征，使用基于随机森林的迭代填充方法
# numeric_transformer = Pipeline(steps=[
#     ('imputer', IterativeImputer(estimator=RandomForestRegressor(random_state=0), max_iter=10, random_state=0)),
#     ('scaler', StandardScaler())
# ])

# # 对于类别型特征，使用最频繁值填充
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # 应用预处理
# print("开始预处理训练数据...")
# X_processed = preprocessor.fit_transform(X)
# print("训练数据预处理完成。")

# print("开始预处理测试数据...")
# test_processed = preprocessor.transform(test.drop(columns=['ID']))
# print("测试数据预处理完成。")

# # 如果需要在迭代填充时显示进度条，可以修改 IterativeImputer 的代码
# # 但 sklearn 的 IterativeImputer 本身不支持直接显示进度条
# # 可以通过自定义一个进度条来实现
# class TqdmIterativeImputer(IterativeImputer):
#     def fit(self, X, y=None):
#         with tqdm(total=self.max_iter, desc="迭代填充进度") as pbar:
#             for i in range(self.max_iter):
#                 super().fit(X, y)
#                 pbar.update(1)
#         return self

# # 使用自定义的 TqdmIterativeImputer 替代原来的 IterativeImputer
# numeric_transformer = Pipeline(steps=[
#     ('imputer', TqdmIterativeImputer(estimator=RandomForestRegressor(random_state=0), max_iter=10, random_state=0)),
#     ('scaler', StandardScaler())
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # 重新应用预处理
# print("开始预处理训练数据...")
# X_processed = preprocessor.fit_transform(X)
# print("训练数据预处理完成。")

# print("开始预处理测试数据...")
# test_processed = preprocessor.transform(test.drop(columns=['ID']))
# print("测试数据预处理完成。")


import pandas as pd

# 假设 X_processed 和 test_processed 是处理后的数据
# 由于 ColumnTransformer 的输出是 numpy 数组，我们需要将其转换回 DataFrame

# 获取处理后的特征名
feature_names = preprocessor.get_feature_names_out()

# 将处理后的数据转换为 DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
test_processed_df = pd.DataFrame(test_processed, columns=feature_names)

# 检查训练数据中的缺失值
print("训练数据中的缺失值统计：")
print(X_processed_df.isnull().sum())

# 检查测试数据中的缺失值
print("\n测试数据中的缺失值统计：")
print(test_processed_df.isnull().sum())

# 检查是否有任何缺失值
print("\n训练数据中是否存在缺失值：", X_processed_df.isnull().values.any())
print("测试数据中是否存在缺失值：", test_processed_df.isnull().values.any())


import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

# 假设 X_processed 是经过预处理后的特征数据，y 是目标变量
# 如果您已经有了 X_processed 和 y，可以直接使用

# 将数据分为训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 使用 LassoCV 进行变量筛选
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train, y_train)

# 获取 Lasso 回归的系数
lasso_coef = lasso_cv.coef_

# 获取特征名
feature_names = preprocessor.get_feature_names_out()

# 将系数与特征名对应起来
lasso_results = pd.DataFrame({'Feature': feature_names, 'Coefficient': lasso_coef})

# 筛选出系数不为 0 的特征
selected_features = lasso_results[lasso_results['Coefficient'] != 0]

# 输出筛选后的特征
print("Selected features by Lasso Regression:")
print(selected_features)

# 如果需要，可以将筛选后的特征名提取出来
selected_feature_names = selected_features['Feature'].values

# 获取筛选后的特征在原始数据中的列索引
selected_indices = [list(feature_names).index(name) for name in selected_feature_names]

# 使用筛选后的特征重新构建训练集和测试集
X_train_selected = X_train[:, selected_indices]
X_val_selected = X_val[:, selected_indices]

# 现在可以使用筛选后的特征进行后续的模型训练和评估


import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

# 假设 X_processed 是经过预处理后的特征数据，y 是目标变量
# 如果您已经有了 X_processed 和 y，可以直接使用

# 将数据分为训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 使用 LassoCV 进行变量筛选，设置更严格的正则化参数
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, -0.5, 30))
lasso_cv.fit(X_train, y_train)

# 获取 Lasso 回归的系数
lasso_coef = lasso_cv.coef_

# 获取特征名
feature_names = preprocessor.get_feature_names_out()

# 将系数与特征名对应起来
lasso_results = pd.DataFrame({'Feature': feature_names, 'Coefficient': lasso_coef})

# 筛选出系数不为 0 的特征
selected_features = lasso_results[lasso_results['Coefficient'] != 0]

# 输出筛选后的特征
print("Selected features by Lasso Regression:")
print(selected_features)

# 如果需要，可以将筛选后的特征名提取出来
selected_feature_names = selected_features['Feature'].values

# 获取筛选后的特征在原始数据中的列索引
selected_indices = [list(feature_names).index(name) for name in selected_feature_names]

# 使用筛选后的特征重新构建训练集和测试集
X_train_selected = X_train[:, selected_indices]
X_val_selected = X_val[:, selected_indices]

# 现在可以使用筛选后的特征进行后续的模型训练和评估


import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 假设 X_processed 是经过预处理后的特征数据，y 是目标变量
# 如果您已经有了 X_processed 和 y，可以直接使用

# 将数据分为训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 使用 LassoCV 进行变量筛选
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train, y_train)

# 获取 Lasso 回归的系数
lasso_coef = lasso_cv.coef_

# 获取特征名
feature_names = preprocessor.get_feature_names_out()

# 将系数与特征名对应起来
lasso_results = pd.DataFrame({'Feature': feature_names, 'Coefficient': lasso_coef})

# 计算系数的绝对值
lasso_results['Abs_Coefficient'] = np.abs(lasso_results['Coefficient'])

# 筛选出系数不为 0 的特征
selected_features = lasso_results[lasso_results['Coefficient'] != 0]

# 输出筛选后的特征
print("Selected features by Lasso Regression:")
print(selected_features)

# 如果需要，可以将筛选后的特征名提取出来
selected_feature_names = selected_features['Feature'].values

# 获取筛选后的特征在原始数据中的列索引
selected_indices = [list(feature_names).index(name) for name in selected_feature_names]

# 使用筛选后的特征重新构建训练集和测试集
X_train_selected = X_train[:, selected_indices]
X_val_selected = X_val[:, selected_indices]

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_selected, y_train)

# 获取随机森林的特征重要性
rf_importances = pd.DataFrame({'Feature': selected_feature_names, 'Importance': rf.feature_importances_})

# 将 Lasso 回归的系数与随机森林的特征重要性合并
combined_features = pd.merge(lasso_results, rf_importances, on='Feature', how='inner')

# 计算综合重要性（可以自定义权重）
combined_features['Combined_Importance'] = combined_features['Abs_Coefficient'] + combined_features['Importance']

# 按照综合重要性降序排列
combined_features_sorted = combined_features.sort_values(by='Combined_Importance', ascending=False)

# 选择综合重要性排名前 N 的特征
N = 40  # 选择排名前 20 的特征
selected_features = combined_features_sorted.head(N)

# 输出筛选后的特征
print("Selected features by Combined Importance:")
print(selected_features)

# 如果需要，可以将筛选后的特征名提取出来
selected_feature_names = selected_features['Feature'].values

# 获取筛选后的特征在原始数据中的列索引
selected_indices = [list(feature_names).index(name) for name in selected_feature_names]

# 使用筛选后的特征重新构建训练集和测试集
X_train_selected = X_train[:, selected_indices]
X_val_selected = X_val[:, selected_indices]

# 现在可以使用筛选后的特征进行后续的模型训练和评估


print(selected_feature_names)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 假设您已经得到了筛选后的特征名 selected_feature_names
# 假设 X_processed 是经过预处理后的特征数据，y 是目标变量

# 获取处理后的特征名
feature_names = preprocessor.get_feature_names_out()

# 将处理后的数据转换为 DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# 使用筛选后的特征重新构建训练集和测试集
X_processed_selected = X_processed_df[selected_feature_names].values

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_processed_selected, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 验证集评估
y_pred = rf.predict_proba(X_val)[:, 1]
print(f'Validation ROC AUC: {roc_auc_score(y_val, y_pred)}')


# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 验证集评估
y_pred = rf.predict_proba(X_val)[:, 1]
print(f'Validation ROC AUC: {roc_auc_score(y_val, y_pred)}')


# # XGBoost模型
# xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
# print(f'XGBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_xgb)}')

# # LightGBM模型
# lgb_model = lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=42)
# lgb_model.fit(X_train, y_train)
# y_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
# print(f'LightGBM Validation ROC AUC: {roc_auc_score(y_val, y_pred_lgb)}')




# # 指定一个已存在的目录作为 train_dir
# train_dir = '/kaggle/working/catboost_info'

# # 确保目录存在
# import os
# if not os.path.exists(train_dir):
#     os.makedirs(train_dir)

# # 使用指定的 train_dir
# cb_model = cb.CatBoostClassifier(random_state=42, verbose=0, train_dir=train_dir)
# cb_model.fit(X_train, y_train)
# y_pred_cb = cb_model.predict_proba(X_val)[:, 1]
# print(f'CatBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_cb)}')


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import roc_auc_score
# import xgboost as xgb
# import lightgbm as lgb
# import catboost as cb

# # 加载数据
# train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
# test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# # 分离特征和目标变量
# X = train.drop(columns=['ID', 'efs', 'efs_time'])
# y = train['efs']

# # 定义数值型和类别型特征
# numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = X.select_dtypes(include=['object', 'category']).columns

# # 定义预处理管道
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # 应用预处理
# X_processed = preprocessor.fit_transform(X)
# test_processed = preprocessor.transform(test.drop(columns=['ID']))

# # 划分训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 指定一个已存在的目录作为 train_dir
train_dir = '/kaggle/working/catboost_info'

# 确保目录存在
import os
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

# XGBoost模型
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
print(f'XGBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_xgb)}')

# LightGBM模型
lgb_model = lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
print(f'LightGBM Validation ROC AUC: {roc_auc_score(y_val, y_pred_lgb)}')

# CatBoost模型
cb_model = cb.CatBoostClassifier(random_state=42, verbose=0, train_dir=train_dir)
cb_model.fit(X_train, y_train)
y_pred_cb = cb_model.predict_proba(X_val)[:, 1]
print(f'CatBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_cb)}')


# 使用CPU用时一分钟
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', random_state=42),
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=5,
                           verbose=1)

grid_search.fit(X_train, y_train)
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best ROC AUC: {grid_search.best_score_}')

# 使用最佳参数重新训练
best_xgb_model = grid_search.best_estimator_
y_pred_best_xgb = best_xgb_model.predict_proba(X_val)[:, 1]
print(f'Best XGBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_best_xgb)}')





import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 400],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0.01, 0.1, 0.5],
    'reg_lambda': [0.1, 0.5, 1.0]
}

# 初始化XGBoost分类器，使用GPU
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42, tree_method='gpu_hist', gpu_id=0, eval_metric='logloss')

# 使用GridSearchCV进行超参数搜索
grid_search = GridSearchCV(estimator=xgb_clf,
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)  # 使用所有可用的CPU核心加速搜索

# 拟合训练数据
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳ROC AUC分数
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best ROC AUC: {grid_search.best_score_}')

# 使用最佳参数重新训练模型
best_xgb_model = grid_search.best_estimator_

# 在验证集上进行预测
y_pred_best_xgb = best_xgb_model.predict_proba(X_val)[:, 1]

# 计算验证集上的ROC AUC分数
print(f'Best XGBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_best_xgb)}')





# 模型融合
from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(estimators=[
    ('rf', rf),
    ('xgb', best_xgb_model),
    ('lgb', lgb_model),
    ('cb', cb_model)
], voting='soft')

ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict_proba(X_val)[:, 1]
print(f'Ensemble Validation ROC AUC: {roc_auc_score(y_val, y_pred_ensemble)}')


# 使用最佳模型对测试集进行预测
test_predictions = ensemble_model.predict_proba(test_processed)[:, 1]

# 生成提交文件
submission = pd.DataFrame({'ID': test['ID'], 'prediction': test_predictions})

# 确保保存到 /kaggle/working/ 目录
submission.to_csv('/kaggle/working/submission.csv', index=False)

print('Submission file saved as /kaggle/working/submission.csv')


# 计算分层一致性指数（Stratified C-index）
from sklearn.metrics import roc_auc_score

# 假设我们有一个函数来计算分层C-index
def stratified_c_index(y_true, y_pred, race_groups):
    c_indices = []
    for group in race_groups.unique():
        group_mask = race_groups == group
        c_index = roc_auc_score(y_true[group_mask], y_pred[group_mask])
        c_indices.append(c_index)
    mean_c_index = np.mean(c_indices)
    std_c_index = np.std(c_indices)
    return mean_c_index - std_c_index

# 计算分层C-index
race_groups = train['race_group']
stratified_c_index_score = stratified_c_index(y, ensemble_model.predict_proba(X_processed)[:, 1], race_groups)
print(f'Stratified C-index: {stratified_c_index_score}')

