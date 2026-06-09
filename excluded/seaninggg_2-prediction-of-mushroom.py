!pip install dython


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


# 导入必要的库 Import necessary libraries
import numpy as np  # 用于数值计算 For numerical computations
import pandas as pd  # 用于数据处理 For data manipulation
import matplotlib.pyplot as plt  # 用于数据可视化 For data visualization

# 机器学习相关库 Machine Learning related libraries
from sklearn.model_selection import train_test_split  # 数据集拆分 Data splitting
import gc  # 垃圾回收 Garbage collection
import seaborn as sns  # 数据可视化工具 Data visualization tool

# 数据预处理 Data Preprocessing
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder  # 类别编码 Label Encoding
from dython.nominal import associations  # 计算分类变量之间的相关性 Calculate correlation between categorical variables

# 交互式可视化 Interactive Visualization
import plotly.express as px  # 高级数据可视化 Advanced data visualization
import plotly.graph_objects as go  # 交互式绘图 Interactive plotting

# 处理缺失值 Missing Value Handling
from sklearn.impute import KNNImputer  # KNN 插补法填充缺失值 KNN-based imputation

# XGBoost 分类模型 XGBoost Classifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor  # XGBoost 回归模型 XGBoost Regressor


# 读取数据集 Load datasets
df_sub = pd.read_csv("/kaggle/input/playground-series-s4e8/sample_submission.csv")  # 读取样本提交文件 Load sample submission file
df_train = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")  # 读取训练集 Load training dataset
df_test = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")  # 读取测试集 Load test dataset


# 显示训练集的前 5 行数据 Display the first 5 rows of the training dataset
df_train.head()


# 显示测试集的前 5 行数据 Display the first 5 rows of the test dataset
df_test.head()


# 获取训练集和测试集的维度（行数, 列数）
# Get the shape (number of rows, number of columns) of the training and test datasets
df_train.shape, df_test.shape


# 删除 ID 列
# Drop the 'id' column
df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])

# 显示训练集的信息，包括数据类型和缺失值情况
# Display information about the training dataset
# including data types and missing values
df_train.info()
print(df_train.isnull().sum())


# 选择数据集中所有的分类特征（数据类型为 'object' 的列）
# Select all categorical features (columns with data type 'object')
categorical_columns = df_train.select_dtypes(include=['object']).columns
# 计算每个分类特征的唯一值数量
# Count the number of unique values in each categorical feature
unique_values = {col: df_train[col].nunique() for col in categorical_columns}
# 输出每个分类特征的唯一值个数
# Print the number of unique values for each categorical feature
for col, unique_count in unique_values.items():
    print(f"{col}: {unique_count} unique values")
# 进行垃圾回收，释放内存（适用于大数据集）
# Perform garbage collection to free up memory (useful for large datasets)
gc.collect()


# 选择测试集中所有的分类特征（数据类型为 'object' 的列）
# Select all categorical features (columns with data type 'object') in the test dataset
categorical_columns = df_test.select_dtypes(include=['object']).columns
# 计算每个分类特征的唯一值数量
# Count the number of unique values in each categorical feature
unique_values = {col: df_test[col].nunique() for col in categorical_columns}
# 输出每个分类特征的唯一值个数
# Print the number of unique values for each categorical feature
for col, unique_count in unique_values.items():
    print(f"{col}: {unique_count} unique values")
# 进行垃圾回收，释放内存（适用于大数据集）
# Perform garbage collection to free up memory (useful for large datasets)
gc.collect()


# 获取测试集和训练集的所有列名
# Get all column names from the test and training datasets
df_test.columns, df_train.columns


# 计算训练集中每列的缺失值比例（百分比）
# Calculate the percentage of missing values for each column in the training dataset
missing_train = df_train.isna().mean() * 100
# 计算测试集中每列的缺失值比例（百分比）
# Calculate the percentage of missing values for each column in the test dataset
missing_test = df_test.isna().mean() * 100


# 输出训练集中缺失值超过 10% 的列
# Print columns in the training dataset with more than 10% missing values
print("Columns in df_train with more than 10% missing values:")
# 筛选出缺失值比例大于 10% 的列，并输出
# Filter and print columns where the missing value percentage is greater than 10%
print(missing_train[missing_train > 10])


# 输出测试集中缺失值超过 10% 的列
# Print columns in the test dataset with more than 10% missing values
print("\nColumns in df_test with more than 10% missing values:")
# 筛选出缺失值比例大于 10% 的列，并输出
# Filter and print columns where the missing value percentage is greater than 10%
print(missing_test[missing_test > 10])


# 计算训练集中每列的缺失值比例（百分比）
# Calculate the percentage of missing values for each column in the training dataset
missing_values = df_train.isnull().mean() * 100
# 仅保留存在缺失值的列
# Keep only columns with missing values
missing_values = missing_values[missing_values > 0]
# 按缺失值比例从高到低排序
# Sort columns by missing value percentage in descending order
missing_values = missing_values.sort_values(ascending=False)


# 设置图表大小 (10, 6)
# Set figure size (10, 6)
plt.figure(figsize=(10, 6))

# 绘制缺失值分布的条形图
# Plot a bar chart showing the distribution of missing values
sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')

# 旋转 x 轴标签，防止重叠
# Rotate x-axis labels to prevent overlap
plt.xticks(rotation=90)

# 设置 x 轴和 y 轴的标签
# Set labels for x-axis and y-axis
plt.xlabel('Features')
plt.ylabel('Percentage of Missing Values')

# 设置图表标题
# Set chart title
plt.title('Missing Values Distribution in df_train')

# 显示图表
# Display the plot
plt.show()


# 设定缺失值阈值（95%），如果某列缺失值超过 95%，则删除
# Set the missing value threshold (95%), if a column has more than 95% missing values, it will be removed
missing_threshold = 0.95

# 找出缺失值超过阈值的列
# Identify columns with missing values greater than the threshold
high_missing_columns = df_train.columns[df_train.isnull().mean() > missing_threshold]

# 从训练集和测试集中删除这些高缺失率的列
# Drop these high-missing-value columns from both the training and test datasets
df_train = df_train.drop(columns=high_missing_columns)
df_test = df_test.drop(columns=high_missing_columns)

# 指定目标变量（分类标签）
# Define the target variable (class label)
target = 'class'


# 遍历训练集的所有列
# Iterate through all columns in the training dataset
for column in df_train.columns:

    # 检查当前列是否存在缺失值
    # Check if the column has missing values
    if df_train[column].isnull().any():

        # 如果是类别（object）类型，则使用众数填充
        # If the column is categorical (object type), fill missing values with mode (most frequent value)
        if df_train[column].dtype == 'object':
            mode_value = df_train[column].mode()[0]  # 计算众数 Compute mode
            df_train[column].fillna(mode_value, inplace=True)
            df_test[column].fillna(mode_value, inplace=True)  # 确保测试集使用相同填充值 Ensure the test set uses the same mode value

        # 如果是数值型数据，则使用中位数填充
        # If the column is numerical, fill missing values with the median
        else:
            median_value = df_train[column].median()  # 计算中位数 Compute median
            df_train[column].fillna(median_value, inplace=True)
            df_test[column].fillna(median_value, inplace=True)  # 确保测试集使用相同填充值 Ensure the test set uses the same median value


# 计算前 10,000 行数据的特征相关性，包括类别特征
# Compute feature correlations for the first 10,000 rows, including categorical features
associations_df = associations(df_train[:10000], nominal_columns='all', plot=False)
# 提取相关性矩阵
# Extract the correlation matrix
corr_matrix = associations_df['corr']
plt.figure(figsize=(20, 8))
plt.gcf().set_facecolor('#FFFDD0')
# 绘制相关性矩阵的热力图
# Plot the heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix including Categorical Features')
plt.show()


# 复制训练数据的前 10,000 行
# Copy the first 10,000 rows of the training dataset
df_train1 = df_train[:10000].copy()

# 统计不同帽形（cap-shape）和帽颜色（cap-color）组合的数量
# Count the occurrences of different cap-shape and cap-color combinations
feature_counts = df_train1.groupby(['cap-shape', 'cap-color']).size().reset_index(name='count')

# 创建旭日图（Sunburst Chart）可视化帽形和帽颜色的分布
# Create a Sunburst Chart to visualize the distribution of cap shape and cap color
fig = px.sunburst(feature_counts, path=['cap-shape', 'cap-color'], values='count',
                  color='count', color_continuous_scale='Viridis',
                  title='Sunburst Chart of Cap Shape and Cap Color Distribution')

# 调整图表布局（标题居中，设置图表宽度和高度）
# Adjust chart layout (center title, set chart width and height)
fig.update_layout(title_text='Sunburst Chart of Cap Shape and Cap Color Distribution',
                  title_x=0.5, width=900, height=600)

# 显示图表
# Display the plot
fig.show()


# 统计不同帽形（cap-shape）和帽颜色（cap-color）组合的数量
# Count occurrences of different cap-shape and cap-color combinations
flow_data = df_train1.groupby(['cap-shape', 'cap-color']).size().reset_index(name='count')

# 生成唯一的标签列表（帽形和帽颜色）
# Generate a unique list of labels (cap-shape and cap-color)
labels = list(pd.concat([flow_data['cap-shape'], flow_data['cap-color']]).unique())

# 创建标签映射字典，将类别名称映射到唯一的索引编号
# Create a dictionary to map category names to unique index numbers
label_map = {label: idx for idx, label in enumerate(labels)}

# 将帽形映射到索引，作为桑基图的源节点
# Map cap-shape to indices as the source nodes in the Sankey diagram
sources = flow_data['cap-shape'].map(label_map).tolist()

# 将帽颜色映射到索引，作为桑基图的目标节点
# Map cap-color to indices as the target nodes in the Sankey diagram
targets = flow_data['cap-color'].map(label_map).tolist()

# 获取每种帽形-帽颜色组合的数量，用于表示流量大小
# Get the count of each cap-shape and cap-color combination, representing the flow size
values = flow_data['count'].tolist()


# 创建桑基图（Sankey Diagram）可视化帽形到帽颜色的流动关系
# Create a Sankey Diagram to visualize the flow from cap-shape to cap-color
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,  # 设置节点之间的间距 Set spacing between nodes
        thickness=20,  # 设置节点的厚度 Set node thickness
        line=dict(color='black', width=0.5),  # 设置节点边框颜色和宽度 Set node border color and width
        label=labels  # 设置节点标签 Set node labels
    ),
    link=dict(
        source=sources,  # 连接的源节点 Indices of source nodes
        target=targets,  # 连接的目标节点 Indices of target nodes
        value=values  # 连接的权重（流量大小） Flow values between nodes
    )
)])

# 更新图表布局（标题居中，对齐方式，大小等）
# Update chart layout (title alignment, width, height, etc.)
fig.update_layout(
    title_text='Sankey Chart of Cap Shape to Cap Color Flow',  # 设置标题 Set chart title
    title_x=0.5,  # 让标题居中 Align the title to the center
    width=1000,  # 设置图表宽度 Set chart width
    height=600  # 设置图表高度 Set chart height
)

# 显示图表
# Display the plot
fig.show()


# # 提取与 'class' 相关的所有特征，并按相关性降序排序
correlation_with_class = corr_matrix['class'].drop('class', errors='ignore').dropna().sort_values(ascending=False)

# 设定相关性阈值（筛选最相关的特征）
threshold = 0.2  # 可以调整此值
filtered_correlation = correlation_with_class[abs(correlation_with_class) > threshold]

# 绘制柱状图
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=filtered_correlation.values, y=filtered_correlation.index, palette="magma", ax=ax)

# 设置标签和标题
ax.set_xlabel("Correlation with Edibility (Class)", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
ax.set_title("Feature Correlation with Edibility (Class)", fontsize=14)

# 显示数值标签
for i, v in enumerate(filtered_correlation.values):
    ax.text(v + 0.01, i, f"{v:.2f}", color='black', va='center', fontsize=12)

plt.show()



# 统计不同帽形（cap-shape）和帽颜色（cap-color）组合的数量
# Count occurrences of different cap-shape and cap-color combinations
feature_counts = df_train1.groupby(['cap-shape', 'cap-color']).size().reset_index(name='count')

# 创建堆叠柱状图（Stacked Bar Chart）展示帽形和帽颜色的分布
# Create a stacked bar chart to visualize the distribution of cap shape and cap color
fig = px.bar(feature_counts, x='cap-shape', y='count', color='cap-color',
             title='Crosstab Chart of Cap Shape and Cap Color',  # 设置图表标题 Set chart title
             labels={'cap-shape': 'Cap Shape', 'count': 'Count', 'cap-color': 'Cap Color'},  # 设置轴标签 Set axis labels
             color_discrete_sequence=px.colors.qualitative.Plotly,  # 使用 Plotly 颜色方案 Use Plotly color scheme
             text='count')  # 在柱状图上显示计数值 Display count values on the bars

# 调整图表布局（标题居中，对齐方式，X/Y 轴名称，堆叠模式）
# Adjust chart layout (title alignment, X/Y axis labels, stacked mode)
fig.update_layout(
    title_text='Crosstab Chart of Cap Shape and Cap Color',  # 设置标题 Set title
    title_x=0.5,  # 让标题居中 Center the title
    xaxis_title='Cap Shape',  # 设置 X 轴标签 Set X-axis label
    yaxis_title='Count',  # 设置 Y 轴标签 Set Y-axis label
    barmode='stack'  # 设定为堆叠柱状图 Set to stacked bar chart mode
)

# 显示图表
# Display the plot
fig.show()


# 选择缺失值超过 95% 的训练集列
# Select columns in the training dataset with more than 95% missing values
cols_to_drop_train = missing_train[missing_train > 95].index
# 选择缺失值超过 95% 的测试集列
# Select columns in the test dataset with more than 95% missing values
cols_to_drop_test = missing_test[missing_test > 95].index
# 从训练集中删除这些高缺失率的列
# Drop these high-missing-value columns from the training dataset
df_train = df_train.drop(columns=cols_to_drop_train)
# 从测试集中删除这些高缺失率的列
# Drop these high-missing-value columns from the test dataset
df_test = df_test.drop(columns=cols_to_drop_test)
# 进行垃圾回收，释放内存（适用于大数据集）
# Perform garbage collection to free up memory (useful for large datasets)
gc.collect()


# 定义 KNN 插补函数，用于填充缺失值
# Define a KNN imputation function to fill missing values
def knn_impute(df, n_neighbors=5):
    df_encoded = df.copy()  # 复制原始数据 Copy the original data

    # 将类别变量转换为数值编码，以便 KNN 插补能够处理
    # Convert categorical variables to numerical encoding for KNN imputation
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

    # 使用 KNNImputer 进行缺失值填充
    # Use KNNImputer to fill missing values
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_encoded), columns=df_encoded.columns)

    # 还原类别变量的原始编码
    # Restore original categorical encoding
    for col in df.select_dtypes(include='object').columns:
        df_imputed[col] = df_imputed[col].round().astype(int).map(
            dict(enumerate(df[col].astype('category').cat.categories))
        )

    return df_imputed

# 对训练集和测试集应用 KNN 插补
# Apply KNN imputation to the training and test datasets
df_train_imputed = knn_impute(df_train, n_neighbors=5)
df_test_imputed = knn_impute(df_test, n_neighbors=5)

# 选择训练集中所有的类别变量（排除目标变量 'class'）
# Select all categorical columns in the training dataset (excluding the target variable 'class')
cat_cols_train = df_train_imputed.select_dtypes(include=['object']).columns
cat_cols_train = cat_cols_train[cat_cols_train != 'class']

# 使用 OrdinalEncoder 对类别变量进行编码（适用于树模型）
# Use OrdinalEncoder to encode categorical variables (suitable for tree-based models)
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 训练集类别变量进行编码
# Encode categorical variables in the training dataset
df_train_imputed[cat_cols_train] = ordinal_encoder.fit_transform(df_train_imputed[cat_cols_train].astype(str))

# 测试集类别变量进行编码（使用相同的编码映射）
# Encode categorical variables in the test dataset (using the same encoding mapping)
df_test_imputed[cat_cols_train] = ordinal_encoder.transform(df_test_imputed[cat_cols_train].astype(str))


# 显示填充缺失值后的训练集前 5 行
# Display the first 5 rows of the imputed training dataset
df_train_imputed.head()


# 显示填充缺失值后的测试集前 5 行
# Display the first 5 rows of the imputed test dataset
df_test_imputed.head()


# 将填充后的训练集替换原始训练集
# Replace the original training dataset with the imputed version
df_train = df_train_imputed

# 将填充后的测试集替换原始测试集
# Replace the original test dataset with the imputed version
df_test = df_test_imputed

# 显示填充后的测试集前 5 行
# Display the first 5 rows of the imputed test dataset
df_test.head()


# 初始化标签编码器（LabelEncoder）
# Initialize the LabelEncoder
le = LabelEncoder()

# 对目标变量 'class' 进行编码（转换为数值）
# Encode the target variable 'class' (convert categorical labels to numerical values)
df_train['class'] = le.fit_transform(df_train['class'])

# 提取目标变量 (y) 和特征变量 (X)
# Extract the target variable (y) and feature variables (X)
y = df_train['class']
X = df_train.drop(['class'], axis=1)  # 移除目标变量，仅保留特征 Drop the target variable, keep only features

# 将数据集拆分为训练集和测试集（80% 训练，20% 测试）
# Split the dataset into training and testing sets (80% train, 20% test)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 按类别比例划分，确保类别分布一致 Stratify ensures class distribution remains consistent
)
gc.collect()


# XGBoost 分类器
# XGBoost Classifier
import time
import gc
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Starting XGBoost training...")  # 输出训练开始信息 Print training start message
start_time = time.time()
# 降低数据精度，以减少内存占用
# Convert data to lower precision to reduce memory usage
train_X = train_X.astype(np.float32)
test_X = test_X.astype(np.float32)
# 采样 20% 训练数据，以减少计算负担
# Sample 20% of the training data to reduce computational load
train_sampled = train_X.sample(frac=0.2, random_state=42)
train_y_sampled = train_y.loc[train_sampled.index]
# 定义 XGBoost 超参数搜索空间
# Define the hyperparameter search space for XGBoost
param_grid = {
    'alpha': [0.01, 0.1, 1],  # L1 正则化参数 L1 regularization parameter
    'subsample': [0.6, 0.8, 1.0],  # 训练时使用的样本比例 Subsampling ratio for training
    'colsample_bytree': [0.4, 0.6, 0.8],  # 每棵树使用的特征比例 Feature sampling ratio per tree
    'max_depth': range(5, 15, 2),  # 决策树最大深度 Maximum tree depth
    'min_child_weight': range(1, 10, 2),  # 控制叶子节点的最小样本权重 Minimum sum of instance weight in a leaf
    'gamma': [0, 1e-6, 1e-3],  # 分裂节点的最小损失 reduction Minimum loss reduction for node splitting
    'n_estimators': [50, 100, 200]  # 训练的树的数量 Number of trees
}
# 进行超参数搜索
# Perform hyperparameter tuning using RandomizedSearchCV
search = RandomizedSearchCV(
    XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42),
    param_distributions=param_grid,
    n_iter=20,  # 迭代 20 次（减少搜索时间） Perform 20 iterations (reduce search time)
    cv=3,  # 3 折交叉验证 3-fold cross-validation
    n_jobs=-1,  # 使用所有可用 CPU 核心 Use all available CPU cores
    verbose=1  # 输出搜索过程的详细信息 Print detailed search process
)
# 训练超参数搜索模型
# Train the hyperparameter search model
search.fit(train_sampled, train_y_sampled)
# 输出最优参数和最佳交叉验证得分
# Print the best parameters and best cross-validation score
print(f'Best Parameters: {search.best_params_}, Best CV Score: {search.best_score_:.4f}')
# 使用最优参数训练最终 XGBoost 模型
# Train the final XGBoost model using the best parameters
model = XGBClassifier(
    **search.best_params_,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)
# 训练 XGBoost 模型
# Train the XGBoost model
XGB = model.fit(
    train_sampled,
    train_y_sampled,
    eval_set=[(test_X, test_y)]
)
# 在测试集上进行预测
# Make predictions on the test set
y_pred = XGB.predict(test_X)
# 计算评估指标
# Compute evaluation metrics
accuracy = accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred, average='macro')
recall = recall_score(test_y, y_pred, average='macro')
f1 = f1_score(test_y, y_pred, average='macro')
# 输出模型评估结果
# Print model evaluation results
print("XGBoost Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Training time: {time.time() - start_time:.2f} seconds\n")
gc.collect()


# 决策树分类器（内存优化版）
# Decision Tree Classifier (Memory Optimized)
import time
import gc
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Starting Decision Tree training...")  # 输出训练开始信息 Print training start message
start_time = time.time()
# 降低数据精度，以减少内存占用
# Convert data to lower precision to reduce memory usage
train_X = train_X.astype(np.float32)
test_X = test_X.astype(np.float32)
# 采样 20% 训练数据，以减少计算负担
# Sample 20% of the training data to reduce computational load
train_sampled = train_X.sample(frac=0.2, random_state=42)
train_y_sampled = train_y.loc[train_sampled.index]
# 定义决策树的超参数搜索空间
# Define the hyperparameter search space for Decision Tree
param_grid = {
    'criterion': ['gini', 'entropy'],  # 选择使用基尼系数或信息增益 Choose between Gini impurity and Information Gain
    'max_depth': range(5, 20, 3),  # 决策树最大深度 Maximum depth of the decision tree
    'min_samples_split': range(2, 10, 2),  # 节点拆分的最小样本数 Minimum samples required to split a node
    'min_samples_leaf': range(5, 10)  # 叶子节点的最小样本数 Minimum number of samples per leaf node
}
# 进行超参数搜索
# Perform hyperparameter tuning using RandomizedSearchCV
search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,  # 迭代 20 次（减少搜索时间） Perform 20 iterations (reduce search time)
    cv=3,  # 3 折交叉验证 3-fold cross-validation
    n_jobs=-1,  # 并行计算（使用所有可用 CPU 核心） Parallel computation (use all available CPU cores)
    verbose=1  # 输出搜索过程的详细信息 Print detailed search process
)
# 训练超参数搜索模型
# Train the hyperparameter search model
search.fit(train_sampled, train_y_sampled)
# 输出最优参数和最佳交叉验证得分
# Print the best parameters and best cross-validation score
print(f'Best Parameters: {search.best_params_}, Best CV Score: {search.best_score_:.4f}')
# 使用最优参数训练最终的决策树模型
# Train the final Decision Tree model using the best parameters
clf = DecisionTreeClassifier(random_state=42, **search.best_params_)
clf.fit(train_sampled, train_y_sampled)
# 在测试集上进行预测
# Make predictions on the test set
y_pred = clf.predict(test_X)
# 计算评估指标
# Compute evaluation metrics
accuracy = accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred, average='macro')
recall = recall_score(test_y, y_pred, average='macro')
f1 = f1_score(test_y, y_pred, average='macro')
# 输出模型评估结果
# Print model evaluation results
print("Decision Tree Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Training time: {time.time() - start_time:.2f} seconds\n")
gc.collect()


# 逻辑回归分类器
# Logistic Regression Classifier
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Starting Logistic Regression training...")  # 输出训练开始信息 Print training start message
start_time = time.time()
# 训练逻辑回归模型
# Train the Logistic Regression model
clf = LogisticRegression(random_state=42, max_iter=1000)  # 设置最大迭代次数，确保收敛 Set max iterations to ensure convergence
clf.fit(train_X, train_y)
# 在测试集上进行预测
# Make predictions on the test set
y_pred = clf.predict(test_X)
# 计算评估指标
# Compute evaluation metrics
accuracy = accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred, average='macro')
recall = recall_score(test_y, y_pred, average='macro')
f1 = f1_score(test_y, y_pred, average='macro')
# 输出模型评估结果
# Print model evaluation results
print("Logistic Regression Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Training time: {time.time() - start_time:.2f} seconds\n")
gc.collect()


# 随机森林分类器
# Random Forest Classifier
# 通过减少计算量来加快训练速度：
# - 降低 n_estimators（减少树的数量）
# - 限制 max_depth（防止树过深，提高效率）
# - 设置 max_features='sqrt'（降低每棵树的计算量）
# - 启用 n_jobs=-1（并行计算，提高速度）
# - 启用 warm_start=True（增量训练，避免重复计算）

import time
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Starting Random Forest training...")  # 输出训练开始信息 Print training start message
start_time = time.time()
# 训练随机森林模型
# Train the Random Forest model
clf = RandomForestClassifier(
    n_estimators=50,  # 树的数量（减少计算量） Number of trees (reduces computation)
    max_depth=10,  # 限制树的深度（防止过拟合，提高效率） Limit tree depth (prevents overfitting and improves efficiency)
    max_features='sqrt',  # 每棵树使用 sqrt(特征数) 个特征 Feature selection strategy (square root of total features)
    n_jobs=-1,  # 并行计算（使用所有可用 CPU 核心） Parallel computation (use all available CPU cores)
    random_state=42,  # 固定随机种子以保证结果可复现 Set random seed for reproducibility
    warm_start=True  # 允许增量训练（可以提高效率） Enable incremental training (avoids redundant calculations)
)
# 拟合模型
# Fit the model
clf.fit(train_X, train_y)
# 在测试集上进行预测
# Make predictions on the test set
y_pred = clf.predict(test_X)
# 计算评估指标
# Compute evaluation metrics
accuracy = accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred, average='macro')
recall = recall_score(test_y, y_pred, average='macro')
f1 = f1_score(test_y, y_pred, average='macro')
# 输出模型评估结果
# Print model evaluation results
print("Random Forest Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Training time: {time.time() - start_time:.2f} seconds\n")
gc.collect()


#SVM model using Stochastic Gradient Descent (SGD)
import time
from sklearn.linear_model import SGDClassifier  # 使用 SGDClassifier（随机梯度下降 SVM）
from sklearn.preprocessing import MinMaxScaler  # 归一化数据（MinMaxScaler 适用于 SVM）
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA  # 主成分分析（降维）

def svm_classifier(train_X, train_y, test_X, test_y):
    """
    训练基于 SGD 的 SVM 模型并评估其表现。
    Train an SVM model using Stochastic Gradient Descent (SGD) and evaluate its performance.
    """
    start_time = time.time()
    print("Optimizing data...")  # 输出优化数据的消息 Print message indicating data optimization
    # 1归一化数据（提升收敛速度）
    # Normalize data to improve convergence speed
    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    # 2 降维 (如果特征维度过高)
    # Apply PCA if the feature dimension is too high
    if train_X.shape[1] > 100:
        print("High feature dimension detected, applying PCA for dimensionality reduction...")
        pca = PCA(n_components=100)  # 限制最多 100 维 Reduce dimensions to a maximum of 100
        train_X = pca.fit_transform(train_X)
        test_X = pca.transform(test_X)
    print("Training SGD-SVM model...")  # 输出训练开始信息 Print training start message
    # 3️训练 SGD-SVM
    # Train the SGD-based SVM model
    clf = SGDClassifier(loss='hinge', max_iter=5000, tol=1e-4, verbose=True)
    clf.fit(train_X, train_y)
    # 4预测
    # Make predictions on the test set
    y_pred = clf.predict(test_X)
    # 5计算评估指标
    # Compute evaluation metrics
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred, average='macro')
    recall = recall_score(test_y, y_pred, average='macro')
    f1 = f1_score(test_y, y_pred, average='macro')
    # 输出模型评估结果
    # Print model evaluation results
    print("SGD-SVM Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Training time: {time.time() - start_time:.2f} seconds\n")
    return clf
# 训练和评估 SGD-SVM
# Train and evaluate the SGD-SVM model
svm_model = svm_classifier(train_X, train_y, test_X, test_y)
gc.collect()

