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


# ===========================================================
# Step 1: Environment Setup and Data Loading
# ===========================================================
# Install required packages  # 安装必要的包（Kaggle环境中需要）
!pip install contextily --quiet
# 导入所需的库
import numpy as np  # 数值计算库
import pandas as pd # 数据处理库
import matplotlib.pyplot as plt # 数据可视化库
import seaborn as sns # 高级数据可视化库
import plotly.express as px
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 数据集划分
from sklearn.compose import ColumnTransformer  # 列转换器
from sklearn.pipeline import Pipeline  # 数据处理管道
from sklearn.metrics import mean_squared_error  # 评估指标
from sklearn.cluster import MiniBatchKMeans  # 快速聚类算法
from sklearn.impute import SimpleImputer  # 缺失值填充
import lightgbm as lgb  # 轻量级梯度提升框架
import geopandas as gpd  # 地理数据处理
import contextily as ctx  # 地图背景
from shapely.geometry import Point  # 点几何对象
%matplotlib inline 

# Set visualization style # 设置可视化风格
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8) # 设置默认图表大小

# Load data  # 加载数据
train = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv') # 训练集
test = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv')  # 测试集
sample_submission = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/sample_submission.csv')  # 提交样例

print("Step 1 complete: Data loaded")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")


# ===========================================================
# Step 2: Enhanced Exploratory Data Analysis (EDA)  # Step 2: 增强型探索性数据分析（EDA）
# ===========================================================
def perform_eda(df):
    # 1. Price distribution analysis   # 1. 价格分布分析
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(df['price'], bins=50, kde=True, ax=axes[0])
    axes[0].set_title('Original Price Distribution')
    axes[0].set_xlabel('Price (£)')
    
    sns.histplot(np.log1p(df['price']), bins=50, kde=True, ax=axes[1])
    axes[1].set_title('Log-Transformed Price Distribution')
    axes[1].set_xlabel('Log(Price)')
    plt.tight_layout()
    plt.savefig('price_distribution.png')
    plt.show()
    
    # 2. Price trends over time  # 2. 时间趋势分析
    plt.figure(figsize=(14, 6))
    
    # Yearly trend   # 年度趋势
    plt.subplot(1, 2, 1)
    yearly_avg = df.groupby('sale_year')['price'].median()
    sns.lineplot(x=yearly_avg.index, y=yearly_avg.values, marker='o')
    plt.title('Median Price by Year')
    plt.xlabel('Year')
    plt.ylabel('Median Price (£)')
    plt.grid(True)
    
    # Monthly trend  # 月度趋势
    plt.subplot(1, 2, 2)
    monthly_avg = df.groupby('sale_month')['price'].median()
    sns.barplot(x=monthly_avg.index, y=monthly_avg.values)
    plt.title('Median Price by Month')
    plt.xlabel('Month')
    plt.ylabel('Median Price (£)')
    plt.tight_layout()
    plt.savefig('time_trends.png')
    plt.show()
    
    # 3. Numerical feature analysis  # 3. 数值特征分析 - 探索房价与关键数值特征的关系
    num_features = ['floorAreaSqM', 'bedrooms', 'bathrooms', 'livingRooms']
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(num_features):
        plt.subplot(2, 2, i+1)
        sns.scatterplot(x=feature, y='price', data=df, alpha=0.3)
        plt.title(f'Price vs {feature}')
        plt.ylabel('Price (£)')
    plt.tight_layout()
    plt.savefig('numeric_features.png')
    plt.show()
    
    # 4. Categorical feature analysis  # 4. 分类特征分析 - 探索不同类别特征对房价的影响
    cat_features = ['propertyType', 'tenure', 'currentEnergyRating']
    plt.figure(figsize=(18, 12))
    for i, feature in enumerate(cat_features):# 遍历分类特征列表，针对每个特征进行分析
        plt.subplot(2, 2, i+1)# 处理类别过多的特征，只取前10个类别
        if df[feature].nunique() > 10:
            top_categories = df[feature].value_counts().index[:10]
            data_subset = df[df[feature].isin(top_categories)]
        else:
            data_subset = df.copy()# 类别数≤10时保留全量数据
        
        # Use boxplot for price distribution  # 使用箱线图展示价格分布
        sns.boxplot(x=feature, y='price', data=data_subset)
        plt.xticks(rotation=45)# 旋转x轴标签
        plt.title(f'Price Distribution by {feature}')
        plt.ylabel('Price (£)')
        
        # Add median price annotations# 添加中位数价格标注
        medians = data_subset.groupby(feature)['price'].median().sort_values()
        for j, (_, median_val) in enumerate(medians.items()):
            plt.text(j, median_val, f'£{median_val:,.0f}', 
                     horizontalalignment='center', 
                     verticalalignment='bottom', 
                     fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('categorical_features.png')
    plt.show()
    
    # 5. Location-based analysis   #  地理空间分析 - 探索地理位置对房价的影响
    # Create GeoDataFrame for mapping # 创建地理数据框架
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Convert to Web Mercator for contextily # 转换为Web Mercator投影坐标系
    gdf = gdf.to_crs(epsg=3857)
    
    # Plot price density # 绘制价格密度图
    ax = gdf.plot(figsize=(14, 10), column='price', cmap='viridis', 
                 markersize=10, alpha=0.7, legend=True,
                 legend_kwds={'label': "Price (£)", 'shrink': 0.7})
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)# 添加底图
    plt.title('London Property Price Distribution')
    plt.axis('off')
    plt.savefig('geo_distribution.png')
    plt.show()
    
    # 6. Correlation analysis #相关性分析 - 探索特征间的相关关系
    # Select numerical features for correlation
    corr_features = num_features + ['price', 'latitude', 'longitude']
    corr_df = df[corr_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap='coolwarm', 
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()
    
    # 7. Price by number of rooms # 房间数量分析 - 探索总房间数对房价的影响
    plt.figure(figsize=(12, 8))
    room_counts = df[['bedrooms', 'bathrooms', 'livingRooms']].sum(axis=1)
    sns.boxplot(x=room_counts, y='price', data=df)
    plt.title('Price Distribution by Total Room Count')
    plt.xlabel('Total Rooms (Bedrooms + Bathrooms + Living Rooms)')
    plt.ylabel('Price (£)')
    plt.tight_layout()
    plt.savefig('price_by_room_count.png')
    plt.show()
    
    # 8. Price per square meter analysis  #每平方米价格分析 - 探索单位面积价格的地理分布
    df['price_per_sqm'] = df['price'] / (df['floorAreaSqM'] + 1)  # 避免除以零
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='longitude', y='latitude', 
                    hue='price_per_sqm', size='price_per_sqm',
                    data=df, palette='viridis', alpha=0.7)
    plt.title('Price per Square Meter in London')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig('price_per_sqm.png')
    plt.show()

print("Performing comprehensive EDA...")
perform_eda(train)
print("Step 2 complete: Enhanced EDA visualizations created")


# ===========================================================
# Step 3: Feature Engineering (Optimized)
# ===========================================================
def create_features(df):
    """
    创建新特征以增强模型预测能力
    包括地理特征、房间特征、时间特征等
    """

    df = df.copy()
    
    # 1. 地理特征
    # 填充缺失的经纬度（使用中位数）
    df['latitude'] = df['latitude'].fillna(df['latitude'].median())
    df['longitude'] = df['longitude'].fillna(df['longitude'].median())
    
    # 快速地理聚类（使用MiniBatchKMeans提高效率）
    if len(df) > 10:
        kmeans = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=1000)
        df['geo_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
    else:
        df['geo_cluster'] = -1
    
   # 中心伦敦标识（根据经纬度范围）
    df['central_london'] = ((df['longitude'].between(-0.15, 0.05)) & 
                           (df['latitude'].between(51.48, 51.53))).astype(int)
    
    # 2. 房间特征
    # 处理缺失值：卧室填充0后替换为1.5（中位数）
    df['bedrooms'] = df['bedrooms'].fillna(0).replace(0, 1.5)
    df['bathrooms'] = df['bathrooms'].fillna(0) # 浴室填充0
    df['livingRooms'] = df['livingRooms'].fillna(1)# 总房间数
    
     # 计算衍生特征
    df['size_per_bedroom'] = df['floorAreaSqM'] / (df['bedrooms'] + 1e-3)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['livingRooms']
    df['price_per_sqm'] = df['price'] / (df['floorAreaSqM'] + 1) if 'price' in df.columns else None
    
    # 3. 时间特征
    if 'sale_year' in df.columns:
        df['years_since_2000'] = df['sale_year'] - 2000# 距离2000年的年数
    
    # 4. 邮政编码特征
    if 'outcode' in df.columns:
        df['outcode'] = df['outcode'].fillna('Unknown')# 填充缺失值
         # 计算邮政编码出现频率
        df['outcode_frequency'] = df.groupby('outcode')['outcode'].transform('count')
    
    # 5. 能源评级
    if 'currentEnergyRating' in df.columns:
        df['currentEnergyRating'] = df['currentEnergyRating'].fillna('Unknown')# 填充缺失值
    
    return df

print("Applying optimized feature engineering...")
train = create_features(train) # 为训练集创建新特征
test = create_features(test)# 为测试集创建新特征

print("Step 3 complete: Feature engineering done")


# ===========================================================
# Step 4: Data Preprocessing and Validation Split Step 4: 数据预处理与验证集划分
# ===========================================================
# Define feature types  # 定义特征类型
num_features = [
    'floorAreaSqM', 'bedrooms', 'bathrooms', 
    'latitude', 'longitude', 'size_per_bedroom',
    'years_since_2000', 'central_london', 'total_rooms'
]# 数值特征列表

cat_features = [
    'propertyType', 'tenure', 'currentEnergyRating', 
    'outcode', 'geo_cluster'
]# 分类特征列表

# 创建预处理管道
# Create preprocessing pipeline  
preprocessor = ColumnTransformer(
    transformers=[  # 数值特征处理：缺失值填充（中位数）+ 标准化
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features),# 分类特征处理：缺失值填充（众数）+ 独热编码
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ]), cat_features)
])
# 准备数据
# Prepare data
X = train.drop('price', axis=1)# 特征矩阵
y = np.log1p(train['price'])  # 目标变量（对数变换处理偏态分布）

# 创建验证集划分（90%训练，10%验证）
# Create validation split  # 创建预处理管道
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42  # 随机种子确保可复现性
)

print("Step 4 complete: Data preprocessing done")
print(f"Training size: {X_train.shape}, Validation size: {X_val.shape}")


# ===========================================================
# Step 5: Model Training (Optimized for Speed) # Step 5: 模型训练（速度优化版）
# ===========================================================
print("Training LightGBM model (optimized for speed)...") 

# Preprocess data # 预处理数据
X_train_preprocessed = preprocessor.fit_transform(X_train) # 拟合预处理器并转换训练集
X_val_preprocessed = preprocessor.transform(X_val)  # 转换验证集

# Convert to LightGBM Dataset # 转换为LightGBM数据集格式
train_data = lgb.Dataset(X_train_preprocessed, label=y_train)
val_data = lgb.Dataset(X_val_preprocessed, label=y_val, reference=train_data)

# Set parameters for fast training # 设置快速训练参数
params = {
    'objective': 'regression',# 回归任务
    'metric': 'rmse', # 评估指标（均方根误差）
    'learning_rate': 0.1, # 较高的学习率加速收敛
    'num_leaves': 31, # 叶子数量
    'max_depth': 5, # 树的最大深度
    'min_data_in_leaf': 20, # 叶子节点最小数据量
    'feature_fraction': 0.8, # 每次迭代随机选择80%特征
    'bagging_fraction': 0.8, # 随机选择80%数据进行训练
    'bagging_freq': 1, # 每1次迭代执行bagging
    'verbosity': -1, # 静默模式
    'seed': 42 # 随机种子
}

# Train with early stopping  # 使用早停法训练模型
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000, # 最大迭代次数
    valid_sets=[val_data], # 验证集
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True), # 早停法（50轮无改进停止）
        lgb.log_evaluation(period=50) # 每50轮打印日志
    ]
)

# Validation predictions # 验证集预测
val_pred = model.predict(X_val_preprocessed)

# Evaluation metrics # 评估指标
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE (log scale): {rmse:.4f}")
print(f"Validation RMSE (original price): £{np.expm1(rmse):,.2f}")

print("Step 5 complete: Model trained")


# ===========================================================
# Step 6: Feature Importance and Model Analysis # Step 6: 特征重要性与模型分析
# ===========================================================
# Get feature importances  # 获取特征重要性（基于信息增益）
feature_importances = model.feature_importance(importance_type='gain')

# Get feature names  # 获取特征名称
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
all_feature_names = np.concatenate([num_features, cat_feature_names])

# Create importance dataframe  # 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False).head(20)  # 取前20个最重要特征

# 1. Feature importance visualization # 1. 特征重要性可视化
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance (Gain)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# 2. Actual vs Predicted prices  # 2. 实际价格 vs 预测价格
val_results = pd.DataFrame({
    'Actual': np.expm1(y_val),  # 反向转换实际价格
    'Predicted': np.expm1(val_pred)  # 反向转换预测价格
})

plt.figure(figsize=(10, 8))
sns.scatterplot(x='Actual', y='Predicted', data=val_results, alpha=0.4)
plt.plot([0, val_results['Actual'].max()], [0, val_results['Actual'].max()], 'r--')  # 理想预测线
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price (£)')
plt.ylabel('Predicted Price (£)')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.show()

# 3. Residual analysis  # 3. 残差分析（预测误差）
val_results['Residual'] = val_results['Actual'] - val_results['Predicted']
plt.figure(figsize=(12, 6))
sns.histplot(val_results['Residual'], bins=50, kde=True)
plt.title('Residual Distribution')
plt.xlabel('Prediction Error (£)')
plt.savefig('residual_distribution.png')
plt.show()

# 4. Error by feature values  # 4. 特征值与误差的关系分析
val_results_with_features = pd.concat([val_results, X_val.reset_index(drop=True)], axis=1)

# Error by number of bedrooms  # 卧室数量与预测误差的关系
plt.figure(figsize=(12, 6))
sns.boxplot(x='bedrooms', y='Residual', data=val_results_with_features)
plt.title('Prediction Error by Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Prediction Error (£)')
plt.savefig('error_by_bedrooms.png')
plt.show()

# Error by property type  # 房产类型与预测误差的关系
plt.figure(figsize=(14, 8))
top_property_types = val_results_with_features['propertyType'].value_counts().index[:10] # 取前10种房产类型
sns.boxplot(x='propertyType', y='Residual', 
            data=val_results_with_features[val_results_with_features['propertyType'].isin(top_property_types)])
plt.title('Prediction Error by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Prediction Error (£)')
plt.xticks(rotation=45)  # 旋转x轴标签
plt.savefig('error_by_property_type.png')
plt.show()

print("Step 6 complete: Comprehensive model analysis done")


# ===========================================================
# Step 7: Full Model Training and Prediction  # Step 7: 全量模型训练与预测
# ===========================================================
# Preprocess full training data  # 预处理全量训练数据
X_full_preprocessed = preprocessor.fit_transform(X)  # 使用全量数据拟合预处理器
y_full = y  # 全量目标变量

# Create full dataset # 创建全量数据集
full_data = lgb.Dataset(X_full_preprocessed, label=y_full)

# Train final model with optimal iterations # 使用最优迭代次数训练最终模型
optimal_iterations = model.best_iteration  # 从验证中获取最优迭代次数
print(f"Training final model with {optimal_iterations} iterations...")

final_model = lgb.train(
    params,
    full_data,
    num_boost_round=optimal_iterations # 使用最优迭代次数
)
# 预处理测试数据
# Preprocess test data
X_test_preprocessed = preprocessor.transform(test) # 使用训练好的预处理器转换测试集

# Generate test predictions # 生成测试集预测
print("Generating predictions...")
test_pred = np.expm1(final_model.predict(X_test_preprocessed)) # 反向转换预测价格
test_pred = np.maximum(test_pred, 0)  # Ensure non-negative # 确保预测价格为非负

print("Step 7 complete: Predictions generated")


# ===========================================================
# Step 8: Create Submission File  Step 8: 创建提交文件
# ===========================================================
submission = pd.DataFrame({  # 创建提交DataFrame
    'ID': test['ID'], # 测试集ID
    'price': test_pred  # 预测价格
})

# Save submission # 保存提交文件
submission.to_csv('submission.csv', index=False) # 不保存索引列
print("\nSubmission file saved: submission.csv")

# Final checks # 最终检查
print("\nFinal checks:")
print(f"Submission rows: {len(submission)}")
print(f"Test set rows: {len(test)}")
print(f"Sample submission rows: {len(sample_submission)}")
if len(submission) == len(test):
    print("✅ Row count match")
else:
    print("❌ Row count mismatch")

print("All steps complete! Ready to submit.")

