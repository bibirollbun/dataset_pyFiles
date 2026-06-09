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


#2024423310119 林俊仁
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 定义评估函数
def mapk(y_true, y_pred, k=5):
    """计算 Mean Average Precision @ k"""
    def apk(actual, predicted, k=5):
        if len(predicted) > k:
            predicted = predicted[:k]
        score, num_hits = 0.0, 0.0
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / min(len(actual), k) if actual else 0.0
    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)])

# 加载数据（使用英语列名）
try:
    train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    print("Data loaded successfully")
except Exception as e:
    print(f"Data loading failed: {e}")
    # 创建模拟数据用于演示
    print("Using simulated data for demonstration...")
    np.random.seed(42)
    train_data = pd.DataFrame({
        'id': range(1000),
        'Temperature': np.random.normal(25, 5, 1000),
        'Humidity': np.random.normal(60, 15, 1000),
        'Moisture': np.random.normal(50, 20, 1000),
        'Soil Type': np.random.choice(['Sandy', 'Loamy', 'Clayey'], 1000),
        'Crop Type': np.random.choice(['Wheat', 'Corn', 'Rice'], 1000),
        'Nitrogen': np.random.normal(60, 30, 1000),
        'Potassium': np.random.normal(50, 25, 1000),
        'Phosphorous': np.random.normal(40, 20, 1000),
        'Fertilizer Name': np.random.choice(['Nitrogen', 'Phosphorus', 'Potassium', 'Compound'], 1000)
    })
    test_data = pd.DataFrame({
        'id': range(1000, 1500),
        'Temperature': np.random.normal(25, 5, 500),
        'Humidity': np.random.normal(60, 15, 500),
        'Moisture': np.random.normal(50, 20, 500),
        'Soil Type': np.random.choice(['Sandy', 'Loamy', 'Clayey'], 500),
        'Crop Type': np.random.choice(['Wheat', 'Corn', 'Rice'], 500),
        'Nitrogen': np.random.normal(60, 30, 500),
        'Potassium': np.random.normal(50, 25, 500),
        'Phosphorous': np.random.normal(40, 20, 500),
    })

# 数据探索
print("\nData basic information:")
train_data.info()

print("\nNumber of rows and columns in the dataset:")
rows, columns = train_data.shape

if rows < 100 or columns < 5:
    print("The dataset is too small, which may affect model performance")

# 特征和目标变量
X = train_data.drop(['id', 'Fertilizer Name'], axis=1)
y = train_data['Fertilizer Name']
X_test = test_data.drop('id', axis=1)

# 检查列名
print("\nTraining set column names:", X.columns.tolist())
print("Test set column names:", X_test.columns.tolist())

# 分离数值和分类特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# 目标变量编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 数据预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 添加独热编码
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 模型定义
xgb_model = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

lgbm_model = LGBMClassifier(
    objective='multiclass',
    metric='multi_logloss',
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

# 集成模型
ensemble_model = VotingClassifier(
    estimators=[('xgb', xgb_model), ('lgbm', lgbm_model)],
    voting='soft'
)

# 构建完整管道
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ensemble_model)
])

# 训练模型
print("\nStarting model training...")
pipeline.fit(X, y_encoded)

# 预测并计算MAP@5
print("\nCalculating model performance metrics...")
y_pred_proba = pipeline.predict_proba(X)
top5_indices = np.argsort(y_pred_proba, axis=1)[:, -5:][:, ::-1]
top5_labels = [label_encoder.inverse_transform(pred) for pred in top5_indices]

map5_score = mapk([[fertilizer] for fertilizer in y], top5_labels)
print(f"Training set MAP@5 score: {map5_score:.4f}")

# === 可视化部分（英文）===

# 1. 目标变量分布
plt.figure(figsize=(10, 6))
ax = y.value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Fertilizer Types')
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# 添加数值标签
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.show()

# 2. 数值特征相关性热力图
plt.figure(figsize=(12, 10))
correlation = X[numeric_features].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", 
            cmap='coolwarm', square=True, linewidths=.5,
            cbar_kws={"shrink": .8})
plt.title('Numeric Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# 3. 特征重要性（基于XGBoost）
plt.figure(figsize=(10, 8))
# 修正访问方式：直接通过estimators_获取XGB模型
if hasattr(pipeline.named_steps['classifier'].estimators_[0], 'feature_importances_'):
    xgb_model = pipeline.named_steps['classifier'].estimators_[0]
    xgb_importance = xgb_model.feature_importances_
    
    # 获取特征名称（包括独热编码后的）
    preprocessor_fit = preprocessor.fit(X)
    feature_names = preprocessor_fit.get_feature_names_out()
    
    importance_df = pd.Series(xgb_importance, index=feature_names).sort_values(ascending=True)
    importance_df.tail(10).plot(kind='barh', color='lightgreen')  # 只显示前10个重要特征
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
else:
    print("Model does not support feature importance calculation")

# 4. 数值特征分布
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features[:9], 1):  # 只显示前9个特征
    plt.subplot(3, 3, i)
    sns.histplot(X[feature], kde=True)
    plt.title(f'{feature} Distribution')
plt.tight_layout()
plt.show()

# 5. 分类特征分布
plt.figure(figsize=(15, 6))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(1, len(categorical_features), i)
    X[feature].value_counts().plot(kind='bar')
    plt.title(f'{feature} Distribution')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. 预测概率分布（针对第一个样本）
plt.figure(figsize=(10, 6))
sample_idx = 0
probs = pipeline.predict_proba(X.iloc[[sample_idx]])[0]
classes = label_encoder.classes_

sns.barplot(x=classes, y=probs, color='salmon')
plt.title(f'Fertilizer Type Prediction Probabilities for Sample {sample_idx}')
plt.xlabel('Fertilizer Type')
plt.ylabel('Probability')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nData visualization completed!")

