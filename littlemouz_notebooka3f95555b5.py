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



import numpy as np
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 检查GPU支持
print("检查GPU支持...")
USE_GPU = False  # Kaggle环境中GPU存在问题，暂时禁用
print("当前使用CPU模式")

# =====================
# 自定义评估指标 MAP@5
# =====================
def map5_score(y_true, y_pred):
    """计算Mean Average Precision @ 5 (MAP@5)"""
    top5 = np.argsort(-y_pred, axis=1)[:, :5]
    
    ap_scores = []
    for i in range(len(y_true)):
        actual = y_true[i]
        predicted = top5[i]
        
        ap = 0.0
        correct = 0
        for k in range(5):
            if predicted[k] == actual:
                correct += 1
                ap += correct / (k + 1)
        
        ap_scores.append(ap / min(correct, 1) if correct > 0 else 0.0)
    
    return np.mean(ap_scores)

# =====================
# 数据准备与特征工程
# =====================
print("正在读取数据并进行特征工程...")
start_time = time.time()

# 读取数据
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# 显示实际列名
print("\n训练集实际列名:", train.columns.tolist())
print("\n训练集数据类型:")
print(train.dtypes)

# 创建列名映射字典
column_mapping = {}
for col in train.columns:
    # 简化列名：移除特殊字符、空格，转换为小写
    simplified = re.sub(r'[^a-zA-Z0-9]', '', col).lower()
    
    # 识别关键特征
    if 'nitrogen' in simplified or 'n' == simplified:
        column_mapping['N'] = col
    elif 'phosphorus' in simplified or 'p' == simplified:
        column_mapping['P'] = col
    elif 'potassium' in simplified or 'k' == simplified:
        column_mapping['K'] = col
    elif 'temp' in simplified:
        column_mapping['temperature'] = col
    elif 'humid' in simplified:
        column_mapping['humidity'] = col
    elif 'ph' in simplified:
        column_mapping['ph'] = col
    elif 'rain' in simplified:
        column_mapping['rainfall'] = col
    elif 'soil' in simplified or 'type' in simplified:
        column_mapping['soil_type'] = col
    elif 'fertilizer' in simplified:
        column_mapping['fertilizer'] = col

# 确保所有必要列都有映射
required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type', 'fertilizer']
for col in required_columns:
    if col not in column_mapping:
        column_mapping[col] = col  # 使用标准名称作为后备

print("\n列名映射:")
for standard, actual in column_mapping.items():
    print(f"{standard} -> {actual}")

# 重命名列以标准化
train = train.rename(columns={v: k for k, v in column_mapping.items() if k in required_columns})
test = test.rename(columns={v: k for k, v in column_mapping.items() if k in required_columns and v in test.columns})

# 确保所有必要列都存在
for col in required_columns:
    if col not in train.columns:
        train[col] = np.nan
    if col not in test.columns:
        test[col] = np.nan

print("\n标准化后列名:", train.columns.tolist())

# 高级特征工程
print("\n正在进行高级特征工程...")

# 基础特征
train['NP_ratio'] = train['N'] / (train['P'] + 1e-5)
train['NK_ratio'] = train['N'] / (train['K'] + 1e-5)
train['PK_ratio'] = train['P'] / (train['K'] + 1e-5)
train['N+P+K'] = train['N'] + train['P'] + train['K']
train['temp_humidity'] = train['temperature'] * train['humidity']
train['nutrient_balance'] = (train['N'] + train['P']) / (train['K'] + 1e-5)
train['rain_temp_ratio'] = train['rainfall'] / (train['temperature'] + 1e-5)
train['ph_temp_interaction'] = train['ph'] * train['temperature']

test['NP_ratio'] = test['N'] / (test['P'] + 1e-5)
test['NK_ratio'] = test['N'] / (test['K'] + 1e-5)
test['PK_ratio'] = test['P'] / (test['K'] + 1e-5)
test['N+P+K'] = test['N'] + test['P'] + test['K']
test['temp_humidity'] = test['temperature'] * test['humidity']
test['nutrient_balance'] = (test['N'] + test['P']) / (test['K'] + 1e-5)
test['rain_temp_ratio'] = test['rainfall'] / (test['temperature'] + 1e-5)
test['ph_temp_interaction'] = test['ph'] * test['temperature']

# 编码目标变量
print("编码目标变量...")
le = LabelEncoder()
train['fertilizer_encoded'] = le.fit_transform(train['fertilizer'])
n_classes = len(le.classes_)

# 准备特征和目标
features = train.columns.drop(['id', 'fertilizer', 'fertilizer_encoded']).tolist()
print(f"\n最终使用 {len(features)} 个特征")

# 分离数值特征和分类特征
numerical_features = [col for col in features if train[col].dtype in ['int64', 'float64']]
categorical_features = [col for col in features if train[col].dtype == 'object']

print(f"数值特征 ({len(numerical_features)}): {numerical_features}")
print(f"分类特征 ({len(categorical_features)}): {categorical_features}")

# 创建预处理管道
print("创建数据预处理管道...")
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# 预处理数据
print("预处理训练集和测试集...")
X = train[features]
y = train['fertilizer_encoded']

# 使用ColumnTransformer预处理数据
X_preprocessed = preprocessor.fit_transform(X)
test_preprocessed = preprocessor.transform(test[features])

# 获取特征名称
if categorical_features:
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numerical_features, cat_feature_names])
else:
    all_feature_names = numerical_features

print(f"预处理后特征数量: {len(all_feature_names)}")

data_prep_time = time.time() - start_time
print(f"数据准备完成! 耗时: {data_prep_time:.2f}秒")

# =====================
# 模型定义 - 优化配置
# =====================
# 定义多个模型
models = {
    'xgb': XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',  # 修复：移除了重复的eval_metric
        num_class=n_classes,
        n_estimators=800,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        tree_method='hist',  # 使用hist方法
        device='cpu',  # 强制使用CPU
        early_stopping_rounds=50,
        verbosity=0  # 控制详细输出
    ),
    'lgbm': LGBMClassifier(
        objective='multiclass',
        num_class=n_classes,
        n_estimators=800,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        device='cpu',  # 强制使用CPU
        verbose=-1
    ),
    'catboost': CatBoostClassifier(
        loss_function='MultiClass',
        iterations=800,
        learning_rate=0.05,
        depth=8,
        random_state=RANDOM_STATE,
        task_type='CPU',  # 强制使用CPU
        verbose=0
    ),
    'rf': RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )
}

# 打印模型配置
print("\n模型配置:")
for model_name, model in models.items():
    print(f"{model_name}: {model.__class__.__name__}")

# =====================
# 模型训练与集成
# =====================
print("\n开始模型训练与集成...")
start_time = time.time()

# 使用3折交叉验证训练多个模型
n_folds = 3
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

# 存储每个模型的预测结果
test_preds = {model_name: np.zeros((len(test), n_classes)) for model_name in models}
oof_preds = {model_name: np.zeros((len(X_preprocessed), n_classes)) for model_name in models}
scores = {model_name: [] for model_name in models}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_preprocessed, y)):
    print(f"\n=== 折叠 {fold+1}/{n_folds} ===")
    X_train, X_val = X_preprocessed[train_idx], X_preprocessed[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    for model_name, model in models.items():
        print(f"训练 {model_name}...")
        
        try:
            # 特殊处理CatBoost
            if model_name == 'catboost':
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=100,
                    verbose=0
                )
            # 处理XGBoost
            elif model_name == 'xgb':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=0
                )
            # 处理LightGBM
            elif model_name == 'lgbm':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)]
                )
            # 处理其他模型
            else:
                model.fit(X_train, y_train)
            
            # 在验证集上预测
            val_preds = model.predict_proba(X_val)
            oof_preds[model_name][val_idx] = val_preds
            
            # 计算并存储分数
            fold_score = map5_score(y_val.values, val_preds)
            scores[model_name].append(fold_score)
            print(f"{model_name} 折叠 {fold+1} MAP@5: {fold_score:.5f}")
            
            # 在测试集上预测
            test_preds[model_name] += model.predict_proba(test_preprocessed) / n_folds
            
        except Exception as e:
            print(f"训练 {model_name} 时出错: {str(e)}")
            # 使用简单模型作为后备
            backup_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
            backup_model.fit(X_train, y_train)
            val_preds = backup_model.predict_proba(X_val)
            oof_preds[model_name][val_idx] = val_preds
            fold_score = map5_score(y_val.values, val_preds)
            scores[model_name].append(fold_score)
            print(f"后备模型 {model_name} 折叠 {fold+1} MAP@5: {fold_score:.5f}")
            test_preds[model_name] += backup_model.predict_proba(test_preprocessed) / n_folds

# 计算每个模型的平均分数
for model_name in models:
    if len(scores[model_name]) > 0:
        model_score = np.mean(scores[model_name])
        print(f"\n{model_name} 平均 MAP@5: {model_score:.5f} ± {np.std(scores[model_name]):.5f}")
    else:
        print(f"\n{model_name} 没有有效分数")

# =====================
# 模型集成
# =====================
print("\n进行模型集成...")

# 加权平均集成
weights = {
    'xgb': 0.4,
    'lgbm': 0.3,
    'catboost': 0.2,
    'rf': 0.1
}

# 创建加权平均预测
weighted_avg_preds = np.zeros((len(test), n_classes))
for model_name, weight in weights.items():
    weighted_avg_preds += test_preds[model_name] * weight

# =====================
# 生成提交文件
# =====================
print("\n生成测试集预测...")
# 获取每个样本的前5个预测
top5_preds = np.argsort(-weighted_avg_preds, axis=1)[:, :5]

# 将数字标签转换为原始肥料名称
top5_fertilizers = []
for preds in top5_preds:
    top5_fertilizers.append(' '.join(le.inverse_transform(preds)))

# 创建提交文件
submission['fertilizer'] = top5_fertilizers
submission.to_csv('submission.csv', index=False)

print("\n提交文件已生成!")
print("前5个样本的预测结果:")
print(submission.head())

# 可视化模型性能
model_names = list(models.keys())
model_scores = [np.mean(scores[name]) for name in model_names if len(scores[name]) > 0]

if model_scores:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=model_scores, y=model_names, palette='viridis')
    plt.title('模型性能比较 (MAP@5)', fontsize=14)
    plt.xlabel('MAP@5')
    plt.xlim(0.8, 0.95)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

total_time = time.time() - start_time
print(f"\n所有处理完成! 总耗时: {total_time:.2f}秒")
print("请下载提交文件: submission.csv")

