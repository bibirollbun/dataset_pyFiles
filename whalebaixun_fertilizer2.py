##学号：20224423320208 姓名胡凯乐
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib
import os
import tempfile
from scipy.stats import skew

def apk(actual, predicted, k=5):
    """计算单个样本的Average Precision@k"""
    if len(predicted) > k:
        predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / min(1, k)

def mapk(actual, predicted, k=5):
    """计算整个数据集的Mean Average Precision@k"""
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def inverse_transform_2d(encoder, arr_2d):
    flattened = arr_2d.ravel()
    transformed = encoder.inverse_transform(flattened)
    return transformed.reshape(arr_2d.shape)

# 数据加载
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 改进的标准化列名函数
def standardize_columns(df):
    # 创建列名映射字典
    column_mapping = {}
    for col in df.columns:
        clean_col = col.strip().lower()
        clean_col = clean_col.replace(' ', '_')
        
        # 修正拼写错误
        if 'temp' in clean_col and 'hum' not in clean_col:
            clean_col = 'temperature'
        elif 'phos' in clean_col:
            clean_col = 'phosphorus'
        elif 'fertilizer' in clean_col:
            clean_col = 'fertilizer'
        
        column_mapping[col] = clean_col
    
    # 重命名列
    df = df.rename(columns=column_mapping)
    
    return df

# 应用列名标准化
train_data = standardize_columns(train_data)
test_data = standardize_columns(test_data)

# 手动修正列名
column_corrections = {
    'temperaturere': 'temperature',
    'phosphorusus': 'phosphorus',
    'phosphorous': 'phosphorus',
    'temparature': 'temperature'
}

for wrong, correct in column_corrections.items():
    if wrong in train_data.columns:
        train_data = train_data.rename(columns={wrong: correct})
    if wrong in test_data.columns:
        test_data = test_data.rename(columns={wrong: correct})

# 目标列处理
target_col = 'fertilizer'
if target_col not in train_data.columns:
    target_candidates = [col for col in train_data.columns if 'fertilizer' in col]
    if target_candidates:
        target_col = target_candidates[0]
    else:
        raise KeyError("训练集中未找到肥料类型列")

# 特征工程增强
def enhanced_feature_engineering(df):
    df = df.copy()
    
    # 基础比值特征
    df['n_p_ratio'] = df['nitrogen'] / (df['phosphorus'] + 1e-6)
    df['n_k_ratio'] = df['nitrogen'] / (df['potassium'] + 1e-6)
    df['p_k_ratio'] = df['phosphorus'] / (df['potassium'] + 1e-6)
    
    # 交互特征
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['temp_moisture'] = df['temperature'] * df['moisture']
    df['humidity_moisture'] = df['humidity'] * df['moisture']
    
    # 组合特征
    df['npk_sum'] = df['nitrogen'] + df['phosphorus'] + df['potassium']
    df['npk_product'] = df['nitrogen'] * df['phosphorus'] * df['potassium']
    
    # 环境指数
    df['env_index'] = 0.5*df['temperature'] + 0.3*df['humidity'] + 0.2*df['moisture']
    
    # 营养平衡特征
    df['nutrient_balance'] = np.abs(df['n_p_ratio'] - 1) + np.abs(df['n_k_ratio'] - 1)
    
    # 土壤-作物组合
    if 'soil_type' in df and 'crop_type' in df:
        df['soil_crop'] = df['soil_type'].astype(str) + "_" + df['crop_type'].astype(str)
    
    return df

# 应用增强的特征工程
train_data = enhanced_feature_engineering(train_data)
test_data = enhanced_feature_engineering(test_data)

# 准备训练数据
drop_cols = ['id', 'name']
for col in drop_cols:
    if col in train_data.columns:
        train_data = train_data.drop(col, axis=1)
    if col in test_data.columns:
        test_data = test_data.drop(col, axis=1)

X = train_data.drop(target_col, axis=1, errors='ignore')
y = train_data[target_col]

# 编码目标变量
le_fertilizer = LabelEncoder()
y_encoded = le_fertilizer.fit_transform(y)

# 处理分类特征
categorical_cols = []
feature_encoders = {}

# 新增的soil_crop特征处理
if 'soil_crop' in X.columns:
    categorical_cols.append('soil_crop')
    le = LabelEncoder()
    combined = pd.concat([X['soil_crop'], test_data['soil_crop']], axis=0)
    le.fit(combined)
    feature_encoders['soil_crop'] = le
    X['soil_crop'] = le.transform(X['soil_crop'])
    test_data['soil_crop'] = le.transform(test_data['soil_crop'])

# 原始分类特征处理
for col in ['soil_type', 'crop_type']:
    if col in X.columns:
        categorical_cols.append(col)
        le = LabelEncoder()
        combined = pd.concat([X[col], test_data[col]], axis=0)
        le.fit(combined)
        feature_encoders[col] = le
        X[col] = le.transform(X[col])
        if col in test_data.columns:
            test_data[col] = le.transform(test_data[col])

# 数值特征处理 - 处理偏态分布
numerical_cols = [col for col in X.columns if col not in categorical_cols and col != 'soil_crop']

# 对偏态特征进行对数变换
for col in numerical_cols:
    if skew(X[col]) > 1.0:  # 偏度大于1视为偏态
        X[col] = np.log1p(X[col])
        if col in test_data.columns:
            test_data[col] = np.log1p(test_data[col])

# 特征缩放
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])

# 优化后的XGBoost参数配置
optimized_params = {
    'objective': 'multi:softprob',
    'num_class': len(le_fertilizer.classes_),
    'eval_metric': 'mlogloss',
    'learning_rate': 0.05,  # 降低学习率
    'max_depth': 8,         # 增加深度
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'gamma': 0.3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 2000,   # 增加树的数量
    'seed': 42,
    'tree_method': 'hist',
    'grow_policy': 'lossguide'
}

# 交叉验证设置
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
models = []
val_scores = []

# 使用交叉验证训练模型
print(f"\n开始使用 {n_folds} 折交叉验证训练XGBoost模型...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"\n===== 训练折叠 {fold+1}/{n_folds} =====")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        optimized_params,
        dtrain,
        num_boost_round=optimized_params['n_estimators'],
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    # 验证集评估
    val_pred_proba = model.predict(dval)
    val_pred_top5 = np.argsort(-val_pred_proba, axis=1)[:, :5]
    map5_score = mapk(y_val, val_pred_top5, k=5)
    val_scores.append(map5_score)
    print(f'折叠 {fold+1} MAP@5: {map5_score:.6f}')
    
    models.append(model)

# 计算平均验证分数
mean_val_score = np.mean(val_scores)
print(f"\n平均验证 MAP@5: {mean_val_score:.6f}")

# 模型保存
model_dir = tempfile.mkdtemp()
for i, model in enumerate(models):
    model_path = os.path.join(model_dir, f'xgboost_model_fold{i+1}.model')
    model.save_model(model_path)

encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
feature_encoder_path = os.path.join(model_dir, 'feature_encoders.joblib')
scaler_path = os.path.join(model_dir, 'scaler.joblib')

joblib.dump(le_fertilizer, encoder_path)
joblib.dump(feature_encoders, feature_encoder_path)
joblib.dump(scaler, scaler_path)

print(f"\n模型和编码器已保存至临时目录: {model_dir}/")

# 测试集预测 - 使用所有模型进行集成
print("\n开始测试集预测...")
dtest = xgb.DMatrix(test_data)
test_pred_proba = np.zeros((test_data.shape[0], len(le_fertilizer.classes_)))

for model in models:
    test_pred_proba += model.predict(dtest)
    
test_pred_proba /= len(models)  # 平均概率
test_pred_top5 = np.argsort(-test_pred_proba, axis=1)[:, :5]

# 将预测索引转换回原始标签
test_pred_labels = inverse_transform_2d(le_fertilizer, test_pred_top5)

# 生成提交文件
original_test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission_preds = [' '.join(pred) for pred in test_pred_labels]

submission = pd.DataFrame({
    'id': original_test_data['id'],
    'Fertilizer Name': submission_preds
})

submission.to_csv('submission.csv', index=False)
print("\n提交文件已保存: submission.csv")

print("\n" + "="*50)
print("="*50)

