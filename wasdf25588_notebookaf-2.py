# 方法1：使用 TensorFlow 检测
try:
    import tensorflow as tf
    print("TensorFlow 版本:", tf.__version__)
    print("GPU 可用:", tf.config.list_physical_devices('GPU') != [])
except:
    print("未安装 TensorFlow")


# 学号: XXX, 姓名: XXX
# Kaggle竞赛：预测最佳肥料 (MAP@5评分) - GPU加速版

# 1. 导入必要库
import pandas as pd
import numpy as np
import warnings
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import top_k_accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import gc

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)

# 检查GPU可用性
def check_gpu_support():
    print("="*50)
    print("GPU支持检查:")
    
    # LightGBM GPU支持
    try:
        lgb_gpu = lgb.LGBMClassifier(device='gpu')
        print("LightGBM GPU支持: 是")
    except:
        print("LightGBM GPU支持: 否")
    
    # XGBoost GPU支持
    try:
        xgb_gpu = xgb.XGBClassifier(tree_method='gpu_hist')
        print("XGBoost GPU支持: 是")
    except:
        print("XGBoost GPU支持: 否")
    
    # CatBoost GPU支持
    try:
        cb_gpu = CatBoostClassifier(task_type='GPU')
        print("CatBoost GPU支持: 是")
    except:
        print("CatBoost GPU支持: 否")
    
    print("="*50)

# 2. 数据加载
start_time = time.time()
print("开始数据加载...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
print(f"数据加载完成! 耗时: {time.time()-start_time:.2f}秒")

# 保存测试集ID用于最终提交
test_ids = test_df['id'].copy()

# 检查GPU支持
check_gpu_support()

# 3. 特征工程 (GPU友好型)
def feature_engineering(df):
    # 复制数据框，避免修改原始数据
    df = df.copy()
    
    # 数值特征交互
    if 'Temperature' in df and 'Humidity' in df:
        df['Temp_Humidity'] = df['Temperature'] * df['Humidity']
    if 'Moisture' in df and 'Humidity' in df:
        df['Moisture_Humidity'] = df['Moisture'] * df['Humidity']
    if 'Nitrogen' in df and 'Phosphorous' in df:
        df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    if 'Nitrogen' in df and 'Potassium' in df:
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    if 'Phosphorous' in df and 'Potassium' in df:
        df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    
    # 多项式特征
    if 'Temperature' in df:
        df['Temp_squared'] = df['Temperature'] ** 2
    if 'Humidity' in df:
        df['Humidity_squared'] = df['Humidity'] ** 2
    
    return df

# 对训练集和测试集应用特征工程
start_time = time.time()
print("\n开始特征工程...")
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)
print(f"特征工程完成! 耗时: {time.time()-start_time:.2f}秒")

# 4. 数据预处理 (内存优化)
def preprocess_data(train_df, test_df):
    # 目标列
    target_col = 'Fertilizer Name'
    
    # 检查目标列
    if target_col not in train_df.columns:
        available_cols = train_df.columns.tolist()
        raise ValueError(f"目标列 '{target_col}' 不存在于训练数据中。可用列: {available_cols}")
    
    print("\n训练数据列名:", train_df.columns.tolist())
    
    # 分离特征和目标
    X_train = train_df.drop(columns=[target_col, 'id'])  # 移除ID列
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=['id'])  # 移除ID列

    # 识别特征类型
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    # 处理缺失值
    for col in numeric_cols:
        mean_val = X_train[col].mean()
        X_train[col].fillna(mean_val, inplace=True)
        X_test[col].fillna(mean_val, inplace=True)
    
    for col in categorical_cols:
        mode_val = X_train[col].mode()[0]
        X_train[col].fillna(mode_val, inplace=True)
        X_test[col].fillna(mode_val, inplace=True)

    # 处理分类变量
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    # 标准化数值特征
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, y_train, X_test

# 执行预处理
start_time = time.time()
print("\n开始数据预处理...")
try:
    X_train, y_train, X_test = preprocess_data(train_df, test_df)
    print(f"预处理完成! 耗时: {time.time()-start_time:.2f}秒")
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    
    # 释放内存 - 但保留test_ids
    del train_df
    gc.collect()
except ValueError as e:
    print("预处理错误:", e)
    raise

# 对目标变量编码
start_time = time.time()
print("\n目标变量编码...")
le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train)
class_names = le_target.classes_
n_classes = len(class_names)
print(f"目标类别数: {n_classes}, 耗时: {time.time()-start_time:.2f}秒")

# 检查类别分布
class_counts = pd.Series(y_train_encoded).value_counts()
print("\n类别分布:")
print(class_counts)

# 5. 模型训练 - 使用分层K折交叉验证和GPU加速
print("\n===== 开始模型训练 (使用GPU加速) =====")
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# 初始化存储预测的数组
test_preds = np.zeros((X_test.shape[0], n_classes))
val_preds = np.zeros((X_train.shape[0], n_classes))

# 存储模型
models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
    fold_start_time = time.time()
    print(f"\n===== 训练Fold {fold+1}/{n_folds} =====")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
    
    # 计算类别权重
    class_weights = {}
    for i in range(n_classes):
        class_weights[i] = len(y_tr) / (n_classes * np.sum(y_tr == i))
    
    # 模型列表
    fold_models = []
    
    # LightGBM模型 (GPU加速)
    print("\n[LightGBM] 使用GPU训练...")
    lgb_start = time.time()
    lgb_model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=127,
        learning_rate=0.05,
        n_estimators=1500,
        max_depth=9,
        objective='multiclass',
        num_class=n_classes,
        class_weight=class_weights,
        random_state=42 + fold,
        n_jobs=-1,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    fold_models.append(('lgb', lgb_model))
    print(f"LightGBM训练完成! 耗时: {time.time()-lgb_start:.2f}秒")
    
    # XGBoost模型 (GPU加速)
    print("\n[XGBoost] 使用GPU训练...")
    xgb_start = time.time()
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42 + fold,
        tree_method='gpu_hist',
        gpu_id=0,
        predictor='gpu_predictor',
        verbosity=0
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )
    fold_models.append(('xgb', xgb_model))
    print(f"XGBoost训练完成! 耗时: {time.time()-xgb_start:.2f}秒")
    
    # CatBoost模型 (GPU加速)
    print("\n[CatBoost] 使用GPU训练...")
    cb_start = time.time()
    cb_model = CatBoostClassifier(
        loss_function='MultiClass',
        iterations=1500,
        learning_rate=0.05,
        depth=9,
        l2_leaf_reg=3,
        class_weights=class_weights,
        random_seed=42 + fold,
        task_type='GPU',
        devices='0:0',
        verbose=100
    )
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=100
    )
    fold_models.append(('cb', cb_model))
    print(f"CatBoost训练完成! 耗时: {time.time()-cb_start:.2f}秒")
    
    models.append(fold_models)
    
    # 验证集预测
    print("\n生成验证集预测...")
    val_lgb_pred = lgb_model.predict_proba(X_val)
    val_xgb_pred = xgb_model.predict_proba(X_val)
    val_cb_pred = cb_model.predict_proba(X_val)
    
    # 加权平均概率
    weights = [0.4, 0.3, 0.3]  # LightGBM权重最高
    val_avg_pred = weights[0] * val_lgb_pred + weights[1] * val_xgb_pred + weights[2] * val_cb_pred
    val_preds[val_idx] = val_avg_pred
    
    # 测试集预测
    print("生成测试集预测...")
    test_lgb_pred = lgb_model.predict_proba(X_test)
    test_xgb_pred = xgb_model.predict_proba(X_test)
    test_cb_pred = cb_model.predict_proba(X_test)
    
    # 加权平均概率
    test_avg_pred = weights[0] * test_lgb_pred + weights[1] * test_xgb_pred + weights[2] * test_cb_pred
    test_preds += test_avg_pred / n_folds
    
    # 计算当前fold的MAP@5
    top_k_indices = np.argsort(-val_avg_pred, axis=1)[:, :5]
    mapk_score = top_k_accuracy_score(y_val, val_avg_pred, k=5)
    print(f"Fold {fold+1} 验证集 MAP@5: {mapk_score:.4f}")
    
    # 释放内存
    del lgb_model, xgb_model, cb_model, val_lgb_pred, val_xgb_pred, val_cb_pred, test_lgb_pred, test_xgb_pred, test_cb_pred
    gc.collect()
    
    print(f"Fold {fold+1} 总耗时: {time.time()-fold_start_time:.2f}秒")

# 6. 整体验证集评估
print("\n===== 整体验证集评估 =====")
# 获取Top-K预测索引
val_top_k_indices = np.argsort(-val_preds, axis=1)[:, :5]

# 计算MAP@5
mapk_score = top_k_accuracy_score(y_train_encoded, val_preds, k=5)
print(f"整体验证集 MAP@5: {mapk_score:.4f}")

# 7. 生成测试集预测
print("\n===== 生成测试集预测 =====")
# 获取Top-K预测索引
test_top_k_indices = np.argsort(-test_preds, axis=1)[:, :5]

# 将数字索引转换为实际标签
test_labels = []
for sample in test_top_k_indices:
    decoded_sample = [class_names[idx] for idx in sample]
    test_labels.append(decoded_sample)

# 生成提交文件（符合MAP@5要求）
submission = pd.DataFrame({
    'id': test_ids,
    'top_5_fertilizers': [' '.join(preds) for preds in test_labels]
})

submission.to_csv('submission.csv', index=False)
print("\n===== 提交文件已生成! =====")
print(f"文件路径: submission.csv")
print(f"前5行提交数据:\n{submission.head()}")
print("\n===== 所有步骤完成! =====")

