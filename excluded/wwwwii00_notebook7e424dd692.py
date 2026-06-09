import os
print(os.listdir('/kaggle/input'))  # 查看所有输入数据目录


import os
print(os.listdir('/kaggle/input/playground-series-s5e6/'))


!pip install --upgrade lightgbm


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import label_ranking_average_precision_score
import lightgbm as lgb
import gc
from datetime import datetime

# 1. 数据加载与预处理
def load_data(use_kaggle_path=True):
    """加载数据并进行内存优化"""
    try:
        if use_kaggle_path:
            train_path = '/kaggle/input/playground-series-s5e6/train.csv'
            test_path = '/kaggle/input/playground-series-s5e6/test.csv'
        else:
            train_path = './train.csv'
            test_path = './test.csv'
            
        # 检查Kaggle数据路径是否存在
        if use_kaggle_path and not os.path.exists(train_path):
            print(f"警告: Kaggle数据路径不存在: {train_path}")
            print("尝试列出Kaggle输入目录...")
            try:
                print(os.listdir('/kaggle/input'))
            except:
                print("无法访问Kaggle输入目录")
            use_kaggle_path = False
        
        print(f"加载训练数据: {train_path if use_kaggle_path else os.path.abspath(train_path)}")
        train_data = pd.read_csv(train_path if use_kaggle_path else train_path)
        print(f"加载测试数据: {test_path if use_kaggle_path else os.path.abspath(test_path)}")
        test_data = pd.read_csv(test_path if use_kaggle_path else test_path)
        
        print(f"训练数据形状: {train_data.shape}")
        print(f"测试数据形状: {test_data.shape}")
        return train_data, test_data
    
    except Exception as e:
        print(f"数据加载错误: {e}")
        print(f"当前工作目录: {os.getcwd()}")
        print("可用文件:", os.listdir() if os.path.exists(os.getcwd()) else "无法列出文件")
        raise

# 2. 特征工程
def engineer_features(df, is_train=True):
    """创建关键特征"""
    # 基础养分比例特征
    eps = 1e-8
    df['NP_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + eps)
    df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + eps)
    df['PK_ratio'] = df['Phosphorous'] / (df['Potassium'] + eps)
    
    # 气候组合特征
    df['temp_humidity'] = df['Temparature'] * df['Humidity']
    df['soil_moisture'] = df['Soil Type'].astype(str) + "_" + df['Moisture'].astype(str)
    
    # 土壤-养分交互特征
    soil_types = df['Soil Type'].unique()
    for nutrient in ['Nitrogen', 'Potassium', 'Phosphorous']:
        for soil_type in soil_types:
            df[f'{nutrient}_{soil_type}'] = df[nutrient] * (df['Soil Type'] == soil_type)
    
    # 作物-环境交互特征
    crop_types = df['Crop Type'].unique()
    for crop in crop_types:
        df[f'{crop}_temp'] = df['Temparature'] * (df['Crop Type'] == crop)
        df[f'{crop}_humidity'] = df['Humidity'] * (df['Crop Type'] == crop)
    
    # 分箱特征
    for col in ['Temparature', 'Humidity', 'Moisture']:
        df[f'{col}_bin'] = pd.qcut(df[col], 8, labels=False, duplicates='drop')
    
    # 统计特征
    nutrient_cols = ['Nitrogen', 'Potassium', 'Phosphorous']
    df['nutrient_mean'] = df[nutrient_cols].mean(axis=1)
    df['nutrient_std'] = df[nutrient_cols].std(axis=1)
    
    # 非线性变换
    for col in nutrient_cols + ['Temparature', 'Humidity', 'Moisture']:
        df[f'{col}_log'] = np.log1p(df[col])
        df[f'{col}_sqrt'] = np.sqrt(df[col])
    
    return df, []

# 3. 数据预处理
def preprocess_data(train_data, test_data):
    """预处理数据并进行内存优化"""
    # 定义特征和目标
    features = [col for col in train_data.columns if col not in ['id', 'Fertilizer Name']]
    target = 'Fertilizer Name'
    
    # 处理类别特征
    cat_cols = ['Soil Type', 'Crop Type', 'soil_moisture']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    combined = pd.concat([train_data[cat_cols], test_data[cat_cols]])
    encoder.fit(combined)
    train_data[cat_cols] = encoder.transform(train_data[cat_cols])
    test_data[cat_cols] = encoder.transform(test_data[cat_cols])
    
    # 目标变量编码
    le_target = LabelEncoder()
    y = le_target.fit_transform(train_data[target])
    X = train_data[features].astype(np.float32)  # 降低数据精度
    test_features = test_data[features].astype(np.float32)
    
    # 释放内存
    del train_data, test_data; gc.collect()
    return X, y, test_features, le_target, features

# 4. 优化MAP@5计算
def calculate_map5(y_true, y_score):
    """优化的MAP@5计算函数，处理大矩阵时更稳定"""
    n_samples, n_classes = y_true.shape
    map5_scores = []
    
    for i in range(n_samples):
        relevant = np.where(y_true[i])[0]
        if len(relevant) == 0:
            continue  # 跳过没有相关类别的样本
            
        # 对分数进行降序排序
        sorted_indices = np.argsort(-y_score[i])
        top5 = sorted_indices[:5]
        
        # 计算相关类别在top5中的排名
        ranks = []
        for label in relevant:
            if label in top5:
                rank = np.where(sorted_indices == label)[0][0] + 1  # 排名从1开始
                ranks.append(rank)
        
        if ranks:
            # 计算平均精度
            avg_precision = sum((i+1)/rank for i, rank in enumerate(ranks)) / len(ranks)
            map5_scores.append(avg_precision)
    
    return np.mean(map5_scores) if map5_scores else 0

# 5. 模型训练
def train_model(X, y, test_features, features, le_target, n_splits=3):
    """使用优化参数训练模型"""
    # 存储预测结果
    oof_probs = np.zeros((len(X), len(le_target.classes_)))
    test_probs = np.zeros((len(test_features), len(le_target.classes_)))
    
    # 分层K折
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        
        # 划分数据集
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 优化的LightGBM参数
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(le_target.classes_),
            n_estimators=800,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=45,
            min_child_samples=30,
            reg_alpha=0.05,
            reg_lambda=0.05,
            colsample_bytree=0.85,
            subsample=0.9,
            random_state=42 + fold,
            n_jobs=-1,
            verbosity=-1
        )
        
        # 训练模型
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # 记录预测结果
        oof_probs[val_idx] = model.predict_proba(X_val)
        test_probs += model.predict_proba(test_features) / n_splits
        
        # 计算MAP@5（使用优化函数）
        y_true = np.zeros((len(y_val), len(le_target.classes_)))
        for i, label in enumerate(y_val):
            y_true[i, label] = 1
        map5_score = calculate_map5(y_true, oof_probs[val_idx])
        print(f"Fold {fold+1} MAP@5: {map5_score:.4f}")
        
        # 清理内存
        del X_train, X_val, y_train, y_val, model; gc.collect()
    
    # 计算整体分数
    y_true_all = np.zeros((len(y), len(le_target.classes_)))
    for i, label in enumerate(y):
        y_true_all[i, label] = 1
    cv_map5_score = calculate_map5(y_true_all, oof_probs)
    print(f"\n整体交叉验证 MAP@5: {cv_map5_score:.4f}")
    
    return test_probs, cv_map5_score, le_target

# 6. 生成提交文件
def create_submission(test_data, test_probs, le_target):
    """生成提交文件"""
    # 获取前5个预测结果
    top5_indices = np.argsort(-test_probs, axis=1)[:, :5]
    top5_labels = le_target.inverse_transform(top5_indices.flatten())
    top5_labels = top5_labels.reshape(len(test_data), 5)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'id': test_data['id'],
        'Fertilizer Name': [' '.join(labels) for labels in top5_labels]
    })
    submission.to_csv('submission.csv', index=False)
    print("提交文件已保存: submission.csv")
    return submission

# 7. 主函数
def main():
    """主函数，协调整个流程"""
    start_time = datetime.now()
    print(f"开始时间: {start_time.strftime('%H:%M:%S')}")
    
    try:
        # 1. 数据加载
        print("\n===== 数据加载 =====")
        train_data, test_data = load_data(use_kaggle_path=True)
        
        # 2. 特征工程
        print("\n===== 特征工程 =====")
        train_data, _ = engineer_features(train_data, is_train=True)
        test_data, _ = engineer_features(test_data, is_train=False)
        
        # 3. 数据预处理
        print("\n===== 数据预处理 =====")
        X, y, test_features, le_target, features = preprocess_data(train_data, test_data)
        
        # 4. 模型训练
        print("\n===== 模型训练 =====")
        test_probs, cv_score, le_target = train_model(
            X, y, test_features, features, le_target, n_splits=3
        )
        
        # 5. 生成提交文件
        print("\n===== 生成提交文件 =====")
        submission = create_submission(test_data, test_probs, le_target)
        
        # 计算总运行时间
        end_time = datetime.now()
        print(f"\n总运行时间: {(end_time - start_time).total_seconds():.2f}秒")
        return cv_score, submission
        
    except Exception as e:
        end_time = datetime.now()
        print(f"\n执行错误: {e}")
        print(f"运行时间: {(end_time - start_time).total_seconds():.2f}秒")
        return None, None

if __name__ == "__main__":
    cv_score, submission = main()
    if cv_score:
        print(f"\n最终交叉验证 MAP@5 分数: {cv_score:.4f}")

