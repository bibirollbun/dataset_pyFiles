import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import label_ranking_average_precision_score
import xgboost as xgb
import time
import warnings

warnings.filterwarnings('ignore')

# 模拟生成土壤特征数据
def generate_soil_data(num_samples=10000):
    """生成模拟土壤特征数据"""
    np.random.seed(42)
    
    # 基础特征
    data = {
        'nitrogen': np.random.normal(50, 15, num_samples),  # 氮含量 (ppm)
        'phosphorus': np.random.normal(30, 10, num_samples),  # 磷含量 (ppm)
        'potassium': np.random.normal(150, 40, num_samples),  # 钾含量 (ppm)
        'ph': np.random.normal(6.5, 1.2, num_samples),  # pH值
        'temperature': np.random.normal(25, 5, num_samples),  # 温度 (°C)
        'moisture': np.random.normal(40, 10, num_samples),  # 湿度 (%)
        'organic_matter': np.random.normal(3.5, 1.0, num_samples),  # 有机质含量 (%)
        'soil_type': np.random.choice(['sandy', 'clay', 'loam', 'silt'], num_samples),  # 土壤类型
    }
    
    df = pd.DataFrame(data)
    
    # 基于特征生成肥料类型标签
    def assign_fertilizer(row):
        if row['nitrogen'] < 40 and row['phosphorus'] < 25 and row['potassium'] < 120:
            return 'NPK_20-20-20'
        elif row['phosphorus'] > 35 and row['potassium'] < 130:
            return 'DAP'
        elif row['nitrogen'] > 60 and row['moisture'] > 45:
            return 'Urea'
        elif row['potassium'] > 180 and row['ph'] < 6.0:
            return 'MOP'
        elif row['organic_matter'] < 2.5 and row['phosphorus'] < 30:
            return 'NPK_15-15-15'
        elif row['ph'] > 7.0 and row['phosphorus'] > 30:
            return 'SSP'
        elif row['temperature'] > 28 and row['moisture'] < 35:
            return 'NPK_10-26-26'
        else:
            return 'NPK_17-17-17'
    
    df['fertilizer'] = df.apply(assign_fertilizer, axis=1)
    return df

# MAP@5计算函数
def map_at_5(y_true, y_pred_proba):
    """计算Mean Average Precision at 5 (MAP@5)"""
    # 获取每个样本前5个预测
    top5_preds = np.argsort(-y_pred_proba, axis=1)[:, :5]
    
    # 计算每个样本的平均精度
    ap_scores = []
    for i in range(len(y_true)):
        actual = y_true[i]
        preds = top5_preds[i]
        precision = 0.0
        correct = 0
        for k in range(min(5, len(preds))):
            if preds[k] == actual:
                correct += 1
                precision += correct / (k + 1)
        ap_scores.append(precision / min(correct, 1) if correct > 0 else 0.0)
    
    return np.mean(ap_scores)

# 特征工程
def feature_engineering(df):
    """创建新特征"""
    # 营养元素比例
    df['np_ratio'] = df['nitrogen'] / (df['phosphorus'] + 1e-5)
    df['nk_ratio'] = df['nitrogen'] / (df['potassium'] + 1e-5)
    df['pk_ratio'] = df['phosphorus'] / (df['potassium'] + 1e-5)
    
    # 综合指标
    df['nutrient_balance'] = (df['nitrogen'] + df['phosphorus'] + df['potassium']) / 3
    df['acidity_level'] = np.where(df['ph'] < 6.0, 'acidic', 
                                  np.where(df['ph'] > 7.5, 'alkaline', 'neutral'))
    
    # 土壤类型编码
    soil_dummies = pd.get_dummies(df['soil_type'], prefix='soil')
    df = pd.concat([df, soil_dummies], axis=1)
    
    # 酸碱度编码
    acidity_dummies = pd.get_dummies(df['acidity_level'], prefix='acidity')
    df = pd.concat([df, acidity_dummies], axis=1)
    
    return df

# 主函数
def main():
    # 1. 生成模拟数据
    print("生成模拟土壤数据...")
    soil_df = generate_soil_data(10000)
    
    # 2. 特征工程
    print("执行特征工程...")
    soil_df = feature_engineering(soil_df)
    
    # 3. 数据预处理
    print("预处理数据...")
    # 编码目标变量
    le = LabelEncoder()
    soil_df['fertilizer_encoded'] = le.fit_transform(soil_df['fertilizer'])
    
    # 选择特征
    features = ['nitrogen', 'phosphorus', 'potassium', 'ph', 'temperature', 
               'moisture', 'organic_matter', 'np_ratio', 'nk_ratio', 'pk_ratio', 
               'nutrient_balance', 'soil_sandy', 'soil_clay', 'soil_loam', 'soil_silt',
               'acidity_acidic', 'acidity_alkaline', 'acidity_neutral']
    
    X = soil_df[features]
    y = soil_df['fertilizer_encoded']
    
    # 标准化数值特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. 训练XGBoost模型
    print("训练XGBoost模型...")
    start_time = time.time()
    
    # 设置模型参数
    params = {
        'objective': 'multi:softprob',
        'num_class': len(le.classes_),
        'eval_metric': 'mlogloss',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'n_estimators': 500
    }
    
    # 使用交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f"\n训练Fold {fold+1}/5")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 训练模型
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=50
        )
        models.append(model)
        
        # 验证集预测
        y_val_pred_proba = model.predict(dval)
        map_score = map_at_5(y_val.values, y_val_pred_proba)
        cv_results.append(map_score)
        print(f"Fold {fold+1} MAP@5: {map_score:.4f}")
    
    # 平均交叉验证分数
    mean_cv_map = np.mean(cv_results)
    print(f"\n平均交叉验证MAP@5: {mean_cv_map:.4f}")
    
    # 6. 测试集评估
    print("\n在测试集上评估模型...")
    dtest = xgb.DMatrix(X_test)
    
    # 组合所有模型的预测
    test_pred_proba = np.zeros((X_test.shape[0], len(le.classes_)))
    for model in models:
        test_pred_proba += model.predict(dtest)
    test_pred_proba /= len(models)
    
    # 计算MAP@5
    test_map = map_at_5(y_test.values, test_pred_proba)
    print(f"测试集MAP@5: {test_map:.4f}")
    
    # 7. 特征重要性
    print("\n分析特征重要性...")
    fig, ax = plt.subplots(figsize=(12, 8))
    xgb.plot_importance(models[0], ax=ax, max_num_features=15)
    plt.title('特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    # 8. 预测示例
    print("\n预测示例:")
    sample_idx = np.random.randint(0, len(X_test), 5)
    for i, idx in enumerate(sample_idx):
        sample = X_test[idx]
        actual_fertilizer = le.inverse_transform([y_test.iloc[idx]])[0]
        
        # 获取预测概率
        dsample = xgb.DMatrix(sample.reshape(1, -1))
        proba = np.zeros((1, len(le.classes_)))
        for model in models:
            proba += model.predict(dsample)
        proba /= len(models)
        
        # 获取前5个预测
        top5_idx = np.argsort(-proba[0])[:5]
        top5_fertilizers = le.inverse_transform(top5_idx)
        top5_proba = proba[0][top5_idx]
        
        print(f"\n样本 {i+1} - 实际肥料: {actual_fertilizer}")
        print("前5个预测:")
        for j, (fert, prob) in enumerate(zip(top5_fertilizers, top5_proba)):
            print(f"  {j+1}. {fert} ({prob:.4f})")
    
    # 9. 训练时间
    training_time = time.time() - start_time
    print(f"\n总训练时间: {training_time:.2f}秒")
    
    # 10. 保存模型
    # 实际应用中应保存模型，这里省略

if __name__ == "__main__":
    main()

