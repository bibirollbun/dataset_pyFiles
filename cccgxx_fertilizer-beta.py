print("2024423320201，陈果鑫")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略特定警告
warnings.filterwarnings('ignore', category=FutureWarning)

# 修正的MAP@3计算函数（符合评分规则）
def map3_score(y_true, y_pred):
    """
    计算Mean Average Precision @ 3
    :param y_true: 真实标签数组 (n_samples,)
    :param y_pred: 预测标签数组 (n_samples, 3)
    :return: MAP@3 分数
    """
    total_ap = 0.0
    for true, pred in zip(y_true, y_pred):
        ap = 0.0
        correct_count = 0
        seen = set()  # 跟踪已看到的正确标签
        
        for k, p in enumerate(pred[:3], 1):
            # 检查预测是否正确且尚未被计入
            if p == true and p not in seen:
                correct_count += 1
                seen.add(p)
                ap += correct_count / k
        
        # 如果找到正确答案，除以min(3, 实际相关项数)
        if correct_count > 0:
            ap /= min(3, correct_count)
        total_ap += ap
    
    return total_ap / len(y_true)

# 创建模拟数据集
def create_synthetic_dataset(n_samples=1000):
    """生成具有农学合理性的模拟土壤肥料数据集"""
    np.random.seed(42)
    
    # 基础土壤特征
    data = {
        'nitrogen': np.random.uniform(5, 50, n_samples),    # 氮含量 (ppm)
        'phosphorus': np.random.uniform(2, 40, n_samples),   # 磷含量 (ppm)
        'potassium': np.random.uniform(10, 60, n_samples),   # 钾含量 (ppm)
        'pH': np.random.uniform(4.0, 8.5, n_samples),        # pH值
        'temperature': np.random.uniform(10, 35, n_samples),  # 温度 (°C)
        'moisture': np.random.uniform(15, 80, n_samples),     # 湿度 (%)
        'organic_matter': np.random.uniform(0.5, 5.0, n_samples),  # 有机质含量 (%)
    }
    
    # 类别特征
    soil_types = ['sandy', 'clay', 'loamy', 'silt']
    regions = ['north', 'south', 'east', 'west']
    data['soil_type'] = np.random.choice(soil_types, n_samples)
    data['region'] = np.random.choice(regions, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 添加从750000开始的唯一ID
    df['id'] = range(750000, 750000 + n_samples)
    
    # 根据农学规则定义肥料类型
    fertilizer_rules = []
    fertilizers = ['NPK_20-20-20', 'Urea', 'DAP', 'MOP', 'SSP', 
                  'NPK_15-15-15', 'Slow-Release', 'Lime', 'Gypsum', 'Ammonium_Sulfate']
    
    for i in range(n_samples):
        n = df.at[i, 'nitrogen']
        p = df.at[i, 'phosphorus']
        k = df.at[i, 'potassium']
        ph = df.at[i, 'pH']
        moisture = df.at[i, 'moisture']
        
        # 根据NPK比例和pH值决定肥料类型
        if n < 20 and p < 15 and k < 25:
            fert = 'NPK_20-20-20'  # 均衡肥料
        elif n < 15 and p > 20:
            fert = 'DAP'  # 磷酸二铵
        elif k < 20 and ph > 7.0:
            fert = 'MOP'  # 氯化钾
        elif n < 10:
            fert = 'Urea'  # 尿素
        elif p < 10 and ph < 6.5:
            fert = 'SSP'  # 过磷酸钙
        elif moisture < 30:
            fert = 'Slow-Release'  # 缓释肥料
        elif ph < 5.5:
            fert = 'Gypsum'  # 石膏（用于酸性土壤）
        elif ph > 7.5:
            fert = 'Lime'  # 石灰（用于碱性土壤）
        else:
            fert = 'NPK_15-15-15'  # 标准复合肥
        
        fertilizer_rules.append(fert)
    
    # 添加随机变异
    noise = np.random.choice(fertilizers, size=n_samples, p=[0.6] + [0.4/9]*9)
    
    # 结合规则和随机选择
    df['fertilizer'] = np.where(np.random.random(n_samples) > 0.2, 
                              fertilizer_rules, 
                              noise)
    
    return df

# 数据预处理
def preprocess_data(df):
    """处理缺失值、编码类别特征"""
    # 数值特征用中位数填充
    num_cols = ['nitrogen', 'phosphorus', 'potassium', 'pH', 'temperature', 'moisture', 'organic_matter']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # 类别特征用众数填充
    cat_cols = ['soil_type', 'region']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # 编码标签
    le = LabelEncoder()
    if 'fertilizer' in df.columns:  # 只在训练集上编码
        df['fertilizer_label'] = le.fit_transform(df['fertilizer'])
    
    return df, le

# 特征工程
def feature_engineering(df):
    """创建有农学意义的新特征"""
    # NPK比例特征
    df['np_ratio'] = df['nitrogen'] / (df['phosphorus'] + 1e-6)
    df['nk_ratio'] = df['nitrogen'] / (df['potassium'] + 1e-6)
    df['pk_ratio'] = df['phosphorus'] / (df['potassium'] + 1e-6)
    
    # 温湿度交互特征
    df['temp_humidity'] = df['temperature'] * df['moisture']
    
    # pH分组
    df['pH_group'] = pd.cut(df['pH'], 
                           bins=[0, 5.5, 6.5, 7.5, 14],
                           labels=['acidic', 'slightly_acidic', 'neutral', 'alkaline'])
    
    # 土壤类型编码
    soil_dummies = pd.get_dummies(df['soil_type'], prefix='soil')
    df = pd.concat([df, soil_dummies], axis=1)
    
    # 地区编码
    region_dummies = pd.get_dummies(df['region'], prefix='region')
    df = pd.concat([df, region_dummies], axis=1)
    
    # 有机质分类
    df['organic_level'] = pd.cut(df['organic_matter'],
                               bins=[0, 1.0, 2.0, 5.0],
                               labels=['low', 'medium', 'high'])
    
    return df

# 应用农学规则修正预测
def apply_agronomic_rules(predictions, soil_data, le):
    """应用农学规则修正预测结果（Top3）"""
    # 将预测索引转换为肥料名称
    decoded_preds = []
    for pred_row in predictions:
        decoded_row = le.inverse_transform(pred_row)
        decoded_preds.append(decoded_row)
    
    # 规则1: 极酸土壤(pH<5.5)不应推荐碱性肥料
    if 'pH' in soil_data.columns:
        acid_soils = soil_data['pH'] < 5.5
        for i in range(len(soil_data)):
            if acid_soils.iloc[i]:
                row = decoded_preds[i]
                if 'Lime' in row:
                    # 移除石灰推荐，添加石膏
                    new_row = [p for p in row if p != 'Lime']
                    if 'Gypsum' not in new_row and len(new_row) < 3:
                        new_row.append('Gypsum')
                    decoded_preds[i] = new_row[:3]
    
    # 规则2: 干旱地区优先推荐缓释肥料
    if 'moisture' in soil_data.columns:
        dry_regions = soil_data['moisture'] < 25
        for i in range(len(soil_data)):
            if dry_regions.iloc[i]:
                row = decoded_preds[i]
                if 'Urea' in row and 'Slow-Release' not in row:
                    # 替换尿素为缓释肥料
                    new_row = ['Slow-Release' if p == 'Urea' else p for p in row]
                    decoded_preds[i] = new_row[:3]
    
    # 规则3: 高有机质土壤减少氮肥推荐
    if 'organic_matter' in soil_data.columns:
        high_organic = soil_data['organic_matter'] > 3.0
        for i in range(len(soil_data)):
            if high_organic.iloc[i]:
                row = decoded_preds[i]
                if 'Urea' in row:
                    # 降低尿素优先级
                    new_row = [p for p in row if p != 'Urea']
                    if len(new_row) < 3:
                        new_row.append('Urea')
                    decoded_preds[i] = new_row[:3]
    
    return decoded_preds

# 主流程
def main():
    # 1. 创建训练数据集
    print("Generating training dataset (15,000 samples)...")
    train_df = create_synthetic_dataset(15000)
    
    # 2. 创建测试数据集 (250,000 samples)
    print("Generating test dataset (250,000 samples)...")
    test_df = create_synthetic_dataset(250000)
    
    # 3. 预处理训练数据
    print("Preprocessing training data...")
    train_df, le = preprocess_data(train_df)
    
    # 4. 特征工程
    print("Feature engineering...")
    train_df = feature_engineering(train_df)
    
    # 5. 准备训练数据
    feature_cols = [col for col in train_df.columns if col not in 
                   ['fertilizer', 'fertilizer_label', 'soil_type', 'region', 'pH_group', 'organic_level', 'id']]
    X_train = train_df[feature_cols]
    y_train = train_df['fertilizer_label']
    
    # 6. 准备测试数据
    print("Preparing test data...")
    # 对测试集应用相同的预处理和特征工程
    test_df, _ = preprocess_data(test_df)  # 注意：测试集不需要标签编码
    test_df = feature_engineering(test_df)
    X_test = test_df[feature_cols]
    
    # 7. 训练XGBoost模型
    print("Training model...")
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=len(le.classes_),
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42
    )
    
    # 使用训练集进行训练
    model.fit(X_train, y_train)
    
    # 8. 预测Top3肥料（分批处理以避免内存问题）
    print("Generating predictions for 250,000 samples...")
    batch_size = 50000
    all_preds = []
    
    for i in range(0, len(X_test), batch_size):
        print(f"  Predicting batch {i//batch_size + 1}/{(len(X_test)//batch_size)+1}")
        batch = X_test.iloc[i:i+batch_size]
        proba = model.predict_proba(batch)
        top3_indices = np.argsort(proba, axis=1)[:, ::-1][:, :3]
        all_preds.extend(top3_indices)
    
    # 9. 应用农学规则修正
    print("Applying agronomic rules...")
    top3_fertilizers = apply_agronomic_rules(all_preds, X_test, le)
    
    # 10. 创建符合要求的提交文件
    print("Creating submission file...")
    submission_data = []
    
    # 获取测试集的ID
    test_ids = test_df['id'].values
    
    for i in range(len(X_test)):
        # 获取当前样本的预测结果（3个肥料名称）
        fert_list = top3_fertilizers[i][:3]  # 只取前3个
        # 确保没有重复项
        unique_fert_list = []
        for fert in fert_list:
            if fert not in unique_fert_list:
                unique_fert_list.append(fert)
        # 用空格连接3个肥料名称
        fert_str = ' '.join(unique_fert_list[:3])
        
        submission_data.append({
            'id': test_ids[i],
            'Fertilizer Name': fert_str
        })
    
    submission_df = pd.DataFrame(submission_data)
    
    # 11. 验证行数
    if len(submission_df) != 250000:
        print(f"Warning: Submission has {len(submission_df)} rows, expected 250000")
    else:
        print(f"Submission has correct number of rows: 250,000")
    
    # 12. 特征重要性分析
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(importance.head(10))
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance.head(15))
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved to feature_importance.png")
    
    # 13. 保存结果
    submission_df.to_csv('fertilizer_predictions.csv', index=False)
    print("Predictions saved to fertilizer_predictions.csv")
    
    # 14. 在训练集上评估模型性能
    print("\nEvaluating model performance on training set...")
    train_preds = model.predict_proba(X_train)
    train_top3 = np.argsort(train_preds, axis=1)[:, ::-1][:, :3]
    train_map3 = map3_score(y_train.values, train_top3)
    print(f"Training MAP@3: {train_map3:.4f}")
    
    # 15. 样本预测解释
    if len(X_test) > 0:
        sample_idx = np.random.randint(len(X_test))
        explain_sample(X_test.iloc[sample_idx], model, le, test_df.iloc[sample_idx])
    
    return submission_df

def explain_sample(sample, model, le, test_row):
    """解释单个样本的预测结果"""
    # 确保输入是二维数组 (1个样本)
    proba = model.predict_proba(sample.values.reshape(1, -1))[0]  # 获取第一个样本的概率
    
    # 正确获取top3索引 - 处理一维数组
    top3_idx = np.argsort(proba)[::-1][:3]  # 直接取前3个最大值的索引
    
    # 转换索引为肥料名称
    top3_fert = le.inverse_transform(top3_idx)
    top3_proba = proba[top3_idx]
    
    print("\n=== Sample Prediction Explanation ===")
    print(f"Sample ID: {test_row['id']}")
    if 'nitrogen' in sample:
        print(f" - Nitrogen: {sample['nitrogen']:.1f} ppm")
    if 'phosphorus' in sample:
        print(f" - Phosphorus: {sample['phosphorus']:.1f} ppm")
    if 'potassium' in sample:
        print(f" - Potassium: {sample['potassium']:.1f} ppm")
    if 'pH' in sample:
        print(f" - pH: {sample['pH']:.1f}")
    if 'temperature' in sample:
        print(f" - Temperature: {sample['temperature']:.1f}°C")
    if 'moisture' in sample:
        print(f" - Moisture: {sample['moisture']:.1f}%")
    if 'organic_matter' in sample:
        print(f" - Organic Matter: {sample['organic_matter']:.1f}%")
    if 'soil_type' in test_row:  # 使用原始数据获取类别值
        print(f" - Soil Type: {test_row['soil_type']}")
    
    print("\nTop 3 Predictions:")
    for i, (fert, prob) in enumerate(zip(top3_fert, top3_proba), 1):
        print(f"{i}. {fert} (Probability: {prob:.2%})")
    
    # 农学合理性解释
    print("\nAgronomic Explanation:")
    if 'pH' in sample and sample['pH'] < 5.5:
        print("- Acidic soil (pH<5.5): Gypsum is recommended for pH adjustment")
    if 'phosphorus' in sample and sample['phosphorus'] < 10:
        print("- Low phosphorus soil (P<10ppm): Phosphorus fertilizers like DAP or SSP are needed")
    if 'potassium' in sample and sample['potassium'] < 20:
        print("- Low potassium soil (K<20ppm): Potassium fertilizers like MOP are needed")
    if 'moisture' in sample and sample['moisture'] < 25:
        print("- Dry conditions (moisture<25%): Slow-release fertilizers reduce nutrient loss")
    if 'organic_matter' in sample and sample['organic_matter'] > 3.0:
        print("- High organic matter soil (>3%): Nitrogen requirement may be reduced")

if __name__ == "__main__":
    submission = main()
    print("Process completed successfully!")

