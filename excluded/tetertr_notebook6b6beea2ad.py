import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. 数据准备
np.random.seed(42)
num_samples = 1000
num_features = 5
# 定义英文特征名称（根据实际含义对应调整）
feature_names = ['Nitrogen Content', 'Phosphorus Content', 'Potassium Content', 'pH Value', 'Humidity']  
fertilizer_types = ['NPK_20-20-20', 'Urea', 'DAP', 'MOP', 'Diammonium Phosphate']

# 生成随机特征数据
X = np.random.rand(num_samples, num_features)
# 生成随机肥料类型标签
y = np.random.choice(fertilizer_types, num_samples)

# 标签编码：用完整类别列表初始化编码器
le = LabelEncoder()
le.fit(fertilizer_types)
y_encoded = le.transform(y)

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 2. 模型训练
model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric=['mlogloss', 'map@3'],
    n_estimators=500,
    learning_rate=0.05
)
model.fit(X_train, y_train)

# 3. 获取 Top5 预测
probabilities = model.predict_proba(X_test)
top5_indices = np.argsort(probabilities, axis=1)[:, -5:][:, ::-1]
top5_predictions = le.classes_[top5_indices]

# 4. 计算 MAP@5 分数
def calculate_map_at_5(y_true_encoded, y_pred_top5_str, le):
    map5 = 0.0
    for true_label_enc, pred_labels in zip(y_true_encoded, y_pred_top5_str):
        true_label_str = le.inverse_transform([true_label_enc])[0]
        true_set = {true_label_str}
        
        ap = 0.0
        hit = 0.0
        for i, pred_label in enumerate(pred_labels):
            if pred_label in true_set:
                hit += 1.0
                ap += hit / (i + 1.0)
        if hit > 0:
            ap /= hit
        map5 += ap
    return map5 / len(y_true_encoded)

map5_score = calculate_map_at_5(y_test, top5_predictions, le)
print(f"MAP@5 Score: {map5_score:.4f}")

# 5. 特征重要性分析及可视化（全英文显示）
feature_importances = model.feature_importances_
print("Feature Importances:", feature_importances)

# 可视化特征重要性（英文标题、坐标轴标签、特征名称）
plt.bar(feature_names, feature_importances)
plt.xlabel('Feature Name')
plt.ylabel('Importance')
plt.title('Feature Importance Distribution')
plt.xticks(rotation=45)  # 旋转 x 轴标签，避免长名称重叠
plt.tight_layout()  # 调整布局，让标签完整显示
plt.show()

# 6. 结果解释（农学解释可按需保留中文或调整，这里示范保留中文辅助理解）
def explain_prediction(sample_index, X_test, y_test, top5_predictions, le, feature_names):
    sample_features = X_test[sample_index]
    sample_top5 = top5_predictions[sample_index]
    true_label_enc = y_test[sample_index]
    true_label_str = le.inverse_transform([true_label_enc])[0]  
    
    print(f"\nSample {sample_index} - True Label: {true_label_str}")
    print("Top 5 Predictions:", sample_top5)
    print("Feature Values:")
    for name, value in zip(feature_names, sample_features):
        print(f"  {name}: {value:.4f}")
    
    # 根据假设的特征含义，做简单农学解释示例（保留中文便于理解业务逻辑）
    np_content, p_content, k_content, ph_value, humidity = sample_features
    print("\n农学合理性解释：")
    # 以 NPK_20-20-20 这类平衡型肥料为例，假设适合氮磷钾相对均衡场景
    if true_label_str == 'NPK_20-20-20':
        if (0.15 < np_content < 0.3) and (0.15 < p_content < 0.3) and (0.15 < k_content < 0.3):
            print("  样本氮、磷、钾含量相对均衡，推荐 NPK_20-20-20 这类平衡型肥料符合土壤养分需求")
        else:
            print("  虽然推荐了 NPK_20-20-20，但当前土壤氮磷钾含量可能并非特别均衡，需结合实际情况验证")
    # 若真实标签是 Urea（尿素，高氮），假设适合低氮场景
    elif true_label_str == 'Urea':
        if np_content < 0.2:
            print("  土壤氮含量较低，尿素（Urea）是高氮肥料，推荐符合补充氮素需求")\
    # 可继续扩展其他肥料类型的解释逻辑...

explain_prediction(0, X_test, y_test, top5_predictions, le, feature_names)

# 7. 保存结果与模型
np.savetxt("fertilizer_predictions.txt", top5_predictions, fmt='%s')
model.save_model("fertilizer_predictor_model.json")

