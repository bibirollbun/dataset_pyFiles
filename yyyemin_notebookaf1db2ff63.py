import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
np.random.seed(42)

# ---------------------------
# 1. 数据加载与探索
# ---------------------------
print("加载数据...")
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

print(f"训练集形状: {train.shape}")
print(f"测试集形状: {test.shape}")
print(f"示例提交文件形状: {sample_submission.shape}")

# 查看数据集基本信息
print("\n训练集基本信息:")
train.info()

# 分离特征和目标变量
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
test_ids = test['id']
X_test = test.drop('id', axis=1)

# ---------------------------
# 2. 特征工程 - 分类特征编码
# ---------------------------
print("\n开始特征工程...")

# 分离数值特征和分类特征
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"数值特征: {numeric_cols}")
print(f"分类特征: {categorical_cols}")

# 对分类特征进行标签编码
print("\n对分类特征进行标签编码...")
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])  # 应用相同的编码到测试集

# ---------------------------
# 3. 目标变量编码
# ---------------------------
print("\n执行目标变量编码...")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ---------------------------
# 4. 划分训练集和验证集
# ---------------------------
print("\n划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------------------
# 5. 验证LightGBM版本
# ---------------------------
print(f"\nLightGBM版本: {lgb.__version__}")

# ---------------------------
# 6. XGBoost模型训练
# ---------------------------
print("\n训练XGBoost模型...")
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': len(encoder.classes_),
    'eval_metric': ['mlogloss'],
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_lambda': 1.2,
    'reg_alpha': 0.05,
    'seed': 42
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

evals_result = {}
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,
    evals_result=evals_result,
    verbose_eval=50
)

# 评估XGBoost模型
print("\n评估XGBoost模型...")
xgb_pred_proba = xgb_model.predict(dval)
xgb_pred = np.argmax(xgb_pred_proba, axis=1)
xgb_accuracy = accuracy_score(y_val, xgb_pred)
print(f"XGBoost验证集准确率: {xgb_accuracy:.4f}")

# 手动计算MAP@5
def calculate_map_at_k(y_true, y_pred_proba, k=5):
    map_score = 0
    n_samples = len(y_true)
    
    for i in range(n_samples):
        true_label = y_true[i]
        top_k_indices = np.argsort(y_pred_proba[i])[::-1][:k]
        
        correct_count = 0
        precision_sum = 0
        
        for pos, pred_idx in enumerate(top_k_indices, start=1):
            if pred_idx == true_label:
                correct_count += 1
                precision_sum += correct_count / pos
        
        if correct_count > 0:
            map_score += precision_sum / correct_count
    
    return map_score / n_samples

xgb_map5 = calculate_map_at_k(y_val, xgb_pred_proba, k=5)
print(f"XGBoost验证集MAP@5分数: {xgb_map5:.4f}")

# ---------------------------
# 7. LightGBM模型训练（修复版本）
# ---------------------------
print("\n训练LightGBM模型...")
lgb_params = {
    'objective': 'multiclass',
    'num_class': len(encoder.classes_),
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.05,
    'reg_lambda': 1.2,
    'seed': 42,
    'verbosity': -1  # 减少日志输出
}

# 使用原生API训练模型
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_val, label=y_val)

lgb_model = lgb.train(
    params=lgb_params,
    train_set=lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_eval],
    valid_names=['validation'],
    early_stopping_rounds=50,
    verbose_eval=50,
    callbacks=[lgb.log_evaluation(50)]  # 明确设置回调
)

# 评估LightGBM模型
print("\n评估LightGBM模型...")
lgb_pred_proba = lgb_model.predict(X_val)
lgb_pred = np.argmax(lgb_pred_proba, axis=1)
lgb_accuracy = accuracy_score(y_val, lgb_pred)
print(f"LightGBM验证集准确率: {lgb_accuracy:.4f}")

lgb_map5 = calculate_map_at_k(y_val, lgb_pred_proba, k=5)
print(f"LightGBM验证集MAP@5分数: {lgb_map5:.4f}")

# ---------------------------
# 8. 模型融合（简单平均）
# ---------------------------
print("\n进行模型融合...")
blend_pred_proba = (xgb_pred_proba + lgb_pred_proba) / 2
blend_pred = np.argmax(blend_pred_proba, axis=1)
blend_accuracy = accuracy_score(y_val, blend_pred)
print(f"融合模型验证集准确率: {blend_accuracy:.4f}")

blend_map5 = calculate_map_at_k(y_val, blend_pred_proba, k=5)
print(f"融合模型验证集MAP@5分数: {blend_map5:.4f}")

# ---------------------------
# 9. 对测试集进行预测
# ---------------------------
print("\n对测试集进行预测...")
xgb_test_pred_proba = xgb_model.predict(dtest)
lgb_test_pred_proba = lgb_model.predict(X_test)

# 融合预测结果
blend_test_pred_proba = (xgb_test_pred_proba + lgb_test_pred_proba) / 2
best_pred_indices = np.argmax(blend_test_pred_proba, axis=1)
best_pred_labels = encoder.inverse_transform(best_pred_indices)

# ---------------------------
# 10. 生成提交文件
# ---------------------------
print("\n生成提交文件...")
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': best_pred_labels
})

# 验证提交文件格式
print("\n提交文件前5行:")
print(submission.head())
print("\n提交文件列名:")
print(submission.columns.tolist())

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存为 'submission.csv'，包含 {len(submission)} 行")

# ---------------------------
# 11. 特征重要性可视化
# ---------------------------
plt.figure(figsize=(12, 5))

# XGBoost特征重要性
plt.subplot(1, 2, 1)
xgb.plot_importance(xgb_model, height=0.8, importance_type='gain')
plt.title('XGBoost特征重要性')
plt.tight_layout()

# LightGBM特征重要性
plt.subplot(1, 2, 2)
lgb.plot_importance(lgb_model, height=0.8)
plt.title('LightGBM特征重要性')
plt.tight_layout()

plt.savefig('feature_importance.png')
plt.show()

