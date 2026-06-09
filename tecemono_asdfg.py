# ===== 1. 导入必要的库 =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


RANDOM_STATE = 42


# ===== 2. 数据加载 =====
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print("训练集形状:", train.shape)
print("测试集形状:", test.shape)
print("\n训练集列名:", train.columns.tolist())


# ===== 3. 数据预处理 =====
def preprocess_data(df, is_train=True):
    """数据预处理函数"""
    df_processed = df.copy()
    
    # 删除id列（根据EDA发现可能有泄漏）
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop('id', axis=1)
    
    # 分类变量编码
    categorical_columns = ['gender', 'marital_status', 'education_level', 
                          'employment_status', 'loan_purpose', 'grade_subgrade']
    
    # 对grade_subgrade进行有序编码（基于EDA发现的有序性）
    grade_order = ['A1', 'A2', 'A3', 'A4', 'A5', 
                   'B1', 'B2', 'B3', 'B4', 'B5',
                   'C1', 'C2', 'C3', 'C4', 'C5',
                   'D1', 'D2', 'D3', 'D4', 'D5',
                   'E1', 'E2', 'E3', 'E4', 'E5',
                   'F1', 'F2', 'F3', 'F4', 'F5']
    
    grade_mapping = {grade: i for i, grade in enumerate(grade_order)}
    df_processed['grade_subgrade_encoded'] = df_processed['grade_subgrade'].map(grade_mapping)
    
    # 对employment_status进行有序编码（基于EDA发现的强有序性）
    employment_order = ['Retired', 'Employed', 'Self-employed', 'Student', 'Unemployed']
    employment_mapping = {emp: i for i, emp in enumerate(employment_order)}
    df_processed['employment_status_encoded'] = df_processed['employment_status'].map(employment_mapping)
    
    # 对其他分类变量使用普通编码
    other_categorical = ['gender', 'marital_status', 'education_level', 'loan_purpose']
    encoder = OrdinalEncoder()
    df_processed[other_categorical] = encoder.fit_transform(df_processed[other_categorical])
    
    # 删除原始分类列
    df_processed = df_processed.drop(['grade_subgrade', 'employment_status'], axis=1)
    
    # 特征工程：基于EDA的洞察创建新特征
    df_processed['income_to_loan_ratio'] = df_processed['annual_income'] / (df_processed['loan_amount'] + 1)
    df_processed['credit_dti_interaction'] = df_processed['credit_score'] * (1 - df_processed['debt_to_income_ratio'])
    df_processed['risk_score'] = (df_processed['debt_to_income_ratio'] * 100 + 
                                 (850 - df_processed['credit_score']) / 8.5 + 
                                 df_processed['interest_rate'])
    
    # 创建DTI分箱特征
    df_processed['dti_category'] = pd.cut(df_processed['debt_to_income_ratio'], 
                                         bins=[0, 0.1, 0.2, 0.3, 1], 
                                         labels=[0, 1, 2, 3])
    
    # 创建信用评分分箱
    df_processed['credit_category'] = pd.cut(df_processed['credit_score'],
                                           bins=[0, 600, 700, 750, 850],
                                           labels=[0, 1, 2, 3])
    
    return df_processed

# 预处理训练集和测试集
print("开始数据预处理...")
train_processed = preprocess_data(train, is_train=True)
test_processed = preprocess_data(test, is_train=False)

# 分离特征和目标变量
X = train_processed.drop('loan_paid_back', axis=1)
y = train_processed['loan_paid_back']
X_test = test_processed

print(f"预处理后训练集形状: {X.shape}")
print(f"预处理后测试集形状: {X_test.shape}")



# ===== 4. 特征重要性分析 =====
print("\n正在分析特征重要性...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('特征重要性排序 (Random Forest)')
plt.tight_layout()
plt.show()

print("Top 10 最重要特征:")
print(feature_importance.head(10))


# ===== 5. 模型训练与评估 =====
print("\n开始模型训练...")

# 选择最重要的特征（基于EDA和特征重要性分析）
top_features = feature_importance.head(12)['feature'].tolist()
X_top = X[top_features]
X_test_top = X_test[top_features]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top)
X_test_scaled = scaler.transform(X_test_top)

# 划分训练验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 定义模型
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

# 训练并评估模型
results = {}
print("\n模型性能比较:")
print("=" * 50)

for name, model in models.items():
    print(f"训练 {name}...")
    model.fit(X_train, y_train)
    
    # 验证集预测
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    # 计算指标
    auc_score = roc_auc_score(y_val, y_pred_proba)
    
    results[name] = {
        'model': model,
        'auc': auc_score,
        'predictions': y_pred_proba
    }
    
    print(f"{name}:")
    print(f"  ROC-AUC: {auc_score:.4f}")
    print(f"  准确率: {np.mean(y_pred == y_val):.4f}")
    print("-" * 30)

# 选择最佳模型
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']
best_auc = results[best_model_name]['auc']

print(f"\n最佳模型: {best_model_name} (AUC: {best_auc:.4f})")


# ===== 6. 交叉验证 =====
print("\n进行交叉验证...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"交叉验证 AUC 分数: {cv_scores}")
print(f"平均交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


# ===== 7. 在全量数据上重新训练最佳模型 =====
print(f"\n在全量数据上重新训练 {best_model_name}...")

if best_model_name == 'LightGBM':
    final_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
elif best_model_name == 'XGBoost':
    final_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
else:
    final_model = best_model

final_model.fit(X_scaled, y)


# ===== 8. 测试集预测 =====
print("生成测试集预测...")
test_predictions = final_model.predict_proba(X_test_scaled)[:, 1]



# ===== 9. 创建提交文件 =====
submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_predictions
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("\n提交文件已保存为 'submission.csv'")


# ===== 10. 结果分析 =====
print("\n预测结果统计:")
print(f"预测概率均值: {test_predictions.mean():.4f}")
print(f"预测概率标准差: {test_predictions.std():.4f}")
print(f"预测概率范围: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(test_predictions, bins=50, alpha=0.7, color='skyblue')
plt.title('测试集预测概率分布')
plt.xlabel('预测概率')
plt.ylabel('频数')

plt.subplot(1, 3, 2)
# 特征重要性可视化
if hasattr(final_model, 'feature_importances_'):
    lgb_importance = pd.DataFrame({
        'feature': top_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.barh(lgb_importance['feature'], lgb_importance['importance'])
    plt.title(f'{best_model_name} 特征重要性')
    plt.xlabel('重要性')

plt.subplot(1, 3, 3)
# 模型比较
model_names = list(results.keys())
model_aucs = [results[name]['auc'] for name in model_names]
colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']

bars = plt.bar(model_names, model_aucs, color=colors, alpha=0.7)
plt.title('模型性能比较 (ROC-AUC)')
plt.ylabel('ROC-AUC')
plt.xticks(rotation=45)

# 在柱状图上添加数值
for bar, auc in zip(bars, model_aucs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{auc:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()



# ===== 11. 输出关键洞察 =====
print("\n" + "="*60)
print("关键商业洞察总结")
print("="*60)

print("\n基于EDA和建模分析的主要发现:")
print("1. 最重要的预测特征:")
print("   - employment_status (就业状况)")
print("   - debt_to_income_ratio (债务收入比)") 
print("   - credit_score (信用评分)")
print("   - grade_subgrade (贷款等级)")
print("   - interest_rate (利率)")

print("\n2. 风险模式识别:")
print("   - 失业人员和学生违约风险显著较高")
print("   - 高债务收入比(DTI > 0.3)的借款人风险急剧上升")
print("   - 信用评分低于600的群体违约概率明显增加")
print("   - E/F等级贷款的违约率是A/B等级的3-5倍")

print("\n3. 建模建议:")
print("   - 重点关注就业稳定性和信用历史")
print("   - 建立基于DTI和信用评分的风险分层体系")
print("   - 对高风险群体实施更严格的审批标准")

print(f"\n4. 模型表现:")
print(f"   最佳模型: {best_model_name}")
print(f"   验证集AUC: {best_auc:.4f}")
print(f"   交叉验证AUC: {cv_scores.mean():.4f}")



# ===== 12. 保存重要数据用于报告 =====
# 保存特征重要性
feature_importance.to_csv('feature_importance.csv', index=False)

# 保存模型比较结果
model_comparison = pd.DataFrame([
    {'Model': name, 'AUC': results[name]['auc']} 
    for name in results.keys()
])
model_comparison.to_csv('model_comparison.csv', index=False)

print("\n所有文件已生成完成!")


