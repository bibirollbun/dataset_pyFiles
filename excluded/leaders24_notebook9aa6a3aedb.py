import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

# ===================== 1. 加载数据并生成风险权重 =====================
X_train = pd.read_csv('../input/playground-series-s5e11/train.csv')
X_test = pd.read_csv('../input/playground-series-s5e11/test.csv')
testID = X_test['id']
y = X_train['loan_paid_back']

# 定义风险特征阈值（基于散点图分析的高风险区域）
def calculate_risk_weight(row):
    weight = 1.0  # 基础权重
    
    # 风险因子1：高负债收入比 + 低信用分
    if row['debt_to_income_ratio'] > 0.3 and row['credit_score'] < 600:
        weight *= 2.0
    
    # 风险因子2：高利率 + 低信用分
    if row['interest_rate'] > 15 and row['credit_score'] < 600:
        weight *= 2.0
    
    # 风险因子3：高贷款金额 + 低收入
    if row['loan_amount'] > 30000 and row['annual_income'] < 100000:
        weight *= 1.5
    
    # 风险因子4：未还款样本（目标变量为0）额外加权
    if row['loan_paid_back'] == 0:
        weight *= 1.8
    
    return weight

# 生成训练集样本权重
X_train['sample_weight'] = X_train.apply(calculate_risk_weight, axis=1)
sample_weights = X_train['sample_weight'].values

# ===================== 2. 特征工程（保留之前的优化） =====================
# 对数变换
X_train['annual_income_log'] = np.log1p(X_train['annual_income'])
X_train['loan_amount_log'] = np.log1p(X_train['loan_amount'])
X_test['annual_income_log'] = np.log1p(X_test['annual_income'])
X_test['loan_amount_log'] = np.log1p(X_test['loan_amount'])

# 特征交互
X_train['credit_interest_ratio'] = X_train['credit_score'] / (X_train['interest_rate'] + 1e-6)
X_test['credit_interest_ratio'] = X_test['credit_score'] / (X_test['interest_rate'] + 1e-6)

# 类别特征编码
cat_cols = X_train.select_dtypes(exclude=np.number).columns
oh = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
oh_X = oh.fit_transform(X_train[cat_cols])
oh_X = pd.DataFrame(oh_X, columns=oh.get_feature_names_out(cat_cols))
oh_t = oh.transform(X_test[cat_cols])
oh_t = pd.DataFrame(oh_t, columns=oh.get_feature_names_out(cat_cols))

# 合并特征
num_cols = ['annual_income_log', 'loan_amount_log', 'credit_score', 
            'interest_rate', 'debt_to_income_ratio', 'credit_interest_ratio']
X = pd.concat([X_train[num_cols], oh_X], axis=1)
X_test_final = pd.concat([X_test[num_cols], oh_t], axis=1)

# ===================== 3. 带样本权重的模型训练 =====================
model = xgb.XGBClassifier(
    random_state=42,
    learning_rate=0.08,
    n_estimators=250,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1
)

# 交叉验证（使用样本权重）
cv_score = cross_val_score(
    model, X, y, 
    cv=5, 
    scoring='roc_auc',
    fit_params={'sample_weight': sample_weights}  # 传入样本权重
)
print(f"带权重的交叉验证AUC：{cv_score.mean():.4f}")

# 训练模型
model.fit(X, y, sample_weight=sample_weights)

# ===================== 4. 预测与提交 =====================
final_pred = model.predict_proba(X_test_final)[:, 1]
submission = pd.DataFrame({'id': testID, 'loan_paid_back': final_pred})
submission.to_csv('weighted_submission.csv', index=False)

























