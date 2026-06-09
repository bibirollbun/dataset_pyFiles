import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
from scipy.stats import rankdata

# 数据导入
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
train


# 构造新变量
for df in [train, test]:
    df['temp_change_rate'] = df['temparature'].diff().fillna(0)
    df['humidity_change_trend'] = df['humidity'].diff().fillna(0)

RMV = ['rainfall','id']
FEATURES = [c for c in train.columns if c not in RMV]


# 贝叶斯优化函数
def xgb_cv(max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
    """贝叶斯优化目标函数"""
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'n_estimators': 10000,
        'eval_metric': 'auc',
        'early_stopping_rounds': 80
    }
    
    cv_scores = []
    train_auc_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(train):
        x_train = train.loc[train_idx, FEATURES]
        y_train = train.loc[train_idx, 'rainfall']
        x_val = train.loc[val_idx, FEATURES]
        y_val = train.loc[val_idx, 'rainfall']
        
        model = XGBClassifier(**params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            verbose=False
        )
        
        train_pred = model.predict_proba(x_train)[:, 1]
        val_pred = model.predict_proba(x_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        
        cv_scores.append(val_auc)
        train_auc_scores.append(train_auc)
    
    # 计算过拟合指标
    mean_train_auc = np.mean(train_auc_scores)
    mean_val_auc = np.mean(cv_scores)
    overfit_gap = mean_train_auc - mean_val_auc
    print(f"Train AUC = {mean_train_auc:.3f}, Val AUC = {mean_val_auc:.3f}, Overfitting = {overfit_gap:.3f}")
    
    # # 如果过拟合严重（训练AUC比验证高0.05以上），惩罚最终得分
    # if overfit_gap > 0.05:
    #     return mean_val_auc * 0.9  # 对过拟合的参数组合进行惩罚
    # elif overfit_gap > 0.1:
    #     return mean_val_auc * 0.8
    
    return mean_val_auc


# 定义参数空间
pbounds = {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.8, 1),
    'colsample_bytree': (0.8, 1),
    'gamma': (0, 1),
    'reg_alpha': (0.1, 1),
    'reg_lambda': (0.1, 1)
}

# 运行贝叶斯优化
optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=100, n_iter=300)

# 获取最佳参数
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
print('最佳参数:', best_params)


# 5折交叉验证预测
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

fold_train_auc = []
fold_val_auc = []

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index, FEATURES]
    y_train = train.loc[train_index, "rainfall"]
    x_valid = train.loc[test_index, FEATURES]
    y_valid = train.loc[test_index, "rainfall"]
    x_test = test[FEATURES]

    model = XGBClassifier(
        **best_params,
        n_estimators=10000,
        early_stopping_rounds=80,
        eval_metric='auc'
    )
    
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )
    
    # 记录训练集预测结果
    train_pred = model.predict_proba(x_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    
    # 验证集预测
    val_pred = model.predict_proba(x_valid)[:, 1]
    val_auc = roc_auc_score(y_valid, val_pred)
    
    # 存储结果
    oof_xgb[test_index] = val_pred
    fold_train_auc.append(train_auc)
    fold_val_auc.append(val_auc)
    
    # 测试集预测
    pred_xgb += model.predict_proba(x_test)[:, 1]

pred_xgb /= FOLDS
sample = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sample.rainfall = pred_xgb
sample.to_csv("submission_without_ensemble.csv",index=False)
print(sample.head())


# 过拟合分析
mean_train_auc = np.mean(fold_train_auc)
mean_val_auc = np.mean(fold_val_auc)
overfit_gap = mean_train_auc - mean_val_auc

print("\n" + "="*50)
print(f"Average Training AUC:   {mean_train_auc:.4f}")
print(f"Average Validation AUC: {mean_val_auc:.4f}")
print(f"Overfitting Gap:        {overfit_gap:.4f}")

if overfit_gap > 0.05:
    print("\n⚠️ 警告：检测到严重过拟合！")
    print("建议操作：")
    print("- 增加正则化系数（reg_alpha/reg_lambda）")
    print("- 降低模型复杂度（max_depth）")
    print("- 增加更多训练数据")
    print("- 进行特征选择")
elif overfit_gap > 0.02:
    print("\nℹ️ 注意：检测到中度过拟合")
    print("考虑进行适度的正则化或降低复杂度")
else:
    print("\n✅ 模型表现出良好的泛化性能")

# 可视化训练与验证AUC分布
plt.figure(figsize=(10, 5))
plt.boxplot([fold_train_auc, fold_val_auc], 
           labels=['Training AUC', 'Validation AUC'])
plt.title("AUC Distribution Comparison")
plt.ylabel("AUC Score")
plt.grid(True)
plt.show()


# 集成结果并保存最终结果
best_public = pd.read_csv("/kaggle/input/0-96245-lda-lgs-ensemble-for-rainfall-pred/submission.csv")
best_public = best_public.rainfall.values

sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub.rainfall = -0.05 * rankdata( pred_xgb ) + 1.05 * rankdata( best_public )
sub.rainfall = rankdata( sub.rainfall ) / len(sub)

# 进行条件赋值
sub['rainfall'] = sub['rainfall'].apply(lambda x: 1 if x > 0.95 else 0 if x < 0.05 else x)

# 保存结果
sub.to_csv(f"submission_ensemble.csv",index=False)
print(sub.head())

