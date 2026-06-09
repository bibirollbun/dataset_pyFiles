# 学号: 2024423320103, 姓名: 戴梓建
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import time
import warnings
warnings.filterwarnings('ignore')

# MAP@5评估函数（优化版）
def map5_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)[:, :5]
    
    score = 0
    for i in range(len(y_true)):
        correct_idx = np.where(y_pred[i] == y_true[i])[0]
        if len(correct_idx) > 0:
            score += 1 / (correct_idx[0] + 1)
    return score / len(y_true)

# 1. 数据准备（使用Kaggle路径）
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 精简特征工程（只保留高效特征）
def preprocess(df):
    # 仅保留最有效的特征组合
    df['NPK_ratio'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    return df

train = preprocess(train)
test = preprocess(test)

# 编码类别特征
cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# 编码目标变量
le_fert = LabelEncoder()
train['Fertilizer Name'] = le_fert.fit_transform(train['Fertilizer Name'])

# 特征选择（基于重要性分析）
features = ['Temparature', 'Humidity', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Potassium', 'Phosphorous', 'NPK_ratio']
X = train[features]
y = train['Fertilizer Name']
X_test = test[features]

# 2. 优化LightGBM参数（加速关键）
params = {
    'boosting_type': 'goss',  # 使用GOSS提升速度[5,8](@ref)
    'objective': 'multiclass',
    'num_class': len(le_fert.classes_),
    'metric': 'multi_logloss',
    'learning_rate': 0.1,     # 提高学习率减少迭代次数
    'num_leaves': 63,         # 平衡精度和速度[6](@ref)
    'max_depth': 7,           # 限制深度防止过深[5](@ref)
    'min_data_in_leaf': 50,   # 增加叶节点最小样本数[6](@ref)
    'max_bin': 128,           # 减少直方图分桶数加速[6](@ref)
    'feature_fraction': 0.7,  # 特征采样加速[8](@ref)
    'verbosity': -1,
    'num_threads': 4,         # 利用多核并行[6](@ref)
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
}

# 3. 优化交叉验证策略
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 减少折数
test_preds = np.zeros((len(X_test), len(le_fert.classes_)))
oof_preds = np.zeros((len(X), len(le_fert.classes_)))
oof_labels = np.zeros(len(X))

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"\nFold {fold+1} - Starting at {time.strftime('%X')}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    
    # 使用回调函数和早停机制
    model = lgb.train(params, 
                      train_set,
                      num_boost_round=500,  # 减少最大迭代次数
                      valid_sets=[train_set, val_set],
                      callbacks=[
                          lgb.early_stopping(stopping_rounds=30, verbose=True),
                          lgb.log_evaluation(period=50)
                      ])
    
    # 验证集预测
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    oof_labels[val_idx] = y_val
    
    # 测试集预测（使用模型最佳迭代）
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits
    
    # 验证集评估
    val_top5 = np.argsort(val_pred, axis=1)[:, ::-1][:, :5]
    map5_val = map5_score(y_val, val_top5)
    print(f"Fold {fold+1} MAP@5: {map5_val:.4f} - Time: {time.time()-start_time:.1f}s")

# 整体OOF评估
oof_top5 = np.argsort(oof_preds, axis=1)[:, ::-1][:, :5]
map5_oof = map5_score(oof_labels, oof_top5)
print(f"\nOverall OOF MAP@5: {map5_oof:.4f} - Total Time: {time.time()-start_time:.1f}s")

# 4. 生成Top5预测
test_top5 = np.argsort(test_preds, axis=1)[:, ::-1][:, :5]
test_fertilizers = le_fert.inverse_transform(test_top5.reshape(-1))
test_fertilizers = test_fertilizers.reshape(test_top5.shape)

# 5. 创建提交文件
submission = test[['id']].copy()
submission['Fertilizer Name'] = [' '.join(row) for row in test_fertilizers.astype(str)]

# 保存结果
submission.to_csv('submission.csv', index=False)
print("\nOptimized submission file saved as submission.csv")


