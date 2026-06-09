# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sub_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train_df.head()


train_df.info()


train_df1 = train_df.dropna()

# # 用前一个有效值填充 winddirection 缺失值
# test_df['winddirection'] = test_df['winddirection'].fillna(method='ffill')



train_df1.info()


import pandas as pd
from scipy.stats import pearsonr

# Calculate p-values of correlations between 'rainfall' and other columns
correlations = train_df.drop(columns=['id']).apply(lambda col: pearsonr(train_df['rainfall'], col)[1])

# Display the results
print(correlations)



train= train_df1.drop(columns=['id','rainfall','day','mintemp','winddirection'])
label = train_df1['rainfall']
test = test_df.drop(columns=['id','day','mintemp','winddirection'])


label.value_counts()


import lightgbm as lgb
import numpy as np
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 1. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.2, random_state=42)

# 2. 计算正负样本比例（用于 scale_pos_weight）
neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_pos_weight_value = neg_count / pos_count  # 直接计算实际比例

# 3. 定义目标函数（用于Optuna的超参数优化）
def objective(trial):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',  # 设置评估指标为 'auc'（可以使用其他指标，如 'binary_error'，'logloss' 等）
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 
                                               scale_pos_weight_value*0.5, 
                                               scale_pos_weight_value*1.5),  # 围绕实际比例搜索
        'verbosity': -1
    }

    # 使用回调函数实现早停
    callbacks = [
        lgb.early_stopping(stopping_rounds=10),
        lgb.log_evaluation(period=0)
    ]

    model = lgb.train(
        params,
        train_set=lgb.Dataset(X_train, y_train),
        num_boost_round=1000,  # 设置较大的迭代次数，由早停控制
        valid_sets=[lgb.Dataset(X_test, y_test)],
        valid_names=['valid'],  # 添加验证集名称，便于调试
        callbacks=callbacks
    )

    # 预测概率值
    pred_proba = model.predict(X_test)
    return roc_auc_score(y_test, pred_proba)


# 4. Optuna优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, timeout=3600)

# 5. 使用最佳参数训练最终模型
best_params = study.best_params
best_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1
})

final_model = lgb.train(
    best_params,
    train_set=lgb.Dataset(X_train, y_train),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# 6. 评估与阈值优化
probs = final_model.predict(X_test)

# 默认阈值 0.5 的评估
print("\n默认阈值 0.5 的评估结果：")
print(classification_report(y_test, (probs > 0.5).astype(int)))

# 寻找最佳阈值（基于F1-score）
from sklearn.metrics import f1_score
thresholds = np.linspace(0.1, 0.5, 50)
best_threshold = 0.5
best_f1 = 0
for thresh in thresholds:
    current_f1 = f1_score(y_test, (probs > thresh).astype(int))
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = thresh

print(f"\n最佳阈值: {best_threshold:.3f} (F1-score: {best_f1:.3f})")
print(classification_report(y_test, (probs > best_threshold).astype(int)))

# 输出特征重要性
lgb.plot_importance(final_model, importance_type='gain', figsize=(10, 6))


# 预测
probs = final_model.predict(test)

# 获取测试集的 'id' 列，假设 test_df 中有 'id' 列
ids = test_df['id']

# 构造结果 DataFrame
result = pd.DataFrame({
    'id': ids,
    'rainfall': probs  # 输出预测为 1 的概率
})

# 保存结果到 CSV 文件
result.to_csv('prediction_results.csv', index=False)

# 打印结果以查看
print(result)




