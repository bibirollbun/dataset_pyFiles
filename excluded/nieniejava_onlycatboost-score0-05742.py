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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.simplefilter('ignore')

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("\nTrain Info:")
train.info()
print("\nTest Info:")
test.info()
print("\nTrain Describe:")
train.describe()


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

print("\nSex Distribution:")
print(train['Sex'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=train)
plt.title('Distribution of Sex')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Sex', y='Calories', data=train)
plt.title('Calories by Sex')
plt.show()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Sex_encoded'] = le.fit_transform(train['Sex'])

corr = train[numerical_features + ['Calories', 'Sex_encoded']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# train = train.drop('Sex_encoded', axis=1)

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)

train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')

X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])
X_test = test.drop(columns=['id'])

FEATURES = X.columns.tolist()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
import time

FOLDS = 5
RANDOM_STATE = 42

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

cat_features = ['Sex']  
model_params = {
    # 核心参数
    'iterations': 10000,          # 增加总迭代次数
    'learning_rate': 0.02,        # 适当提高学习率加速收敛
    'depth': 16,                  # 增加树深捕捉复杂模式
    'l2_leaf_reg': 0.5,           # 降低 L2 正则化强度，减少约束
    
    # 过拟合控制
    'grow_policy': 'Lossguide',   # 按损失优化节点生长方向
    'min_data_in_leaf': 50,       # 防止叶节点过小
    'random_strength': 1.5,       # 增加特征选择随机性
    
    # 类别特征处理
    'cat_features': ['Sex'],
    'one_hot_max_size': 10,       # 对低基数类别特征自动独热编码
    'max_ctr_complexity': 4,      # 允许更高阶类别特征组合
    
    # 采样策略
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 1.2,   # 提高温度增强样本多样性
    
    # 训练配置
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 500, # 增加早停耐心
    'use_best_model': True,       # 强制使用最佳模型
    'random_seed': 42,
    'verbose': 100
}

results = {
    'oof': np.zeros(len(train)),
    'pred': np.zeros(len(test)),
    'rmsle': [],
    'train_times': []
}

print("=== Training CatBoost ===")


for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold+1}")
    

    x_train, y_train = X.iloc[train_idx], y[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
    

    model = CatBoostRegressor(**model_params)
    

    start_time = time.time()
    

    model.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid),
        use_best_model=True
    )
    

    train_time = time.time() - start_time
    results['train_times'].append(train_time)
    

    oof_pred = model.predict(x_valid)
    test_pred = model.predict(X_test)
    

    results['oof'][valid_idx] = oof_pred
    results['pred'] += test_pred / FOLDS
    

    rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
    results['rmsle'].append(rmsle)
    
    print(f"Fold {fold+1} RMSLE: {rmsle:.4f}")
    print(f"Training time: {train_time:.1f} sec")


mean_rmsle = np.mean(results['rmsle'])
std_rmsle = np.std(results['rmsle'])
print(f"\nCatBoost Performance:")
print(f"Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")
print(f"Average training time: {np.mean(results['train_times']):.1f} sec")


feature_importance = model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('CatBoost Feature Importance')
plt.show()

y_preds = np.expm1(results['pred'])
y_preds = np.clip(y_preds, 1, 314)  

submission['Calories'] = y_preds
submission.to_csv('catboost_submission.csv', index=False)

print("\nSubmission Summary:")
print(f"Predictions Mean: {y_preds.mean():.2f}")
print(f"Predictions Median: {np.median(y_preds):.2f}")
print("\nSubmission Head:")
print(submission.head())




