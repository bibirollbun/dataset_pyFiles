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


# train_df['temp_range'] = train_df['maxtemp'] - train_df['mintemp']
# train_df['temp_diff_max_avg'] = train_df['maxtemp'] - train_df['temparature']
# train_df['temp_diff_avg_min'] = train_df['temparature'] - train_df['mintemp']


# test_df['temp_range'] = test_df['maxtemp'] - test_df['mintemp']
# test_df['temp_diff_max_avg'] = test_df['maxtemp'] - test_df['temparature']
# test_df['temp_diff_avg_min'] = test_df['temparature'] - test_df['mintemp']


train_df1 = train_df.dropna()

# test_df['winddirection'] = test_df['winddirection'].fillna(0)

test_df['winddirection'] = test_df['winddirection'].interpolate()




train_df1.info()


train= train_df1.drop(columns=['id','rainfall'])
label = train_df1['rainfall']
test = test_df.drop(columns=['id'])


label.value_counts()


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import optuna
import numpy as np
from sklearn.preprocessing import StandardScaler

x = train
y = label


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================== 逻辑回归超参数优化 ==================

def objective_lr(trial):
    
    params = {
    'C': trial.suggest_float('C', 1e-5, 100, log=True),  # 正则化强度参数
    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs']),  # 优化算法
    'max_iter': trial.suggest_int('max_iter', 200, 500),  # 增加最大迭代次数
    'random_state': 42
    }

    # 使用逻辑回归模型
    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)
    pred = model.predict_proba(X_test_scaled)[:, 1]  # 预测类别概率
    return roc_auc_score(y_test, pred)  # 返回AUC作为优化目标

study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objective_lr, n_trials=50)

best_lr = study_lr.best_params
best_lr['random_state'] = 42


print("逻辑回归最佳参数:", best_lr)

# ================== 训练最佳模型并评估 ==================

lr_model = LogisticRegression(**best_lr)
lr_model.fit(X_train_scaled, y_train)


lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, lr_pred)

print(f"逻辑回归模型的AUC: {auc:.5f}")


test.info()




probabilities = lr_model.predict_proba(test)[:, 1]  # 获取预测为1的概率

ids = test_df['id']


result = pd.DataFrame({
'id': ids,  
'rainfall': probabilities  # 输出为1的概率
})


result.to_csv('prediction_results.csv', index=False)


result




