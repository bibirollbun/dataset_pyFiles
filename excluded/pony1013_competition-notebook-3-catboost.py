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
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler 

train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test_id = test_df['id']
train_df = train_df.drop(columns='id')
test_df = test_df.drop(columns='id')
y_train = train_df['Calories']
X_train = train_df.drop(columns='Calories')
cat_features = ['Sex'] 


from sklearn.model_selection import train_test_split

# 分割数据集 (测试集占比 20%，随机种子固定保证可复现性)
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.2,   # 验证集比例，常用 0.2-0.3
    random_state=42, # 固定随机种子确保每次分割结果一致
    stratify=y_train # 若目标变量分布不均衡时建议启用分层抽样（尤其分类任务）
)

print("训练集样本数:", len(X_train_split))
print("验证集样本数:", len(X_val))


model = CatBoostRegressor(
    iterations=15000,
    learning_rate=0.7,
    depth=7,
    l2_leaf_reg=0.5,
    eval_metric='RMSE',
    min_data_in_leaf=100,
    loss_function='RMSE',
    verbose=500,
    grow_policy='SymmetricTree',
    task_type= 'GPU',
    cat_features=cat_features
)



from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

model.fit(
    X=X_train,
    y=y_train,
    eval_set = (X_val, y_val)
)

y_pred = model.predict(X_val)
rmsle = np.sqrt(((np.log1p(y_pred) - np.log1p(y_val))**2).mean())
print(f"RMSLE: {rmsle:.4f}")

s = model.predict(test_df)
mean_value = np.mean(s)
s = [mean_value if value < 0 else value for value in s]
submission = pd.DataFrame(
    {
        'id' : test_id, 
        'Calories' : s
    }
)
print(submission)
for car in submission['Calories']:
    if car < 0:
        print("\n Negative value found : ", car)

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Data submitted!")




