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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train_df.head()
len(train_df)


print(*list(train_df.columns),sep='\n')


train_df['loan_paid_back'].value_counts()
# 预测目标， 注意到正例和负例有较大的不均衡


object_cols = train_df.select_dtypes(include=['object']).columns.tolist()
object_cols


for col in object_cols:
    print(f'{col}:')
    print(sorted(train_df[col].unique().tolist()),'\n')


X = train_df.drop(columns=['loan_paid_back'])
y = train_df['loan_paid_back']


X['interest_per_year']=X['loan_amount']*X['interest_rate']


numeric_cols = train_df.select_dtypes(exclude=['object']).columns.tolist()


numeric_cols.remove('id')
numeric_cols


for col in object_cols:
    X[col] = X[col].astype('category')
X.drop('id',axis=1,inplace=True)


X.columns


numeric_df = train_df[numeric_cols]
corr_matrix = numeric_df.corr('spearman')


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True)
plt.title('heat_map')
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
print(len(X_train),len(X_val))


import xgboost as xgb


model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        enable_categorical=True,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50
    )


model.fit(
    X_train, y_train,
    eval_set=[[X_train,y_train],(X_val, y_val)],  
    verbose=True
)


results = model.evals_result()

# 提取 AUC 数据
epochs = range(len(results['validation_0']['auc']))
    
# 训练集 AUC
plt.plot(epochs, results['validation_0']['auc'], 
             label='Train AUC', linewidth=2, color='blue')
    
# 验证集 AUC
plt.plot(epochs, results['validation_1']['auc'], 
             label='Validation AUC', linewidth=2, color='red')


sample_submission.head()




