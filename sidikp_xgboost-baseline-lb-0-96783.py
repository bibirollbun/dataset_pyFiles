import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
import matplotlib.patheffects as pe

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import xgboost as xgb

pd.options.display.max_columns = 500

import os
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


print("="*25, '\n Data Train')
display(train.head(2))
display(train.shape)

print("="*25,  '\n Data Test')
display(test.head(2))
display(test.shape)


train['is_test'] = 0
test['is_test'] = 1

full_df = pd.concat([train, test])
num_col = full_df.select_dtypes(include=['number']).columns.to_list()
cat_col = full_df.select_dtypes(include=['object']).columns.to_list()

le = LabelEncoder()
for col in cat_col:
    full_df[col] = le.fit_transform(full_df[col])

train = full_df[full_df['is_test'] == 0].copy()
test = full_df[full_df['is_test'] == 1].copy()

for df in [train, test]:
    df.drop(columns=['is_test', 'id'], inplace = True)

X = train.drop(columns=['y'])
y = train['y']

display(train.head(2))
display(test.head(2))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

num_round = 20000
evallist = [(dtrain, 'train'), (dtest, 'test')]

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
}

plst = list(params.items())

model = xgb.train(
    plst,
    dtrain,
    num_round,
    evallist,
    verbose_eval=100,
    early_stopping_rounds=250
)


submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission.head(2)


dprediction_data = xgb.DMatrix(test[X_train.columns])
prediction = model.predict(dprediction_data)

submission['y'] = prediction
submission.to_csv('solution.csv', index=False)
pd.read_csv('solution.csv').head(5)

