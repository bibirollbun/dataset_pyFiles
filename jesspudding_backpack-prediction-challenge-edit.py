# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col = 'id')
# train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col = 'id')
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv",index_col = 'id')
submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


# train = pd.concat([train,train_extra],axis=0)


train.shape, train_extra.shape, submission.shape, test.shape


train.isna().sum().sum(), test.isna().sum().sum()


print(train.isna().sum())
print(test.isna().sum())

## testì—� ê²°ì¸¡ì§€ ì�ˆì�Œ. ê²°ì¸¡ì—� ëŒ€ë¹„í•´ì„œ í›ˆë ¨ì�„ ì‹œì¼œì•¼ í•¨


train.info()


train.describe(include = 'all')


train.columns


train.nunique()


train['Brand'].unique(), train['Brand'].value_counts()


sns.regplot( data = train, x = 'Weight Capacity (kg)', y = 'Price')


# sns.regplot( data = train_extra, x = 'Weight Capacity (kg)', y = 'Price')


# sns.boxplot( data = train_extra, x = 'Brand', y = 'Price')


train.loc[:,['Compartments','Weight Capacity (kg)','Price']].corr()


## ê²°ì¸¡ì¹˜ ì²˜ë¦¬ë°©ë²•
## trainì—� ê²°ì¸¡ì¹˜ ìœ ë¬´ í™•ì�¸
# ë²”ì£¼í˜• : ë¹ˆë�„ë†’ì�€ ê²ƒìœ¼ë¡œ ëŒ€ì²´, ê²°ì¸¡ ë²”ì£¼ ìƒ�ì„± (Unknown)
# ìˆ˜ì¹˜í˜• : í�‰ê· , ì¤‘ì•™ê°’
# ëª¨ë�¸ : imputation


train.isna().sum()


train.info()


train.select_dtypes('object').columns


object_columns = [col for col in train.columns if train[col].dtype == 'O']


for col in object_columns:    
    train.loc[train[col].isna(),col] = 'Unknown'
    test.loc[test[col].isna(),col] = 'Unknown'


train.loc[train['Weight Capacity (kg)'].isna(),'Weight Capacity (kg)'] = train['Weight Capacity (kg)'].mean()
test.loc[test['Weight Capacity (kg)'].isna(),'Weight Capacity (kg)'] = train['Weight Capacity (kg)'].mean()


train['Size'].unique() 


train['spec_weight'] = train['Weight Capacity (kg)'] / train['Compartments']
test['spec_weight'] = test['Weight Capacity (kg)'] / test['Compartments']


y = train['Price']
X = train.drop('Price',axis=1)
X_test = test.copy()


train.shape


kf = KFold(n_splits = 5, random_state = 42, shuffle = True)

scores = []
y_pred = np.zeros(X_test.shape[0])

for i,(train_idx, valid_idx) in enumerate(kf.split(X)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(iterations = 10000,
                          cat_features = object_columns,
                          learning_rate = 0.1)
    model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], early_stopping_rounds = 100, verbose = 0)

    y_pred_val  = model.predict(X_valid)
    y_pred += model.predict(X_test) / 5
    
    score = mean_squared_error(y_valid, y_pred_val, squared = False)
    scores.append(score)
    print(f'{i}th Fold RMSE: {np.round(score,3)}')
    # print(f'{i}th Fold, n_train: {train_idx},n_valid: {valid_idx}')

print(np.mean(scores))


# X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size = 0.2 , random_state = 42, shuffle = True)


# model = CatBoostRegressor(iterations = 10000,
#                           cat_features = object_columns,
#                           learning_rate = 0.1)
# model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], early_stopping_rounds = 100)


# y_pred = model.predict(X_test)
# y_pred

submission['Price'] = y_pred


y.mean(), y_pred.mean()


submission.to_csv('submission.csv',index=False)

