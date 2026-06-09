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



sample = pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2023-5th/sample_submission.csv', encoding='utf-8-sig')
train = pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2023-5th/train.csv', encoding='cp949')
test = pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2023-5th/test.csv', encoding='cp949')


train.isna().sum()


test.isna().sum()


train.info()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

num_atrribs = train.select_dtypes(include=np.number).columns.to_list()
cat_attribs = train.select_dtypes(include='object').columns.to_list()

num_atrribs.remove('price')



from sklearn.metrics import mean_squared_error
X = train.drop(columns='price', axis=1)
y = train['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore')



X_train_num = scaler.fit_transform(X_train[num_atrribs])
X_train_num = pd.DataFrame(X_train_num, columns = num_atrribs, index= X_train.index)
X_train_cat = encoder.fit_transform(X_train[cat_attribs]).toarray()
X_train_cat = pd.DataFrame(X_train_cat, columns=encoder.get_feature_names_out(), index=X_train.index)
X_train = pd.concat([X_train_num, X_train_cat], axis=1)

X_test_num = scaler.transform(X_test[num_atrribs])
X_test_num = pd.DataFrame(X_test_num, columns = num_atrribs, index= X_test.index)
X_test_cat = encoder.transform(X_test[cat_attribs]).toarray()
X_test_cat = pd.DataFrame(X_test_cat, columns=encoder.get_feature_names_out(), index=X_test.index)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)



estimators = {
    'lin_reg': LinearRegression(),
    'enet_reg': ElasticNet(alpha=0.6),
    'rf_reg': RandomForestRegressor(random_state=42,n_estimators=500),
    'xgb_reg': XGBRegressor(n_estimators=500)
}


results = {}
for name, model in estimators.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = rmse



results


best_model = min(results, key=results.get)
best_model


results[best_model]


test = pd.read_csv('/kaggle/input/big-data-analytics-certification-kr-2023-5th/test.csv', encoding='cp949')

test_num = scaler.transform(test[num_atrribs])
test_num = pd.DataFrame(test_num, columns = num_atrribs, index= test.index)
test_cat = encoder.transform(test[cat_attribs]).toarray()
test_cat = pd.DataFrame(test_cat, columns=encoder.get_feature_names_out(), index=test.index)
test = pd.concat([test_num, test_cat], axis=1)


model = XGBRegressor(n_estimators=500)
model.fit(X_train, y_train)
pred = model.predict(test)
pred


pred = pd.DataFrame(pred, columns=['price'],index=test.index)
pred = pred.reset_index()
pred.columns = ['id', 'price']
pred


sample.head(10)


result = pred.copy()
result.to_csv('result.csv', encoding='cp949', index=False)
result.to_csv('submission.csv', encoding='cp949', index=False)

pd.read_csv('submission.csv', encoding='cp949').head(10)




