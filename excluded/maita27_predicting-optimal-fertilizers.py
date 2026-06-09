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


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train


categorical = ['Soil Type', 'Crop Type']
numerical = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


from sklearn.preprocessing import OneHotEncoder
    
def preprocess_df(df, categorical, numerical, mode = 'train'):
    df = df.copy()
    df.ffill(inplace=True)

    for column in categorical:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        transformed = encoder.fit_transform(df[[column]])

        # Make column names like "Soil Type_Clay", "Soil Type_Sandy"
        col_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
        one_hot_df = pd.DataFrame(transformed, columns=col_names, index=df.index)

        df = pd.concat([df.drop(columns=[column]), one_hot_df], axis=1)
    if mode == 'train':
        fertilizer_names = {
            '28-28': 0,
            '17-17-17': 1,
            '10-26-26': 2,
            'DAP': 3,
            '20-20': 4,
            '14-35-14': 5,
            'Urea': 6
            }
        df['Fertilizer Name'] = df['Fertilizer Name'].map(fertilizer_names)

    return df


train = preprocess_df(train, categorical, numerical)


train


x_train = train.drop(columns = ['id', 'Fertilizer Name'])
y_train = train['Fertilizer Name']


import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100,
                          max_depth=2,
                          learning_rate=1)

model.fit(x_train, y_train)


test = preprocess_df(test, categorical, numerical, mode = 'test')


x_test = test.drop(columns = ['id'])
test['preds'] = model.predict(x_test)


fertilizer_numbers = {
    0 : '28-28',
    1 : '17-17-17',
    2 : '10-26-26',
    3 : 'DAP',
    4 : '20-20',
    5 : '14-35-14',
    6 : 'Urea'
}
test['preds'] = test['preds'].map(fertilizer_numbers)


proba = model.predict_proba(x_test)


top3 = np.argsort(proba, axis = 1)[:, -3:][:, ::-1]


mapped_top3 = np.array([[fertilizer_numbers[i] for i in row] for row in top3])


predictions = []
for row in mapped_top3:
    pred = ''
    for string in row:
        pred = pred + string + ' '
    predictions.append(pred)


test['preds'] = predictions


submission = pd.DataFrame({
    'id' : test['id'],
    'Fertilizer Name' : test['preds']
})


submission.to_csv('submission.csv', index = False)




