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


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train


train.dtypes


train = train.drop(columns=['id'])


# Encode target
fertilizer_le = LabelEncoder()
train['Fertilizer_Label'] = fertilizer_le.fit_transform(train['Fertilizer Name'])

# Encode Soil and Crop
soil_le = LabelEncoder()
crop_le = LabelEncoder()

train['Soil_Type_Label'] = soil_le.fit_transform(train['Soil Type'])
train['Crop_Type_Label'] = crop_le.fit_transform(train['Crop Type'])

test['Soil_Type_Label'] = soil_le.transform(test['Soil Type'])
test['Crop_Type_Label'] = crop_le.transform(test['Crop Type'])


X = train[[
    'Temparature', 'Humidity', 'Moisture',
    'Nitrogen', 'Phosphorous', 'Potassium',
    'Soil_Type_Label', 'Crop_Type_Label'
]]
y = train['Fertilizer_Label']

X_test = test[[
    'Temparature', 'Humidity', 'Moisture',
    'Nitrogen', 'Phosphorous', 'Potassium',
    'Soil_Type_Label', 'Crop_Type_Label'
]]


model = XGBClassifier(
    n_estimators=129,
    max_depth=10,
    learning_rate=0.2002,
    subsample=0.9347,
    colsample_bytree=0.5184,
    gamma=0.027,
    min_child_weight=9,
    reg_alpha=1.31e-5,
    reg_lambda=0.1834,
    objective='multi:softprob',
    num_class=7,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X, y)


proba = model.predict_proba(X_test)
top_3_preds = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
top_3_labels = np.vectorize(lambda x: fertilizer_le.inverse_transform([x])[0])(top_3_preds)
final_preds = [' '.join(row) for row in top_3_labels]


submission = submission[['id']].copy()
submission['Fertilizer Name'] = final_preds
submission.to_csv('submission.csv', index=False)


