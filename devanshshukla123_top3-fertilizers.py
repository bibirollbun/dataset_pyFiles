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


#data input
train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
train=train.drop(columns=['id'])
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.model_selection import train_test_split


# data preprocesssing
train['Temp_Humid'] = train['Temparature'] * train['Humidity']
test['Temp_Humid'] = test['Temparature'] * test['Humidity']

features=['Temp_Humid','Moisture','Soil Type','Crop Type','Nitrogen','Potassium','Phosphorous']
x1=train[features]
y2=train['Fertilizer Name']
x1=x1.copy()
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
x1['Soil Type'] = le1.fit_transform(x1['Soil Type'])
test['Soil Type'] = le1.fit_transform(test['Soil Type'])
x1['Crop Type'] = le2.fit_transform(x1['Crop Type'])
test['Crop Type'] = le2.fit_transform(test['Crop Type'])
y2 = le3.fit_transform(y2)




x_train, x_val, y_train, y_val = train_test_split(x1, y2, test_size=0.2, random_state=42)

# XGBoost 
xgb = XGBClassifier(
    objective='multi:softprob',  
    num_class=len(np.unique(y2)),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
xgb.fit(x_train, y_train)

probs = xgb.predict_proba(test[features])




top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1] #sorting the probab list to get top3




top3_str = [' '.join(le3.inverse_transform(row)) for row in top3_indices]
#final submission
submission = pd.DataFrame({
    "ID": test["id"],
    "Fertilizer Name": top3_str
})
submission.to_csv("submission.csv", index=False)
print('done submission')

