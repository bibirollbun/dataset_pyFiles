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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


train=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


test_ids=test['id']
y_train=train['accident_risk']
train=train.drop(columns=['id','accident_risk'])
test=test.drop(columns=['id'])


bool_cols=['road_signs_present','public_road','holiday','school_season']
print(f"Converting boolean columns:{bool_cols}")
for col in bool_cols:
    train[col]=train[col].astype(int)
    test[col]=test[col].astype(int)
object_cols=['road_type','lighting','weather','time_of_day']
print(f"One-hot encoding categorical columns: {object_cols}") 
X_train = pd.get_dummies(train,columns=object_cols, drop_first=False)
X_test = pd.get_dummies(test,columns=object_cols, drop_first=False)
print("Aligning columns between training and test sets...")
X_train, X_test = X_train.align(X_test,join='outer',axis=1,fill_value=0)
print(f"Training data shape after processing: {X_train.shape}")
print(f"Test data shape after processing: {X_test.shape}")


model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)


model.fit(X_train, y_train)
predictions = model.predict(X_test)
submission_df = pd.DataFrame({
    'id': test_ids,
    'accident_risk': predictions
})
submission_df['accident_risk'] = np.clip(predictions, 0, 1)


submission_df.to_csv('submission.csv', index=False)
print("--- submission.csv created successfully! ---")
print(submission_df.head())

