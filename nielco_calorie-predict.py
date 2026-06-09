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


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train.columns


cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp']
X_train = train[cols]
y_train = train["Calories"]
X_test = test[cols]


X_train['Height_m'] = X_train['Height'] / 100
X_train['BMI'] = X_train['Weight'] / (X_train['Height_m'] ** 2)
X_train['Age_Group'] = X_train['Age'].apply(lambda x: 'young' if x < 30 else 'middle' if x < 60 else 'old')
X_train['HR_per_min'] = X_train['Heart_Rate'] / X_train['Duration']
X_train['Body_Load'] = X_train['Weight'] * X_train['Duration'] * X_train['Heart_Rate'] / 10000
X_train['High_Temp'] = (X_train['Body_Temp'] > 37.5).astype(int)
X_train['Is_Male'] = (X_train['Sex'] == 'male').astype(int)

X_test['Height_m'] = X_test['Height'] / 100
X_test['BMI'] = X_test['Weight'] / (X_test['Height_m'] ** 2)
X_test['Age_Group'] = X_test['Age'].apply(lambda x: 'young' if x < 30 else 'middle' if x < 60 else 'old')
X_test['HR_per_min'] = X_test['Heart_Rate'] / X_test['Duration']
X_test['Body_Load'] = X_test['Weight'] * X_test['Duration'] * X_test['Heart_Rate'] / 10000
X_test['High_Temp'] = (X_test['Body_Temp'] > 37.5).astype(int)
X_test['Is_Male'] = (X_test['Sex'] == 'male').astype(int)


X_train


from xgboost import XGBRegressor

model = XGBRegressor(
    objective='reg:squarederror',  # 기본 회귀
    max_depth=5,
    learning_rate=0.1,
    n_estimators=300,  # 더 복잡한 패턴 학습 가능
    subsample=0.8,     # 과적합 방지
    colsample_bytree=0.8,
    random_state=42
)


cat_cols = X_train.select_dtypes(include='object').columns.tolist()

X = pd.get_dummies(X_train, columns=cat_cols)
X_test = pd.get_dummies(X_test, columns=cat_cols)


model.fit(X, y_train)


prediction = model.predict(X_test)


prediction = prediction.clip(min=0)


(prediction<0).sum()


submission["Calories"] = prediction
submission.to_csv("submission.csv", index = False)


from sklearn.ensemble import RandomForestRegressor

model2 = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)


model2.fit(X, y_train)


prediction2 = model2.predict(X_test)


submission["Calories"] = prediction2
submission.to_csv("submission.csv", index = False)


print(prediction)
print(prediction2)




