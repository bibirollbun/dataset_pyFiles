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


from sklearn.linear_model import LogisticRegression


!pwd


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

X = train_df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = train_df['Fertilizer Name']

X = pd.get_dummies(X, drop_first=True)
# print([X.columns])
# print(train_df['Soil Type'].unique())
# print(train_df['Crop Type'].unique())


model = LogisticRegression(max_iter=2000)
model.fit(X.values, y.values)

def predict_top3(X):
    sample = X.values.reshape(1, -1)
    probs = model.predict_proba(sample)[0]

    class_probs = list(zip(model.classes_, probs))

    top3 = sorted(class_probs, key=lambda x: x[1], reverse=True)[:3]

    return ' '.join([item[0] for item in top3])

# predict_top3(X.loc[0])


test_X = test_df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
test_X = pd.get_dummies(test_X, drop_first=True)
pred = test_X.apply(predict_top3, axis=1)


sample_submission_df = test_df[["id"]].copy()
sample_submission_df["Fertilizer Name"] = pred
sample_submission_df.to_csv("sample_submission.csv", index=False, encoding="UTF-8")
sample_submission_df




