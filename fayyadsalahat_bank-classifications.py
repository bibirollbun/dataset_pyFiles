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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


train_data.info()


test_data.info()


train_data.describe()


train_data["y"]


test_data.describe()


from sklearn.preprocessing import OrdinalEncoder


import typing
def convert_category_to_numb(datafarame :pd.DataFrame, series :pd.Series):
    enc = OrdinalEncoder()
    datafarame[[series]] = enc.fit_transform(datafarame[[series]])
    datafarame[series]= datafarame[series].astype("int64")




# for train data
for item in (train_data.loc[:, train_data.dtypes == object].columns):
    convert_category_to_numb(train_data,item)



# for test data
for item in (test_data.loc[:, test_data.dtypes == object].columns):
    convert_category_to_numb(test_data,item)



train_data


test_data


train_data_new = train_data.drop(["id"],axis=1)


train_data_new


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


numeric_coloumns = list(train_data_new.columns)
pipeline = ColumnTransformer([
    ("Standard", StandardScaler(), numeric_coloumns,)
])
scaled_data = pd.DataFrame(pipeline.fit_transform(train_data_new), columns=list(train_data_new.columns))



test_data_new = test_data.drop(["id"],axis=1)
numeric_coloumns = list(test_data_new.columns)
pipeline = ColumnTransformer([
    ("Standard", StandardScaler(), numeric_coloumns,)
])
scaled_data_test = pd.DataFrame(pipeline.fit_transform(test_data_new), columns=list(test_data_new.columns))



scaled_data


from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split


label = train_data["y"]
scaled_data = scaled_data.drop("y", axis=1)



train_x, test_x, train_y, test_y = train_test_split(scaled_data, label, test_size=0.2, random_state=42)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()


random_forest.fit(train_x, train_y)
random_forest_prediction = random_forest.predict(test_x)
random_forest_score = random_forest.score(train_x, train_y)
random_forest_accuracy = round(random_forest.score(train_x, train_y) * 100, 3)


print(random_forest_accuracy)
print(random_forest_score)


from sklearn.metrics import roc_auc_score


random_forest_prediction_porba = random_forest.predict_proba(test_x)


auc_score = roc_auc_score(test_y, random_forest_prediction_porba[:,1])


print("ROC-AUC:", auc_score)


test_pred_prob = random_forest.predict_proba(scaled_data_test)[:,1]


# Create submission file
submission = pd.DataFrame({
    'id': test_data.index,
    'y': test_pred_prob
})
submission.to_csv(f"bank_submission.csv", index=False)





