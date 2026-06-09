# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

#Download Kaggle Dataset
# ! pip install kaggle
# ! mkdir ~/.kaggle
# ! cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
# ! chmod 600 ~/.kaggle/kaggle.json
# ! kaggle competitions download playground-series-s5e2
# ! unzip /content/playground-series-s5e2.zip
# ! pip install dask[dataframe]

#External Libraries
! pip install sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
!pip install catboost
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


! kaggle competitions download -c playground-series-s5e2


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
print("size of train_data ",len(train_data))
training_extra_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
print("size of training_extra_data ", len(training_extra_data))
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test_data = test_data.drop(columns=["id"])
print("size of test_data ", len(test_data))
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
print("size of sample_submission ", len(sample_submission))


training_data = pd.concat([train_data, training_extra_data], ignore_index=True)
print("size before drop na", len(training_data))
training_data = training_data.dropna()
training_data = training_data.drop(columns=['id'])
print("size after drop na", len(training_data))


# Load dataset
data = training_data  # Replace with actual dataset path
submission = test_data  # Replace with actual submission file path

# Handle missing values
def handle_missing_values(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical with mode
        else:
            df[col].fillna(df[col].median(), inplace=True)  # Fill numerical with median
    return df

data = handle_missing_values(data)
submission = handle_missing_values(submission)

# Encode categorical variables
label_encoders = {}
categorical_columns = data.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    submission[col] = le.transform(submission[col])  # Transform submission data with the same encoder
    label_encoders[col] = le

# Splitting data
X = data.drop(columns=["Price"])
y = data["Price"]
X_train = X
y_train = y


train_data = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'boosting_type': 'gbdt'
}

model = lgb.train(params, train_data, num_boost_round=1000)


submission["Price"] = model.predict(submission.drop(columns=["Price"], errors='ignore'))
sample_submission['Price'] = submission["Price"]


sample_submission.to_csv("lgbm.csv", index=False)




