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

# Load datasets
train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
test = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
val = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/val.csv')
sample_submission = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')

# Basic Overview
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("Validation Shape:", val.shape)

# First few rows
print("\nTrain Head:\n", train.head())

# Missing values
print("\nMissing values in Train:\n", train.isnull().sum())

# Data types
print("\nData Types:\n", train.dtypes)



#To predict the Lap_Time_Seconds for each rider in the test.csv file based on the features available.




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')

# ğŸ”� 2.1 Check & Fill Missing Values
# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Fill numerical missing values with mean
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Fill categorical missing values with most frequent (mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# ğŸ”¡ 2.2 Encode Categorical Columns
# You can apply Label Encoding to each categorical column
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # store the encoder in case needed later

# ğŸ“� 2.3 Feature Scaling (Optional but good practice for some models)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ğŸ§¹ 2.4 Drop Irrelevant or Leakage Columns
# Drop columns like Unique ID or anything not useful for modeling
cols_to_drop = ['Unique ID', 'Rider_ID']  # add/remove based on EDA
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# âœ… Final shape of preprocessed dataset
print("Shape after preprocessing:", df.shape)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/train.csv")
test_df = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/test.csv")
val_df = pd.read_csv("/kaggle/input/burnout-datathon-ieeecsmuj/val.csv")

#  Set target variable
y = train_df['Lap_Time_Seconds']
X = train_df.drop(columns=['Lap_Time_Seconds', 'Unique ID', 'Rider_ID'], errors='ignore')


num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include='object').columns


num_imputer = SimpleImputer(strategy='mean')
X[num_features] = num_imputer.fit_transform(X[num_features])
test_df[num_features] = num_imputer.transform(test_df[num_features])
val_df[num_features] = num_imputer.transform(val_df[num_features])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_features] = cat_imputer.fit_transform(X[cat_features])
test_df[cat_features] = cat_imputer.transform(test_df[cat_features])
val_df[cat_features] = cat_imputer.transform(val_df[cat_features])


label_encoders = {}
for col in cat_features:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    test_df[col] = encoder.transform(test_df[col])
    val_df[col] = encoder.transform(val_df[col])
    label_encoders[col] = encoder


scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])
test_df[num_features] = scaler.transform(test_df[num_features])
val_df[num_features] = scaler.transform(val_df[num_features])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

#  XGBoost Model Training
model = XGBRegressor(n_estimators=500,
                     learning_rate=0.05,
                     max_depth=10,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     random_state=42,
                     tree_method='hist')
model.fit(X_train, y_train)

y_pred_valid = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print("Validation RMSE:", rmse)


test_predictions = model.predict(test_df.drop(columns=['Unique ID', 'Rider_ID'], errors='ignore'))

submission = pd.DataFrame({
    'Unique ID': test_df['Unique ID'],
    'Lap_Time_Seconds': test_predictions
})
submission.to_csv("submission.csv", index=False)




