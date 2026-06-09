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
import os


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




train_path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
test_path = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
submission_path = "/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv"


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df = pd.read_csv(submission_path)


print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print("Sample Submission Shape:", submission_df.shape)




print("Train Data Preview:")
display(train_df.head())

print("\nTest Data Preview:")
display(test_df.head())


print("\nTrain Data Info:")
train_df.info()




missing_values = train_df.isnull().sum() / len(train_df) * 100
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)


print("Columns with Missing Values:\n")
display(missing_values)





train_df.drop(columns=['tce_match', 'mrd_hct'], inplace=True)
test_df.drop(columns=['tce_match', 'mrd_hct'], inplace=True)

print("Remaining Columns:", train_df.shape[1])




num_cols = list(set(train_df.select_dtypes(include=['float64', 'int64']).columns) & set(test_df.columns))

train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())
test_df[num_cols] = test_df[num_cols].fillna(test_df[num_cols].median())

print("Missing values in train:", train_df.isnull().sum().sum())
print("Missing values in test:", test_df.isnull().sum().sum())




cat_cols = list(set(train_df.select_dtypes(include=['object']).columns) & set(test_df.columns))


train_df[cat_cols] = train_df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))
test_df[cat_cols] = test_df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

print("Missing values in train after filling categorical:", train_df.isnull().sum().sum())
print("Missing values in test after filling categorical:", test_df.isnull().sum().sum())



from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

for col in cat_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

print("Categorical features encoded successfully!")




train_df.drop(columns=['ID'], inplace=True)
test_df.drop(columns=['ID'], inplace=True)

print("Unnecessary columns dropped!")



from sklearn.preprocessing import StandardScaler


target_cols = ['efs', 'efs_time']


X_train = train_df.drop(columns=target_cols, errors='ignore')  
y_train = train_df[target_cols]  


X_test = test_df[X_train.columns]  


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed successfully!")



import pandas as pd


train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

drop_cols = ["sex_match", "vent_hist", "gvhd_proph", "dri_score"]
train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

num_cols = train_df.select_dtypes(include=["float64", "int64"]).columns
train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())

common_num_cols = [col for col in num_cols if col in test_df.columns]
test_df[common_num_cols] = test_df[common_num_cols].fillna(test_df[common_num_cols].median())

cat_cols = train_df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)


train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)


print(f"Missing values in train: {train_df.isnull().sum().sum()}")
print(f"Missing values in test: {test_df.isnull().sum().sum()}")



print(train_df.columns)




X = train_df.drop(columns=['ID', 'efs'])  
y = train_df['efs']  


X = pd.get_dummies(X, drop_first=True)


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MSE: {mse}")


X_test = test_df.drop(columns=['ID'])  
X_test = pd.get_dummies(X_test, drop_first=True)  
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_test_pred = model.predict(X_test)

submission = pd.DataFrame({'ID': test_df['ID'], 'prediction': y_test_pred})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")



from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


val_predictions = model.predict(X_val)


rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f'RMSE: {rmse}')

r2 = r2_score(y_val, val_predictions)
print(f'R² Score: {r2}')


