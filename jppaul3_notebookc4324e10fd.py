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


train =pd.read_csv("/kaggle/input/gds-exercise-3-2025-/train.csv")


train.head(5)


df =train.drop(["S_CONSTELLATION_ENG","S_CONSTELLATION","S_DEC_TXT","S_RA_TXT","S_DEC_STR","S_RA_STR","S_NAME_HIP","S_NAME","S_NAME","P_UPDATE","P_YEAR","P_DETECTION","P_DISCOVERY_FACILITY","releasedate","pl_pubdate","decstr","rastr","sy_refname","st_refname","pl_refname","rowid","hostname","pl_letter","hd_name","hip_name","tic_id","gaia_id","discoverymethod","disc_refname","disc_pubdate","disc_locale","disc_telescope","disc_instrument"],axis =1)
df.head(5)


df =df.dropna(axis=1,thresh=4000)


df.shape


from sklearn.preprocessing import LabelEncoder
for col in df.columns:
  # Check if the column's data type is 'object'
  if df[col].dtype == 'object':
    # Initialize LabelEncoder
    le = LabelEncoder()
    try:
        df[col] = le.fit_transform(df[col])
    except TypeError as e:
        print(f"Error encoding column '{col}': {e}")
        df[col] = df[col].fillna("Nan")
        df[col] = le.fit_transform(df[col])



y = df["P_HABITABLE"]  #output feature


# Fill NaN values with the mean of each column
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())


df = df.drop("P_HABITABLE", axis=1)


df.shape


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred)
acc =accuracy_score(y_test,y_pred)
print(f"F1 Score: {f1}")
print(f"Accuracy Score: {acc}")



test = pd.read_csv("/kaggle/input/gds-exercise-3-2025-/test.csv")


id = test["rowid"]


test = test[df.columns]
test.shape


for col in test.columns:
  # Check if the column's data type is 'object'
  if test[col].dtype == 'object':
    # Initialize LabelEncoder
    le = LabelEncoder()
    try:
        test[col] = le.fit_transform(test[col])
    except TypeError as e:
        print(f"Error encoding column '{col}': {e}")
        test[col] = test[col].fillna("Nan")
        test[col] = le.fit_transform(test[col])


# Fill NaN values with the mean of each column
for col in test.columns:
    if pd.api.types.is_numeric_dtype(test[col]):
        test[col] = test[col].fillna(test[col].mean())


predicted =rf_classifier.predict(test)


submission = pd.DataFrame({ 'ID': id,'P_HABITABLE': predicted})


for index, rows in submission.iterrows():
    if rows["ID"] == np.nan:
        rows["ID"] = submission["ID"].max()


submission.isna().sum()


submission.to_csv("submission.csv", index =False)

