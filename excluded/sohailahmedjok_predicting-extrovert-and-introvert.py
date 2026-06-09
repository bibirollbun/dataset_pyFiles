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


#loading all data

train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submition = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


# checking columns and understanding the data
train_data.head()


# checking null values
train_data.isnull().sum()

# filling null values with zero in numerical category
train_data[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]] = train_data[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]].fillna(0)  

# filling null values to "NO" in Object or String Columns
train_data[["Stage_fear", "Drained_after_socializing"]] = train_data[["Stage_fear", "Drained_after_socializing"]].fillna("No") 

#checking null values after filling, For verification
train_data.isnull().sum()


# Encoding String Data of yes = 1 and no = 0 for better machine learning efficiency
train_data[["Stage_fear", "Drained_after_socializing"]] = train_data[["Stage_fear", "Drained_after_socializing"]].replace({"Yes": 1, "No": 0})

# Encoding Personality Type as Extrovert = 1 & Introvert = 0
train_data["Personality"] = train_data["Personality"].replace({"Extrovert": 1, "Introvert": 0})

# Checking data
train_data.head()



# importing machine learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Droping not needed coloumns from X variable 
X = train_data.drop(["Personality", "id"], axis = 1)

# selecting target variable from the dataset
y = train_data["Personality"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)

# Now is the time for evalution
y_predict = model.predict(X_val)
print("Accuracy: ", accuracy_score(y_val, y_predict))


test_data[["Stage_fear", "Drained_after_socializing"]] = test_data[["Stage_fear", "Drained_after_socializing"]].fillna("No") 

# Encoding String Data of yes = 1 and no = 0 for better machine learning efficiency "For test data"
test_data[["Stage_fear", "Drained_after_socializing"]] = test_data[["Stage_fear", "Drained_after_socializing"]].replace({"Yes": 1, "No": 0})

# filling null values with zero in numerical category
test_data[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]] = test_data[["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]].fillna(0)  

test_data.head()


X_test = test_data.drop('id', axis=1)
test_preds = model.predict(X_test)
decoded_preds = pd.Series(test_preds).replace({1: "Extrovert", 0: "Introvert"})


# submitting the result as csv file

submission = pd.DataFrame({
    'id': test_data['id'],
    'Personality': decoded_preds
})
submission.to_csv('submission.csv', index=False)


print(submission.head())





