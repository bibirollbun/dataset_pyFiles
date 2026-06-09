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


#load the data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


#fill numerical missing values with median of the column
numerical_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside','Friends_circle_size', 'Post_frequency']
for column in numerical_columns :
    median_value = train_df[column].median() 
    train_df[column] = train_df[column].fillna(median_value)
    test_df[column] = test_df[column].fillna(test_df[column].median())
    


#fill categorical missing values with mode of the column
categorical_columns = ['Stage_fear' , 'Drained_after_socializing']
for column in categorical_columns :
    median_value = train_df[column].mode() 
    train_df[column] = train_df[column].fillna(median_value)
    test_df[column] = test_df[column].fillna(test_df[column].mode()[0])


#covert categorical vales into numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['Stage_fear'] = le.fit_transform(train_df['Stage_fear'])
train_df['Drained_after_socializing'] = le.fit_transform(train_df['Drained_after_socializing'])
train_df['Personality'] = le.fit_transform(train_df['Personality'])


#split data into training and testing data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality']

X_train, X_value, y_train, y_value = train_test_split(X, y, test_size=0.2, random_state=42)

#train model 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_value)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_value, rf_preds))
print(classification_report(y_value, rf_preds))



# convert the features of test data to string
test_df['Stage_fear'] = test_df['Stage_fear'].astype(str)
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].astype(str)


from sklearn.preprocessing import LabelEncoder
import numpy as np

# Function to safely label encode with 'Unknown' handling
def safe_label_encode(train_col, test_col):
    le = LabelEncoder()
    le.fit(train_col)

    # Add 'Unknown' if not present
    test_col = test_col.apply(lambda x: x if x in le.classes_ else 'Unknown')
    if 'Unknown' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'Unknown')

    return le.transform(test_col)

# Apply on test set
test_df['Stage_fear'] = safe_label_encode(train_df['Stage_fear'], test_df['Stage_fear'])
test_df['Drained_after_socializing'] = safe_label_encode(train_df['Drained_after_socializing'], test_df['Drained_after_socializing'])



#make predictions on test data
x_test = test_df.drop(['id'] , axis=1)
test_pred = rf_model.predict(x_test)
#convert the features and label to its original form
test_pred_labels = le.inverse_transform(test_pred)
# make the submission
submission_df = sample_submission.copy()
submission_df['Personality'] = test_pred_labels 
submission_df.to_csv("submission.csv" , index= False)


#check your submission_df
submission_df.head()

