import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler # For normalizing
from xgboost import XGBClassifier


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_data = train_data.drop_duplicates()


train_data.head()


train_nan_columns = train_data.columns[train_data.isna().any()].tolist()
train_nan_columns


train_data['Time_spent_Alone'] = train_data['Time_spent_Alone'].fillna(train_data['Time_spent_Alone'].median())
train_data['Stage_fear'] = train_data['Stage_fear'].fillna(train_data['Stage_fear'].mode())
train_data['Social_event_attendance'] = train_data['Social_event_attendance'].fillna(train_data['Social_event_attendance'].mean())
train_data['Going_outside'] = train_data['Going_outside'].fillna(train_data['Going_outside'].mean())
train_data['Drained_after_socializing'] = train_data['Drained_after_socializing'].fillna(train_data['Drained_after_socializing'].mode())
train_data['Friends_circle_size'] = train_data['Friends_circle_size'].fillna(train_data['Friends_circle_size'].median())
train_data['Post_frequency'] = train_data['Post_frequency'].fillna(train_data['Post_frequency'].median())


test_nan_columns = test_data.columns[test_data.isna().any()].tolist()
test_nan_columns


test_data['Time_spent_Alone'] = test_data['Time_spent_Alone'].fillna(test_data['Time_spent_Alone'].median())
test_data['Stage_fear'] = test_data['Stage_fear'].fillna(test_data['Stage_fear'].mode())
test_data['Social_event_attendance'] = test_data['Social_event_attendance'].fillna(test_data['Social_event_attendance'].mean())
test_data['Going_outside'] = test_data['Going_outside'].fillna(test_data['Going_outside'].mean())
test_data['Drained_after_socializing'] = test_data['Drained_after_socializing'].fillna(test_data['Drained_after_socializing'].mode())
test_data['Friends_circle_size'] = test_data['Friends_circle_size'].fillna(test_data['Friends_circle_size'].median())
test_data['Post_frequency'] = test_data['Post_frequency'].fillna(test_data['Post_frequency'].median())


x = train_data.drop('Personality', axis=1)
y = train_data['Personality']


# Initialize encoder
encoder = LabelEncoder()

# Convert labels to numbers
y = encoder.fit_transform(y)


# sanity check
print(x.columns)
print(len(x)==len(y))
print(len(x.columns))


# Splitting the train data into train and validation
train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=44)


print('Length of training data: ',len(train_data))
print('Length of testing data: ',len(test_data))


print('Length of train data: ',len(train_x))
print('Length of val data: ', len(val_x))


columns = train_x.columns


columns


train_x.describe()


# converting the Stage_fear column to one hot-encoding
train_x = pd.get_dummies(train_x, columns=['Stage_fear'])
val_x = pd.get_dummies(val_x, columns=['Stage_fear'])
test_data = pd.get_dummies(test_data, columns=['Stage_fear'])


# converting the Drained_after_socializing column to one hot-encoding
train_x = pd.get_dummies(train_x, columns=['Drained_after_socializing'])
val_x = pd.get_dummies(val_x, columns=['Drained_after_socializing'])
test_data = pd.get_dummies(test_data, columns=['Drained_after_socializing'])


# dropping id as it is just sequence and not contributing anything to result
train_x = train_x.drop('id', axis=1)
val_x = val_x.drop('id', axis=1)
test_data = test_data.drop('id', axis=1)


train_x = train_x.astype(np.float32)
val_x = val_x.astype(np.float32)
train_y = train_y.astype(np.int32)
val_y = val_y.astype(np.int32)


input_dim = len(train_x.columns)
n_classes = 2


# Train XGBClassifier
model = XGBClassifier(objective='binary:logistic', random_state=42, n_estimators=200)


model.fit(train_x, train_y,eval_set=[(val_x, val_y)])


y_pred_prob = model.predict_proba(test_data)


y_pred = y_pred_prob.argmax(axis=1)


y_pred[1]


# For going from one hot encoding to labels
y_res = encoder.inverse_transform(y_pred)


# Sanity check for result and test_data
len(test_data)==len(y_res)
print('Len test data: ',len(test_data))
print('Len y_res: ',len(y_res))


# Save submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission["Personality"] = y_res
submission.to_csv("submission.csv", index=False)
print('Sumission done!')
submission.head()

