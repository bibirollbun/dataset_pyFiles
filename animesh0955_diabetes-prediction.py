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


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.head()


df = df.drop(columns=['id'])


df.info()


import matplotlib.pyplot as plt 
import seaborn as sns 



plt.figure(figsize=(15, 15))

plt.subplot(3,3,1)
sns.histplot(df['age'], bins=10, kde=True)

plt.subplot(3,3,2)
sns.histplot(df['alcohol_consumption_per_week'], bins=10, kde=True)

plt.subplot(3,3,3)
sns.histplot(df['physical_activity_minutes_per_week'], bins=10, kde=True)

plt.subplot(3,3,4)
sns.histplot(df['sleep_hours_per_day'], bins=10, kde=True)

plt.subplot(3,3,5)
sns.histplot(df['screen_time_hours_per_day'], bins=10, kde=True)

plt.subplot(3,3,6)
sns.histplot(df['bmi'], bins=10, kde=True)

plt.subplot(3,3,7)
sns.histplot(df['waist_to_hip_ratio'], bins=10, kde=True)

plt.subplot(3,3,8)
sns.histplot(df['systolic_bp'], bins=10, kde=True)

plt.subplot(3,3,9)
sns.histplot(df['diastolic_bp'], bins=10, kde=True)



plt.figure(figsize=(15, 15))

plt.subplot(3,3,1)
sns.histplot(df['heart_rate'], bins=10, kde=True)

plt.subplot(3,3,2)
sns.histplot(df['cholesterol_total'], bins=10, kde=True)

plt.subplot(3,3,3)
sns.histplot(df['hdl_cholesterol'], bins=10, kde=True)

plt.subplot(3,3,4)
sns.histplot(df['ldl_cholesterol'], bins=10, kde=True)

plt.subplot(3,3,5)
sns.histplot(df['triglycerides'], bins=10, kde=True)

plt.subplot(3,3,6)
sns.histplot(df['family_history_diabetes'], bins=10, kde=True)

plt.subplot(3,3,7)
sns.histplot(df['hypertension_history'], bins=10, kde=True)

plt.subplot(3,3,8)
sns.histplot(df['cardiovascular_history'], bins=10, kde=True)


cap_99 = df['physical_activity_minutes_per_week'].quantile(0.99)
cap_95 = df['physical_activity_minutes_per_week'].quantile(0.95)
cap_99, cap_95


upper_limit = df['physical_activity_minutes_per_week'].quantile(.99)

df['physical_activity_minutes_per_week'] = df['physical_activity_minutes_per_week'].clip(upper=upper_limit)


cap_99 = df['triglycerides'].quantile(0.99)
cap_95 = df['triglycerides'].quantile(0.95)
cap_99, cap_95


upper_limit = df['triglycerides'].quantile(.99)

df['triglycerides'] = df['triglycerides'].clip(upper=upper_limit)


cap_99 = df['ldl_cholesterol'].quantile(0.99)
cap_95 = df['ldl_cholesterol'].quantile(0.95)
cap_99, cap_95


upper_limit = df['ldl_cholesterol'].quantile(.99)

df['ldl_cholesterol'] = df['ldl_cholesterol'].clip(upper=upper_limit)


print(df['gender'].value_counts())
print(df['ethnicity'].value_counts())
print(df['education_level'].value_counts())
print(df['income_level'].value_counts())
print(df['smoking_status'].value_counts())
print(df['employment_status'].value_counts())


df['diagnosed_diabetes'].value_counts()/df.shape[0]


x = df.drop(columns=['diagnosed_diabetes'])
y = df['diagnosed_diabetes']


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, stratify=y)


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ohe = OneHotEncoder(drop='first')
oe = OrdinalEncoder()


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()


df.columns


from sklearn.compose import ColumnTransformer
ohe_cols = ['gender', 'ethnicity', 'smoking_status', 'employment_status']

edu_order = ['No formal', 'Highschool', 'Graduate', 'Postgraduate']
income_order = ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
oe_cols = ['education_level', 'income_level']

num_cols = ['age', 'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 'systolic_bp', 
            'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
            'triglycerides']

pt_cols = ['alcohol_consumption_per_week','physical_activity_minutes_per_week', 'waist_to_hip_ratio', 'family_history_diabetes', 'hypertension_history','cardiovascular_history']

transformer = ColumnTransformer([
    ('ohe', ohe, ohe_cols),
    ('oe', oe, oe_cols), 
    ('scale', scale, num_cols),
    ('pt', pt, pt_cols)
])


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

model1 = Pipeline([
    ('transformer', transformer), 
    ('clf', lr)
])

model1.fit(x_train, y_train)
y_pred = model1.predict(x_val)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
print(accuracy_score(y_val, y_pred))
print(f1_score(y_val, y_pred))
print(roc_auc_score(y_val, y_pred))


from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(max_depth=10)

model2 = Pipeline([
    ('transformer', transformer), 
    ('clf', dt)
])

model2.fit(x_train, y_train)
y_pred = model2.predict(x_val)

print(accuracy_score(y_val, y_pred))
print(f1_score(y_val, y_pred))
print(roc_auc_score(y_val, y_pred))


from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    bootstrap=True,
    n_jobs=-1
)

model3 = Pipeline([
    ('transformer', transformer), 
    ('clf', rf)
])

model3.fit(x_train, y_train)
y_pred = model3.predict(x_val)

print(accuracy_score(y_val, y_pred))
print(f1_score(y_val, y_pred))
print(roc_auc_score(y_val, y_pred))


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()

model4 = Pipeline([
    ('transformer', transformer), 
    ('clf', gb)
])

model4.fit(x_train, y_train)
y_pred = model4.predict(x_val)

print(accuracy_score(y_val, y_pred))
print(f1_score(y_val, y_pred))
print(roc_auc_score(y_val, y_pred))


estimators = [('model1', model1), ('model2', model2), ('model3', model3), ('model4', model4)]


from sklearn.ensemble import StackingClassifier
stc = StackingClassifier(
    estimators=estimators,
    cv=5,
    n_jobs=-1,
    verbose=1
)

stc.fit(x_train, y_train)
y_pred = stc.predict(x_val)

print(accuracy_score(y_val, y_pred))
print(f1_score(y_val, y_pred))
print(roc_auc_score(y_val, y_pred))


test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
test_df.head()
test_id = test_df['id']
test_df = test_df.drop(columns=['id'])
test_df.head()


test_df_predict = stc.predict(test_df)


solution = pd.DataFrame({
    'id': test_id, 
    'diagnosed_diabetes': test_df_predict
})


solution['diagnosed_diabetes'].value_counts()


solution.to_csv('submission.csv')


x_train_trf = transformer.fit_transform(x_train)
x_val_trf = transformer.transform(x_val)


import tensorflow as tf 
from tensorflow import keras 
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout


model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(31,)))
model.add(BatchNormalization())
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))


model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['auc'])

history = model.fit(x_train_trf, y_train, validation_data=(x_val_trf, y_val), epochs=20, batch_size=64)


import matplotlib.pyplot as plt 
plt.plot(history.history['auc'], label='auc')
plt.plot(history.history['val_auc'], label='val_auc')
plt.legend()


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
test_df.head()


test_id = test_df['id']
test_df = test_df.drop(columns=['id'])
test_df.head()


test_df  = transformer.transform(test_df)


test_df_predict = model.predict(test_df)


test_df_predict


test_df_predict = test_df_predict.ravel()


test_df_predict


test_id


test_df_predict = test_df_predict.squeeze()   # NOW 1-D

submission = pd.DataFrame({
    'id': test_id,
    'diagnosed_diabetes': test_df_predict
})

submission.to_csv("submission.csv", index=False)


pd.read_csv('/kaggle/working/submission.csv')




