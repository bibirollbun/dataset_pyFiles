# import library for exploring dataset
import numpy as np
import pandas as pd


# read the dataset csv
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train.head()


test.head()


sample_submission.head()


train.info()


test.info()


sample_submission.info()


print('train Sex unique = ',train['Sex'].unique())
print('test Sex unique = ',test['Sex'].unique())


sex_mapping = {'male':'0','female':'1'}
train['Sex'] = (train['Sex'].replace(sex_mapping)).astype(float)
test['Sex'] = (test['Sex'].replace(sex_mapping)).astype(float)


# BMI (Body Mass Index)
train['BMI'] = train['Weight'] / ( (train['Height'] / 100) ** 2 )
test['BMI'] = test['Weight'] / ( (test['Height'] / 100) ** 2 )

# Age × Heart Rate
train['Heart_Activity'] = train['Age'] * train['Heart_Rate']
test['Heart_Activity'] = test['Age'] * test['Heart_Rate']

# Duration × Heart Rate (Total Effort)
train['Heart_Stress'] = train['Duration'] * train['Heart_Rate']
test['Heart_Stress'] = test['Duration'] * test['Heart_Rate']

# Duration × Body Temp (Thermal Load)
train['Duration_BodyTemp'] = train['Duration'] * train['Body_Temp']
test['Duration_BodyTemp'] = test['Duration'] * test['Body_Temp']

# Weight × Duration (Mechanical Load)
train['Calories_Burned'] = train['Weight'] * train['Duration']
test['Calories_Burned'] = test['Weight'] * test['Duration']

# Heart / Duration
train['Heart_Rate_Minute'] = train['Heart_Rate'] / train['Duration']
test['Heart_Rate_Minute'] = test['Heart_Rate'] / test['Duration']

# Body_Temp - 37 
train['Body_Temp_Dev'] = train['Body_Temp'] -37
test['Body_Temp_Dev'] = test['Body_Temp'] - 37

# Age / Height
train['Body_Structure'] = train['Age'] / train['Height']
test['Body_Structure'] = test['Age'] / test['Height']

# Age * Heart_Rate
train['Activity_Impact'] = train['Age'] * train['Heart_Rate']
test['Activity_Impact'] = test['Age'] * test['Heart_Rate']

# Height * Weight * Duration
train['Calorie_Prediction'] = train['Height'] * train['Weight'] * train['Duration']
test['Calorie_Prediction'] = test['Height'] * test['Weight'] * test['Duration']


# Map Age Groups to Decimal Codes
age_group_mapping = {
    'Young_Adult': 0,
    'Adult': 1,
    'Middle_Aged': 2,
    'Older': 3
}

# Map Age into Age Groups
train['Age_Group'] = pd.cut(train['Age'], 
                         bins=[0, 25, 35, 50, 100], 
                         labels=['Young_Adult', 'Adult', 'Middle_Aged', 'Older'])

test['Age_Group'] = pd.cut(test['Age'], 
                         bins=[0, 25, 35, 50, 100], 
                         labels=['Young_Adult', 'Adult', 'Middle_Aged', 'Older'])

train['Age'] = (train['Age_Group'].map(age_group_mapping)).cat.codes.astype(float)
train = train.drop(columns=['Age_Group'])

test['Age'] = (test['Age_Group'].map(age_group_mapping)).cat.codes.astype(float)
test = test.drop(columns=['Age_Group'])


train.head()


test.head()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.show()


# drop id column on train dataset
train = train.drop(columns=['id'])

# train data split
X = train.loc[:, train.columns != 'Calories']
y = train['Calories']

# test data split
X_new = test.iloc[:, 1:]


import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=7, random_state=42)
xgb_model.fit(X, y)


predictions_xgb_model = xgb_model.predict(X_new)


# save model to sample_sumbission
sample_submission['Calories'] = predictions_xgb_model
sample_submission['Calories'] = sample_submission['Calories'].abs()
sample_submission.to_csv('submission.csv', index=False)

# see the sample_submission head
sample_submission.head()

