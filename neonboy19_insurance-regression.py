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

sampled_df = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')

test_df = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


fraction = 0.1  # 10% of the data
sampled_df = sampled_df.sample(frac=fraction, random_state=42)


sampled_df.info()


sampled_df.head()


sampled_df.duplicated().sum()


sampled_df.isnull().sum()


Nan_Cols = []

for col in sampled_df.columns:

    if sampled_df[col].isnull().any() == True:

        print(f"{col} dtype : {sampled_df[col].dtype}")

        


for col in sampled_df.columns:
    if sampled_df[col].dtype == 'object':  
        mode_value = sampled_df[col].mode()[0]  
        sampled_df[col].fillna(mode_value, inplace=True)
    else:  
        mean_value = sampled_df[col].mean()  # Calculate mean
        sampled_df[col].fillna(mean_value, inplace=True)


for col in test_df.columns:
    if test_df[col].dtype == 'object':  
        mode_value = test_df[col].mode()[0]  
        test_df[col].fillna(mode_value, inplace=True)
    else:  
        mean_value = test_df[col].mean()  # Calculate mean
        test_df[col].fillna(mean_value, inplace=True)


for col in sampled_df.columns:

    if sampled_df[col].isnull().any() == True:

        print(f"{col} dtype : {sampled_df[col].dtype}")


sampled_df.describe()


import seaborn as sns
import matplotlib.pyplot as plt

def numerical_plots(data):

    for col in data.columns:

        if data[col].dtype != 'object':

            sns.histplot(data[col], kde=True, bins=20)
            plt.title(f'{col} Distribution')
            plt.show()

numerical_plots(sampled_df)


            


def categorical_plots(data):
    
    for col in data.columns:
        
        if data[col].dtype == 'object':  
            
            sns.countplot(x=data[col])
            plt.title(f'{col} Distribution')
            plt.show()

categorical_plots(sampled_df)


sampled_df.info()


constant_cols = [col for col in sampled_df.columns if sampled_df[col].nunique() == 1]
# sampled_df = sampled_df.drop(constant_cols, axis=1)



constant_cols


for col in sampled_df.columns:

    if sampled_df[col].dtype == 'object':

        print(f"{col} has {sampled_df[col].value_counts()} uniq values")


from sklearn.preprocessing import LabelEncoder

binary_cols = ['Gender', 'Smoking Status']
for col in binary_cols:
    le = LabelEncoder()
    sampled_df[col] = le.fit_transform(sampled_df[col])


binary_cols = ['Gender', 'Smoking Status']
for col in binary_cols:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])


sampled_df = pd.get_dummies(sampled_df, columns=[
    'Marital Status', 'Education Level', 'Occupation', 
    'Location', 'Policy Type', 'Customer Feedback', 
    'Exercise Frequency', 'Property Type'
], drop_first=True) 


test_df = pd.get_dummies(test_df, columns=[
    'Marital Status', 'Education Level', 'Occupation', 
    'Location', 'Policy Type', 'Customer Feedback', 
    'Exercise Frequency', 'Property Type'
], drop_first=True) 


sampled_df['Policy Start Year'] = pd.to_datetime(sampled_df['Policy Start Date']).dt.year
sampled_df['Policy Start Month'] = pd.to_datetime(sampled_df['Policy Start Date']).dt.month
sampled_df['Policy Start Day'] = pd.to_datetime(sampled_df['Policy Start Date']).dt.day
sampled_df['Policy Start DayofWeek'] = pd.to_datetime(sampled_df['Policy Start Date']).dt.dayofweek
sampled_df['Policy Duration (Days)'] = (pd.Timestamp('now') - pd.to_datetime(sampled_df['Policy Start Date'])).dt.days


test_df['Policy Start Year'] = pd.to_datetime(test_df['Policy Start Date']).dt.year
test_df['Policy Start Month'] = pd.to_datetime(test_df['Policy Start Date']).dt.month
test_df['Policy Start Day'] = pd.to_datetime(test_df['Policy Start Date']).dt.day
test_df['Policy Start DayofWeek'] = pd.to_datetime(test_df['Policy Start Date']).dt.dayofweek
test_df['Policy Duration (Days)'] = (pd.Timestamp('now') - pd.to_datetime(test_df['Policy Start Date'])).dt.days


sampled_df = sampled_df.drop(columns=['Policy Start Date'])


test_df = test_df.drop(columns=['Policy Start Date'])


sampled_df.info()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



X = sampled_df.drop(columns=['Premium Amount', 'id'])  
y = sampled_df['Premium Amount']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse}")


test_df.isnull().sum()


test_features = test_df.drop(columns=['id']) 
test_predictions = model.predict(test_features)

submission = pd.DataFrame({
    'id': test_df['id'],  
    'Premium Amount': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

