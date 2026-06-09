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


df=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


df.head()


df.duplicated().sum()


df.describe()


df.isnull().sum()


from sklearn.ensemble import RandomForestClassifier


from sklearn.preprocessing import StandardScaler
numerical_features = df.columns[1:-1]  # Exclude 'day'

# Applying StandardScaler
scaler = StandardScaler()
X=df.drop(columns=['id','rainfall'])
X=scaler.fit_transform(X[numerical_features])
y=df['rainfall']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=41)


model=RandomForestClassifier(n_estimators=2)
model.fit(X_train,y_train)


y_pred=model.predict(X_test)


from sklearn.metrics import accuracy_score


accuracy_score(y_pred,y_test)


from sklearn.linear_model import LogisticRegression


lmodel=LogisticRegression()
lmodel.fit(X_train,y_train)
y_pred=lmodel.predict(X_test)


accuracy_score(y_pred,y_test)


sub_df=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')





# test_df=scaler.fit_transform(test_df[1:])


test_df.isnull().sum()


test_df['winddirection'].mode()[0]


test_df['winddirection'].fillna(test_df['winddirection'].mode()[0],inplace=True)


test_id = test_df.iloc[:, 0]  # Extract the first column (ID)
test_features = test_df.iloc[:, 1:]  # Select all other numerical features

# Apply the previously fitted scaler (trained on train data)
test_scaled = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns)

# Add back the ID column
test_scaled.insert(0, 'ID', test_id)

# Print scaled test data
print(test_scaled)


test_df.iloc[:,1:].values


test_scaled['rainfall']=lmodel.predict(test_scaled.iloc[:,1:].values)


sub_df=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# Create a mapping series from test_df
rainfall_mapping = test_scaled.set_index('ID')['rainfall']

# Update sub_df's rainfall using the mapping
sub_df['rainfall'] = sub_df['id'].map(rainfall_mapping)


sub_df


sub_df.to_csv('submission.csv', index=False)




