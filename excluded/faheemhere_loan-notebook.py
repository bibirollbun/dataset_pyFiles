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


train_data=pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
test_data=pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")


train_data.info()


test_data.info()


X_trainf= train_data.drop(columns=['id','loan_status'])
X_testf=test_data.drop(columns=['id'])
y=train_data['loan_status']


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_array = encoder.fit_transform(X_trainf[['loan_intent', 'cb_person_default_on_file']])

# Convert back to DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['loan_intent', 'cb_person_default_on_file']))
X_trainf = X_trainf.drop(columns=['loan_intent', 'cb_person_default_on_file']).join(encoded_df)



X_trainf



encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_array = encoder.fit_transform(X_testf[['loan_intent', 'cb_person_default_on_file']])

# Convert back to DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['loan_intent', 'cb_person_default_on_file']))
X_testf = X_testf.drop(columns=['loan_intent', 'cb_person_default_on_file']).join(encoded_df)



X_testf


X_trainf.columns


X_testf.columns


from sklearn.preprocessing import OrdinalEncoder

# Define categories manually (if order is known)
categories = [
    ['OWN','RENT','MORTGAGE','OTHER'],   # For col1
    [ 'A','B','C','D','E','F','G']  # For col2
]

encoder = OrdinalEncoder(categories=categories)
X_trainf[['person_home_ownership', 'loan_grade']] = encoder.fit_transform(X_trainf[['person_home_ownership', 'loan_grade']])



from sklearn.preprocessing import OrdinalEncoder

# Define categories manually (if order is known)
categories = [
    ['OWN','RENT','MORTGAGE','OTHER'],   # For col1
    [ 'A','B','C','D','E','F','G']  # For col2
]

encoder = OrdinalEncoder(categories=categories)
X_testf[['person_home_ownership', 'loan_grade']] = encoder.fit_transform(X_testf[['person_home_ownership', 'loan_grade']])



X_testf


cat_col=[]
for col in train_data.columns:
    print('-'*50)
    if len(train_data[col].unique())<10:
        print(f"Categorical Column: {col}")
        print(f"Number of Categorical value: {train_data[col].unique()}")

        cat_col.append(col)
        print("*"*50)
    else:
        print(f"{col} is NOT a categorical column")
        print('^'*50)


 cat_col


from sklearn.preprocessing import StandardScaler

# Step 1: Identify numerical columns
num_cols =['person_age', 'person_income',
       'person_emp_length', 'loan_amnt', 'loan_int_rate',
       'loan_percent_income',
       ]

# Step 2: Initialize the StandardScaler
scaler = StandardScaler()

# Step 3: Loop through each numerical column and apply scaling
for col in num_cols:
    X_trainf[col] = scaler.fit_transform(X_trainf[[col]])  # Apply scaling column-wise





from sklearn.preprocessing import StandardScaler

# Step 1: Identify numerical columns
num_cols =['person_age', 'person_income',
       'person_emp_length', 'loan_amnt', 'loan_int_rate',
       'loan_percent_income',
       ]

# Step 2: Initialize the StandardScaler
scaler = StandardScaler()

# Step 3: Loop through each numerical column and apply scaling
for col in num_cols:
    X_testf[col] = scaler.fit_transform(X_testf[[col]])  # Apply scaling column-wise





from sklearn.ensemble import RandomForestClassifier

# Instantiate the RandomForestClassifier with proper syntax
model = RandomForestClassifier(max_depth=20, min_samples_split=10, n_estimators=200)

# Fit the model on the training data
model.fit(X_trainf, y)

# Predict on the test data
y_pred = model.predict(X_testf)



submission_df = pd.DataFrame({
    'id': test_data['id'],  # Ensure this matches the format required by the competition
    'Prediction': y_pred   # Replace with your actual predictions
})

# Save to a CSV file for submission
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created!")


