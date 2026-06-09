# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')

test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


train.info()
print('========================\n')
test.info()


train.isnull().sum()


test.isnull().sum()


train['loan_status'].value_counts()


train['person_home_ownership'].value_counts()


test['person_home_ownership'].value_counts()


train['loan_intent'].value_counts()


test['loan_intent'].value_counts()


train['loan_grade'].value_counts()


test['loan_grade'].value_counts()


train['cb_person_default_on_file'].value_counts()


test['cb_person_default_on_file'].value_counts()


# Drop redundant

train = train.drop(columns=['id'])


# Label encoding for binary and ordinal 

train['cb_person_default_on_file'] = train['cb_person_default_on_file'].map(
    {'Y':0,
     'N':1}
)

train['loan_grade'] = train['loan_grade'].map(
    {'A':1,
     'B':2,
     'C':3,
     'D':4,
     'E':5,
     'F':6,
     'G':7}
)


# One hot encoding for nominal

# Initialize encoders for each column
pho_ecd = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
lit_ecd = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Encode each column
pho_encoded = pho_ecd.fit_transform(train[['person_home_ownership']])
lit_encoded = lit_ecd.fit_transform(train[['loan_intent']])

# Convert to DataFrames
pho_encoded_df = pd.DataFrame(pho_encoded, columns=pho_ecd.get_feature_names_out(['person_home_ownership']))
lit_encoded_df = pd.DataFrame(lit_encoded, columns=lit_ecd.get_feature_names_out(['loan_intent']))

# Drop the original columns from the original DataFrame
train_dropped = train.drop(columns=['person_home_ownership', 'loan_intent'])

# Combine the results with the remaining dataset
train = pd.concat([train_dropped.reset_index(drop=True), pho_encoded_df, lit_encoded_df], axis=1)


train.columns


train.isnull().sum()


# Label encoding for binary and ordinal 

test['cb_person_default_on_file'] = test['cb_person_default_on_file'].map(
    {'Y':0,
     'N':1}
)

test['loan_grade'] = test['loan_grade'].map(
    {'A':1,
     'B':2,
     'C':3,
     'D':4,
     'E':5,
     'F':6,
     'G':7}
)


# One hot encoding for nominal

# Encode each column
pho_encoded_test = pho_ecd.transform(test[['person_home_ownership']])
lit_ecdoded_test = lit_ecd.transform(test[['loan_intent']])

# Convert to DataFrames
pho_encoded_df_test = pd.DataFrame(pho_encoded_test, columns=pho_ecd.get_feature_names_out(['person_home_ownership']))
lit_encoded_df_test = pd.DataFrame(lit_ecdoded_test, columns=lit_ecd.get_feature_names_out(['loan_intent']))

# Drop the original columns from the original DataFrame
test_dropped = test.drop(columns=['person_home_ownership', 'loan_intent'])

# Combine the results with the remaining dataset
test = pd.concat([test_dropped.reset_index(drop=True), pho_encoded_df_test, lit_encoded_df_test], axis=1)


test.columns


test.isnull().sum()


X_test_all = test.drop(columns=['id'])
X_id = test['id']



X_test_all.columns


X_test_all.isnull().sum()


X_id.head(5)


X_id.shape


X_id.info()


from sklearn.feature_selection import chi2

X = train.drop(columns=['loan_status'])
y = train['loan_status']

chi2_stats, p_values = chi2(X, y)

# Create a DataFrame for results
chi2_results = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_stats,
    'P-value': p_values
}).sort_values(by='Chi2 Score', ascending=False)

print(chi2_results)


# Correct condition for selecting features with Chi2 Score > 1000
selected_features = chi2_results[chi2_results['Chi2 Score'] > 1000]

# Display the result
print(selected_features)


X = train[['person_income','loan_amnt','loan_int_rate','loan_grade','person_emp_length','person_home_ownership_RENT','person_home_ownership_MORTGAGE']]
y = train['loan_status']

X_test = test[['person_income','loan_amnt','loan_int_rate','loan_grade','person_emp_length','person_home_ownership_RENT','person_home_ownership_MORTGAGE']]


X.shape


X_test.shape


train = train.drop(columns=['loan_status'])


train.shape


X_test_all.shape


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


clf = LogisticRegression(random_state=0 ,max_iter=5000)
scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')
print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=2, random_state=0)
scores = cross_val_score(rfc, X, y, cv=10, scoring='f1_macro')
print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))


from sklearn.ensemble import GradientBoostingClassifier


gbc = GradientBoostingClassifier(max_depth=2, random_state=0)
scores = cross_val_score(gbc, X, y, cv=10, scoring='f1_macro')
print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))


clf_o = LogisticRegression(random_state=0 ,max_iter=5000)
scores = cross_val_score(clf_o, train, y, cv=10, scoring='f1_macro')
print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))


rfc_o = RandomForestClassifier(max_depth=2, random_state=0)
scores = cross_val_score(rfc_o, train, y, cv=10, scoring='f1_macro')
print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))


gbc_o = GradientBoostingClassifier(max_depth=2, random_state=0)
scores = cross_val_score(gbc_o, train, y, cv=10, scoring='f1_macro')
print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))


gbc_o.fit(train, y)



pred = gbc_o.predict(X_test_all)


submission = pd.DataFrame({
    'id': X_id,
    'loan_status': pred
})

submission.to_csv('submission.csv', index=False)

