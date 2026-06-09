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


df = df.drop('education_level', axis = 1)



df['ldl_cholesterol'].unique()


df.columns


X = df.drop('diagnosed_diabetes', axis = 1)
y = df['diagnosed_diabetes']


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1,stratify = y)


df.head()


from sklearn.preprocessing import OneHotEncoder


num_col = ['id', 'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', ]

cat_col = ['gender', 'ethnicity', 'income_level',
       'smoking_status', 'employment_status']


encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
X_train_en = encoder.fit_transform(X_train[cat_col])
X_test_en = encoder.transform(X_test[cat_col])

encoded_col_names = encoder.get_feature_names_out(cat_col)


X_train_en = pd.DataFrame(X_train_en, columns=encoded_col_names, index = X_train.index)
X_test_en = pd.DataFrame(X_test_en, columns=encoded_col_names, index = X_test.index)


X_train_bin = X_train[['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']]
X_test_bin  = X_test[['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']]


X_train_cat = pd.concat([X_train_en, X_train_bin], axis = 1)
X_test_cat = pd.concat([X_test_en, X_test_bin], axis = 1)



from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scl = scaler.fit_transform(X_train[num_col])
X_test_scl = scaler.transform(X_test[num_col])

X_train_scl = pd.DataFrame(X_train_scl, columns=num_col, index = X_train.index)
X_test_scl = pd.DataFrame(X_test_scl, columns=num_col, index = X_test.index)


X_train_fin = pd.concat([X_train_scl, X_train_cat], axis = 1)
X_test_fin = pd.concat([X_test_scl, X_test_cat], axis = 1)


X_train_fin.head()


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_fin, y_train)



y_pred = model.predict(X_test_fin)

print(model.score(X_train_fin, y_train))
print(model.score(X_test_fin, y_test))


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train = train.drop('education_level', axis = 1)
test = test.drop('education_level', axis = 1)



X = train.drop('diagnosed_diabetes', axis = 1)
y = train['diagnosed_diabetes']


from sklearn.preprocessing import OneHotEncoder


num_col = ['id', 'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', ]

cat_col = ['gender', 'ethnicity', 'income_level',
       'smoking_status', 'employment_status']


encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
X_en = encoder.fit_transform(X[cat_col])
test_en = encoder.transform(test[cat_col])

encoded_col_names = encoder.get_feature_names_out(cat_col)


X_en = pd.DataFrame(X_en, columns=encoded_col_names, index = X.index)
test_en = pd.DataFrame(test_en, columns=encoded_col_names, index = test.index)


X_bin = X[['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']]
test_bin  = test[['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']]

X_cat = pd.concat([X_en, X_bin], axis = 1)
test_cat = pd.concat([test_en, test_bin], axis = 1)



from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_scl = scaler.fit_transform(X[num_col])
test_scl = scaler.transform(test[num_col])

X_scl = pd.DataFrame(X_scl, columns=num_col, index = X.index)
test_scl = pd.DataFrame(test_scl, columns=num_col, index = test.index)


X_fin = pd.concat([X_scl, X_cat], axis = 1)
test_fin = pd.concat([test_scl, test_cat], axis = 1)


X_fin.head()


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)
model.fit(X_fin,y)


y_pred = model.predict_proba(test_fin)[:, 1]



# Create submission dataframe
submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": y_pred
})


submission.to_csv("submission.csv", index=False)







