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


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
train


test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test


submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission


train.dtypes


train.isnull().sum()


from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

numeric_cols = train.select_dtypes(include=['number']).columns
train[numeric_cols] = imputer.fit_transform(train[numeric_cols])

print(train.isnull().sum())


X = train.drop(columns=['Listening_Time_minutes'], axis=1)  
y = train['Listening_Time_minutes']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


numeric_features = X_train.select_dtypes(include=['number']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns


preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


from lightgbm import LGBMRegressor


model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMRegressor())
])

model.fit(X_train, y_train)




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)



mae = mean_absolute_error(y_test, y_pred)
mae


r2 = r2_score(y_test, y_pred)
r2


test_predictions = model.predict(test)



submission = pd.DataFrame({
    'id': test['id'],  
    'Listening_Time_minutes': test_predictions
})

submission.to_csv('sub.csv', index=False)


submission


from xgboost import XGBRegressor

model1 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8
    ))
])


model1.fit(X_train, y_train)



y_pred1 = model1.predict(X_test)
r2 = r2_score(y_test, y_pred1)
r2


test_predictions = model1.predict(test)



submission = pd.DataFrame({
    'id': test['id'],  
    'Listening_Time_minutes': test_predictions
})

submission.to_csv('sub1.csv', index=False)
submission


model = model1.named_steps['regressor']

preprocessor = model1.named_steps['preprocessor']

if hasattr(model, 'feature_importances_'):
    numeric_feature_names = preprocessor.transformers_[0][2]
    
    ohe = preprocessor.transformers_[1][1]  
    categorical_feature_names = ohe.get_feature_names_out(preprocessor.transformers_[1][2])
    
    feature_names = list(numeric_feature_names) + list(categorical_feature_names)
    
    importance = model.feature_importances_
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    print(feature_importances)



train.head()


train.shape


train.dtypes


train.columns


X = train.drop(columns=['Listening_Time_minutes', 'id', ], axis=1)  
y = train['Listening_Time_minutes']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder

scale = ['Host_Popularity_percentage', 'Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']
ohe = ['Genre', 'Publication_Time', 'Episode_Sentiment', 'Publication_Day']
ore = ['Podcast_Name', 'Episode_Title']


preprocessor1 = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), scale),
        ('ohe', OneHotEncoder(), ohe),
        ('ore', OrdinalEncoder(), ore)
    ])


label = LabelEncoder()

y_train_label = label.fit_transform(y_train)
y_test_label = label.transform(y_test)


model2 = Pipeline([
    ('preprocessor1', preprocessor),
    ('regressor', LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=-1, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0, random_state=42, n_jobs=-1))
])

model2.fit(X_train, y_train_label)


y_pred2 = model2.predict(X_test)
r2 = r2_score(y_test_label, y_pred2)
r2


test_predictions2 = model2.predict(test)



submission = pd.DataFrame({
    'id': test['id'],  
    'Listening_Time_minutes': test_predictions2
})

submission.to_csv('sub2.csv', index=False)
submission

