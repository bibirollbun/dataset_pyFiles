# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train.head()


target = 'Personality'


X_train = train.drop(columns=['id', target])
y_train = train[target]


from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)


# print(X_train.head())
y_train_encoded, len(X_train.columns)


numerical_features = X_train.select_dtypes(include=['float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns


numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('transformer', FunctionTransformer(np.log1p)),
    ('scaler', RobustScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline, categorical_pipeline


preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
],remainder='passthrough')

preprocessor


from sklearn.metrics import classification_report, f1_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


models = {
    'Logistic Regression' : LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest' : RandomForestClassifier(random_state=42),
    'XGBoost' : XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM' : LGBMClassifier(random_state=42),
    'CatBoost' : CatBoostClassifier(random_state=42)
}

results ={}


for model_name, model in models.items():
    print(f'Training {model_name} ')

    full_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('sampler', SMOTE(random_state=42)),
        ('model', model)
    ])

    full_pipeline.fit(X_train, y_train_encoded)

    y_pred = full_pipeline.predict(X_train)

    ac_score = accuracy_score(y_train_encoded, y_pred)
    results[model_name] = ac_score


results_df = pd.DataFrame(results.items(), columns=['Model', 'Accuracy Score'])
results_df = results_df.sort_values(by='Accuracy Score', ascending=False)
results_df


def submit(y_pred):
    y_sub_str = le.inverse_transform(y_pred)

    submission = pd.DataFrame({
        'id' : test['id'],
        'Personality': y_sub_str
    })

    submission.to_csv('submission.csv', index=False)

    print(submission.head())


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
X_test = test.drop(columns=['id'])


# rf_pipeline = ImbPipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('sampler', SMOTE(random_state=42)),
#     ('model', RandomForestClassifier(random_state=42))
# ])

# rf_pipeline.fit(X_train, y_train_encoded)
# y_pred_test_rf = rf_pipeline.predict(X_test)

# rf_pipeline


# submit(y_pred_test_rf)


# catb_pipeline = ImbPipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('sampler', SMOTE(random_state=42)),
#     ('model', CatBoostClassifier(random_state=42))
# ])

# catb_pipeline.fit(X_train, y_train_encoded)
# y_pred_test_catb = catb_pipeline.predict(X_test)

# catb_pipeline


# submit(y_pred_test_catb)


xgb_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('sampler', SMOTE(random_state=42)),
    ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

xgb_pipeline.fit(X_train, y_train_encoded)
y_pred_test_xgb = xgb_pipeline.predict(X_test)

xgb_pipeline


submit(y_pred_test_xgb)

