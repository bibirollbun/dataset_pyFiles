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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')


df_train.isnull().sum()


df_train.dtypes


from sklearn.impute import SimpleImputer
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']


# --- For numeric columns (skip columns if they're completely NaN!)
num_impute_cols = [col for col in num_cols if not df_train[col].isnull().all()]
num_imputer = SimpleImputer(strategy='median')
df_train[num_impute_cols] = pd.DataFrame(
    num_imputer.fit_transform(df_train[num_impute_cols]),
    columns=num_impute_cols,
    index=df_train.index
)
df_test[num_impute_cols] = pd.DataFrame(
    num_imputer.transform(df_test[num_impute_cols]),
    columns=num_impute_cols,
    index=df_test.index
)

# --- For categorical columns (skip all-NaN ones)
cat_impute_cols = [col for col in cat_cols if not df_train[col].isnull().all()]
cat_imputer = SimpleImputer(strategy='most_frequent')
df_train[cat_impute_cols] = pd.DataFrame(
    cat_imputer.fit_transform(df_train[cat_impute_cols]),
    columns=cat_impute_cols,
    index=df_train.index
)
df_test[cat_impute_cols] = pd.DataFrame(
    cat_imputer.transform(df_test[cat_impute_cols]),
    columns=cat_impute_cols,
    index=df_test.index
)


df_train.isnull().sum()


from sklearn.impute import SimpleImputer

# --- Get numeric and categorical columns (excluding id and Personality)
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']

# --- For numeric columns (skip columns if they're completely NaN!)
num_impute_cols = [col for col in num_cols if not df_train[col].isnull().all()]
num_imputer = SimpleImputer(strategy='median')
df_train[num_impute_cols] = pd.DataFrame(
    num_imputer.fit_transform(df_train[num_impute_cols]),
    columns=num_impute_cols,
    index=df_train.index
)
df_test[num_impute_cols] = pd.DataFrame(
    num_imputer.transform(df_test[num_impute_cols]),
    columns=num_impute_cols,
    index=df_test.index
)

# --- For categorical columns (skip all-NaN ones)
cat_impute_cols = [col for col in cat_cols if not df_train[col].isnull().all()]
cat_imputer = SimpleImputer(strategy='most_frequent')
df_train[cat_impute_cols] = pd.DataFrame(
    cat_imputer.fit_transform(df_train[cat_impute_cols]),
    columns=cat_impute_cols,
    index=df_train.index
)
df_test[cat_impute_cols] = pd.DataFrame(
    cat_imputer.transform(df_test[cat_impute_cols]),
    columns=cat_impute_cols,
    index=df_test.index
)


df_train.head()


binary_map = {'No': 0, 'Yes': 1}

for col in ['Stage_fear', 'Drained_after_socializing']:
    df_train[col] = df_train[col].map(binary_map)
    df_test[col] = df_test[col].map(binary_map)


df_train.head()


X = df_train.drop('Personality', axis=1)
y = df_train['Personality']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


from sklearn.preprocessing import LabelEncoder

# Encode y_train and y_test
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)  # Introvert -> 0, Extrovert -> 1 (or vice versa)
y_test_encoded = le.transform(y_test)        # Apply same transformation to y_test


from xgboost import XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV


# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1,0.05, 0.3],
    'subsample': [0.7,0.8, 1],
    'colsample_bytree': [0.7, 0.8,1],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}

# Grid search
from sklearn.model_selection import StratifiedKFold

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='accuracy', cv=cv_strategy, verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train_encoded)


# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


y_pred = best_model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)


print("Accuracy:", accuracy_score(y_test, y_pred_labels))
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))


# Ensure df_test has the same columns as X_train (no 'id' column)
df_test_final = df_test[X_train.columns]  # Enforce same feature set as training

# Predict and decode
predictions = best_model.predict(df_test_final)
decoded_predictions = le.inverse_transform(predictions)


submission = pd.DataFrame({'Personality': decoded_predictions}, index=df_test.index)
submission.to_csv('submission.csv')
submission.head()




