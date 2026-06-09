!pip install imbalanced-learn


import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub_df   = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train_df.head()


train_df.info()


display(train_df.isnull().sum())
display(test_df.isnull().sum())


train_df.describe()


train_df['Personality'].value_counts()


categorical_cols = train_df.select_dtypes(include='object').columns
display(categorical_cols)

numerical_cols = train_df.select_dtypes(exclude='object').columns
display(numerical_cols)


train_df['Stage_fear'] = train_df['Stage_fear'].map({'Yes': 1, 'No': 0})
train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
train_df['Personality'] = train_df['Personality'].map({'Introvert': 1, 'Extrovert': 0})

test_df['Stage_fear'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})


train_df[categorical_cols].head(10)


knn_i = KNNImputer(n_neighbors=5).set_output(transform='pandas')

train_df = knn_i.fit_transform(train_df)
test_df  = knn_i.fit_transform(test_df)

train_df.head(10)


train_df.isnull().sum()


RANDOM_STATE = 2474


rf_model = RandomForestClassifier(random_state=RANDOM_STATE)


cv_stratified = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 9, 11, 13]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv_stratified,
    n_jobs=-1,
    verbose=2,
)


X = train_df.drop(columns='Personality', axis='columns')
y = train_df['Personality']


grid_search.fit(X, y)


grid_search.best_score_


brf_model = BalancedRandomForestClassifier(random_state=RANDOM_STATE)

grid_search_brf = GridSearchCV(
    estimator=brf_model,
    param_grid={'n_estimators': [100], 'max_depth': [7]},
    scoring='accuracy',
    cv=cv_stratified,
    n_jobs=-1,
    verbose=2,
)

grid_search_brf.fit(X, y)


grid_search_brf.best_score_


rf_model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_estimators=100,
    max_depth=7,
    n_jobs=-1
)

rf_model.fit(X, y)


y_pred = rf_model.predict(test_df)
y_pred = np.select([y_pred==0, y_pred!=0], ['Extrovert', 'Introvert'], 'None')


sub_df['Personality'] = y_pred


sub_df.to_csv('submission.csv', index=False)

