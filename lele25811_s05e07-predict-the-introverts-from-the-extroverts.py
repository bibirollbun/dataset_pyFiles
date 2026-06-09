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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df.head()


train_df.isnull().sum()


test_df.isnull().sum()


train_df.dtypes


def fillna_by_cols(df):
    float_cols=['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    obj_cols=['Stage_fear', 'Drained_after_socializing']
    for col in float_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in obj_cols:
        df[col] = df[col].fillna('U')
    return df


train_df = fillna_by_cols(train_df)
test_df = fillna_by_cols(test_df)


train_df.isnull().sum()


test_df.isnull().sum()


from sklearn.preprocessing import LabelEncoder

cols = ['Stage_fear', 'Drained_after_socializing']
for col in cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

le = LabelEncoder()
train_df['Personality'] = le.fit_transform(train_df['Personality'])


train_df.head()


test_df.head()


test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)


y = train_df['Personality']
train_df = train_df.drop(['Personality', 'id'], axis=1)


from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(train_df, y, discrete_features='auto')
mi_df = pd.DataFrame({'Feature': train_df.columns, 'Mutual Information': mi_scores})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# Mostra
print(mi_df)


import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data=mi_df, x='Mutual Information', y='Feature')
plt.title("Mutual Information tra feature e target")
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2)


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgbc = XGBClassifier(objective="binary:logistic", 
                    use_label_encoder=False,
                    eval_metric='logloss'
)

param_grid = {
    'n_estimators': [1000, 1500],
    'max_depth': [4, 6],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'gamma': [1, 5],
}

grid_search = GridSearchCV(
    estimator=xgbc,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best score (CV):", grid_search.best_score_)


best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test ,predictions)
plt.figure(figsize=(8,4))
sns.heatmap(cm,annot=True,fmt="g",cmap="summer")
plt.show()


submission_predictions = best_model.predict(test_df)


submission_predictions.shape, test_ids.shape


submission_predictions_mapped = np.where(submission_predictions == 1, 'Introvert', 'Extrovert')


submission = pd.DataFrame({'id': test_ids.values, 'Personality': submission_predictions_mapped})

submission.head(5)


submission.to_csv('/kaggle/working/submission.csv', index=False)

