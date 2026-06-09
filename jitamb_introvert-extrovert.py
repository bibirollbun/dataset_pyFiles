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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_train.head(10)


df_train.shape


df_train.info()


df_train = df_train.drop('id', axis = 1)


df_train['Personality'].value_counts()


13699/(13699+4825) *100


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


cat_features = ['Stage_fear', 'Drained_after_socializing']


df_train['Stage_fear'] = (df_train['Stage_fear']=='Yes').astype(int)
df_train['Drained_after_socializing'] = (df_train['Drained_after_socializing']=='Yes').astype(int)


X = df_train.drop('Personality', axis = 1)
y = df_train['Personality']


y = (y == "Extrovert").astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
catboost_weights = dict(zip([0, 1], class_weights))


cbc = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='Accuracy',
    class_weights=catboost_weights,
    verbose=0,
    random_state=42
)


param_dist = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'iterations': [500, 800, 1000, 1200],
    'l2_leaf_reg': [1, 3, 5, 7, 10],
    'bagging_temperature': [0, 0.2, 0.5, 1],
    'border_count': [32, 64, 128, 254]
}


model = RandomizedSearchCV(
    estimator = cbc,
    param_distributions = param_dist,
    n_iter=25,
    n_jobs=-1,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose = 2,
    random_state=42
)


model.fit(X_train, y_train, early_stopping_rounds=50)


y_pred = model.predict(X_test)


acc = accuracy_score(y_test, y_pred)


acc


from sklearn.metrics import f1_score


f1_score(y_test, y_pred)


print(f"Test Accuracy: {acc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df_test.head()


df_test['Stage_fear'] = (df_test['Stage_fear']=='Yes').astype(int)
df_test['Drained_after_socializing'] = (df_test['Drained_after_socializing']=='Yes').astype(int)


y_test_pred = model.predict(df_test.drop('id', axis = 1))


inverse_label_map = {0: 'Introvert', 1: 'Extrovert'}
y_test_pred_cat = pd.Series(y_test_pred).map(inverse_label_map)


df_sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
df_sample_sub.head()


df_submit = pd.DataFrame({
    'id': df_test['id'],
    'Personality': y_test_pred_cat,
})


df_submit.head(10)


df_submit.to_csv('/kaggle/working/submission.csv', index=False)

