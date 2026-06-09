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


!unzip -q /kaggle/input/flight-delays-fall-2018/sample_submission.csv.zip -d /kaggle/working/
!unzip -q /kaggle/input/flight-delays-fall-2018/flight_delays_train.csv.zip -d /kaggle/working/
!unzip -q /kaggle/input/flight-delays-fall-2018/flight_delays_test.csv.zip -d /kaggle/working/


train_df = pd.read_csv('/kaggle/working/flight_delays_train.csv')
test_df = pd.read_csv('/kaggle/working/flight_delays_test.csv')


train_df


test_df


train_df.isnull().sum()


test_df.isnull().sum()


cols = ['Month', 'DayofMonth', 'DayOfWeek']
print(f"train:\n {train_df[cols].nunique()}")
print(f"test:\n {test_df[cols].nunique()}")


c = 'c-8'
c = c.replace('c-', '')
print(c)


def convert_c_prefix(df):
    cols = ['Month', 'DayofMonth', 'DayOfWeek']

    for col in cols: 
        df[col] = df[col].str.replace('c-', '').astype(int)
    return df

train_df = convert_c_prefix(train_df)
test_df = convert_c_prefix(test_df)


train_df.head(10)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_df['dep_delayed_15min'] = le.fit_transform(train_df['dep_delayed_15min'])


from category_encoders import OrdinalEncoder

cols = ['UniqueCarrier', 'Origin', 'Dest']

for col in cols:
    encoder = OrdinalEncoder(handle_unknown='value')
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])


train_df.head(10)


y = train_df['dep_delayed_15min']
train_df = train_df.drop(['dep_delayed_15min'], axis=1)


# Classi Sbilanciate, 0: 80%, 1: 20%
y.value_counts()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2)

len(X_train), len(X_test), len(y_train), len(y_test) 


X_train


y_test


from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

# Calcolo dello sbilanciamento tra classi
ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

models = {
    'xgb': {
        'model': XGBClassifier(objective = 'binary:logistic',
                               device='cuda',  # Usa 'cuda' invece dei parametri deprecati
                               tree_method='hist',  # Obbligatorio con device='cuda'
                               scale_pos_weight=ratio,
                              ),
        'params': {
            'max_depth': [5, 6, 8, 10],
            'n_estimators': [500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1],
        }
    },
    'catboost': {
        'model': CatBoostClassifier(auto_class_weights='Balanced',
                                   task_type='GPU',  # Abilita GPU
                                   devices='0:1'  # Usa la prima GPU
                                   ),
        'params': {
            'max_depth': [ 5, 6, 8, 10],
            'n_estimators': [500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    }
}

#**CATBOOST**
#Migliori parametri: {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 1500}
#Miglior ROC AUC (CV): 0.7385609264207375

best_models = {}

for model_name, config in models.items():
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        scoring='roc_auc',
        cv=5,
        verbose=1
    )

    grid_search.fit(X_train, y_train, verbose=False)

    best_models[model_name] = {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }
    
print(f"\n**{model_name.upper()}**")
print("Migliori parametri:", grid_search.best_params_)
print("Miglior ROC AUC (CV):", grid_search.best_score_)


best_model = best_models['catboost']['best_model']
print(f"Catboost params: {best_model.get_params()}")


prediction = best_model.predict(X_test)


prob_prediction = best_model.predict_proba(X_test)[:, 1].round(3)


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(prediction, y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N', 'Y'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


from sklearn.metrics import classification_report

print(classification_report(y_test, prediction, target_names=['N', 'Y']))


from sklearn.metrics import roc_auc_score

print(f"ROC AUC: {roc_auc_score(y_test, prob_prediction)}")


best_model.fit(train_df, y, verbose=False)


sample_df = pd.read_csv('/kaggle/working/sample_submission.csv')
sample_df.head(5)


submission_predictions = best_model.predict_proba(test_df)[:, 1].round(3)
submission_predictions


submission_df = pd.DataFrame({'id': range(0, len(submission_predictions)), 'dep_delayed_15min': submission_predictions})
submission_df.head(5)


submission_df.to_csv('/kaggle/working/submission.csv', index=False)

