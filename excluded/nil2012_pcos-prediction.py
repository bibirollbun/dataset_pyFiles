# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pcos_train = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')
pcos_test = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')


pcos_train.head()


pcos_train.info()


pcos_test.info()


pcos_train.isna().sum()


pcos_test.isna().sum()


categorical_columns = pcos_train.select_dtypes(include=['object']).columns


categorical_columns = categorical_columns[categorical_columns != 'PCOS']


for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=pcos_train[column]) 
    sns.histplot(data=pcos_test[column])
    plt.title(f'Distribution of {column}')
    plt.legend(['Train', 'Test'])
    plt
    plt.show()


binary_columns = ['Hormonal_Imbalance','Hyperandrogenism','Hirsutism','Conception_Difficulty','Insulin_Resistance']


def binary_group(column_value):
    if column_value in ['Yes', 'Yes Significantly', 'Somewhat', 'Yes, diagnosed by a doctor']:
        return 'Yes'
    elif column_value == 'No':
        return 'No'
    else:
        return 'Missing'


for column in binary_columns:
    pcos_train[column] = pcos_train[column].apply(binary_group)
    pcos_test[column] = pcos_test[column].apply(binary_group)


for column in binary_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=pcos_train[column]), sns.histplot(data=pcos_test[column])
    plt.legend(['Train', 'Test'])
    plt.title(f'Distribution of {column}')
    plt.show()


other_categorical_columns = ['Exercise_Frequency', 'Exercise_Duration', 'Exercise_Benefit', 'Exercise_Type', 'Sleep_Hours']


for column in other_categorical_columns:
    for keys, values in pcos_train[column].value_counts().items():
        if values < 2:
            pcos_train[column] = pcos_train[column].replace(keys, 'Noise')


for column in other_categorical_columns:
    for keys, values in pcos_test[column].value_counts().items():
        if values < 2:
            pcos_test[column] = pcos_test[column].replace(keys, 'Noise')


def exercise_frequency_group(exercise_frequency):
    if exercise_frequency == '6-8 Times a Week' or exercise_frequency == 'Daily' or exercise_frequency == '6-8 hours':
        return '6-8 Times a Week'
    elif exercise_frequency == 'Less than 6-8 Times a Week' or exercise_frequency == 'Less than 6 hours':
        return 'Less than 6-8 Times a Week'
    elif exercise_frequency == '3-4 Times a Week':
        return '3-4 Times a Week'
    elif exercise_frequency == '1-2 Times a Week' or exercise_frequency == '1/2 Times a Week':
        return '1-2 Times a Week'
    elif exercise_frequency == 'Never':
        return 'Never'
    elif exercise_frequency == 'Rarely':
        return 'Rarely'
    elif exercise_frequency == 'Noise':
        return 'Noise'
    else:
        return 'Missing'


pcos_train['Exercise_Frequency'] = pcos_train['Exercise_Frequency'].apply(exercise_frequency_group)
pcos_test.Exercise_Frequency = pcos_test.Exercise_Frequency.apply(exercise_frequency_group)


def exercise_duration_group(exercise_duration):
    if exercise_duration == '30 minutes':
        return '30 minutes'
    elif exercise_duration == '45 minutes' or exercise_duration == '30 minutes to 1 hour':
        return '45 minutes'
    elif exercise_duration == 'Less than 30 minutes':
        return '25 minutes'
    elif exercise_duration == 'More than 30 minutes':
        return '40 minutes'
    elif exercise_duration == '20 minutes':
        return '20 minutes'
    elif exercise_duration == 'Not Applicable':
        return 'Not Applicable'
    elif exercise_duration == 'Noise':
        return 'Noise'
    else:
        return 'Missing'


pcos_train['Exercise_Duration'] = pcos_train['Exercise_Duration'].apply(exercise_duration_group)
pcos_test['Exercise_Duration'] = pcos_test['Exercise_Duration'].apply(exercise_duration_group)


def exercise_benefit_group(exercise_benefit):
    if exercise_benefit == 'Not at All':
        return 'Not at All'
    elif exercise_benefit == 'Not Much':
        return 'Not Much'
    elif exercise_benefit == 'Somewhat':
        return 'Somewhat'
    elif exercise_benefit == 'Yes Significantly':
        return 'Yes Significantly'
    elif exercise_benefit == 'Noise':
        return 'Noise'
    else:
        return 'Missing'


pcos_train['Exercise_Benefit'] = pcos_train['Exercise_Benefit'].apply(exercise_benefit_group)
pcos_test['Exercise_Benefit'] = pcos_test['Exercise_Benefit'].apply(exercise_benefit_group)


def exercise_type_group(exercise_type):
    if exercise_type == 'Cardio (e.g.' or exercise_type == 'Cardio (e.g., running, cycling, swimming)' or exercise_type == 'Cardio (e.g., running, cycling, swimming), None':
        return 'Cardio'

    elif exercise_type == 'Strength training' or exercise_type == 'Strength training (e.g.' or exercise_type == 'Strength (e.g.' or exercise_type == 'Strength training (e.g., weightlifting, resistance exercises)':
        return 'Strength'
    elif exercise_type == 'Noise':
        return 'Noise'
    elif exercise_type == 'Flexibility and balance (e.g.' or exercise_type == 'Flexibility and balance (e.g., yoga, pilates)' or exercise_type == 'Flexibility and balance (e.g., yoga, pilates), None':
        return 'Flexibility'
    elif exercise_type == 'Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises)':
        return 'Cardio and Strength'
    elif exercise_type == 'Cardio (e.g., running, cycling, swimming), Flexibility and balance (e.g., yoga, pilates)':
        return 'Cardio and Flexibility'
    elif exercise_type == 'Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)':
        return 'Cardio, Strength, and Flexibility'
    elif exercise_type == 'Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)':
        return 'Strength and Flexibility'
    elif exercise_type == 'High-intensity interval training (HIIT)':
        return 'HIIT'
    elif exercise_type == 'No Exercise':
        return 'No Exercise'
    
    else:
        return 'Missing'
    


pcos_train['Exercise_Type'] = pcos_train['Exercise_Type'].apply(exercise_type_group)
pcos_test['Exercise_Type'] = pcos_test['Exercise_Type'].apply(exercise_type_group)


def sleep_hours_group(sleep_hours):
    if sleep_hours == '6-8 hours':
        return '7 hours'
    elif sleep_hours == 'Less than 6 hours' or sleep_hours == '3-4 hours':
        return '3-4 hours'
    elif sleep_hours == '9-12 hours':
        return '10 hours'
    elif sleep_hours == 'Noise':
        return 'Noise'
    else:
        return 'Missing'


pcos_train['Sleep_Hours'] = pcos_train['Sleep_Hours'].apply(sleep_hours_group)
pcos_test['Sleep_Hours'] = pcos_test['Sleep_Hours'].apply(sleep_hours_group)


pcos_train.Age.unique()


pcos_test.Age.unique()


def age_group(age):
    if age == '20-25':
        return 22
    elif age == '22-25':
        return 23
    elif age == '15-20':
        return 17
    elif age == '25-25':
        return 25
    elif age == 'Less than 20' or age == 'Less than 20-25' or age == '20' or age == 'Less than 20)':
        return 20
    elif age == '25-30' or age == '30-25':
        return 27
    elif age == '30-30':
        return 30
    elif age == '30-35':
        return 32
    elif age == '30-40':
        return 35
    elif age == '35-44':
        return 40
    elif age == '45 and above':
        return 45
    elif age == '45-49':
        return 47
    elif age == '50-60':
        return 55


pcos_train['Age'] = pcos_train['Age'].apply(age_group)
pcos_test['Age'] = pcos_test['Age'].apply(age_group)


from sklearn.impute import SimpleImputer, KNNImputer


pcos_train.Weight_kg = SimpleImputer(strategy='mean').fit_transform(pcos_train[['Weight_kg']])
pcos_test.Weight_kg = SimpleImputer(strategy='mean').fit_transform(pcos_test[['Weight_kg']])


pcos_train.Age = SimpleImputer(strategy='mean').fit_transform(pcos_train[['Age']])
pcos_test.Age = SimpleImputer(strategy='mean').fit_transform(pcos_test[['Age']])


pcos_train.info()
pcos_test.info()


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


for column in other_categorical_columns:
    pcos_train[column] = OrdinalEncoder().fit_transform(pcos_train[[column]])
    pcos_test[column] = OrdinalEncoder().fit_transform(pcos_test[[column]])


for column in binary_columns:
    pcos_train[column] = LabelEncoder().fit_transform(pcos_train[column])
    pcos_test[column] = LabelEncoder().fit_transform(pcos_test[column])


pcos_train.PCOS = LabelEncoder().fit_transform(pcos_train['PCOS'])


X = pcos_train.drop(columns=['ID', 'PCOS'])  # specify the column(s) to drop
y = pcos_train['PCOS']  # target variable


feat_test = pcos_test.copy()
feat_test = feat_test.drop(columns=['ID'])


from sklearn.model_selection import train_test_split  # import the train_test_split function


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # split the data


from xgboost import XGBRegressor


from sklearn.metrics import roc_auc_score


xgb_test = XGBRegressor()  # initialize the XGBoost classifier
xgb_test.fit(x_train, y_train)  # fit the model to the training data
y_pred_xgbtest = xgb_test.predict(x_test)  # make predictions on the test data
roc_auc = roc_auc_score(y_test, y_pred_xgbtest)  # calculate the ROC AUC score
print("ROC AUC Score:", roc_auc)  # print the ROC AUC score
print()


import optuna


def objective_xgb(trial):
    
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_params = {
        
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.05, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.05, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 40),
        'eval_metric': 'auc',
    }
    
    model = XGBRegressor(**xgb_params)
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    preds = model.predict(val_x)
    score = roc_auc_score(val_y, preds)
    
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective_xgb, n_trials=50)


print(study.best_params)
print(study.best_value)


best_params = study.best_params


xgb_model = XGBRegressor(**best_params)
xgb_model.fit(X, y)
y_pred_xgb = xgb_model.predict(feat_test)


df = pd.DataFrame({'ID': pcos_test['ID'], 'PCOS': y_pred_xgb})


df.to_csv('PCOS_prediction_notebook.csv', index=False)

