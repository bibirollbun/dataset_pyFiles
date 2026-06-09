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


# load data
df_train = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv',
                       low_memory=False)
df_test = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv',
                      low_memory=False)

df_sub = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/sample_submission.csv')


df_train.head(5)


df_train.info()


df_train.isna().sum()


df_test.isna().sum()


df_train = df_train.dropna()
df_train.isna().sum()


df_test = df_test.apply(lambda x: x.fillna(x.value_counts().index[0]))
df_test.isna().sum()


import matplotlib.pyplot as plt

# aesthetics
default_color_1 = 'blue'
default_color_2 = 'green'
default_color_3 = 'darkred'

# define features and target
features_num = ['Weight_kg']

features_cat = ['Age','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
                'Conception_Difficulty', 'Insulin_Resistance',
                'Exercise_Frequency', 'Exercise_Type', 'Exercise_Duration',
                'Sleep_Hours', 'Exercise_Benefit']

target = 'PCOS'


# plot histograms (train and test)
for f in features_num:
    plt.figure(figsize=(12,3))
    ax1 = plt.subplot(1,2,1)
    df_train[f].plot(kind='hist', bins=20, color=default_color_1)
    plt.title(f + ' - Train')
    plt.grid()
    ax2 = plt.subplot(1,2,2, sharex=ax1)
    df_test[f].plot(kind='hist', bins=20, color=default_color_2)
    plt.title(f + ' - Test')
    plt.grid()
    plt.show()


# plot categorical feature distributions (train and test)
for f in features_cat:
    plt.figure(figsize=(14,3))
    ax1 = plt.subplot(1,2,1)
    df_train[f].value_counts().sort_index().plot(kind='bar', color=default_color_1)
    plt.title(f + ' - Train')
    plt.grid()
    ax2 = plt.subplot(1,2,2)
    df_test[f].value_counts().sort_index().plot(kind='bar', color=default_color_2)
    plt.title(f + ' - Test')
    plt.grid()
    plt.show()


# check categories of Age in train set
df_train.Age.value_counts()


# simplify age structure - training data
df_train['Age_Group'] = 'MISSING'
# 20-25
df_train.loc[df_train.Age=='20-25', 'Age_Group'] = '20t25'
# translate all that are < 20 in level "lt20"
df_train.loc[df_train.Age=='15-20', 'Age_Group'] = 'lt20'
df_train.loc[df_train.Age=='Less than 20', 'Age_Group'] = 'lt20'
df_train.loc[df_train.Age=='Less than 20-25', 'Age_Group'] = 'lt20'
# translate all that are > 25 in level "gt25"
df_train.loc[df_train.Age=='35-44', 'Age_Group'] = 'gt25'
df_train.loc[df_train.Age=='25-30', 'Age_Group'] = 'gt25'
df_train.loc[df_train.Age=='45 and above', 'Age_Group'] = 'gt25'
df_train.loc[df_train.Age=='30-35', 'Age_Group'] = 'gt25'
df_train.loc[df_train.Age=='30-25', 'Age_Group'] = 'gt25'
df_train.loc[df_train.Age=='30-40', 'Age_Group'] = 'gt25'
# check results
df_train['Age_Group'].value_counts()


df_train = df_train.drop(['Age'], axis=1)


# simplify age structure - test data
df_test['Age_Group'] = 'MISSING'
# 20-25
df_test.loc[df_test.Age=='20-25', 'Age_Group'] = '20t25'
df_test.loc[df_test.Age=='20', 'Age_Group'] = '20t25'
df_test.loc[df_test.Age=='22-25', 'Age_Group'] = '20t25'
df_test.loc[df_test.Age=='25-25', 'Age_Group'] = '20t25'
# translate all that are < 20 in level "lt20"
df_test.loc[df_test.Age=='Less than 20', 'Age_Group'] = 'lt20'
df_test.loc[df_test.Age=='Less than 20-25', 'Age_Group'] = 'lt20'
df_test.loc[df_test.Age=='Less than 20)', 'Age_Group'] = 'lt20'
# translate all that are > 25 in level "gt25"
df_test.loc[df_test.Age=='30-35', 'Age_Group'] = 'gt25'
df_test.loc[df_test.Age=='35-44', 'Age_Group'] = 'gt25'
df_test.loc[df_test.Age=='30-30', 'Age_Group'] = 'gt25'
df_test.loc[df_test.Age=='50-60', 'Age_Group'] = 'gt25'
df_test.loc[df_test.Age=='30-40', 'Age_Group'] = 'gt25'
df_test.loc[df_test.Age=='45-49', 'Age_Group'] = 'gt25'
# check results
df_test['Age_Group'].value_counts()


df_test = df_test.drop(['Age'], axis=1)


df_train.Exercise_Type.value_counts()


# simplify Exercise_Type structure - training data
df_train['Exercise_Type_Clean'] = 'MISSING'

# replace values
df_train.loc[df_train.Exercise_Type=='No Exercise', 'Exercise_Type_Clean'] = 'No Exercise'
df_train.loc[df_train.Exercise_Type=='Cardio (e.g., running, cycling, swimming)', 'Exercise_Type_Clean'] = 'Cardio'
df_train.loc[df_train.Exercise_Type=='Cardio (e.g.', 'Exercise_Type_Clean'] = 'Cardio'
df_train.loc[df_train.Exercise_Type=='Flexibility and balance (e.g., yoga, pilates)', 'Exercise_Type_Clean'] = 'Flexibility'
df_train.loc[df_train.Exercise_Type=='Strength training (e.g., weightlifting, resistance exercises)', 'Exercise_Type_Clean'] = 'Strength'
df_train.loc[df_train.Exercise_Type=='Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises)', 'Exercise_Type_Clean'] = 'Flexibility+Strength'
df_train.loc[df_train.Exercise_Type=='Cardio (e.g., running, cycling, swimming), Flexibility and balance (e.g., yoga, pilates)', 'Exercise_Type_Clean'] = 'Cardio+Flexibility'
df_train.loc[df_train.Exercise_Type=='High-intensity interval training (HIIT)', 'Exercise_Type_Clean'] = 'HIIT'
df_train.loc[df_train.Exercise_Type=='Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates),', 'Exercise_Type_Clean'] = 'Cardio+Flexibiliy+Strength'
df_train.loc[df_train.Exercise_Type=='Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)', 'Exercise_Type_Clean'] = 'Strength'
df_train.loc[df_train.Exercise_Type=='Flexibility and balance (e.g., yoga, pilates), None', 'Exercise_Type_Clean'] = 'Flexibility'
df_train.loc[df_train.Exercise_Type=='Cardio (e.g., running, cycling, swimming), None', 'Exercise_Type_Clean'] = 'Cardio'
df_train.loc[df_train.Exercise_Type=='Strength training', 'Exercise_Type_Clean'] = 'Strength'
df_train.loc[df_train.Exercise_Type=='Strength training (e.g.', 'Exercise_Type_Clean'] = 'Strength'
df_train.loc[df_train.Exercise_Type=='Somewhat', 'Exercise_Type_Clean'] = 'Somewhat'
df_train.loc[df_train.Exercise_Type=='Flexibility and balance (e.g.', 'Exercise_Type_Clean'] = 'Flexibility'
     
# check results
df_train['Exercise_Type_Clean'].value_counts()


df_train = df_train.drop(['Exercise_Type'], axis=1)


# simplify Exercise_Type structure - test data
df_test['Exercise_Type_Clean'] = 'MISSING'

# replace values
df_test.loc[df_test.Exercise_Type=='Cardio (e.g.', 'Exercise_Type_Clean'] = 'Cardio'
df_test.loc[df_test.Exercise_Type=='No Exercise', 'Exercise_Type_Clean'] = 'No Exercise'
df_test.loc[df_test.Exercise_Type=='Flexibility and balance (e.g.', 'Exercise_Type_Clean'] = 'Flexibility'
df_test.loc[df_test.Exercise_Type=='Strength training (e.g.', 'Exercise_Type_Clean'] = 'Strength'
df_test.loc[df_test.Exercise_Type=='Strength training', 'Exercise_Type_Clean'] = 'Strength'
df_test.loc[df_test.Exercise_Type=='Yes Significantly', 'Exercise_Type_Clean'] = 'Other'
df_test.loc[df_test.Exercise_Type=='No', 'Exercise_Type_Clean'] = 'No Exercise'
df_test.loc[df_test.Exercise_Type=='Sleep_Benefit', 'Exercise_Type_Clean'] = 'MISSING'
df_test.loc[df_test.Exercise_Type=='Not Applicable', 'Exercise_Type_Clean'] = 'MISSING'
df_test.loc[df_test.Exercise_Type=='Somewhat', 'Exercise_Type_Clean'] = 'Somewhat'
df_test.loc[df_test.Exercise_Type=='Strength (e.g.', 'Exercise_Type_Clean'] = 'Strength'
# check results
df_test['Exercise_Type_Clean'].value_counts()


df_test = df_test.drop(['Exercise_Type'], axis=1)


# plot new categorical features (train and test)
for f in ['Age_Group', 'Exercise_Type_Clean']:
    plt.figure(figsize=(14,3))
    ax1 = plt.subplot(1,2,1)
    df_train[f].value_counts().sort_index().plot(kind='bar', color=default_color_1)
    plt.title(f + ' - Train')
    plt.grid()
    ax2 = plt.subplot(1,2,2)
    df_test[f].value_counts().sort_index().plot(kind='bar', color=default_color_2)
    plt.title(f + ' - Test')
    plt.grid()
    plt.show()


df_train.Sleep_Hours.value_counts()


df_train['Sleep'] = 'MISSING'

df_train.loc[df_train.Sleep_Hours=='6-8 hours', 'Sleep'] = '68h'
df_train.loc[df_train.Sleep_Hours=='Less than 6 hours', 'Sleep'] = 'l6h'
df_train.loc[df_train.Sleep_Hours=='9-12 hours','Sleep'] = 'b9h'
df_train.loc[df_train.Sleep_Hours=='More than 12 hours', 'Sleep'] = 'b9h'
df_train.loc[df_train.Sleep_Hours=='3-4 hours', 'Sleep'] = 'l6h'


df_train['Sleep'].value_counts()


df_train = df_train.drop(['Sleep_Hours'], axis=1)


df_test.Sleep_Hours.value_counts()


df_test['Sleep'] = 'MISSING'

df_test.loc[df_test.Sleep_Hours=='6-8 hours', 'Sleep'] = '68h'
df_test.loc[df_test.Sleep_Hours=='Less than 6 hours', 'Sleep'] = 'l6h'
df_test.loc[df_test.Sleep_Hours=='9-12 hours','Sleep'] = 'b9h'
df_test.loc[df_test.Sleep_Hours=='3-4 hours', 'Sleep'] = 'l6h'
df_test.loc[df_test.Sleep_Hours=='6-8 Times a Week', 'Sleep'] = '68h'
df_test.loc[df_test.Sleep_Hours=='6-12 hours', 'Sleep'] = '68h'
df_test.loc[df_test.Sleep_Hours=='20 minutes', 'Sleep'] = 'l6h'




df_test['Sleep'].value_counts()


df_test = df_test.drop(['Sleep_Hours'], axis=1)


df_train.Hormonal_Imbalance.value_counts()


df_train.loc[df_train.Hormonal_Imbalance=='No, Yes, not diagnosed by a doctor', 'Hormonal_Imbalance'] = 'Yes'
df_train.loc[df_train.Hormonal_Imbalance=='Yes Significantly', 'Hormonal_Imbalance'] = 'Yes'



df_train.Hormonal_Imbalance.value_counts()


df_test.Hormonal_Imbalance.value_counts()


df_train.Hirsutism.value_counts()


df_train.loc[df_train.Hirsutism=='No, Yes, not diagnosed by a doctor', 'Hirsutism'] = 'Yes'


df_train.Hirsutism.value_counts()


df_test.Hirsutism.value_counts()


plt.figure(figsize=(4,3))
df_train[target].value_counts().plot(kind='bar', color=default_color_3)
plt.title(target)
plt.grid()
plt.show()


features = ['Weight_kg',
 'Age_Group',
 'Hormonal_Imbalance',
 'Hyperandrogenism',
 'Hirsutism',
 'Conception_Difficulty',
 'Insulin_Resistance',
 'Exercise_Frequency',
 'Exercise_Type_Clean',
 'Exercise_Duration',
 'Sleep',
 'Exercise_Benefit']
features


import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = df_train[features]
y = df_train[target].map({'No': 0, 'Yes': 1})

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cat_features = [col for col in X.columns if X[col].dtype == 'object']

def objective(trial):
    cat_params = dict(
        iterations=trial.suggest_int("iterations", 100, 1000),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        depth=trial.suggest_int("depth", 4, 15),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0, 2.5),
        random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        task_type='GPU',
        early_stopping_rounds=200,
        verbose=False
    )
    
    model = CatBoostRegressor(**cat_params)
    X_train_pool = Pool(X_train, y_train, cat_features=cat_features)
    X_valid_pool = Pool(X_val, y_val, cat_features=cat_features)
    model.fit(X=X_train_pool, eval_set=X_valid_pool)
    
    y_pred = model.predict(X_val).clip(0,1)
    score = mean_squared_error(y_val, y_pred)
    
    return score


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)


best_params = study.best_params

final_model = CatBoostRegressor(**best_params)
final_model


X_train_pool = Pool(X, y, cat_features=cat_features)
final_model.fit(X=X_train_pool)

pred_test = final_model.predict(df_test[features]).clip(0,1)


# create submission file
df_sub_GLM = df_sub.copy()
df_sub_GLM[target] = pred_test
df_sub_GLM.to_csv('submission_GLM.csv', index=False)
df_sub_GLM.head(10)

