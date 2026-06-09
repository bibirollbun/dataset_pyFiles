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





from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


le = LabelEncoder()
def CodificarLabelEncode(Dataset, columns, LabelEncoder):
    LabelEncoder.fit(Dataset[columns])
    Dataset['le_'+columns] = le.transform(Dataset[columns])
    return Dataset


oe = OrdinalEncoder()

def CodificarOrdinalEncoder(dataset, column, encoder):
    """
    Codifica uma coluna categórica usando OrdinalEncoder e
    adiciona uma nova coluna codificada ao DataFrame.
    """
    # O erro estava aqui: precisamos passar [[column]] para garantir 2D
    encoder.fit(dataset[[column]])  
    dataset['oe_' + column] = encoder.transform(dataset[[column]])
    return dataset


def get_dummies1(dataset, column):
    result = ''
    for elemento in column:
        print(elemento)
        temp = pd.get_dummies(dataset[elemento])
        dataset = pd.concat([dataset, temp], axis=1)
    return dataset   


def get_bool(dataset, column):
    temp = dataset[column].replace({True:1,False:0})
    temp = temp
    return temp


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train.dtypes


test.dtypes


train.isnull().sum()


test.isnull().sum()


train.shape


test.shape


train['family_history_diabetes'].unique()


train['family_history_diabetes'] = get_bool(train, 'family_history_diabetes')
train['cardiovascular_history'] = get_bool(train, 'cardiovascular_history')



train['diagnosed_diabetes'] = get_bool(train, 'diagnosed_diabetes')


test['family_history_diabetes'] = get_bool(test, 'family_history_diabetes')
test['cardiovascular_history'] = get_bool(test, 'cardiovascular_history')


train = get_dummies1(train, ['employment_status'])


test = get_dummies1(test, ['employment_status'])


train = get_dummies1(train, ['smoking_status'])


test = get_dummies1(test, ['smoking_status'])


train.select_dtypes(include='object')


train.loc[ : ,train.columns != 'diagnosed_diabetes']


train.select_dtypes(include="object")


train['education_level'].unique()


train['income_level'].unique()


train['smoking_status'].unique()


train['employment_status'].unique()


train = get_dummies1(train, ['gender'])


test = get_dummies1(test, ['gender'])


train = get_dummies1(train, ['education_level'])


test = get_dummies1(test, ['education_level'])


train = get_dummies1(train, ['income_level'])


test = get_dummies1(test, ['income_level'])


train = get_dummies1(train, ['smoking_status'])


test = get_dummies1(test, ['smoking_status'])


train = get_dummies1(train, ['employment_status'])


test = get_dummies1(test, ['employment_status'])


train = get_dummies1(train, ['ethnicity'])


test = get_dummies1(test, ['ethnicity'])


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


train.loc[ : , train.dtypes == 'object' ]


train.columns


train['bmi']


train.loc[ : , train.dtypes == 'object' ]


train.loc[ train['diagnosed_diabetes'] == 1]['bmi'].unique()


i=0
bmi = []
while i <= 38:
    bmi.append( train.loc[train['bmi'] > i,:]['diagnosed_diabetes'].value_counts(normalize=True) )
    i += 1
    


df = pd.DataFrame({
    'iteracao': range(len(bmi)),
    'prop_1': [s.get(1.0, 0) for s in bmi],
    'prop_0': [s.get(0.0, 0) for s in bmi],
})


import matplotlib.pyplot as plt


# plt.figure(figsize=(12,6))
# plt.plot(df['iteracao'], df['prop_1'], label='Diagnosed = 1')
# plt.plot(df['iteracao'], df['prop_0'], label='Diagnosed = 0')

# plt.xlabel('Iteração')
# plt.ylabel('Proporção')
# plt.title('Evolução da proporção de diabetes')
# plt.legend()
# plt.grid(True)
# plt.show()


train['physical_activity_minutes_per_week'].describe()


i=0
physical_activity_minutes_per_week = []
while i <= 747:
    physical_activity_minutes_per_week.append( train.loc[train['physical_activity_minutes_per_week'] > i,:]['diagnosed_diabetes'].value_counts(normalize=True) )
    i += 1


df = pd.DataFrame({
    'iteracao': range(len(physical_activity_minutes_per_week)),
    'prop_1': [s.get(1.0, 0) for s in physical_activity_minutes_per_week],
    'prop_0': [s.get(0.0, 0) for s in physical_activity_minutes_per_week],
})


plt.figure(figsize=(12,6))
plt.plot(df['iteracao'], df['prop_1'], label='Diagnosed = 1')
plt.plot(df['iteracao'], df['prop_0'], label='Diagnosed = 0')

plt.xlabel('Phisical activity minutes per work')
plt.ylabel('Proporção')
plt.title('Evolução da proporção de diabetes')
plt.legend()
plt.grid(True)
plt.show()


train.dtypes == 'int64'


train.loc[ : ,  train.dtypes == 'int64'].columns.values


train.columns


# boxplot = train.boxplot(column=['bmi', 'physical_activity_minutes_per_week', 'diet_score'])  


train['physical_activity_minutes_per_week'].describe()


train[train['physical_activity_minutes_per_week'] <= 10]['diagnosed_diabetes'].value_counts(normalize=True)


np.sort(train[train['diagnosed_diabetes'] == 1]['physical_activity_minutes_per_week'].unique())


train['family_history_diabetes'].unique()


def alcohol_risk(drinks):
    if drinks <= 3:
        return 0   # baixo
    elif drinks <= 7:
        return 1   # moderado
    else:
        return 2   # alto


train['alcohol_risk'] = train['alcohol_consumption_per_week'].apply(alcohol_risk)
train['alcohol_risk']


test['alcohol_risk'] = test['alcohol_consumption_per_week'].apply(alcohol_risk)
test['alcohol_risk']


train['prediabetes_risk_score'] = (
    (train['bmi'] >= 25).astype(int) +
    (train['physical_activity_minutes_per_week'] < 150).astype(int) +
    (train['hypertension_history'] == 1).astype(int) +
    (train['cholesterol_total'] >= 200).astype(int) +
    (train['triglycerides'] >= 150).astype(int) +
    (train['family_history_diabetes'] == 1).astype(int) +
    (train['age'] >= 45).astype(int) + 
    (train['alcohol_risk'] == 2).astype(int)
)


test['prediabetes_risk_score'] = (
    (test['bmi'] >= 25).astype(int) +
    (test['physical_activity_minutes_per_week'] < 150).astype(int) +
    (test['hypertension_history'] == 1).astype(int) +
    (test['cholesterol_total'] >= 200).astype(int) +
    (test['triglycerides'] >= 150).astype(int) +
    (test['family_history_diabetes'] == 1).astype(int) +
    (test['age'] >= 45).astype(int) + 
    (test['alcohol_risk'] == 2).astype(int)
)


# train = criar_prediabetes_risk(train)
# train


# test = criar_prediabetes_risk(test)
# test


train['has_hypertension'] = (
    (train['systolic_bp'] >= 130) | 
    (train['diastolic_bp'] >= 80)
).astype(int)


test['has_hypertension'] = (
    (test['systolic_bp'] >= 130) | 
    (test['diastolic_bp'] >= 80)
).astype(int)


# train = train[['diet_score',
#        'sleep_hours_per_day', 'screen_time_hours_per_day',
#        'waist_to_hip_ratio','heart_rate',
#         'hdl_cholesterol', 'ldl_cholesterol',
#        'income_level', 'smoking_status', 
#        'cardiovascular_history','Asian', 'Black', 'Hispanic',
#         'prediabetes_risk_score', 'has_hypertension', 'diagnosed_diabetes']]


# test = test[['diet_score',
#        'sleep_hours_per_day', 'screen_time_hours_per_day',
#        'waist_to_hip_ratio','heart_rate',
#         'hdl_cholesterol', 'ldl_cholesterol',
#        'income_level', 'smoking_status', 
#        'cardiovascular_history','Asian', 'Black', 'Hispanic',
#         'prediabetes_risk_score', 'has_hypertension']]


train.loc[ :, train.dtypes != 'object' ].drop(columns=['diagnosed_diabetes'])


from sklearn.feature_selection import mutual_info_classif

mx = train.loc[ :, train.dtypes != 'object' ].drop(columns=['diagnosed_diabetes'])
my = train['diagnosed_diabetes']

mi = mutual_info_classif(mx, my, random_state=42)

mi_series = pd.Series(mi, index=mx.columns).sort_values(ascending=False)

mi_series


# Olhando pela correlação podemos ver que....
# As features não tem correlação nas features novas. Ou seja, não reflete na melhoria do modelo.
# E agora, quais os possíveis motivos?
# Podemos ter como principais problemas: 
# -Dados feitos de forma aleatória. Já que são dados criados por aprendizado profundo de outro modelo. Não representando aquele modelo em si.
# -Falta alinhamento e testes mais precisos com novas ordenações das features.
# -Features não foram bem feitas.


X = train.loc[ : ,( train.columns != 'diagnosed_diabetes' ) ]


X.dtypes != 'object'


X = X.loc[:, X.dtypes != 'object']


X = X.loc[:, (X.columns != 'id')]


y = train['diagnosed_diabetes']


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# model_LogisticRegression = LogisticRegression(solver="newton-cholesky", random_state=0, max_iter=1000).fit(X_train, y_train)


# model_XGBClassifier = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')


X_trainfold = X_train.loc[:,X_train.columns != 'id']


X_trainfold


X_trainfold['diet_score']


from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Define base classifiers
estimators = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('xgb', XGBClassifier(random_state=42))
]




# Define the meta-classifier (final estimator)
# Logistic Regression is a common choice for the final step in classification stacking
stk_clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=LogisticRegression(),
    cv=5, # Use cross-validation
    passthrough=False,
    verbose=0
)


X_trainfold = X_trainfold.loc[:, ~X_trainfold.columns.duplicated()]
X_trainfold


X_test = X_test.loc[:, ~X_test.columns.duplicated()]
X_test


X_trainfold = X_trainfold[['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'Employed', 'Retired', 'Student',
       'Unemployed', 'Current', 'Former', 'Never', 'Female', 'Male', 'Other',
       'Graduate', 'Highschool', 'No formal', 'Postgraduate', 'High', 'Low',
       'Lower-Middle', 'Middle', 'Upper-Middle', 'Asian', 'Black', 'Hispanic',
       'White', 'alcohol_risk', 'prediabetes_risk_score', 'has_hypertension']]


X_trainfold.columns


X_trainfold[['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'Employed', 'Retired', 'Student',
       'Unemployed', 'Current', 'Former', 'Never', 'Female', 'Male', 'Other',
       'Graduate', 'Highschool', 'No formal', 'Postgraduate', 'High', 'Low',
       'Lower-Middle', 'Middle', 'Upper-Middle', 'Asian', 'Black', 'Hispanic',
       'White', 'alcohol_risk', 'prediabetes_risk_score', 'has_hypertension']]


X_test[['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'Employed', 'Retired', 'Student',
       'Unemployed', 'Current', 'Former', 'Never', 'Female', 'Male', 'Other',
       'Graduate', 'Highschool', 'No formal', 'Postgraduate', 'High', 'Low',
       'Lower-Middle', 'Middle', 'Upper-Middle', 'Asian', 'Black', 'Hispanic',
       'White', 'alcohol_risk', 'prediabetes_risk_score', 'has_hypertension']]


# adsadasd


X_test[['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'Employed', 'Retired', 'Student',
       'Unemployed', 'Current', 'Former', 'Never', 'Female', 'Male', 'Other',
       'Graduate', 'Highschool', 'No formal', 'Postgraduate', 'High', 'Low',
       'Lower-Middle', 'Middle', 'Upper-Middle', 'Asian', 'Black', 'Hispanic',
       'White', 'alcohol_risk', 'prediabetes_risk_score', 'has_hypertension']].columns


# X_tr.columns
X_test





# from sklearn.model_selection import KFold

# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# for train_idx, val_idx in kf.split(X_trainfold):
#     X_tr, X_val = X_trainfold.iloc[train_idx], X_trainfold.iloc[val_idx]
#     y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#     # treina o modelo
#     # model_LogisticRegression.fit(X_tr, y_tr)
#     # model_XGBClassifier.fit(X_tr, y_tr)
#     stk_clf.fit(X_tr, y_tr)

#     # avalia no validation
#     # pred = model_LogisticRegression.predict(X_val)
#     # pred_model_XGBClassifier = model_XGBClassifier.predict(X_val)    
#     pred_stk_clf = stk_clf.predict(X_val)
    


stk_clf.fit(X_trainfold, y_train)


# y_scores = stk_clf.predict_proba(X_test)[:, 1]


y_scores = stk_clf.predict_proba(X_test[['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'Employed', 'Retired', 'Student',
       'Unemployed', 'Current', 'Former', 'Never', 'Female', 'Male', 'Other',
       'Graduate', 'Highschool', 'No formal', 'Postgraduate', 'High', 'Low',
       'Lower-Middle', 'Middle', 'Upper-Middle', 'Asian', 'Black', 'Hispanic',
       'White', 'alcohol_risk', 'prediabetes_risk_score', 'has_hypertension']])[:, 1]


from sklearn.metrics import roc_auc_score


roc_auc = roc_auc_score(y_test, y_scores)
roc_auc


# roc_auc_score(y, model_LogisticRegression.predict_proba(X.loc[:,X.columns != 'id'])[:, 1])


# roc_auc_score(y, model_XGBClassifier.predict_proba(X.loc[:,X.columns != 'id'])[:, 1])


# sub_test = test.loc[:,X_val.columns]


# sub_test = sub_test.loc[:, ~sub_test.columns.duplicated()]


sub_test = test.loc[:,X_trainfold.columns].loc[:, ~test.loc[:,X_trainfold.columns].columns.duplicated()]


# feature = X.loc[:,X.columns != 'id'].columns


# sub_test[X_tr.columns]


res = sub_test[X_trainfold.columns]


submit = stk_clf.predict_proba(res)[:, 1]


submission_final = pd.DataFrame({

        "id": test['id'],

        "diagnosed_diabetes": submit

    })


## Submit notebooks to the challenge. Final

submission_final.to_csv('submission.csv', index=False)


print(" Arquivo submission.csv pronto ")

