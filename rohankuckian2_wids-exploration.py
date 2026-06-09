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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


matrix = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
solutions = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')
categorical = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')
quantitative = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx')


solutions


solutions[['ADHD_Outcome', 'Sex_F']].value_counts()





matrix.set_index('participant_id', inplace=True)
solutions.set_index('participant_id', inplace=True)
categorical.set_index('participant_id', inplace=True)
quantitative.set_index('participant_id', inplace=True)
solutions = solutions.reindex(columns=['Sex_F', 'ADHD_Outcome'])


concat_mat = pd.concat([matrix, categorical, quantitative, solutions], axis=1)
concat_mat.fillna(0, inplace=True)


from sklearn.model_selection import train_test_split

X = concat_mat.iloc[:, :-2]
y = concat_mat['Sex_F']
X.shape, y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression

lrmodel = LogisticRegression()
lrmodel.fit(X_train, y_train)
y_pred = lrmodel.predict(X_test)


(y_pred == y_test).sum() * 100/len(y_pred)


(y_pred == 1).sum()


pred_df = pd.DataFrame(y_pred, columns=['Pred_Sex'], index=y_test.index)
y_test = pd.concat([pd.DataFrame(y_test), pred_df], axis=1)


(y_test['Sex_F'] == y_test['Pred_Sex']).sum()








X_2_train = concat_mat[concat_mat.index.isin(X_train.index)].iloc[:, :-1]
X_2_test = concat_mat[concat_mat.index.isin(X_test.index)].iloc[:, :-1]
y_2_train = concat_mat[concat_mat.index.isin(X_train.index)].loc[:, 'ADHD_Outcome']
y_2_test = concat_mat[concat_mat.index.isin(X_test.index)].loc[:, 'ADHD_Outcome']


X_2_test['Sex_F'] = y_test['Pred_Sex']


lrmodel_2 = LogisticRegression(solver='lbfgs', max_iter=10000)
lrmodel_2.fit(X_2_train, y_2_train)
y_2_pred = lrmodel_2.predict(X_2_test)


np.unique(y_2_pred, return_counts=True)


np.unique(y_2_test, return_counts=True)


(y_2_pred == y_2_test).sum() * 100/len(y_2_pred)


np.unique(y_2_pred, return_counts=True)


(X_2_test.sort_index()['Sex_F'] != y_test.sort_index()['Pred_Sex']).sum()


X_2_test['Sex_F']


sample_submission = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
sample_submission.set_index('participant_id', inplace=True)



matrix_test = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
categorical_test = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
quantitative_test = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')

matrix_test.set_index('participant_id', inplace=True)
categorical_test.set_index('participant_id', inplace=True)
quantitative_test.set_index('participant_id', inplace=True)

concat_mat_test = pd.concat([matrix_test, categorical_test, quantitative_test], axis=1)
concat_mat_test.fillna(0, inplace=True)


for i in range(len(concat_mat_test.columns)):
    if concat_mat_test.columns[i] != X_train.columns[i]:
        print(concat_mat_test.columns[i], X_train.columns[i])


X_pred_1 = concat_mat_test
y_pred_1 = lrmodel.predict(X_test_1)


X_pred_2 = X_pred_1.copy()
X_pred_2['Sex_F'] = y_pred_1
y_pred_2 = lrmodel_2.predict(X_pred_2)


sample_submission['Sex_F'] = y_pred_1
sample_submission['ADHD_Outcome'] = y_pred_2
sample_submission.to_csv('baseline_submission_logistic.csv')


sample_submission


X_train.shape


X_train


solutions


print(matrix.isna().sum().sum(), categorical.isna().sum().sum(), quantitative.isna().sum().sum())


categorical


for column in categorical.columns:
    print(column)
    categorical[column] = categorical[column].fillna(categorical[column].mode()[0])


for column in quantitative.columns[1:]:
    print(column)
    quantitative[column] = quantitative[column].fillna(quantitative[column].median())
    # print(quantitative[column].median())





matrix.set_index('participant_id', inplace=True)
solutions.set_index('participant_id', inplace=True)
categorical.set_index('participant_id', inplace=True)
quantitative.set_index('participant_id', inplace=True)
solutions = solutions.reindex(columns=['Sex_F', 'ADHD_Outcome'])

concat_mat = pd.concat([matrix, categorical, quantitative, solutions], axis=1)

X = concat_mat.iloc[:, :-2]
y = concat_mat['Sex_F']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lrmodel = LogisticRegression()
lrmodel.fit(X_train, y_train)
y_pred = lrmodel.predict(X_test)

print((y_pred == y_test).sum() * 100/len(y_pred))


X_2_train = concat_mat[concat_mat.index.isin(X_train.index)].iloc[:, :-1]
X_2_test = concat_mat[concat_mat.index.isin(X_test.index)].iloc[:, :-1]
y_2_train = concat_mat[concat_mat.index.isin(X_train.index)].loc[:, 'ADHD_Outcome']
y_2_test = concat_mat[concat_mat.index.isin(X_test.index)].loc[:, 'ADHD_Outcome']
pred_df = pd.DataFrame(y_pred, columns=['Pred_Sex'], index=y_test.index)
y_test = pd.concat([pd.DataFrame(y_test), pred_df], axis=1)

X_2_test['Sex_F'] = y_test['Pred_Sex']
lrmodel_2 = LogisticRegression(solver='lbfgs', max_iter=10000)
lrmodel_2.fit(X_2_train, y_2_train)
y_2_pred = lrmodel_2.predict(X_2_test)

print((y_2_pred == y_2_test).sum() * 100/len(y_2_pred))


matrix_test = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
categorical_test = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
quantitative_test = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')

for column in categorical_test.columns:
    categorical_test[column] = categorical_test[column].fillna(categorical_test[column].mode()[0])

for column in quantitative_test.columns[1:]:
    quantitative_test[column] = quantitative_test[column].fillna(quantitative_test[column].median())


matrix_test.set_index('participant_id', inplace=True)
categorical_test.set_index('participant_id', inplace=True)
quantitative_test.set_index('participant_id', inplace=True)

concat_mat_test = pd.concat([matrix_test, categorical_test, quantitative_test], axis=1)

X_pred_1 = concat_mat_test
y_pred_1 = lrmodel.predict(X_pred_1)

X_pred_2 = X_pred_1.copy()
X_pred_2['Sex_F'] = y_pred_1
y_pred_2 = lrmodel_2.predict(X_pred_2)

sample_submission = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
sample_submission.set_index('participant_id', inplace=True)
sample_submission['Sex_F'] = y_pred_1
sample_submission['ADHD_Outcome'] = y_pred_2
sample_submission.to_csv('imputed_logistic.csv')


oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe_array = oh_encoder.fit_transform(categorical)
feature_names = oh_encoder.get_feature_names_out()
categorical_encoded = pd.DataFrame(ohe_array, columns=feature_names)
categorical_encoded.index = categorical.index
concat_mat = pd.concat([matrix, categorical_encoded, quantitative, solutions], axis=1)

X = concat_mat.iloc[:, :-2]
y = concat_mat['Sex_F']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lrmodel = LogisticRegression(solver='lbfgs', max_iter=10000)
lrmodel.fit(X_train, y_train)
y_pred = lrmodel.predict(X_test)

print((y_pred == y_test).sum() * 100/len(y_pred))


X_2_train = concat_mat[concat_mat.index.isin(X_train.index)].iloc[:, :-1]
X_2_test = concat_mat[concat_mat.index.isin(X_test.index)].iloc[:, :-1]
y_2_train = concat_mat[concat_mat.index.isin(X_train.index)].loc[:, 'ADHD_Outcome']
y_2_test = concat_mat[concat_mat.index.isin(X_test.index)].loc[:, 'ADHD_Outcome']
pred_df = pd.DataFrame(y_pred, columns=['Pred_Sex'], index=y_test.index)
y_test = pd.concat([pd.DataFrame(y_test), pred_df], axis=1)

X_2_test['Sex_F'] = y_test['Pred_Sex']
lrmodel_2 = LogisticRegression(solver='lbfgs', max_iter=10000)
lrmodel_2.fit(X_2_train, y_2_train)
y_2_pred = lrmodel_2.predict(X_2_test)

print((y_2_pred == y_2_test).sum() * 100/len(y_2_pred))


ohe_array_test = oh_encoder.transform(categorical_test)
categorical_test_encoded = pd.DataFrame(ohe_array_test, columns=feature_names)
categorical_test_encoded.index = categorical_test.index

concat_mat_test = pd.concat([matrix_test, categorical_test_encoded, quantitative_test], axis=1)

X_pred_1 = concat_mat_test
y_pred_1 = lrmodel.predict(X_pred_1)

X_pred_2 = X_pred_1.copy()
X_pred_2['Sex_F'] = y_pred_1
y_pred_2 = lrmodel_2.predict(X_pred_2)

sample_submission = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
sample_submission.set_index('participant_id', inplace=True)
sample_submission['Sex_F'] = y_pred_1
sample_submission['ADHD_Outcome'] = y_pred_2
sample_submission.to_csv('imputed_ohe_logistic.csv')


concat_mat


zscore_scaler = StandardScaler()

zscore_scaled_array = zscore_scaler.fit_transform(quantitative)
quantitative_zscore = pd.DataFrame(zscore_scaled_array, columns=quantitative.columns)
quantitative_zscore.index = quantitative.index
concat_mat = pd.concat([matrix, categorical_encoded, quantitative_zscore, solutions], axis=1)

X = concat_mat.iloc[:, :-2]
y = concat_mat['Sex_F']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lrmodel = LogisticRegression(solver='lbfgs', max_iter=10000)
lrmodel.fit(X_train, y_train)
y_pred = lrmodel.predict(X_test)

print((y_pred == y_test).sum() * 100/len(y_pred))


X_2_train = concat_mat[concat_mat.index.isin(X_train.index)].iloc[:, :-1]
X_2_test = concat_mat[concat_mat.index.isin(X_test.index)].iloc[:, :-1]
y_2_train = concat_mat[concat_mat.index.isin(X_train.index)].loc[:, 'ADHD_Outcome']
y_2_test = concat_mat[concat_mat.index.isin(X_test.index)].loc[:, 'ADHD_Outcome']
pred_df = pd.DataFrame(y_pred, columns=['Pred_Sex'], index=y_test.index)
y_test = pd.concat([pd.DataFrame(y_test), pred_df], axis=1)

X_2_test['Sex_F'] = y_test['Pred_Sex']
lrmodel_2 = LogisticRegression(solver='lbfgs', max_iter=10000)
lrmodel_2.fit(X_2_train, y_2_train)
y_2_pred = lrmodel_2.predict(X_2_test)

print((y_2_pred == y_2_test).sum() * 100/len(y_2_pred))


zscore_scaler.transform(quantitative_test).shape


zscore_scaled_array_test = zscore_scaler.transform(quantitative_test)
quantitative_test_zscore = pd.DataFrame(zscore_scaled_array_test, columns=quantitative_test.columns)
quantitative_test_zscore.index = quantitative_test.index

concat_mat_test = pd.concat([matrix_test, categorical_test_encoded, quantitative_test_zscore], axis=1)

X_pred_1 = concat_mat_test
y_pred_1 = lrmodel.predict(X_pred_1)

X_pred_2 = X_pred_1.copy()
X_pred_2['Sex_F'] = y_pred_1
y_pred_2 = lrmodel_2.predict(X_pred_2)

sample_submission = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
sample_submission.set_index('participant_id', inplace=True)
sample_submission['Sex_F'] = y_pred_1
sample_submission['ADHD_Outcome'] = y_pred_2
sample_submission.to_csv('imputed_ohe_zscore_logistic.csv')


quantitative_minmax


minmax_scaler = MinMaxScaler()

minmax_scaled_array = minmax_scaler.fit_transform(quantitative)
quantitative_minmax = pd.DataFrame(minmax_scaled_array, columns=quantitative.columns)
quantitative_minmax.index = quantitative.index
concat_mat = pd.concat([matrix, categorical_encoded, quantitative_minmax, solutions], axis=1)

X = concat_mat.iloc[:, :-2]
y = concat_mat['Sex_F']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lrmodel = LogisticRegression(solver='lbfgs', max_iter=10000)
lrmodel.fit(X_train, y_train)
y_pred = lrmodel.predict(X_test)

print((y_pred == y_test).sum() * 100/len(y_pred))


X_2_train = concat_mat[concat_mat.index.isin(X_train.index)].iloc[:, :-1]
X_2_test = concat_mat[concat_mat.index.isin(X_test.index)].iloc[:, :-1]
y_2_train = concat_mat[concat_mat.index.isin(X_train.index)].loc[:, 'ADHD_Outcome']
y_2_test = concat_mat[concat_mat.index.isin(X_test.index)].loc[:, 'ADHD_Outcome']
pred_df = pd.DataFrame(y_pred, columns=['Pred_Sex'], index=y_test.index)
y_test = pd.concat([pd.DataFrame(y_test), pred_df], axis=1)

X_2_test['Sex_F'] = y_test['Pred_Sex']
lrmodel_2 = LogisticRegression(solver='lbfgs', max_iter=10000)
lrmodel_2.fit(X_2_train, y_2_train)
y_2_pred = lrmodel_2.predict(X_2_test)

print((y_2_pred == y_2_test).sum() * 100/len(y_2_pred))


minmax_scaled_array_test = minmax_scaler.transform(quantitative_test)
quantitative_test_minmax = pd.DataFrame(minmax_scaled_array_test, columns=quantitative_test.columns)
quantitative_test_minmax.index = quantitative_test.index

concat_mat_test = pd.concat([matrix_test, categorical_test_encoded, quantitative_test_minmax], axis=1)

X_pred_1 = concat_mat_test
y_pred_1 = lrmodel.predict(X_pred_1)

X_pred_2 = X_pred_1.copy()
X_pred_2['Sex_F'] = y_pred_1
y_pred_2 = lrmodel_2.predict(X_pred_2)

sample_submission = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
sample_submission.set_index('participant_id', inplace=True)
sample_submission['Sex_F'] = y_pred_1
sample_submission['ADHD_Outcome'] = y_pred_2
sample_submission.to_csv('imputed_ohe_minmax_logistic.csv')




