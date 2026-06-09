import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
dataset


print('Dataset Data types')
print(dataset.info())
print('-----'*10)
print('Duplicates')
print(dataset.duplicated().sum())
print('-----'*10)
print(dataset.isna().sum())


pairplot = sns.pairplot(data=dataset, diag_kind='kde')
pairplot.fig.set_size_inches(12, 8)
plt.show()


plt.figure(figsize=(7, 5))

counts = dataset['loan_paid_back'].value_counts(normalize=True) * 100 #This will reduce the from almost 500k rows to just 2 values without losing the shape of the data.
labels = ['Paid back', 'Did not pay back'] #Labeling
pie = plt.pie(counts, labels=labels, startangle=90) #PieChart

plt.title('Paid back?')
print('The perentages of', counts)
plt.show()


plt.figure(figsize=(12, 8))

most_frequent_loan_purposes = dataset['loan_purpose'].value_counts()
top_10 = most_frequent_loan_purposes.head(10)

plt.bar(top_10.index, top_10.values)
plt.xticks(rotation=25, ha='right')
plt.xlabel('Loan Purpose')
plt.ylabel('Count')
plt.show()


print(dataset['education_level'].unique())
print(dataset['grade_subgrade'].unique())


from sklearn.preprocessing import OrdinalEncoder
edu_lvl = [['Other', 'High School',"Bachelor's", "Master's", 'PhD']]
order = [['A1','A2','A3','A4','A5',
         'B1','B2','B3','B4','B5',
         'C1','C2','C3','C4','C5',
         'D1','D2','D3','D4','D5',
         'E1','E2','E3','E4','E5',
         'F1','F2','F3','F4','F5',
         'G1','G2','G3','G4','G5']]
oe = OrdinalEncoder(categories=edu_lvl)
dataset[['education_level']] = oe.fit_transform(dataset[['education_level']])
oe2 = OrdinalEncoder(categories=order)
dataset[['grade_subgrade']] = oe2.fit_transform(dataset[['grade_subgrade']])


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')
columns = ['gender', 'marital_status', 'employment_status', 'loan_purpose']
encoded = encoder.fit_transform(dataset[columns])
encoded_df = pd.DataFrame(encoded, columns= encoder.get_feature_names_out(columns))
dataset = dataset.drop(columns=columns).reset_index(drop=True)
dataset = pd.concat([dataset, encoded_df], axis=1)


relevent_cols = dataset[['annual_income', 'debt_to_income_ratio',	'credit_score',	'loan_amount',	'interest_rate', 'education_level', 'grade_subgrade', 'loan_paid_back']]
plt.figure(figsize=(12, 8))
sns.heatmap(relevent_cols.corr(), cmap='coolwarm', center=0)
plt.title('Correlation Heatmap', fontsize=20)
plt.show()


#Assigning Independent and Dependent Variebles and splitting into train and test splits:

X = dataset.drop(['id', 'loan_paid_back'], axis=1)
y = dataset['loan_paid_back']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


import xgboost as xg

negative, positive = np.bincount(y_train)
scalers = positive / negative

xgmodel = xg.XGBClassifier(n_estimators = 120, 
                           max_depth = 6, 
                           learning_rate = 0.1, 
                           subsample = 0.8, 
                           colsample_bytree=0.8, 
                           eval_metric='auc',
                           reg_lambda = 1.0, #L2 Term to push not so important features values away while maintaining them to a certain point (l2 = loss + lambda * Sum(|wi|)
                           use_label_encoder = False,
                           scale = scalers) 
xgmodel.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

y_pred = xgmodel.predict(X_test)
y_pred_proba = xgmodel.predict_proba(X_test)[:, 1] #Probabiltites of class 1

print('Fixed Predictions: ')
print('accuracy_scor:', accuracy_score(y_test, y_pred))
print('Confusion Matri:', confusion_matrix(y_test, y_pred))

print('---' * 10)

print('Probabilites Predictions: ')
print('ROC:', roc_auc_score(y_test, y_pred_proba))


xg.plot_importance(xgmodel, max_num_features=10, importance_type='gain')


test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
print(test.info())
print('-' * 20)
print('Duplicates', test.duplicated().sum())
print('-' * 20)
print('Nan values:', test.isna().sum())


from sklearn.preprocessing import OrdinalEncoder

test_encoded = test

edu_lvl = [['Other', 'High School',"Bachelor's", "Master's", 'PhD']]
order = [['A1','A2','A3','A4','A5',
         'B1','B2','B3','B4','B5',
         'C1','C2','C3','C4','C5',
         'D1','D2','D3','D4','D5',
         'E1','E2','E3','E4','E5',
         'F1','F2','F3','F4','F5',
         'G1','G2','G3','G4','G5']]
oe = OrdinalEncoder(categories=edu_lvl)
test_encoded[['education_level']] = oe.fit_transform(test_encoded[['education_level']])
oe2 = OrdinalEncoder(categories=order)
test_encoded[['grade_subgrade']] = oe2.fit_transform(test_encoded[['grade_subgrade']])


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')
columns = ['gender', 'marital_status', 'employment_status', 'loan_purpose']
encoded = encoder.fit_transform(test_encoded[columns])
encoded_df = pd.DataFrame(encoded, columns= encoder.get_feature_names_out(columns))
test_encoded = test_encoded.drop(columns=columns).reset_index(drop=True)
test_encoded = pd.concat([test_encoded, encoded_df], axis=1)


X2 = test_encoded.drop(['id'], axis=1)


probabilities_of_loan_paid_back_predictions = xgmodel.predict_proba(X2)[:, 1]


test['loan_paid_back'] = probabilities_of_loan_paid_back_predictions
test


submission_dataset_probability_of_loan_paid_back = test[['id','loan_paid_back']]
submission_dataset_probability_of_loan_paid_back


submission = pd.DataFrame(submission_dataset_probability_of_loan_paid_back)

submission.to_csv('submission.csv', index=False)

