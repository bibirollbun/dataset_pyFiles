import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)


import pandas as pd
import numpy as np

train = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
test = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')
sample_submission = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/sample_submission.csv')

print(train.shape)
print(train.columns)
train.head()


train.info()


gender_dummies = pd.get_dummies(train['Gender'], prefix='Gender')

train = train.drop('Gender', axis=1)
train = pd.concat([train, gender_dummies], axis=1)



print(train.columns)
print(test.columns)


print(train.isnull().sum())
print(test.isnull().sum())


unique_geographies = train['Geography'].unique()
print(f"Jami {len(unique_geographies)} mamlakat bor: {unique_geographies}")


geography_counts = train.groupby('Geography')['CustomerId'].count()
print(geography_counts)


train = pd.get_dummies(train, columns=['Geography'], drop_first=False)
test = pd.get_dummies(test, columns=['Geography'], drop_first=False)


train[['Geography_France', 'Geography_Spain', 'Geography_Germany']] = train[['Geography_France', 'Geography_Spain', 'Geography_Germany']].astype(int)
test[['Geography_France', 'Geography_Spain', 'Geography_Germany']] = test[['Geography_France', 'Geography_Spain', 'Geography_Germany']].astype(int)


train


train.info()


train = train.drop(columns=['Surname'])
test = test.drop(columns=['Surname'])


train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


train['Age_Bin_Senior'] = (train['Age'] >= 50).astype(int)

train['Products_1'] = (train['NumOfProducts'] == 1).astype(int)
train['Products_2'] = (train['NumOfProducts'] == 2).astype(int)
train['Products_3'] = (train['NumOfProducts'] == 3).astype(int)
train['Products_3_plus'] = (train['NumOfProducts'] >= 3).astype(int)
train['Age_Products_1'] = train['Age'] * train['Products_1']
train['Germany_Products_1'] = train['Geography_Germany'] * train['Products_1']
train['Germany_Senior'] = train['Geography_Germany'] * train['Age_Bin_Senior']
train['Senior_Products'] = train['Age_Bin_Senior'] * train['NumOfProducts']

train['Products_Bin'] = pd.cut(train['NumOfProducts'], bins=[0, 1, 2, 5], labels=['Single', 'Double', 'Multiple'])
train = pd.get_dummies(train, columns=['Products_Bin'], drop_first=True)  # Single olib tashlanadi

final_columns_train = [
    'Exited', 'Age', 'NumOfProducts', 'Geography_Germany', 'Products_1',
    'Products_2', 'Products_3', 'Products_3_plus', 'Age_Bin_Senior',
    'Age_Products_1', 'Germany_Products_1', 'Germany_Senior',
    'Products_Bin_Multiple', 'Products_Bin_Double', 'Senior_Products'
]
train = train[final_columns_train]

numeric_cols = train.select_dtypes(include=['int64', 'float64', 'uint8']).columns
correlation_with_exited = train[numeric_cols].corr()['Exited'].sort_values(ascending=False)
print("Train dataseti korrelyatsiyasi:")
print(correlation_with_exited)


test['Age_Bin_Senior'] = (test['Age'] >= 50).astype(int)

test['Products_1'] = (test['NumOfProducts'] == 1).astype(int)
test['Products_2'] = (test['NumOfProducts'] == 2).astype(int)
test['Products_3'] = (test['NumOfProducts'] == 3).astype(int)
test['Products_3_plus'] = (test['NumOfProducts'] >= 3).astype(int)
test['Age_Products_1'] = test['Age'] * test['Products_1']
test['Germany_Products_1'] = test['Geography_Germany'] * test['Products_1']
test['Germany_Senior'] = test['Geography_Germany'] * test['Age_Bin_Senior']
test['Senior_Products'] = test['Age_Bin_Senior'] * test['NumOfProducts']

test['Products_Bin'] = pd.cut(test['NumOfProducts'], bins=[0, 1, 2, 5], labels=['Single', 'Double', 'Multiple'])
test = pd.get_dummies(test, columns=['Products_Bin'], drop_first=True)  # Single olib tashlanadi

final_columns_test = [
    'Age', 'NumOfProducts', 'Geography_Germany', 'Products_1',
    'Products_2', 'Products_3', 'Products_3_plus', 'Age_Bin_Senior',
    'Age_Products_1', 'Germany_Products_1', 'Germany_Senior',
    'Products_Bin_Multiple', 'Products_Bin_Double', 'Senior_Products'
]
test = test[final_columns_test]


train.corr()['Exited'].sort_values(ascending=False)



print("Dataset hajmi:", train.shape)
print("Sonli ustunlar:", train.select_dtypes(include=['int64', 'float64']).columns.tolist())


train.info()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_exited.index, y=correlation_with_exited.values, palette='coolwarm')
plt.title('Exited ustuni bilan korrelyatsiya')
plt.xticks(rotation=45)  
plt.show()



print(test.columns)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X = train.drop(columns=['Exited'])
y = train['Exited']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

val_preds = lr_model.predict_proba(X_val)[:, 1]
roc_score = roc_auc_score(y_val, val_preds)
print(f"Validation ROC AUC: {roc_score:.4f}")


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X = train.drop(columns=['Exited'])
y = train['Exited']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train_scaled, y_train)

val_preds = rf_model.predict_proba(X_val_scaled)[:, 1]

roc_score = roc_auc_score(y_val, val_preds)
print(f"Validation ROC AUC: {roc_score:.4f}")



X_test = test.drop(columns=['Exited'], errors='ignore')  
X_test_scaled = scaler.transform(X_test)
test_preds = rf_model.predict_proba(X_test_scaled)[:, 1]
test['Exited_Probabilities'] = test_preds
test.to_excel('test_predictions.xlsx', index=False)
print(test[['Exited_Probabilities']].head())


test.info()


import pandas as pd
sample = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/sample_submission.csv')

sample['Exited'] = test['Exited_Probabilities']
sample.to_csv('submission.csv', index=False)
sample




