pip install -U scikit-learn==1.3.2 imbalanced-learn==0.11.0


!pip install imbalanced-learn


import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer  
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, confusion_matrix, classification_report



import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/webbee-internship-2025/train.csv')
test = pd.read_csv('/kaggle/input/webbee-internship-2025/test.csv')
test_copy = test.copy()


test.head()


train.head()


train.info()


train.describe(include='all')


train = train.drop(['id','Surname','CustomerId'],axis=1)
test = test.drop(['id', 'Surname', 'CustomerId'], axis=1)


x_values = train.select_dtypes(include=['number'])
x_values = x_values.drop('Exited',axis=1)



corr_matrix = train.select_dtypes(include='number').corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt='.1f',cmap="Blues")
plt.show() 



plt.figure(figsize=(10, 8))
sns.histplot(data=train, x='Age', hue='Exited', kde=False, bins=30, element='step', stat='count', common_norm=False)
plt.title('DistribuiÃ§Ã£o da Idade por Status de SaÃ­da (Exited)')
plt.xlabel('Idade')
plt.ylabel('Contagem')
plt.legend(title='Saiu (Exited)', labels=['Ficaram (0)', 'SaÃ­ram (1)'])
plt.show()



plt.figure(figsize=(10, 8))
sns.countplot(data=train, x='Exited', hue='Geography')
plt.title('Contagem de Clientes que SaÃ­ram ou Permaneceram por RegiÃ£o')
plt.xlabel('Saiu (Exited)')
plt.ylabel('Contagem')
plt.legend(title='Geography')
plt.show()



plt.figure(figsize=(10, 8))
sns.countplot(x='Exited', data=train)
plt.title('label Distribution')
plt.show()



fig, axis = plt.subplots(nrows=4, ncols=2, figsize=(16, 22))

for ax, x_value in zip(axis.flat, x_values):
    sns.kdeplot(data=train, x=x_value, fill=True,hue='Exited', common_norm=False, alpha=0.5, ax=ax)
    ax.set_title(f'{x_value.capitalize()}')
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 22))

for i, x_value in enumerate(x_values):
    ax = axes.flatten()[i] 
    sns.boxplot(data=train, x='Exited', y=x_value, hue='Exited', ax=ax,palette=["#006992", "#ff7d00"])
    ax.set_title(f'{x_value.capitalize()}')
    ax.set_ylabel(x_value.capitalize())
plt.tight_layout()
plt.show()


train['CardAndActive'] = ((train['HasCrCard'] == 1) & (train['IsActiveMember'] == 1)).astype(int)
test['CardAndActive'] = ((test['HasCrCard'] == 1) & (test['IsActiveMember'] == 1)).astype(int)


categorical_columns = train.select_dtypes(include=['object', 'category']).columns
numerical_columns = train.select_dtypes(include=['float64', 'int64']).columns
numerical_columns = numerical_columns.drop('Exited')


def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

features_to_clean = ['Age', 'EstimatedSalary']
train = remove_outliers_iqr(train, features_to_clean)


def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':  
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())  
    return df

train = fill_missing_values(train)
test = fill_missing_values(test)


def _one_hot_encode_columns(df, categorical_columns):
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df_encoded

train = _one_hot_encode_columns(train,categorical_columns)
test = _one_hot_encode_columns(test,categorical_columns)


def scale_numerical_features(df, numerical_columns):
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

train = scale_numerical_features(train,numerical_columns)
test = scale_numerical_features(test,numerical_columns)


X = train.drop('Exited', axis=1)
y = train['Exited'] 


adasyn = ADASYN(random_state=42)
X, y = adasyn.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


X_train.describe(include='all')


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=13,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.7,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"AcurÃ¡cia no conjunto de teste: {acc:.4f}")
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred) 

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()



y_test_proba = model.predict_proba(test)[:, 1]

submission = pd.DataFrame({
    'id': test_copy['id'],
    'target': y_test_proba
})

submission.to_csv('submission4.csv', index=False)

