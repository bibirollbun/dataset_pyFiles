import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train_data.head()


## Shape of this data
print(f"Shape of training data : {train_data.shape}")
print(f"Number of rows in training data : {train_data.shape[0]}")
print(f"Number of columns in training data : {train_data.shape[1]}")


## Describing the data and getting info
train_data.describe()


## getting info for the data
train_data.info()


## Drop the id column 
train_data = train_data.drop(columns=['id'])
train_data.head()


import matplotlib.pyplot as plt
import seaborn as sns


## Categorical columns and Numerical columns
categorical_cols = train_data.select_dtypes(include='object').columns
numerical_cols = train_data.select_dtypes(exclude='object').columns

print(f"Categorical columns : {categorical_cols}")
print(f"Numerical columns : {numerical_cols}")


## Countplot for Catgeorical column
## Creates 1 row x 3 columns
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes = axes.flatten()

for i, feature in enumerate(categorical_cols):
    sns.countplot(x=feature, data=train_data, ax=axes[i])
    axes[i].set_title(f"{feature}")
    axes[i].tick_params(axis='x', rotation=60)

## Improve spacing
plt.tight_layout()
plt.show()


## Prabability distribution for Numerical columns 
fig, axes = plt.subplots(5, 1, figsize=(14, 20))
axes = axes.flatten()

for i, feature in enumerate(numerical_cols):
    sns.histplot(x=feature, data=train_data, ax=axes[i], kde=True)
    axes[i].set_title(f"{feature}")
    axes[i].tick_params(axis='x', rotation=60)

## Improve spacing
plt.tight_layout()
plt.show()


## using hue as "Personality" column
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes = axes.flatten()

for i, feature in enumerate(categorical_cols):
    sns.countplot(x=feature, data=train_data, ax=axes[i], hue='Personality')
    axes[i].set_title(f"{feature}")
    axes[i].tick_params(axis='x', rotation=60)

## Improve spacing
plt.tight_layout()
plt.show()


## Prabability distribution for Numerical columns with hue as "Personality"
fig, axes = plt.subplots(5, 1, figsize=(14, 20))
axes = axes.flatten()

for i, feature in enumerate(numerical_cols):
    sns.histplot(x=feature, data=train_data, ax=axes[i], kde=True, hue="Personality")
    axes[i].set_title(f"{feature}")
    axes[i].tick_params(axis='x', rotation=60)

## Improve spacing
plt.tight_layout()
plt.show()


X_features = train_data.drop(columns=['Personality'])
y_labels = train_data['Personality']


## numerical and categorical columns
numerical_columns = X_features.select_dtypes(exclude='object').columns
categorical_columns = X_features.select_dtypes(include='object').columns

print(f"Numerical columns : {numerical_columns}")
print(f"Categorical columns : {categorical_columns}")


from sklearn.model_selection import train_test_split
X_train_features, X_test_features, y_train_labels, y_test_labels = train_test_split(X_features, y_labels, test_size=0.2, random_state=42, shuffle=True)
print(f"Number of features for training : {X_train_features.shape[0]}")
print(f"Number of features for testing : {X_test_features.shape[0]}")


## using Label encoder for Labels and OHE for categorical features
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

label_enc = LabelEncoder()
ct = ColumnTransformer([
    ("one_hot_encoding", OneHotEncoder(), categorical_columns),
    ("standard_scaler", StandardScaler(), numerical_columns)
])

ct.fit(X_train_features)
label_enc.fit(y_train_labels)


X_train_new = ct.transform(X_train_features)
y_train_new = label_enc.transform(y_train_labels)
X_test_new = ct.transform(X_test_features)
y_test_new = label_enc.transform(y_test_labels)

## checking the dtypes and shapes
print(f"Shape of X_train_new : {X_train_new.shape}")
print(f"Data type of X_train_new : {type(X_train_new)}")


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.01, objective='binary:logistic')
xgb_model.fit(X_train_new, y_train_new)

y_preds = xgb_model.predict(X_test_new)
xgb_accuracy = accuracy_score(y_test_new, y_preds)
print(xgb_accuracy)


## function to generate submission.csv

def generate_submission(model, test_data):
    ids = test_data['id']
    X = test_data.drop(columns=['id'])
    X = ct.transform(X)
    y_preds = model.predict(X)
    ids = pd.Series(ids)
    labels = label_enc.inverse_transform(y_preds)
    labels = pd.Series(labels)
    submission_df = pd.DataFrame({
        'id': ids,
        'Personality': labels
    })

    submission_df.to_csv("Submission.csv", index=False)


generate_submission(xgb_model, test_data)




