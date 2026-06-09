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
import seaborn as sns
import warnings
!pip install optuna
warnings.filterwarnings("ignore")


# import the datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


# Quick preview
train.shape  # Returns (number_of_rows, number_of_columns)


train.head()  # Displays the first 5 rows


train.info()


train.isnull().sum()


# Step 1: Visualize Target Variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Fertilizer Name', data=train)
plt.title('Distribution of Fertilizer Name')
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Step 2: Visualize Numerical Features
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# Step 3: Visualize Categorical Features
categorical_cols = ['Soil Type', 'Crop Type']
plt.figure(figsize=(12, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 2, i)
    sns.countplot(x=col, data=train)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Step 4: Numerical Features vs. Target
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Fertilizer Name', y=col, data=train)
    plt.title(f'{col} vs. Fertilizer Name')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Step 5: Categorical Features vs. Target
soil_fertilizer_crosstab = pd.crosstab(train['Soil Type'], train['Fertilizer Name'], normalize='index')
plt.figure(figsize=(10, 6))
sns.heatmap(soil_fertilizer_crosstab, annot=True, cmap='Blues', fmt='.2f')
plt.title('Soil Type vs. Fertilizer Name (Normalized)')
plt.show()


# Step 6: Outliers
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, 6, i)
    sns.boxplot(y=train[col])
    plt.title(col)
plt.tight_layout()
plt.show()


# Step 7: Correlation Matrix
correlation_matrix = train[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


print("Unique Soil Types:")
print(train['Soil Type'].unique())

print("\nUnique Crop Types:")
print(train['Crop Type'].unique())

print("\nUnique Fertilizer Types:")
print(train['Fertilizer Name'].unique())



test.shape


test.head()


test=test.drop(["id"],axis=1)


test.info()


test.isnull().sum()


test.duplicated().sum()


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Convert categorical columns to numeric in train dataset
train['Soil Type'] = label_encoder.fit_transform(train['Soil Type'])
train['Crop Type'] = label_encoder.fit_transform(train['Crop Type'])
train['Fertilizer Name'] = label_encoder.fit_transform(train['Fertilizer Name'])

# Convert categorical columns to numeric in test dataset
test['Soil Type'] = label_encoder.fit_transform(test['Soil Type'])
test['Crop Type'] = label_encoder.fit_transform(test['Crop Type'])


train.info()


from sklearn.model_selection import train_test_split

X = train.drop(['Fertilizer Name', 'id'], axis=1)
y = train['Fertilizer Name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"X_train: ",X_train.shape)
print(f"X_test: ",X_test.shape)
print(f"y_train: ",y_train.shape)
print(f"y_test: ",y_test.shape)


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit only on the training data
scaler.fit(X_train)

# Transform all datasets using the same scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_original_test_scaled = scaler.transform(test)



import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def objective(trial):
    # Define hyperparameters to search
    params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
    }
    
    clf = DecisionTreeClassifier(**params, random_state=42)
    
    # Use cross-validation on the training set to avoid overfitting
    score = cross_val_score(clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
    return np.mean(score)



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


best_params = study.best_params
print("Best Hyperparameters:", best_params)

final_model = DecisionTreeClassifier(**best_params, random_state=42)
final_model.fit(X_train_scaled, y_train)


y_pred = final_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Validation Accuracy:", accuracy)

