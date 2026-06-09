# Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


data_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

data_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


data_train.head()


data_test.head()


# Shape of the dataset

rows, cols = data_train.shape
print(f"The Train dataset contains: Rows: {rows}, Columns: {cols}")

rows, cols = data_test.shape
print(f"The Test dataset contains: Rows: {rows}, Columns: {cols}")


# Data information

data_train.info()

data_test.info()


print('Train Data Columns: ')
data_train.columns


print('Test Data Columns: ')
data_test.columns


data_train.drop(columns = ['id'], inplace = True)

data_test.drop(columns = ['id'], inplace = True)


data_train.head(2)


# Missing Values in Train Data 

data_train.isnull().sum()


# Missing Values in Test Data 

data_test.isnull().sum()


#  Train data Summary Statistics 

print('Train Data Summary Statistics: ')
data_train.describe()


# Test Data Summary Statistics

print('Test Data Summary Statistics: ')
data_test.describe()


# Target column value counts

data_train['y'].value_counts()


# Target column distribution

plt.figure(figsize=(6,4))
sns.countplot(x = 'y', data = data_train, palette = "Paired")
plt.title("Target Column Distribution", fontsize = 14)
plt.xlabel("Target", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()



# Categorical features in Train Data

categorical_features = [data for data in data_train.columns if data_train[data].dtype == 'object']

for col in data_train.columns:
    if col in categorical_features:
        print(col, data_train[col].unique())
        print("-" * 50)


# Value count for categorical features

for col in categorical_features:
        print(f"\nValue counts for {col}:")
        print(data_train[col].value_counts())


# Numerical features in Train Data

numerical_features = [data for data in data_train.columns if data_train[data].dtype != 'object']
print(numerical_features)


# Distribution plots for numerical features
print("Distribution of Numerical Features: ")

plt.figure(figsize=(15, 10))

for i, col in enumerate(numerical_features, 1):
    plt.subplot(len(numerical_features)//3 + 1, 3, i) 
    sns.histplot(data_train[col], kde = True, bins = 30, color = "darkorange")
    plt.title(f"{col}", fontsize=11)
    plt.xlabel("")
    
plt.tight_layout()
plt.show()


# Box plots for outlier detection

print("Box Plots for Outlier Detection: ")

for col in numerical_features:
    plt.figure(figsize = (8, 4))
    sns.boxplot(x = data_train[col], color = 'seagreen')
    plt.title(f'Box plot of {col}')
    plt.show()



# Checking for Multicollinearity in the Numerical Dataset

sns.heatmap(data_train.select_dtypes("number").drop(columns = "y").corr()); 


from sklearn.preprocessing import LabelEncoder

def label_encoding(data_train, data_test):
    """
    Encodes categorical features in training and test datasets 
    using LabelEncoder. Unseen categories in test data are mapped to -1.
    """

    # Make copies so original data is untouched
    data_train_enc = data_train.copy()
    data_test_enc = data_test.copy()

    label_encoders = {}

    # Encode categorical features in training set
    for column in data_train_enc.columns:
        if data_train_enc[column].dtype == 'object':
            le = LabelEncoder()
            data_train_enc[column] = le.fit_transform(data_train_enc[column].astype(str))
            label_encoders[column] = le

    # Encode categorical features in test set using train encoders
    for column in data_test_enc.columns:
        if column in label_encoders:
            le = label_encoders[column]
            data_test_enc[column] = data_test_enc[column].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        elif data_test_enc[column].dtype == 'object':
            data_test_enc[column] = -1  # default encoding for unseen categorical column

    return data_train_enc, data_test_enc, label_encoders


data_train, data_test, label_encoders = label_encoding(data_train, data_test)


data_train.head()


data_test.head()


from sklearn.preprocessing import StandardScaler

def data_standardization(data_train, data_test, target_variable):
    """
    Standardize numerical features in both train and test datasets 
    using StandardScaler, while preserving the target column in train data.
    """

    # Store the target column separately from the training data, only features should be scaled
    target_values = data_train[target_variable]
    data_train = data_train.drop(columns = [target_variable])

    # Ensure train and test have the same set of feature columns
    common_columns = data_train.columns.intersection(data_test.columns)
    data_train = data_train[common_columns]
    data_test = data_test[common_columns]

    std_scaler = StandardScaler()

    # Fit the scaler on train data, then transform both train and test
    data_train_scaled = pd.DataFrame(std_scaler.fit_transform(data_train), columns = common_columns)
    data_test_scaled = pd.DataFrame(std_scaler.transform(data_test), columns = common_columns)

    # Reattach the target column back to the scaled training data
    data_train_scaled[target_variable] = target_values.reset_index(drop = True)

    
    return data_train_scaled, data_test_scaled


data_train_scaled, data_test_scaled = data_standardization(data_train, data_test, 'y')


data_train_scaled.head()


data_test_scaled.head()


X = data_train_scaled.drop(columns = ['y'])
Y = data_train_scaled['y']


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 42)


X.shape, X_train.shape, X_test.shape


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score


# Base Models

cat_model = CatBoostClassifier(n_estimators = 500, verbose = 0, random_state = 42)
lgb_model = LGBMClassifier(n_estimators = 500, random_state = 42, verbose = -1)
xgb_model = XGBClassifier(n_estimators = 500, random_state = 42, use_label_encoder = False, eval_metric = 'logloss')
rf_model = RandomForestClassifier(n_estimators = 300, random_state = 42)


voting_clf = VotingClassifier(
    estimators = [('cat', cat_model), ('lgb', lgb_model), ('xgb', xgb_model), ('rf', rf_model)],
    voting = 'soft', n_jobs = -1
)


stacking_clf = StackingClassifier(
    estimators=[('cat', cat_model), ('lgb', lgb_model), ('xgb', xgb_model), ('rf', rf_model)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv = 5, n_jobs = -1
)


models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "LightGBM": lgb_model,
    "CatBoost": cat_model,
    "Voting Ensemble": voting_clf,
    "Stacking Ensemble": stacking_clf
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    print(f"ROC AUC: {roc_auc_score(Y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(Y_test, y_pred):.4f}")
    print(classification_report(Y_test, y_pred))

    cm = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix:\n", cm)


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sample_submission.head()


sample_submission['y'].shape


prediction = voting_clf.predict_proba(data_test_scaled)


prediction


# Extracting probabilities of positive class

Y_probabilities = prediction[:, 1]
Y_probabilities



Y_probabilities.shape


# Replace the target column with your predicted probabilities

sample_submission['y'] = Y_probabilities


sample_submission.head()


sample_submission.to_csv('submission.csv', index = False)
print('Submission file saved.')


import os
os.listdir("/kaggle/working")


import os, shutil

os.makedirs("/kaggle/outputs", exist_ok=True)

# Copy the file
shutil.copy("/kaggle/working/submission.csv", "/kaggle/outputs/submission.csv")




