import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# import library
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
df.info()


# Drop the 'ID' column
df = df.drop(columns=['id'])


# Check the range of numeric variables
# List of numeric variables
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Check min max for each numeric variable
for feature in numeric_features:
    min_feature = df[feature].min()
    max_feature = df[feature].max()
    print(f"{feature}: Min: {min_feature}, Max: {max_feature}")


# separate '-1' from pdays

# new binary column: was the client contacted before?
df['pdays_contacted_or_not'] = df['pdays'].apply(lambda x: 0 if x == -1 else 1)

# replace -1 with a 0 for original column
df['pdays'] = df['pdays'].replace(-1, 0)


# Check unique value of categorical variables
# List of categorical variables
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

# Check unique values for each categorical variable
for feature in categorical_features:
    unique_features = df[feature].unique()
    print(f"Unique values for {feature}: {unique_features}")


# convert month to numerical values
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

df['month'] = df['month'].str.lower().map(month_mapping)


# convert binary feature to '0' & '1' 
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    df[col] = df[col].map({'no': 0, 'yes': 1})


# one-hot encoding for other categorical features
cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


# Data splitting
X = df.drop(columns=['y']) # features
y = df['y'] # target variable

# Split into at 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# XG Boost
xgb = XGBClassifier(random_state=42, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

# Predict probabilities
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

# Predict classes
y_pred_xgb = xgb.predict(X_test)

# ROC AUC
roc_auc = roc_auc_score(y_test, y_proba_xgb)
print(f"ROC AUC Score: {roc_auc:.4f}")


# Prepare to submit test set
df_test.info()


# Keep the 'ID' column separate
id_test = df_test['id']  

# Drop the 'ID' column from df_test
df_test = df_test.drop(columns=['id'])


# preprocess pdays
df_test['pdays_contacted_or_not'] = df_test['pdays'].apply(lambda x: 0 if x == -1 else 1)
df_test['pdays'] = df_test['pdays'].replace(-1, 0)

# preprocess month
df_test['month'] = df_test['month'].str.lower().map(month_mapping)

# preprocess binary feature
for col in binary_cols:
    df_test[col] = df_test[col].map({'no': 0, 'yes': 1})

# one-hot encoding
df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)


# Feature scaling
df_test = scaler.transform(df_test)


# Predict using XG Boost
y_proba_xgb_2 = xgb.predict_proba(df_test)[:, 1]
y_proba_xgb_2


# Create a DataFrame with 'ID' and 'y' columns
output = pd.DataFrame({'id': id_test, 'y': y_proba_xgb_2})
output.head()


output.to_csv('submission.csv', index=False)

