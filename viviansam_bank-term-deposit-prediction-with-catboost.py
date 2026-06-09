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
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
df.info()


df.head()


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


# Check unique value of categorical variables
# List of categorical variables
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

# Check unique values for each categorical variable
for feature in categorical_features:
    unique_features = df[feature].unique()
    print(f"Unique values for {feature}: {unique_features}")


# separate '-1' from pdays

# new binary column: was the client contacted before?
df['pdays_contacted_or_not'] = df['pdays'].apply(lambda x: 0 if x == -1 else 1)

# replace -1 with a 0 for original column
df['pdays'] = df['pdays'].replace(-1, 0)


# Data splitting
X = df.drop(columns=['y']) # features
y = df['y'] # target variable

# Split into at 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Identify categorical columns
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()


# Train CatBoost
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    eval_metric='AUC',
    verbose=100,
    random_state=42
)

model.fit(X_train, y_train, cat_features=cat_features, 
          eval_set=(X_test, y_test), early_stopping_rounds=50)


# ROC AUC Score
y_prob = model.predict_proba(X_test)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))


# Prepare to submit test set
df_test.info()


# Keep the 'ID' column separate
id_test = df_test['id']  

# Drop the 'ID' column from df_test
df_test = df_test.drop(columns=['id'])


# preprocess pdays
df_test['pdays_contacted_or_not'] = df_test['pdays'].apply(lambda x: 0 if x == -1 else 1)
df_test['pdays'] = df_test['pdays'].replace(-1, 0)


# Predict
y_prob_test = model.predict_proba(df_test)[:, 1]


# Create a DataFrame with 'ID' and 'y' columns
output = pd.DataFrame({'id': id_test, 'y': y_prob_test})
output.head()


output.to_csv('submission.csv', index=False)

