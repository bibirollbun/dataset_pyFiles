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


df_train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_train.info()


df_train.head(2)


df_test.info()


df_train.isna().sum()


# Filled Social_event_attendance based on Personality                      
df_train['Social_event_attendance'] = df_train.groupby('Personality')['Social_event_attendance']\
    .transform(lambda x: x.fillna(x.mean()))


# Filled Social_event_attendance based on Personality                      
df_test['Social_event_attendance'] = df_train.groupby('Personality')['Social_event_attendance']\
    .transform(lambda x: x.fillna(x.mean()))


# Social Event Attendance
df_train.loc[(~df_train.Stage_fear.isna()) & (df_train.Social_event_attendance <=2.0) ,['Stage_fear']].value_counts().plot.bar()


# Filled Stage_fear based on Social_event_attendance
df_train['Stage_fear'] = df_train.groupby('Social_event_attendance')['Stage_fear']\
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))


# Filled Stage_fear based on Social_event_attendance
df_test['Stage_fear'] = df_train.groupby('Social_event_attendance')['Stage_fear']\
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))


df_train.isna().sum()


# List of target numeric columns
numerical_cols = [
    'Time_spent_Alone', 
    'Going_outside',  
    'Friends_circle_size', 
    'Post_frequency'
]


# Convert columns to numeric (if not already), coerce errors to NaN
for col in numerical_cols:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce')


# Convert columns to numeric (if not already), coerce errors to NaN
for col in numerical_cols:
    df_test[col] = pd.to_numeric(df_test[col], errors='coerce')


# Impute missing values based on Personality group means
for col in numerical_cols:
    df_train[col] = df_train.groupby('Personality')[col].transform(
        lambda x: x.fillna(x.mean())
    )


# Impute missing values based on Personality group means
for col in numerical_cols:
    df_test[col] = df_train.groupby('Personality')[col].transform(
        lambda x: x.fillna(x.mean())
    )


df_train['Stage_fear'] = df_train.groupby('Social_event_attendance')['Stage_fear'] \
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))


df_test['Stage_fear'] = df_train.groupby('Social_event_attendance')['Stage_fear'] \
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))


df_train.Drained_after_socializing.unique()


df_train['Drained_after_socializing'] = df_train.groupby('Personality')['Drained_after_socializing']\
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))


df_test['Drained_after_socializing'] = df_train.groupby('Personality')['Drained_after_socializing']\
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))


df_train.isna().sum()


df_test.isna().sum()


df_train.columns


df_train.Personality.value_counts()


df_train.info()


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Step 3: Prepare Data
X = df_train.drop(columns=['id', 'Personality'])
y = df_train['Personality']  # Target: Extrovert / Introvert


df_test_prep=df_test.drop(columns=['id'])


# Define categorical feature indices
cat_features = ['Stage_fear', 'Drained_after_socializing']


# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



# Step 5: Build CatBoostClassifier
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='F1',
    class_weights=[1, 3],  # Extrovert:Introvert ~ 3:1 imbalance
    verbose=50,
    random_seed=42
)


# Step 6: Train the model
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))


# Step 7: Predict and Evaluate
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


personality=model.predict(df_test_prep)


df_test['Personality']=personality


df_test.head(2)


df_test.info()


submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': personality  # this should be the output from CatBoostClassifier.predict
})


# Save to CSV
submission.to_csv('submission.csv', index=False)




