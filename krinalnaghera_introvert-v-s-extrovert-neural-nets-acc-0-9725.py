# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly
import tensorflow as tf


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


train.describe()


train.columns


train.groupby("Personality").count()


train.isnull().sum()


import seaborn as sns
column_list = ['Time_spent_Alone', 'Social_event_attendance',
       'Going_outside', 'Friends_circle_size',
       'Post_frequency']
for column in column_list:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Personality', y=column, data=train)
        plt.title(f'Relationship between Personality and {column}')
        plt.show()


# @title Stage_fear vs Time_spent_Alone

from matplotlib import pyplot as plt
import seaborn as sns
figsize = (12, 1.2 * len(train['Stage_fear'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(train, x='Time_spent_Alone', y='Stage_fear', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)


# Check the percentage of missing values per column
missing_values_percentage = train.isnull().sum() / len(train) * 100
print("Percentage of missing values per column:")
print(missing_values_percentage)

# Option 1: Impute missing values with the mean (for numerical columns)
for col in train.select_dtypes(include=np.number).columns:
    if train[col].isnull().any():
        train[col].fillna(int(train[col].mean()), inplace=True)

#Option 2: Impute missing values with the mode (for categorical columns)
for col in train.select_dtypes(include='object').columns:
    if train[col].isnull().any():
        train[col].fillna(train[col].mode()[0], inplace=True)

# Verify that there are no more missing values
print("\nMissing values after handling:")
print(train.isnull().sum())


# Encode the categorical columns and target column "Personality"

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

encoder_stage_fear = OneHotEncoder(sparse_output=False)
encoder_drained = OneHotEncoder(sparse_output=False)

stage_fear_reshaped = train['Stage_fear'].values.reshape(-1, 1)
drained_reshaped = train['Drained_after_socializing'].values.reshape(-1, 1)

encoded_stage_fear = encoder_stage_fear.fit_transform(stage_fear_reshaped)
encoded_drained = encoder_drained.fit_transform(drained_reshaped)

stage_fear_cols = [f'Stage_fear_{cat}' for cat in encoder_stage_fear.categories_[0]]
drained_cols = [f'Drained_after_socializing_{cat}' for cat in encoder_drained.categories_[0]]

encoded_df_stage_fear = pd.DataFrame(encoded_stage_fear, columns=stage_fear_cols, index=train.index)
encoded_df_drained = pd.DataFrame(encoded_drained, columns=drained_cols, index=train.index)

train = train.drop(['Stage_fear', 'Drained_after_socializing'], axis=1)
train = pd.concat([train, encoded_df_stage_fear, encoded_df_drained], axis=1)


# Convert 'Personality' to binary
train['Personality'] = train['Personality'].apply(lambda x: 1 if x == 'Introvert' else 0)

train.head()


import sklearn
from sklearn.model_selection import train_test_split

X = train.drop(['Personality', 'id'], axis=1)
y = train['Personality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, verbose=1)


prediction = model.predict(X_test)
y_pred = (prediction > 0.5).astype(int).flatten()

confusion_mtx = sklearn.metrics.confusion_matrix(y_test, y_pred)
confusion_mtx


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


missing_values_percentage = test.isnull().sum() / len(test) * 100
print("Percentage of missing values per column:")
print(missing_values_percentage)

# Option 1: Impute missing values with the mean (for numerical columns)
for col in test.select_dtypes(include=np.number).columns:
    if test[col].isnull().any():
        test[col].fillna(int(test[col].mean()), inplace=True)

#Option 2: Impute missing values with the mode (for categorical columns)
for col in test.select_dtypes(include='object').columns:
    if test[col].isnull().any():
        test[col].fillna(test[col].mode()[0], inplace=True)

# Verify that there are no more missing values
print("\nMissing values after handling:")
print(test.isnull().sum())


stage_fear_reshaped = test['Stage_fear'].values.reshape(-1, 1)
drained_reshaped = test['Drained_after_socializing'].values.reshape(-1, 1)

# Fit and transform the columns
encoded_stage_fear = encoder_stage_fear.fit_transform(stage_fear_reshaped)
encoded_drained = encoder_drained.fit_transform(drained_reshaped)

encoded_df_stage_fear_test = pd.DataFrame(encoded_stage_fear, columns=stage_fear_cols, index=test.index)
encoded_df_drained_test = pd.DataFrame(encoded_drained, columns=drained_cols, index=test.index)

test = test.drop(['Stage_fear', 'Drained_after_socializing', 'id'], axis=1)
test = pd.concat([test, encoded_df_stage_fear_test, encoded_df_drained_test], axis=1)

# Make predictions
predictions = model.predict(test)

predicted_classes = (predictions > 0.5).astype(int)

predicted_labels = ['Introvert' if pred[0] == 1 else 'Extrovert' for pred in predicted_classes]


# Convert predictions to a DataFrame
predicted_df = pd.DataFrame({'Predicted_Personalities': predicted_labels})

# Save to CSV
predicted_df.to_csv('predicted_personalities.csv', index=False)

print("Predictions saved to 'predicted_personalities.csv'")




