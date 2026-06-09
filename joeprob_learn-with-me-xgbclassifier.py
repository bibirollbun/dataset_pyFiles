# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_set = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id') # set the column 'id' as the index

print(f"The train set contains {train_set.shape[0]} rows and {train_set.shape[1]} colums")
train_set.head()


test_set = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col='id')

print(f"The test set contains {test_set.shape[0]} rows and {test_set.shape[1]} colums")
test_set.head()


# General information
train_set.info()


# Overview of null values
train_set.isnull().sum()


# Class imbalance
train_set[["Personality"]].value_counts()


# create a function to make a KDE plot for both the train and test set
# for more information, see https://seaborn.pydata.org/tutorial/distributions.html#tutorial-kde
def plot_kde(data, cols, target=None):
    sns.set(style="whitegrid")
    
    for col in cols:
        plt.figure(figsize=(5, 3))
        if target != None:
            sns.kdeplot(data=data, x=col, hue=target, fill=True, common_norm=False, alpha=0.5)
        else:
            sns.kdeplot(data=data, x=col, fill=True, common_norm=False, alpha=0.5)
        plt.title(f'KDE Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.tight_layout()
        plt.show()


numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

# Class distributions over numerical features
plot_kde(data=train_set, cols=numerical_features, target='Personality')


categorical_features = ["Drained_after_socializing", "Stage_fear"]

# display the distribution of categories per Personality class
for col in categorical_features:
    distribution = train_set[[col, 'Personality']].groupby(by=[col, 'Personality']).size().unstack(fill_value=0)
    print(distribution)
    print("")


test_set.info()


test_set.isnull().sum()


plot_kde(data=test_set, cols=numerical_features)


# display the distribution of categories
for col in categorical_features:
    print(test_set[[col]].value_counts())
    print("")


# Map categorical values to numerical values
for column in categorical_features:
    train_set[column] = train_set[column].map({'Yes': 1, 'No': 0})
    test_set[column] = test_set[column].map({'Yes': 1, 'No': 0})

train_set['Personality'] = train_set['Personality'].map({'Introvert': 1, 'Extrovert': 0})


train_set.head()


# Interaction features
train_set['Alone_to_Social_Ratio'] = train_set['Time_spent_Alone'] / (train_set['Social_event_attendance'] + 1)
test_set['Alone_to_Social_Ratio'] = test_set['Time_spent_Alone'] / (test_set['Social_event_attendance'] + 1)

train_set['Social_Comfort_Index'] = (train_set['Friends_circle_size'] + train_set['Post_frequency'] - train_set['Stage_fear']) / 3
test_set['Social_Comfort_Index'] = (test_set['Friends_circle_size'] + test_set['Post_frequency'] - test_set['Stage_fear']) / 3

train_set['Social_Overload'] = train_set['Drained_after_socializing'] * train_set['Social_event_attendance']
test_set['Social_Overload'] = test_set['Drained_after_socializing'] * test_set['Social_event_attendance']


X = train_set.drop(["Personality"], axis=1)
y = train_set[["Personality"]]


X.head()


y.head()


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


model_base = XGBClassifier(random_state=42)

model_base.fit(X_train, y_train)

prediction_base = model_base.predict(X_val)

from sklearn.metrics import accuracy_score
accuracy_base = accuracy_score(y_val, prediction_base)

print(accuracy_base)


# Get feature importances
importances = model_base.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for easy sorting and plotting
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print top features
print(feat_imp_df)


model_early = XGBClassifier(n_estimators=1000,       # Large number to allow early stopping
                            learning_rate=0.005,
                            use_label_encoder=False,
                            eval_metric='logloss')    # Required to suppress warning


model_early.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=5,  # Stop if no improvement after 10 rounds
                verbose=False)

predictions_early = model_early.predict(X_val)
accuracy_early = accuracy_score(y_val, predictions_early)

print(accuracy_early)


# Get feature importances
importances = model_early.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for easy sorting and plotting
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print top features
print(feat_imp_df)


predictions = model_early.predict(test_set)

output = pd.DataFrame({'id': test_set.index, 'Personality': predictions})
output['Personality'] = output['Personality'].map({1: 'Introvert', 0: 'Extrovert'}) # reverse the mapping of the predictions

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

