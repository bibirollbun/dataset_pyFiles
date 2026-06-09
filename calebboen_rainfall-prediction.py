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


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train.head()


train.info()


test.info()


train.shape, test.shape


train.isnull().sum()


test.isnull().sum()


# check for duplicated entries
train.duplicated().sum(), test.duplicated().sum()


train = train.drop(columns=['id'])
test = test.drop(columns=['id'])
train.head()


numerical_variables = ['winddirection', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall' 
categorical_variables = []

train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()


# Analysis of all NUMERICAL features

# Define a custom color palette
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
train['Dataset'] = 'Train'
test['Dataset'] = 'Test'

variables = [col for col in train.columns if col in numerical_variables]

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train, test]), x=variable, y
    ="Dataset", palette=custom_palette)
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN, TEST]")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each variable
for variable in variables:
    create_variable_plots(variable)


# Drop the 'Dataset' column after analysis
train.drop('Dataset', axis=1, inplace=True)
test.drop('Dataset', axis=1, inplace=True)


plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), annot=True)


data = train.copy()
data['season'] = data['day'] % 365

def get_season(day):
    month = (day % 365) // 30 + 1
    if month in [12, 1, 2]:
        return 0 #'Winter'
    elif month in [3, 4, 5]:
        return 1 #'Spring'
    elif month in [6, 7, 8]:
        return 2 #'Summer'
    else:
        return 3 #'Autumn'

data['season'] = data['day'].apply(get_season)
# data = pd.get_dummies(data, columns=['season'], dtype=int)
data.head()


data['day_of_year'] = data['day'] % 365
data['sin_day'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
data['cos_day'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

data.head()


data['temp_range'] = data['maxtemp'] - data['mintemp']
data['temp_dew_diff'] = data['temparature'] - data['dewpoint']

data['humid_temp'] = data['humidity'] * data['temparature']
data['cloud_sun_ratio'] = data['cloud'] / (data['sunshine'] + 1)

data['wind_speed_category'] = pd.cut(data['windspeed'], bins=[0, 10, 20, 30, 50, 100], labels=[1, 2, 3, 4, 5])
data['wind_speed_category'] = data['wind_speed_category'].astype('int')

# data['rainfall_lag1'] = data['rainfall'].shift(1).fillna(0)
# data['rainfall_lag3'] = data['rainfall'].shift(3).fillna(0)

data.head()


drop_cols = ['day', 'day_of_year', 'maxtemp',]
data.drop(columns=drop_cols, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(data.drop('rainfall', axis=1), data['rainfall'], 
                                                    stratify=data['rainfall'], random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc_cols  = ['pressure', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'humid_temp',	'cloud_sun_ratio']
X_train[sc_cols] = sc.fit_transform(X_train[sc_cols])
X_test[sc_cols] = sc.transform(X_test[sc_cols])

X_train.head()


models = {
    'Logistic_Reg' : LogisticRegression(),
    'SVC' : LinearSVC(),
    'DT' : DecisionTreeClassifier(),
    'RF' : RandomForestClassifier(),
    'XGB': XGBClassifier(),
    'Cat' : CatBoostClassifier(verbose=0),
    'LGB': LGBMClassifier(verbose=0),
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}

for name, model in models.items():
    scores = cross_val_score(model, data.drop('rainfall', axis=1), data['rainfall'], cv=cv, scoring="roc_auc")
    cv_scores[name] = scores
    print(f"{name}: Mean ROC-AUC = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")

print("\nDetailed Scores:")
for name, scores in cv_scores.items():
    print(f"{name}: {scores}")


xgb = XGBClassifier()
xgb.fit(X_train, y_train)

importances = xgb.feature_importances_
columns = X_train.columns

sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(y=np.array(columns)[sorted_idx], width=importances[sorted_idx], color="green")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.show()


cat = CatBoostClassifier(verbose=0)
cat.fit(X_train, y_train)

importances = cat.feature_importances_
columns = X_train.columns

sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(y=np.array(columns)[sorted_idx], width=importances[sorted_idx], color="purple")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("CatBoost Feature Importance")
plt.show()


lgb = LGBMClassifier()
lgb.fit(X_train, y_train)

importances = lgb.feature_importances_
columns = X_train.columns

sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(y=np.array(columns)[sorted_idx], width=importances[sorted_idx], color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("LGB Feature Importance")
plt.show()


model = LogisticRegression()
model.fit(X_train, y_train)
roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


model.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))


def process_test(data:pd.DataFrame):
    data['season'] = data['day'].apply(get_season)
    
    data['day_of_year'] = data['day'] % 365
    data['sin_day'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
    data['cos_day'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

    data['temp_range'] = data['maxtemp'] - data['mintemp']
    data['temp_dew_diff'] = data['temparature'] - data['dewpoint']

    data['humid_temp'] = data['humidity'] * data['temparature']
    data['cloud_sun_ratio'] = data['cloud'] / (data['sunshine'] + 1)
    # print(data.head())

    data['wind_speed_category'] = pd.cut(data['windspeed'], bins=[0, 10, 20, 30, 50, 100], labels=[1, 2, 3, 4, 5])
    data['wind_speed_category'] = data['wind_speed_category'].astype('int')
    
    # data['rainfall_lag1'] = data['rainfall'].shift(1).fillna(0)
    # data['rainfall_lag3'] = data['rainfall'].shift(3).fillna(0)
    drop_cols = ['day', 'day_of_year', 'maxtemp',]
    data.drop(columns=drop_cols, inplace=True)
    data[sc_cols] = sc.transform(data[sc_cols])
    
    data.fillna(0, inplace=True, axis=0)
    return data


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv').set_index('id')


test = process_test(test)
test.head()


submission = model.predict_proba(test)[:, 1]
subfile = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
subfile['rainfall'] = submission

subfile.to_csv('submission.csv', index=False)

subfile.head()




