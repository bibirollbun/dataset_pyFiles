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


## Training data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


## Sample size
train_data.shape


## Exploration 
train_data.head()


train_data.dtypes


## Missing data analysis 
percentage_missing = round(train_data.isna().sum()/train_data.shape[0],2)
percentage_missing.sort_values(ascending = False)


## Data split for outcomes 
import matplotlib.pyplot as plt


train_data.Personality.value_counts().plot(kind = 'bar')
plt.title('Split between Extroverts and Introverts')
plt.show()

print(f'Split is:\n {train_data.Personality.value_counts()/ train_data.shape[0]}')


## numeric 
train_data['Post_frequency'] = train_data['Post_frequency'].fillna(train_data['Post_frequency'].mean())
train_data['Going_outside'] = train_data['Going_outside'].fillna(train_data['Post_frequency'].mean())


## Check
print(train_data['Post_frequency'].mean())
print(train_data['Going_outside'].mean())


missing_cols = percentage_missing[percentage_missing > 0].index

missing_numeric = [col for col in missing_cols if train_data[col].dtype != object]
missing_categorical = [col for col in missing_cols if train_data[col].dtype == object]

## check
print(missing_numeric)
print(missing_categorical)


## Cleaning numeric 
def impute_mean_numeric(df, col):
    df[col] = df[col].fillna(df[col].mean())
    return df[col]

for col in missing_numeric:
    train_data[col] = impute_mean_numeric(train_data, col)
    print(train_data[col].isna().sum())


# ## Repeat for categorical 
def impute_mode_categorical(df, col):
    df[col] = df[col].fillna(df[col].mode()[0])
    return(df[col])

for col in missing_categorical:
    train_data[col] = impute_mode_categorical(train_data, col)
    print(train_data[col].isna().sum())


categorical_cols = [col for col in train_data.columns if train_data[col].dtype == 'O']
print(categorical_cols)
numeric_cols = [col for col in train_data.columns if train_data[col].dtype != 'O' and col != 'id']
print(numeric_cols)


## The categorical columns are all binary -- no need for OHE
train_data[categorical_cols].nunique()
for col in categorical_cols:
    print(train_data[col].value_counts())


import seaborn as sns

for numeric_col in numeric_cols:
    sns.violinplot(data = train_data, x = 'Personality', y = numeric_col)
    plt.show()


## Encode binaries 
train_data['Stage_fear'] = train_data['Stage_fear'].map({'No':0, 'Yes':1})
train_data['Drained_after_socializing'] = train_data['Drained_after_socializing'].map({'No':0, 'Yes':1})
train_data['Personality_Extrovert'] = train_data['Personality'].map({'Introvert':0, 'Extrovert':1})


from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt

# Mosaic plot of Stage_fear vs Personality
mosaic(train_data, ['Stage_fear', 'Personality_Extrovert'])

plt.title('Stage Fear vs Personality')
plt.ylabel('Stage Fear')
plt.show()

# # Mosaic plot of Drained_after_socializing vs Personality
mosaic(train_data, ['Drained_after_socializing', 'Personality_Extrovert'])
plt.title('Drained After Socializing vs Personality')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


## Save separately 
data = train_data.copy()
data.drop(['Personality', 'id'], axis=1, inplace=True)

## Train test split
X = data.drop('Personality_Extrovert', axis=1)
y = data['Personality_Extrovert']

## Stratify such as class proportions stay the same
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


from sklearn.linear_model import LogisticRegressionCV

# Build pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegressionCV(
        Cs=10,           # Try 10 inverse regularization strengths
        cv=5,            # 5-fold cross-validation
        class_weight='balanced',  # handle class imbalance
        penalty='l2',    # ridge regularization (default)
        solver='lbfgs',
        scoring='f1',    # optimize for F1
        random_state=42
    ))
])

pipe.fit(X_train, y_train)

## test predictions
y_pred = pipe.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))


from xgboost import XGBClassifier

xgb = XGBClassifier(
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # balance classes
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("XGBoost:")
print(classification_report(y_test, y_pred_xgb))



## hyperparam tuning 
params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=params,
    n_iter=20,
    scoring='f1',
    cv=5,
    verbose=1,
    random_state=42
)
search.fit(X_train, y_train)



print("Best params:", search.best_params_)
print("Best F1 score:", search.best_score_)


## test data
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
X_kaggle = test_df.copy()

# Drop id column (but keep for submission)
ids = X_kaggle['id']

# repeat encoding
X_kaggle['Stage_fear'] = X_kaggle['Stage_fear'].map({'No':0, 'Yes':1})
X_kaggle['Drained_after_socializing'] = X_kaggle['Drained_after_socializing'].map({'No':0, 'Yes':1})

# Drop 'id' column for model input
X_kaggle = X_kaggle.drop('id', axis=1)

## PREDICT USING TRAINED PIPELINE
y_kaggle_pred = pipe.predict(X_kaggle)


## convert back to string factors
y_kaggle_labels = pd.Series(y_kaggle_pred).map({0: 'Introvert', 1: 'Extrovert'})

## create submission
submission = pd.DataFrame({
    'id': ids,
    'Personality': y_kaggle_labels
})

## submit 
submission.to_csv("/kaggle/working/submission.csv", index=False)






