# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV


train = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv")


train.shape


test.shape


train.info()



# we need to handle non values
train.isnull().sum()


train = train.drop(columns=['id','Name'])
test = test.drop(columns=['Name'])


numeric_columns = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
                   'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours',
                   'Financial Stress', 'Depression']


categorical_columns = ['Gender', 'City', 'Working Professional or Student',
                      'Profession', 'Sleep Duration', 'Dietary Habits',
                      'Degree', 'Have you ever had suicidal thoughts ?',
                      'Family History of Mental Illness']

for feature in categorical_columns:
  unique = train[feature].unique()
  print(f"column: {feature}")
  print(f"Unique Values: {unique}\n")


#cleaning the data
sleepmode = train['Sleep Duration'].mode()[0]
dietmode = train['Dietary Habits'].mode()[0]

train = train.replace({'Sleep Duration': {'Indore': sleepmode ,'Work_Study_Hours': sleepmode , 'Unhealthy': sleepmode,
"Pune": sleepmode, "Sleep_Duration": sleepmode, "than 5 hours": sleepmode, 'Moderate': sleepmode, 'No':sleepmode,
"40-45 hours": "4-5 hours", "55-66 hours": "5-6 hours", "35-36 hours": "4-6 hours", "45-48 hours":sleepmode,
"45": sleepmode, '49 hours': sleepmode}})

train = train.replace({'Dietary Habits':{
    'Yes': "Healthy", '1.0': "Healthy", 'Pratham': dietmode,  'BSc': dietmode, 'Gender': dietmode,
    '3': dietmode, 'M.Tech': dietmode, 'Mihir': dietmode,'Vegas': dietmode,'Male': dietmode,
    'Indoor': dietmode,'Male': dietmode,'Class 12': dietmode, '2': dietmode,  'Less Healthy' :"Unhealthy",
    "Hormonal":dietmode, 'Electrician':dietmode, 'No': 'Unhealthy','No Healthy':'Unhealthy', "More Healthy": "Healthy",
    'Less than Healthy': "Unhealthy"
}})


test = test.replace({'Sleep Duration': {'Indore': sleepmode ,'Work_Study_Hours': sleepmode , 'Unhealthy': sleepmode,
"Pune": sleepmode, "Sleep_Duration": sleepmode, "than 5 hours": sleepmode, 'Moderate': sleepmode, 'No':sleepmode,
"40-45 hours": "4-5 hours", "55-66 hours": "5-6 hours", "35-36 hours": "4-6 hours", "45-48 hours":sleepmode,
"45": sleepmode , '49 hours': sleepmode}})

test = test.replace({'Dietary Habits':{
    'Yes': "Healthy", '1.0': "Healthy", 'Pratham': dietmode,  'BSc': dietmode, 'Gender': dietmode,
    '3': dietmode, 'M.Tech': dietmode, 'Mihir': dietmode,'Vegas': dietmode,'Male': dietmode,
    'Indoor': dietmode,'Male': dietmode,'Class 12': dietmode, '2': dietmode,  'Less Healthy' :"Unhealthy",
    "Hormonal":dietmode, 'Electrician':dietmode, 'No': 'Unhealthy','No Healthy':'Unhealthy', "More Healthy": "Healthy",
    'Less than Healthy': "Unhealthy"
}})

print(train['Dietary Habits'].unique(),
train['Sleep Duration'].unique())


#finding missing values or typos in numerical values
for feature in numeric_columns:
  unique = train[feature].unique()
  print(f"column: {feature}")
  print(f"Unique Values: {unique}\n")


# handling null values using mean for numeric datas and mode for numerical datas
train[numeric_columns] = train[numeric_columns].fillna(train[numeric_columns].mean())
test[[col for col in numeric_columns if col != 'Depression']] = test[[col for col in numeric_columns if col != 'Depression']].fillna(test[[col for col in numeric_columns if col != 'Depression']].mean())

for col in categorical_columns:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])


print(train.isnull().sum())
print(train.head())


# train test split
X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=['Depression']), train['Depression'], test_size=0.25, random_state=42 )
print(X_train.shape, X_test.shape)


# normalization and one hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [col for col in numeric_columns if col != 'Depression']),
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
,  remainder='drop')

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
test = preprocessor.transform(test)


import xgboost as xgb


xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic', 
    n_estimators=200,            
    learning_rate=0.1,
    max_depth=8,
    use_label_encoder=False,      
    eval_metric='logloss',       
    random_state=42
)



xgb_classifier.fit(X_train, y_train)

y_pred = xgb_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



from sklearn.model_selection import GridSearchCV
from collections import Counter

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2], 
    'max_depth': [5, 8, 10, 13], 
    'scale_pos_weight': [1, round(Counter(y_train)[0] / Counter(y_train)[1], 2)]
}

bestXGB = GridSearchCV(
    estimator=xgb_classifier,
    param_grid=param_grid,
    cv=5, 
    scoring='roc_auc', 
    verbose=1,
    n_jobs=-1
)


bestXGB.fit(X_train, y_train)
print(f"\nBest parameters found: {bestXGB.best_params_}")
print(f"Best cross-validation score ({bestXGB.scoring}): {bestXGB.best_score_:.4f}")


best_xgb_model = bestXGB.best_estimator_

y_pred_best = best_xgb_model.predict(X_test)
y_pred_proba_best = best_xgb_model.predict_proba(X_test)[:, 1]

accuracy_best = accuracy_score(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_pred_proba_best) 

print(f"Test Set Accuracy: {accuracy_best:.4f}")
print(f"Test Set ROC AUC: {roc_auc_best:.4f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred_best))




# Plot a learning curve
train_sizes, train_scores, test_scores = learning_curve(best_xgb_model, X_train, y_train, cv=5, n_jobs=-1,scoring='roc_auc', train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.title("Learning Curve (ROC AUC Score)")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid()
plt.show()


sample_submission = pd.read_csv("/kaggle/input/playground-series-s4e11/sample_submission.csv")
y_pred = best_xgb_model.predict(test)
submission_df = pd.DataFrame({'id': sample_submission['id'], 'Depression': y_pred})
submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

