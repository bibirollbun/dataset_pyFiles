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


cat_df = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')
cat_df.shape


quantdf = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx')
quantdf.shape


import matplotlib.pyplot as plt
quantdf.hist(figsize=(15, 10), bins=20)
plt.suptitle('Histograms of Quantitative Data')
plt.show()


cat_df.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of all columns in cat_df')
plt.show()


connectome_df = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
connectome_df.shape


sol_df = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')
y_adhd = sol_df['ADHD_Outcome']
y_female = sol_df['Sex_F']


feats = sol_df.merge(cat_df, on = 'participant_id', how = 'left')
feats = feats.merge(quantdf, on = 'participant_id', how = 'left')
feats = feats.merge(connectome_df, on = 'participant_id', how = 'left')
feats.shape


feats_imputed = feats.drop(columns=['participant_id','ADHD_Outcome','Sex_F'])
feats_imputed = feats_imputed.fillna(feats_imputed.mean())


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# For warnings
import warnings
warnings.filterwarnings('ignore')


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feats_imputed, y_adhd, test_size=0.2, random_state=42)

# Define the classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Dictionary to store the ROC AUC scores
roc_auc_scores = {}

# create a adhdpred_dfdataframe to store the predictions
adhdpred_df = pd.DataFrame()
adhdpred_df['participant_id'] = feats['participant_id']

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    roc_auc_scores[name] = roc_auc
    print(f'{name}: ROC AUC = {roc_auc:.4f}')

# Find the best scoring model
best_model_name = max(roc_auc_scores, key=roc_auc_scores.get)
best_model_score = roc_auc_scores[best_model_name]
print(f'\nBest Model: {best_model_name} with ROC AUC = {best_model_score:.4f}')


test_cat_df = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
test_quantdf = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')
test_connectome_df = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')


# Merge test dataframes
test_feats = test_cat_df.merge(test_quantdf, on='participant_id', how='left')
test_feats = test_feats.merge(test_connectome_df, on='participant_id', how='left')  
test_feats_imputed = test_feats.drop(columns=['participant_id'])
test_feats_imputed = test_feats_imputed.fillna(test_feats_imputed.mean())
test_feats_imputed.shape


# use the above svc model to predict the test set, outoput is 0 or 1
y_test_pred = classifiers['Support Vector Machine'].predict(test_feats_imputed)
y_test_pred.shape
# Save the predictions to a CSV file
submission_df = pd.DataFrame({
    'participant_id': test_cat_df['participant_id'],
    'ADHD_Outcome': y_test_pred
})
submission_df.head()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feats_imputed, y_female, test_size=0.2, random_state=42)

# Define the classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Dictionary to store the ROC AUC scores
roc_auc_scores = {}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    roc_auc_scores[name] = roc_auc
    print(f'{name}: ROC AUC = {roc_auc:.4f}')

# Find the best scoring model
best_model_name = max(roc_auc_scores, key=roc_auc_scores.get)
best_model_score = roc_auc_scores[best_model_name]
print(f'\nBest Model: {best_model_name} with ROC AUC = {best_model_score:.4f}')


# use the above svc model to predict the test set
y_fem_pred = classifiers['Logistic Regression'].predict(test_feats_imputed)
y_fem_pred.shape
# Save the predictions to a CSV file
submission_df = pd.DataFrame({
    'participant_id': test_cat_df['participant_id'],
    'ADHD_Outcome': y_test_pred,
    'sex_F': y_fem_pred
})
submission_df.to_csv('submission.csv', index=False)
submission_df.head()


