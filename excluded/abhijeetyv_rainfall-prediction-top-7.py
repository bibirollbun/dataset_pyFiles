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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier




train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

print(f'Train Shape: {train.shape}')
print(f'Test Shape: {test.shape}')

print(train.head())

print(test.head())





print(train.info())
print(train.describe())





missing = train.isnull().sum()
print('Missing Values:')
print(missing[missing > 0])





sns.countplot(x='rainfall', data=train)
plt.title('Target Distribution')
plt.show()





plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()





features = [col for col in train.columns if col not in ['id', 'rainfall']]
X = train[features]
y = train['rainfall']





imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)





scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=features)




# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)





rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
val_preds = rf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f'Baseline RandomForest AUC: {roc_auc:.4f}')







# Baseline Model with Hyperparameter Tuning
rf = RandomForestClassifier(n_estimators= 260, max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42)
rf.fit(X_train, y_train)
val_preds = rf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f'RandomForest AUC with Hyperparameters: {roc_auc:.4f}')


import matplotlib.pyplot as plt
import seaborn as sns

# Assume 'rf' is your trained Random Forest model
feature_importances = rf.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': features,  # 'features' should be the list of your input features
    'Importance': feature_importances
})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Random Forest Feature Importance')
plt.show()



from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features (degree=2) for all features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)

# Train model with polynomial features
model.fit(X_train_poly, y_train)






X_val_poly = poly.fit_transform(X_val)
val_preds = model.predict_proba(X_val_poly)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f'poly AUC with Hyperparameters: {roc_auc:.4f}')


#logistic regression

clf = LogisticRegression(random_state=0)

clf.fit(X_train, y_train)
val_preds = clf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f'Logistic regression AUC: {roc_auc:.4f}')





# 6. Stacking Model

rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42)
xgb = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, random_state=42)
stack_model = StackingClassifier(estimators=[('rf', rf), ('xgb', xgb)], final_estimator=LogisticRegression(), cv=5)

stack_model.fit(X_train, y_train)

val_preds = stack_model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f'Stacking Model AUC: {roc_auc:.4f}')





xgb = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)

val_preds = xgb.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f'XGB Model AUC: {roc_auc:.4f}')


# from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# def grid_search_xgb(X_train, y_train):
#     # Create an XGBoost model
#     model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    
#     # Define the parameter grid to search
#     param_grid = {
#         'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Tune learning rate
#         'max_depth': [3 , 6, 10],                   # Tune max depth
#         'n_estimators': [50, 100, 200],            # Tune number of estimators
#         'subsample': [0.8, 0.9, 1.0],              # Tune subsample ratio
#         'colsample_bytree': [0.8, 0.9, 1.0],       # Tune column subsample ratio
#         'gamma': [0, 0.1, 0.2],                    # Tune gamma
#     }
    
#     # Initialize GridSearchCV
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
#                                scoring='accuracy', n_jobs=-1, cv=2, verbose= 2)
    
#     # Fit the grid search
#     grid_search.fit(X_train, y_train)
    
#     # Get the best parameters and best score
#     best_params = grid_search.best_params_
#     best_score = grid_search.best_score_
    
#     print(f"Best Parameters: {best_params}")
#     print(f"Best Score: {best_score:.4f}")
    
#     # Return the best model
#     return grid_search.best_estimator_


# grid_search_xgb(X_train, y_train)


xgb = XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=0.9, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=0, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=3,
              max_leaves=None, min_child_weight=None,
              monotone_constraints=None, multi_strategy=None, n_estimators=50,
              n_jobs=None, num_parallel_tree=None, random_state=None)

xgb.fit(X_train, y_train)

val_preds = xgb.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f'XGB Model AUC: {roc_auc:.4f}')


# rf.fit(X, y)
rf.fit(X_train, y_train)

val_preds = rf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f'RF Model AUC: {roc_auc:.4f}')




test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

#save id 
test_ids = test['id'].copy()

# Drop 'id' column and apply transformations (imputation and scaling)
test = test.drop('id', axis=1)
test = pd.DataFrame(imputer.transform(test[features]), columns=features)
test = pd.DataFrame(scaler.transform(test), columns=features)

# Make predictions with the model

test_preds = model.predict_proba(poly.fit_transform(test))[:, 1]
# test_preds = model.predict_proba(test)[:, 1]

# Create the submission DataFrame with 'id' and 'rainfall'
submission = pd.DataFrame({'id': test_ids, 'rainfall': test_preds})

# Write the submission file
submission.to_csv('submission.csv', index=False)

# Print and check the first few rows
print(submission.head())
print('done')










