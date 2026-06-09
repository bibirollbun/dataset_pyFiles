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


# import Libs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import lightgbm as lgb


# import data

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train = train.drop(columns=['id'])
train


train.columns



train.isna().sum()


# Dimention of dataset
print(f'Data dimention for train set is :{train.shape}')
print(f'Data dimention for test set is :{test.shape}')



train.dtypes


# target varaible distribution
import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize= (8, 8))
sns.histplot(train['accident_risk'], kde=True, bins=50)
plt.title('Distribution of Accident Risk')
plt.xlabel('Accident Risk')
plt.show()



# Correlation heatmap

plt.figure(figsize = (8,8))

num_cols = train.select_dtypes(include=[np.number]).columns
corr_matrix = train[num_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Define features and target

X = train.drop(['accident_risk'], axis=1)
y = train['accident_risk']



test


# Identify categorical features

cate_features = X.select_dtypes(exclude=[np.number]).columns.tolist()



# Split the data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)



print(X_train.shape)
X_val.shape


# models

catboost_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=cate_features,
    random_seed=42,
    verbose=200,
    early_stopping_rounds=50,
    loss_function='RMSE'    
)


light_model = lgb.LGBMRegressor(
    learning_rate=0.1,
    max_depth=-5,
    random_state=42
    
)


catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=200)


# prediction on validation set

y_pred = catboost_model.predict(X_val)


# Calculate RMSE

val_rmse = mean_squared_error(y_val, y_pred, squared=False)
val_rmse


# Cross-validation

cv_scores = cross_val_score(catboost_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(cv_scores)


cv_rmse.mean()



cv_rmse.std()


# feature importance

feature_importance = catboost_model.get_feature_importance()


feature_names = X.columns


importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
})



importance_df = importance_df.sort_values('importance', ascending=False)


plt.figure()
sns.barplot(data=importance_df, x='importance', y='feature')


plt.scatter(y_val, y_pred, alpha=0.1, marker='.', s=2)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')


# Submission

test_feature = test.drop('id', axis=1)


test_predictions = catboost_model.predict(test_feature)


# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': test_predictions
})

# Save submission file
submission.to_csv('submission.csv', index=False)

print(f"Submission shape: {submission.shape}")
print("\nFirst 5 rows of submission:")
display(submission.head())


# Save your trained model for Stack Overflow Code Challenge #10 
# https://stackoverflow.com/beta/challenges/79780240/challenge-10-road-safety-challenge-joint-with-kaggle
catboost_model.save_model('accident_risk_model.cbm')

# Verify
import os
print(f"Model file size: {os.path.getsize('accident_risk_model.cbm') / (1024*1024):.2f} MB")




