import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import pylab
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
#train['price_log'] = np.log1p(train['Price'])
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train1 = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.dropna(inplace=True)
train1.dropna(inplace=True)
train1


#for col in test:
#    if test[col].dtype == 'object':
#        if test[col].isnull().any():
#            print(test[col])
#        test[col] = test[col].fillna('not listed')
        
#    if test[col].dtype == 'int' or test[col].dtype == 'float':
#        test[col] = test[col].fillna(-1)




X = train.drop(columns=['id', 'Price'])
y = train['Price']
X_test = test.drop('id', axis=1)

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('not listed')
        X_test[col] = X_test[col].fillna('not listed')
    else:
        X[col] = X[col].fillna(-1)
        X_test[col] = X_test[col].fillna(-1)


full = pd.concat([X, X_test], axis=0)

# One-hot encode
full = pd.get_dummies(full)

# Split back into train/test
X_encoded = full.iloc[:len(X), :]
X_test_encoded = full.iloc[len(X):, :]
        


#y = train1.pop('Price')
#X = train1
#X_test = test


X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape


param_dist = {
    'n_estimators': [200, 300, 500, 700],
    'max_depth': [15, 25, 35, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the base RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Setup RandomizedSearchCV with 20 iterations and 3-fold cross-validation
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Run the hyperparameter search
random_search.fit(X_train, y_train)

# Print the best hyperparameters and the best RMSE score from cross-validation
print("Best params:", random_search.best_params_)
print("Best CV RMSE:", -random_search.best_score_)

# Use the best estimator found to predict on the validation set
best_model = random_search.best_estimator_
val_preds = best_model.predict(X_val)

# Calculate RMSE on validation set
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("Validation RMSE with best model:", rmse)


#preds_log = model.predict(X_test_encoded)
y_pred = best_model.predict(X_val)
stats.probplot(y_pred, dist="norm", plot=pylab)
pylab.show()


#train['price_log'] = np.log1p(train['Price'])
test_preds = best_model.predict(X_test_encoded)
#preds_log = model.predict(X_test_encoded)
submission = pd.DataFrame()
submission['id'] = test['id']
#submission['price'] = np.expm1(preds_log)
submission['price'] = test_preds
submission.to_csv('submission_rf.csv', index=False)
submission




