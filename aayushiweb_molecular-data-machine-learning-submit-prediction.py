# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_csv('/kaggle/input/molecular-machine-learning/train.csv')
test=pd.read_csv('/kaggle/input/molecular-machine-learning/test.csv')


train.head()


print(train.describe())
print(train.isnull().sum())
train['T80'].hist(bins=30)


priority_features = ['TDOS4.0', 'NumHeteroatoms', 'Mass', 
                    'HOMO', 'LUMO', 'PrimeExcite(eV)']


y = train['T80']

drop_cols = ['Batch_ID', 'T80', 'Smiles']
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=['Batch_ID', 'Smiles'])

X.columns = X.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_test = X_test[X.columns]

X.fillna(X.mean(), inplace=True)
X_test.fillna(X.mean(), inplace=True)  


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from scipy.stats import randint


lasso = LassoCV(cv=5, random_state=42)
feature_selector = SelectFromModel(lasso)



pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('select', feature_selector),
    ('model', RandomForestRegressor(random_state=42))
])

param_distributions = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(5, 50),
    'model__min_samples_split': randint(2, 10),
    'model__min_samples_leaf': randint(1, 10)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)



search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X, np.log1p(y))


print(f"\nBest CV RMSE: {-search.best_score_:.4f}")
print("Best hyperparameters:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")

best_model = search.best_estimator_
preds = np.expm1(best_model.predict(X_test))

selected_features = X.columns[best_model.named_steps['select'].get_support()]
print("\nSelected features:")
print(list(selected_features))

submission = pd.DataFrame({
    'Batch_ID': test['Batch_ID'],
    'T80': preds
})
submission.to_csv('submission.csv', index=False)


print(submission)





