import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from scipy.stats import randint
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/molecular-machine-learning/train.csv')
test = pd.read_csv('/kaggle/input/molecular-machine-learning/test.csv')

y = train['T80']

drop_cols = ['Batch_ID', 'T80', 'Smiles']
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=['Batch_ID', 'Smiles'])

X.columns = X.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_test = X_test[X.columns]

X.fillna(X.mean(), inplace=True)
X_test.fillna(X.mean(), inplace=True)  


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

