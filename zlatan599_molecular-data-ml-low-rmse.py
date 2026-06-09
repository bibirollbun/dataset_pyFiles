!pip install rdkit-pypi

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import randint, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error

train_set = pd.read_csv('/kaggle/input/molecular-machine-learning/train.csv')
test_set = pd.read_csv('/kaggle/input/molecular-machine-learning/test.csv')


train_set.head()


train_set.info()


train_set.describe()


print("ALL GREAT")


y = train_set['T80']

drop_cols = ['Batch_ID', 'T80', 'Smiles']
X = train_set.drop(columns=drop_cols)
X_test = test_set.drop(columns=['Batch_ID', 'Smiles'])

X.columns = X.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_test = X_test[X.columns]

X.fillna(X.mean(), inplace=True)
X_test.fillna(X.mean(), inplace=True) 

categorical_cols = X.select_dtypes(include='object').columns
numeric_cols = X.select_dtypes(include=np.number).columns

from scipy.stats import skew
skewed_feat = X[numeric_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_features = skewed_feat[skewed_feat > 0.8].index.tolist()
print("High skew features:", skewed_features)
skewed_features = [col for col in skewed_features if col in X.columns]
skewed_features = [col for col in skewed_features if (X[col] > 0).all()]

numeric_cols = [col for col in numeric_cols if col not in skewed_features]


log_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
    StandardScaler(),
)

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = ColumnTransformer([
    ("log", log_transformer, skewed_features),
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])


alphas = [0.1, 0.01, 0.001, 0.0001]
lasso = LassoCV(cv=5, alphas = alphas, max_iter = 8000, random_state=42) #my name ;)
feature_selector = SelectFromModel(lasso)

pipeline = Pipeline([
    ('scale', preprocessor),
    ('select', feature_selector),
    ('model', RandomForestRegressor(random_state=42)) #my name ;)
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
    n_iter=100,
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
    'Batch_ID': test_set['Batch_ID'],
    'T80': preds
})
submission.to_csv('submission_final2.csv', index=False)

