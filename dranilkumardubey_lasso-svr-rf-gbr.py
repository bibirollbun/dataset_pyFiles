import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train


train['Sex'] = train['Sex'].replace({'male': 1, 'female': 0})
train['Sex']


train


test


test['Sex'] = test['Sex'].replace({'male': 1.0, 'female': 0.0})
test['Sex']


test


train.info()


test.info()


X = train.drop(columns=["id", "Calories"])
y = train["Calories"]
X_test = test.drop(columns=["id"])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=36)


models = {
    'lasso': (Lasso(), {'alpha': [0.001, 0.01, 0.1, 1, 10]}),
    'svr': (SVR(), {'C': [1, 10], 'epsilon': [0.1, 0.2], 'kernel': ['rbf']}),
    'rf': (RandomForestRegressor(random_state=36), {'n_estimators': [100], 'max_depth': [10, 20]}),
    'gbr': (GradientBoostingRegressor(random_state=36), {'n_estimators': [100], 'learning_rate': [0.05, 0.1]})
}


best_models = {}
for name, (model, params) in models.items():
    grid = GridSearchCV(model, param_grid=params, scoring='r2', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"{name} best R2 on CV: {grid.best_score_:.4f}")


hybrid = VotingRegressor(estimators=[
    ('lasso', best_models['lasso']),
    ('svr', best_models['svr']),
    ('rf', best_models['rf']),
    ('gbr', best_models['gbr'])
])
hybrid.fit(X_train, y_train)


y_pred = hybrid.predict(X_val)
print("Hybrid R2 Score:", r2_score(y_val, y_pred))
print("Hybrid RMSE:", mean_squared_error(y_val, y_pred, squared=False))


final_preds = hybrid.predict(X_test_scaled)
submission["Calories"] = final_preds
submission.to_csv("submission.csv", index=False)




