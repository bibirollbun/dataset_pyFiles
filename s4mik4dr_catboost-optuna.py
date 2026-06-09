import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import optuna


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


combined_train_data = pd.concat([train_data, train_extra_data], ignore_index=True)


combined_train_data['Price'] = np.log1p(combined_train_data['Price'])


numeric_features = combined_train_data.select_dtypes(include=['float64', 'int64']).columns.drop('Price')
categorical_features = combined_train_data.select_dtypes(include=['object']).columns


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


X = combined_train_data.drop('Price', axis=1)
y = combined_train_data['Price']

X_preprocessed = preprocessor.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'task_type': 'GPU',
        'verbose': False
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params


catboost_model = CatBoostRegressor(**best_params)
catboost_model.fit(X_train, y_train, verbose=100)


y_pred = catboost_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"MSE (CatBoost): {mse}")


X_test_preprocessed = preprocessor.transform(test_data)


test_predictions = catboost_model.predict(X_test_preprocessed)


test_predictions = np.expm1(test_predictions)


submission = pd.DataFrame({'id': test_data['id'], 'Price': test_predictions})
submission.to_csv("submission.csv", index=False)
print("Файл submission.csv сохранен.")

