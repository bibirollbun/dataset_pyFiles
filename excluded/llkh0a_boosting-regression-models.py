import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("\nTrain Info:")
train.info()
print("\nTest Info:")
test.info()
print("\nTrain Describe:")
train.describe()


plt.figure(figsize=(10, 6))
sns.histplot(train['Calories'], bins=50, kde=True)
plt.title('Distribution of Calories')
plt.xlabel('Calories')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(train['Calories']), bins=50, kde=True)
plt.title('Distribution of Log(Calories + 1)')
plt.xlabel('Log(Calories + 1)')
plt.ylabel('Count')
plt.show()


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']



plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

print("\nSex Distribution:")
print(train['Sex'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=train)
plt.title('Distribution of Sex')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Sex', y='Calories', data=train)
plt.title('Calories by Sex')
plt.show()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Sex_encoded'] = le.fit_transform(train['Sex'])

corr = train[numerical_features + ['Calories', 'Sex_encoded']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Prepare data
X = train[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex_encoded']]
y = train['Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Elastic Net': ElasticNet(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42)
}


# Evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsle = np.sqrt(mean_squared_log_error(y_test, np.maximum(0, y_pred)))
    results[name] = rmsle

# Display results
for name, rmsle in results.items():
    print(f'{name}: RMSLE = {rmsle:.4f}')


best_model = min(results, key=results.get)
print(f'\nBest Model: {best_model} with RMSLE = {results[best_model]:.4f}')


test['Sex_encoded'] = le.transform(test['Sex'])
ids = test['id']
test.drop(columns=['id'])
test = test[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp','Sex_encoded']]
#submission for each model
for name, model in models.items():
    y_pred = np.clip(model.predict(test), a_min=0, a_max=None)
    submission_df = pd.DataFrame({'id': ids, 'Calories': y_pred})
    submission_df.to_csv(f'submission_{name}.csv', index=False)
    print(f'Submission file for {name} created: submission_{name}.csv')



import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np


# Define objective function for XGBoost
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
    }
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsle = np.sqrt(mean_squared_log_error(y_test, np.maximum(0, y_pred)))
    return rmsle


# Define objective function for LightGBM
def objective_lgbm(trial):
# Adjust LightGBM hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        'max_depth': trial.suggest_int('max_depth', -1, 10),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10)
    }
    model = LGBMRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsle = np.sqrt(mean_squared_log_error(y_test, np.maximum(0, y_pred)))
    return rmsle


# Define objective function for CatBoost
def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0)
    }
    model = CatBoostRegressor(**params, verbose=0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsle = np.sqrt(mean_squared_log_error(y_test, np.maximum(0, y_pred)))
    return rmsle


# Run Optuna studies
# study_xgb = optuna.create_study(direction='minimize')
# study_xgb.optimize(objective_xgb, n_trials=20)
# print('Best parameters for XGBoost:', study_xgb.best_params)
# print('Best RMSLE for XGBoost:', study_xgb.best_value)




# study_lgbm = optuna.create_study(direction='minimize')
# study_lgbm.optimize(objective_lgbm, n_trials=20)
# print('Best parameters for LightGBM:', study_lgbm.best_params)
# print('Best RMSLE for LightGBM:', study_lgbm.best_value)




# study_catboost = optuna.create_study(direction='minimize')
# study_catboost.optimize(objective_catboost, n_trials=20)
# print('Best parameters for CatBoost:', study_catboost.best_params)
# print('Best RMSLE for CatBoost:', study_catboost.best_value)


# print('Best parameters for CatBoost:', study_catboost.best_params)
# print('Best parameters for LightGBM:', study_lgbm.best_params)
# print('Best parameters for XGBoost:', study_xgb.best_params)


#using best params from optuna study of the previous version of this notebook
lgbm_best_params     = {'n_estimators': 861, 'learning_rate': 0.07414834307911929, 'num_leaves': 43, 'min_data_in_leaf': 37, 'max_depth': 10, 'feature_fraction': 0.8873877337635245, 'bagging_fraction': 0.7695289271584665, 'bagging_freq': 3}
xgb_best_pamras      = {'n_estimators': 976, 'max_depth': 10, 'learning_rate': 0.019958650721817035, 'subsample': 0.6722659271541026, 'colsample_bytree': 0.849966898586394}
catboost_best_params = {'iterations': 756, 'depth': 9, 'learning_rate': 0.05626275170216383, 'l2_leaf_reg': 0.008347575844872637}


import matplotlib.pyplot as plt

# Train CatBoost
catboost_params = catboost_best_params
catboost_model = CatBoostRegressor(**catboost_params, verbose=100, random_state=42)
catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)



# Train LightGBM
lgbm_params = lgbm_best_params
lgbm_model = LGBMRegressor(**lgbm_params, random_state=42)
lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse')



# Train XGBoost
xgb_params = xgb_best_pamras
xgb_model = XGBRegressor(**xgb_params, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', verbose=True)



# Generate predictions and submission files
models = {"CatBoost": catboost_model, "LightGBM": lgbm_model, "XGBoost": xgb_model}
for name, model in models.items():
    predictions = np.clip(model.predict(test), a_min=0, a_max=None)
    submission = pd.DataFrame({'id': ids, 'Calories': predictions})
    submission.to_csv(f'submission_{name} with params tuned.csv', index=False)
    print(f'Submission file for {name} created: submission_{name}.csv')

