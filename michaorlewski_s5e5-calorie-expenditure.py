import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
train_df


train_df.info()


assert train_df.isna().sum().sum() + test_df.isna().sum().sum() == 0


train_df['BMI'] = train_df['Weight'] / ((train_df['Height']/100)**2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height']/100) **2)


bmr_men = lambda w,h,a: 88.362 + (13.397*w) + (4.799*h) - (5.677*a)
bmr_women = lambda w,h,a: 447.593 + (9.247*w) + (3.098*h) - (4.330*a)

train_df['BMR'] = train_df.apply(lambda r: bmr_men(r['Weight'], r['Height'], r['Age'])
                                 if r['Sex'] == 'male'
                                 else bmr_women(r['Weight'], r['Height'], r['Age']),
                                axis=1)
test_df['BMR'] = test_df.apply(lambda r: bmr_men(r['Weight'], r['Height'], r['Age'])
                                 if r['Sex'] == 'male'
                                 else bmr_women(r['Weight'], r['Height'], r['Age']),
                                axis=1)


def calories_burned(row):
    h = row['Heart_Rate']
    w = row['Weight']
    a = row['Age']
    d = row['Duration']
    
    if row['Sex'] == 'male':
        cpm = (-55.0969 + (0.6309 * h) + (0.1988 * w) + (0.2017 * a)) / 4.184
    else:
        cpm = (-20.4022 + (0.4472 * h) - (0.1263 * w) + (0.074 * a)) / 4.184
        
    return cpm * d

train_df['Est_Calories'] = train_df.apply(calories_burned, axis=1)
test_df['Est_Calories'] = test_df.apply(calories_burned, axis=1)


train_num_cols = train_df.select_dtypes(exclude=['object']).columns.tolist()
train_num_cols.remove('Calories')

plt.figure(figsize=(12, 8)).suptitle('Numerical features histograms (training data)')
for i, col in enumerate(train_num_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(train_df[col], bins=30)

plt.tight_layout()
plt.show()


test_num_cols = test_df.select_dtypes(exclude=['object']).columns.tolist()

plt.figure(figsize=(12, 8)).suptitle('Numerical features histograms (test data)')
for i, col in enumerate(train_num_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(test_df[col], bins=30)

plt.tight_layout()
plt.show()


plt.title('Calorie Expenditure histogram')
sns.histplot(train_df['Calories'], bins=30)


plt.figure(figsize=(8, 6))
sns.heatmap(train_df.drop(columns=['Sex']).corr(), annot=True)


sns.boxplot(x=train_df['Sex'], y=train_df['Calories'])


X = train_df.drop('Calories', axis=1)
y = train_df['Calories']

X_test = test_df


num_cols = X.select_dtypes(exclude=['object']).columns

scaler = StandardScaler()
scaler.fit(X[num_cols])

X[num_cols] = scaler.transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

X


X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X['Sex_male'] = X['Sex_male'].astype(int)
X_test['Sex_male'] = X_test['Sex_male'].astype(int)

X


def objective(trial):
    xgb = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 1000),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3),
        max_depth=trial.suggest_int('max_depth', 3, 12),
        gamma=trial.suggest_float('gamma', 0, 5),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
        reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
        n_jobs=-1,
    )

    # clip negative predictions to zero otherwise, the mean squared log error cannot be computed
    model = TransformedTargetRegressor(regressor=xgb,
        func=lambda x: x, inverse_func=lambda x: np.maximum(0, x))

    cv = KFold(n_splits=4)
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_log_error').mean()
    return -1 * score

study = optuna.create_study(direction='minimize', study_name='XGBRegressor')
# study.optimize(objective, n_trials=20) # uncomment to optimize parameters


best_params = {'n_estimators': 290,
               'learning_rate': 0.09035682136488198,
               'max_depth': 11,
               'gamma': 1.9619844689205799,
               'min_child_weight': 3,
               'subsample': 0.800425708343804,
               'colsample_bytree': 0.8970988741636009,
               'reg_alpha': 4.082772550667748,
               'reg_lambda': 2.2420793860740686}

xgb = XGBRegressor(**best_params, n_jobs=-1)
model_xgb = TransformedTargetRegressor(regressor=xgb, func=lambda x: x, inverse_func=lambda x: np.maximum(0, x))
scores = cross_validate(model_xgb, X, y, scoring='neg_mean_squared_log_error', return_train_score=True)

train_score = -scores['train_score'].mean()
test_score = -scores['test_score'].mean()

print('XGBRegressor')
print(f'Train MSLE = {train_score:.6f}')
print(f'Test MSLE = {test_score:.6f}')


def objective(trial):
    lgbm = LGBMRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 500),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3),
        num_leaves=trial.suggest_int('num_leaves', 31, 4096),
        max_depth=trial.suggest_int('max_depth', 3, 12),
        min_split_gain=trial.suggest_float('min_split_gain', 0, 5),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
        reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
        n_jobs=-1,
        verbose=-1,
    )

    # clip negative predictions to zero otherwise, the mean squared log error cannot be computed
    model = TransformedTargetRegressor(regressor=lgbm,
        func=lambda x: x, inverse_func=lambda x: np.maximum(0, x))

    cv = KFold(n_splits=4)
    score = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_log_error').mean()
    return -1 * score

study = optuna.create_study(direction='minimize', study_name='LGBMRegressor')
# study.optimize(objective, n_trials=20) # uncomment to optimize parameters


best_params = {'n_estimators': 352,
               'learning_rate': 0.05002905043307179,
               'num_leaves': 2856,
               'max_depth': 11,
               'min_split_gain': 3.6686668245573046,
               'min_child_weight': 7,
               'subsample': 0.8055588188891772,
               'colsample_bytree': 0.9718147641239651,
               'reg_alpha': 3.1072119236989515,
               'reg_lambda': 0.8997034093254324}

lgbm = LGBMRegressor(**best_params, n_jobs=-1, verbose=-1)
model_lgbm = TransformedTargetRegressor(regressor=lgbm, func=lambda x: x, inverse_func=lambda x: np.maximum(0, x))
scores = cross_validate(model_lgbm, X, y, scoring='neg_mean_squared_log_error', return_train_score=True)

train_score = -scores['train_score'].mean()
test_score = -scores['test_score'].mean()

print('LGBMRegressor')
print(f'Train MSLE = {train_score:.6f}')
print(f'Test MSLE = {test_score:.6f}')


# model_xgb.fit(X, y)
# preds_xgb = model_xgb.predict(X_test)

# model_lgbm.fit(X, y)
# preds_lgbm = model_lgbm.predict(X_test)

# model_preds = (preds_xgb + preds_lgbm) / 2


# submission = pd.DataFrame({'id': X_test.index, 'Calories': model_preds})
# submission.to_csv('/kaggle/working/submission_model.csv', index=False)


estimators = [('XGB', model_xgb),
              ('LGBM', model_lgbm)]

from sklearn.linear_model import Ridge

ridge = Ridge()
model_ridge = TransformedTargetRegressor(regressor=ridge, func=lambda x: x, inverse_func=lambda x: np.maximum(0, x))
final_estimator = model_ridge

model_stacked = StackingRegressor(estimators, final_estimator)
model_stacked.fit(X, y)
stacked_preds = model_stacked.predict(X_test)


submission = pd.DataFrame({'id': X_test.index, 'Calories': stacked_preds})
submission.to_csv('/kaggle/working/submission_stacked.csv', index=False)

