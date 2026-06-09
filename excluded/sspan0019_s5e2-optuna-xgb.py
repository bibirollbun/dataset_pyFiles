import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')

train = pd.concat([train, train_extra], axis=0, ignore_index=True)


print(train.shape, test.shape)



print(train.dtypes)


CAT_COLS = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
NUM_COLS = ['Compartments', 'Weight Capacity (kg)', 'Price']


print('TRAIN')

for col in CAT_COLS:
    unique_values = train[col].dropna().unique()
    print(f'{col}: {sorted(unique_values)}')

print()

print('TEST')

for col in CAT_COLS:
    unique_values = test[col].dropna().unique()
    print(f'{col}: {sorted(unique_values)}')


print('TRAIN')

for col in CAT_COLS:
    print(f'{col}: {train[col].isnull().mean():.4f}%')

print()

print('TEST')

for col in CAT_COLS:
    print(f'{col}: {test[col].isnull().mean():.4f}%')


print(train[NUM_COLS].describe())


print('TRAIN')

for col in NUM_COLS:
    print(f'{col}: {train[col].isnull().mean():.4f}%')

print()

print('TEST')

for col in NUM_COLS:

    if col == 'Price':
        continue

    print(f'{col}: {test[col].isnull().mean():.4f}%')


for col in NUM_COLS:
    train[col].plot(kind='hist', title=col)
    plt.show()


for col in NUM_COLS:
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    print(f'{col}: {train[(train[col] < lower_bound) | (train[col] > upper_bound)].shape[0]} outliers')


train = train.dropna()
test = test.dropna()


NOMINAL_CAT_COLS = ['Brand', 'Material', 'Style', 'Color']

train = pd.get_dummies(train, columns=NOMINAL_CAT_COLS)
test = pd.get_dummies(test, columns=NOMINAL_CAT_COLS)


train['Size'] = train['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3})
test['Size']  = test['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3})

train['Laptop Compartment'] = train['Laptop Compartment'].map({'No': 0, 'Yes': 1})
test['Laptop Compartment']  = test['Laptop Compartment'].map({'No': 0, 'Yes': 1})

train['Waterproof'] = train['Waterproof'].map({'No': 0, 'Yes': 1})
test['Waterproof']  = test['Waterproof'].map({'No': 0, 'Yes': 1})


NUM_COLS.remove('Price')

scaler = MinMaxScaler()
scaler.fit(train[NUM_COLS])

train[NUM_COLS] = scaler.transform(train[NUM_COLS])
test[NUM_COLS] = scaler.transform(test[NUM_COLS])


train.shape


train.head()


target       = train['Price']
predictors   = train.drop(columns=['Price'])
train_predictors, val_predictors, train_target, val_target = train_test_split(predictors, target, train_size=0.8, random_state=42)


def objective(trial):
    params = {        
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True), 
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.),
        "gamma": trial.suggest_float("gamma", 0.01, 1.0),
        "verbosity": 0, 
        "device": 'gpu', 
        "tree_method": 'gpu_hist'
    } 

    xgb_model = xgb.XGBRegressor(**params)
    score = cross_val_score(xgb_model, train_predictors, train_target, n_jobs=-1, cv=3, scoring='neg_mean_squared_error').mean()
    rmse = np.sqrt(-score)
    return rmse

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(), study_name='XGB Regression')
optuna.logging.set_verbosity(optuna.logging.DEBUG)
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_score = study.best_value

print("Best parameters:", best_params)
print("Best score:", best_score)


initial_model = xgb.XGBRegressor(**best_params)
initial_model.fit(train_predictors, train_target)


val_predictions = initial_model.predict(val_predictors)
val_rmse = np.sqrt(mean_squared_error(val_target, val_predictions))
print(f'Validation RMSE: {val_rmse}')


final_model = xgb.XGBRegressor(**best_params)
final_model.fit(predictors, target)


test_predictions = final_model.predict(test)
test['Price'] = test_predictions
test[['Price']].to_csv('s5e2-submission.csv', index=True)

