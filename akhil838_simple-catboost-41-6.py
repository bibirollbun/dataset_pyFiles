import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import sys
sys.path.append('/kaggle/input/russian-car-plates-prices-prediction')
from supplemental_english import REGION_CODES, GOVERNMENT_CODES
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
import holidays
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor 

def smape(A, F):
    A = np.array(A)
    F = np.array(F)
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))



train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv').drop(columns =['id'],axis =1 )
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv').drop(columns =['id'],axis =1 )


def extract_plate(df):
    df['plate_series1'] = df['plate'].apply(lambda x: x[0])
    df['plate_series2'] = df['plate'].apply(lambda x: x[4:6])
    df['plate_series'] = df['plate_series1'] + df['plate_series2']
    df['plate_regst_code'] = df['plate'].apply(lambda x: x[1:4]).astype(int)
    df['plate_region_code'] = df['plate'].apply(lambda x: x[6:])

    #df = df.drop(columns=['plate','plate_series1','plate_series2'],axis=1)

    return df

train = extract_plate(train)
test = extract_plate(test)


REGION_CODES2 = {}
for region, codes in REGION_CODES.items():
    for c in codes:
        # if c in REGION_CODES2:
        #     print(c)
        REGION_CODES2[c] = region
train['region'] = train['plate_region_code'].apply(lambda x: REGION_CODES2[x])
test['region'] = test['plate_region_code'].apply(lambda x: REGION_CODES2[x])
#DUPLICATED REGION CODE 81 85 84


GOVERNMENT_CODES2 = defaultdict(lambda: defaultdict(dict))
for key, value in GOVERNMENT_CODES.items():
    region, r, code = key
    discription, forbidden, advantage, significance = value
    GOVERNMENT_CODES2[region][range(r[0],r[1]+1)][code] = [discription, forbidden, advantage, significance]


def add_prorities(df):
    prorities = []
    def govt_vehicles(row):
        series = row['plate_series']
        register_code = row['plate_regst_code']
        region_code = row['plate_region_code']

        discription, forbidden, advantage, significance,govt = 'No Description', 0, 0, 0, 0
        if series in GOVERNMENT_CODES2:
            codes = GOVERNMENT_CODES2[series]
            for r in codes:
                if register_code in r:
                    numbers = codes[r]
                    if region_code in numbers:
                        values = numbers[region_code]
                        discription, forbidden, advantage, significance = values
                        govt = 1
                        #print(series, register_code, region_code,values)
        prorities.append([forbidden, advantage, significance,govt,discription])

    df[['plate_series', 'plate_regst_code', 'plate_region_code']].apply(govt_vehicles, axis=1)
    res_df = pd.DataFrame(prorities, columns = ['forbidden_to_buy', 'has_advantage', 'significance_level','govt_vehicle','description'])
    return res_df
train[['forbidden_to_buy', 'has_advantage', 'significance_level','govt_vehicle','description']] = add_prorities(train)
test[['forbidden_to_buy', 'has_advantage', 'significance_level','govt_vehicle','description']] = add_prorities(test)


train['plate_region_code']= train['plate_region_code'].astype(int)
test['plate_region_code']= test['plate_region_code'].astype(int)

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])


def extract_time(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    df['days_from_initial_listing'] = df.groupby('plate')['date'].transform('min')
    df['days_from_initial_listing'] = (df['date'] - df['days_from_initial_listing']).dt.days
    df['months_from_initial_listing'] = round(df['days_from_initial_listing'] / 30,3)
    df['years_from_initial_listing'] = round(df['months_from_initial_listing'] / 12, 3)

    df['year_end'] = df['month'] == 12

    df['listing_num'] = df.groupby('plate')['date'].rank(method='dense').astype(int)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df
train = extract_time(train)
test = extract_time(test)


train.sort_values(by=['plate','date'], ascending=True)


train_df = train.drop(['date', 'plate',], axis=1)
test_df = test.drop(['date', 'plate',], axis=1)


train_df.info()


train_df.nunique()


from sklearn.preprocessing import OrdinalEncoder
OE = OrdinalEncoder()
cols = ['plate_series1','plate_series2','region','plate_series','description']
OE.fit(pd.concat([train_df[cols],test_df[cols]],axis=0))

train_df[cols] = OE.transform(train_df[cols])
test_df[cols] = OE.transform(test_df[cols])


# cat_columns = ['forbidden_to_buy', 'has_advantage', 'significance_level', 'govt_vehicle','plate_series1', 'plate_series2', 'plate_series', 'region','month','day','year','plate_region_code']


plt.figure(figsize = (30,20))
for i, col in enumerate(train_df.columns):
    plt.subplot(4,5,i+1)
    sns.kdeplot(train_df[col])




def pred_kfold(model, X, y):
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in folds.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = smape(np.expm1(y_test), np.expm1(y_pred))

        scores.append(error)

    return np.mean(scores)



train_df.columns


cols_to_drop = ['price']#['price','year','plate_series1','plate_series2']
X = train_df.drop( cols_to_drop, axis=1)
y = np.log1p(train_df['price'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def sample_train():
    def pred(model, name):
        loss = pred_kfold(model,X,y)

        # model.fit(X_train,y_train)
        # pred = model.predict(X_test)
        # loss = smape(np.expm1(y_test), np.expm1(pred))
        print(f'{name} regressor:',loss)

    pred(CatBoostRegressor(verbose=False), 'catboost')
    pred(LGBMRegressor(verbose=-1), 'lightgbm')
    pred(XGBRegressor(), 'xgboost')

sample_train()


model = CatBoostRegressor(verbose=False)
model.fit(X_train, y_train)
pd.DataFrame(list(zip(model.feature_importances_, model.feature_names_)))



# def objective(trial):
#     params = {
#         "iterations": trial.suggest_int("iterations", 500, 3000),
#         "depth": trial.suggest_int("depth", 4, 12),
#         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
#         "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 10.0, log=True),  # Bayesian tuning
#         "border_count": trial.suggest_int("border_count", 32, 255),
#         "bootstrap_type": "Bayesian",  # Bayesian bootstrap
#         #"grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
#         #"max_leaves": trial.suggest_int("max_leaves", 4, 64),  # Only used for Lossguide
#         #"feature_border_type": trial.suggest_categorical("feature_border_type", ["Median", "Uniform", "GreedyLogSum"]),
#         "verbose": False,
#     }

#     # Additional tuning for bootstrap type

#     # Train model
#     model = CatBoostRegressor(**params)  # Change to `CatBoostClassifier` for classification

#     # error = pred_kfold(model, X,y)  # RMSE (change for classification)
#     model.fit(X_train, y_train)
#     pred = model.predict(X_test)
#     error = smape(np.expm1(y_test), np.expm1(pred))
#     return error  # Optuna minimizes the metric

# # Run optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)

# # Best parameters
# print("Best hyperparameters:", study.best_params)


# Best hyperparameters: {'iterations': 960, 'depth': 6, 'learning_rate': 0.2960657270383293, 'l2_leaf_reg': 2.6092590243312563, 'random_strength': 1.5417410279196098, 'bagging_temperature': 0.9626961525566045, 'border_count': 249} 40.6
#{'iterations': 2000, 'depth': 6, 'learning_rate': 0.15130142672624944, 'l2_leaf_reg': 0.0067260418929035, 'random_strength': 0.07744115950948842, 'bagging_temperature': 3.6946905329999615, 'border_count': 251} 40.3
# {'iterations': 2766, 'depth': 6, 'learning_rate': 0.15549073438546138, 'l2_leaf_reg': 2.576664565494825, 'random_strength': 0.005531499970826704, 'bagging_temperature': 0.15377564143769956, 'border_count': 255, 'min_data_in_leaf': 7} 40.02
# {'iterations': 2538, 'depth': 8, 'learning_rate': 0.08470141619243111, 'l2_leaf_reg': 0.006372137071839507, 'random_strength': 0.29981761433181997, 'bagging_temperature': 0.11869968385680696, 'border_count': 223, 'min_data_in_leaf': 50} 40.3 kfold

model = CatBoostRegressor(**{'iterations': 2538, 'depth': 8, 'learning_rate': 0.08470141619243111, 'l2_leaf_reg': 0.006372137071839507, 'random_strength': 0.29981761433181997, 'bagging_temperature': 0.11869968385680696, 'border_count': 223, 'min_data_in_leaf': 50},verbose=False)


pred_kfold(model, X,y)


model.fit(X, y)


pred = np.expm1(model.predict(test_df.drop(cols_to_drop, axis=1)))


submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')


submission


submission['price'] = pred


submission


submission.to_csv('submission.csv', index=False)




