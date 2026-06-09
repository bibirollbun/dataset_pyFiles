import numpy as np 
import pandas as pd 

import os



ross_df = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv', low_memory=False)
store_df = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
test_df = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
submission_df = pd.read_csv('/kaggle/input/rossmann-store-sales/sample_submission.csv')


ross_df


store_df


test_df


submission_df


merged_df = ross_df.merge(store_df, how='left', on='Store')
merged_test_df = test_df.merge(store_df, how='left', on='Store')


merged_df


merged_df.info()


def split_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week


split_date(merged_df)
split_date(merged_test_df)


merged_df


merged_df[merged_df.Open == 0].Sales.value_counts()


merged_df = merged_df[merged_df.Open == 1].copy()


def comp_months(df):
    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df['CompetitionOpen'].map(lambda x: 0 if x<0 else x).fillna(0)


comp_months(merged_df)
comp_months(merged_test_df)


merged_df


merged_df[['Date', 'CompetitionDistance', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth','CompetitionOpen']].sample(20)


def check_promo_month(row):
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',              
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    try: 
        months = (row['PromoInterval'] or '').split(',')
        if row['PromoOpen'] and month2str[row['Month']] in months:
            return 1
        else:
            return 0
    except Exception:
        return 0

def promo_cols(df):
    df['Promo2Open'] = 12 * (df.Year - df.Promo2SinceYear) + (df.WeekOfYear - df.Promo2SinceWeek) *7/30.5
    df['Promo2Open'] = df['Promo2Open'].map(lambda x: 0 if x<0 else x).fillna(0)*df['Promo2']
    df['IsPromo2Month'] = df.apply(check_promo_month, axis=1)*df['Promo2']


promo_cols(merged_df)
promo_cols(merged_test_df)


merged_df[['Date', 'Promo2', 'Promo2SinceYear','Promo2SinceWeek', 'PromoInterval','Promo2Open','IsPromo2Month']].sample(20)


merged_df.columns


input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
              'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpen', 
              'Day', 'Month', 'Year', 'WeekOfYear',  'Promo2', 
              'Promo2Open', 'IsPromo2Month']
target_col = 'Sales'


inputs = merged_df[input_cols].copy()
targets = merged_df[target_col].copy()


test_inputs = merged_test_df[input_cols].copy()


numeric_cols = ['Store', 'Promo', 'SchoolHoliday', 
              'CompetitionDistance', 'CompetitionOpen', 'Promo2', 'Promo2Open', 'IsPromo2Month',
              'Day', 'Month', 'Year', 'WeekOfYear',  ]
categorical_cols = ['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment']


inputs[numeric_cols].isna().sum()


test_inputs[numeric_cols].isna().sum()


max_distance = inputs.CompetitionDistance.max()


inputs['CompetitionDistance'].fillna(max_distance, inplace=True)
test_inputs['CompetitionDistance'].fillna(max_distance, inplace=True)


max_distance = inputs.CompetitionDistance.max()


inputs['CompetitionDistance'].fillna(max_distance, inplace = True)
test_inputs['CompetitionDistance'].fillna(max_distance, inplace = True)


from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler().fit(inputs[numeric_cols])


inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(sparse = False, handle_unknown = 'ignore').fit(inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))


inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


X= inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols+encoded_cols]


X


from xgboost import XGBRegressor


model = XGBRegressor(random_state=42, n_jobs= -1, n_estimators=20, max_depth =4)


%%time
model.fit(X, targets)


preds = model.predict(X)


from sklearn.metrics import mean_squared_error

def rmse(a, b):
    return mean_squared_error(a, b, squared=False)

def rmspe(a, b, eps=1e-8):
    a, b = np.array(a), np.array(b)
    return np.sqrt(np.mean(((b - a) / (a + eps)) ** 2))



rmse(preds, targets)


merged_df.Sales.min(), merged_df.Sales.max()


import matplotlib.pyplot as plt
from xgboost import plot_tree
from matplotlib.pylab import rcParams
%matplotlib inline

rcParams['figure.figsize'] = 30,30


plot_tree(model, rankdir='LR')


plot_tree(model, rankdir='LR', num_trees=1);


plot_tree(model, rankdir='LR', num_trees=19);


trees = model.get_booster().get_dump()


len(trees)


print(trees[0])


importance_df = pd.DataFrame({'feature': X.columns, 
                              'importance': model.feature_importances_}).sort_values('importance', ascending=False)


importance_df.head(10)


import seaborn as sns
plt.figure(figsize=(10,6))
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature');


from sklearn.model_selection import KFold


def train_and_evaluate(X_train, train_targets, X_val, val_targets, **params):
    model = XGBRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, train_targets)
    train_rmse = rmse(model.predict(X_train), train_targets)
    train_rmspe = rmspe(model.predict(X_train), train_targets)
    val_rmse = rmse(model.predict(X_val), val_targets)
    val_rmspe = rmspe(model.predict(X_val), val_targets)
    return model, train_rmse, val_rmse, train_rmspe, val_rmspe


kfold = KFold(n_splits=5)


models = []

for train_idxs, val_idxs in kfold.split(X):
    X_train, train_targets = X.iloc[train_idxs], targets.iloc[train_idxs]
    X_val, val_targets = X.iloc[val_idxs], targets.iloc[val_idxs]
    model, train_rmse, val_rmse, train_rmspe, val_rmspe= train_and_evaluate(X_train, 
                                                     train_targets, 
                                                     X_val, 
                                                     val_targets, 
                                                     max_depth=4, 
                                                     n_estimators=20)
    models.append(model)
    print('Train RMSE: {}, Validation RMSE, : {}, Train RMSPE: {}, Validation RMSPE: {}'.format(train_rmse, val_rmse, train_rmspe, val_rmspe))


import numpy as np

def predict_avg(models, inputs):
    return np.mean([model.predict(inputs) for model in models], axis=0)


preds = predict_avg(models, X)


preds


from sklearn.model_selection import train_test_split


X_train, X_val, train_targets, val_targets = train_test_split(X, targets, test_size=0.1)


def test_params(**params):
    model = XGBRegressor(n_jobs=-1, random_state=42, **params)
    model.fit(X_train, train_targets)
    train_rmse = rmse(model.predict(X_train), train_targets)
    train_rmspe = rmspe(model.predict(X_train), train_targets)
    val_rmse = rmse(model.predict(X_val), val_targets)
    val_rmspe = rmspe(model.predict(X_val), val_targets)
    
    print('Train RMSE: {}, Validation RMSE, : {}, Train RMSPE: {}, Validation RMSPE: {}'.format(train_rmse, val_rmse, train_rmspe, val_rmspe))


test_params(n_estimators=10)


test_params(n_estimators=30)


test_params(n_estimators=100)


test_params(n_estimators=240)


test_params(max_depth=2)


test_params(max_depth=5)


test_params(max_depth=10)


test_params(n_estimators=50, learning_rate=0.01)


test_params(n_estimators=50, learning_rate=0.1)


test_params(max_depth = 10, n_estimators=150, learning_rate=0.2)


test_params(n_estimators=50, learning_rate=0.9)


test_params(n_estimators=50, learning_rate=0.99)


test_params(booster='gblinear')


model = XGBRegressor(n_jobs=-1, random_state=42, n_estimators=150, 
                     learning_rate=0.2, max_depth=10, subsample=0.9, 
                     colsample_bytree=0.7)


%%time
model.fit(X, targets)


test_preds = model.predict(X_test)


submission_df['Sales']  = test_preds


test_df.Open.isna().sum()


submission_df['Sales'] = submission_df['Sales'] * test_df.Open.fillna(1.)


submission_df




