import os 
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit  
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import train_test_split  
import xgboost as xgb
from xgboost import XGBRegressor
import catboost as cb
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

import pickle
import time
import joblib
from joblib import dump, load


df_train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv', low_memory=False, parse_dates=["Date"])
df_store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
df_com = pd.merge(df_train, df_store, on='Store', how='inner')


df_com.info()


df_com['Date'] = pd.to_datetime(df_com['Date'])
# Set 'Date' as the index and drop the original column

df_com['Day'] = df_com['Date'].dt.day
df_com['Month'] = df_com['Date'].dt.month
df_com['Year'] = df_com['Date'].dt.year

df_com = df_com.set_index('Date', drop=True)



df_com.head(5)


df_com.isna().sum()


df_com = df_com.dropna(subset=['CompetitionDistance'])


df_com.isna().sum()


def oulier_check(df, name):
  for n in name:
    plt.figure(figsize=(14,10))
    sns.boxplot(df[n])
    plt.title(f'Distribution of {n}')
    plt.tight_layout()
    plt.show()


oulier_check(df_com,[ 'Customers', 'CompetitionDistance',	'CompetitionOpenSinceMonth',	'CompetitionOpenSinceYear', 'DayOfWeek', 'Promo2SinceWeek',	'Promo2SinceYear'])



def plot_his(df, name):
    df[name] = df[name].replace([np.inf, -np.inf], np.nan)  # xá»­ lÃ½ inf

    rows = int(np.ceil(len(name) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(name):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {col}', fontsize=14)

    # XÃ³a cÃ¡c Ã´ subplot dÆ° náº¿u cÃ³
    for j in range(len(name), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


plot_his(df_com, [	'CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 'Promo2SinceWeek','Promo2SinceYear','PromoInterval']) 


def count_plot(df, name):

    rows = int(np.ceil(len(name) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(name):
        sns.barplot(x = df[name].value_counts().index, y = df[name].value_counts())
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
    

    for j in range(len(name), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()



plt.figure(figsize=(12, 8))
sns.barplot(x = df_com['CompetitionOpenSinceMonth'].value_counts().index, y = df_com['CompetitionOpenSinceMonth'].value_counts())
plt.title(f'Distribution of CompetitionOpenSinceMonth')
plt.show()


def fill_nan_by_distribution(df, col):
    np.random.seed(42)
    value_counts = df[col].value_counts(normalize=True)
    
    n_missing = df[col].isna().sum()
    
    sampled_values = np.random.choice(
        value_counts.index, 
        size=n_missing, 
        p=value_counts.values
    )
    
    df.loc[df[col].isna(), col] = sampled_values
    return df


df_com = fill_nan_by_distribution(df_com , 'CompetitionOpenSinceMonth')


plt.figure(figsize=(12, 8))
sns.barplot(x = df_com['CompetitionOpenSinceMonth'].value_counts().index, y = df_com['CompetitionOpenSinceMonth'].value_counts())
plt.title(f'Distribution of CompetitionOpenSinceMonth')
plt.show()


df_com = fill_nan_by_distribution(df_com , 'CompetitionOpenSinceYear')


plt.figure(figsize=(12, 8))
sns.barplot(x = df_com['CompetitionOpenSinceYear'].value_counts().index, y = df_com['CompetitionOpenSinceYear'].value_counts())
plt.title(f'Distribution of CompetitionOpenSinceYear')
plt.show()


def fill_nan_by_zero(df, col):
    df[col] = df[col].fillna(0) 
    return df


df_com = fill_nan_by_zero(df_com , 'Promo2SinceWeek')


plt.figure(figsize=(12, 8))
sns.barplot(x = df_com['Promo2SinceWeek'].value_counts().index, y = df_com['Promo2SinceWeek'].value_counts())
plt.title(f'Distribution of Promo2SinceWeek')
plt.show()


df_com = fill_nan_by_zero(df_com , 'Promo2SinceYear')


plt.figure(figsize=(12, 8))
sns.barplot(x = df_com['Promo2SinceYear'].value_counts().index, y = df_com['Promo2SinceYear'].value_counts())
plt.title(f'Distribution of Promo2SinceYear')
plt.show()


df_com = fill_nan_by_zero(df_com , 'PromoInterval')


plt.figure(figsize=(12, 8))
sns.barplot(x = df_com['PromoInterval'].value_counts().index, y = df_com['PromoInterval'].value_counts())
plt.title(f'Distribution of PromoInterval')
plt.show()


oulier_check(df_com,['CompetitionDistance'])


df_com['CompetitionDistance'] = np.log(df_com['CompetitionDistance'])


oulier_check(df_com,['CompetitionDistance'])


length_data = df_com.shape[0]


df_rate_null = df_com[[ 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']]
rate = (df_rate_null.isna().sum() / length_data) * 100
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=rate.index, y=rate)
plt.xticks(rotation=90)
plt.ylabel("Missing Rate (%)")
plt.title("Rate")

# ThÃªm nhÃ£n tá»· lá»‡ trÃªn Ä‘áº§u má»—i cá»™t
for i, v in enumerate(rate):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()


# df_com = df_com.drop(columns = ['PromoInterval'])


print(df_com.shape[0])


df_com = df_com[df_com['Open'] == 1]
df_com = df_com.drop(columns = ['Open'])


plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_com["Customers"], y=df_com["Sales"], alpha=0.5)

plt.xlabel("Customers")
plt.ylabel("Sales")
plt.title("Má»‘i quan há»‡ giá»¯a Customers  vÃ  Sales")
plt.show()


corre = df_com[ ['Customers','DayOfWeek','Sales']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corre, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


df_com = df_com.drop(columns = ['Customers'])


df_com = df_com.drop(columns = ['Store'])


df_com['CompetitionOpen'] = 12 * (df_com.Year - df_com.CompetitionOpenSinceYear) + (df_com.Month - df_com.CompetitionOpenSinceMonth)


plt.figure(figsize = (15, 8) )
sns.barplot(data=df_com, x='CompetitionOpen', y='Sales')
plt.title('Sales by CompetitionOpen')
plt.xlabel('CompetitionOpen')
plt.ylabel('Sales')
plt.show()


df_com['lag1'] = df_com['Sales'].shift(1)
df_com = df_com.dropna()


categorical_cols = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']

df_com_onehot = pd.get_dummies(df_com, columns=categorical_cols, drop_first=True).astype(int)


mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
mapping_PromoInterval = {0:0, 'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3,}
df_com_la = df_com.copy()


df_com_la['StoreType'] = df_com_la['StoreType'].map(mappings)
df_com_la['Assortment'] = df_com_la['Assortment'].map(mappings)
df_com_la['StateHoliday'] = df_com_la['StateHoliday'].map(mappings)
df_com_la['PromoInterval'] = df_com_la['PromoInterval'].map(mapping_PromoInterval)


df_com_la.isna().sum()


df_encode = df_com.copy()
categorical_col = ['StateHoliday']

# One-hot encode
df_encode = pd.get_dummies(df_encode, columns=categorical_col, drop_first=True)

# Only cast the new one-hot columns to int
onehot_cols = [col for col in df_encode.columns if any(cat in col for cat in categorical_col)]
df_encode[onehot_cols] = df_encode[onehot_cols].astype(int)

df_encode['StoreType'] = df_encode['StoreType'].map(mappings)
df_encode['Assortment'] = df_encode['Assortment'].map(mappings)
df_encode['PromoInterval'] = df_encode['PromoInterval'].map(mapping_PromoInterval)


def normalize_data(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_val_scaled= scaler.transform(X_val) 
    X_test_scaled = scaler.transform(X_test) 
    return X_train_scaled, X_val_scaled, X_test_scaled


def split_data(df):
    df.sort_index(inplace=True)

    total_len = len(df)
    
    train_end = int(0.7 * total_len)
    val_end = int(0.9 * total_len) 
    
    X = df.drop(columns=['Sales'])
    y = df['Sales']
    
    X_train = X.iloc[:train_end]
    X_val= X.iloc[train_end:val_end]
    X_test= X.iloc[val_end:]
    
    y_train = (y.iloc[:train_end])
    y_val = (y.iloc[train_end:val_end])
    y_test = (y.iloc[val_end:])
    return (X_train,y_train ), (X_val,y_val ), (X_test,y_test )


def rmspe(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    mask = y != 0
    return np.sqrt(np.mean(((yhat[mask] / y[mask]) - 1) ** 2))



def try_model(model, X_train_scaled, y_train, X_val_scaled, y_val):
    # Fit the model
    start = time.time()
    model.fit(X_train_scaled, y_train)
    end = time.time()
    
    # Generate predictions
    train_preds = model.predict(X_train_scaled)
    val_preds = model.predict(X_val_scaled)
    
    # Compute RMSE
    train_rmspe = rmspe(y_train,train_preds)
    val_rmspe = rmspe(y_val,val_preds)
    print(f'Time of train: {end - start}s')
    print(f'MSE: {mean_squared_error(y_val,val_preds)}')
    print(train_rmspe, val_rmspe)
    return train_rmspe, val_rmspe


(X_train_onehot,y_train_onehot ), (X_val_onehot,y_val_onehot ), (X_test_onehot,y_test_onehot ) = split_data(df_com_onehot)


plt.figure(figsize=(15,8))
ax = y_train_onehot.plot()
y_val_onehot.plot(ax=ax)
y_test_onehot.plot(ax=ax)
plt.legend(['Train', 'Val', 'Test'])
plt.title('Data Split')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()


X_train_scaled_onehot, X_val_scaled_onehot, X_test_scaled_onehot = normalize_data(X_train_onehot,X_val_onehot, X_test_onehot ) 


# model_onehot = CatBoostRegressor(verbose=0)
# try_model(model_onehot, X_train_onehot, y_train_onehot, X_val_onehot, y_val_onehot)


# model_onehot= XGBRegressor()
# try_model(model_onehot, X_train_scaled_onehot, y_train_onehot,X_val_scaled_onehot,y_val_onehot  )


model_onehot = LGBMRegressor()
train_rmspe, val_rmspe = try_model(model_onehot, X_train_scaled_onehot, y_train_onehot,X_val_scaled_onehot,y_val_onehot)


(X_train_la,y_train_la ), (X_val_la,y_val_la ), (X_test_la,y_test_la ) = split_data(df_com_la)


X_train_scaled_la, X_val_scaled_la, X_test_scaled_la = normalize_data(X_train_la,X_val_la, X_test_la ) 


# model_la = CatBoostRegressor(verbose=0)
# try_model(model_la, X_train_la, y_train_la,X_val_la,y_val_la)


# model_la= XGBRegressor()
# try_model(model_la, X_train_scaled_la, y_train_la,X_val_scaled_la,y_val_la)


model_la = LGBMRegressor()
train_rmspe, val_rmspe = try_model(model_la, X_train_scaled_la, y_train_la,X_val_scaled_la,y_val_la  )


(X_train_es,y_train_es ), (X_val_es,y_val_es ), (X_test_es,y_test_es ) = split_data(df_encode)


X_train_scaled_es, X_val_scaled_es, X_test_scaled_es = normalize_data(X_train_es, X_val_es, X_test_es ) 


# model_es= CatBoostRegressor(verbose=0)
# try_model(model_es, X_train_es, y_train_es,X_val_es,y_val_es)


model_es= LGBMRegressor()
try_model(model_es, X_train_scaled_es, y_train_es,X_val_scaled_es,y_val_es)


importances = model_es.feature_importances_
feature_names = X_train_es.columns
indices = np.argsort(importances)

# Váº½ biá»ƒu Ä‘á»“
plt.figure(figsize=(8, 6))

plt.title('Ä�á»™ quan trá»�ng cá»§a tá»«ng biáº¿n (Dá»± Ä‘oÃ¡n xu hÆ°á»›ng)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

plt.xlabel('Feature Importance')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


df_cb = df_encode.drop(columns = [ 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c'])



(X_train1,y_train1 ), (X_val1,y_val1 ), (X_test1,y_test1 ) = split_data(df_cb)
 


X_train_scaled1,X_val_scaled1,X_test_scaled1   = normalize_data(X_train1,X_val1,X_test1  ) 


# model_es= XGBRegressor()
# try_model(model_es, X_train_scaled1, y_train1,X_val_scaled1,y_val1  )


# model_es= CatBoostRegressor(verbose=0)
# try_model(model_es, X_train1, y_train1,X_val1,y_val1)


model_es = LGBMRegressor()
try_model(model_es, X_train_scaled1, y_train1,X_val_scaled1,y_val1  )


# param_grid = {
#     'learning_rate': [0.09, 0.1],
#     'iterations': [ 300, 400],
#     'depth': [8,9, 10],
#     'l2_leaf_reg': [1, 2],
#     'random_strength': [1, 2, 3],
# }



# tscv = TimeSeriesSplit(n_splits=5)
# rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# model_cb = CatBoostRegressor(
#      random_state=42,
#     task_type='GPU',
#     devices='0',      
#     verbose=0
# )

# grid_search = GridSearchCV(
#     estimator=model_cb,
#     param_grid=param_grid,
#     scoring=rmspe_scorer,    
#     cv=tscv,                    
#     verbose=0,
# )

# grid_search.fit(X_train_scaled_es, y_train_es)

# best_model_cb = grid_search.best_estimator_

# print("Best parameters found: ", grid_search.best_params_)

# y_pred_cb = best_model_cb.predict(X_val_scaled_es)
# print("RMSPE score on validation:", rmspe(y_val_es, y_pred_cb))
# print(f'MSE: {mean_squared_error(y_val_es, y_pred_cb)}')


# def rmspe_catboost(y_pred, dtrain):
#     y_true = dtrain.get_label()
#     non_zero_idx = y_true != 0
#     if np.any(non_zero_idx):
#         error = rmspe(y_true[non_zero_idx], y_pred[non_zero_idx])
#     else:
#         error = 0.0
#     return 'RMSPE', error



# dtrain = Pool(X_train1, y_train1)
# dval = Pool(X_val1, y_val1)

# model = CatBoostRegressor(
#     iterations=10000,
#     learning_rate=0.1,
#     depth=6,
#     early_stopping_rounds=100,
#     verbose=100,
#     task_type='GPU',
#     devices='0'
# )

# model.fit(dtrain, eval_set=dval, use_best_model=True)

# # Predict
# y_pred = model.predict(dval)

# score = rmspe(y_val1, y_pred)
# print(f'RMSPE: {score:.5f}')



# param_grid = {
#     'learning_rate': [0.01, 0.02, 0.1],
#     'max_depth': [5, 7, 8],
#     'n_estimators': [100, 200],
#     'gamma': [0, 0.1, 0.5],
#     'min_child_weight': [1, 5, 6],
#     "device" : ["cuda"]
# }



# from sklearn.model_selection import TimeSeriesSplit

# tscv = TimeSeriesSplit(n_splits=5)
# rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# # ğŸ‘‰ Khá»Ÿi táº¡o model
# model_xgb = XGBRegressor(
#     random_state=42
# )

# # ğŸ‘‰ GridSearchCV
# grid_search = GridSearchCV(
#     estimator=model_xgb,
#     param_grid=param_grid,
#     scoring=rmspe_scorer,    
#     cv=3,                    
#     verbose=0,
# )

# # Fit
# grid_search.fit(X_train_scaled1, y_train1)

# # Best model
# best_model_xgb = grid_search.best_estimator_

# print("Best parameters found: ", grid_search.best_params_)

# # Predict & Ä‘Ã¡nh giÃ¡

# # âœ¨ Predict
# y_pred_xgb = best_model_xgb.predict(X_val_scaled1)
# print("RMSPE score on validation:", rmspe(y_val1, y_pred_xgb))



# def rmspe_xgboost(y_pred, dtrain):
#     y_true = dtrain.get_label()
#     # Avoid division by zero
#     non_zero_idx = y_true != 0
#     if np.any(non_zero_idx):
#         error = rmspe(y_true[non_zero_idx], y_pred[non_zero_idx])
#     else:
#         error = 0.0
#     return 'RMSPE', error



# # Táº¡o DMatrix cho XGBoost
# dtrain = xgb.DMatrix(X_train_scaled1, label=y_train1)
# dval = xgb.DMatrix(X_val_scaled1, label=y_val1)

# # Bá»™ tham sá»‘ XGBoost chuáº©n Ä‘á»ƒ báº¯t Ä‘áº§u
# params = {
#     'objective': 'reg:squarederror',    # bÃ i toÃ¡n regression
#     'learning_rate': 0.1,               # small learning rate
#     'max_depth': 8,                      # cÃ¢y sÃ¢u vá»«a pháº£i
#     'min_child_weight': 5,                # giáº£m overfitting
#     'gamma': 0,                         # thÃªm regularization
# }

# # Train vá»›i Early stopping
# model = xgb.train(
#     params,
#     dtrain,
#     num_boost_round=10000,
#     evals=[(dtrain, 'train'), (dval, 'eval')],
#     feval=rmspe_xgboost,
#     early_stopping_rounds=100,
#     verbose_eval=100
# )

# # Predict
# y_pred = model.predict(dval)

# score = rmspe(y_val1, y_pred)
# print(f'RMSPE: {score:.5f}')



param_grid = {
    'num_leaves': [31, 64, 128],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'bagging_fraction': [0.6, 0.8, 1.0],
    'min_data_in_leaf': [10, 20, 50]
}


tscv = TimeSeriesSplit(n_splits=5)
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# ğŸ‘‰ Khá»Ÿi táº¡o model
model_lgbm = LGBMRegressor(
    random_state=42
)

# ğŸ‘‰ GridSearchCV
grid_search = GridSearchCV(
    estimator=model_lgbm,
    param_grid=param_grid,
    scoring=rmspe_scorer,    
    cv=tscv,                    
    verbose=0,
)

# Fit
grid_search.fit(X_train_scaled1, y_train1)




# Best model
best_model_lgbm = grid_search.best_estimator_

print("Best parameters found: ", grid_search.best_params_)

# Predict & Ä‘Ã¡nh giÃ¡

# âœ¨ Predict
y_pred_lgbm = best_model_lgbm.predict(X_val_scaled1)
print("RMSPE score on validation:", rmspe(y_val1, y_pred_lgbm))


def rmspe_lgbm(y_pred, dtrain):
    y_true = dtrain.get_label()
    non_zero_idx = y_true != 0
    if np.any(non_zero_idx):
        error = rmspe(y_true[non_zero_idx], y_pred[non_zero_idx])
    else:
        error = 0.0
    return 'RMSPE', error, False 


dtrain = lgb.Dataset(X_train_scaled1, label=y_train1)
dval = lgb.Dataset(X_val_scaled1, label=y_val1)


params = {
    'num_leaves': 128,
    'max_depth': 15,
    'bagging_fraction': 0.6,
    'min_data_in_leaf': 10
}                   


model = lgb.train(
    params,
    dtrain,
    num_boost_round=10000,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'valid'],
    feval=rmspe_lgbm,
    callbacks=[lgb.early_stopping(stopping_rounds=100),
              lgb.log_evaluation(period=100)]
    
)

y_pred = model.predict(X_val_scaled1, num_iteration=model.best_iteration)

score = rmspe(y_val1, y_pred)
print(f'RMSPE: {score:.5f}')



# dtest = xgb.DMatrix(X_test_scaled1, label=y_test1)
# y_pred_test = model.predict(dtest)

# print(f'RMSPE: {rmspe(y_test1, y_pred_test)}')




# dtest = Pool(X_test1, label=y_test1)
# y_pred_test = model.predict(dtest)

# print(f'RMSPE: {rmspe(y_test1, y_pred_test)}')



start = time.time()
y_pred_test = model.predict(X_test_scaled1)
end = time.time()
print(f'RMSPE: {rmspe(y_test1, y_pred_test)}')
print(f'MSE: {mean_squared_error(y_test1, y_pred_test)}')
print(f'R2:{r2_score(y_test1, y_pred_test)}')
print(f'Time of test: {end - start} s')


model_bytes = pickle.dumps(model)
model_size = len(model_bytes)

print(f"KÃ­ch thÆ°á»›c mÃ´ hÃ¬nh: {model_size / 1024 ** 2:.2f} MB")

joblib.dump(model, 'LightGBM_model.pkl')

np.save('y_hat_XGB',y_pred_test)

