# Importing necessary libraries
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit  # For time series cross-validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For feature scaling
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

import os  # For interacting with the operating system



# Load the training dataset with specified options
df_train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv', low_memory=False, parse_dates=["Date"])

# Load the store dataset
df_store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

# Merge the datasets on the 'Store' column using inner join
df_com = pd.merge(df_train, df_store, on='Store', how='inner')


print(df_com.shape[0])


df_com['Date'] = pd.to_datetime(df_com['Date'])
# Set 'Date' as the index and drop the original column

df_com['Day'] = df_com['Date'].dt.day
df_com['Month'] = df_com['Date'].dt.month
df_com['Year'] = df_com['Date'].dt.year

df_com = df_com.set_index('Date', drop=True)



df_com.head(5)


df_com = df_com[df_com['Open'] == 1]
df_com = df_com.drop(columns = ['Open'])


df_com.isna().sum()


df_com = df_com.dropna(subset=['CompetitionDistance'])
df_com = df_com.drop(columns = ['Store'])


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
    df[name] = df[name].replace([np.inf, -np.inf], np.nan)  # xử lý inf

    rows = int(np.ceil(len(name) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(name):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {col}', fontsize=14)

    # Xóa các ô subplot dư nếu có
    for j in range(len(name), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



plot_his(df_com, [	'CompetitionOpenSinceMonth',	'CompetitionOpenSinceYear', 'Promo2SinceWeek',	'Promo2SinceYear',	'PromoInterval']) # biến liên tục


# df_com = df_com.drop(columns = ['Promo2SinceWeek','Promo2SinceYear','PromoInterval'])


def count_plot(df, name):

    rows = int(np.ceil(len(name) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(name):
        sns.barplot(x = df[name].value_counts().index, y = df[name].value_counts())
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
    
    # Xóa các ô subplot dư nếu có
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
    # Lấy các giá trị không NaN
    value_counts = df[col].value_counts(normalize=True)
    
    # Lấy số lượng NaN
    n_missing = df[col].isna().sum()
    
    # Lấy mẫu theo phân phối gốc
    sampled_values = np.random.choice(
        value_counts.index, 
        size=n_missing, 
        p=value_counts.values
    )
    
    # Gán lại vào các vị trí NaN
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
    # Gán giá trị 0 vào các vị trí NaN trong cột
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


df_com = df_com.drop(columns = ['PromoInterval'])


df_com.isna().sum()


categorical_cols = ['StateHoliday', 'StoreType', 'Assortment']

df_com_onehot = pd.get_dummies(df_com, columns=categorical_cols, drop_first=True).astype(int)


df_com_onehot.head()


mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
df_com_la = df_com.copy()

df_com_la['StoreType'] = df_com_la['StoreType'].map(mappings)
df_com_la['Assortment'] = df_com_la['Assortment'].map(mappings)
df_com_la['StateHoliday'] = df_com_la['StateHoliday'].map(mappings)




df_com_la.head()


df_com_onehot = df_com_onehot.drop(columns= ['Customers'])
df_com_la = df_com_la.drop(columns= ['Customers'])


def split_data(df):
    df.sort_index(inplace=True)

    total_len = len(df)
    
    train_end = int(0.7 * total_len)
    val_end = int(0.9 * total_len)  # 70% + 20% = 90%
    
    # Split features and target
    X = df.drop(columns=['Sales'])
    y = df['Sales']
    
    # Manual split
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
def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)


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


# from sklearn.preprocessing import MinMaxScaler

# scaler_onehot = MinMaxScaler()
# X_train_scaled_onehot = scaler_onehot.fit_transform(X_train_onehot) 
# X_val_scaled_onehot = scaler_onehot.transform(X_val_onehot) 
# X_test_scaled_onehot = scaler_onehot.transform(X_test_onehot) 


def try_model(model, X_train_scaled, y_train, X_val_scaled, y_val):
    # Fit the model
    model.fit(X_train_scaled, y_train)
#
    # Generate predictions
    train_preds = model.predict(X_train_scaled)
    val_preds = model.predict(X_val_scaled)
    
    # Compute RMSE
    train_rmse = rmspe(y_train,train_preds)
    val_rmse = rmspe(y_val,val_preds)
    return train_rmse, val_rmse 


# from xgboost import XGBRegressor

# model_onehot= XGBRegressor()
# try_model(model_onehot, X_train_scaled_onehot, y_train_onehot,X_val_scaled_onehot,y_val_onehot  )


from catboost import CatBoostRegressor, Pool

model_onehot = CatBoostRegressor(verbose=0)
try_model(model_onehot, X_train_onehot, y_train_onehot, X_val_onehot, y_val_onehot)


(X_train_la,y_train_la ), (X_val_la,y_val_la ), (X_test_la,y_test_la ) = split_data(df_com_la)


# from sklearn.preprocessing import MinMaxScaler

# scaler_onehot = MinMaxScaler()
# X_train_scaled_la = scaler_onehot.fit_transform(X_train_la) 
# X_val_scaled_la= scaler_onehot.transform(X_val_la) 
# X_test_scaled_la = scaler_onehot.transform(X_test_la) 


# from xgboost import XGBRegressor

# model_la= XGBRegressor()
# try_model(model_la, X_train_scaled_la, y_train_la,X_val_scaled_la,y_val_la  )


from catboost import CatBoostRegressor, Pool

model_la = CatBoostRegressor(verbose=0)
try_model(model_la, X_train_la, y_train_la,X_val_la,y_val_la)


importances = model_la.feature_importances_
feature_names = X_train_la.columns
indices = np.argsort(importances)

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))

plt.title('Độ quan trọng của từng biến (Dự đoán xu hướng)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

plt.xlabel('Feature Importance')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# # Assuming you have already defined your X_train_la, y_train_la, and X_test
# np.random.seed(42)
# model_xgb1 = XGBRegressor()

# params = {
#     "n_estimators" : [50, 100, 150] ,
# }

# grid = GridSearchCV(estimator=model_xgb1, param_grid=params, n_jobs=-1, verbose=1, cv =None)
# grid.fit(X_train_scaled_la, y_train_la)

# print("Best parameters:", grid.best_params_)
# best_model_xgb = grid.best_estimator_
# pred_grid = best_model_xgb.predict(X_val_scaled_la)
# print("RMSPE:", rmspe(y_val_la, pred_grid))



# model_xgb_best1 = XGBRegressor( n_estimators = 150, booster='gbtree', device = 'cuda', random_state = 42)
# model_xgb_best1.fit(X_train_scaled_la, y_train_la)

# y_hat_xgb1 = model_xgb_best1.predict(X_val_scaled_la)
# print(rmspe(y_val_la, y_hat_xgb1))


# from sklearn.ensemble import RandomForestRegressor



# model_rf1= RandomForestRegressor()
# params = {
#     "n_estimators" : [50, 100, 150] , 
# }

# grid = GridSearchCV(estimator=model_rf1, param_grid=params, n_jobs=-1, verbose=1)
# grid.fit(X_train_scaled_la, y_train_la)

# best_model_rf = grid.best_estimator_
# print("Best parameters:", grid.best_params_)
# pred_grid = best_model_rf.predict(X_val_scaled_la)
# print("RMSPE:", rmspe(y_val_la, pred_grid))


# model_rf_best1 = RandomForestRegressor(n_estimators = 150, random_state = 42)
# model_rf_best1.fit(X_train_scaled_la, y_train_la)
# y_hat_rf1 = model_rf_best1.predict(X_val_scaled_la)
# print(rmspe(y_val_la, y_hat_rf1))


# dump(model_rf_best1, 'model_rf_best1.joblib')


# importances = model_rf_best1.feature_importances_
# feature_names = X_train_la.columns
# indices = np.argsort(importances)

# # Vẽ biểu đồ
# plt.figure(figsize=(8, 6))

# plt.title('Độ quan trọng của từng biến (Dự đoán xu hướng)')
# plt.barh(range(len(indices)), importances[indices], align='center')
# plt.yticks(range(len(indices)), [feature_names[i] for i in indices])

# plt.xlabel('Feature Importance')
# plt.tight_layout()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()


# np.random.seed(42)
# model_cb1 = CatBoostRegressor( task_type="GPU", devices='0', verbose = 0)

# params = {
#     'iterations': [50, 100, 150]
# }

# grid = GridSearchCV(estimator=model_cb1, param_grid=params, n_jobs=1, verbose=1, cv =3)
# grid.fit(X_train_la, y_train_la)

# print("Best parameters:", grid.best_params_)
# best_model_cb = grid.best_estimator_
# pred_grid = best_model_cb.predict(X_val_la)
# print("RMSPE:", rmspe(y_val_la, pred_grid))


model_cb_best1 = CatBoostRegressor(task_type="GPU", devices='0',verbose=0 , random_state = 42)
model_cb_best1.fit(X_train_la, y_train_la)

y_hat_cb1 = model_cb_best1.predict(X_val_la)
print(rmspe(y_val_la, y_hat_cb1))


# y_ensemble = (y_hat_rf1 + y_hat_xgb1 * 0.75)  / (2)
# print(rmspe(y_val_la, y_ensemble))


# df_xgb = df_com_la.drop(columns = ['Year', 'StateHoliday'])
# df_rf =  df_com_la.drop(columns = ['SchoolHoliday', 'StateHoliday'])
df_cb = df_com_la.drop(columns = ['SchoolHoliday', 'StateHoliday'])


# (X_train1,y_train1 ), (X_val1,y_val1 ), (X_test1,y_test1 ) = split_data(df_xgb)
# (X_train2,y_train2 ), (X_val2,y_val2 ), (X_test2,y_test2 ) = split_data(df_rf)
(X_train3,y_train3 ), (X_val3,y_val3 ), (X_test3,y_test3 ) = split_data(df_cb)


# scaler1 = MinMaxScaler()
# X_train_scaled1 = scaler1.fit_transform(X_train1) 
# X_val_scaled1= scaler1.transform(X_val1) 
# X_test_scaled1 = scaler1.transform(X_test1) 


# # Assuming you have already defined your X_train_la, y_train_la, and X_test
# np.random.seed(42)
# model_xgb2 = XGBRegressor()

# params = {
#     "n_estimators" : [50, 100, 150] ,
# }

# grid = GridSearchCV(estimator=model_xgb2, param_grid=params, n_jobs=-1, verbose=1, cv =None)
# grid.fit(X_train_scaled1, y_train1)

# print("Best parameters:", grid.best_params_)
# best_model_xgb = grid.best_estimator_
# pred_grid = best_model_xgb.predict(X_val_scaled1)
# print("RMSPE:", rmspe(y_val1, pred_grid))



# model_xgb_best2 = XGBRegressor( n_estimators = 150, booster='gbtree', device = 'cuda', random_state = 42)
# model_xgb_best2.fit(X_train_scaled1, y_train1)

# y_hat_xgb2 = model_xgb_best2.predict(X_val_scaled1)
# print(rmspe(y_val1, y_hat_xgb2))


# scaler2 = MinMaxScaler()
# X_train_scaled2 = scaler2.fit_transform(X_train2) 
# X_val_scaled2 = scaler2.transform(X_val2) 
# X_test_scaled2 = scaler2.transform(X_test2) 


# model_rf2= RandomForestRegressor()
# params = {
#     "n_estimators" : [ 50, 100, 150] , # Removed extra space
# }

# grid = GridSearchCV(estimator=model_rf2, param_grid=params, n_jobs=-1, verbose=1, cv =3)
# grid.fit(X_train_scaled2, y_train2)

# best_model_rf = grid.best_estimator_
# print("Best parameters:", grid.best_params_)
# pred_grid = best_model_rf.predict(X_val_scaled2)
# print("RMSPE:", rmspe(y_val2, pred_grid))


# model_rf_best2 = RandomForestRegressor(n_estimators = 100, random_state = 42)
# model_rf_best2.fit(X_train_scaled2, y_train2)
# y_hat_rf2 = model_rf_best2.predict(X_val_scaled2)
# print(rmspe(y_val2, y_hat_rf2))


# from joblib import dump, load

# # Save the model
# dump(model_rf_best2, 'random_rft_best2.joblib')


# y_ensemble2 = (y_hat_rf2 + y_hat_xgb2 * 0.75) / (2)
# print(rmspe(y_val2, y_ensemble2))


# Assuming you have already defined your X_train_la, y_train_la, and X_test
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer

tscv = TimeSeriesSplit(n_splits=3)
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

np.random.seed(42)
model_cb2 = CatBoostRegressor()

params = {
    "iterations" : [50, 100, 150] ,
}

grid = GridSearchCV(estimator=model_cb2, param_grid=params, verbose=1, cv =tscv, scoring=rmspe_scorer)
grid.fit(X_train3, y_train3)

print("Best parameters:", grid.best_params_)
best_model_cb = grid.best_estimator_
pred_grid = best_model_cb.predict(X_val3)
print("RMSPE:", rmspe(y_val3, pred_grid))


model_cb_best2 = CatBoostRegressor(iterations = 150 , task_type="GPU", devices='0',verbose=0 , random_state = 42)
model_cb_best2.fit(X_train3, y_train3)

y_hat_cb2 = model_cb_best2.predict(X_val3)
print(rmspe(y_val3, y_hat_cb2))


# y_pred1 = model_rf_best1.predict(X_test_scaled_la)
# y_pred2 = model_xgb_best1.predict(X_test_scaled_la)
# y_pred3 = model_rf_best2.predict(X_test_scaled2)
# y_pred4 = model_xgb_best2.predict(X_test_scaled1)
y_pred5 = model_cb_best1.predict(X_test_la)
y_pred6 = model_cb_best2.predict(X_test3)


# np.save('y_pred1.npy', y_pred1)
# np.save('y_pred2.npy', y_pred2)
# np.save('y_pred3.npy', y_pred3)
# np.save('y_pred4.npy', y_pred4)
np.save('y_pred5.npy', y_pred5)
np.save('y_pred6.npy', y_pred6)



# print('rmspe của model xgb', rmspe(y_test_la, y_pred2))
# print('rmspe của model xgb with select', rmspe(y_test_la, y_pred4))
print('rmspe của model cb', rmspe(y_test_la, y_pred5))
print('rmspe của model cb with select', rmspe(y_test_la, y_pred6))


# y_ensemble1 = (y_pred1 + y_pred2 * 0.75) / 2
# y_ensemble2 = (y_pred3  + y_pred4 * 0.75) / 2

# print(rmspe(y_test1, y_ensemble1))
# print(rmspe(y_test2, y_ensemble2))

