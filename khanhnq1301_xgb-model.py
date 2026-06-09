# Importing necessary libraries
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit  # For time series cross-validation
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For feature scaling
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor

from joblib import dump, load
import time

import os  # For interacting with the operating system



# Load the training dataset with specified options
df_train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv', low_memory=False, parse_dates=["Date"])

# Load the store dataset
df_store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

# Merge the datasets on the 'Store' column using inner join
df_com = pd.merge(df_train, df_store, on='Store', how='inner')


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


col_name = 'CompetitionDistance'

plt.figure(figsize=(14, 6))  # Set width to fit both plots nicely

# Boxplot (left)
plt.subplot(1, 2, 1)
plt.boxplot(df_com[col_name])
plt.title(f'Boxplot of {col_name}')

# Histogram (right)
plt.subplot(1, 2, 2)
sns.histplot(df_com[col_name], kde=True)  # Assuming you meant df[col_name]
plt.title(f'Distribution of {col_name}')

plt.tight_layout()
plt.show()


df_com['CompetitionDistance'] = np.log(df_com['CompetitionDistance'])


plt.figure(figsize=(14, 6))  # Set width to fit both plots nicely

# Boxplot (left)
plt.subplot(1, 2, 1)
plt.boxplot(df_com[col_name])
plt.title(f'Boxplot of {col_name}')

# Histogram (right)
plt.subplot(1, 2, 2)
sns.histplot(df_com[col_name], kde=True)  # Assuming you meant df[col_name]
plt.title(f'Distribution of {col_name}')

plt.tight_layout()
plt.show()


length_data = df_com.shape[0]


df_rate_null = df_com[[ 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']]
rate = (df_rate_null.isna().sum() / length_data) * 100
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=rate.index, y=rate)
plt.xticks(rotation=90)
plt.ylabel("Missing Rate (%)")
plt.title("Rate")

# Thêm nhãn tỷ lệ trên đầu mỗi cột
for i, v in enumerate(rate):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()


# df_com = df_com.drop(columns = ['Promo2SinceWeek','Promo2SinceYear','PromoInterval'])


# fill nan by 0
df_com[['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']]= df_com[['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']].fillna(0)



print(df_com.shape[0])


df_com = df_com[df_com['Open'] == 1]
df_com = df_com.drop(columns = ['Open'])


len(df_com)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_com["Customers"], y=df_com["Sales"], alpha=0.5)

plt.xlabel("Customers")
plt.ylabel("Sales")
plt.title("Mối quan hệ giữa Customers  và Sales")
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


df_com.head()


categorical_cols = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']

df_com_onehot = pd.get_dummies(df_com, columns=categorical_cols, drop_first=True).astype(int)


mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
mapping_PromoInterval = {'0':0, 'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3,}
df_com_la = df_com.copy()


df_com_la['StoreType'] = df_com_la['StoreType'].map(mappings)
df_com_la['Assortment'] = df_com_la['Assortment'].map(mappings)
df_com_la['StateHoliday'] = df_com_la['StateHoliday'].map(mappings)
df_com_la['PromoInterval'] = df_com_la['PromoInterval'].map(mapping_PromoInterval)



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



from sklearn.preprocessing import MinMaxScaler

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



def try_model(model, X_train_scaled, y_train, X_val_scaled, y_val):
    # Fit the model
    start = time.time()
    model.fit(X_train_scaled, y_train)
    end = time.time()
    
    # Generate predictions
    train_preds = model.predict(X_train_scaled)
    val_preds = model.predict(X_val_scaled)
    
    # Compute RMSE
    train_rmse = rmspe(y_train,train_preds)
    val_rmse = rmspe(y_val,val_preds)
    print(f'Time of train: {end - start}s')
    print(f'MSE: {mean_squared_error(y_val,val_preds)}')
    return train_rmse, val_rmse 


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


from xgboost import XGBRegressor

model_onehot= XGBRegressor()
try_model(model_onehot, X_train_scaled_onehot, y_train_onehot,X_val_scaled_onehot,y_val_onehot  )


(X_train_la,y_train_la ), (X_val_la,y_val_la ), (X_test_la,y_test_la ) = split_data(df_com_la)


X_train_scaled_la, X_val_scaled_la, X_test_scaled_la = normalize_data(X_train_la,X_val_la, X_test_la ) 



model_la= XGBRegressor()
try_model(model_la, X_train_scaled_la, y_train_la,X_val_scaled_la,y_val_la  )


(X_train_es,y_train_es ), (X_val_es,y_val_es ), (X_test_es,y_test_es ) = split_data(df_encode)


X_train_scaled_es, X_val_scaled_es, X_test_scaled_es = normalize_data(X_train_es, X_val_es, X_test_es ) 



model_es= XGBRegressor()
try_model(model_es, X_train_scaled_es, y_train_es,X_val_scaled_es,y_val_es  )


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


df_xgb = df_com_la.drop(columns = [ 'StateHoliday', 'Year'])
# df_xgb = df_com_la.copy()




(X_train1,y_train1 ), (X_val1,y_val1 ), (X_test1,y_test1 ) = split_data(df_xgb)
 


X_train_scaled1, X_val_scaled1, X_test_scaled1 = normalize_data(X_train1,X_val1,X_test1  ) 


model_check= XGBRegressor()
try_model(model_check, X_train_scaled1, y_train1,X_val_scaled1,y_val1 )


param_grid = {
    'learning_rate': [0.01, 0.1, 0.05],
    'max_depth': [ 7, 8, 9],
    'n_estimators': [150, 200],
    'gamma': [0, 0.1],
    'min_child_weight': [2, 6,7],
    "device" : ["cuda"]
}



from sklearn.model_selection import TimeSeriesSplit
import time

# Time series split
tscv = TimeSeriesSplit(n_splits=5)

# Custom RMSPE scoring
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

model_xgb = XGBRegressor(
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_grid,
    scoring=rmspe_scorer,
    cv=tscv,
    verbose=0,
)

# Fit
grid_search.fit(X_train_scaled_la, y_train_la)

# Best model
best_model_xgb = grid_search.best_estimator_

print("Best parameters found: ", grid_search.best_params_)

# ✨ Predict
y_pred_xgb = best_model_xgb.predict(X_val_scaled_la)
print("RMSPE score on validation:", rmspe(y_val_la, y_pred_xgb))
print(f"MSE: {mean_squared_error(y_val_la, y_pred_xgb)}")


def rmspe_xgboost(y_pred, dtrain):
    y_true = dtrain.get_label()
    # Avoid division by zero
    non_zero_idx = y_true != 0
    if np.any(non_zero_idx):
        error = rmspe(y_true[non_zero_idx], y_pred[non_zero_idx])
    else:
        error = 0.0
    return 'RMSPE', error



# Tạo DMatrix cho XGBoost
dtrain = xgb.DMatrix(X_train_scaled_la, label=y_train_la)
dval = xgb.DMatrix(X_val_scaled_la, label=y_val_la)

# Bộ tham số XGBoost chuẩn để bắt đầu
params = {
    'objective': 'reg:squarederror',    # bài toán regression
    'learning_rate': 0.1,               # small learning rate
    'max_depth': 9,                      # cây sâu vừa phải
    'min_child_weight': 2,                # giảm overfitting
    'gamma': 0,                         # thêm regularization
}

# Train với Early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=10000,
    evals=[(dtrain, 'train'), (dval, 'eval')],
    feval=rmspe_xgboost,
    early_stopping_rounds=100,
    verbose_eval=100
)

# Predict
y_pred = model.predict(dval)

score = rmspe(y_val_la, y_pred)
print(f'RMSPE: {score:.5f}')
print(f"MSE: {mean_squared_error(y_val_la, y_pred_xgb)}")


start_time = time.time()
dtest = xgb.DMatrix(X_test_scaled_la, label=y_test_la)
y_pred_test = model.predict(dtest)
end_time = time.time()

print(f'Time for predict: {end_time - start_time}s')
print(f'RMSPE: {rmspe(y_test_la, y_pred_test)}')
print(f'MSE: {mean_squared_error(y_test_la, y_pred_test)}')
print(f'R2: {r2_score(y_test_la, y_pred_test)}')

import pickle

model_bytes = pickle.dumps(model)
model_size = len(model_bytes)

print(f"Kích thước mô hình: {model_size / 1024 ** 2:.2f} MB")



np.save('y_hat_XGB',y_pred_test)


import joblib
joblib.dump(model, 'model_XGBoost.joblib')

