import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


sub = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
sub


train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
train


test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
test


print("TRAIN shape:",train.shape)
print("TEST shape:",test.shape)
print("-------------------------")
print("Both have the same column length.")
print("*Differences in the number of rows are:",test.shape[0] - train.shape[0])
print("The test set has more row values.")


print("TRAIN \n",train.dtypes.value_counts(), "\n")
print("TEST \n",test.dtypes.value_counts())


print("TRAIN SET:",train.isnull().sum().value_counts())
print("TEST SET:",test.isnull().sum().value_counts())


train.index


test.index


train.index.duplicated().sum()


# Is the training dataset in the proper order?
print((train.index.to_series().diff().dt.total_seconds() < 0).sum())

# The test set is hidden, so we don't need to control it.


test.index.diff().value_counts()


train.index.diff().value_counts().sort_index()


total = 0

# How many time gaps totally we have?
def my_function(minute):
    global total
    summary = sum(train.index.diff() == pd.Timedelta(f"{minute}min"))
    total += summary * minute
    return total

for i in range(0,36):
    if i == 1:
        continue
    else:
        my_function(i)

print(f"We have '{total}' missing lines & minutes in total.")


# Let's mask it to see which part of time has a problem
mask = train.index.to_series().diff() != pd.Timedelta("1min")


train.index.to_series().diff()[mask]


train.loc['2023-04-08 08:29:00':'2023-04-08 08:31:00']


train.loc['2023-04-10 02:41:00':'2023-04-10 03:00:00']


plt.figure(figsize=(10,5))
plt.plot(train.index, train["bid_qty"], linestyle="-", linewidth=0.3)
plt.grid(True)
plt.show()


plt.figure(figsize=(10,5))
plt.plot(train.index, train["label"], linestyle="-", linewidth=0.3)
plt.grid(True)
plt.show()


train.isna().sum().value_counts()


full_index = pd.date_range(start=train.index.min(), end=train.index.max(), freq='T')


df_full = train.reindex(full_index)


rolling_means = df_full.rolling(window=12000, min_periods=1).mean()


train_filled = df_full.fillna(rolling_means)


print(train.shape)
print(1426 + train.shape[0])
train_filled.shape


train_filled.isna().sum().value_counts()


del train
del df_full
del full_index
del rolling_means


nan_rows = train_filled[train_filled.isna().any(axis=1)]
nan_rows.head()


nan_columns = train_filled.columns[train_filled.isna().any()].tolist()
print(nan_columns)


print(train_filled["X696"][0])
print(train_filled["X697"][0])
print(train_filled["X698"][0])
print(train_filled["X698"][1])


nan_rows["X697"]


nan_rows["X698"]


train_filled.dtypes.value_counts()


import sys
sys.float_info


print(1.8e308) 
print(-1.8e308)  


a = 1.8e308
print(a)


import numpy as np

a = 1.8e308         
b = np.float128(a)  

print(b)           


train_filled.drop(nan_columns, axis=1, inplace=True)
test.drop(nan_columns, axis=1, inplace=True)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


train_size = int(len(train_filled) * 0.8)
print((train_size))

train_set = train_filled.iloc[:train_size]
test_set = train_filled.iloc[train_size:]
print(train_set.shape)
print(test_set.shape)
print("------------- \n")

y_train = train_set['label']
X_train = train_set.drop('label', axis=1)
print(y_train.shape)
print(X_train.shape)
print("------------- \n")

y_test = test_set['label']
X_test = test_set.drop('label', axis=1)
print(y_test.shape)
print(X_test.shape)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


model = XGBRegressor(
    n_estimators=100,       
    learning_rate=0.05,     
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=1,    
    verbose=2                   
)


y_pred = model.predict(X_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")


from xgboost import plot_importance

# Show top 20 features by gain
plot_importance(model, importance_type='gain', max_num_features=20, height=0.5)
plt.title("XGBoost - Most Important 20 Features (Gain)")
plt.tight_layout()
plt.show()


plt.plot(y_test.values, label='Real')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("Real vs Predict")
plt.show()


test.drop(["label"], axis=1, inplace=True)


print(test.shape)
test.head()


predictions = model.predict(test)


print(len(sub))
print(len(predictions))


sub


sub["prediction"] = predictions


sub


sub.to_csv("submission.csv", index=False)

