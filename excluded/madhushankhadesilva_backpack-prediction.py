import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


train


test


train.describe()


desc = pd.DataFrame(index=train.columns.to_list())
desc["type"] = train.dtypes
desc["count"] = train.count()
desc["nunique"] = train.nunique()
desc["null"] = train.isnull().sum()
desc


train[train.isnull().any(axis=1)]


train.info(memory_usage='deep')


train.columns = train.columns.str.lower()
test.columns = test.columns.str.lower()


train.rename(columns={"weight capacity (kg)" : "weight_capacity_kg"}, inplace= True)
test.rename(columns={"weight capacity (kg)" : "weight_capacity_kg"}, inplace= True)

train.rename(columns={"laptop compartment" : "laptop_compartment"}, inplace= True)
test.rename(columns={"laptop compartment" : "laptop_compartment"}, inplace= True)



train = train.drop(columns = "id", axis=1)
test = test.drop(columns = "id", axis=1)



train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

# train.dropna(axis=0,inplace=True)
# test.dropna(axis=0,inplace=True)


train["compartments"] = train["compartments"].astype("int8")
test["compartments"] = test["compartments"].astype("int8")



train["laptop_compartment"].unique()
# train["waterproof"].unique()


from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_and_add(df, column):
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = one_hot_encoder.fit_transform(df[[column]])
    encoded_columns = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out([column]))
    encoded_columns.index = df.index
    df = pd.concat([df, encoded_columns], axis=1)
    df = df.drop(columns=[column])
    return df

columns_to_encode = ['laptop_compartment', 'waterproof']

for col in columns_to_encode:
    train = one_hot_encode_and_add(train, col)
    test = one_hot_encode_and_add(test, col)


# yes_no_map = {"No": 0, "Yes": 1, -1 : -1}
# train["laptop_compartment"] = train["laptop_compartment"].map(yes_no_map)
# train["waterproof"] = train["waterproof"].map(yes_no_map)


# train["laptop_compartment"] = train["laptop_compartment"].astype("int8")
# train["waterproof"] = train["waterproof"].astype("int8")


train.head()


train['color'].unique()


train['brand'] = pd.Categorical(train['brand'], categories=['Jansport', 'Under Armour', 'Nike', 'Adidas', 'Puma', 'nan'], ordered=True)
train['brand'] = train['brand'].cat.codes

train['material'] = pd.Categorical(train['material'], categories=['Leather', 'Canvas', 'Nylon', 'Polyester', "nan"], ordered=True)
train['material'] = train['material'].cat.codes

train['size'] = pd.Categorical(train['size'], categories=['Medium', 'Small', 'Large', "nan"], ordered=True)
train['size'] = train['size'].cat.codes

train['style'] = pd.Categorical(train['style'], categories=['Tote', 'Messenger', 'Backpack', "nan"], ordered=True)
train['style'] = train['style'].cat.codes

train['color'] = pd.Categorical(train['color'], categories=['Black', 'Green', 'Red', 'Blue', 'Gray', 'Pink', "nan"], ordered=True)
train['color'] = train['color'].cat.codes



test['brand'] = pd.Categorical(test['brand'], categories=['Jansport', 'Under Armour', 'Nike', 'Adidas', 'Puma', "nan"], ordered=True)
test['brand'] = test['brand'].cat.codes

test['material'] = pd.Categorical(test['material'], categories=['Leather', 'Canvas', 'Nylon', 'Polyester', "nan"], ordered=True)
test['material'] = test['material'].cat.codes

test['size'] = pd.Categorical(test['size'], categories=['Medium', 'Small', 'Large', "nan"], ordered=True)
test['size'] = test['size'].cat.codes

test['style'] = pd.Categorical(test['style'], categories=['Tote', 'Messenger', 'Backpack', "nan"], ordered=True)
test['style'] = test['style'].cat.codes

test['color'] = pd.Categorical(test['color'], categories=['Black', 'Green', 'Red', 'Blue', 'Gray', 'Pink', "nan"], ordered=True)
test['color'] = test['color'].cat.codes


train.describe()


all_numeric_cols = train.select_dtypes(include=[np.number]).columns
all_numeric_cols


import seaborn as sns
import matplotlib.pyplot as plt

columns_to_plot = all_numeric_cols

num_cols = 3
num_rows = int(np.ceil(len(columns_to_plot) / num_cols))

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 4 * num_rows)) 

axes = axes.flatten()

for i, column in enumerate(columns_to_plot):
    sns.boxplot(y=train[column], ax=axes[i])
    axes[i].set_title(f"Boxplot - {column}")
    axes[i].grid(False)

plt.tight_layout()
plt.show()


train.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


categorical_features = train.select_dtypes(include=['int8','int32', 'int64'] ).columns.to_list()
continuous_features = train.select_dtypes(include=['float32','float64']).columns.to_list()


# for i, col in enumerate(categorical_features): 
#     plt.subplot(5, 3, i+1) 
#     sns.countplot(data=train, x=col, hue='price') 
#     plt.title(col) 

# plt.subplots_adjust(hspace=0.5, wspace=0.3) 
# plt.show()


# plt.figure(figsize=(16, 4))
# for i, col in enumerate(continuous_features):
#     plt.subplot(5, 3, i+1)
#     sns.violinplot(data=train, x='price', y=col)
# plt.tight_layout()
# plt.show()


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

X = train.drop(columns=['price'], axis=1)  
y = train['price']

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)

train_X = pd.DataFrame(train_X_scaled, columns=train_X.columns, index=train_X.index)
val_X = pd.DataFrame(val_X_scaled, columns=val_X.columns, index=val_X.index)


from sklearn.metrics import mean_squared_error 
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_X, train_y)
y_pred = model.predict(val_X)
rmse = np.sqrt(mean_squared_error(val_y, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse}')


from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train_X, train_y)
y_pred = model.predict(val_X)
rmse = np.sqrt(mean_squared_error(val_y, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse}')


from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(train_X, train_y)
y_pred = model.predict(val_X)
rmse = np.sqrt(mean_squared_error(val_y, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse}')


from sklearn.metrics import mean_squared_error 
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(train_X, train_y)
y_pred = model.predict(val_X)
rmse = np.sqrt(mean_squared_error(val_y, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse}')


from sklearn.metrics import mean_squared_error 
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(train_X, train_y)
y_pred = model.predict(val_X)
rmse = np.sqrt(mean_squared_error(val_y, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse}')


# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential 
# from keras.layers import LSTM, Dense 

# scaler = MinMaxScaler(feature_range=(0, 1)) 
# scaled_data = scaler.fit_transform(train['price'].values.reshape(-1, 1))

# X_train, y_train = [], [] 
# for i in range(60, len(scaled_data)): 
#     X_train.append(scaled_data[i-60:i, 0])
#     y_train.append(scaled_data[i, 0]) 
# X_train, y_train = np.array(X_train), np.array(y_train)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# model = Sequential() 
# model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1))) 
# model.add(LSTM(units=50))
# model.add(Dense(units=1))

# model.compile(optimizer='adam', loss='mean_squared_error') 
# model.fit(X_train, y_train, epochs=100, batch_size=32) # Make predictions 
# test_data = scaled_data[-60:] 
# test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1)) 
# predictions = model.predict(test_data) 
# predictions = scaler.inverse_transform(predictions)


final_model = XGBRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)
predictions = final_model.predict(test)
predictions


submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
submission


submission['Price'] = predictions
submission


submission.to_csv("submission.csv", index=False)

