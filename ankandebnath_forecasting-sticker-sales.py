import numpy as np
import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


# Check for infinite values
print("\nInfinite values:\n", train.isin([np.inf, -np.inf]).sum())

# Check for NaN values
print("\nNaN values:\n", train.isna().sum())

# Select only numeric columns
numeric_columns = train.select_dtypes(include=[np.number]).columns

# Check for very large values in numeric columns
print("\nVery large values:\n", (train[numeric_columns].abs() > 1e6).sum())

# Check for very small values in numeric columns
print("\nVery small values:\n", (train[numeric_columns].abs() < 1e-6).sum())

# Describe the DataFrame
print("\nDataFrame description:\n", train.describe())


train.info()


train = train.dropna()
train


print('\nFor column `country`:')
print(train['country'].value_counts())

print('\nFor column `store`:')
print(train['store'].value_counts())

print('\nFor column `product`:')
print(train['product'].value_counts())


cols_to_encode = ['country', 'store', 'product']
train = pd.get_dummies(train, columns=cols_to_encode)

bool_cols = train.select_dtypes(include='bool').columns
train[bool_cols] = train[bool_cols].astype(int)

'''from category_encoders import BinaryEncoder

# Columns to encode
cols_to_encode = ['country', 'store', 'product']

# Initialize the BinaryEncoder
binary_encoder = BinaryEncoder(cols=cols_to_encode)

# Fit and transform the data
train = binary_encoder.fit_transform(train)'''


# Convert the 'date' column to datetime
train['date'] = pd.to_datetime(train['date'])

# Extract the 'Date', 'Month', and 'Year' columns
train['Date'] = train['date'].dt.day
train['Month'] = train['date'].dt.month
train['Year'] = train['date'].dt.year
train['DayOfWeek'] = train['date'].dt.dayofweek
train['Is_Weekend'] = train['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Cyclic encoding for 'Month'
train['Month_sin'] = np.sin(2 * np.pi * train['Month'] / 12)
train['Month_cos'] = np.cos(2 * np.pi * train['Month'] / 12)

# Cyclic encoding for 'Date'
train['Date_sin'] = np.sin(2 * np.pi * train['Date'] / 31)
train['Date_cos'] = np.cos(2 * np.pi * train['Date'] / 31)

# Cyclic encoding for 'DayOfWeek'
train['DayOfWeek_sin'] = np.sin(2 * np.pi * train['DayOfWeek'] / 7)
train['DayOfWeek_cos'] = np.cos(2 * np.pi * train['DayOfWeek'] / 7)

# Drop the original 'Date' and 'Month' columns if you no longer need them
train = train.drop(columns=['Date', 'Month'])

# Drop the original 'date' column
train = train.drop(columns=['date', 'id', 'DayOfWeek'])


train


import matplotlib.pyplot as plt
import seaborn as sns

df = train.copy()

# transformation
from scipy import stats

# Box-Cox transformation
df['num_sold_boxcox'], lambda_ = stats.boxcox(df['num_sold'] + 1)


# Plotting the distributions
plt.figure(figsize=(14, 6))

# Original distribution
plt.subplot(1, 2, 1)
sns.histplot(df['num_sold'], kde=True, bins=30)
plt.title('Distribution of num_sold (Original)')
plt.xlabel('num_sold')
plt.ylabel('Frequency')

# Log-transformed distribution
plt.subplot(1, 2, 2)
sns.histplot(df['num_sold_boxcox'], kde=True, bins=30)
plt.title('Distribution of num_sold (Box-Cox)')
plt.xlabel('---')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


lambda_


train['num_sold'] = np.log1p(train['num_sold'])


from sklearn.model_selection import train_test_split

X = train.drop(columns=['num_sold'])
y = train['num_sold']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

# XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
mse_xgb = mean_squared_error(y_val, y_pred_xgb)
print(f'XGBoost MSE: {mse_xgb}')

# LightGBM
lgb_model = lgb.LGBMRegressor(objective='regression', random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_val)
mse_lgb = mean_squared_error(y_val, y_pred_lgb)
print(f'LightGBM MSE: {mse_lgb}')

# CatBoost
catboost_model = CatBoostRegressor(verbose=0, random_state=42)
catboost_model.fit(X_train, y_train)
y_pred_catboost = catboost_model.predict(X_val)
mse_catboost = mean_squared_error(y_val, y_pred_catboost)
print(f'CatBoost MSE: {mse_catboost}')

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
mse_rf = mean_squared_error(y_val, y_pred_rf)
print(f'Random Forest MSE: {mse_rf}')


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Neural Network
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1))
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model with validation data and callbacks
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Load the best model
nn_model.load_weights('best_model.keras')

# Predict and evaluate
y_pred_nn = nn_model.predict(X_val)
mse_nn = mean_squared_error(y_val, y_pred_nn)
print(f'Neural Network MSE: {mse_nn}')



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
model = SimpleNN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_torch = model(X_val_tensor)
    mse_torch = mean_squared_error(y_val_tensor.numpy(), y_pred_torch.numpy())
    print(f'Neural Network (PyTorch) MSE: {mse_torch}')


# Train the final model
best_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
best_model.fit(X, y)

# Predict and evaluate
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f'Best Model MSE: {mse}')


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Check for infinite values
print("\nInfinite values:\n", test.isin([np.inf, -np.inf]).sum())

# Check for NaN values
print("\nNaN values:\n", test.isna().sum())

# Select only numeric columns
numeric_columns = test.select_dtypes(include=[np.number]).columns

# Check for very large values in numeric columns
print("\nVery large values:\n", (test[numeric_columns].abs() > 1e6).sum())

# Check for very small values in numeric columns
print("\nVery small values:\n", (test[numeric_columns].abs() < 1e-6).sum())

# Describe the DataFrame
print("\nDataFrame description:\n", test.describe())


test.info()


print('\nFor column `country`:')
print(test['country'].value_counts())

print('\nFor column `store`:')
print(test['store'].value_counts())

print('\nFor column `product`:')
print(test['product'].value_counts())


cols_to_encode = ['country', 'store', 'product']
test = pd.get_dummies(test, columns=cols_to_encode)

bool_cols = test.select_dtypes(include='bool').columns
test[bool_cols] = test[bool_cols].astype(int)

'''test['num_sold'] = 0
test = binary_encoder.transform(test)
test = test.drop(columns=['num_sold'])'''

# Convert the 'date' column to datetime
test['date'] = pd.to_datetime(test['date'])

# Extract the 'Date', 'Month', and 'Year' columns
test['Date'] = test['date'].dt.day
test['Month'] = test['date'].dt.month
test['Year'] = test['date'].dt.year
test['DayOfWeek'] = test['date'].dt.dayofweek
test['Is_Weekend'] = test['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Cyclic encoding for 'Month'
test['Month_sin'] = np.sin(2 * np.pi * test['Month'] / 12)
test['Month_cos'] = np.cos(2 * np.pi * test['Month'] / 12)

# Cyclic encoding for 'Date'
test['Date_sin'] = np.sin(2 * np.pi * test['Date'] / 31)
test['Date_cos'] = np.cos(2 * np.pi * test['Date'] / 31)

# Cyclic encoding for 'DayOfWeek'
test['DayOfWeek_sin'] = np.sin(2 * np.pi * test['DayOfWeek'] / 7)
test['DayOfWeek_cos'] = np.cos(2 * np.pi * test['DayOfWeek'] / 7)

# Drop the original 'Date' and 'Month' columns if you no longer need them
test = test.drop(columns=['Date', 'Month'])

# Drop the original 'date' column
temp = test.copy()
test = test.drop(columns=['date','id', 'DayOfWeek'])


test


#best_model = catboost_model
predictions = best_model.predict(test)
predictions = np.expm1(predictions)
#predictions = np.maximum(predictions, 0)

# Create the DataFrame
df = pd.DataFrame({
    'id': temp['id'],
    'num_sold': predictions.flatten()
})

# Save the DataFrame as a CSV file
df.to_csv('submission33.csv', index=False)

print("CSV file saved successfully.")


min(predictions)




