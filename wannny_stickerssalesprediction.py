# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df


df.isnull().sum()


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


print(train.shape, test.shape)


test.head()


train.isnull().sum()


train.head()


"""
    Preferably we wanted to map the nan value of num_sold
    with the mean of grouping of country, store, product.
    Unfortunatly some of it does not exist.
"""

grouped_train = train.groupby(['country','store','product'])
grouped_train['num_sold'].mean().isnull().sum()




# Replace NaN values with the mean of each group
train['num_sold'] = train.groupby(['country', 'store','product'])['num_sold'].transform(lambda x: x.fillna(x.mean()))



"""
    We fill the na rows of num_sold col
    with the mean of each group.
"""
train.isnull().sum()


"""
    fill the rest with the grouping of Country, product
"""
train['num_sold'] = train.groupby(['country','product'])['num_sold'].transform(lambda x: x.fillna(x.mean()))



train.isnull().sum()


# for (country, store), group_data in grouped_train:
#     print(f"Group: Country = {country}, Store = {store}")
#     print(group_data)
#     print("-" * 50)  # Separator for readability



train.head()


# Exporatory Data Analysis
print("Number of Unique Country: ", train['country'].nunique(),'\n')

print("Number of Unique Store: ", train['store'].nunique(),'\n')

print("Number of Unique Product: ", train['product'].nunique())


date = pd.to_datetime(train['date'])

sold_item = train['num_sold']

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(date, sold_item, marker='o', linestyle='-', color='b')

# Customize the plot
plt.title('Sales Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Sales', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()



# We could do dummies coding for each categorial column.
encoded_train = pd.get_dummies(train,columns=['country','store','product'],drop_first=True)
encoded_test = pd.get_dummies(test,columns=['country','store','product'],drop_first=True)


encoded_test


encoded_train


encoded_train.drop(columns=['id'],inplace = True)
encoded_test.drop(columns=['id'],inplace = True)


encoded_train


encoded_train.columns


# Extract date features
encoded_train['date'] = pd.to_datetime(encoded_train['date'])

encoded_train['year'] = encoded_train['date'].dt.year
encoded_train['month'] = encoded_train['date'].dt.month
encoded_train['day'] = encoded_train['date'].dt.day
encoded_train['day_of_week'] = encoded_train['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
encoded_train['is_weekend'] = encoded_train['day_of_week'].isin([5, 6]).astype(int)
encoded_train['week_of_year'] = encoded_train['date'].dt.isocalendar().week


encoded_test['date'] = pd.to_datetime(encoded_test['date'])

encoded_test['year'] = encoded_test['date'].dt.year
encoded_test['month'] = encoded_test['date'].dt.month
encoded_test['day'] = encoded_test['date'].dt.day
encoded_test['day_of_week'] = encoded_test['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
encoded_test['is_weekend'] = encoded_test['day_of_week'].isin([5, 6]).astype(int)
encoded_test['week_of_year'] = encoded_test['date'].dt.isocalendar().week


encoded_train.head()


encoded_test.head()


print(encoded_train['date'].dtype)  # Should display: datetime64[ns]



encoded_train.columns.to_list()


encoded_train.drop(columns=['date'],inplace=True)
encoded_test.drop(columns=['date'],inplace = True)

encoded_train.columns.to_list()


encoded_train.shape[1]
# Finish data cleaning with total of 18 columns


encoded_train['week_of_year'].unique()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler


X = encoded_train.drop(columns=['num_sold'])
Y = encoded_train['num_sold']
x_val = encoded_test


x_val


# Normalize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
x_val = pd.DataFrame(scaler.fit_transform(x_val), columns=x_val.columns)

# Initialize scaler
scaler_y = MinMaxScaler()

# Y = pd.DataFrame(scaler_y.fit_transform(Y),columns = Y.columns)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
x_val = np.array(x_val)

# Reshape data for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])

# Fit and transform the target
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))


X_test.shape, x_val.shape


X_test.dtype, x_val.dtype


import torch
import torch.nn as nn
import torch.optim as optim

class SalesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(SalesLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # LSTM output
        out, _ = self.lstm(x)
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Pass through the fully connected layer
        out = self.fc(out)
        return out

# Model parameters
input_dim = X_train.shape[2]
hidden_dim = 128
output_dim = 1
num_layers = 3

model = SalesLSTM(input_dim, hidden_dim, output_dim, num_layers)



loss_idx = []
loss_list = []


# Loss function and optimizer
# Define a function for MAPE
# def mean_absolute_percentage_error(y_pred, y_true):
#     epsilon = 1e-8
#     return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1)
x_val_tensor = torch.tensor(x_val,dtype=torch.float32)

# Training loop
num_epochs = 250
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    with torch.no_grad():
        loss_idx.append(epoch)
        loss_list.append(loss.item())

    # loss.append
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(loss_idx, loss_list, marker='o', linestyle='-', color='b')

# Customize the plot
plt.title('Loss', fontsize=16)
plt.xlabel('iterations', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()



y_train_scaled



y_test_scaled





# Evaluate on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    # y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    test_loss = criterion(y_pred, y_test_tensor)
    # print(y_pred)
    print(f'Test Loss: {test_loss.item():.4f}')

# Convert predictions to NumPy for further analysis
y_pred = y_pred.numpy()

# Rescaled our y prediction back.
y_pred_rescaled = scaler_y.inverse_transform(y_pred)


y_pred_rescaled[:5]


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test[100:200], label='Actual', marker='o')
plt.plot(y_pred_rescaled[100:200], label='Predicted', marker='x')
plt.legend()
plt.title('Actual vs Predicted Sales')
plt.xlabel('Sample Index')
plt.ylabel('Number of Sales')
plt.grid(True)
plt.show()



test.head()


y_val = model(x_val_tensor)


y_val = y_val.detach().numpy()


y_val_rescaled = scaler_y.inverse_transform(y_val)
y_val_rescaled


y_val_rescaled.astype(np.int64)
y_val_rescaled = y_val_rescaled.reshape(-1)


ids = test['id']
ids


# Write output
out = pd.DataFrame()
out['num_sold'] = pd.DataFrame({
    'num_sold':y_val_rescaled
})
out['id'] = test['id']



out.head()


submission = pd.DataFrame()
submission['id'] = out['id']
submission['num_sold'] = out['num_sold']


submission


submission.to_csv("/kaggle/working/submission3.csv",index=False)

