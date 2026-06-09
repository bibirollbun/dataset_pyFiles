# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s4e7/train.csv")
df_train.shape


df_train.head(3)


df_train.info()


df_train.Policy_Sales_Channel.value_counts()


df_train.Vehicle_Age.value_counts()


df_train.Response.value_counts()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in df_train.columns:
    if df_train[col].dtype == 'object' :
        df_train[col] = le.fit_transform(df_train[col])


df_train.head(3)


cols_to_scale = ['Age','Region_Code' ,'Annual_Premium','Policy_Sales_Channel','Vintage']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Step 2: Fit and transform only the selected columns
scaled_values = scaler.fit_transform(df_train[cols_to_scale])

# Step 3: Convert the scaled values back to a DataFrame
scaled_df = pd.DataFrame(scaled_values, columns=cols_to_scale, index=df_train.index)

# Step 4: Replace original columns with scaled ones
df_train[cols_to_scale] = scaled_df


df_train.head(3)


df_train.columns


from sklearn.model_selection import train_test_split, cross_val_score

X=df_train[['Gender', 'Age', 'Driving_License', 'Region_Code',
       'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
       'Policy_Sales_Channel', 'Vintage']].values
y=df_train[['Response']].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=9)


X_train.shape


import torch
import torch.nn as nn
import torch.nn.functional as F

# Seed for reproducibility
torch.manual_seed(0)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


class InsuranceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)


import torch.optim as optim
model = InsuranceClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 1000

for epoch in range(epochs):
    # Forward pass: Compute predicted values
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)  # <-- fixed line

    # Backward pass: Compute gradients
    optimizer.zero_grad()
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.2f}")


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure model is in eval mode and no gradient tracking
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)

    # Apply sigmoid since using BCEWithLogitsLoss (for binary classification)
    outputs = torch.sigmoid(outputs)

    # Apply threshold of 0.5
    predicted = (outputs > 0.5).int()

# Flatten if output shape is (N, 1)
y_pred = predicted.view(-1).cpu().numpy()
y_true = y_test_tensor.view(-1).cpu().numpy()

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred))


cm = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

