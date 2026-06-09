# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
np.seterr(invalid='ignore')


df_train=pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")
df_sample=pd.read_csv("/kaggle/input/playground-series-s4e12/sample_submission.csv")
df_train.head()


df_train.columns


df_train.isnull().sum()


df_train.shape


df1=df_train.dropna()
df1.shape


(1200000-384004)/1200000


df_train.dtypes


df_train.columns=["_".join(i.split()).lower() for i in df_train.columns]


df_cat=df_train.select_dtypes(include=["object"])
df_num=df_train.select_dtypes(exclude=["object"])


df_cat.dtypes.index


df_cat.isnull().sum()[df_cat.isnull().sum()>0]


df_cat["marital_status"].value_counts(dropna=False)


df_cat["occupation"].value_counts(dropna=False)


# customer_feedback
df_cat["customer_feedback"].value_counts(dropna=False)


df_cat.loc[:, 'marital_status'] = df_cat['marital_status'].fillna('unknown')
# Define the probabilities
probs = df_cat['occupation'].value_counts(normalize=True)
# Sample based on probabilities
df_cat['occupation'] = df_cat['occupation'].apply(
    lambda x: np.random.choice(probs.index, p=probs.values) if pd.isnull(x) else x
)

probs = df_cat['customer_feedback'].value_counts(normalize=True)
# Sample based on probabilities
df_cat['customer_feedback'] = df_cat['customer_feedback'].apply(
    lambda x: np.random.choice(probs.index, p=probs.values) if pd.isnull(x) else x
)


df_cat["policy_start_date"] = pd.to_datetime(df_cat["policy_start_date"])

df_cat["year"] = df_cat["policy_start_date"].dt.year
df_cat["month"] = df_cat["policy_start_date"].dt.month
df_cat["day"] = df_cat["policy_start_date"].dt.day
df_cat["day_of_week"] = df_cat["policy_start_date"].dt.dayofweek  # Monday=0, Sunday=6
df_cat["week_of_year"] = df_cat["policy_start_date"].dt.isocalendar().week
df_cat["quarter"] = df_cat["policy_start_date"].dt.quarter



df_cat["is_weekend"] = df_cat["policy_start_date"].dt.weekday >= 5  # 1 for Sat/Sun, 0 for others



from pandas.tseries.holiday import USFederalHolidayCalendar

cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df_cat["policy_start_date"].min(), end=df_cat["policy_start_date"].max())

df_cat["is_holiday"] = df_cat["policy_start_date"].isin(holidays)



df_cat["month_sin"] = np.sin(2 * np.pi * df_cat["month"] / 12)
df_cat["month_cos"] = np.cos(2 * np.pi * df_cat["month"] / 12)



df_cat.isnull().sum()


df_num.isnull().sum()[df_num.isnull().sum()>0]


df_num['age']=df_num['age'].fillna(df_num['age'].median())  # Median is better for skewed distributions
df_num['annual_income']=df_num['annual_income'].fillna(df_num['annual_income'].mean())
df_num['number_of_dependents']=df_num['number_of_dependents'].fillna(df_num['number_of_dependents'].mode()[0])
df_num['health_score']=df_num['health_score'].fillna(df_num['health_score'].median())
df_num['previous_claims']=df_num['previous_claims'].fillna(0)
df_num['vehicle_age']=df_num['vehicle_age'].fillna(df_num['vehicle_age'].mode()[0])
df_num['credit_score']=df_num['credit_score'].fillna(df_num['credit_score'].mean())
df_num['insurance_duration']=df_num['insurance_duration'].fillna(df_num['insurance_duration'].mode()[0])


df_train0=pd.concat([df_cat, df_num], axis=1)
df_train0.shape


# Assuming df is your DataFrame
categorical_cols = df_train0.select_dtypes(include=['object', 'category']).columns
label_encoders = {}  # Dictionary to store encoders for future use

for col in categorical_cols:
    le = LabelEncoder()
    df_train0[col] = le.fit_transform(df_train0[col])  # Transform categorical values into numerical labels
    label_encoders[col] = le  # Store the encoder for inverse transformation if needed later



df_train0.dtypes


df_train0.drop(columns=["policy_start_date"], inplace=True)


def detect_outliers_iqr(df, columns):
    outlier_indices = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices[col] = outliers

    return outlier_indices

# Columns to check for outliers (excluding categorical/binary)
numerical_cols = [
    "age", "annual_income", "number_of_dependents", "health_score",
    "previous_claims", "vehicle_age", "credit_score", "insurance_duration",
    "premium_amount"
]

outliers_iqr = detect_outliers_iqr(df_train0, numerical_cols)
outliers_iqr



from scipy import stats

def detect_outliers_zscore(df, columns, threshold=3):
    outlier_indices = {}
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = df[z_scores > threshold].index
        outlier_indices[col] = outliers
    return outlier_indices

outliers_zscore = detect_outliers_zscore(df_train0, numerical_cols)
outliers_zscore



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --------------- Identify Features ---------------
# Categorical (label-encoded)
categorical_cols = ["year", "month", "day", "day_of_week", "week_of_year", "quarter", "is_weekend", "is_holiday"]

# Continuous numerical columns (to be standardized)
numerical_cols = ["age", "annual_income", "number_of_dependents", "health_score",
                  "previous_claims", "vehicle_age", "credit_score", "insurance_duration"]

all_numerical_cols=categorical_cols+numerical_cols
# all_numerical_cols
target_col="premium_amount"


# ðŸ”¹ Standardize all numerical features
scaler = StandardScaler()
df_train0[all_numerical_cols] = scaler.fit_transform(df_train0[all_numerical_cols])

# ðŸ”¹ Split Data
X = df_train0[all_numerical_cols].values
y = df_train0[target_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



y_train


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


X_train.shape


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # Batching and shuffling for training
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)  # No shuffling for evaluation




# Define the neural network model with more layers
class ComplexRegressionNN(nn.Module):
    def __init__(self):
        super(ComplexRegressionNN, self).__init__()
        self.layer1 = nn.Linear(16, 128)  # Input to 1st hidden layer
        self.batch_norm1 = nn.BatchNorm1d(128)  # Batch Normalization
        self.dropout1 = nn.Dropout(0.3)  # Dropout to prevent overfitting
        
        self.layer2 = nn.Linear(128, 256)  # 1st to 2nd hidden layer
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.layer3 = nn.Linear(256, 512)  # 2nd to 3rd hidden layer
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.3)
        
        self.layer4 = nn.Linear(512, 256)  # 3rd to 4th hidden layer
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.3)
        
        self.layer5 = nn.Linear(256, 128)  # 4th to 5th hidden layer
        self.batch_norm5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.3)
        
        self.output_layer = nn.Linear(128, 1)  # 5th hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.layer1(x)))  # ReLU + BatchNorm1
        x = self.dropout1(x)  # Dropout

        x = torch.relu(self.batch_norm2(self.layer2(x)))  # ReLU + BatchNorm2
        x = self.dropout2(x)  # Dropout
        
        x = torch.relu(self.batch_norm3(self.layer3(x)))  # ReLU + BatchNorm3
        x = self.dropout3(x)  # Dropout
        
        x = torch.relu(self.batch_norm4(self.layer4(x)))  # ReLU + BatchNorm4
        x = self.dropout4(x)  # Dropout
        
        x = torch.relu(self.batch_norm5(self.layer5(x)))  # ReLU + BatchNorm5
        x = self.dropout5(x)  # Dropout
        
        x = self.output_layer(x)  # Output layer (no activation for regression)
        return x



# Instantiate the model and move it to GPU if available
model = ComplexRegressionNN().to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model with DataLoader on GPU
num_epochs = 50
for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    for inputs, labels in train_loader:
        # Move inputs and labels to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        # Move inputs and labels to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}')


import torch
import numpy as np

# Assuming the model is already trained and you're using the same device (GPU/CPU)
# Assuming you've defined and fitted a scaler (e.g., StandardScaler) for scaling the input features

# Step 1: Prepare a single input for prediction (16 features)
# Replace these values with the actual input values for prediction
single_input = np.array([[25, 50000, 2, 80, 1, 5, 750, 3, 2022, 12, 15, 2, 50, 3, 1, 0]])  # Example

# Step 2: Standardize the input using the same scaler used during training
single_input_scaled = scaler.transform(single_input)  # Apply the same scaling as training data

# Step 3: Convert the input to a PyTorch tensor
single_input_tensor = torch.tensor(single_input_scaled, dtype=torch.float32)

# Step 4: Move the input tensor to the same device (GPU/CPU) as the model
single_input_tensor = single_input_tensor.to(device)

# Step 5: Set the model to evaluation mode (important to disable dropout and batch norm)
model.eval()

# Step 6: Perform prediction (without gradient calculation)
with torch.no_grad():
    prediction = model(single_input_tensor)

# Step 7: Convert the prediction to CPU and then to NumPy for easier interpretation
prediction = prediction.cpu().numpy()

# Step 8: Print the prediction
print(f"Prediction for input {single_input[0]}: {prediction[0][0]:.4f}")



prediction


df_test




