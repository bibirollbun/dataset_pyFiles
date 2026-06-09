# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are 1ble in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install --upgrade scikit-learn scikit-survival



import sklearn
import sksurv
print("scikit-learn version:", sklearn.__version__)
print("scikit-survival version:", sksurv.__version__)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel


# Set Seed for Reproducibility
SEED = 10
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



# Load Data
df_train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
df_test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")



train_data =pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_data =pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


# EDA & Data Quality Check
print(df_train.info())
print(df_train.describe())
print(df_train.isnull().sum())


# Identify numerical and categorical columns
numerical_cols = df_train.select_dtypes(include=['float64', 'int64']).columns.drop(['efs', 'efs_time'], errors='ignore')
categorical_cols = df_train.select_dtypes(include=['object']).columns

# Handle missing values for numerical features using median
df_train[numerical_cols] = df_train[numerical_cols].apply(lambda x: x.fillna(x.median()))
df_test[numerical_cols.intersection(df_test.columns)] = df_test[numerical_cols.intersection(df_test.columns)].apply(lambda x: x.fillna(x.median()))

# Handle missing values for categorical features using mode
df_train[categorical_cols] = df_train[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))
df_test[categorical_cols.intersection(df_test.columns)] = df_test[categorical_cols.intersection(df_test.columns)].apply(lambda x: x.fillna(x.mode()[0]))


print(df_train.isnull().sum().sum()) 
print(df_test.isnull().sum().sum())


from sklearn.preprocessing import LabelEncoder

# Define ordinal categorical features (these will use Label Encoding)
ordinal_features = ['tbi_status', 'dri_score']  # Add more if needed

# Apply Label Encoding
le = LabelEncoder()
for col in ordinal_features:
    df_train[col] = le.fit_transform(df_train[col])
    if col in df_test.columns:  # Ensure the column exists in df_test
        df_test[col] = le.transform(df_test[col])

# Apply One-Hot Encoding to nominal features
df_train = pd.get_dummies(df_train, columns=['ethnicity', 'graft_type'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['ethnicity', 'graft_type'], drop_first=True)

# Align columns for test set (ensure both datasets have the same features)
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

print("Encoding completed. Shape of train set:", df_train.shape)
print("Shape of test set:", df_test.shape)



from sklearn.preprocessing import LabelEncoder, StandardScaler

# Identify categorical columns (object type)
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()

# Apply Label Encoding to all categorical variables
le = LabelEncoder()
for col in categorical_cols:
    df_train[col] = le.fit_transform(df_train[col])
    if col in df_test.columns:  # Ensure column exists in test set
        df_test[col] = le.transform(df_test[col])

# Define numerical feature columns (excluding ID and target variables)
feature_cols = [col for col in df_train.columns if col not in ['ID', 'efs_time', 'efs']]

# Standardize numerical features
scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train[feature_cols])
df_test_scaled = scaler.transform(df_test[feature_cols])

# Convert back to DataFrame
df_train_scaled = pd.DataFrame(df_train_scaled, columns=feature_cols)
df_test_scaled = pd.DataFrame(df_test_scaled, columns=feature_cols)

print("Feature scaling completed successfully.")



from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# Convert survival labels into structured array
y_train_structured = np.array([(df_train['efs'].iloc[i], df_train['efs_time'].iloc[i]) 
                               for i in range(len(df_train))], dtype=[('event', '?'), ('time', '<f8')])

# Split into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(df_train_scaled, y_train_structured, test_size=0.2, random_state=42)

# Train Cox Model
cox_model = CoxPHSurvivalAnalysis().fit(X_train, y_train)

# Make Predictions
cox_preds = cox_model.predict(X_val)

# Evaluate with C-Index
cox_cindex = concordance_index_censored(y_val['event'], y_val['time'], cox_preds)[0]
print(f"Cox Model C-Index: {cox_cindex:.4f}")



from sksurv.ensemble import RandomSurvivalForest

# Train Random Survival Forest
rsf_model = RandomSurvivalForest(n_estimators=10, min_samples_split=10, min_samples_leaf=10, random_state=10)
rsf_model.fit(X_train, y_train)

# Make Predictions
rsf_preds = rsf_model.predict(X_val)

# Evaluate with C-Index
rsf_cindex = concordance_index_censored(y_val['event'], y_val['time'], rsf_preds)[0]
print(f"Random Survival Forest C-Index: {rsf_cindex:.4f}")



import xgboost as xgb

# Convert survival time into log-scale for better regression stability
y_train_xgb = np.log1p(df_train['efs_time'])

# Ensure alignment with X_train after the train-test split
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    df_train_scaled, y_train_xgb, test_size=0.2, random_state=42
)

# Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_xgb, y_train_xgb)

# Make Predictions
xgb_preds = xgb_model.predict(X_val_xgb)

# Evaluate with C-Index
xgb_cindex = concordance_index_censored(y_val['event'], y_val['time'], xgb_preds)[0]
print(f" XGBoost C-Index: {xgb_cindex:.4f}")






import torch
import torch.nn as nn
import torch.optim as optim
from sksurv.metrics import concordance_index_censored

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define DeepSurv Model
class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super(DeepSurv, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Convert structured array to standard NumPy array
y_train_time = np.array(y_train['time'].tolist(), dtype=np.float32)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train_time, dtype=torch.float32).view(-1, 1).to(device)

print(" Data successfully converted to PyTorch tensors.")

# Initialize Model
model = DeepSurv(X_train.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train Model
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Make Predictions
X_val_torch = torch.tensor(X_val.values, dtype=torch.float32).to(device)
deep_preds = model(X_val_torch).detach().cpu().numpy().flatten()

# Evaluate with C-Index
deep_cindex = concordance_index_censored(y_val['event'], y_val['time'], deep_preds)[0]
print(f"DeepSurv C-Index: {deep_cindex:.4f}")



# now lets try changing some values

class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super(DeepSurv, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)  
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)  
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)  
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.nn.functional.mish(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.nn.functional.mish(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.nn.functional.mish(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # Linear activation for risk scores
        return x
y_train_time = np.log1p(np.array(y_train['time'].tolist(), dtype=np.float32))


# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train_time, dtype=torch.float32).view(-1, 1).to(device)

print(" Data successfully converted to PyTorch tensors.")

# Initialize Model
model = DeepSurv(X_train.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

criterion = nn.MSELoss()

# Train Model
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Make Predictions
X_val_torch = torch.tensor(X_val.values, dtype=torch.float32).to(device)
deep_preds = model(X_val_torch).detach().cpu().numpy().flatten()

# Evaluate with C-Index
deep_cindex = concordance_index_censored(y_val['event'], y_val['time'], deep_preds)[0]
print(f"DeepSurv C-Index: {deep_cindex:.4f}")






import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# Convert survival labels into structured array
y_train_structured = np.array([(df_train['efs'].iloc[i], df_train['efs_time'].iloc[i]) 
                               for i in range(len(df_train))], dtype=[('event', '?'), ('time', '<f8')])

# Train Random Survival Forest on the FULL dataset
rsf_model = RandomSurvivalForest(n_estimators=10, min_samples_split=10, min_samples_leaf=5, random_state=10)
rsf_model.fit(df_train_scaled, y_train_structured)

# Make Predictions on Validation Set
rsf_preds = rsf_model.predict(X_val)

# Evaluate Final C-Index
rsf_cindex = concordance_index_censored(y_val['event'], y_val['time'], rsf_preds)[0]
print(f"Final RSF C-Index: {rsf_cindex:.4f}")



# Generate Test Predictions using the trained RSF Model
test_preds = rsf_model.predict(df_test_scaled)

# Create Submission File
submission = pd.DataFrame({"ID": df_test["ID"], "prediction": test_preds})
submission.to_csv("submission.csv", index=False)

print("Submission file saved as 'submission.csv'")





