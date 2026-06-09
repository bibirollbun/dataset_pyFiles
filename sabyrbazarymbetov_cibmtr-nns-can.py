!python -m pip install -qq --no-index --find-links=/kaggle/input/library-for-cibmtr \
autogluon \
lifelines


from metric import score

from autogluon.tabular import TabularDataset, TabularPredictor

import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import TargetEncoder, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from xgboost import XGBRegressor
import xgboost as xgb

import time
import random


import torch
import torch.nn as nn
import torch.optim as optim

from eda_utility_library import categorize_columns, plot_pie_charts, violin_plots, missing_data_summary



pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# Read the input
train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

data_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')


# C-index utility to use with valid
def c_index(valid, preds):
    y_true = valid[['ID', 'efs', 'efs_time', 'race_group']].copy()
    y_pred = valid[['ID']].copy()
    y_pred['prediction'] = preds
    
    m = score(y_true, y_pred, 'ID')
    return m


def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    """
    Transform using survival probability estimates
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
    # Get survival probabilities at each time point
    y = kmf.survival_function_at_times(df[time_col]).values
    
    # Adjust for censoring
    # censored_mask = df[event_col] == 0
    #y[censored_mask] = y[censored_mask] * 1.2  # Increase survival prob for censored
    
    return y

train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')


RMV = ['ID', 'efs', 'efs_time', 'y']
BASIC = list(set(train.columns) - set(RMV))
print(f'There are {len(BASIC)} basic features: {BASIC}')


# Print the column types that exist in the given data
categorized_columns = categorize_columns(train, rmv=RMV)
for col_type in categorized_columns.keys():
    if len(categorized_columns[col_type]) > 0:
        print(col_type)


CATEGORICAL = categorized_columns['categorical']
DISCRETE = categorized_columns['discrete']
CONTINUOUS = categorized_columns['continuous']


# Treat Discrete as Categorical
for col in DISCRETE:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)


# Treat missing values as a new category
for col in CATEGORICAL+DISCRETE:
    train[col].fillna('NAN', inplace=True)
    test[col].fillna('NAN', inplace=True)


# OHE for Categorical and Discrete
ohe = OneHotEncoder(handle_unknown='error', sparse_output=False)
dummy = ohe.fit_transform(train[CATEGORICAL+DISCRETE])
OHE_COLUMNS = list(ohe.get_feature_names_out())

# Apply OHE to train and test
train[OHE_COLUMNS] = dummy
test[OHE_COLUMNS] = ohe.transform(test[CATEGORICAL+DISCRETE])


imputer = SimpleImputer(strategy='mean')

train[CONTINUOUS] = imputer.fit_transform(train[CONTINUOUS])
test[CONTINUOUS] = imputer.transform(test[CONTINUOUS])


scaler = RobustScaler()
train[CONTINUOUS] = scaler.fit_transform(train[CONTINUOUS])
test[CONTINUOUS] = scaler.transform(test[CONTINUOUS])


train_NN = train[OHE_COLUMNS+CONTINUOUS]
test_NN = test[OHE_COLUMNS+CONTINUOUS] 


X_train, X_val, y_train, y_val = train_test_split(train_NN, train['y'], test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)

y_train = y_train.view(-1, 1)
y_val = y_val.view(-1, 1)

X_test = torch.tensor(test_NN.values, dtype=torch.float32)



input_size = X_train.shape[1]


# NN Architechture 
class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Model, Loss, Optimizer
model = RegressionNN(input_size=input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_val)
    test_loss = criterion(y_pred, y_val)
    print(f"Test Loss: {test_loss.item():.4f}")


_, X_val, _, y_val = train_test_split(train, train['y'], test_size=0.2, random_state=42)
X_val_NN = X_val[OHE_COLUMNS+CONTINUOUS]
X_val_NN = torch.tensor(X_val_NN.values, dtype=torch.float32)

valid = X_val.copy()
preds= model(X_val_NN).detach().numpy().astype(np.float32).flatten()

m = c_index(valid, preds)

print(f'Val C-Index: {m}')


predictions = model(X_test)
predictions = predictions.detach().numpy().flatten()


ss = pd.DataFrame({'ID':test['ID'], 'prediction':predictions})
ss.to_csv('submission.csv', index=False)

