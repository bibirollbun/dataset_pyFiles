import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
print(f'train dataset {train_df.shape}')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
print(f'test dataset {test_df.shape}')
extra_train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
print(f'Extra training dataset {extra_train_df.shape}')                     


train_df.head(5)


train_df.isnull().sum()


test_df.isnull().sum()


train_df.info()


numeric_col = train_df.drop(columns=['id','Price']).select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_col = train_df.drop(columns=['id','Price']).select_dtypes(include=['object']).columns.tolist()


numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessing into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_col),
        ('cat', categorical_transformer, categorical_col)
    ]
)


X = train_df.drop(columns=['id','Price'])
y = train_df['Price']

X = preprocessor.fit_transform(X)

X_test = test_df.drop(columns=['id'])
X_test = preprocessor.transform(X_test)

# encoded_columns = numeric_col.tolist() + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_col))

# X = pd.DataFrame(X, columns=encoded_columns)
# X_test = pd.DataFrame(X_test, columns=encoded_columns)

X.shape


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.2)

print(X_train.shape)


randomReg = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=3, ccp_alpha=0.5, random_state=42)


randomReg.fit(X_train,y_train)


X_pred = randomReg.predict(X_val)
rmse = mean_squared_error(y_val, X_pred,  squared=False)
print(f'RMSE : {rmse:.2f}')


test_pred = randomReg.predict(X_test)


tree_reg = DecisionTreeRegressor(splitter='best', random_state=42, max_depth=10, min_samples_leaf=3)


tree_reg.fit(X_train, y_train)


X_predict = tree_reg.predict(X_val)


rmse_2 = mean_squared_error(y_val, X_predict,  squared=False)
print(f'RMSE : {rmse_2:.2f}')


test_pred_2 = tree_reg.predict(X_test)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# Sparse matrix কে dense numpy array তে রূপান্তর
X_dense = X.toarray()
X_test_dense = X_test.toarray()

X_tensor = torch.tensor(X_dense, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1,1)

X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32)

X_train, X_valid, y_train, y_valid = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create tensor dataset..............
train_ds = TensorDataset(X_train,y_train)
valid_ds = TensorDataset(X_valid,y_valid)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)


class Backpack_NN(nn.Module):
    def __init__(self, input_size):
        super(Backpack_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,32)
        self.fc4 = nn.Linear(32,1)

        self.dp = nn.Dropout(p=0.35)  # Regularization........
        self.fc_drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = F.relu(self.fc2(x))
        x = self.fc_drop(x)
        x = F.relu(self.fc3(x))
        
        return self.fc4(x)
    


model = Backpack_NN(input_size=X_tensor.shape[1])
model = model.to(device)


criterian = nn.MSELoss()
#criterian = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)


train_losses =[]
val_losses = []
best_val_loss = float('inf')
epochs = 30

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for x, y in train_loader:
        x , y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterian(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss+= loss
    
    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)


    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for xv, yv in valid_loader:
            xv, yv = xv.to(device), yv.to(device)

            val_pred = model(xv)
            loss_val = criterian(val_pred, yv)
            running_val_loss+= loss_val

    valid_loss = running_val_loss / len(valid_loader)
    val_losses.append(valid_loss)

    print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")


model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    test_preds = torch.tensor(model(X_test_tensor).squeeze())


test_preds = test_preds.cpu().numpy()


submission = pd.DataFrame({
    'id' : test_df['id'],
    'Price' : test_preds
})

submission.to_csv('submission.csv', index=False)

submission.head(5)







