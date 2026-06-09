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


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import polars as pl
%matplotlib inline


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")



# train = pl.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
# test = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
# sample_submission = pl.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")


train.head(5)


test.head(5)


train.replace([np.inf, -np.inf], np.nan, inplace=True)  # for Pandas

# # Replace inf and -inf with null (NaN equivalent)
# train = train.drop(['timestamp'])

# train = train.with_columns([
#     pl.when(pl.col(col).is_infinite()).then(None).otherwise(pl.col(col)).alias(col)
#     for col in train.columns
# ])

# train = train.fill_null(strategy="zero")  # or "zero", "forward", etc.


train.isnull().sum().sort_values(ascending=False)

#train.select(pl.all().null_count())


null_cols = train.isnull().sum().sort_values(ascending=False)[lambda x: x >0].index
null_cols

train.drop(columns=null_cols, inplace=True)
train.shape


# train = train[:300000]
# train.shape


X = train.drop(columns=['label']) #for pandas
# X = train.drop('label')
y = train['label']
X.shape


from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=200)
X_selected = selector.fit_transform(X, y)

mask = selector.get_support()

# Get names of selected features
selected_features = X.columns[mask]
print(selected_features)


cols_to_keep = ['bid_qty','ask_qty','buy_qty','sell_qty','volume'] + list(selected_features)

X = X[cols_to_keep]


X = X[:200000]

y = y[:200000]

X.shape


#y


type(X)


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

print("Shape of X_train:",X_train.shape)
print("Shape of X_test:",X_test.shape)

print("Shape of y_train:",y_train.shape)
print("Shape of y_test:",y_test.shape)


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer


numeric_cols = [fea for fea in X.columns if X[fea].dtype != '0']
len(numeric_cols)


scaler = MinMaxScaler()


transformer = ColumnTransformer(
    transformers =[
        ('standard_scalling' , scaler, numeric_cols),
    ], remainder = 'passthrough'
)

X_train_tr = transformer.fit_transform(X_train)
X_test_tr = transformer.transform(X_test)


type(X_train_tr)


#test=test.drop(columns=['label']) ## dropping target feature from test dataframe
test = test[cols_to_keep]
test.head(5)
test_trf = transformer.transform(test)


## Model Training and Model Selection
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error


## Creating a function to evaluat model
def evaluation(true, predicted):
    mae = mean_absolute_error(true, predicted)
    me = mean_squared_error(true, predicted)
    mse = np.sqrt(me)
    r2 = r2_score(true, predicted)

    r = np.corrcoef(true, predicted)[0,1]

    print(f"Pearson Correlation Coefficient: {r}")
    print("R2 Score:{:.4f}".format(r2))
    print("MAE:{:.4f}".format(mae))
    print("MSE:{:.4f}".format(mse))
    #print("RMSE:{:.4f}".format(rmse))
    

    return 0


## Model Training and Model Selection

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost 
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor


## Model training
models={
    # "Linear_Regression":LinearRegression(),

    # "Linear_Regression_with_params": LinearRegression(
    #                     fit_intercept=True,                 
    #                     copy_X=True,              
    #                     n_jobs=-1,                
    #                     positive=False            
    #                     ),
    
    # "Lasso":Lasso(),
    
    # "Ridge":Ridge(),
    
    # "ElasticNet":ElasticNet(),
    
    # "DecisionTreeRegressor":DecisionTreeRegressor(),
    
    # "DecisionTreeRegressor_with_params":DecisionTreeRegressor(
    #                                     criterion='squared_error',   
    #                                     splitter='best',             
    #                                     max_depth=10,                
    #                                     min_samples_split=10,       
    #                                     min_samples_leaf=4,         
    #                                     max_features='sqrt',        
    #                                     random_state=42             
    #                                     ),
    
    # "AdaBoost":AdaBoostRegressor(),
    
    # "GradientBoost":GradientBoostingRegressor(),
    
    "XGBRegressor":XGBRegressor(
                    max_depth=20,
                    colsample_bytree=0.75,
                    subsample=0.9,
                    n_estimators=1000,
                    learning_rate=0.001,
                    gamma=0.01,
                    max_delta_step=2,
                    eval_metric="rmse",
                    enable_categorical=True,
                    device = 'cuda'),
    
    # "LGBMRegressor":LGBMRegressor(
    #                 n_estimators=1000,
    #                 learning_rate=0.005,
    #                 max_depth=7,
    #                 num_leaves=31,
    #                 min_child_samples=20,
    #                 subsample=0.8,
    #                 colsample_bytree=0.8,
    #                 random_state=42,
    #                 n_jobs=-1
    #                 ),
    
    # "CatBoostRegressor":CatBoostRegressor(
    #                     iterations= 3500,
    #                     depth= 12,
    #                     loss_function= 'RMSE',
    #                     l2_leaf_reg= 3,
    #                     random_seed= 42,
    #                     eval_metric= 'RMSE',
    #                     silent=True
    #                     ),
    
    # "RandomForest":RandomForestRegressor(
    #                 n_estimators=300,
    #                 max_depth=10,
    #                 min_samples_split=5,
    #                 min_samples_leaf=2,
    #                 max_features='sqrt',
    #                 random_state=42,
    #                 n_jobs=-1
    #                 ),
}



sample_submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
#sample_submission = sample_submission[:30000]
id_column = sample_submission['ID']
len(id_column)


# model_name_list = []
# corrcoef_list = []

# for i in range(len(list(models))):
#     model_name = list(models.keys())[i]
#     model = list(models.values())[i]

#     print(".............",model_name, "........................\n")

#     model.fit(X_train_tr, y_train) ## Train Model on X_train

#     ## Make Predictions.............................
#     y_train_pred=model.predict(X_train_tr)
#     y_test_pred=model.predict(X_test_tr)

#     print()
#     print("Evaluating Train Dataset")
#     evaluation(y_train,y_train_pred)

#     print(f"\n{'-'*50}\n")
    
#     print("Evaluating Test Dataset")
#     evaluation(y_test,y_test_pred)
#     print("="*60)
#     print("\n")

    
#     ### appending the vlaues in list 
#     model_name_list.append(model_name)
#     corrcoef_list.append(np.corrcoef(y_test, y_test_pred)[0, 1])

#     ## prediction
#     prediction = model.predict(test_trf)

#     result = pd.DataFrame(
#     {
#         'ID':id_column,
#         'prediction':prediction
#     }
#     )
 
#     result.to_csv('submission.csv',index=False)
#     print("File saved as '{}_prediction.csv'....".format(model_name))
#     print()



# ## creating dataframe contains model name and their performance on X_test 
# performance_df = pd.DataFrame({
#     'ML Algo Name': model_name_list,
#     'Pearson Correlation Coefficient': corrcoef_list
# })

# performance_df


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# X.info()


# Sparse matrix কে dense numpy array তে রূপান্তর
# X_dense = X.toarray()
# X_test_dense = test.toarray()

X_tensor = torch.tensor(X_train_tr, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

X_test_tensor = torch.tensor(X_test_tr, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

#X_train, X_valid, y_train, y_valid = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)


# Create tensor dataset..............
train_ds = TensorDataset(X_tensor,y_tensor)
valid_ds = TensorDataset(X_test_tensor,y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False)


class NN_model(nn.Module):
    def __init__(self, input_size):
        super(NN_model, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,1)


        self.dp = nn.Dropout(p=0.35)  # Regularization........
        self.fc_drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.dp(x)
        x = F.relu(self.fc2(x))
        #x = self.dp(x)
        x = F.relu(self.fc3(x))
       # x = self.fc_drop(x)
        x = F.relu(self.fc4(x))
        
        return self.fc5(x)
    


model = NN_model(input_size=X_tensor.shape[1])
model = model.to(device)

#summary(model)
print(model.parameters())
criterian = nn.MSELoss()
#criterian = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)


# training the model..............................
train_losses =[]
val_losses = []
best_val_loss = float('inf')
epochs = 70

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


#len(id_column)


test_trf_tensor = torch.tensor(test_trf, dtype=torch.float32)
type(test_trf_tensor)


len(test_trf_tensor)


model.eval()
with torch.no_grad():
    X_test_tensor = test_trf_tensor.to(device)
    test_pred = model(X_test_tensor).squeeze()



test_preds = test_pred.cpu().numpy()

submission = pd.DataFrame({
    'ID' : id_column,
    'prediction' : test_preds
})

submission.to_csv('submission.csv', index=False)

submission.head(5)


#test_preds


#y_test


# evaluation(y_test.values, test_preds)

