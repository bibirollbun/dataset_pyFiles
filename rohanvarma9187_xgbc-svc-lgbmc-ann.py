from tqdm import tqdm
from scipy.stats import randint, uniform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

import xgboost as xgb

import lightgbm as lgb
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"

df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df


df.isna().sum()


df_test.isna().sum()


df_test['winddirection'] = df['winddirection'].fillna(df_test['winddirection'].mean())
df_test.isna().sum()


df.corr()['rainfall']


X = df.drop('rainfall', axis=1)
y = df['rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y)
len(X_train), len(X_test), len(y_train), len(y_test)


y.value_counts()


def fit_and_eval_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    pred_probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, pred_probs)



def submit_model(model):
    pred_probs_submission = model.predict_proba(df_test)[:, 1]
    submission = pd.DataFrame(
        {
            'rainfall' : pred_probs_submission
        },
        index = df_test['id']
    )
    print(submission.head())
    submission.to_csv('/kaggle/working/submission.csv')
    return submission


params = {
    'n_estimators': randint(50, 301),          
    'learning_rate': uniform(0.01, 0.5),     
    'max_depth': randint(4, 9),                 
    'subsample': uniform(0.8, 0.2),             
    'reg_alpha': uniform(0.1, 1),               
    'reg_lambda': uniform(0.1, 1),              
}

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss'
)

rs_model = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,  
    n_iter=100,                                
    cv=3,                                     
    scoring='roc_auc',                        
    verbose=1,                               
    n_jobs=-1,                                
    random_state=42                           
)

rs_model.fit(X_train, 
             y_train)



best_params = rs_model.best_params_
best_params


model1 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    **best_params
)

fit_and_eval_model(model1, X_train, y_train)


# sub1 = submit_model(model1)


model2 = SVC(
    kernel = 'rbf',
    probability = True,
    verbose = 1,
    C = 7,
)
fit_and_eval_model(model2, X_train, y_train)


# sub2 = submit_model(model2)


params = {
    'objective': 'binary',  
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.4,
    'feature_fraction': 0.9
}

train_data = lgb.Dataset(X_train, y_train)
test_data = lgb.Dataset(X_test, y_test)
model3 = lgb.train(
    params, 
    train_data, 
    num_boost_round=100, 
    valid_sets=[test_data], 
    callbacks=[lgb.early_stopping(stopping_rounds=10)],
)

pred_probs = model3.predict(X_test)
roc_auc_score(y_test, pred_probs)


def submit_model(model):
    pred_probs_submission = model.predict(df_test)
    submission = pd.DataFrame(
        {
            'rainfall' : pred_probs_submission
        },
        index = df_test['id']
    )
    print(submission.head())
    submission.to_csv('/kaggle/working/submission.csv')
    return submission
    
sub3 = submit_model(model3)


class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
        self.l1 = nn.Linear(input_size, 16)
        self.bn1 = nn.BatchNorm1d(16)  
        self.drop1 = nn.Dropout(0.3)   

        self.l2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)   
        self.drop2 = nn.Dropout(0.3)   

        self.l3 = nn.Linear(8, 1)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)    
        x = self.relu(x)
        x = self.drop1(x) 

        x = self.l2(x)
        x = self.bn2(x)    
        x = self.relu(x)
        # x = self.drop2(x)  

        x = self.l3(x)  
        return x



X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_size = X_train.shape[1]
model = ANN(input_size)
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 500
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    all_labels = []
    all_probs = []

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch).squeeze()  
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy() 
        all_probs.extend(probs)
        all_labels.extend(y_batch.detach().cpu().numpy())

    train_auc = roc_auc_score(all_labels, all_probs)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}, Train ROC AUC: {train_auc:.4f}")



model.eval()
test_labels = []
test_probs = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch).squeeze()
        probs = torch.sigmoid(logits)  
        test_probs.extend(probs.cpu().numpy())
        test_labels.extend(y_batch.cpu().numpy())

test_auc = roc_auc_score(test_labels, test_probs)
print(f"Test ROC AUC: {test_auc:.4f}")


def submit_model(model, df_test):
    model.eval()  
    X_test_tensor = torch.tensor(df_test.values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test_tensor).squeeze() 
        pred_probs_submission = torch.sigmoid(logits).numpy() 
    submission = pd.DataFrame(
        {
            'rainfall': pred_probs_submission
        },
        index=df_test['id']  
    )
    print(submission.head())
    submission.to_csv('/kaggle/working/submission.csv')
    print("Submission file saved as 'submission.csv'")
    return submission

sub4 = submit_model(model, df_test)




