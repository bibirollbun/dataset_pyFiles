import pandas as pd
import numpy as np
from collections import OrderedDict
import os#interact with operation system
from sklearn.model_selection import KFold
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset,TensorDataset
import warnings
#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')
import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

train=pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/train.csv").fillna(0)
test=pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/test.csv").fillna(0)
sub=pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/sample_submission.csv')
sub.head()



train.shape




columns = train.columns
columns


FEATURES=[c for c in test.columns if c!='geology_id']
TARGETS=[c for c in sub.columns if c!='geology_id']

solution=train[['geology_id']+TARGETS].copy()
train_sub=train[['geology_id']+TARGETS].copy()
train.head()


train.describe()


train.iloc[:,1:300].min().min()



class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(300, 200),
            nn.Tanh(),
            #nn.Dropout(0.1),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 3000)
        )
                
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)



folds = 5
epochs = 50
loss_fn = nn.HuberLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")





from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()
scaler2 = StandardScaler()

kf = KFold(n_splits=folds, random_state=42, shuffle=True)

X_num,y = np.log(31+train[FEATURES].values),train[TARGETS].values

scaler.fit(X_num)
X_num = scaler.transform(X_num)
X_num_test = np.log(31+test[FEATURES].values)
X_num_test = scaler.transform(X_num_test)

scaler2.fit(y)
y = scaler2.transform(y)
test_dl = DataLoader(TensorDataset(torch.tensor(X_num_test, dtype=torch.float32),
                                   ),
                     batch_size=1024,shuffle=False)
test_tabm = np.zeros((folds,len(test),len(TARGETS)))

for i, (train_index, val_index) in enumerate(kf.split(train[FEATURES])):
    X_num_train,X_num_val = X_num[train_index],X_num[val_index]
    y_train,y_val = y[train_index],y[val_index]
    train_dl = DataLoader(TensorDataset(torch.tensor(X_num_train, dtype=torch.float32), 
                                        torch.tensor(y_train, dtype=torch.float32)),
                                        batch_size=128, shuffle=True)
    valid_dl = DataLoader(TensorDataset(torch.tensor(X_num_val, dtype=torch.float32), 
                                        torch.tensor(y_val, dtype=torch.float32)), 
                                        batch_size=128, shuffle=False)
    model=MLP()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-2 ,
    )
    print("< model training >")
    for epoch in range(epochs):
        model.train() 
        with tqdm(train_dl, total=len(train_dl), leave=True) as phar:
            for train_tensor in phar:
                optimizer.zero_grad()
                X_num_train,y_train = [t.to(device) for t in train_tensor]
                output = model(X_num_train)
                loss = loss_fn((output-y_train.mean(dim=0))/y_train.std(dim=0), (y_train-y_train.mean(dim=0))/y_train.std(dim=0)  )
                loss.backward()
                optimizer.step()
                
                phar.set_postfix(
                    OrderedDict(    epoch=f'{epoch+1}/{epochs}',
                                    loss=f'{loss.item():.6f}'  )
                )
                phar.update(1)
        model.eval()
        valid_preds,valid_targets= [],[]
        for valid_tensor in valid_dl:
            X_num_val,y_val = [t.to(device) for t in valid_tensor]
            with torch.no_grad():
                output = model(X_num_val)
            valid_preds.append(output.cpu().numpy())
        valid_preds = np.concatenate(valid_preds)
        train_sub.loc[val_index,TARGETS]=valid_preds
    print("< model prediction >")
    model.eval()
    test_preds = []
    with torch.no_grad():
        for test_tensor in test_dl:
            X_num_test = [t.to(device) for t in test_tensor][0]
            output = model(X_num_test)
            test_preds.append(output.cpu().numpy())
    test_tabm[i] = np.concatenate(test_preds)
        


import pandas.api.types

class ParticipantVisibleError(Exception):
     pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    NEGATIVE_PART = -299
    LARGEST_CHUNK = 600
    SMALLEST_CHUNK = 350
    TOTAL_REALIZATIONS = 10
    sigma_2 = np.ones((LARGEST_CHUNK+NEGATIVE_PART-1))
    from_ranges = [1, 61, 245]
    to_ranges_excl = [61, 245, 301]
    log_slopes = [1.0406028049510443, 0.0, 7.835345062351012]
    log_offsets = [-6.430669850650689, -2.1617411566043896, -45.24876794412965]
    for growth_mode in range(len(from_ranges)):
        for i in range(from_ranges[growth_mode], to_ranges_excl[growth_mode]):
            sigma_2[i-1] = np.exp(np.log(i)*log_slopes[growth_mode]+log_offsets[growth_mode])

    sigma_2 *= 6000
  
    cov_matrix_inv_diag = 1. / sigma_2  # Inverse of the diagonal elements
    p = 1./TOTAL_REALIZATIONS
    num_rows = solution.shape[0]
    ps = np.full((num_rows, TOTAL_REALIZATIONS), p)

    exp_misfit = np.zeros((num_rows, TOTAL_REALIZATIONS))

    num_columns = LARGEST_CHUNK + NEGATIVE_PART - 1

    full_submission = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))
    full_solution = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))
    for k in range(TOTAL_REALIZATIONS):
        misfit = np.zeros((num_rows, num_columns))
        for i in range(num_columns):
            if k == 0:
                column_name = str(i+1)
            else:
                column_name = f"r_{k}_pos_{i+1}"
            misfit[:,i] = solution[column_name].values - submission[column_name].values
            full_submission[:, k, i] = submission[column_name].values
            full_solution[:, k, i] = solution[column_name].values
        misfit_scaled = misfit * cov_matrix_inv_diag
        inner_product = np.sum(misfit_scaled * misfit, axis=1)
        exp_misfit_cur = np.exp(inner_product)
        exp_misfit[:,k] = exp_misfit_cur
    nll = -np.log(np.sum(ps*exp_misfit))
    computed_score = nll.mean()
    
    # Check for infinite scores
    if np.isinf(computed_score):
        raise ParticipantVisibleError(f"Your score is {computed_score}, which means there is room for improvement.")

    return computed_score
print(f"final CV:{score(solution,train_sub,'geology_id')}")






sub.shape


sub[TARGETS]=scaler2.inverse_transform(test_tabm.mean(axis=0))

sub.to_csv("submission.csv",index=None)
sub.head()


