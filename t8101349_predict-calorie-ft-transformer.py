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


pip install -U scikit-learn



!pip install rtdl --quiet


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rtdl import FTTransformer
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df = df.drop('id',axis = 1)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test = df_test.drop('id',axis=1)
df = df.drop_duplicates()


df.describe()


df.isnull().sum()


sns.heatmap(df.corr(numeric_only=True), annot = True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()



sns.histplot(df["Calories"], kde=True)
plt.title("原始 Calories 分布")
plt.show()

sns.histplot(np.log1p(df["Calories"]), kde=True)
plt.title("Log1p 後的 Calories 分布")
plt.show()



#label encoding
df['Age_bin'] = pd.cut(df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 100], labels=False)



#df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
#Heart_Rate*Duration
#df["HR_dur"] = df["Heart_Rate"] * df["Duration"]
#(Body_Temp − mean)² 
#df["Temp_squar"] = (df["Body_Temp"] - df["Body_Temp"].mean())**2



categorical_cols = "Sex"

le = LabelEncoder()
df[categorical_cols] = le.fit_transform(df[categorical_cols])




# standard normalize
numerical_cols = ["Duration","Heart_Rate","Body_Temp","Weight","Height"]

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])



df


def feature_engineering(df : pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:


    #df['male'] = (df['Sex'] == 'male').astype(int)
    #df['female'] = (df['Sex'] == 'female').astype(int)
    df['Age_bin'] = pd.cut(df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 100], labels=False)
    #df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    #df["HR_dur"] = df["Heart_Rate"] * df["Duration"]
    #df["Temp_squar"] = (df["Body_Temp"] - df["Body_Temp"].mean())**2
    df[categorical_cols] = le.transform(df[categorical_cols])
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    
    return df.drop(["Age"], axis = 1 )


y = np.log1p(df["Calories"])   #logp1
#y = df["Calories"]
X = df.drop(['Calories',"Age"], axis = 1 )



X_test = feature_engineering(df_test,numerical_cols,categorical_cols)


# Dataset 定義
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        else:
            return self.X_num[idx], self.X_cat[idx]



"""
# 模型與訓練函數
def train_model(X_train, y_train, X_val, y_val, cat_idx):
    model = FTTransformer(
        n_num_features=X_train.shape[1],
        cat_cardinalities=[],
        token_dim=64,
        n_blocks=3,
        attention_dropout=0.2,
        ffn_dropout=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)

    for epoch in range(20):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            preds = model(xb).squeeze().cpu().numpy()
            val_preds.append(preds)
        val_preds = np.concatenate(val_preds)
    return model, val_preds
"""


def to_tensor(x):
    return torch.tensor(x.values, dtype=torch.float32)

def train_model(X_train, y_train, X_val, y_val, cat_idx):
    cat_cols = ['Sex', 'Age_bin']  
    cat_cardinalities = [X_train[col].nunique() for col in cat_cols]

    model = FTTransformer(
        n_cont_features=len(numerical_cols),
        cat_cardinalities=cat_cardinalities,
        d_out=1,

    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Dataloader
    train_ds = TensorDataset(to_tensor(X_train), torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
    val_ds = TensorDataset(to_tensor(X_val), torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    # 驗證
    model.eval()
    val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            val_preds.append(model(xb).squeeze().cpu())
    val_preds = torch.cat(val_preds).numpy()

    return model, val_preds



def to_tensor(x):
    return torch.tensor(x.values, dtype=torch.float32)

def train_model(X_train, y_train, X_val, y_val, cat_cols, numerical_cols):
    # 分離數值與類別欄位
    X_train_num = torch.tensor(X_train[numerical_cols].values, dtype=torch.float32)
    X_train_cat = torch.tensor(X_train[cat_cols].values, dtype=torch.long)
    X_val_num = torch.tensor(X_val[numerical_cols].values, dtype=torch.float32)
    X_val_cat = torch.tensor(X_val[cat_cols].values, dtype=torch.long)

    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    cat_cardinalities = [X_train[col].nunique() for col in cat_cols]

    model = FTTransformer.make_default(
        n_num_features=len(numerical_cols),
        cat_cardinalities=cat_cardinalities,
        d_out=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Dataloader
    train_ds = TensorDataset(X_train_num, X_train_cat, y_train_t)
    val_ds = TensorDataset(X_val_num, X_val_cat, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    # Training loop
    model.train()
    for epoch in range(10):
        for xb_num, xb_cat, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb_num.to(device), xb_cat.to(device))
            loss = loss_fn(preds, yb.to(device))
            loss.backward()
            optimizer.step()

    # 驗證
    model.eval()
    val_preds = []
    with torch.no_grad():
        for xb_num, xb_cat, _ in val_loader:
            preds = model(xb_num.to(device), xb_cat.to(device))
            val_preds.append(preds.squeeze().cpu())
    val_preds = torch.cat(val_preds).numpy()

    return model, val_preds


cat_idx = [X.columns.get_loc(categorical_cols)]
cat_cols = ['Sex', 'Age_bin'] 

def encode_categoricals(df, cat_cols):
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

# Encode 全部資料（確保 test 也有用同樣的編碼）
X_all = pd.concat([X, X_test])
X_all_encoded, encoders = encode_categoricals(X_all, cat_cols)

# 再切回來
X = X_all_encoded.iloc[:len(X)]
X_test = X_all_encoded.iloc[len(X):]


# 訓練流程
FOLDS = 20
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))


           
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model, val_pred = train_model(X_train, y_train, X_val, y_val, cat_cols, numerical_cols)
    oof_preds[val_idx] = val_pred

    #rmse = mean_squared_error(np.expm1(y_val), np.expm1(val_pred), squared=False)
    mse = mean_squared_error(np.expm1(y_val), np.expm1(val_pred))
    rmse = np.sqrt(mse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")
    
    # 測試集預測
    test_num_tensor = torch.tensor(X_test[numerical_cols].values, dtype=torch.float32)
    test_cat_tensor = torch.tensor(X_test[cat_cols].values, dtype=torch.long)
    
    test_dataset = TabularDataset(test_num_tensor,test_cat_tensor)
    test_loader = DataLoader(test_dataset, batch_size=1024)
    model.eval()
    test_fold_preds = []
    
    with torch.no_grad():
        for xb_num, xb_cat in test_loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            preds = model(xb_num, xb_cat).squeeze().cpu().numpy()
            test_fold_preds.append(preds)
    
    test_preds += np.expm1(np.concatenate(test_fold_preds)) / FOLDS

# ➤ 最終 RMSE
#final_rmse = mean_squared_error(np.expm1(y), np.expm1(oof_preds), squared=False)
final_mse = mean_squared_error(np.expm1(y), np.expm1(oof_preds))
final_rmse = np.sqrt(final_mse)
print(f"\nOverall RMSE: {final_rmse:.4f}")



submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

submission["Calories"] = test_preds
submission.to_csv("submission.csv", index=False)
print('submission saved')
submission.head()

