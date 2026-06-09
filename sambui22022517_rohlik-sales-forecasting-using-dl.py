# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

import timm
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from PIL import Image
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from multiprocessing import Manager as MemoryManager
import shutil
import random
import cv2


# Global config
SEED = 26092004
root_dir = '/kaggle/input/rohlik-sales-forecasting-challenge-v2'
extend_dir = '/kaggle/input/extended-calendar-dataset-for-rohlik-challenge'
epochs = 35
batch_size = 2**11
num_models = 9
model_idx = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Loader
data_loader_params = {
    'batch_size': batch_size, 
    'shuffle': True,
}

# Model
model_params = {
    "hidden_size": 2048,
    "output_size": 1,
    "num_attn_heads": 8,
    "num_layers": 15,
    "dropout": 0.1
}

# Trainer
optim_params = {
    'lr': 1e-3,
    'weight_decay': 1e-4,
}
scheduler_params = {
    'mode': 'min', 
    'factor': 0.9, 
    'patience': 2, 
    'verbose': True, 
}
trainer_params = {
    'weight_init': False,
    'custom_weight_initializer': None,
    'device': device,
}

# training
train_func_params = {
    'epochs': epochs, 
    'logging_frequence': 1 if epochs <= 10 else epochs // 10, 
    'detect_best_model': True, 
    'save_at_end': False,
    'save_only_network': True,
    'save_name': f'rohlik-sales-forecasting-{model_idx}.pth',
    'load_best_model_at_end': True, 
    'start_detect_best_at_epoch': 0, 
    'start_apply_scheduler': None, 
}


def set_seed(seed: int = 42):
    """Cài đặt seed để đảm bảo tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu dùng nhiều GPU
    
    # Đảm bảo các phép tính trên CUDA luôn nhất quán
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(SEED)


class RohlikDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return torch.from_numpy(self.X.iloc[idx].to_numpy()).float(), None
        return torch.from_numpy(self.X.iloc[idx].to_numpy()).float(), self.y.iloc[idx]


# class MSEWithNegativePenalty(nn.Module):
#     def __init__(self, lambda_penalty=0.1):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.lambda_penalty = lambda_penalty

#     def forward(self, pred, target):
#         mse_loss = self.mse(pred, target)
#         penalty = self.lambda_penalty * torch.relu(-pred).mean()  # Phạt các giá trị âm
#         return mse_loss + penalty

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_funcs = nn.ModuleList([
            nn.SmoothL1Loss(), 
        ])

    def forward(self, pred, target):
        loss = 0
        for loss_func in self.loss_funcs:
            loss += loss_func(pred, target)
        loss /= len(self.loss_funcs)
        return loss


class GeGLU(nn.Module):
    def __init__(self, emb_channels, ffn_size):
        super().__init__()
        self.wi_0 = nn.Linear(emb_channels, ffn_size, bias=False)
        self.wi_1 = nn.Linear(emb_channels, ffn_size, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x_gelu = self.act(self.wi_0(x))
        x_linear = self.wi_1(x)
        x = x_gelu * x_linear
        return x

class Feed_Forward(nn.Module):
    def __init__(self, in_channels, ffw_channels, dropout=0.1):
        super().__init__()
        
        self.ln1 = GeGLU(in_channels, ffw_channels)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = GeGLU(ffw_channels, in_channels)
        
    def forward(self, x):
        '''
        input:  [N, H, W, channels]
        output: [N, H, W, channels]
        '''
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.ln2(x)
        return x

class Transformer_Encoder_Layer(nn.Module):
    def __init__(self, channels, num_attn_heads, ffw_channels, dropout=0.1):
        super().__init__()
        
        self.attn_norm = nn.LayerNorm(channels)
        self.attn_layer = nn.MultiheadAttention(channels, num_attn_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.ffw_norm = nn.LayerNorm(channels)
        self.ffw_layer = Feed_Forward(channels, ffw_channels, dropout)
        self.ffw_dropout = nn.Dropout(dropout)
        
    def forward(self, adp_pos_imgs):
        """
        input:  [N, H, W, channels]
        output: [N, H, W, channels]
        """
        _x = adp_pos_imgs
        x = self.attn_norm(adp_pos_imgs)
        x, _ = self.attn_layer(x, x, x)
        x = self.attn_dropout(x)
        x = x + _x
        
        _x = x
        x = self.ffw_norm(x)
        x = self.ffw_layer(x)
        x = self.ffw_dropout(x)
        x = x + _x
        return x

class RohlikModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_attn_heads, num_layers, dropout=0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_attn_heads = num_attn_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size), 
            *[Transformer_Encoder_Layer(self.hidden_size, self.num_attn_heads, self.hidden_size, self.dropout) for _ in range(num_layers)],
            nn.Linear(self.hidden_size, self.output_size), 
        )

    def forward(self, X):
        """
        input:  [N, input_size]
        output: [N, output_size]
        """
        return self.model(X)


class Trainer:
    def __init__(
        self, 
        model, 
        lossfunc, optimizer, scheduler, 
        weight_init=False, custom_weight_initializer=None, 
        device='cpu'
    ):
        self.model = model.to(device)
        
        self.lossfuncs = lossfunc
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.weight_init = weight_init
        self.custom_weight_initializer = custom_weight_initializer

        self.device = device
        
        self.save_best_model = False
        
        if self.weight_init:
            if self.custom_weight_initializer:
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weight)
                
    def xavier_init_weight(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)
    
    def save_model(self, dirname='/kaggle/working/network_params', filename='full_model.pth', only_network=False):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        state_dicts = {
            'network_params': self.model.state_dict(),
            'scheduler_params': self.scheduler.state_dict(),
            'optimizer_params': self.optimizer.state_dict(),
        }
        if only_network:
            return torch.save(self.model.state_dict(), os.path.join(dirname, filename))
        return torch.save(state_dicts, os.path.join(dirname, filename))
    
    def load_model(self, dirname='/kaggle/working/network_params', filename='full_model.pth'):
        best_model_path = f'{dirname}/{filename}'
        if os.path.exists(best_model_path):
            try:
                best_model_state = torch.load(best_model_path, weights_only=True, map_location=self.device)['network_params']
            except:
                best_model_state = torch.load(best_model_path, weights_only=True, map_location=self.device)
            self.model.load_state_dict(best_model_state)
            print(f'Model is loaded from {best_model_path}')
        else:
            print(f'Model is not exist at {best_model_path}')
                
    def _train_model(self, train_loader):
        total_loss = 0
        self.model.train()
        for imgs, labels in train_loader:
            self.optimizer.zero_grad()
            
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            pred_labels = self.model(imgs).squeeze(1)
            
            loss = self.lossfuncs(pred_labels, labels)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            del imgs, labels
            torch.cuda.empty_cache()
        return total_loss / len(train_loader)

    def _eval_model(self, val_loader):
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                pred_labels = self.model(imgs).squeeze(1)
                
                loss = self.lossfuncs(pred_labels, labels)
                total_loss += loss.item()

                del imgs, labels
                torch.cuda.empty_cache()
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs, 
              logging_frequence=10, 
              detect_best_model=False, 
              save_at_end=False, save_only_network=False, save_name='full_model.pth',
              load_best_model_at_end=True, start_detect_best_at_epoch=10, 
              start_apply_scheduler=None, 
             ):

        all_train_loss = []
        all_val_loss = []
        saved = []
        min_loss = float('inf')
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss = 0
            self.model.train()
            for imgs, labels in train_loader:
                self.optimizer.zero_grad()
                
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                pred_labels = self.model(imgs).squeeze(1)
                
                loss = self.lossfuncs(pred_labels, labels)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
    
                del imgs, labels
                torch.cuda.empty_cache()
            train_loss /= len(train_loader)
                
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    pred_labels = self.model(imgs).squeeze(1)
                    
                    loss = self.lossfuncs(pred_labels, labels)
                    val_loss += loss.item()
    
                    del imgs, labels
                    torch.cuda.empty_cache()
            val_loss /= len(val_loader)
                    
            if start_apply_scheduler:
                if epoch >= start_apply_scheduler:
                    self.scheduler.step(val_loss)

            # Save best model
            if detect_best_model and val_loss < min_loss and epoch >= start_detect_best_at_epoch:
                min_loss = val_loss
                self.save_model(filename=f'{save_name}', only_network=save_only_network)
                self.save_best_model = True
                saved.append(1)
            else:
                saved.append(0)

            # Logging
            if (epoch + 1) % logging_frequence == 0:
                print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Time: {time.time() - start_time:.3f}")

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

        if (not self.save_best_model or save_at_end) and epochs > 0:
            self.save_model(filename=f'{save_name}', only_network=save_only_network)
            self.save_best_model = True
            saved[-1] = 1

        if load_best_model_at_end:
            self.load_model(filename=f'{save_name}')
        return all_train_loss, all_val_loss, saved


train = pd.read_csv(f'{root_dir}/sales_train.csv', parse_dates=['date'])
inventory = pd.read_csv(f'{root_dir}/inventory.csv')
test = pd.read_csv(f'{root_dir}/sales_test.csv', parse_dates=['date'])
calendar_extended = pd.read_csv(f'{extend_dir}/calendar_enriched_2025-01-05.csv', parse_dates=['date'])


## Pre-processing of train datasaet
train = train.drop(columns=['availability'])
train.dropna(subset=['sales'], inplace=True)

## Stacking of train and test dataset
test['sales'] = 0
df = pd.concat([train, test], ignore_index=True).sort_values('date')
df = df.merge(calendar_extended, on=['date', 'warehouse'], how='left')
df = df.merge(inventory, on=['unique_id', 'warehouse'], how='left')
df['date'] = pd.to_datetime(df['date'])

del train, test


## Add date feature
df['date_month'] = df['date'].dt.month
df['date_day'] = df['date'].dt.day
df['date_weekofyear'] = df['date'].dt.isocalendar().week
df['date_weekday'] = df['date'].dt.weekday 
df['date_dayofyear'] = df['date'].dt.dayofyear
df['date_year_sin'] = np.sin((df['date_year'] - df['date_year'].min()) / (df['date_year'].max() - df['date_year'].min()) * 2 * np.pi)
df['date_month_sin'] = np.sin(df['date_month'] / 12 * 2 * np.pi)


## Add lag feature
PERIODS = [14, 16, 18, 21, 30, 60, 90, 120, 180, 270, 350, 600, 1000]
for shift in PERIODS:
    df[f'product_sales_{shift}'] = df.groupby(['warehouse','name'])['sales'].shift(periods=shift)


# 10. Biến xu hướng thời gian
df["trend"] = df.groupby(['warehouse','name'])["sales"].diff().fillna(0)

# 11. Lấy giảm giá cao nhất
df["max_discount"] = df.filter(like="discount").max(axis=1)

# 12. Biến nhị phân có giảm giá hay không
df["has_discount"] = (df["max_discount"] > 0).astype(int)

# 13. Tổng hợp mức giảm giá trong 30 ngày qua
df["discount_magnitude"] = df.groupby("unique_id")["max_discount"].rolling(window=30, min_periods=1).sum().reset_index(level=0, drop=True)

# 15. Trung bình động 7 ngày
grouped_sales = df.groupby("unique_id")["sales"]
df["sales_ma_7"] = grouped_sales.shift(1).rolling(window=7).mean()

# 16. Độ lệch chuẩn doanh số
df["sales_std_7"] = grouped_sales.shift(1).rolling(window=7).std()

# 17. Đánh dấu ngày có doanh số thấp bất thường
df["low_sales_flag"] = (df["sales"] < df["sales"].quantile(0.1)).astype(int)

# 18. Tổng tồn kho của kho hàng
df["total_stock_per_warehouse"] = df.groupby("warehouse")["unique_id"].transform("count")

# 19. Tạo biến vòng quay hàng tồn kho
df["inventory_turnover"] = df["sales"] / (df["total_stock_per_warehouse"] + 1e-6)


df.loc[:, df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.mean()))


## Chuẩn hóa các cột số gốc
other_scaler = StandardScaler()
other_numeric_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in ('unique_id', 'sales')]
df[other_numeric_cols] = other_scaler.fit_transform(df[other_numeric_cols])

sales_scaler = StandardScaler()
df['sales'] = sales_scaler.fit_transform(df[['sales']])


df = df.drop(columns=df.select_dtypes(include=['object', 'category']).columns)


## Splitting of datasets
train_start_date  = '2023-06-03'
train_end_date  = '2024-06-02'

train = df[(df['date'] >= train_start_date) & (df['date'] <= train_end_date)]
test  = df[(df['date'] >  train_end_date)]

X_train = train.drop(['sales', 'date', 'unique_id', 'product_unique_id'], axis=1)
y_train = train['sales']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

X_test = test.drop(['sales', 'date', 'unique_id', 'product_unique_id'], axis=1)

del df

X_train.shape, X_test.shape


trainset = RohlikDataset(X_train, y_train)
valset = RohlikDataset(X_val, y_val)
testset = RohlikDataset(X_test)

train_loader = DataLoader(trainset, **data_loader_params)
val_loader = DataLoader(valset, **data_loader_params)


my_model = RohlikModel(X_train.shape[1], **model_params)
optimizer = optim.Adam(my_model.parameters(), **optim_params)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
loss_fn = CustomLoss()
trainer = Trainer(my_model, loss_fn, optimizer, scheduler, **trainer_params)


def test_model():
    test_img = torch.rand(X_train.shape[1]).to(device)
    test_img = test_img.unsqueeze(0)
    model = RohlikModel(X_train.shape[1], **model_params).to(device)
    model.eval()
    with torch.no_grad():
        res = model(test_img)
        print(res.shape)
        print(res.item())
    del test_img, model
    torch.cuda.empty_cache()
test_model()


start_time = time.time()
all_train_loss, all_val_loss, saved = trainer.train(train_loader, val_loader, **train_func_params)
print(time.time() - start_time)


training_log = {
    'Epochs': list(range(1, len(all_train_loss) + 1)),
    'Train Loss': all_train_loss,
    'Val Loss': all_val_loss,
    'Saved': saved, 
}

# Tạo DataFrame
df = pd.DataFrame(training_log)

# Vẽ biểu đồ tương ứng
sns.lineplot(data=df, x='Epochs', y='Train Loss', label='Train Loss', color='blue')  # Biểu đồ đường Train Loss
sns.lineplot(data=df, x='Epochs', y='Val Loss', label='Val Loss', color='orange')  # Biểu đồ đường Val Loss
saved_epochs = df[df['Saved'] == 1]  # Lọc ra những epoch được lưu
plt.scatter(saved_epochs['Epochs'], saved_epochs['Val Loss'], color='red', label='Saved Model', zorder=3, marker='o', s=30)

# Thêm tiêu đề và chú thích
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


y_pred = []
for x, _ in testset:
    pred = trainer.model(x.unsqueeze(0).to(device)).item()
    y_pred.append(pred)
y_pred = np.array(y_pred)
y_pred = sales_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

test['id'] = test['unique_id'].astype(str) + "_" + test['date'].astype(str)
test['sales_hat'] = y_pred
test[['id','sales_hat']].to_csv("submission.csv",index=False)




