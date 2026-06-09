!pip install torch_geometric


!pip install torch-sparse


!pip install torch-scatter


import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")


import torch_sparse


# Ğ‘Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğµ Ğ±Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞºĞ¸
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ğ§Ñ‚Ğ¾Ğ±Ñ‹ ĞºÑ€Ğ°Ñ�Ğ¸Ğ²Ğ¾ Ğ¾Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶Ğ°Ğ»Ğ¸Ñ�ÑŒ Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¸
import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
#plt.style.use("seaborn-dark")


#PATH_INPUT_BASE = '../data/raw/ieee-fraud-detection'
PATH_INPUT_BASE = '/kaggle/input/ieee-fraud-detection'


#PATH_OUTPUT_BASE = '../data/processing/'
PATH_OUTPUT_BASE = '/kaggle/working/'


# Ğ˜Ğ¼Ğ¿Ğ¾Ñ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ±Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞºĞ¸
import pandas as pd
import numpy as np

# ĞŸÑƒÑ‚ÑŒ Ğº Ğ´Ğ°Ğ½Ğ½Ñ‹Ğ¼
PATH_TRANSACTION = f'{PATH_INPUT_BASE}/train_transaction.csv'
PATH_IDENTITY = f'{PATH_INPUT_BASE}/train_identity.csv'

# Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
transaction = pd.read_csv(PATH_TRANSACTION)
identity = pd.read_csv(PATH_IDENTITY)

print(f"Ğ Ğ°Ğ·Ğ¼ĞµÑ€ train_transaction: {transaction.shape}")
print(f"Ğ Ğ°Ğ·Ğ¼ĞµÑ€ train_identity: {identity.shape}")

# Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ğ¿Ğ¾ TransactionID
train = transaction.merge(identity, how='left', on='TransactionID')

print(f"Ğ˜Ñ‚Ğ¾Ğ³Ğ¾Ğ²Ñ‹Ğ¹ Ñ€Ğ°Ğ·Ğ¼ĞµÑ€ train: {train.shape}")

# Ğ‘Ñ‹Ñ�Ñ‚Ñ€Ñ‹Ğ¹ Ğ¾Ñ�Ğ¼Ğ¾Ñ‚Ñ€ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
print(train.head())



# 4. ĞŸÑ€Ğ¸Ğ²ĞµĞ´ĞµĞ½Ğ¸Ğµ Ñ‚Ğ¸Ğ¿Ğ¾Ğ²
# ĞšĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ñ�Ğ²Ğ½Ğ¾ Ğ¸Ğ·Ğ²ĞµÑ�Ñ‚Ğ½Ñ‹
categorical_features = [
    'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
    'DeviceType', 'DeviceInfo'
]

# identity ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
identity_categoricals = [col for col in train.columns if col.startswith('id_')]
categorical_features += identity_categoricals

# M-Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ â€” Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ñ‹Ğµ (Ğ½Ğ¾ Ğ¸Ğ½Ğ¾Ğ³Ğ´Ğ° Ğ¼Ğ¾Ğ³ÑƒÑ‚ Ğ±Ñ‹Ñ‚ÑŒ NaN)
m_features = [col for col in train.columns if col.startswith('M')]
categorical_features += m_features

# Ğ�Ñ�Ñ‚Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ñ‚Ğµ, Ñ‡Ñ‚Ğ¾ Ğ¾Ñ�Ñ‚Ğ°Ğ»Ğ¸Ñ�ÑŒ Ğ² train
categorical_features = [col for col in categorical_features if col in train.columns]

# ĞšĞ°Ñ�Ñ‚ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ² str/categorical
for col in categorical_features:
    train[col] = train[col].astype('category')


import pandas as pd
import matplotlib.pyplot as plt

# 1. Ğ”Ğ¾Ğ»Ñ� Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ² Ğ¿Ğ¾ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ°Ğ¼
missing_col = train.isnull().mean().sort_values(ascending=False)
high_null_cols = missing_col[missing_col > 0.3]  # Ğ¿Ğ¾Ñ€Ğ¾Ğ³ 30%

print(f"ğŸ§¹ Ğ ĞµĞºĞ¾Ğ¼ĞµĞ½Ğ´ÑƒĞµÑ‚Ñ�Ñ� ÑƒĞ´Ğ°Ğ»Ğ¸Ñ‚ÑŒ {len(high_null_cols)} ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº (>{30}% Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ²):")
print(high_null_cols)

# 2. Ğ”Ğ¾Ğ»Ñ� Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ² Ğ¿Ğ¾ Ñ�Ñ‚Ñ€Ğ¾ĞºĞ°Ğ¼
missing_row = train.isnull().mean(axis=1)
high_null_rows = missing_row[missing_row > 0.5]

print(f"\nğŸ§¹ Ğ ĞµĞºĞ¾Ğ¼ĞµĞ½Ğ´ÑƒĞµÑ‚Ñ�Ñ� ÑƒĞ´Ğ°Ğ»Ğ¸Ñ‚ÑŒ {len(high_null_rows)} Ñ�Ñ‚Ñ€Ğ¾Ğº (>{50}% Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ²)")


# ĞŸĞ¾Ñ€Ğ¾Ğ³: ÑƒĞ´Ğ°Ğ»Ğ¸Ğ¼ Ğ²Ñ�Ğµ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ñ‹, Ğ³Ğ´Ğµ > 30% Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ²
threshold_col = 0.3
missing_ratio_col = train.isnull().mean()

cols_to_drop = missing_ratio_col[missing_ratio_col > threshold_col].index.tolist()
df_cleaned = train.drop(columns=cols_to_drop)

print(f"Ğ£Ğ´Ğ°Ğ»ĞµĞ½Ğ¾ {len(cols_to_drop)} ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº:", cols_to_drop)



# ĞŸĞ¾Ñ€Ğ¾Ğ³: ÑƒĞ´Ğ°Ğ»Ğ¸Ğ¼ Ñ�Ñ‚Ñ€Ğ¾ĞºĞ¸, Ğ³Ğ´Ğµ > 50% Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ñ‹
threshold_row = 0.5
missing_ratio_row = train.isnull().mean(axis=1)

rows_to_drop = train[missing_ratio_row > threshold_row].index
df_cleaned = df_cleaned.drop(index=rows_to_drop)

print(f"Ğ£Ğ´Ğ°Ğ»ĞµĞ½Ğ¾ {len(rows_to_drop)} Ñ�Ñ‚Ñ€Ğ¾Ğº")


df_cleaned.shape


# Ğ§Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ Ğ¸ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
num_cols = df_cleaned.select_dtypes(include=['number']).columns
cat_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns


for col in num_cols:
    if df_cleaned[col].isnull().any():
        median = df_cleaned[col].median()
        df_cleaned[col].fillna(median, inplace=True)


for col in cat_cols:
    if df_cleaned[col].isnull().any():
        mode = df_cleaned[col].mode(dropna=True)
        if not mode.empty:
            df_cleaned[col].fillna(mode[0], inplace=True)
        else:
            df_cleaned[col].fillna('Unknown', inplace=True)


missing_summary = df_cleaned.isnull().sum()
print("Ğ�Ñ�Ñ‚Ğ°Ğ²ÑˆĞ¸ĞµÑ�Ñ� Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸ Ğ¿Ğ¾ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ°Ğ¼:\n", missing_summary[missing_summary > 0])


df_modified = df_cleaned.copy()


# Ğ�Ğ°Ğ¹Ğ´ĞµĞ¼ Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ
min_timestamp = df_modified['TransactionDT'].min()

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ Ğ¼ĞµÑ‚ĞºĞ¸ Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğ¹
df_modified['Relative_TransactionDT'] = df_modified['TransactionDT'] - min_timestamp

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ² Ğ´Ğ½Ğ¸, Ñ‡Ğ°Ñ�Ñ‹, Ğ¼Ğ¸Ğ½ÑƒÑ‚Ñ‹ Ğ¸ Ñ‚.Ğ´.
df_modified['Transaction_day'] = df_modified['Relative_TransactionDT'] // (24 * 60 * 60)  # Ğ² Ğ´Ğ½Ñ�Ñ…
df_modified['Transaction_hour'] = (df_modified['Relative_TransactionDT'] // 3600) % 24  # Ğ² Ñ‡Ğ°Ñ�Ğ°Ñ…
df_modified['Transaction_weekday'] = (df_modified['Relative_TransactionDT'] // (3600*24)) % 7
df_modified['Transaction_day'] = df_modified['Transaction_day'].astype(int)
df_modified['Transaction_hour'] = df_modified['Transaction_hour'].astype(int)
df_modified['Transaction_weekday'] = df_modified['Transaction_weekday'].astype(int)


import numpy as np

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ° TransactionAMT
df_modified['log_TransactionAmt'] = np.log1p(df_modified['TransactionAmt'])


bins = [0, 100, 1000, 5000, 10000, np.inf]
labels = ['Low', 'Medium', 'High', 'Very High', 'Extremely High']

df_modified['TransactionAmt_binned'] = pd.cut(df_modified['TransactionAmt'], bins=bins, labels=labels)

# Ğ�Ğ»ÑŒÑ‚ĞµÑ€Ğ½Ğ°Ñ‚Ğ¸Ğ²Ğ½Ñ‹Ğ¹ Ñ�Ğ¿Ğ¾Ñ�Ğ¾Ğ±:  Ğ‘Ğ¸Ğ½Ğ½Ğ¸Ğ½Ğ³ Ñ�ÑƒĞ¼Ğ¼Ñ‹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ Ğ¿Ğ¾ ĞºĞ²Ğ°Ğ½Ñ‚Ğ¸Ğ»Ñ�Ğ¼
#train['TransactionAmt_bin'] = pd.qcut(train['TransactionAmt'], q=10, duplicates='drop')


# Ğ Ğ°Ñ�Ñ�Ñ‡Ğ¸Ñ‚Ğ°ĞµĞ¼ Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºÑƒ Ğ´Ğ»Ñ� Ğ²Ñ‹Ñ�Ğ²Ğ»ĞµĞ½Ğ¸Ñ� Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¾Ğ²
q1 = df_modified['TransactionAmt'].quantile(0.25)  # 25-Ğ¹ Ğ¿ĞµÑ€Ñ†ĞµĞ½Ñ‚Ğ¸Ğ»ÑŒ
q3 = df_modified['TransactionAmt'].quantile(0.75)  # 75-Ğ¹ Ğ¿ĞµÑ€Ñ†ĞµĞ½Ñ‚Ğ¸Ğ»ÑŒ
iqr = q3 - q1  # Ğ˜Ğ½Ñ‚ĞµÑ€ĞºĞ²Ğ°Ñ€Ñ‚Ğ¸Ğ»ÑŒĞ½Ñ‹Ğ¹ Ñ€Ğ°Ğ·Ğ¼Ğ°Ñ…

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr


df_modified['isOutlier'] = ((df_modified['TransactionAmt'] < lower_bound) | (df_modified['TransactionAmt'] > upper_bound)).astype(int)


from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler

#numeric_features = ['log_TransactionAmt', 'Transaction_hour', 'Transaction_weekday']
#categorical_features = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'DeviceType', 'DeviceInfo']

# Ğ§Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ Ğ¸ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸
numeric_features = df_modified.select_dtypes(include=['number']).columns
categorical_features = df_modified.select_dtypes(include=['object', 'category']).columns

base_train = df_modified.copy()



# Ğ›ĞµĞ³ĞºĞ°Ñ� Ğ¿Ñ€ĞµĞ´Ğ¾Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºĞ°
for col in categorical_features:
    #base_train[col] = base_train[col].fillna('unknown')
    base_train[col] = base_train[col].astype(str)
    le = LabelEncoder()
    base_train[col] = le.fit_transform(base_train[col])



# 2. Ğ�Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºĞ° Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² (Ğ·Ğ°Ğ¿Ğ¾Ğ»Ğ½ĞµĞ½Ğ¸Ğµ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ² Ğ¼ĞµĞ´Ğ¸Ğ°Ğ½Ğ¾Ğ¹)
#scaler = StandardScaler()
#base_train[numeric_features] = scaler.fit_transform(base_train[numeric_features])


base_train.head()


#base_train.to_csv(f'{PATH_OUTPUT_BASE}/processing/df_preprocessing.csv', index=False)


# ĞŸÑ€Ğ¾Ñ�Ñ‚Ğ°Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ
#X = base_train[numeric_features + categorical_features]
X = base_train.drop(labels=['TransactionID', 'TransactionDT', 'Relative_TransactionDT', 'TransactionAmt', 'isOutlier',  'isFraud'], axis=1)
y = base_train['isFraud']


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)








import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()


class GCNWithLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x.squeeze()


import torch
from torch import nn
from torch_geometric.loader import NeighborLoader

def train(
    model, 
    loader: NeighborLoader, 
    criterion=None, 
    optimizer=None, 
    scheduler=None, 
    epochs=100, 
    device="cpu", 
    grad_clip=None,
    verbose=True):

    model.to(device)
    model.train()

    # ĞµÑ�Ğ»Ğ¸ Ğ½Ğµ Ğ¿ĞµÑ€ĞµĞ´Ğ°Ğ½ criterion - Ğ·Ğ°Ğ´Ğ°Ñ‘Ğ¼ BCEWithLogitsLoss
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    # ĞµÑ�Ğ»Ğ¸ Ğ½Ğµ Ğ¿ĞµÑ€ĞµĞ´Ğ°Ğ½ Ğ¾Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ°Ñ‚Ğ¾Ñ€ - Ğ·Ğ°Ğ´Ğ°Ñ‘Ğ¼ AdamW + weight_decay
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # ĞœĞ¾Ğ¶Ğ½Ğ¾ Ğ´Ğ¾Ğ±Ğ°Ğ²Ğ¸Ñ‚ÑŒ ReduceLROnPlateau Ğ¸Ğ»Ğ¸ StepLR
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    for epoch in range(epochs):
        epoch_loss = 0
        total = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = criterion(logits[:batch.batch_size], batch.y[:batch.batch_size].float())
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_size = batch.batch_size
            epoch_loss += loss.item() * batch_size  # Ğ”Ğ°Ñ‘Ğ¼ Ğ²ĞµÑ� Ğ¿Ğ¾ Ñ‡Ğ¸Ñ�Ğ»Ñƒ Ğ±Ğ°Ñ‚Ñ‡
            total += batch_size

        avg_loss = epoch_loss / total if total > 0 else 0
        scheduler.step(avg_loss)
        if verbose:
            print(f"Epoch {epoch+1:03d}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.5f}")





from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# ğŸ“Š Ğ¢ĞµÑ�Ñ‚
@torch.no_grad()
def evaluate(model, loader: NeighborLoader, name="Model"):
    model.eval()

    all_true = []
    all_probs = []
    all_preds = []
    for batch in loader:
        # Ğ•Ñ�Ğ»Ğ¸ Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµÑˆÑŒ device:
        # batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        # Ğ¢Ğ¾Ğ»ÑŒĞºĞ¾ Ğ´Ğ»Ñ� Ñ†ĞµĞ½Ñ‚Ñ€Ğ°Ğ»ÑŒĞ½Ğ¾Ğ³Ğ¾ Ğ±Ğ°Ñ‚Ñ‡Ğ°:
        logits = logits[:batch.batch_size]
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        true = batch.y[:batch.batch_size].long()
        all_true.append(true.cpu())
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())

    all_true = torch.cat(all_true).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()


    # ğŸ�¯ ROC AUC
    fpr, tpr, _ = roc_curve(all_true, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # ğŸ�¯ PR AUC
    precision, recall, _ = precision_recall_curve(all_true, all_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"{name} (PR AUC = {pr_auc:.2f})")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    return {
        "Accuracy": accuracy_score(all_true, all_preds),
        "F1": f1_score(all_true, all_preds),
        "AUC": roc_auc_score(all_true, all_probs),
        "Precision": precision_score(all_true, all_preds),
        "Recall": recall_score(all_true, all_preds)
    }


import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


import pandas as pd
#df = pd.read_csv('../data/processing/df_preprocessing.csv') # Ğ¢Ğ²Ğ¾Ğ¹ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚
df = base_train.copy()
df = df.sort_values('TransactionDT').reset_index(drop=True)

print(df.columns.tolist())








import random

MAX_GROUP_SIZE = 300
PAIR_PER_NODE = 5  # Ğ´Ğ»Ñ� Ğ±Ğ¾Ğ»ÑŒÑˆĞ¸Ñ… Ğ³Ñ€ÑƒĞ¿Ğ¿: Ñ�ĞºĞ¾Ğ»ÑŒĞºĞ¾ Ñ�Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ñ‹Ñ… Ñ€Ñ‘Ğ±ĞµÑ€ Ğ´Ğ»Ñ� ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¹ Ğ½Ğ¾Ğ´Ñ‹

edge_list = []

for col in ['card1', 'card2', 'card3', 'Relative_TransactionDT',  'log_TransactionAmt', 'P_emaildomain']:
    groups = df.groupby(col)
    for val, group in groups:
        if pd.isnull(val): continue
        tx_ids = group.index.tolist()
        n = len(tx_ids)
        if n < 2:
            continue
        if n > MAX_GROUP_SIZE:
            # Ğ¡Ñ�Ğ¼Ğ¿Ğ»Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¿Ğ°Ñ€Ñ‹
            for idx in tx_ids:
                sampled = random.sample(tx_ids, PAIR_PER_NODE)
                for s in sampled:
                    if idx != s:
                        edge_list.append((idx, s))
                        edge_list.append((s, idx))
        else:
            # Ğ’Ñ�Ğµ-Ğ²Ñ�Ğµ Ğ¿Ğ°Ñ€Ñ‹
            for i in range(n):
                for j in range(i+1, n):
                    edge_list.append((tx_ids[i], tx_ids[j]))
                    edge_list.append((tx_ids[j], tx_ids[i]))
print(f'Ğ Ñ‘Ğ±ĞµÑ€ Ğ²Ñ�ĞµĞ³Ğ¾: {len(edge_list)}')


import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edges_from(edge_list[:100])
nx.draw(G, with_labels=True, node_size=120)
plt.show()


import torch
from torch_geometric.data import Data


feature_cols = [
    
    'card4', 'card5', 'card6', 'addr1', 'addr2', 'ProductCD',
    'Transaction_day', 'Transaction_hour',
    'Transaction_weekday',  'TransactionAmt_binned',
    'isOutlier',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D4', 'D10', 'D15', 'M6', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321'
]
x = torch.tensor(df[feature_cols].values, dtype=torch.float)    # X â€” Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ğ° Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² (np.array/pd.DataFrame.values) [num_nodes, num_features]
y = torch.tensor(df['isFraud'].values, dtype=torch.long)  # [num_nodes]

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµv Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº Ñ€Ñ‘Ğ±ĞµÑ€ (edge_list) Ğ² Ñ‚ĞµĞ½Ğ·Ğ¾Ñ€ PyTorch
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

data = Data(x=x, edge_index=edge_index, y=y)
data


from torch_geometric.loader import NeighborLoader

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")
#data = data.to(device)

N_positive = (y == 1).sum().item()
N_negative = (y == 0).sum().item()
pos_weight = torch.tensor([N_negative / N_positive], dtype=torch.float).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ĞŸÑƒÑ�Ñ‚ÑŒ Ñƒ Ñ‚ĞµĞ±Ñ� Ğ³Ñ€Ğ°Ñ„ data ĞºĞ°Ğº Ğ²Ñ‹ÑˆĞµ

batch_size = 512  # Ğ¡ĞºĞ¾Ğ»ÑŒĞºĞ¾ ÑƒĞ·Ğ»Ğ¾Ğ² (Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹) Ğ² Ğ¾Ğ´Ğ½Ğ¾Ğ¼ Ğ±Ğ°Ñ‚Ñ‡Ğµ
num_neighbors = [10, 5]  # Ğ¡ĞºĞ¾Ğ»ÑŒĞºĞ¾ Ñ�Ğ¾Ñ�ĞµĞ´ĞµĞ¹ Ñ�Ğ¾Ğ±Ğ¸Ñ€Ğ°Ñ‚ÑŒ Ğ½Ğ° 1-Ğ¼ Ğ¸ 2-Ğ¼ Ñ�Ğ»Ğ¾Ñ�Ñ…

train_loader = NeighborLoader(
    data,
    num_neighbors=num_neighbors,
    batch_size=batch_size,
    shuffle=False,  # Ğ² Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğ¸ Ğ´Ğ¾Ğ»Ğ¶ĞµĞ½ Ğ±Ñ‹Ñ‚ÑŒ shuffle
)




!pip install torch-sparse


import torch_sparse


model_simple_gnn = SimpleGNN(input_dim=data.x.shape[1], hidden_dim=32).to(device)

train(model=model_simple_gnn, loader=train_loader, criterion=criterion, epochs=5)


evaluate(model=model_simple_gnn, loader=train_loader, name="SimpleGNN")


model_linear_gnn = GCNWithLinear(input_dim=data.x.shape[1], hidden_dim=32).to(device)

train(model=model_linear_gnn, loader=train_loader, criterion=criterion, epochs=5)


evaluate(model=model_linear_gnn, loader=train_loader, name="GCNWithLinear")


import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def train_one_epoch(model, loader, criterion, optimizer, device='cpu', grad_clip=None):
    model.train()
    total_loss = 0
    total = 0
    all_preds = []
    all_targets = []
    sigmoid = torch.nn.Sigmoid()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        y_true = batch.y[:batch.batch_size].float()
        y_pred = logits[:batch.batch_size]
        loss = criterion(y_pred, y_true)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch.batch_size
        total_loss += loss.item() * batch_size
        total += batch_size
        with torch.no_grad():
            all_preds.append(sigmoid(y_pred).cpu())
            all_targets.append(y_true.cpu())
    
    avg_loss = total_loss / total
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    bin_preds = (preds >= 0.5).astype(np.float32)
    acc = accuracy_score(targets, bin_preds)
    try:
        auc = roc_auc_score(targets, preds)
    except:
        auc = np.nan
    return avg_loss, acc, auc


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0
    total = 0
    all_preds = []
    all_targets = []
    sigmoid = torch.nn.Sigmoid()

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        y_true = batch.y[:batch.batch_size].float()
        y_pred = logits[:batch.batch_size]
        loss = criterion(y_pred, y_true)
        batch_size = batch.batch_size
        total_loss += loss.item() * batch_size
        total += batch_size
        all_preds.append(sigmoid(y_pred).cpu())
        all_targets.append(y_true.cpu())
    
    avg_loss = total_loss / total
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    bin_preds = (preds >= 0.5).astype(np.float32)
    acc = accuracy_score(targets, bin_preds)
    try:
        auc = roc_auc_score(targets, preds)
    except:
        auc = np.nan
    return avg_loss, acc, auc


def fit_gnn(
    model,
    train_loader,
    val_loader,
    criterion=None,
    optimizer=None,
    scheduler=None,
    epochs=100,
    device='cpu',
    grad_clip=None,
    early_stopping_rounds=10,
    verbose=True
):
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
               'train_auc': [], 'val_auc': []}
    best_val_loss = float('inf')
    patience = 0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip)
        val_loss, val_acc, val_auc = validate_one_epoch(
            model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)

        # logging
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        if verbose:
            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.5f}"
            )

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_rounds:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.4f}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


model, history = fit_gnn(
    model_simple_gnn, 
    train_loader, 
    train_loader, 
    epochs=5,
    device="cpu",
    grad_clip=2.0,
    early_stopping_rounds=10
)


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(history['train_loss'], label='Train loss')
plt.plot(history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.title('Loss dynamics')
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history['train_acc'], label='Train accuracy')
plt.plot(history['val_acc'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.title('Accuracy dynamics')
plt.show()


plt.plot(history['train_auc'], label='Train ROC-AUC')
plt.plot(history['val_auc'], label='Validation ROC-AUC')
plt.legend()
plt.title("ROC-AUC per epoch")
plt.show()




