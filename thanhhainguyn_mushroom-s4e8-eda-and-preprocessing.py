import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import cÃ¡c cÃ´ng cá»¥ tiá»�n xá»­ lÃ½ vÃ  chia dá»¯ liá»‡u
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# CÃ i Ä‘áº·t chung cho biá»ƒu Ä‘á»“
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

DATA_PATH = '/kaggle/input/playground-series-s4e8/train.csv'


df_raw = pd.read_csv(DATA_PATH)

df = df_raw.drop('id', axis=1)
    
print(f"KÃ­ch thÆ°á»›c dá»¯ liá»‡u: {df.shape}")

df.head(12)


# 1. XÃ¡c Ä‘á»‹nh cÃ¡c loáº¡i thuá»™c ttÃ­nh
df.info()
numerical_cols = [col for col in df.columns if df[col].dtype != 'object']
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'class']

# 2. Kiá»ƒm tra giÃ¡ trá»‹ khuyáº¿t (NaN)
nan_counts = df.isnull().sum()
print(f'\n{nan_counts}')

# 3. Thá»‘ng kÃª mÃ´ táº£ thuá»™c tÃ­nh kiá»ƒu sá»‘
print(f'\n{df[numerical_cols].describe().T}')


# 1. Trá»±c quan hÃ³a Tá»· lá»‡ Dá»¯ liá»‡u khuyáº¿t
nan_counts = df.isnull().sum()
nan_percent = (nan_counts / len(df)) * 100
nan_df = nan_percent[nan_percent > 0].sort_values(ascending=False)
plt.figure(figsize=(6, 4))
sns.barplot(x=nan_df.values, y=nan_df.index, palette='viridis')
plt.title('Tá»· lá»‡ GiÃ¡ trá»‹ khuyáº¿t (NaN) trong cÃ¡c Cá»™t', fontsize=16)
plt.xlabel('Tá»· lá»‡ (%)', fontsize=12)
plt.ylabel('TÃªn Cá»™t', fontsize=12)
plt.show()

# 2. PhÃ¢n tÃ­ch Biáº¿n Má»¥c tiÃªu (class)
plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df, palette={'p': 'red', 'e': 'green'})
plt.title('PhÃ¢n bá»‘ lá»›p (e = \'Ä‚n Ä‘Æ°á»£c\', p = \'Ä�á»™c\')', fontsize=15)
plt.show()
print()

# 3. PhÃ¢n bá»‘ cá»§a cÃ¡c Biáº¿n Sá»‘
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

plt.figure(figsize=(18, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'PhÃ¢n bá»‘ cá»§a {col}', fontsize=14)
plt.tight_layout()
plt.show()
print()

# 4. Ma tráº­n tÆ°Æ¡ng quan giá»¯a cÃ¡c Biáº¿n Sá»‘
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Ma tráº­n TÆ°Æ¡ng quan cá»§a 3 Cá»™t Sá»‘', fontsize=15)
plt.show()
print()

# 5. Má»‘i quan há»‡: Biáº¿n Sá»‘ vs. Má»¥c tiÃªu (Boxplots)
plt.figure(figsize=(18, 6))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x='class', y=col, data=df, palette={'p': 'red', 'e': 'green'})
    plt.title(f'{col} vá»›i Class (e = \'Ä‚n Ä‘Æ°á»£c\', p = \'Ä�á»™c\')', fontsize=14)
plt.tight_layout()
plt.show()
print('\n\n')

# 6. Má»‘i quan há»‡: Biáº¿n PhÃ¢n loáº¡i vs. Má»¥c tiÃªu (Countplots)
# Demo vá»›i 3 cá»™t phÃ¢n loáº¡i
cols_to_plot_cat = ['gill-color', 'habitat', 'cap-shape'] 

plt.figure(figsize=(18, 6))
for i, col in enumerate(cols_to_plot_cat):
    plt.subplot(1, 3, i + 1)
    # Láº¥y 10 giÃ¡ trá»‹ Ä‘áº§u Ä‘á»ƒ biá»ƒu Ä‘á»“ gá»�n gÃ ng
    order = df[col].value_counts().index[:10] 
    sns.countplot(data=df, x=col, hue='class', 
                  palette={'p': 'red', 'e': 'green'}, 
                  order=order)
    plt.title(f'PhÃ¢n bá»‘ {col} theo Class', fontsize=14)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 1. TÃ¡ch X, y (nhÆ°ng lÃ  X_raw, chÆ°a xá»­ lÃ½)
y_raw = df['class']
X_raw = df.drop('class', axis=1)
y = y_raw.map({'p': 1, 'e': 0})

# 2. TÃ¡ch (Split) TRÆ¯á»šC Ä‘á»ƒ tiáº¿t kiá»‡m bá»™ nhá»›
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# --- 3. Xá»¬ LÃ� X_train (Há»�c há»�i) ---
X_train = X_train_raw.copy()

# Ä�á»‹nh nghÄ©a cÃ¡c ngÆ°á»¡ng
DROP_THRESHOLD = 60.0
IMPUTE_THRESHOLD = 1.0

# LÆ°u trá»¯ cÃ¡c giÃ¡ trá»‹ há»�c Ä‘Æ°á»£c tá»« Train
impute_values = {}
cols_to_drop_final = []
cols_add_missing_flag = [] # DÃ¹ng Ä‘á»ƒ xá»­ lÃ½ X_test

# (Giáº£ Ä‘á»‹nh numerical_cols vÃ  categorical_cols Ä‘Ã£ Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a á»Ÿ Cell 1.3)

# --- 3a. Xá»­ lÃ½ NaN (Train) ---
# TÃ­nh % NaN chá»‰ má»™t láº§n trÃªn X_train
nan_percent_train = (X_train.isnull().sum() / len(X_train)) * 100

for col in X_train.columns:
    percent_missing = nan_percent_train[col]
    
    if percent_missing > DROP_THRESHOLD:
        cols_to_drop_final.append(col)
        
    elif percent_missing > IMPUTE_THRESHOLD: # Thiáº¿u vá»«a (1-60%)
        if col in categorical_cols:
            X_train[col] = X_train[col].fillna('Missing')
        elif col in numerical_cols:
            # Chiáº¿n lÆ°á»£c: Ä�iá»�n median VÃ€ thÃªm cá»� 'is_missing'
            median_val = X_train[col].median()
            impute_values[col] = median_val # LÆ°u láº¡i median
            cols_add_missing_flag.append(col) # Nhá»› tÃªn cá»™t nÃ y
            X_train[f'{col}_is_missing'] = X_train[col].isnull().astype(int) # Táº¡o cá»�
            X_train[col] = X_train[col].fillna(median_val) # Ä�iá»�n
            
    elif percent_missing > 0: # Thiáº¿u Ã­t (< 1%)
        if col in numerical_cols:
            median_val = X_train[col].median()
            impute_values[col] = median_val # LÆ°u láº¡i
            X_train[col] = X_train[col].fillna(median_val)
        elif col in categorical_cols:
            mode_val = X_train[col].mode()[0]
            impute_values[col] = mode_val # LÆ°u láº¡i
            X_train[col] = X_train[col].fillna(mode_val)

# XÃ³a cÃ¡c cá»™t Ä‘Ã£ xÃ¡c Ä‘á»‹nh
X_train = X_train.drop(columns=cols_to_drop_final)

# --- 3b. Chuáº©n hÃ³a (Scale) (Train) ---
# XÃ¡c Ä‘á»‹nh cÃ¡c cá»™t sá»‘ Má»šI (bao gá»“m cáº£ cá»� '_is_missing')
numerical_cols_new = [col for col in X_train.columns if X_train[col].dtype != 'object']
scaler = StandardScaler()
# Fit vÃ  Transform trÃªn X_train
X_train[numerical_cols_new] = scaler.fit_transform(X_train[numerical_cols_new])

# --- 3c. OHE (Train) ---
categorical_cols_new = [col for col in X_train.columns if X_train[col].dtype == 'object']
X_train = pd.get_dummies(X_train, columns=categorical_cols_new, drop_first=True)

# --- 4. Xá»¬ LÃ� X_test (Ã�p dá»¥ng) ---
X_test = X_test_raw.copy()

# --- 4a. Xá»­ lÃ½ NaN (Test) ---
# XÃ³a cÃ¡c cá»™t tÆ°Æ¡ng tá»±
X_test = X_test.drop(columns=cols_to_drop_final)

# Táº¡o cá»� 'is_missing' cho cÃ¡c cá»™t tÆ°Æ¡ng á»©ng (Sá»‘, 1-60%)
for col in cols_add_missing_flag:
    X_test[f'{col}_is_missing'] = X_test[col].isnull().astype(int)

# Ä�iá»�n khuyáº¿t (dÃ¹ng giÃ¡ trá»‹ tá»« `impute_values` hoáº·c 'Missing')
for col in X_test.columns:
    if col in impute_values: # (Sá»‘, <1%), (PhÃ¢n loáº¡i, <1%), (Sá»‘, 1-60%)
        X_test[col] = X_test[col].fillna(impute_values[col])
    elif col in categorical_cols and nan_percent_train.get(col, 0) > IMPUTE_THRESHOLD: # (PhÃ¢n loáº¡i, 1-60%)
         X_test[col] = X_test[col].fillna('Missing')

# An toÃ n: Ä�iá»�n 0 vÃ o cÃ¡c cá»™t sá»‘ cÃ²n láº¡i náº¿u lá»¡ bá»‹ NaN (hiáº¿m)
num_cols_test_temp = [c for c in X_test.columns if X_test[c].dtype != 'object']
X_test[num_cols_test_temp] = X_test[num_cols_test_temp].fillna(0)


# --- 4b. Chuáº©n hÃ³a (Scale) (Test) ---
# Ã�p dá»¥ng scaler Ä‘Ã£ fit (numerical_cols_new Ä‘Ã£ Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a tá»« X_train)
X_test[numerical_cols_new] = scaler.transform(X_test[numerical_cols_new])

# --- 4c. OHE (Test) ---
# (categorical_cols_new Ä‘Ã£ Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a tá»« X_train)
X_test = pd.get_dummies(X_test, columns=categorical_cols_new, drop_first=True)

# --- 5. Ä�á»’NG Bá»˜ Cá»˜T (Quan trá»�ng) ---
# Ä�áº£m báº£o X_train vÃ  X_test cÃ³ chÃ­nh xÃ¡c cÃ¡c cá»™t giá»‘ng nhau
X_train_cols = set(X_train.columns)
X_test_cols = set(X_test.columns)

missing_in_test = X_train_cols - X_test_cols
missing_in_train = X_test_cols - X_train_cols

# ThÃªm cá»™t cÃ²n thiáº¿u vÃ o X_test (nhanh hÆ¡n dÃ¹ng loop)
if missing_in_test:
    X_test = pd.concat(
        [X_test, pd.DataFrame(0, index=X_test.index, columns=list(missing_in_test))],
        axis=1
    )

# ThÃªm cá»™t cÃ²n thiáº¿u vÃ o X_train (hiáº¿m, nhÆ°ng Ä‘á»ƒ an toÃ n)
if missing_in_train:
    X_train = pd.concat(
        [X_train, pd.DataFrame(0, index=X_train.index, columns=list(missing_in_train))],
        axis=1
    )

# Sáº¯p xáº¿p láº¡i thá»© tá»± cá»™t cá»§a Test theo Train
X_test = X_test[X_train.columns]

# Gom bá»™ nhá»› (tÃ¹y chá»�n)
X_train = X_train.copy()
X_test = X_test.copy()

# --- 6. IN Káº¾T QUáº¢ CUá»�I CÃ™NG ---
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


import os

OUTPUT_DIR = '/kaggle/working/'
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Táº¡o thÆ° má»¥c náº¿u chÆ°a tá»“n táº¡i

file_paths = {
    "X_train": f"{OUTPUT_DIR}X_train_processed.parquet",
    "X_test": f"{OUTPUT_DIR}X_test_processed.parquet",
    "y_train": f"{OUTPUT_DIR}y_train_processed.csv",
    "y_test": f"{OUTPUT_DIR}y_test_processed.csv"
}

try:
    X_train.to_parquet(file_paths["X_train"])
    X_test.to_parquet(file_paths["X_test"])

    y_train.to_csv(file_paths["y_train"], index=False)
    y_test.to_csv(file_paths["y_test"], index=False)

    print(f"{OUTPUT_DIR}\n")

    for name, path in file_paths.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"ğŸ“� {os.path.basename(path):30s}  ({size_mb:6.2f} MB)")

except Exception as e:
    print(f"Lá»—i khi lÆ°u file: {e}")


