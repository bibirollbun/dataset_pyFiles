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


from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import recall_score, precision_score


def eval(model, X_val, y_val):
    y_pred = model.predict(X_val)
    # ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹
    y_pred_proba = model.predict_proba(X_val)[:, 1]


    # Recall
    recall = recall_score(y_val, y_pred)
    print(f"Recall: {recall:.4f}")

    # Precision
    precision = precision_score(y_val, y_pred)
    print(f"Precision: {precision:.4f}")

    # ROC-AUC
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    print(f'ROC-AUC: {roc_auc:.4f}')

    # PR-AUC
    pr_auc = average_precision_score(y_val, y_pred_proba)
    print(f'PR-AUC: {pr_auc:.4f}')

    # --- ROC-AUC ---
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--')  # Ğ´Ğ¸Ğ°Ğ³Ğ¾Ğ½Ğ°Ğ»ÑŒ
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # --- PR-AUC ---
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)

    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.4f})', color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()



def show_feature_details(df, col, top=10):
    print(f'\nĞ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ°: {col}')
    print(df[col].value_counts(normalize=True).head(top))

    fature_counts = df[col].value_counts(normalize=True) * 100

    # ĞŸĞ¾Ñ�Ñ‚Ñ€Ğ¾ĞµĞ½Ğ¸Ğµ ĞºÑ€ÑƒĞ³Ğ¾Ğ²Ğ¾Ğ¹ Ğ´Ğ¸Ğ°Ğ³Ñ€Ğ°Ğ¼Ğ¼Ñ‹
    plt.figure(figsize=(10,10))
    plt.pie(fature_counts, labels=fature_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ¿Ğ¾ {col}')
    plt.axis('equal')
    plt.show()


    fraud_rate = df.groupby(col)['isFraud'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12,4))
    fraud_rate.head(top).plot(kind='bar')
    plt.title(f'Ğ”Ğ¾Ğ»Ñ� Ğ¼Ğ¾ÑˆĞµĞ½Ğ½Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ğ¿Ğ¾ {col}')
    plt.ylabel('Ğ”Ğ¾Ğ»Ñ� Ğ¼Ğ¾ÑˆĞµĞ½Ğ½Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ°')
    plt.show()


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

# 3. Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Ğ“Ğ¸Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ğ° Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ² Ğ¿Ğ¾ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ°Ğ¼
ax[0].hist(missing_col, bins=30, color='skyblue', edgecolor='black')
ax[0].set_title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ² Ğ¿Ğ¾ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ°Ğ¼')
ax[0].set_xlabel('Ğ”Ğ¾Ğ»Ñ� Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ğ½Ñ‹Ñ… Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹')
ax[0].set_ylabel('ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº')

# Ğ“Ğ¸Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ğ° Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ² Ğ¿Ğ¾ Ñ�Ñ‚Ñ€Ğ¾ĞºĞ°Ğ¼
ax[1].hist(missing_row, bins=30, color='salmon', edgecolor='black')
ax[1].set_title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ² Ğ¿Ğ¾ Ñ�Ñ‚Ñ€Ğ¾ĞºĞ°Ğ¼')
ax[1].set_xlabel('Ğ”Ğ¾Ğ»Ñ� Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ğ½Ñ‹Ñ… Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹')
ax[1].set_ylabel('ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ�Ñ‚Ñ€Ğ¾Ğº')

plt.tight_layout()
plt.show()



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



df_cleaned['TransactionDT'].head()


import matplotlib.pyplot as plt
import seaborn as sns

# ĞŸĞ¾Ñ�Ğ¼Ğ¾Ñ‚Ñ€Ğ¸Ğ¼ Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ¹ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ¾Ğ¹
print(df_cleaned['isFraud'].value_counts(normalize=True))

# Ğ’Ñ‹Ğ±ĞµÑ€ĞµĞ¼ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ğ´Ğ»Ñ� ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¸
numeric_features = df_cleaned.select_dtypes(include=[np.number])

# ĞŸĞ¾Ñ�Ñ‡Ğ¸Ñ‚Ğ°ĞµĞ¼ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� Ñ� Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ¹ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ¾Ğ¹
correlation_matrix = numeric_features.corr()

# ĞšĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ñ� Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ¹ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ¾Ğ¹
cor_target = correlation_matrix['isFraud'].sort_values(ascending=False)
print("Ğ¢Ğ¾Ğ¿ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ², Ğ½Ğ°Ğ¸Ğ±Ğ¾Ğ»ĞµĞµ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ğ¸Ñ€ÑƒÑ�Ñ‰Ğ¸Ñ… Ñ� isFraud:\n", cor_target.head(15))



# Ğ¢ĞµĞ¿Ğ»Ğ¾Ğ²Ğ°Ñ� ĞºĞ°Ñ€Ñ‚Ğ° Ğ²Ñ�ĞµĞ¹ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğ¹ Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ñ‹
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('ĞŸĞ¾Ğ»Ğ½Ğ°Ñ� ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ°Ñ� Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ğ° Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²', fontsize=18)
plt.show()


# Ñ‚ĞµĞ¿Ğ»Ğ¾Ğ²Ğ°Ñ� ĞºĞ°Ñ€Ñ‚Ğ° Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ñ‚Ğ¾Ğ¿-30 Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
top_features = cor_target.index[1:31]  # Ğ¸Ñ�ĞºĞ»Ñ�Ñ‡Ğ°ĞµĞ¼ Ñ�Ğ°Ğ¼ isFraud
plt.figure(figsize=(14, 12))
sns.heatmap(train[top_features].corr(), cmap='coolwarm', center=0, annot=False, linewidths=0.5)
plt.title('ĞšĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ°Ñ� Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ğ° Ñ‚Ğ¾Ğ¿-30 Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²', fontsize=18)
plt.show()


numeric_features = df_cleaned.select_dtypes(include=['number']).columns
categorical_features = df_cleaned.select_dtypes(include=['object', 'category']).columns

# Ğ Ğ°Ğ·Ğ¼ĞµÑ€Ñ‹ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğ°
print(f'Ğ Ğ°Ğ·Ğ¼ĞµÑ€Ñ‹ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğ°: {df_cleaned.shape[0]} Ñ�Ñ‚Ñ€Ğ¾Ğº, {df_cleaned.shape[1]} Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ².')

# Ğ”Ğ¾Ğ»Ğ¸ ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ²
print(df_cleaned['isFraud'].value_counts(normalize=True))

# Ğ”Ğ¾Ğ»Ñ� Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¾Ğ² Ğ¿Ğ¾ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ°Ğ¼
missing_values = df_cleaned.isnull().mean().sort_values(ascending=False)
print(missing_values[missing_values > 0])

# Ğ¢Ğ¸Ğ¿Ñ‹ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
print('ĞšĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸:', len(categorical_features))
print('Ğ§Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸:', len(numeric_features))



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


# ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ²Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸
plt.figure(figsize=(14,6))
df_modified.groupby('Transaction_day').size().plot()
plt.title('ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ¿Ğ¾ Ğ´Ğ½Ñ�Ğ¼')
plt.xlabel('Ğ”ĞµĞ½ÑŒ')
plt.ylabel('ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹')
plt.show()


# ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¼Ğ¾ÑˆĞµĞ½Ğ½Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ² Ğ²Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸
plt.figure(figsize=(14,6))
df_modified[df_modified['isFraud'] == 1].groupby('Transaction_day').size().plot()
plt.title('ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¼Ğ¾ÑˆĞµĞ½Ğ½Ğ¸Ñ‡ĞµÑ�ĞºĞ¸Ñ… Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ¿Ğ¾ Ğ´Ğ½Ñ�Ğ¼')
plt.xlabel('Ğ”ĞµĞ½ÑŒ')
plt.ylabel('ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¼Ğ¾ÑˆĞµĞ½Ğ½Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²')
plt.show()



# Ğ’Ñ€ĞµĞ¼Ñ� Ñ�ÑƒÑ‚Ğ¾Ğº
plt.figure(figsize=(14,6))
sns.histplot(data=df_modified, x='Transaction_hour', hue='isFraud', bins=24, kde=True, multiple='stack')
plt.title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ¿Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸ Ñ�ÑƒÑ‚Ğ¾Ğº')
plt.xlabel('Ğ§Ğ°Ñ� Ğ´Ğ½Ñ�')
plt.show()


df_modified[['TransactionDT', 'Relative_TransactionDT','Transaction_day', 'Transaction_hour']]


# Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ�ÑƒĞ¼Ğ¼ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹
plt.figure(figsize=(14,6))
sns.histplot(data=df_modified, x='TransactionAmt', hue='isFraud', bins=100, log_scale=True, kde=True)
plt.title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ�ÑƒĞ¼Ğ¼Ñ‹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ² Ğ·Ğ°Ğ²Ğ¸Ñ�Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸ Ğ¾Ñ‚ Ğ¼Ğ¾ÑˆĞµĞ½Ğ½Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ°')
plt.xlabel('Ğ¡ÑƒĞ¼Ğ¼Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸')
plt.show()

# Ğ¢Ğ¾Ğ¿-10 Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ğ¿Ğ¾ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¸ Ñ� isFraud
fraud_corr = df_modified[numeric_features].corr()['isFraud'].drop('isFraud').abs().sort_values(ascending=False)
print('Ğ¢Ğ¾Ğ¿-10 Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ğ¿Ğ¾ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¸ Ñ� isFraud:')
print(fraud_corr.head(10))



col = 'ProductCD'
show_feature_details(train, col)


train['card1'].unique()


col = 'card1'
show_feature_details(train, col)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ğ¡Ñ‡Ğ¸Ñ‚Ğ°ĞµĞ¼ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ¿Ğ¾ ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¼Ñƒ card1
card1_counts = df_modified['card1'].value_counts()

# Ğ§Ñ‚Ğ¾Ğ±Ñ‹ Ğ¸Ğ·Ğ±ĞµĞ¶Ğ°Ñ‚ÑŒ Ğ¿Ñ€Ğ¾Ğ±Ğ»ĞµĞ¼ Ñ� "unknown" (Ğ³Ğ´Ğµ 0 Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹)
card1_counts = card1_counts[card1_counts > 0]

# Ğ›Ğ¾Ğ³Ğ°Ñ€Ğ¸Ñ„Ğ¼Ğ¸Ñ€ÑƒĞµĞ¼ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ´Ğ»Ñ� Ğ±Ğ¾Ğ»ĞµĞµ Ğ½Ğ°Ğ³Ğ»Ñ�Ğ´Ğ½Ğ¾Ğ³Ğ¾ Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ°
plt.figure(figsize=(12, 6))
sns.histplot(np.log1p(card1_counts), bins=100, kde=True)
plt.title('Ğ›Ğ¾Ğ³-Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ¿Ğ¾ card1')
plt.xlabel('log1p(ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹)')
plt.ylabel('Ğ§Ğ¸Ñ�Ğ»Ğ¾ card1')
plt.grid(True)
plt.show()



col = 'card2'
show_feature_details(train, col)


col = 'card3'
show_feature_details(train, col)


col = 'card4'
show_feature_details(train, col)


col = 'card5'
show_feature_details(train, col)


col = 'card6'
show_feature_details(train, col)


col = 'addr1'
show_feature_details(train, col)


col = 'addr2'
show_feature_details(train, col)


col = 'P_emaildomain'
show_feature_details(train, col)


col = 'R_emaildomain'
show_feature_details(train, col)


col = 'DeviceType'
show_feature_details(train, col)


col = 'DeviceInfo'
show_feature_details(train, col)


exclude = [
    'ProductCD', 
    'card1', 
    'card2', 
    'card3', 
    'card4', 
    'card5', 
    'card6', 
    'addr1', 
    'addr2', 
    'P_emaildomain', 
    'R_emaildomain', 
    'DeviceType',
    'DeviceInfo'
]

for col in categorical_features:
    if col in exclude:
        continue
    
    print(f'\nĞ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ°: {col}')
    print(train[col].value_counts(normalize=True).head(10))

    fraud_rate = train.groupby(col)['isFraud'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12,4))
    fraud_rate.head(10).plot(kind='bar')
    plt.title(f'Ğ”Ğ¾Ğ»Ñ� Ğ¼Ğ¾ÑˆĞµĞ½Ğ½Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ğ¿Ğ¾ {col}')
    plt.ylabel('Ğ”Ğ¾Ğ»Ñ� Ğ¼Ğ¾ÑˆĞµĞ½Ğ½Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ°')
    plt.show()



col = 'dist1'
plt.figure(figsize=(14,6))
sns.kdeplot(data=train, x=col, hue='isFraud', common_norm=False)
plt.title(f'Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ {col} Ğ´Ğ»Ñ� Ñ€Ğ°Ğ·Ğ½Ñ‹Ñ… ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ²')
plt.show()



col = 'dist2'
plt.figure(figsize=(14,6))
sns.kdeplot(data=train, x=col, hue='isFraud', common_norm=False)
plt.title(f'Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ {col} Ğ´Ğ»Ñ� Ñ€Ğ°Ğ·Ğ½Ñ‹Ñ… ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ²')
plt.show()



exclude = [
    'isFraud',
    'TransactionID',
    'TransactionDT',
    'dist1',
    'dist2', 
]

for col in numeric_features:
    if col in exclude:
        continue

    
    plt.figure(figsize=(14,6))
    sns.kdeplot(data=train, x=col, hue='isFraud', common_norm=False)
    plt.title(f'Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ {col} Ğ´Ğ»Ñ� Ñ€Ğ°Ğ·Ğ½Ñ‹Ñ… ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ²')
    plt.show()



# ĞšĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ°Ñ� Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ğ° Ğ²Ñ�ĞµÑ… Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
plt.figure(figsize=(20, 16))
corr_matrix = train[numeric_features].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('ĞšĞ°Ñ€Ñ‚Ğ° ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¹ Ğ²Ñ�ĞµÑ… Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²')
plt.show()

# ĞšĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ñ� isFraud
plt.figure(figsize=(12,8))
fraud_corr.head(20).plot(kind='barh')
plt.title('ĞšĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ñ� Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ñ� isFraud')
plt.show()



key_features = ['TransactionAmt', 'dist1', 'dist2', 'D1', 'D2']
for col in key_features:
    if col in train.columns:
        plt.figure(figsize=(12,6))
        sns.boxplot(x='isFraud', y=col, data=train)
        plt.title(f'Boxplot {col} Ğ¿Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ğ¼')
        plt.show()



#for col in categorical_features:
#    rare_values = train[col].value_counts()[train[col].value_counts() < 10].index
#    train[col] = train[col].apply(lambda x: 'rare' if x in rare_values else x)
#    print(f'ĞŸÑ€Ğ¸Ğ·Ğ½Ğ°Ğº {col}: Ğ·Ğ°Ğ¼ĞµĞ½ĞµĞ½Ğ¾ {len(rare_values)} Ñ€ĞµĞ´ĞºĞ¸Ñ… ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¹ Ğ½Ğ° "rare"')



# Ğ½Ğ° Ğ²Ñ�Ñ�ĞºĞ¸Ğ¹ Ñ�Ğ»ÑƒÑ‡Ğ°Ğ¹ Ğ¾Ñ‚Ñ�Ğ¾Ñ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¿Ğ¾ TransactionDT
train = train.sort_values('TransactionDT')

# Ğ�Ğ°Ğ¹Ğ´ĞµĞ¼ Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ
min_timestamp = train['TransactionDT'].min()

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ½Ñ‹Ğµ Ğ¼ĞµÑ‚ĞºĞ¸ Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğ¹
train['Relative_TransactionDT'] = train['TransactionDT'] - min_timestamp

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ² Ğ´Ğ½Ğ¸, Ñ‡Ğ°Ñ�Ñ‹, Ğ¼Ğ¸Ğ½ÑƒÑ‚Ñ‹ Ğ¸ Ñ‚.Ğ´.
train['Transaction_day'] = train['Relative_TransactionDT'] // (24 * 60 * 60)  # Ğ² Ğ´Ğ½Ñ�Ñ…
train['Transaction_hour'] = (train['Relative_TransactionDT'] // 3600) % 24  # Ğ² Ñ‡Ğ°Ñ�Ğ°Ñ…
train['Transaction_weekday'] = (train['Relative_TransactionDT'] // (3600*24)) % 7
train['Transaction_day'] = train['Transaction_day'].astype(int)
train['Transaction_hour'] = train['Transaction_hour'].astype(int)
train['Transaction_weekday'] = train['Transaction_weekday'].astype(int)

# Ğ�Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ TransactionDT
#train['weekday'] = (train['TransactionDT'] // (3600*24)) % 7
#train['hour'] = (train['TransactionDT'] // 3600) % 24




# Ğ“Ñ€ÑƒĞ¿Ğ¿Ğ¸Ñ€Ğ¾Ğ²ĞºĞ° email-Ğ´Ğ¾Ğ¼ĞµĞ½Ğ¾Ğ²
#email_domains = {
#    'google': 'gmail', 'gmail': 'gmail', 'att.net': 'other', 'twc.com': 'other',
#    'scranton.edu': 'other', 'verizon.net': 'other', 'protonmail.com': 'other',
#    'aol.com': 'aol', 'hotmail.com': 'hotmail', 'yahoo.com': 'yahoo', 
#    'yahoo.com.mx': 'yahoo', 'outlook.com': 'microsoft', 'icloud.com': 'apple'
#}
#for col in ['P_emaildomain', 'R_emaildomain']:
#    if col in train.columns:
#        train[col] = train[col].apply(lambda x: email_domains.get(x.split('.')[0], 'other'))


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ĞŸĞ¾Ñ�Ñ‚Ñ€Ğ¾Ğ¸Ğ¼ boxplot Ğ´Ğ»Ñ� Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ° Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¾Ğ² Ğ² TransactionAMT
plt.figure(figsize=(10, 6))
sns.boxplot(x=train['TransactionAmt'])
plt.title('Boxplot Ğ´Ğ»Ñ� Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ° TransactionAMT')
plt.show()

# Ğ Ğ°Ñ�Ñ�Ñ‡Ğ¸Ñ‚Ğ°ĞµĞ¼ Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºÑƒ Ğ´Ğ»Ñ� Ğ²Ñ‹Ñ�Ğ²Ğ»ĞµĞ½Ğ¸Ñ� Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¾Ğ²
q1 = train['TransactionAmt'].quantile(0.25)  # 25-Ğ¹ Ğ¿ĞµÑ€Ñ†ĞµĞ½Ñ‚Ğ¸Ğ»ÑŒ
q3 = train['TransactionAmt'].quantile(0.75)  # 75-Ğ¹ Ğ¿ĞµÑ€Ñ†ĞµĞ½Ñ‚Ğ¸Ğ»ÑŒ
iqr = q3 - q1  # Ğ˜Ğ½Ñ‚ĞµÑ€ĞºĞ²Ğ°Ñ€Ñ‚Ğ¸Ğ»ÑŒĞ½Ñ‹Ğ¹ Ñ€Ğ°Ğ·Ğ¼Ğ°Ñ…

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"Ğ“Ñ€Ğ°Ğ½Ğ¸Ñ†Ñ‹ Ğ´Ğ»Ñ� Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¾Ğ²: {lower_bound:.2f} - {upper_bound:.2f}")

# Ğ’Ñ‹Ğ±Ñ€Ğ¾Ñ�Ñ‹ â€” Ñ�Ñ‚Ğ¾ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ�, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ğ»ĞµĞ¶Ğ°Ñ‚ Ğ·Ğ° Ğ¿Ñ€ĞµĞ´ĞµĞ»Ğ°Ğ¼Ğ¸ Ñ�Ñ‚Ğ¸Ñ… Ğ³Ñ€Ğ°Ğ½Ğ¸Ñ†
outliers = train[(train['TransactionAmt'] < lower_bound) | (train['TransactionAmt'] > upper_bound)]
print(f"Ğ§Ğ¸Ñ�Ğ»Ğ¾ Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¾Ğ² Ğ² TransactionAmt: {outliers.shape[0]}")
print(outliers[['TransactionAmt', 'isFraud']].head())



# Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ñ‹ Ğ¿Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ñƒ isFraud
outliers_fraud = outliers['isFraud'].value_counts(normalize=True)
print(f"Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² Ñ�Ñ€ĞµĞ´Ğ¸ Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¾Ğ²: {outliers_fraud}")



# Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¾Ğ² Ğ½Ğ° Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞµ
plt.figure(figsize=(12, 6))
sns.histplot(train[train['isFraud'] == 0]['TransactionAmt'], kde=True, color='blue', label='Ğ§ĞµÑ�Ñ‚Ğ½Ñ‹Ğµ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸')
sns.histplot(train[train['isFraud'] == 1]['TransactionAmt'], kde=True, color='red', label='ĞœĞ¾ÑˆĞµĞ½Ğ½Ğ¸Ñ‡ĞµÑ�ĞºĞ¸Ğµ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸')
plt.axvline(x=lower_bound, color='black', linestyle='--', label='Ğ�Ğ¸Ğ¶Ğ½Ñ�Ñ� Ğ³Ñ€Ğ°Ğ½Ğ¸Ñ†Ğ° Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ°')
plt.axvline(x=upper_bound, color='black', linestyle='--', label='Ğ’ĞµÑ€Ñ…Ğ½Ñ�Ñ� Ğ³Ñ€Ğ°Ğ½Ğ¸Ñ†Ğ° Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ°')
plt.legend()
plt.title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ�ÑƒĞ¼Ğ¼Ñ‹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ¿Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°Ğ¼')
plt.show()



import numpy as np

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ° TransactionAMT
df_modified['log_TransactionAmt'] = np.log1p(df_modified['TransactionAmt'])



bins = [0, 100, 1000, 5000, 10000, np.inf]
labels = ['Low', 'Medium', 'High', 'Very High', 'Extremely High']

df_modified['TransactionAmt_binned'] = pd.cut(df_modified['TransactionAmt'], bins=bins, labels=labels)

# Ğ�Ğ»ÑŒÑ‚ĞµÑ€Ğ½Ğ°Ñ‚Ğ¸Ğ²Ğ½Ñ‹Ğ¹ Ñ�Ğ¿Ğ¾Ñ�Ğ¾Ğ±:  Ğ‘Ğ¸Ğ½Ğ½Ğ¸Ğ½Ğ³ Ñ�ÑƒĞ¼Ğ¼Ñ‹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸ Ğ¿Ğ¾ ĞºĞ²Ğ°Ğ½Ñ‚Ğ¸Ğ»Ñ�Ğ¼
#train['TransactionAmt_bin'] = pd.qcut(train['TransactionAmt'], q=10, duplicates='drop')



df_modified['isOutlier'] = ((train['TransactionAmt'] < lower_bound) | (train['TransactionAmt'] > upper_bound)).astype(int)


df_modified[['TransactionAmt', 'log_TransactionAmt', 'TransactionAmt_binned']].describe()


# Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ�ÑƒĞ¼Ğ¼ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹
plt.figure(figsize=(14,6))
sns.histplot(data=df_modified, x='log_TransactionAmt', hue='isFraud', bins=100, log_scale=True, kde=True)
plt.title('Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ�ÑƒĞ¼Ğ¼Ñ‹ Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¹ Ğ² Ğ·Ğ°Ğ²Ğ¸Ñ�Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸ Ğ¾Ñ‚ Ğ¼Ğ¾ÑˆĞµĞ½Ğ½Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ°')
plt.xlabel('Log Ğ¡ÑƒĞ¼Ğ¼Ğ° Ñ‚Ñ€Ğ°Ğ½Ğ·Ğ°ĞºÑ†Ğ¸Ğ¸')
plt.show()



#base_columns = [
#    'isFraud',
#    'log_TransactionAmt',
#    'TransactionAmt_binned',
#    'ProductCD',
#    'card4',
#    'card6',
#    'Transaction_hour', 
#    'Transaction_weekday',
#    'P_emaildomain',
#    #'DeviceType',
#    #'DeviceInfo'
#]

#df_modified[base_columns].head()


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


y_train.head()


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


eval(model, X_val, y_val)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# ĞŸĞ¾Ğ»ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
importances = model.feature_importances_

# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ñ�Ğ¿Ğ¸Ñ�ĞºĞ° Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ñ� Ğ¸Ñ… Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ñ�Ğ¼Ğ¸
feature_names = X_train.columns  # Ğ¸Ğ»Ğ¸ Ğ¿Ñ€Ğ¾Ñ�Ñ‚Ğ¾ Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ¸Ğ¼ĞµĞ½ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
important_features = list(zip(feature_names, importances))

# Ğ¡Ğ¾Ñ€Ñ‚Ğ¸Ñ€Ğ¾Ğ²ĞºĞ° Ğ¿Ğ¾ Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸
important_features.sort(key=lambda x: x[1], reverse=True)

# Ğ�Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğµ Ğ½Ğ°Ğ¸Ğ±Ğ¾Ğ»ĞµĞµ Ğ²Ğ°Ğ¶Ğ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
for feature in important_features:
    print(f"Feature: {feature[0]}, Importance: {feature[1]}")


eval(model, X_val, y_val)


import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score

# Ğ¡Ğ¾Ñ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ¿Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸
#train = train.sort_values('TransactionDT')

#X = train.drop(columns=['isFraud'])
#y = train['isFraud']

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    # ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Random Forest
    params = {
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),  # Ğ¸Ñ�Ğ¿Ñ€Ğ°Ğ²Ğ»ĞµĞ½Ğ¾
        'random_state': 42,
        'n_jobs': -1,
    }
    
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°Ñ‘Ğ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ
    model = RandomForestClassifier(**params)
    
    # ĞšÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ� Ğ¿Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='roc_auc')
    
    # Ğ�Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ ROC AUC
    return scores.mean()

# Ğ�Ğ¿Ñ‚Ğ¸Ğ¼Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
sampler = optuna.samplers.TPESampler(seed=42)  # Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼ seed Ğ´Ğ»Ñ� Ğ²Ğ¾Ñ�Ğ¿Ñ€Ğ¾Ğ¸Ğ·Ğ²Ğ¾Ğ´Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50)

print('Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:', study.best_params)




#params = {'n_estimators': 801, 'max_depth': 12, 'min_samples_split': 17, 'min_samples_leaf': 11, 'max_features': 'log2', 'random_state': 42}
#params = {'n_estimators': 538, 'max_depth': 12, 'min_samples_split': 4, 'min_samples_leaf': 19, 'max_features': 'sqrt', 'random_state': 42}
#params = {'n_estimators': 895, 'max_depth': 12, 'min_samples_split': 20, 'min_samples_leaf': 13, 'max_features': 'sqrt'}
#params = {'n_estimators': 821, 'max_depth': 19, 'min_samples_split': 9, 'min_samples_leaf': 5, 'max_features': 'sqrt'}
#params = {'n_estimators': 470, 'max_depth': 20, 'min_samples_split': 9, 'min_samples_leaf': 4, 'max_features': 'sqrt'}
params = {'class_weight': 'balanced', 'n_estimators': 742, 'max_depth': 20, 'min_samples_split': 17, 'min_samples_leaf': 19, 'max_features': 'sqrt'}
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)


# ĞŸĞ¾Ğ»ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
importances = model.feature_importances_

# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ñ�Ğ¿Ğ¸Ñ�ĞºĞ° Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ñ� Ğ¸Ñ… Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ñ�Ğ¼Ğ¸
feature_names = X_train.columns  # Ğ¸Ğ»Ğ¸ Ğ¿Ñ€Ğ¾Ñ�Ñ‚Ğ¾ Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ¸Ğ¼ĞµĞ½ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
important_features = list(zip(feature_names, importances))

# Ğ¡Ğ¾Ñ€Ñ‚Ğ¸Ñ€Ğ¾Ğ²ĞºĞ° Ğ¿Ğ¾ Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸
important_features.sort(key=lambda x: x[1], reverse=True)

# Ğ�Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğµ Ğ½Ğ°Ğ¸Ğ±Ğ¾Ğ»ĞµĞµ Ğ²Ğ°Ğ¶Ğ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
for feature in important_features:
    print(f"Feature: {feature[0]}, Importance: {feature[1]}")


eval(model, X_val, y_val)


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

#
y_proba = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# Ğ¤ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ğ´Ğ»Ñ� Ğ²Ñ‹Ğ±Ğ¾Ñ€Ğ° Ğ¿Ğ¾Ñ€Ğ¾Ğ³Ğ°
def find_threshold_for_recall(y_true, y_scores, target_recall=0.95):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    recall = recall[:-1]
    precision = precision[:-1]

    mask = recall >= target_recall
    if not np.any(mask):
        return None
    
    best_idx = np.argmax(precision[mask])
    return thresholds[mask][best_idx]

optimal_threshold = find_threshold_for_recall(y_val, y_proba, target_recall=0.95)
print('Ğ�Ğ¿Ñ‚Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Ğ¿Ğ¾Ñ€Ğ¾Ğ³ Ğ´Ğ»Ñ� Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ³Ğ¾ recall:', optimal_threshold)


from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
import optuna
import numpy as np

#
optimal_threshold = find_threshold_for_recall(y_val, y_proba, target_recall=0.95)
print('Ğ�Ğ¿Ñ‚Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Ğ¿Ğ¾Ñ€Ğ¾Ğ³ Ğ´Ğ»Ñ� Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ³Ğ¾ recall:', optimal_threshold)


def custom_recall_with_threshold(y_true, y_proba, optimal_threshold=0.5, **kwargs):
    """
    Ğ Ğ°Ñ�Ñ�Ñ‡ĞµÑ‚ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸ recall (Ğ¿Ğ¾Ğ»Ğ½Ğ¾Ñ‚Ñ‹) Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ Ğ·Ğ°Ğ´Ğ°Ğ½Ğ½Ğ¾Ğ³Ğ¾ Ğ¿Ğ¾Ñ€Ğ¾Ğ³Ğ° Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹.

    Parameters:
    y_true (array-like): Ğ˜Ñ�Ñ‚Ğ¸Ğ½Ğ½Ñ‹Ğµ Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ñ‹Ğµ Ğ¼ĞµÑ‚ĞºĞ¸.
    y_proba (array-like): ĞŸÑ€Ğ¾Ğ³Ğ½Ğ¾Ğ·Ğ¸Ñ€ÑƒĞµĞ¼Ñ‹Ğµ Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚Ğ¸.
    optimal_threshold (float): ĞŸĞ¾Ñ€Ğ¾Ğ³ Ğ´Ğ»Ñ� Ğ¿Ñ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ñ� Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹ Ğ² Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ñ‹Ğµ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ�.
    
    Returns:
    float: Recall score.
    """
    # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ² Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ²
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # Ğ’Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµĞ¼ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºÑƒ recall
    return recall_score(y_true, y_pred)


# Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼ TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

n_jobs = -1

def objective(trial):
    params = {
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42,
        'n_jobs': n_jobs,
    }

    model = RandomForestClassifier(**params)
    


    try:
        # Ğ�Ñ†ĞµĞ½ĞºĞ° Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
        #print(f'##### Ğ�Ñ†ĞµĞ½ĞºĞ° Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸')
        scorer = make_scorer(custom_recall_with_threshold, needs_proba=True, optimal_threshold=optimal_threshold)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring=scorer, n_jobs=n_jobs)
        print(f"Trial completed with mean score: {np.nanmean(scores)}")
        return np.nanmean(scores)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return np.nan

# Ğ—Ğ°Ğ¿ÑƒÑ�Ğº Optuna
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50)

print('Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:', study.best_params)


#params = {'n_estimators': 400, 'max_depth': 20, 'min_samples_split': 17, 'min_samples_leaf': 12, 'max_features': 'sqrt'}
#params = {'n_estimators': 418, 'max_depth': 20, 'min_samples_split': 15, 'min_samples_leaf': 1, 'max_features': 'sqrt'}
#params = {'class_weight': 'balanced_subsample', 'n_estimators': 259, 'max_depth': 3, 'min_samples_split': 6, 'min_samples_leaf': 7, 'max_features': 'log2'}
#params = {'class_weight': 'balanced_subsample', 'n_estimators': 751, 'max_depth': 6, 'min_samples_split': 15, 'min_samples_leaf': 17, 'max_features': 'log2'}
params = {'class_weight': 'balanced_subsample', 'n_estimators': 751, 'max_depth': 6, 'min_samples_split': 15, 'min_samples_leaf': 17, 'max_features': 'log2'}
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)


eval(model, X_val, y_val)




