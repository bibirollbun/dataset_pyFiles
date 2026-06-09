!pip install lifelines


!conda install metric


# Ğ˜Ğ¼Ğ¿Ğ¾Ñ€Ñ‚ Ğ²Ñ�ĞµÑ… Ğ½ĞµĞ¾Ğ±Ñ…Ğ¾Ğ´Ğ¸Ğ¼Ñ‹Ñ… Ğ±Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞº
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import warnings

# Ğ£Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²ĞºĞ° Ñ�Ñ‚Ğ¸Ğ»Ñ� Ğ´Ğ»Ñ� Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¾Ğ²
sns.set(style="whitegrid")

# Ğ�Ñ‚ĞºĞ»Ñ�Ñ‡ĞµĞ½Ğ¸Ğµ Ğ¿Ñ€ĞµĞ´ÑƒĞ¿Ñ€ĞµĞ¶Ğ´ĞµĞ½Ğ¸Ğ¹ Ğ´Ğ»Ñ� ÑƒĞ´Ğ¾Ğ±Ñ�Ñ‚Ğ²Ğ° Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ°
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ° Ñ�Ñ‚Ğ¸Ğ»Ñ� Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¾Ğ²
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# ĞŸĞ¾Ñ�Ñ‚Ñ€Ğ¾ĞµĞ½Ğ¸Ğµ Ğ³Ğ¸Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ñ‹ Ğ´Ğ»Ñ� Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸ Ñ�Ğ¾Ğ±Ñ‹Ñ‚Ğ¸Ñ� Ñ� ÑƒĞ»ÑƒÑ‡ÑˆĞµĞ½Ğ¸ĞµĞ¼
plt.figure(figsize=(10, 6))  # Ğ Ğ°Ğ·Ğ¼ĞµÑ€ Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ°
sns.histplot(train.loc[train.efs == 1, "efs_time"], bins=100, kde=True, label="efs=1, Ğ¡Ğ¾Ğ±Ñ‹Ñ‚Ğ¸Ğµ Ğ¿Ñ€Ğ¾Ğ¸Ğ·Ğ¾ÑˆĞ»Ğ¾", color="royalblue", stat="density", linewidth=1.5)
sns.histplot(train.loc[train.efs == 0, "efs_time"], bins=100, kde=True, label="efs=0, Ğ’Ğ¾Ğ·Ğ¼Ğ¾Ğ¶Ğ½Ğ¾ Ñ�Ğ¾Ğ±Ñ‹Ñ‚Ğ¸Ğµ", color="tomato", stat="density", linewidth=1.5)

# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ° Ğ·Ğ°Ğ³Ğ¾Ğ»Ğ¾Ğ²ĞºĞ¾Ğ² Ğ¸ Ğ¿Ğ¾Ğ´Ğ¿Ğ¸Ñ�ĞµĞ¹
plt.xlabel("Ğ’Ñ€ĞµĞ¼Ñ� Ğ½Ğ°Ğ±Ğ»Ñ�Ğ´ĞµĞ½Ğ¸Ñ� (efs_time)", fontsize=14)
plt.ylabel("ĞŸĞ»Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ñ‚ÑŒ", fontsize=14)
plt.title("ğŸ“Š Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸ Ğ´Ğ¾ Ñ�Ğ¾Ğ±Ñ‹Ñ‚Ğ¸Ñ�", fontsize=16, pad=20)
plt.legend(title="Ğ¡Ğ¾Ğ±Ñ‹Ñ‚Ğ¸Ğµ", fontsize=12)

# Ğ£Ğ±Ğ¸Ñ€Ğ°ĞµĞ¼ Ğ»Ğ¸ÑˆĞ½Ğ¸Ğµ Ñ�ĞµÑ‚ĞºĞ¸ Ğ¸ Ğ´Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ»ĞµĞ³ĞºĞ¸Ğµ Ğ»Ğ¸Ğ½Ğ¸Ğ¸
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# Ğ˜Ğ¼Ğ¿Ğ¾Ñ€Ñ‚ KaplanMeierFitter Ğ´Ğ»Ñ� Ğ¿Ñ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ñ�
from lifelines import KaplanMeierFitter

# Ğ¤ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ğ´Ğ»Ñ� Ğ¿Ñ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ñ� Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ²Ñ‹Ğ¶Ğ¸Ğ²Ğ°Ğ½Ğ¸Ñ�
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y

# ĞŸÑ€Ğ¸Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ğ¸ Ğ¸ Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ğ½Ğ¾Ğ²Ğ¾Ğ¹ Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ¹ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ¾Ğ¹
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

# ĞŸĞ¾Ñ�Ñ‚Ñ€Ğ¾ĞµĞ½Ğ¸Ğµ Ğ³Ğ¸Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ñ‹ Ğ´Ğ»Ñ� Ğ½Ğ¾Ğ²Ğ¾Ğ¹ Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ¹ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ¾Ğ¹
plt.figure(figsize=(10, 6))  # Ğ Ğ°Ğ·Ğ¼ĞµÑ€ Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ°
sns.histplot(train.loc[train.efs == 1, "y"], bins=100, kde=True, label="efs=1, Ğ¡Ğ¾Ğ±Ñ‹Ñ‚Ğ¸Ğµ Ğ¿Ñ€Ğ¾Ğ¸Ğ·Ğ¾ÑˆĞ»Ğ¾", color="royalblue", stat="density", linewidth=1.5)
sns.histplot(train.loc[train.efs == 0, "y"], bins=100, kde=True, label="efs=0, Ğ’Ğ¾Ğ·Ğ¼Ğ¾Ğ¶Ğ½Ğ¾ Ñ�Ğ¾Ğ±Ñ‹Ñ‚Ğ¸Ğµ", color="tomato", stat="density", linewidth=1.5)

# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ° Ğ·Ğ°Ğ³Ğ¾Ğ»Ğ¾Ğ²ĞºĞ¾Ğ² Ğ¸ Ğ¿Ğ¾Ğ´Ğ¿Ğ¸Ñ�ĞµĞ¹
plt.xlabel("ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ½Ğ°Ñ� Ñ†ĞµĞ»ĞµĞ²Ğ°Ñ� Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ°Ñ� y", fontsize=14)
plt.ylabel("ĞŸĞ»Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ñ‚ÑŒ", fontsize=14)
plt.title("ğŸ“Š ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ½Ğ°Ñ� Ñ†ĞµĞ»ÑŒ y Ñ� Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ğ¸ĞµĞ¼ Kaplan-Meier Ğ´Ğ»Ñ� Ğ¾Ğ±ĞµĞ¸Ñ… Ñ†ĞµĞ»ĞµĞ¹ (efs Ğ¸ efs_time)", fontsize=16, pad=20)
plt.legend(title="Ğ¡Ğ¾Ğ±Ñ‹Ñ‚Ğ¸Ğµ", fontsize=12)

# Ğ›ĞµĞ³ĞºĞ°Ñ� Ñ�ĞµÑ‚ĞºĞ° Ğ¸ Ñ�Ñ‚Ğ¸Ğ»ÑŒ
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ², ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ ÑƒĞ´Ğ°Ğ»Ñ�ĞµĞ¼ Ğ¸Ğ· Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
RMV = ["ID", "efs", "efs_time", "y"]

# Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ², ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ğ¾Ñ�Ñ‚Ğ°Ñ�Ñ‚Ñ�Ñ� Ğ¿Ğ¾Ñ�Ğ»Ğµ ÑƒĞ´Ğ°Ğ»ĞµĞ½Ğ¸Ñ�
FEATURES = [col for col in train.columns if col not in RMV]

# ĞŸĞµÑ‡Ğ°Ñ‚ÑŒ Ñ‡Ğ¸Ñ�Ğ»Ğ° Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ², ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ğ±ÑƒĞ´ÑƒÑ‚ Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ñ‹ Ğ² Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


# Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½ĞµĞ½Ğ¸Ğµ Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
combined = pd.concat([train, test], axis=0, ignore_index=True)

# ĞšĞ¾Ğ´Ğ¸Ñ€ÑƒĞµĞ¼ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ñ� Ğ¿Ğ¾Ğ¼Ğ¾Ñ‰ÑŒÑ� label encoding
print("ĞœÑ‹ Ğ¿Ñ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµĞ¼ LABEL ENCODING Ğ´Ğ»Ñ� ĞšĞ�Ğ¢Ğ•Ğ“Ğ�Ğ Ğ˜Ğ�Ğ›Ğ¬Ğ�Ğ«Ğ¥ ĞŸĞ Ğ˜Ğ—Ğ�Ğ�ĞšĞ�Ğ’: ", end="")
for c in FEATURES:

    # Ğ•Ñ�Ğ»Ğ¸ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°Ğº ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹, Ğ¿Ñ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµĞ¼ label encoding
    if c in CATS:
        print(f"{c}, ", end="")
        combined[c], _ = combined[c].factorize()  # ĞŸÑ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµĞ¼ Ñ„Ğ°ĞºÑ‚Ğ¾Ñ€Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
        combined[c] -= combined[c].min()  # Ğ¡Ğ¼ĞµÑ‰Ğ°ĞµĞ¼ Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ Ğ½Ğ° 0
        combined[c] = combined[c].astype("int32")  # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ² Ñ‚Ğ¸Ğ¿ int32
        combined[c] = combined[c].astype("category")  # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ² ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Ñ‚Ğ¸Ğ¿
        
    # Ğ”Ğ»Ñ� Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² ÑƒĞ¼ĞµĞ½ÑŒÑˆĞ°ĞµĞ¼ Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ´Ğ¾ 32 Ğ±Ğ¸Ñ‚ Ğ´Ğ»Ñ� Ñ�ĞºĞ¾Ğ½Ğ¾Ğ¼Ğ¸Ğ¸ Ğ¿Ğ°Ğ¼Ñ�Ñ‚Ğ¸
    else:
        if combined[c].dtype == "float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype == "int64":
            combined[c] = combined[c].astype("int32")

# Ğ Ğ°Ğ·Ğ´ĞµĞ»Ñ�ĞµĞ¼ Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ½Ğ¾ Ğ½Ğ° Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ğµ Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


# Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½ĞµĞ½Ğ¸Ğµ Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
combined = pd.concat([train, test], axis=0, ignore_index=True)

# ĞšĞ¾Ğ´Ğ¸Ñ€ÑƒĞµĞ¼ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ñ� Ğ¿Ğ¾Ğ¼Ğ¾Ñ‰ÑŒÑ� label encoding
print("ĞœÑ‹ Ğ¿Ñ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµĞ¼ LABEL ENCODING Ğ´Ğ»Ñ� ĞšĞ�Ğ¢Ğ•Ğ“Ğ�Ğ Ğ˜Ğ�Ğ›Ğ¬Ğ�Ğ«Ğ¥ ĞŸĞ Ğ˜Ğ—Ğ�Ğ�ĞšĞ�Ğ’: ", end="")
for c in FEATURES:

    # Ğ•Ñ�Ğ»Ğ¸ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°Ğº ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹, Ğ¿Ñ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµĞ¼ label encoding
    if c in CATS:
        print(f"{c}, ", end="")
        combined[c], _ = combined[c].factorize()  # ĞŸÑ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµĞ¼ Ñ„Ğ°ĞºÑ‚Ğ¾Ñ€Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
        combined[c] -= combined[c].min()  # Ğ¡Ğ¼ĞµÑ‰Ğ°ĞµĞ¼ Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ Ğ½Ğ° 0
        combined[c] = combined[c].astype("int32")  # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ² Ñ‚Ğ¸Ğ¿ int32
        combined[c] = combined[c].astype("category")  # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ² ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Ñ‚Ğ¸Ğ¿
        
    # Ğ”Ğ»Ñ� Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² ÑƒĞ¼ĞµĞ½ÑŒÑˆĞ°ĞµĞ¼ Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ´Ğ¾ 32 Ğ±Ğ¸Ñ‚ Ğ´Ğ»Ñ� Ñ�ĞºĞ¾Ğ½Ğ¾Ğ¼Ğ¸Ğ¸ Ğ¿Ğ°Ğ¼Ñ�Ñ‚Ğ¸
    else:
        if combined[c].dtype == "float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype == "int64":
            combined[c] = combined[c].astype("int32")

# Ğ Ğ°Ğ·Ğ´ĞµĞ»Ñ�ĞµĞ¼ Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ½Ğ¾ Ğ½Ğ° Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ğµ Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb

# ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ğ²ĞµÑ€Ñ�Ğ¸Ñ� XGBoost
print("Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼Ğ°Ñ� Ğ²ĞµÑ€Ñ�Ğ¸Ñ� XGBoost:", xgb.__version__)


%%time
FOLDS = 10  # ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ„Ğ¾Ğ»Ğ´Ğ¾Ğ² Ğ´Ğ»Ñ� ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# ĞœĞ°Ñ�Ñ�Ğ¸Ğ²Ñ‹ Ğ´Ğ»Ñ� Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ� Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğ¹
oof_xgb = np.zeros(len(train))  # Out-of-Fold Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ´Ğ»Ñ� Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ğ¾Ğ³Ğ¾ Ğ½Ğ°Ğ±Ğ¾Ñ€Ğ°
pred_xgb = np.zeros(len(test))  # ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ´Ğ»Ñ� Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ³Ğ¾ Ğ½Ğ°Ğ±Ğ¾Ñ€Ğ°

# Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ½Ğ° ĞºĞ°Ğ¶Ğ´Ğ¾Ğ¼ Ñ„Ğ¾Ğ»Ğ´Ğµ
for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#" * 25)
    print(f"### Ğ¤Ğ¾Ğ»Ğ´ {i+1}")
    print("#" * 25)
    
    # Ğ Ğ°Ğ·Ğ´ĞµĞ»Ñ�ĞµĞ¼ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ½Ğ° Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½ÑƒÑ� Ğ¸ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½ÑƒÑ� Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸
    x_train = train.loc[train_index, FEATURES].copy()
    y_train = train.loc[train_index, "y"]
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = train.loc[test_index, "y"]
    x_test = test[FEATURES].copy()

    # Ğ˜Ğ½Ğ¸Ñ†Ğ¸Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¸ Ğ¾Ğ±ÑƒÑ‡Ğ°ĞµĞ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ XGBoost
    model_xgb = XGBRegressor(
        device="cuda",  # Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼ GPU Ğ´Ğ»Ñ� ÑƒÑ�ĞºĞ¾Ñ€ĞµĞ½Ğ¸Ñ�
        max_depth=3,  # Ğ“Ğ»ÑƒĞ±Ğ¸Ğ½Ğ° Ğ´ĞµÑ€ĞµĞ²ÑŒĞµĞ²
        colsample_bytree=0.5,  # ĞŸÑ€Ğ¾Ğ¿Ğ¾Ñ€Ñ†Ğ¸Ñ� Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ğ´Ğ»Ñ� Ğ´ĞµÑ€ĞµĞ²Ğ°
        subsample=0.8,  # ĞŸÑ€Ğ¾Ğ¿Ğ¾Ñ€Ñ†Ğ¸Ñ� Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸
        n_estimators=2000,  # ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ´ĞµÑ€ĞµĞ²ÑŒĞµĞ²
        learning_rate=0.02,  # Ğ¡ĞºĞ¾Ñ€Ğ¾Ñ�Ñ‚ÑŒ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�
        enable_categorical=True,  # Ğ’ĞºĞ»Ñ�Ñ‡Ğ°ĞµĞ¼ Ğ¾Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºÑƒ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
        min_child_weight=80,  # ĞœĞ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Ğ²ĞµÑ� Ğ´Ğ¾Ñ‡ĞµÑ€Ğ½Ğ¸Ñ… ÑƒĞ·Ğ»Ğ¾Ğ²
        #early_stopping_rounds=25,  # ĞœĞ¾Ğ¶Ğ½Ğ¾ Ğ²ĞºĞ»Ñ�Ñ‡Ğ¸Ñ‚ÑŒ Ñ€Ğ°Ğ½Ğ½Ñ�Ñ� Ğ¾Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²ĞºÑƒ
    )
    
    # Ğ�Ğ±ÑƒÑ‡Ğ°ĞµĞ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ½Ğ° Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500  # ĞŸĞµÑ‡Ğ°Ñ‚ÑŒ Ğ¿Ñ€Ğ¾Ğ¼ĞµĞ¶ÑƒÑ‚Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¾Ğ²
    )

    # ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ´Ğ»Ñ� Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸ Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ°
    oof_xgb[test_index] = model_xgb.predict(x_valid)  # Out-of-Fold Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ�
    pred_xgb += model_xgb.predict(x_test)  # ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ´Ğ»Ñ� Ñ‚ĞµÑ�Ñ‚Ğ°

# Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğ¹ Ğ´Ğ»Ñ� Ñ‚ĞµÑ�Ñ‚Ğ°
pred_xgb /= FOLDS


from sklearn.metrics import mean_squared_error

# Ğ˜Ñ�Ñ‚Ğ¸Ğ½Ğ½Ñ‹Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� (ID, efs, efs_time, race_group)
y_true = train[["ID", "efs", "efs_time", "race_group"]].copy()

# ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb  # Ğ—Ğ°Ğ¿Ğ¸Ñ�Ñ‹Ğ²Ğ°ĞµĞ¼ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ¸Ğ· Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸

# Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ MSE Ğ´Ğ»Ñ� Ğ¾Ñ†ĞµĞ½ĞºĞ¸
mse = mean_squared_error(y_true["efs_time"], y_pred["prediction"])
print(f"Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞºĞ²Ğ°Ğ´Ñ€Ğ°Ñ‚Ğ¸Ñ‡Ğ½Ğ°Ñ� Ğ¾ÑˆĞ¸Ğ±ĞºĞ°: {mse}")


# Feature Importance with Tufte's Principles
feature_importance = model_xgb.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# Set up figure with increased height for better readability
plt.figure(figsize=(15, 20))  # Height set to 20

# Create horizontal bar chart
bars = plt.barh(
    importance_df["Feature"], 
    importance_df["Importance"], 
    color='royalblue', 
    edgecolor='none'  # No border for a cleaner look
)

# Add numbers at the end of the bars
for bar in bars:
    width = bar.get_width()  # Get the width (importance value)
    plt.text(
        width + 0.005,  # Slightly to the right of the bar
        bar.get_y() + bar.get_height() / 2, 
        f'{width:.4f}',  # Format to 4 decimal places
        va='center', 
        ha='left', 
        color='black', 
        fontsize=10
    )

# Apply Tufte's minimalism:
plt.xlabel("Importance", fontsize=14, labelpad=10)
plt.ylabel("Feature", fontsize=14, labelpad=10)
plt.title("XGBoost KaplanMeier Feature Importance", fontsize=16, pad=15)
plt.gca().invert_yaxis()  # Flip features for better readability
plt.grid(False)  # No grid lines
plt.box(False)  # No chart border

# Display the plot
plt.tight_layout()
plt.show()


# Ğ˜Ğ¼Ğ¿Ğ¾Ñ€Ñ‚ Ğ±Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞº
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb

# Ğ’Ñ‹Ğ²Ğ¾Ğ´ Ğ²ĞµÑ€Ñ�Ğ¸Ğ¸ Ğ±Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞºĞ¸ CatBoost
print("Using CatBoost version", cb.__version__)


import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

# Ğ�Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ° ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Ğ˜Ğ½Ğ¸Ñ†Ğ¸Ğ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ¼Ğ°Ñ�Ñ�Ğ¸Ğ²Ğ¾Ğ² Ğ´Ğ»Ñ� OOF-Ğ¿Ñ€Ğ¾Ğ³Ğ½Ğ¾Ğ·Ğ¾Ğ² Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ñ… Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğ¹
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#" * 25)
    print(f"### Fold {i + 1}")
    print("#" * 25)
    
    # Ğ Ğ°Ğ·Ğ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ğ½Ğ° Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ÑƒÑ� Ğ¸ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½ÑƒÑ� Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸
    x_train = train.loc[train_index, FEATURES].copy()
    y_train = train.loc[train_index, "y"]
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = train.loc[test_index, "y"]
    x_test = test[FEATURES].copy()

    # Ğ˜Ğ½Ğ¸Ñ†Ğ¸Ğ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ¸ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ CatBoost Ğ½Ğ° CPU
    model_cat = CatBoostRegressor(
        task_type="CPU",  # Ğ¯Ğ²Ğ½Ğ¾Ğµ ÑƒĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğµ Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ñ� CPU
        learning_rate=0.1,
        grow_policy='Lossguide',
        # early_stopping_rounds=25,  # ĞœĞ¾Ğ¶Ğ½Ğ¾ Ğ²ĞºĞ»Ñ�Ñ‡Ğ¸Ñ‚ÑŒ Ğ´Ğ»Ñ� ÑƒÑ�ĞºĞ¾Ñ€ĞµĞ½Ğ¸Ñ�
    )
    model_cat.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid),
        cat_features=CATS,
        verbose=250
    )

    # ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ´Ğ»Ñ� OOF Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
    oof_cat[test_index] = model_cat.predict(x_valid)
    pred_cat += model_cat.predict(x_test)

# Ğ£Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ½Ğ¸Ğµ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğ¹ Ğ´Ğ»Ñ� Ñ‚ĞµÑ�Ñ‚Ğ°
pred_cat /= FOLDS

print("ĞšÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ� Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½Ğ°!")


feature_importance = model_cat.get_feature_importance()
importance_df = pd.DataFrame({
    "Feature": FEATURES, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()




