import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


data_train = pd.read_csv("/kaggle/input/mental-health-beetleware/train.csv")
data_test = pd.read_csv("/kaggle/input/mental-health-beetleware/test.csv")

data_train = data_train.drop(columns=["id", "Name"])
data_test = data_test.drop(columns=["id", "Name"])

plt.rcParams["figure.dpi"] = 100


def parse_sleep_to_hours(s):
    if pd.isna(s): 
        return np.nan
    s = str(s).lower().strip()
    rng = re.findall(r'(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)', s)
    if rng:
        a, b = map(float, rng[0])
        return (a+b)/2
    h = re.findall(r'(\d{1,2}):(\d{1,2})', s)
    if h:
        hh, mm = map(int, h[0]); 
        return hh + mm/60
    num = re.findall(r'\d+(?:\.\d+)?', s)
    if num:
        return float(num[0])
    return np.nan


to_convert_cols = ["Academic Pressure", "Work Pressure", "Study Satisfaction", "Job Satisfaction", "Financial Stress"]

for c in to_convert_cols:
    data_train[c] = pd.Categorical(data_train[c])
    data_test[c] = pd.Categorical(data_test[c])


print("Shape(Train):", data_train.shape)
print("\nData Type(Train):\n", data_train.dtypes)


print("Shape(Test):", data_test.shape)
print("\nData Type(Test):\n", data_test.dtypes)


missing_train = data_train.isna().sum().sort_values(ascending=False)
print("\nMissing Value:\n", missing_train.head(10))


missing_test = data_test.isna().sum().sort_values(ascending=False)
print("\nMissing Value:\n", missing_test.head(10))


print("\nTarget distribution (rate):")
print(data_train["Depression"].value_counts(normalize=True).sort_index())


qc_train = pd.DataFrame({
    "column": data_train.columns,
    "dtype": data_train.dtypes.astype(str).values,
    "non_null": data_train.notna().sum().values,
    "nulls": data_train.isna().sum().values,
    "null_%": (data_train.isna().mean()*100).round(2).values
}).sort_values("null_%", ascending=False).reset_index(drop=True)

qc_train


qc_test = pd.DataFrame({
    "column": data_test.columns,
    "dtype": data_test.dtypes.astype(str).values,
    "non_null": data_test.notna().sum().values,
    "nulls": data_test.isna().sum().values,
    "null_%": (data_test.isna().mean()*100).round(2).values
}).sort_values("null_%", ascending=False).reset_index(drop=True)

qc_test


plt.figure(figsize=(5,3))
sns.countplot(x="Depression", data=data_train)
plt.title("Depression Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,4))
missing_train.head(10).plot(kind="barh")
plt.title("Missing Values (Train)")
plt.ylabel("Columns")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,4))
missing_test.head(10).plot(kind="barh")
plt.title("Missing Values (Test)")
plt.ylabel("Columns")
plt.tight_layout()
plt.show()


plt.figure(figsize=(14,10))  
sns.heatmap(data_train.isna(), cbar=False)

plt.title("Missing Data Heatmap (Train)", fontsize=14)
plt.xlabel("Columns", fontsize=12)
plt.ylabel("Rows", fontsize=12)

# eksen ayarlar覺
plt.xticks(rotation=90, fontsize=10)  
plt.yticks([], [])  
plt.tight_layout()
plt.show()


plt.figure(figsize=(14,10))  
sns.heatmap(data_test.isna(), cbar=False)

plt.title("Missing Data Heatmap (Test)", fontsize=14)
plt.xlabel("Columns", fontsize=12)
plt.ylabel("Rows", fontsize=12)

plt.xticks(rotation=90, fontsize=10)  
plt.yticks([], [])  
plt.tight_layout()
plt.show()


data_train["Age"] = data_train["Age"].round().clip(lower=10, upper=100)
data_train["Sleep_Hours"] = data_train["Sleep Duration"].apply(parse_sleep_to_hours)

data_test["Age"] = data_test["Age"].round().clip(lower=10, upper=100)
data_test["Sleep_Hours"] = data_test["Sleep Duration"].apply(parse_sleep_to_hours)

cat_cols = data_train.select_dtypes(include=["object"]).columns.tolist()
num_cols = data_test.select_dtypes(include=["number"]).columns.tolist()

print("Categorical:", cat_cols)
print("Numeric:", num_cols)


for col in num_cols:
    s = data_train[col].dropna().astype(float)
    if s.empty:
        continue
    plt.figure(figsize=(5,3))
    plt.hist(s, bins=30)
    plt.title(f"{col} Histogram")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


for col in num_cols:
    s = data_test[col].dropna().astype(float)
    if s.empty:
        continue
    plt.figure(figsize=(5,3))
    plt.hist(s, bins=30)
    plt.title(f"{col} Histogram")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


for col in cat_cols:
    vc = data_train[col].value_counts(dropna=False).head(20)
    plt.figure(figsize=(7,4))
    vc.sort_values().plot(kind="barh")
    plt.title(f"{col}")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.show()


for col in cat_cols:
    vc = data_test[col].value_counts(dropna=False).head(20)
    plt.figure(figsize=(7,4))
    vc.sort_values().plot(kind="barh")
    plt.title(f"{col}")
    plt.xlabel("Say覺")
    plt.tight_layout()
    plt.show()


target = "Depression"
for col in cat_cols:
    vc_idx = data_train[col].value_counts().head(16).index
    sub = data_train[data_train[col].isin(vc_idx)]
    if sub.empty: 
        continue
    ctab = pd.crosstab(sub[col], sub[target], normalize="index").reindex(vc_idx, fill_value=0.0)
    plt.figure(figsize=(8,4))
    ctab.plot(kind="bar", stacked=True)
    plt.title(f"{col} vs {target} (rate)")
    plt.ylabel("Rate")
    plt.legend(title=target, loc="upper right")
    plt.tight_layout()
    plt.show()


for col in num_cols:
    if col == target:
        continue
    plt.figure(figsize=(6,3.5))
    sns.boxplot(x=target, y=col, data=data_train, showfliers=False)
    plt.title(f"{col} vs {target}")
    plt.tight_layout()
    plt.show()


for col in num_cols:
    if col == target:
        continue
    s0 = data_train.loc[data_train[target]==0, col].dropna().astype(float)
    s1 = data_train.loc[data_train[target]==1, col].dropna().astype(float)
    if s0.empty or s1.empty:
        continue
    bins = np.histogram(np.concatenate([s0, s1]), bins=30)[1]
    h0, _ = np.histogram(s0, bins=bins)
    h1, _ = np.histogram(s1, bins=bins)

    plt.figure(figsize=(6,3.5))
    plt.step(bins[:-1], h0, where="post", label="Dep=0")
    plt.step(bins[:-1], h1, where="post", label="Dep=1")
    plt.title(f"{col} Histogram by {target}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

