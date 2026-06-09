import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

import warnings, os, gc, sys, math, json, random, itertools

from scipy import stats
from scipy.stats import ks_2samp


warnings.filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")
sns.set_palette("crest")
pd.set_option("display.max_columns", 100)


train = pd.read_csv('/kaggle/input/mlolympiadbd2025/train.csv')
test = pd.read_csv('/kaggle/input/mlolympiadbd2025/test.csv')
train_enc = train.replace({"RiskLevel": {0:"Low Risk", 1:"Mid Risk", 2:"High Risk"}})
##encoded switches the ints to categories


def quick_overview(df, name="train"):
    print(f"\n{name.upper()} SHAPE: {df.shape}")
    display(df.head())
    display(df.describe(include="all").T)

quick_overview(train, "train")
quick_overview(test , "test")

print(f"Duplicate rows (train): {train.duplicated().sum()}  |  (test): {test.duplicated().sum()}")



fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(data=train_enc, x="RiskLevel", ax=ax)
ax.set_title("Risk Level Balance")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}", (p.get_x()+.35, p.get_height()+20), ha="center")

plt.ylim(0, 400)
plt.show()

print(train_enc["RiskLevel"].value_counts(normalize=True).rename("proportion"))


train.isnull().sum() 



def plot_kde(data, name, columns=None, figsize=(8, 4), fill=True, max_density=None):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    columns = data.select_dtypes(include='number').columns.tolist()
    plt.figure(figsize=figsize)
    for col in columns:
        sns.kdeplot(data[col], label=col, linewidth=2,clip=(0, None),linestyle="-.")
        
    if max_density is not None:
        plt.ylim(0, max_density)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


for col in test.drop(columns=['Id','Usage']).columns.tolist():
    combo = pd.DataFrame({
        'Train' : train[col],
        'Test' : test[col]
    })
    plot_kde(combo,col)


outlier_summary = {}
for col in train.drop(columns=['Id','Usage']):
    z = np.abs(stats.zscore(train[col]))
    outlier_summary[col] = (z>3).sum()   # 3-σ rule

pd.Series(outlier_summary, name="#outliers (>3σ)").sort_values(ascending=False).to_frame().style.bar()


#Outlier Mask for Blood glucose
bg = train["Blood glucose"]
z    = np.abs(stats.zscore(bg, nan_policy="omit"))
outlier_mask = (z > 3)

outliers= train_enc.loc[outlier_mask, ["Blood glucose", "RiskLevel"]]
base_counts   = train["RiskLevel"].value_counts()
outlier_counts = outliers["RiskLevel"].value_counts()

fig, ax = plt.subplots(figsize=(4,3))
sns.barplot(x=outlier_counts.index, y=outlier_counts.values, ax=ax)
ax.set_title("RiskLevel among 3σ Blood Glucose outliers")
ax.set_ylabel("count")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,.0f}", (p.get_x()+0.3, p.get_height()+30))

plt.show()

# Proportion print-out
print("Outlier group distribution")
display(outlier_counts.to_frame("count")
        .assign(prop=lambda d: d["count"]/d["count"].sum())
        .style.format({"prop": "{:.2%}"}))

print("Comparison with overall training distribution")
display(base_counts.to_frame("count")
        .assign(prop=lambda d: d["count"]/d["count"].sum())
        .style.format({"prop": "{:.2%}"}))


#Outlier Mask for Blood glucose
bt = train["BodyTemp"]
z    = np.abs(stats.zscore(bt, nan_policy="omit"))
outlier_mask = (z > 3)

outliers= train_enc.loc[outlier_mask, ["BodyTemp", "RiskLevel"]]
base_counts   = train["RiskLevel"].value_counts()
outlier_counts = outliers["RiskLevel"].value_counts()

fig, ax = plt.subplots(figsize=(4,3))
sns.barplot(x=outlier_counts.index, y=outlier_counts.values, ax=ax)
ax.set_title("Risk Level among 3σ BodyTemp outliers")
ax.set_ylabel("count")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,.0f}", (p.get_x()+0.3, p.get_height()+30))

plt.show()

# Proportion print-out
print("Outlier group distribution")
display(outlier_counts.to_frame("count")
        .assign(prop=lambda d: d["count"]/d["count"].sum())
        .style.format({"prop": "{:.2%}"}))

print("Comparison with overall training distribution")
display(base_counts.to_frame("count")
        .assign(prop=lambda d: d["count"]/d["count"].sum())
        .style.format({"prop": "{:.2%}"}))


# Numeric vs target
num_cols = train_enc.drop(columns=['Id','Usage']).select_dtypes(include='number').columns.tolist()
fig, axes = plt.subplots(math.ceil(len(num_cols)/3), 3, figsize=(15,4*math.ceil(len(num_cols)/3)))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x="RiskLevel", y=col, data=train_enc, ax=axes[i])
    axes[i].set_title(f"{col} by Risk Level")
plt.tight_layout()
plt.show()


corr = train.drop(columns=['Id','Usage']).corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,cbar_kws={"shrink": 0.8})

plt.xticks(rotation=90, ha='right', fontsize=8) 
plt.yticks(rotation=0, fontsize=8)
plt.title("Pearson Correlation")
plt.show()


target_corr = train.drop(columns=['Id','Usage']).corr()["RiskLevel"].drop(
    "RiskLevel").sort_values()
display(target_corr.to_frame("corr_with_target").style.bar(vmin=-1,vmax=1))

