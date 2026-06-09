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


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
target_col = ['y']


def quick_overview(df, name):
    print(f"\n{name.upper()} SHAPE: {df.shape}")
    display(df.head())
    display(df.describe(include="all").T)

quick_overview(train, "train")
quick_overview(test , "test")

print(f"Duplicate rows (train): {train.duplicated().sum()}  |  (test): {test.duplicated().sum()}")


train.isnull().sum() 



fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(data=train, x="y", ax=ax)
ax.set_title("Target Distribution")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}", (p.get_x()+.35, p.get_height()+5000), ha="center")

plt.ylim(0, 700000)
plt.show()

print(train["y"].value_counts(normalize=True).rename("proportion"))


n_cols = 2
n_rows = math.ceil(len(cat_cols) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    sns.countplot(data=train, x=col, ax=ax)
    ax.set_title(f"{col.capitalize()} Distribution", fontsize = 16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize = 14)
    ax.set_ylim(0, 750000)
    if col == "job":
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize = 12)
        ax.set_ylim(0, 200000)
        

# Turn off any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

for col in cat_cols:
    print(train[col].value_counts(normalize=True).rename("proportion"))


print("87.9% of the full training set is 0")
print("12.1% of the full training set is 1")

for col in cat_cols:
    ct = pd.crosstab(train[col], train["y"], normalize="index")*100
    display(ct.style.format("{:.1f}%").set_caption(f"{col} vs Target"))


n_cols = 3
n_rows = math.ceil(len(num_cols) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()


for i, col in enumerate(num_cols):
    sns.histplot(train[col], ax=axes[i], kde=False)
    axes[i].set_title(f"{col.capitalize()} Distribution")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")


for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.show()


outlier_summary = {}
for col in num_cols:
    z = np.abs(stats.zscore(train[col]))
    outlier_summary[col] = (z>3).sum()   # 3-σ rule

pd.Series(outlier_summary, name="#outliers (>3σ)").sort_values(ascending=False).to_frame().style.bar()


num_cols.remove('day')
for col in num_cols:
    feat = train[col]
    z    = np.abs(stats.zscore(feat, nan_policy="omit"))
    outlier_mask = (z > 3)
    
    outliers= train.loc[outlier_mask, [col, "y"]]
    base_counts   = train["y"].value_counts()
    outlier_counts = outliers["y"].value_counts()
    
    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(x=outlier_counts.index, y=outlier_counts.values, ax=ax)
    ax.set_title("Target Distribution among 3σ " + col + " outliers")
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

num_cols.append('day')


fig, axes = plt.subplots(math.ceil(len(num_cols)/3), 3, figsize=(15,4*math.ceil(len(num_cols)/3)))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x="y", y=col, data=train, ax=axes[i], showfliers=False
)
    axes[i].set_title(f"{col} by Target")


for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


num_cols.append('y')
corr = train[num_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,cbar_kws={"shrink": 0.8})

plt.xticks(rotation=90, ha='right', fontsize=8) 
plt.yticks(rotation=0, fontsize=8)
plt.title("Pearson Correlation")
plt.show()



target_corr = train[num_cols].corr()["y"].drop(
    "y").sort_values()
display(target_corr.to_frame("corr_with_target").style.bar(vmin=-1,vmax=1))

