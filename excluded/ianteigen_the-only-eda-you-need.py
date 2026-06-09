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


def bpm_to_bin(bpm):
    if 0 <= bpm < 101:
        return "0-101"
    elif 101 <= bpm < 120:
        return "101-120"
    elif 120 <= bpm < 136:
        return "120-136"
    elif 136 <= bpm <= 210:
        return "136-210"
    else:
        return "Out of range"


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

cat_cols = ['']
num_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 
            'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'Energy']
target_col = ['BeatsPerMinute']
train['BPM_bins']= train['BeatsPerMinute'].apply(bpm_to_bin)


def quick_overview(df, name):
    print(f"\n{name.upper()} SHAPE: {df.shape}")
    df.info
    display(df.head())
    display(df.describe(include="all").T)
    print(f"\n{name.upper()} MISSING VALUES: \n{df.isnull().sum() }")
    

quick_overview(train, "train")
quick_overview(test , "test")

print(f"Duplicate rows (train): {train.duplicated().sum()}  |  (test): {test.duplicated().sum()}")


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

print("KDE PLOT")
plot_kde(train[target_col], "BeatsPerMinute Distribution")

print("HISTOGRAM")
sns.histplot(train[target_col], kde=False)
plt.title(f"BeatsPerMinute Distribution")
plt.xlabel("BeatsPerMinute")
plt.ylabel("Count")

plt.show()


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


fig, axes = plt.subplots(math.ceil(len(num_cols)/3), 3, figsize=(15,4*math.ceil(len(num_cols)/3)))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(x=col, data=train, ax=axes[i], showfliers=False
)
    axes[i].set_title(f"{col} Boxplot")


for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(math.ceil(len(num_cols)/3), 3, figsize=(15,4*math.ceil(len(num_cols)/3)))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(x="BPM_bins", y=col, data=train, ax=axes[i], showfliers=False
)
    axes[i].set_title(f"{col} Boxplot")


for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


for col in num_cols:
    plot_kde(train[col], f"{col} Distribution")


outlier_summary = {}
for col in num_cols:
    z = np.abs(stats.zscore(train[col]))
    outlier_summary[col] = (z>3).sum()   # 3-σ rule

pd.Series(outlier_summary, name="#outliers (>3σ)").sort_values(ascending=False).to_frame().style.bar()


outlier_cols = ['AcousticQuality', 'VocalContent', 'InstrumentalScore', 'AudioLoudness', 'LivePerformanceLikelihood']

for col in outlier_cols:
    feat = train[col]
    z    = np.abs(stats.zscore(feat, nan_policy="omit"))
    outlier_mask = (z > 3)
    outliers= train.loc[outlier_mask, [col, "BeatsPerMinute"]]
    
    bpm_outliers = train.loc[outlier_mask, "BeatsPerMinute"]
    bpm_non_outliers = train.loc[~outlier_mask, "BeatsPerMinute"]

    print("BPM of Outliers vs Non-Outliers for " + col)
    print(f"Outliers: n={len(bpm_outliers)}, mean={bpm_outliers.mean():.2f}, median={bpm_outliers.median():.2f}, std={bpm_outliers.std():.2f}")
    print(f"Non-outliers: n={len(bpm_non_outliers)}, mean={bpm_non_outliers.mean():.2f}, median={bpm_non_outliers.median():.2f}, std={bpm_non_outliers.std():.2f}")
    print()




num_cols.append('BeatsPerMinute')
corr = train[num_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,cbar_kws={"shrink": 0.8})

plt.xticks(rotation=90, ha='right', fontsize=8) 
plt.yticks(rotation=0, fontsize=8)
plt.title("Pearson Correlation")
plt.show()



target_corr = train[num_cols].corr()["BeatsPerMinute"].drop(
    "BeatsPerMinute").sort_values()
display(target_corr.to_frame("corr_with_target").style.bar(vmin=-1,vmax=1))

