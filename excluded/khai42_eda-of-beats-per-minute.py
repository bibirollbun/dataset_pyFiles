import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')
import cudf
import matplotlib.pyplot as plt
import math
import seaborn as sns


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


features = [col for col in train.columns if col != "id"]
num_features = len(features)
cols = 3
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))

for i, feature in enumerate(features):
    ax = axes[i // cols, i % cols]
    ax.hist(train[feature], bins=50, density=True, alpha=0.6, color='g')
    train[feature].plot(kind="kde", color="red", ax=ax)
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")

for j in range(i+1, rows*cols):
    fig.delaxes(axes[j // cols, j % cols])

plt.tight_layout()
plt.show()


num_features = len(features)
cols = 3
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))

for i, feature in enumerate(features):
    ax = axes[i // cols, i % cols]
    sns.boxplot(x=train[feature], ax=ax, color="skyblue")
    ax.set_xlabel(feature)

for j in range(i+1, rows*cols):
    fig.delaxes(axes[j // cols, j % cols])

plt.tight_layout()
plt.show()


num_features = len(features)
cols = 3
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))

for i, feature in enumerate(features):
    ax = axes[i // cols, i % cols]
        
    train[feature].plot(kind="hist",
                                bins=30,
                                rwidth=0.9,
                                color="gray",
                                alpha=0.7,
                                ax=ax)
    ax.set_title(f"Binned Frequency of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")

for j in range(i+1, rows*cols):
    fig.delaxes(axes[j // cols, j % cols])

plt.tight_layout()
plt.show()


num_features = len(features) - 1
cols = 3
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))

for i, feature in enumerate([f for f in features if f != "BeatsPerMinute"]):
    ax = axes[i // cols, i % cols]
    sns.regplot(x=train[feature],
                y=train["BeatsPerMinute"],
                scatter_kws={'alpha':0.3, 's':10},
                line_kws={'color':'red'},
                ax=ax)
    ax.set_title(f"{feature} vs BeatsPerMinute")

for j in range(i+1, rows*cols):
    fig.delaxes(axes[j // cols, j % cols])

plt.tight_layout()
plt.show()


features_to_compare = [col for col in train.columns if col not in ["id", "BeatsPerMinute"]]
num_features = len(features_to_compare)
cols = 3
rows = math.ceil(num_features / cols)
fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))

for i, feature in enumerate(features_to_compare):
    ax = axes[i // cols, i % cols]
    sns.kdeplot(train[feature], label="Train", ax=ax, color="blue")
    sns.kdeplot(test[feature], label="Test", ax=ax, color="orange")
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.legend()

for j in range(i+1, rows*cols):
    fig.delaxes(axes[j // cols, j % cols])
plt.tight_layout()
plt.show()


from scipy.stats import ks_2samp
features_to_compare = [col for col in train.columns if col not in ["id", "BeatsPerMinute"]]
num_features = len(features_to_compare)
cols = 3
rows = math.ceil(num_features / cols)
fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
ks_results = {}

for i, feature in enumerate(features_to_compare):
    ax = axes[i // cols, i % cols]
    
    
    sns.kdeplot(train[feature], label="Train", ax=ax, color="blue")
    sns.kdeplot(test[feature], label="Test", ax=ax, color="orange")
    
    
    stat, pval = ks_2samp(train[feature], test[feature])
    ks_results[feature] = {"KS_Statistic": stat, "p-value": pval}
    
    ax.set_title(f"{feature} (KS p={pval:.3f})")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.legend()

for j in range(i+1, rows*cols):
    fig.delaxes(axes[j // cols, j % cols])
plt.tight_layout()
plt.show()


ks_df = pd.DataFrame(ks_results).T
ks_sorted = ks_df.sort_values("p-value")
ks_sorted


train["BPM_Group"] = pd.qcut(train["BeatsPerMinute"], q=4, labels=["Low", "Medium", "High", "Very High"])
features_to_plot = ["RhythmScore", "AudioLoudness", "Energy", "MoodScore"]
cols = 2
rows = math.ceil(len(features_to_plot) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(14, 4*rows))

for i, feature in enumerate(features_to_plot):
    ax = axes[i // cols, i % cols]
    sns.boxplot(data=train, x="BPM_Group", y=feature, ax=ax, palette="Set2")
    sns.stripplot(data=train.sample(2000, random_state=42),
                  x="BPM_Group", y=feature, ax=ax, color="black", alpha=0.3)
    ax.set_title(f"{feature} across BPM groups")
plt.tight_layout()
plt.show()


bpm_group_means = train.groupby("BPM_Group").mean(numeric_only=True)
bpm_group_means[["RhythmScore", "AudioLoudness", "Energy", "MoodScore",
                 "AcousticQuality", "InstrumentalScore"]].T.plot(
    kind="bar", figsize=(12,6)
)
plt.title("Mean Feature Values Across BPM Groups")
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()


sns.boxplot(data=train, x="BPM_Group", y="TrackDurationMs", palette="coolwarm")
plt.title("Track Duration Across BPM Groups")
plt.ylabel("Duration (ms)")
plt.show()


train["Tempo_Category"] = pd.cut(train["BeatsPerMinute"],
                                 bins=[0,100,140,300],
                                 labels=["Slow", "Medium", "Fast"])

sns.lmplot(data=train.sample(10000, random_state=42),
           x="Energy", y="AudioLoudness", hue="Tempo_Category",
           scatter_kws={'alpha':0.3, 's':10}, height=6, aspect=1.5)
plt.title("Energy vs Loudness conditioned on Tempo Category")
plt.show()


pivot = pd.pivot_table(train, values="BeatsPerMinute",
                       index=pd.qcut(train["Energy"], 5),
                       columns=pd.qcut(train["MoodScore"], 5),
                       aggfunc="mean")

plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Mean BPM by Energy and MoodScore bins")
plt.show()




