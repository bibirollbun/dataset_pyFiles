import os

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


KAGGLE_DATA_DIR = "/kaggle/input/playground-series-s5e6"
COMP = KAGGLE_DATA_DIR if os.path.exists(KAGGLE_DATA_DIR) else "input"  # local

train = pd.read_csv(f"{COMP}/train.csv")
test = pd.read_csv(f"{COMP}/test.csv")
sample_submission = pd.read_csv(f"{COMP}/sample_submission.csv")

train["Source"] = "train"
test["Source"] = "test"



sample_submission.head()


train.info()
# test (250000) is the same format, but without the target column


train.head()


train["Fertilizer Name"].unique()


train["Soil Type"].unique()


print(train["Fertilizer Name"].value_counts())
sns.histplot(data=train, x="Fertilizer Name");


train.drop("id", axis=1).describe()


quantitative = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]


sns.heatmap(train[quantitative].corr(), annot=True);


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(quantitative):
    pct = pd.crosstab(train[col], train["Fertilizer Name"], normalize="index") * 100
    pct.plot(kind="bar", stacked=True, ax=axes[i])
    axes[i].set_title(f"{col} vs. Fertilizer Name")
    axes[i].tick_params(axis="x", rotation=45)
    
    if col != "Moisture":
        axes[i].legend().set_visible(False)
    else:
        axes[i].legend(bbox_to_anchor=(1, 1), loc="upper left")

plt.tight_layout()
plt.show()




for col in ["Soil Type", "Crop Type"]:
    pct = pd.crosstab(train[col], train["Fertilizer Name"], normalize="index") * 100 
    print(pct)
    pct.plot(kind="bar", stacked=True)
    plt.title(f"{col} vs. Fertilizer distribution")
    plt.ylabel("%")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.show()


ctab = pd.crosstab(train["Soil Type"], train["Crop Type"])
plt.figure(figsize=(6, 4))
sns.heatmap(ctab, annot=False, cmap="Blues")
plt.title("Soil-Crop combination frequency")
plt.tight_layout()

for combination in [
    [train["Soil Type"], train["Crop Type"]],
    [train["Temparature"], train["Potassium"]]
]:
    pct = (
        pd.crosstab(
            index=combination,
            columns=train["Fertilizer Name"],
            normalize="index",
        )
        * 100
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pct, cmap="magma")
    plt.title("Fertilizer distribution by Soil-Crop combination")
    plt.tight_layout()




