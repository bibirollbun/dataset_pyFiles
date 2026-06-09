import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col="id")

print("------- Train Data -------")
print(train.info())
print(train.shape)
print(train.head())

print("\n------- Test Data -------")
print(test.info())
print(test.shape)
print(test.head())



df_numeric = train.drop(columns=["Stage_fear", "Drained_after_socializing", "Personality"] , axis=1)
corr = df_numeric.corr()

sns.heatmap(corr, cmap="coolwarm", annot=True)
print(test.head())


plt.figure(figsize=(15, 10))

# Loop through and plot each histogram
for i, col in enumerate(df_numeric, 1):
    plt.subplot(3, 3, i)
    ax = sns.countplot(x=col, data=train)
    
    plt.xlabel(col)
    plt.ylabel(' ')
    plt.xticks(rotation=45)
    
plt.tight_layout()
plt.show()


df_categorical = train[["Stage_fear", "Drained_after_socializing", "Personality"]]

for i, col in enumerate(df_categorical.columns):
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df_categorical)
    plt.title(f"Count of {col}")
    plt.show()


fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6, 24))

for i, col in enumerate(df_numeric.columns):
    sns.violinplot(x="Personality", y=col, data=train, ax=axes[i])
    axes[i].set_title(f"{col} vs Personality")


sns.pairplot(train, hue="Personality", diag_kind="kde", markers=["o", "s"], height=2.5)

