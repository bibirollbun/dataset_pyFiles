import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Notebook Settings
import warnings
warnings.filterwarnings("ignore") # Ignore warnings from cell outputs


train_data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv").drop(['id'], axis=1)
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


display(train_data.head())


print("\n~~~~~~~~~INFO~~~~~~~~~~~~\n")
display(train_data.info())
print("\n~~~~~~~~~SUMMARY STATISTICS~~~~~~~~~~~~\n")
display(train_data.describe())


# Helper Function To Analyse Distributions Of Numerical Vars
def num_distbox(df):
    num_list = df.select_dtypes(include=np.number).columns.tolist()
    # Plot layout
    fig, axes = plt.subplots(int(np.ceil(len(num_list))), 2, figsize=(10, 25))
    # axes = axes.flatten()
    for i, col in enumerate(num_list):
        sns.histplot(data=df, x=col, ax=axes[i, 0], hue="Sex", kde=True, palette="Set2")
        axes[i, 0].set_title(f"Distribution Of {col}")
        sns.boxplot(data=df, x=col, ax=axes[i, 1], hue="Sex", palette="Set2")
        axes[i, 1].set_title(f"Boxplot Of {col}")

    plt.tight_layout()
    plt.show()
        

num_distbox(train_data)


# ANALYSING RELATIONSHIPS

plt.figure(figsize=(8, 4))
sns.heatmap(
    train_data.select_dtypes(include=np.number).corr(),
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.show()

