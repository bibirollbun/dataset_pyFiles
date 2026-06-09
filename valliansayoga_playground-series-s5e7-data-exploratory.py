# # Comment for uploading to Kaggle
# from src.project_setup import project_setup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use("ggplot")
# paths = project_setup()


# # Local environment
# train = pd.read_csv(paths["data"][-1]).drop("id", axis=1)

# Kaggle environment
import warnings


warnings.filterwarnings("ignore")
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv").drop("id", axis=1)


train.head()


train.info()


categories = train.select_dtypes(object).fillna("Missing")
categories.head()


def plot_categorical_counts(colname: str):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    categories[colname].fillna("Missing").value_counts().sort_values().plot.barh(ax=ax[0])
    ax[1].pie(categories[colname].value_counts(), autopct="%.2f%%")
    
    ax[0].set_xlabel("Original Values", fontsize=10)
    ax[1].set_xlabel("Proportion", fontsize=10)
    
    for a in ax:
        a.set_ylabel(None)
    plt.suptitle(colname, fontsize=14)
    ax[1].legend(categories[colname].value_counts().index, loc="best")
    plt.show()

for cat in categories.columns:
    plot_categorical_counts(cat)



def plot_categorical_heatmap(target: str = "Personality", columns: list = None):
    fig, ax = plt.subplots(1, 2, figsize=(21, 3), layout="constrained")
    sns.heatmap(pd.crosstab(target, columns), annot=True, fmt="g", cbar=False, ax=ax[0], cmap="Greens", robust=True)
    sns.heatmap(pd.crosstab(target, columns, normalize="index"), fmt=".1%", annot=True, cbar=False, ax=ax[1], cmap="Greens", robust=True)
    
    ax[0].set_xlabel("Original Values", fontsize=10)
    ax[1].set_xlabel("Proportion", fontsize=10)
    
    vs_name = "-".join([col.name for col in columns])
    
    plt.suptitle(f"{target.name} vs {vs_name}" , fontsize=14)
    plt.show()
    
plot_categorical_heatmap(categories.Personality, [categories.Drained_after_socializing])
plot_categorical_heatmap(categories.Personality, [categories.Stage_fear])
plot_categorical_heatmap(categories.Personality, [categories.Drained_after_socializing, categories.Stage_fear])


numbers = train.select_dtypes("number")
numbers["Personality"] = train.Personality
numbers.head()


sns.pairplot(numbers, hue="Personality", corner=True)


sns.pairplot(numbers, hue="Personality", kind="reg", corner=True)

