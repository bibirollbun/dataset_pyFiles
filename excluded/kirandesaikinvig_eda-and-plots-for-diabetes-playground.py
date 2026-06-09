#Warnings
import warnings
warnings.simplefilter('ignore')


# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Nice global plotting style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120


# 2. Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
target_col = "diagnosed_diabetes"


# 3. Basic info + missing values
print("="*80)
print("DATA OVERVIEW")
print("="*80)
print("\nTrain shape :", train.shape)
print("Test shape  :", test.shape)

print("\nFirst 5 rows of train:")
display(train.head())

print("\nTrain summary statistics:")
display(train.describe())

print("\nTest summary statistics:")
display(test.describe())

print("\nMissing values")
print("-"*80)
print(f"Total missing in Train: {train.isnull().sum().sum():,}")
print(f"Total missing in Test : {test.isnull().sum().sum():,}")

print("\nInitial Features:")
print("\n ".join(train.columns))

print("\nTest dtypes:")
print(test.dtypes)


# 4. Identify numerical & categorical columns
numcols = [
    col for col in test.columns
    if (test[col].dtype == "int64") or (test[col].dtype == "float64")
]

# explicitly treat these as categorical
catcols = [col for col in test.columns if col not in numcols] + [
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history'
]

# remove the above from numerical list (in case they are there)
for c in ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']:
    if c in numcols:
        numcols.remove(c)

print("\n" + "="*80)
print("FEATURE TYPE SPLIT")
print("="*80)
print("\nNumerical columns:")
print("\n ".join(numcols))

print("\nCategorical columns:")
print("\n ".join(catcols))


# 5. Categorical feature overview (cardinality, options, counts, % with diabetes)
target_col = 'diagnosed_diabetes'

print("\n" + "="*80)
print("CATEGORICAL FEATURES – SUMMARY")
print("="*80)

for col in catcols:
    print(f"\n{'-'*80}")
    print(f"Categorical feature: {col}")
    print(f"Cardinality: {train[col].nunique()}")
    print(f"Unique values: {train[col].unique()}")

    print("\nValue counts:")
    display(train[col].value_counts().to_frame(name="count"))

    # percentage of targets per category (same computation as your code)
    percentages = (
        train.groupby(col)[target_col].sum()
        / train[col].value_counts()
    ).sort_index()

    percentages = (percentages * 100).round(2)
    print("\nPercentage with diabetes (target = 1) by category (%):")
    display(percentages.to_frame(name="% with diabetes"))


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

sns.set_theme(style="whitegrid")

def plot_target_rate_bars(df, catcols, target_col="diagnosed_diabetes", n_cols=3):
    import math

    n_plots = len(catcols)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, n_rows * 4)
    )
    axes = axes.flatten()

    for ax, col in zip(axes, catcols):
        # % with target = 1 in each category
        rate = (
            df.groupby(col)[target_col]
              .mean()
              .reset_index()
        )

        sns.barplot(data=rate, x=col, y=target_col, ax=ax)
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("% with diabetes")
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        # rotate x labels if many categories
        ax.tick_params(axis='x', rotation=30)

        # annotate bars with percentages
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height*100:.1f}%",
                        (p.get_x() + p.get_width()/2., height),
                        ha='center', va='bottom',
                        fontsize=8, rotation=0)

    # remove extra empty axes (if any)
    for ax in axes[len(catcols):]:
        fig.delaxes(ax)

    fig.suptitle("% with diagnosed_diabetes by category", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

# run it
plot_target_rate_bars(train, catcols, target_col="diagnosed_diabetes", n_cols=3)


from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd

def cramers_v(confusion_matrix):
    """Cramér's V association for categorical-categorical."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

results = []

for col in catcols:
    ct = pd.crosstab(train[col], train[target_col])  # contingency table
    chi2, p, dof, expected = chi2_contingency(ct)
    v = cramers_v(ct)

    results.append({
        "feature": col,
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
        "cramers_v": v,
        "n_categories": ct.shape[0]
    })

chi_results = pd.DataFrame(results).sort_values("p_value")

print("\nChi-square association test vs diagnosed_diabetes")
display(chi_results)


#Preparing KDE Plots
import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_kde_grid(df, numcols, target='diagnosed_diabetes', n_cols=3):
    n_plots = len(numcols)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 3)
    )
    axes = axes.flatten()

    shared_handles, shared_labels = None, None

    for ax, col in zip(axes, numcols):
        sns.kdeplot(
            data=df,
            x=col,
            hue=target,
            fill=True,
            common_norm=False,
            alpha=0.5,
            ax=ax
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Build shared legend from the first axes that has one
        if shared_handles is None:
            leg = ax.get_legend()
            if leg is not None:
                shared_handles = leg.legendHandles
                shared_labels = [t.get_text() for t in leg.texts]
                leg.remove()   # remove legend from this subplot
        else:
            # ensure no per-axes legend on subsequent plots
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    # Remove any unused axes (if grid is larger than needed)
    for ax in axes[len(numcols):]:
        fig.delaxes(ax)

    # Add single shared legend for whole figure
    if shared_handles is not None:
        fig.legend(
            shared_handles,
            shared_labels,
            title=target,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=8,
            title_fontsize=9,
        )

    fig.suptitle("\n\nNumerical features – KDE by target", fontsize=30, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


#Viewing KDE Plots
plot_kde_grid(train, numcols, target='diagnosed_diabetes', n_cols=3)


#Preparing Boxplots
import math

def plot_boxplot_grid(df, numcols, target='diagnosed_diabetes', n_cols=3):
    n_plots = len(numcols)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 3)
    )
    axes = axes.flatten()

    for ax, col in zip(axes, numcols):
        sns.boxplot(
            data=df,
            x=target,
            y=col,
            ax=ax
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")   # we'll rely on tick labels 0/1
        ax.set_ylabel("")

    # remove unused axes
    for ax in axes[len(numcols):]:
        fig.delaxes(ax)

    fig.suptitle("Numerical features – boxplots vs target", fontsize=30, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


#Viewing Boxplots
plot_boxplot_grid(train, numcols, target='diagnosed_diabetes', n_cols=3)


#Preparing Histograms
def plot_hist_grid(df, numcols, target='diagnosed_diabetes', n_cols=3):
    n_plots = len(numcols)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 3)
    )
    axes = axes.flatten()

    shared_handles, shared_labels = None, None

    for ax, col in zip(axes, numcols):
        sns.histplot(
            data=df,
            x=col,
            hue=target,
            kde=True,
            bins=40,
            ax=ax,
            element="step"
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # build shared legend once, from first plot that has a legend
        if shared_handles is None:
            leg = ax.get_legend()
            if leg is not None:
                shared_handles = leg.legendHandles
                shared_labels = [t.get_text() for t in leg.texts]
                leg.remove()   # remove from this axes
        else:
            # ensure no per-axes legend
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    # remove unused axes
    for ax in axes[len(numcols):]:
        fig.delaxes(ax)

    # add one shared legend for whole figure
    if shared_handles is not None:
        fig.legend(
            shared_handles,
            shared_labels,
            title=target,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=8,
            title_fontsize=9,
        )

    fig.suptitle("\n\nNumerical features – histograms by target", fontsize=30, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


#Viewing Histograms
plot_hist_grid(train, numcols, target='diagnosed_diabetes', n_cols=3)


# Saw that Physical Activity was notably skewed, so prepare and view a log1p transform
# 9a. Log1p transform of physical_activity_minutes_per_week
train['log_physical_activity'] = np.log1p(train['physical_activity_minutes_per_week'])

print("\nCreated feature: log_physical_activity = log1p(physical_activity_minutes_per_week)")

plt.figure(figsize=(8, 4))
sns.histplot(
    data=train,
    x='log_physical_activity',
    bins=40,
    kde=True,
    edgecolor=None
)
plt.title("Log1p of Physical Activity – distribution")
plt.xlabel("log1p(physical_activity_minutes_per_week)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# Alcohol Consumption was also notably skewed, so prepare and view a log1p transform
# 9b. Log1p transform of alcohol consumption
train['log_alcohol'] = np.log1p(train['alcohol_consumption_per_week'])

print("\nCreated feature: log_alcohol = log1p(alcohol_consumption_per_week)")

plt.figure(figsize=(8, 4))
sns.histplot(
    data=train,
    x='log_alcohol',
    bins=40,
    kde=True,
    edgecolor=None
)
plt.title("Log1p of alcohol consumption – distribution")
plt.xlabel("log1p(alcohol_consumption_per_week)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# 10. Correlation matrix (numerical features + target)

# correlation matrix (numerical features + target)
corr_matrix = train[numcols + ['diagnosed_diabetes']].corr()

plt.figure(figsize=(25, 25))
sns.heatmap(
    corr_matrix,
    annot=True,       # show numbers
    fmt=".2f",        # 2 decimal places
    cmap="coolwarm",  # color map
    center=0,
    linewidths = 0.5
)
plt.title("\nCorrelation matrix: numerical features vs target\n", fontsize = 50)
plt.tight_layout()
plt.show()


#11. Mutual Information Scores
from sklearn.feature_selection import mutual_info_classif

X = train[numcols]
y = train['diagnosed_diabetes']

mi = mutual_info_classif(X, y, random_state=0)
mi_df = pd.DataFrame({'feature': numcols, 'MI': mi}).sort_values("MI", ascending=False)
mi_df

