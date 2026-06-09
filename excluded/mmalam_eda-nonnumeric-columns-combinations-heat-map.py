import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as ws
from itertools import combinations
from scipy.stats import chi2_contingency


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df.head()


len(df)


df.isnull().any()


categorical_cols = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade"
]

for col1, col2 in combinations(categorical_cols, 2):
    pivot = df.pivot_table(
        index=col1, columns=col2, values="loan_paid_back",
        aggfunc=lambda x: (x == 0).mean() * 100
    )
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds")
    plt.title(f"% Defaulters by {col1} and {col2}")
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.show()


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt((chi2 / n) / (min(k - 1, r - 1)))

cramers_results = pd.DataFrame(
    index=categorical_cols + ["loan_paid_back"],
    columns=categorical_cols + ["loan_paid_back"]
)

for col1 in cramers_results.columns:
    for col2 in cramers_results.index:
        cramers_results.loc[col2, col1] = cramers_v(df[col1], df[col2])

plt.figure(figsize=(8,6))
sns.heatmap(cramers_results.astype(float), annot=True, cmap="coolwarm")
plt.title("Cramér’s V Correlation among Categorical Features")
plt.show()

