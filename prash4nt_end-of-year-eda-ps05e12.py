import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", None)


train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


train_df.head()


train_df.shape


train_df.info()


train_df.describe().T


plt.figure(figsize=(8,8))
sns.countplot(x="diagnosed_diabetes", data=train_df)
plt.title("Target Distribution", fontsize=16, weight="bold")
plt.show()


train_df["diagnosed_diabetes"].value_counts(normalize=True)


train_df.isna().mean().sort_values(ascending=False)


plt.figure(figsize=(8,8))
sns.heatmap(train_df.isna(), cbar=False)
plt.title("missing Value Heatmap")
plt.show()

# heatmap, so that we can see the stripes of missing value
# and we can look for any pattern of missingness


TARGET = "diagnosed_diabetes"
ID_COL = "id"

numerical_features = train_df.select_dtypes(include=["int64", "float64"]).columns.drop(
    [TARGET, ID_COL]
)

categorical_features = train_df.select_dtypes(include=["object"]).columns

numerical_features, categorical_features


train_df[numerical_features].hist(
    figsize=(8, 8),
    bins=30,
    edgecolor="black"
)
plt.suptitle("Numerical Feature Distribution", fontsize=16, weight="bold")
plt.show()


fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

for i, col in enumerate(["age", "bmi", "waist_to_hip_ratio", "cholesterol_total", "triglycerides"]):
    sns.boxplot(data=train_df, x=TARGET, y=col, ax=axs[i])
    axs[i].set_title(f"{col.upper()} vs Diabetes Diagnosis")

axs[-1].remove()
plt.suptitle("Numerical Features vs Diabetes Diagnosis", fontsize=16, weight="bold")
plt.tight_layout()
plt.show()


# correlation matrix
corr = train_df[numerical_features.tolist()+["diagnosed_diabetes"]].corr()

plt.figure(figsize=(15, 10))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    annot=True,
    fmt=".2f",
)
plt.title("Numerical Feature Correlation Matrix", fontsize=16, weight="bold")
plt.show()



plt.figure(figsize=(15, 10))

sns.pairplot(
    data=train_df[numerical_features.tolist()+["diagnosed_diabetes"]],
    diag_kind='kde'
)

plt.title("Numerical Feature Relation Matrix", weight="bold")
plt.show()



for col in categorical_features:
    display(
        train_df.groupby(col)[TARGET].mean().sort_values(ascending=False)
    )
    print("-"*40)


fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

for i, col in enumerate(["gender", "ethnicity", "smoking_status", "education_level", "employment_status", "income_level"]):
    sns.barplot(data=train_df, x=col, y=TARGET, ax=axs[i])
    axs[i].set_title(f"{col.upper()} vs Diabetes Rate")

plt.suptitle("Categorical Features vs Diabetes Rate", fontsize=16, weight="bold")
plt.tight_layout()
plt.show()


bmi_pos = train_df.loc[train_df[TARGET] == 1, "bmi"]
bmi_neg = train_df.loc[train_df[TARGET] == 0, "bmi"]

stats.ttest_ind(bmi_pos, bmi_neg, alternative="greater")
# statistical significane answers a yes or no questionâ€”"is there a real difference/effect?"


(bmi_pos.mean() - bmi_neg.mean())
# this checks effect size.
# effect size measures the magnitude of the difference/effect.


sns.kdeplot(
    data=train_df,
    x="physical_activity_minutes_per_week",
    hue=TARGET,
    fill=True
)
plt.title("Physical Activity Distribution by Diabetes Status")
plt.show()


stats.mannwhitneyu(
    train_df.loc[train_df[TARGET]==0, "physical_activity_minutes_per_week"],
    train_df.loc[train_df[TARGET]==1, "physical_activity_minutes_per_week"],
    alternative="greater"
)


contingency = pd.crosstab(
    train_df["family_history_diabetes"],
    train_df[TARGET]
)

contingency


X2 = stats.chi2_contingency(contingency, correction=False)[0]
print(X2)

N = np.sum(contingency)

minimum_dimension=min(contingency.shape)-1

result = np.sqrt((X2/N)/minimum_dimension)

print(result)

