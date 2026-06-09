import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df_train.shape


df_train.info()


df_train.isna().sum()


df_train.duplicated().sum()


df_train.head()


df_train["diagnosed_diabetes"].value_counts()


numeric_cols = df_train.select_dtypes(include="number").columns  # length is 20
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    if col in ["id", "diagnosed_diabetes"]:
        continue
    ax = axes[i]
    sns.kdeplot(
        x=df_train[col],
        ax=ax,
        hue=df_train["diagnosed_diabetes"],
        warn_singular=False,
        fill=True,
        common_norm=False,
    )

plt.tight_layout()


sns.histplot(
    data=df_train,
    x="waist_to_hip_ratio",
    hue="diagnosed_diabetes",
    bins=20,
    alpha=0.5,
    element="step",
    common_norm=False,
)


categorical_but_int = [
    "diagnosed_diabetes",
    "cardiovascular_history",
    "hypertension_history",
    "family_history_diabetes",
    "alcohol_consumption_per_week",
]
df_train[categorical_but_int] = df_train[categorical_but_int].astype(str)


numeric_cols = df_train.select_dtypes(include="number").columns  # length is 20

df_numeric = df_train[numeric_cols]
df_numeric = df_numeric.sample(
    frac=0.01, random_state=42
)  # since pair plot is expensive, we can use 0.01 of the entire dataset

sns.pairplot(df_numeric)


mtrx = df_train.corr(numeric_only=True)
plt.figure(figsize=(20, 16))
sns.heatmap(mtrx, annot=True, fmt=".2f")


numeric_cols = df_train.select_dtypes(include="number").columns
fig, axes = plt.subplots(3, 5, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(
        data=df_train,     
        y="diagnosed_diabetes",
        x=col,               
        ax=ax,
        hue="diagnosed_diabetes"
    )
    ax.set_title(col) 

plt.tight_layout()


categorical_cols = df_train.select_dtypes(include="object").columns
df_cat = df_train[categorical_cols]
df_cat.shape


fig, axes = plt.subplots(4, 3, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    ax = axes[i]
    sns.countplot(ax=ax, data=df_train, x=col, hue="diagnosed_diabetes")


sns.scatterplot(
    data=df_train.sample(frac=0.01),
    x="age",
    y="bmi",
    hue="diagnosed_diabetes",
    alpha=0.6,
)


from scipy.stats import ttest_ind

age_diabetic = df_train[df_train["diagnosed_diabetes"] == "1.0"]["age"]
age_non_diabetic = df_train[df_train["diagnosed_diabetes"] == "0.0"]["age"]

t_stat, p_val = ttest_ind(age_diabetic, age_non_diabetic, equal_var=False)

print(f"Mean Age (Diabetic):     {age_diabetic.mean()}")
print(f"Mean Age (Non-Diabetic): {age_non_diabetic.mean()}")
print("-" * 40)
print(f"T-Statistic: {t_stat}")
print(f"P-Value:     {p_val:.4e}")

if p_val < 0.05:
    print("Statistically Significant (Age differs between groups)")
else:
    print("No Significant Difference")


plt.figure(figsize=(8, 6))
sns.boxplot(data=df_train, x="diagnosed_diabetes", y="age", hue="diagnosed_diabetes")
plt.axhline(
    y=df_train["age"].mean(), color="red", linestyle="--", label="Global Average Age"
)

plt.title("Age Distribution: Diabetics vs Non-Diabetics")
plt.legend()


from scipy.stats import ttest_ind

activity_diabetic = df_train[df_train["diagnosed_diabetes"] == "1.0"][
    "physical_activity_minutes_per_week"
]
activity_non_diabetic = df_train[df_train["diagnosed_diabetes"] == "0.0"][
    "physical_activity_minutes_per_week"
]

t_stat, p_val = ttest_ind(activity_diabetic, activity_non_diabetic, equal_var=False)


print(f"Mean adtivity minutes/week (Diabetic):     {activity_diabetic.mean()}")
print(f"Mean adtivity minutes/week (Non-Diabetic): {activity_non_diabetic.mean()}")
print("-" * 40)
print(f"T-Statistic: {t_stat}")
print(f"P-Value:     {p_val:.4e}")

if p_val < 0.05:
    print("Statistically Significant (Activity differs between groups)")
else:
    print("No Significant Difference")


plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_train,
    x="diagnosed_diabetes",
    y="physical_activity_minutes_per_week",
    hue="diagnosed_diabetes",
)
plt.axhline(
    y=df_train["physical_activity_minutes_per_week"].mean(),
    color="red",
    linestyle="--",
    label="Global Average Activity",
)

plt.title("Activity Distribution: Diabetics vs Non-Diabetics")
plt.legend()


bmi_diabetic = df_train[df_train["diagnosed_diabetes"] == "1.0"]["bmi"]
bmi_non_diabetic = df_train[df_train["diagnosed_diabetes"] == "0.0"]["bmi"]

t_stat, p_val = ttest_ind(bmi_diabetic, bmi_non_diabetic, equal_var=False)

print(f"Mean BMI (Diabetic):     {bmi_diabetic.mean()}")
print(f"Mean BMI (Non-Diabetic): {bmi_non_diabetic.mean()}")
print("-" * 40)
print(f"T-Statistic: {t_stat}")
print(f"P-Value:     {p_val:.4e}")

if p_val < 0.05:
    print("Statistically Significant (BMI differs between groups)")
else:
    print("No Significant Difference")


bp_diabetic = df_train[df_train["diagnosed_diabetes"] == "1.0"]["systolic_bp"]
bp_non_diabetic = df_train[df_train["diagnosed_diabetes"] == "0.0"]["systolic_bp"]

t_stat, p_val = ttest_ind(bp_diabetic, bp_non_diabetic, equal_var=False)

print(f"Mean Systolic BP (Diabetic):     {bp_diabetic.mean()}")
print(f"Mean Systolic BP (Non-Diabetic): {bp_non_diabetic.mean()}")
print("-" * 40)
print(f"T-Statistic: {t_stat}")
print(f"P-Value:     {p_val:.4e}")

if p_val < 0.05:
    print("Statistically Significant (Systolic BP differs between groups)")
else:
    print("No Significant Difference")


from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(
    df_train["family_history_diabetes"], df_train["diagnosed_diabetes"]
)

chi2, p_val, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value:              {p_val:.4e}")
print("-" * 40)

if p_val < 0.05:
    print("Statistically Significant (Family History is linked to Diabetes)")
else:
    print("No Significant Association")

plt.figure(figsize=(8, 6))
sns.countplot(data=df_train, x="family_history_diabetes", hue="diagnosed_diabetes")
plt.title("Diabetes Diagnosis by Family History")
plt.xlabel("Family History (0=No, 1=Yes)")
plt.ylabel("Count")
plt.legend(title="Diagnosed Diabetes")

