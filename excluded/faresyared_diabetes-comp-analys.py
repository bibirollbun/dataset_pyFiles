import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math


# settings
plt.style.use("ggplot")
sns.set(font_scale=1.0)

df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")

print(df.shape)
df.head()


# Basic Info & Types

print("Info")
df.info()

print("\nBasic Describe (numeric)")
display(df.describe().T)

print("\nBasic Describe (categorical)")
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
display(df[cat_cols].describe().T)


numeric_cols = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides'
]

binary_cols = [
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history',
    'diagnosed_diabetes'
]

categorical_cols = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status'
]

id_col = 'id'

print("Numeric columns:", numeric_cols)
print("\nBinary columns:", binary_cols)
print("\nCategorical columns:", categorical_cols)


# حوله ل-int عشان الشكل
df['diagnosed_diabetes'] = df['diagnosed_diabetes'].astype(int)

target_counts = df['diagnosed_diabetes'].value_counts().sort_index()
target_pct = (target_counts / len(df) * 100).round(2)

print("Target Counts")
display(pd.DataFrame({
    "count": target_counts,
    "percent": target_pct
}))

plt.figure(figsize=(5, 4))
sns.barplot(x=target_counts.index.astype(str), y=target_counts.values)
plt.xlabel("diagnosed_diabetes (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.title("Target Distribution")
plt.tight_layout()
plt.show()


df = df.replace([np.inf, -np.inf], np.nan)

n_cols = 3
n_features = len(numeric_cols)
n_rows = math.ceil(n_features / n_cols)

plt.figure(figsize=(5 * n_cols, 4 * n_rows))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(data=df, x=col, kde=True, bins=40)
    plt.title(col)
    plt.xlabel("")

plt.suptitle("Numeric Features Distributions", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


n_cols = 3
n_features = len(numeric_cols)
n_rows = math.ceil(n_features / n_cols)

plt.figure(figsize=(5 * n_cols, 4 * n_rows))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(y=df[col])
    plt.title(col)
    plt.ylabel("")
    plt.tight_layout()

plt.suptitle("Numeric Features Boxplots", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Countplot of {col}")
    plt.tight_layout()
    plt.show()



for col in binary_cols:
    plt.figure(figsize=(4, 4))
    counts = df[col].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()


corr_cols = list(numeric_cols) + list(binary_cols)
corr = df[corr_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Matrix (Numeric + Binary)")
plt.tight_layout()
plt.show()


target = 'diagnosed_diabetes'
corr_with_target = corr[target].drop(target).sort_values(ascending=False)

corr_with_target_df = corr_with_target.to_frame(name='correlation_with_target')
display(corr_with_target_df)

plt.figure(figsize=(6, 8))
sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
plt.title("Correlation with diagnosed_diabetes")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


n_cols = 3
n_features = len(numeric_cols)
n_rows = math.ceil(n_features / n_cols)

plt.figure(figsize=(5 * n_cols, 4 * n_rows))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(data=df, x='diagnosed_diabetes', y=col)
    plt.title(col)
    plt.xlabel("diagnosed_diabetes")
    plt.tight_layout()

plt.suptitle("Numeric Features vs Target (Boxplots)", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


sample_frac = 0.2
df_sample = df.sample(frac=sample_frac, random_state=42)

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df_sample, x=col, hue='diagnosed_diabetes', common_norm=False, fill=True, alpha=0.4)
    plt.title(f"{col} Distribution by diagnosed_diabetes")
    plt.tight_layout()
    plt.show()


for col in categorical_cols:
    rate = (
        df.groupby(col)['diagnosed_diabetes']
          .mean()
          .sort_values(ascending=False)
    )
    
    plt.figure(figsize=(7, 4))
    sns.barplot(x=rate.index, y=rate.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Diabetes Rate (mean of diagnosed_diabetes)")
    plt.title(f"Diabetes Rate by {col}")
    plt.tight_layout()
    plt.show()


for col in ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']:
    crosstab = pd.crosstab(df[col], df['diagnosed_diabetes'], normalize='index') * 100
    print(f"\n=== {col} vs diagnosed_diabetes (row %) ===")
    display(crosstab.round(2))
    
    crosstab_plot = crosstab.reset_index().melt(id_vars=col, var_name='diagnosed_diabetes', value_name='percent')
    
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=crosstab_plot,
        x=col,
        y='percent',
        hue='diagnosed_diabetes'
    )
    plt.title(f"{col} vs diagnosed_diabetes (Row %)")
    plt.ylabel("Percent %")
    plt.tight_layout()
    plt.show()


df_sample = df.sample(frac=0.15, random_state=42)

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_sample, x='age', y='bmi', hue='diagnosed_diabetes', alpha=0.4)
plt.title("Age vs BMI by diagnosed_diabetes")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_sample, x='systolic_bp', y='diastolic_bp', hue='diagnosed_diabetes', alpha=0.4)
plt.title("Systolic vs Diastolic BP by diagnosed_diabetes")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_sample, x='cholesterol_total', y='triglycerides', hue='diagnosed_diabetes', alpha=0.4)
plt.title("Cholesterol vs Triglycerides by diagnosed_diabetes")
plt.tight_layout()
plt.show()


subset_cols = [
    'age',
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'cholesterol_total',
    'diagnosed_diabetes'
]

df_small = df[subset_cols].sample(n=8000, random_state=42)

sns.pairplot(
    df_small,
    hue='diagnosed_diabetes',
    diag_kind='kde',
    corner=True
)
plt.suptitle("Pairplot of Key Features vs diagnosed_diabetes", y=1.02)
plt.show()




