import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import math

from scipy import stats
from scipy.stats import ks_2samp

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
original_data_1 = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")
original_data_2 = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")

original_data = pd.concat([original_data_1, original_data_2], axis=0, ignore_index=True)

train_data.set_index('id', inplace=True)
test_data.set_index('id', inplace=True)

train_data.head()


train_data.info()


train_data.describe()


duplicates_train = train_data.duplicated().sum()
duplicates_test = test_data.duplicated().sum()

print(f"Duplicates in train: {duplicates_train}")
print(f"Duplicates in test: {duplicates_test}")


def missing_table(df):
    mis = df.isna().sum().to_frame("#missing")
    mis["pct"] = 100*mis["#missing"]/len(df)
    return mis[mis["#missing"]>0].sort_values("pct", ascending=False)

display(missing_table(train_data).style.format({"pct":"{:.1f}%"}))
display(missing_table(test_data).style.format({"pct":"{:.1f}%"}))
display(missing_table(original_data).style.format({"pct":"{:.1f}%"}))


fig, axes = plt.subplots(1,2, figsize=(14, 4))
msno.matrix(train_data, ax=axes[0]); axes[0].set_title("Train â€“ missing pattern")
msno.matrix(test_data, ax=axes[1]); axes[1].set_title("Test â€“ missing pattern")
# plt.tight_layout()
plt.show()


for column in sorted(train_data.columns):
    num_distinct_values = train_data[column].nunique()
    print(f"{column}: {num_distinct_values} unique values")


numerical_variables = train_data.select_dtypes(include=[np.number])
correlation_matrix = numerical_variables.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, vmax=0.8, square=True, annot=True, cmap='YlGn')
plt.title('Correlation Matrix', fontsize=15)
plt.show()


# Encode target for point-biserial correlation
num_cols = train_data.select_dtypes(np.number).columns.tolist()
train_data_enc = train_data.replace({"Personality": {"Extrovert": 1, "Introvert": 0}})

target_corr = train_data_enc[num_cols + ["Personality"]].corr()["Personality"].drop("Personality").sort_values()

display(target_corr.to_frame(name="Correlation with Target").style.bar(vmin=-1, vmax=1))


# Count the occurrences
personality_counts = train_data["Personality"].value_counts()

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(
    x=personality_counts.index, 
    y=personality_counts.values,
    palette=["#3498db", "#e74c3c"]  # Blue for Introvert, Red for Extrovert
)
plt.title("Count of Introverts vs Extroverts", fontsize=14)
plt.xlabel("Personality Type")
plt.ylabel("Count")
plt.show()


num_cols = train_data.select_dtypes(include="number").columns
n_cols = 3
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(train_data[col], kde=True, bins=30, ax=axes[i], color='skyblue', label='Train')
    sns.histplot(test_data[col], kde=True, bins=30, ax=axes[i], color='salmon', label='Test')
    sns.histplot(original_data[col], kde=True, bins=30, ax=axes[i], color='green', label='Original')

    axes[i].set_title(f"Histogram of {col}")
    axes[i].legend()
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Select numeric columns
num_cols = train_data.select_dtypes(include="number").columns

# Create subplot grid
n_cols = 3
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.kdeplot(train_data[col].dropna(), ax=axes[i],linewidth=1.5, label = "train", color='skyblue')
    sns.kdeplot(test_data[col].dropna(), ax=axes[i],linewidth=1.5, label = "test",  color='salmon')
    sns.kdeplot(original_data[col].dropna(), ax=axes[i],linewidth=1.5, label = "original", color='green')

    axes[i].set_title(f"Density of {col}", fontsize=12)
    axes[i].legend()
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")

# Hide any extra unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


cat_cols = ["Stage_fear", "Drained_after_socializing"]

for cat_col in cat_cols:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    train_counts = train_data[cat_col].value_counts(normalize=True)
    test_counts = test_data[cat_col].value_counts(normalize=True)
    
    sns.barplot(x=train_counts.index, y=train_counts.values, ax=ax[0], palette="Blues")
    ax[0].set_title(f"Train - {cat_col} (Proportion)")
    ax[0].set_ylabel("Proportion")
     
    sns.barplot(x=test_counts.index, y=test_counts.values, ax=ax[1], palette="Oranges")
    ax[1].set_title(f"Test - {cat_col} (Proportion)")
    ax[1].set_ylabel("")
    
    plt.tight_layout()
    plt.show()


outlier_summary = {}
for col in num_cols:
    z = np.abs(stats.zscore(train_data[col].dropna()))
    outlier_summary[col] = (z > 3).sum()

outliers_df = pd.Series(outlier_summary, name="Outliers (>3Ïƒ)").sort_values(ascending=False).to_frame()
display(outliers_df.style.bar())


outlier_summary = {}
for col in num_cols:
    z = np.abs(stats.zscore(train_data[col].dropna()))
    outlier_summary[col] = (z > 2).sum()

outliers_df = pd.Series(outlier_summary, name="Outliers (>2Ïƒ)").sort_values(ascending=False).to_frame()
display(outliers_df.style.bar())


# Compute 2-Ïƒ mask (ignore NaNs)
tsa = train_data["Time_spent_Alone"]
z    = np.abs(stats.zscore(tsa, nan_policy="omit"))
outlier_mask = (z > 2)

# Subset + counts
outliers      = train_data.loc[outlier_mask, ["Time_spent_Alone", "Personality"]]
base_counts   = train_data["Personality"].value_counts()
outlier_counts = outliers["Personality"].value_counts()

# Plot
fig, ax = plt.subplots(figsize=(4,3))
sns.barplot(x=outlier_counts.index, y=outlier_counts.values, ax=ax)
ax.set_title("Personality among 2Ïƒ Time_spent_Alone outliers")
ax.set_ylabel("count")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,.0f}", (p.get_x()+0.3, p.get_height()+30))

plt.show()

# Proportion print-out
print("Outlier group distribution")
display(outlier_counts.to_frame("count")
        .assign(prop=lambda d: d["count"]/d["count"].sum())
        .style.format({"prop": "{:.2%}"}))

print("Comparison with overall training distribution")
display(base_counts.to_frame("count")
        .assign(prop=lambda d: d["count"]/d["count"].sum())
        .style.format({"prop": "{:.2%}"}))


# Create subplot grid
n_cols = 3
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x="Personality", y=col, data=train_data, ax=axes[i])
    axes[i].set_title(f"{col} by Personality")

# Hide any extra unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
    
plt.tight_layout()
plt.show()


for feature in ["Stage_fear", "Drained_after_socializing", "Personality"]:
    counts = train_data[feature].value_counts()

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f"Distribution of {feature}")
    plt.axis("equal")
    plt.show()

    # Print unique and missing values
    print(f"Number of Unique {feature}: {train_data[feature].nunique()}")
    print(f"Missing Values in {feature}: {train_data[feature].isnull().sum()}")


drift_rows = []
for col in num_cols.union(cat_cols):
    pval = ks_2samp(train_data[col].dropna(), test_data[col].dropna()).pvalue
    drift_rows.append({"feature": col, "KS-pvalue": pval})
    
drift_df = pd.DataFrame(drift_rows).sort_values("KS-pvalue")
drift_df.style.background_gradient(axis=0, cmap="RdYlGn", subset=["KS-pvalue"])

