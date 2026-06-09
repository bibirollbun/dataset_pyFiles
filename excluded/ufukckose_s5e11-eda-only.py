import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Purple Theme ðŸ”®
%matplotlib inline
sns.set_theme(style="whitegrid", palette="Purples")
plt.rcParams["text.color"] = "#4b0082"

# Check Kaggle input folder contents
import os
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


# Initial Exploration
df.head()


df.info()


df.describe()


# Select all numeric columns
num_cols = df.select_dtypes("number").columns
# Drop the ID column and the target variable 'loan_paid_back'
num_cols = num_cols.drop(["id", "loan_paid_back"])
print("Numeric columns selected for histograms:")
print(num_cols)
print("\n")


# Set figure size before creating subplots with .hist()
df[num_cols].hist(bins = 20 ,figsize = (15, 15), color="mediumorchid", edgecolor="purple", linewidth=1.2)
plt.suptitle("Histograms of the Numeric Columns", fontsize=30, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# loan_paid_back Distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="loan_paid_back", color="mediumorchid", edgecolor="purple", linewidth=1.2)
plt.title("Distribution of Loan Paid Back (Target)", fontweight = "bold" , fontsize=30)
plt.xlabel("Loan Paid Back (1 = Yes, 0 = No)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Correlation Matrix Heatmap
plt.figure(figsize=(12, 8))
# Compute correlation matrix - this includes all numeric cols, including the target
corr = df.corr(numeric_only=True)

# Plot heatmap
sns.heatmap(corr, annot=True, cmap="Purples", fmt=".2f", annot_kws={"size": 10})
plt.title("Correlation Matrix", fontsize=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Mean Loan Paid Back by Education & Employment
# ------------------------------
# Create the pivot table
pivot_ee = df.pivot_table(
    values="loan_paid_back",
    index="education_level",
    columns="employment_status",
    aggfunc="mean"
)

plt.figure(figsize=(10, 6)) # Size adjusted for new categories
sns.heatmap(pivot_ee, annot=True, fmt=".2f", cmap="Purples", linewidths=.5)
plt.title("Mean Loan Paid Back by Education Level & Employment Status", fontsize=16)
plt.xlabel("Employment Status")
plt.ylabel("Education Level")
plt.tight_layout()
plt.show()

