import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
print(len(train_df))
print(len(test_df))


display(train_df.head())
display(train_df.describe(include='all'))

# Check missing values
print("\nMissing values:")
display(train_df.isnull().sum().sort_values(ascending=False))

# -------------------------------
# DATA TYPE SUMMARY
# -------------------------------
print("\nData Types:")
display(train_df.dtypes)

# Split numerical and categorical columns
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# -------------------------------
# TARGET DISTRIBUTION
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='loan_paid_back')
plt.title("Target Distribution: Loan Paid Back vs Defaulted")
plt.show()

# -------------------------------
# CORRELATION ANALYSIS
# -------------------------------
plt.figure(figsize=(14,10))
corr = train_df[num_cols].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap of Numerical Variables")
plt.show()

# Top correlations with target
print("\nTop correlations with loan_paid_back:")
display(corr['loan_paid_back'].sort_values(ascending=False))

# -------------------------------
# HISTOGRAMS FOR NUMERICAL FEATURES
# -------------------------------
train_df[num_cols].hist(figsize=(16,14), bins=30)
plt.tight_layout()
plt.show()

# -------------------------------
# CATEGORICAL DISTRIBUTIONS
# -------------------------------
for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(data=train_df, x=col)
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")
    plt.show()

# -------------------------------
# BOX PLOTS FOR NUMERICAL VS TARGET
# -------------------------------
for col in num_cols:
    if col != "loan_paid_back":
        plt.figure(figsize=(8,4))
        sns.boxplot(data=train_df, x='loan_paid_back', y=col)
        plt.title(f"{col} vs Loan Paid Back")
        plt.show()

# -------------------------------
# CORRELATION PAIRPLOT (OPTIONAL)
# -------------------------------
# sns.pairplot(train_df[num_cols + ['loan_paid_back']], corner=True)
# plt.show()

print("\nEDA Completed.")




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r-1, k-1))))

cat_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

cramers_results = {}

for col in cat_cols:
    cm = pd.crosstab(train_df[col], train_df['loan_paid_back'])
    cramers_results[col] = cramers_v(cm)

# Sort by strength
cramers_results = dict(sorted(cramers_results.items(), key=lambda x: x[1], reverse=True))

print("Cramer's V correlation with loan_paid_back:\n")
for k, v in cramers_results.items():
    print(f"{k}: {v:.4f}")

# Barplot visualization
plt.figure(figsize=(10,5))
sns.barplot(x=list(cramers_results.keys()), y=list(cramers_results.values()))
plt.xticks(rotation=45)
plt.title("Cramer's V Correlation of Categorical Features with Loan Paid Back")
plt.show()





