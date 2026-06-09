# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install scikit-posthocs

import scikit_posthocs as sp


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

df = pd.concat([train_df, extra_df], ignore_index=True)


print(f"train dataset consists of {df.shape[0]} rows, {df.shape[1]} columns")
print(f"test dataset consists of {test_df.shape[0]} rows, {test_df.shape[1]} columns\n")

print("Train dataset:")
print(df.info())
print("\nTest dataset")
print(test_df.info())


print(f" % of na values in train dataset \n {round(df.isna().sum()/len(df)*100, 2)}")
print("---")
print(f" \n% of na values in test dataset \n {round(test_df.isna().sum()/len(test_df)*100, 2)}")


print(f"Target variable: Price\n{df['Price'].describe()}")

sns.set_theme(style='darkgrid')

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

sns.histplot(df['Price'], kde=True, ax=axes[0])
axes[0].set_title("Price Distribution (Histogram)")

sns.boxplot(x=df['Price'], ax=axes[1])
axes[1].set_title("Price Distribution (Boxplot)")

plt.tight_layout()

plt.show()


plt.figure(figsize=(6, 6))
stats.probplot(df['Price'], dist="norm", plot=plt)
plt.title("Q-Q Plot of Price")
plt.show()


# Hypothesis
print("H0: Data is normally distributed")
print("H1: Data is not normally distributed")

# Statistical test
dagostino_test = stats.normaltest(df['Price'])
print(f"\nD'Agostino and Pearson's Test: Stat={dagostino_test.statistic:.4f}, p-value={dagostino_test.pvalue:.4f}")

alpha = 0.05
if dagostino_test.pvalue < alpha:
    print("Data is NOT normally distributed (Reject H0)")
else:
    print("Data appears normally distributed (Fail to reject H0)")


for i in df.columns:
    if (i != 'Price') & (i != 'Weight Capacity (kg)') & (i != 'Compartments') & (i != 'id'):
        df[i] = df[i].fillna("unknown")
        print(f"########## {i} ##########\n")
        index_order = df[i].value_counts().index
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[i], order=index_order)
        plt.title(f"Count of {i}")
        plt.show()
        
        targets = df[i].unique()
        print(df[i].value_counts())
        print(f"\nPrice by {i}")
        
        for j in targets:
            temp_df = df[df[i] == j]
            print(f"{j}: mean = {round(temp_df['Price'].mean(),2)}")

        target_groups = [group['Price'].values for name, group in df.groupby(i)]
        kruskal_result = stats.kruskal(*target_groups)
        print(f"\nKruskal-Wallis H Test: H-statistic = {kruskal_result.statistic:.4f}, p-value = {kruskal_result.pvalue:.4f}")
        alpha = 0.05  # Significance level
        if kruskal_result.pvalue < alpha:
            print(f"Result: Significant difference between the prices of at least one pair of {i} (reject H0).")
        else:
            print(f"Result: No significant difference between {i} and prices (fail to reject H0).")

        if kruskal_result.pvalue < 0.05:
            dunn_result = sp.posthoc_dunn(df, val_col='Price', group_col=i, p_adjust='bonferroni')
            print("\nDunn's Test Results (Pairwise Comparisons):")
            print(dunn_result)
            print("\nSignificant Comparisons (p-value < 0.05):")
            significant_comparisons = dunn_result[dunn_result < 0.05]
            print(significant_comparisons)
        else:
            print(f"Kruskal-Wallis test did not show significant differences between {i} and price.")

        plt.figure(figsize=(12, 6))

        sns.boxplot(x=i, y='Price', data=df)

        plt.title(f"Price Distribution by {i}", fontsize=16)
        plt.xlabel(i, fontsize=12)
        plt.ylabel('Price', fontsize=12)

        plt.xticks(rotation=45)

        plt.show()




        


num = df.select_dtypes('float')

num.corr()


corr_matrix = num.corr()

plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


sns.pairplot(df[['Compartments', 'Weight Capacity (kg)', 'Price']])
plt.show()

