import pandas as pd

df_transactions = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
df_transactions.head()


print("DataFrame Info:")
df_transactions.info()

print("\nDescriptive Statistics for Numerical Columns:")
df_transactions.describe()

print("\nDescriptive Statistics for Categorical Columns:")
df_transactions.describe(include='object')


missing_values = df_transactions.isnull().sum()
missing_percentage = (df_transactions.isnull().sum() / len(df_transactions)) * 100

missing_info = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
})

# Filter for columns with at least one missing value and sort by percentage
missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False)

print("Columns with Missing Values and their Percentages (Sorted):")
print(missing_info)


import matplotlib.pyplot as plt
import seaborn as sns

# Create a bar plot for the top N missing columns
n_cols_to_plot = 30 # Adjust as needed for readability
plt.figure(figsize=(12, 8))
sns.barplot(x=missing_info['Missing Percentage'].head(n_cols_to_plot).index,
            y=missing_info['Missing Percentage'].head(n_cols_to_plot))
plt.title(f'Top {n_cols_to_plot} Columns with Missing Values Percentage')
plt.xlabel('Column Name')
plt.ylabel('Missing Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Create a heatmap to visualize the missing data pattern across the DataFrame
plt.figure(figsize=(20, 10))
sns.heatmap(df_transactions.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Pattern Across DataFrame')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()



numerical_cols = df_transactions.select_dtypes(include=['number']).columns

selected_numerical_cols = [
    'TransactionAmt',
    'TransactionDT',
    'isFraud',
    'card1',
    'card2',
    'card3',
    'card5',
    'D1',
    'D2',
    'D3',
    'D4',
    'D5',
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6',
    'V7',
    'V8',
    'V9'
]

selected_numerical_cols = [col for col in selected_numerical_cols if col in numerical_cols]

print(f"Selected {len(selected_numerical_cols)} numerical columns for distribution analysis:")
print(selected_numerical_cols)

import matplotlib.pyplot as plt
import seaborn as sns

for col in selected_numerical_cols:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Distribution of {col}', fontsize=16)

    # Histogram
    sns.histplot(df_transactions[col].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequency')

    # Box plot
    sns.boxplot(y=df_transactions[col].dropna(), ax=axes[1])
    axes[1].set_title(f'Box Plot of {col}')
    axes[1].set_ylabel(col)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

categorical_cols = df_transactions.select_dtypes(include='object').columns

print(f"Identified {len(categorical_cols)} categorical columns:")
print(categorical_cols.tolist())

for col in categorical_cols:
    print(f"\n--- Distribution for column: {col} ---")
    print(df_transactions[col].value_counts())

    plt.figure(figsize=(10, 6))
    sns.countplot(x=df_transactions[col], order=df_transactions[col].value_counts().index)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

numerical_df = df_transactions.select_dtypes(include=['number'])

correlation_matrix = numerical_df.corr()

plt.figure(figsize=(20, 18)) 
sns.heatmap(correlation_matrix, cmap='coolwarm', fmt=".2f", linewidths=.5)

plt.title('Correlation Matrix of Numerical Features', fontsize=20)

plt.show()

