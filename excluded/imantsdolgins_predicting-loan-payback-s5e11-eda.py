# === Import lib ===

import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

# Set a style
sns.set_style("whitegrid")


# === Load Data ===
train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


# === Explore Train Data ===
print("\n â˜‘ï¸� Train - Data Info:")
train_df.info()
print("\n â˜‘ï¸� Train - Statistical Summary:")
display(train_df.describe())
print("\n â˜‘ï¸� Train - First 10 Rows of the Dataset:")
display(train_df.head(10))


# === Explore Test Data ===
print("\n â˜‘ï¸� Test - Data Info:")
test_df.info()
print("\n â˜‘ï¸� Test - Statistical Summary:")
display(test_df.describe())
print("\n â˜‘ï¸� Test - First 10 Rows of the Dataset:")
display(test_df.head(10))


# === 1. Target Variable Analysis ===
print("ğŸ�¯ 1. Target Variable Analysis")
print("---------------------------------")

count_paid = train_df['loan_paid_back'].value_counts().get(1.0, 0)
count_not_paid = train_df['loan_paid_back'].value_counts().get(0.0, 0)

sizes = [count_paid, count_not_paid]
labels = ['Paid (1.0)', 'Not Paid (0.0)']
colors = ['#8338ec', '#ff006e']
explode = (0.05, 0)

# --- Print Report ---
total = sum(sizes)
perc_paid = (count_paid / total) * 100
perc_not_paid = (count_not_paid / total) * 100

print(f"Total Loans in Train Set: {total}")
print(f"âœ… Paid (1.0):     {count_paid} loans ({perc_paid:.1f}%)")
print(f"â�Œ Not Paid (0.0): {count_not_paid} loans ({perc_not_paid:.1f}%)")

plt.figure(figsize=(6, 6))

wedges, texts, autotexts = plt.pie(
    sizes, 
    colors=colors, 
    labels=labels,
    autopct='%1.1f%%',
    startangle=90, 
    explode=explode,
    pctdistance=0.85,
    wedgeprops={'edgecolor': 'white'}
)

centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Loan Payback Status Distribution', fontsize=16, weight='bold')
plt.axis('equal')

plt.text(0, 0, f'Total Loans:\n{total}', ha='center', va='center', fontsize=14, color='black')

for text in texts:
    text.set_color('grey')
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

plt.show()


print("ğŸ”¢ 2.2. Visualizing Numerical Feature Distributions")
print("--------------------------------------------------")

numerical_cols = [
    'annual_income', 
    'debt_to_income_ratio', 
    'credit_score', 
    'loan_amount', 
    'interest_rate'
]

target_map = {1.0: 'Paid (1.0)', 0.0: 'Not Paid (0.0)'}
palette = { 'Paid (1.0)': '#66b3ff', 'Not Paid (0.0)': '#ff9999'}

for col in numerical_cols:
    print(f"\nğŸ“Š Generating plots for: {col}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # --- Plot 1: Histogram / KDE Plot ---
    sns.histplot(
        data=train_df, 
        x=col, 
        hue=train_df['loan_paid_back'].map(target_map), 
        kde=True, 
        ax=ax1,
        palette=palette,
        element="step",
        stat="density",
        common_norm=False
    )
    ax1.set_title(f'Distribution of {col}', fontsize=14, weight='bold')
    ax1.set_xlabel(col, fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    
    # --- Plot 2: Box Plot ---
    sns.boxplot(
        data=train_df, 
        x=train_df['loan_paid_back'].map(target_map), 
        y=col, 
        ax=ax2,
        palette=palette
    )
    ax2.set_title(f'Box Plot of {col} by Payback Status', fontsize=14, weight='bold')
    ax2.set_xlabel('Loan Payback Status', fontsize=12)
    ax2.set_ylabel(col, fontsize=12)
    
    plt.tight_layout()
    plt.show()

print("\nâœ… All numerical plots generated.")


print("ğŸ“¦ 3. Categorical Features Analysis")
print("====================================")

categorical_cols = [
    'gender', 
    'marital_status', 
    'education_level', 
    'employment_status', 
    'loan_purpose', 
    'grade_subgrade'
]

target_map = {1.0: 'Paid (1.0)', 0.0: 'Not Paid (0.0)'}
palette = { 'Paid (1.0)': '#8338ec', 'Not Paid (0.0)': '#ff006e'}

for col in categorical_cols:
    print(f"\n\nğŸ“Š Analysis of Feature: {col}")
    print("--------------------------------" + "-" * len(col))
    
    # --- 1. Value Counts Report ---
    print("\nReport: Value Counts")
    print(train_df[col].value_counts())
    
    # --- 2. Payback Rate Report ---
    print(f"\nReport: Payback Rate by {col}")
    payback_rate = train_df.groupby(col)['loan_paid_back'].mean().sort_values(ascending=False) * 100
    print(payback_rate.round(1).astype(str) + '%')

    # --- 3. Visualization ---
    print("\nVisual: Generating plot...")
    
    if col == 'grade_subgrade':
        plt.figure(figsize=(14, 7))
    else:
        plt.figure(figsize=(12, 6))
    
    sns.countplot(
        data=train_df, 
        x=col, 
        hue=train_df['loan_paid_back'].map(target_map), 
        palette=palette,
        order=train_df[col].value_counts().index
    )
    
    plt.title(f'Distribution of {col} by Payback Status', fontsize=16, weight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel(col, fontsize=12)
    
    if col in ['loan_purpose', 'grade_subgrade']:
        plt.xticks(rotation=45, ha='right')

    plt.legend(title='Loan Payback Status')
    plt.tight_layout()
    plt.show()

print("\nâœ… All categorical plots generated.")


print("ğŸ§© 4. Correlation Analysis")
print("============================")

numerical_cols_with_target = [
    'annual_income', 
    'debt_to_income_ratio', 
    'credit_score', 
    'loan_amount', 
    'interest_rate',
    'loan_paid_back'
]

print("Calculating correlation matrix...")
corr_matrix = train_df[numerical_cols_with_target].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(
    corr_matrix, 
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.5,
    linecolor='white'
)

plt.title('Correlation Heatmap of Numerical Features & Target', fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

print("ğŸ“Š Generating heatmap...")
plt.show()

print("\n--- Correlation with Target (loan_paid_back) ---")
target_corr = corr_matrix['loan_paid_back'].sort_values(ascending=False)
print(target_corr)


print("ğŸ“ˆ 5. Distribution Shape Analysis")
print("=======================================")

numerical_cols = [
    'annual_income', 
    'debt_to_income_ratio', 
    'credit_score', 
    'loan_amount', 
    'interest_rate'
]

print("\n5.1. Skewness and Kurtosis Report (Train Set)")
print("---------------------------------------------")

skew_kurt_report = pd.DataFrame({
    'Skewness': train_df[numerical_cols].skew(),
    'Kurtosis': train_df[numerical_cols].kurtosis()
})
print(skew_kurt_report.round(2))


print("\nğŸ›¡ï¸� 5.2. Train vs. Test Distribution Comparison")
print("---------------------------------------------")
print("ğŸ“Š Generating comparison plots...")

numerical_cols = [
    'annual_income', 
    'debt_to_income_ratio', 
    'credit_score', 
    'loan_amount', 
    'interest_rate'
]

fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('Train vs. Test Distribution Comparison', fontsize=20, weight='bold')

axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    ax = axes[i]
    
    sns.kdeplot(train_df[col], label='Train', color='#66b3ff', shade=True, ax=ax)
    
    sns.kdeplot(test_df[col], label='Test', color='#ff9999', shade=True, ax=ax)
    
    ax.set_title(f'Distribution for {col}', fontsize=14)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()

if len(numerical_cols) < len(axes):
    axes[-1].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

print("\nâœ… All comparison plots generated.")


print("ğŸ�¨ 6. Pair Plot Analysis")
print("===========================")

numerical_cols_with_target = [
    'annual_income', 
    'debt_to_income_ratio', 
    'credit_score', 
    'loan_amount', 
    'interest_rate',
    'loan_paid_back'
]

# --- Create a Sample sample 20,000. ---
print("Sampling 20,000 rows for the pair plot...")
df_sample = train_df[numerical_cols_with_target].sample(n=20000, random_state=42)

# Map target to string labels for a clearer legend
target_map = {1.0: 'Paid (1.0)', 0.0: 'Not Paid (0.0)'}
df_sample['Payback Status'] = df_sample['loan_paid_back'].map(target_map)

print("ğŸ“Š Generating pair plot...")

g = sns.pairplot(
    df_sample, 
    hue='Payback Status',
    palette={'Paid (1.0)': '#8338ec', 'Not Paid (0.0)': '#ff006e'},
    vars=[col for col in numerical_cols_with_target if col != 'loan_paid_back'],
    diag_kind='kde',
    plot_kws={'alpha': 0.3, 's': 10},
    corner=True
)

g.fig.suptitle('Pair Plot of Numerical Features by Payback Status (on 20k Sample)', y=1.02, fontsize=16, weight='bold')
plt.show()

print("\nâœ… Pair plot generated.")

