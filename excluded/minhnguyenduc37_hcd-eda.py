# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import cufflinks as cf

import os
print(os.listdir("../input/home-credit-default-risk"))
dataset_path = "../input/home-credit-default-risk/"


application_train = pd.read_csv(dataset_path + 'application_train.csv')
POS_CASH_balance = pd.read_csv(dataset_path + 'POS_CASH_balance.csv')
bureau_balance = pd.read_csv(dataset_path + 'bureau_balance.csv')
previous_application = pd.read_csv(dataset_path + 'previous_application.csv')
installments_payments = pd.read_csv(dataset_path + 'installments_payments.csv')
credit_card_balance = pd.read_csv(dataset_path + 'credit_card_balance.csv')
bureau = pd.read_csv(dataset_path + 'bureau.csv')
application_test = pd.read_csv(dataset_path + 'application_test.csv')

print('Size of application_train data', application_train.shape)
print('Size of POS_CASH_balance data', POS_CASH_balance.shape)
print('Size of bureau_balance data', bureau_balance.shape)
print('Size of previous_application data', previous_application.shape)
print('Size of installments_payments data', installments_payments.shape)
print('Size of credit_card_balance data', credit_card_balance.shape)
print('Size of bureau data', bureau.shape)


# Define a function to calculate and visualize missing data
def missing_data_summary(df, df_name):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (total / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    print(f"\nTop 20 columns with missing values in {df_name}:")
    display(missing_df.head(20))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=missing_df['Percent'].head(20),
        y=missing_df.index[:20],
        palette="viridis"
    )
    plt.title(f'Top 20 Missing Value Percentages in {df_name}')
    plt.xlabel('Percent Missing')
    plt.ylabel('Column Name')
    plt.tight_layout()
    plt.show()

    return missing_df

# Dictionary of datasets
datasets = {
    "application_train": application_train,
    "POS_CASH_balance": POS_CASH_balance,
    "bureau_balance": bureau_balance,
    "previous_application": previous_application,
    "installments_payments": installments_payments,
    "credit_card_balance": credit_card_balance,
    "bureau": bureau,
    "application_test": application_test
}

# Analyze and visualize missing data for each dataset
missing_data_results = {}
for name, df in datasets.items():
    missing_data_results[name] = missing_data_summary(df, name)



import matplotlib.pyplot as plt
import seaborn as sns

def target_bar_chart_horizontal(df, column="TARGET", title="Target Distribution"):
    counts = df[column].value_counts().sort_index()
    total = counts.sum()
    pct = counts / total * 100

    colors = ['#3CB371', '#FF6F61']  # Medium Sea Green & Light Coral

    plt.figure(figsize=(8, 4))
    sns.barplot(y=counts.index.astype(str), x=counts.values, palette=colors)

    # Add percentage labels at end of bars
    for i, (count, percent) in enumerate(zip(counts.values, pct)):
        plt.text(count + total * 0.005, i, f"{percent:.1f}%", va='center', fontsize=11)

    plt.title(title)
    plt.ylabel("Target")
    plt.xlabel("Count")

    # Short note below the chart
    plt.figtext(0.5, -0.1, "Target: 1 = payment difficulties; 0 = no difficulties",
                ha='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.show()

# Usage
target_bar_chart_horizontal(application_train)



import matplotlib.pyplot as plt
import seaborn as sns

def gender_bar_plot(df, column="CODE_GENDER", title="Gender Distribution"):
    counts = df[column].value_counts()
    total = counts.sum()
    pct = counts / total * 100
    labels = counts.index.tolist()
    
    plt.figure(figsize=(8,5))
    ax = sns.barplot(x=labels, y=counts.values, palette="pastel")
    
    # Add count and percentage labels on top of bars
    for i, (count, percent) in enumerate(zip(counts.values, pct.values)):
        ax.text(i, count + total * 0.01, f"{count}\n({percent:.1f}%)", 
                ha='center', va='bottom', fontsize=11)
    
    plt.title(title, fontsize=16)
    plt.ylabel("Count")
    plt.xlabel("Gender")
    plt.ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.show()

# Usage
gender_bar_plot(application_train)


import matplotlib.pyplot as plt
import seaborn as sns

def education_bar_plot(df, column="NAME_EDUCATION_TYPE", title="Education Distribution"):
    counts = df[column].value_counts().sort_values(ascending=False)
    total = counts.sum()
    pct = counts / total * 100

    plt.figure(figsize=(10,6))
    ax = sns.barplot(x=counts.values, y=counts.index, palette="mako_r")

    # Add counts and percentages on bars
    for i, (count, percent) in enumerate(zip(counts.values, pct.values)):
        ax.text(count + total*0.01, i, f'{count} ({percent:.1f}%)', va='center', fontsize=10)

    plt.title(title, fontsize=16)
    plt.xlabel("Count")
    plt.ylabel("Education Level")
    plt.xlim(0, counts.max()*1.2)
    plt.tight_layout()
    plt.show()

# Usage example
education_bar_plot(application_train)

import matplotlib.pyplot as plt
import seaborn as sns

def income_type_bar_plot(df, column="NAME_INCOME_TYPE", title="Income Type Distribution"):
    counts = df[column].value_counts().sort_values(ascending=False)
    total = counts.sum()
    pct = counts / total * 100

    plt.figure(figsize=(10,6))
    ax = sns.barplot(x=counts.values, y=counts.index, palette="coolwarm")

    # Add counts and percentages on bars
    for i, (count, percent) in enumerate(zip(counts.values, pct.values)):
        ax.text(count + total*0.01, i, f'{count} ({percent:.1f}%)', va='center', fontsize=10)

    plt.title(title, fontsize=16)
    plt.xlabel("Count")
    plt.ylabel("Income Type")
    plt.xlim(0, counts.max()*1.2)
    plt.tight_layout()
    plt.show()

# Usage example
income_type_bar_plot(application_train)

import matplotlib.pyplot as plt
import pandas as pd

def occupation_target_stacked_bar_fixed_spacing_pct_colors(df, occupation_col="OCCUPATION_TYPE", target_col="TARGET", title="Occupation Type Group"):
    # Group and unstack
    grouped = df.groupby([occupation_col, target_col]).size().unstack(fill_value=0)
    
    # Sort by total count descending
    grouped['total'] = grouped.sum(axis=1)
    grouped = grouped.sort_values(by='total', ascending=False)
    grouped = grouped.drop(columns='total')

    # Softer colors for bars
    colors = ['#4A90E2', '#F5A623']  # blue for 0 (good), orange for 1 (bad)

    # Plot stacked bar chart
    ax = grouped.plot(kind='bar', stacked=True, figsize=(14,7),
                      color=colors)

    plt.title(title, fontsize=16)
    plt.xlabel("Occupation Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')

    spacing = grouped.values.max() * 0.05  # fixed spacing

    # Add percentage labels above bars with fixed spacing
    for i, row in enumerate(grouped.itertuples(index=False)):
        total = sum(row)
        if total == 0:
            continue
        pct_0 = (row[0] / total * 100) if len(row) > 0 else 0
        pct_1 = (row[1] / total * 100) if len(row) > 1 else 0

        bar_top = total
        y_pos_target1 = bar_top + spacing
        y_pos_target0 = y_pos_target1 + spacing

        ax.text(i, y_pos_target1, f'{pct_1:.1f}%', ha='center', va='bottom', fontsize=11, color=colors[1], fontweight='bold')
        ax.text(i, y_pos_target0, f'{pct_0:.1f}%', ha='center', va='bottom', fontsize=11, color=colors[0], fontweight='bold')

    plt.ylim(top=grouped.values.max() + spacing * 3)
    plt.legend(title='TARGET', labels=['0', '1'])
    plt.tight_layout()
    plt.show()

# Usage
occupation_target_stacked_bar_fixed_spacing_pct_colors(application_train)



import matplotlib.pyplot as plt
import seaborn as sns

def loan_types_bar_chart(data, title="Loan Types Distribution"):
    plt.figure(figsize=(8,6))
    sns.barplot(x=data.index, y=data.values, palette="pastel")
    plt.title(title, fontsize=16)
    plt.xlabel("Loan Type")
    plt.ylabel("Count")
    
    # Add value labels on top of bars
    for i, v in enumerate(data.values):
        plt.text(i, v + max(data.values)*0.01, str(v), ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Usage
temp = application_train["NAME_CONTRACT_TYPE"].value_counts()
loan_types_bar_chart(temp)



import matplotlib.pyplot as plt
import seaborn as sns

def income_type_bar_chart(df, title="Income sources of Applicant's"):
    plt.figure(figsize=(10,6))
    sns.barplot(x='labels', y='values', data=df, palette='muted')
    plt.title(title, fontsize=16)
    plt.xlabel("Income Type")
    plt.ylabel("Count")

    total = df['values'].sum()
    # Add percentage and count labels on top of bars
    for i, row in df.iterrows():
        pct = row['values'] / total * 100
        label = f"{row['values']}\n({pct:.1f}%)"
        plt.text(i, row['values'] + total*0.01, label, ha='center', fontsize=11)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Usage
temp = application_train["NAME_INCOME_TYPE"].value_counts()
df = pd.DataFrame({'labels': temp.index, 'values': temp.values})
income_type_bar_chart(df)



import matplotlib.pyplot as plt
import seaborn as sns

def family_status_bar_chart(df, title="Family Status of Applicants"):
    plt.figure(figsize=(10,6))
    sns.barplot(x='labels', y='values', data=df, palette='Set2')
    plt.title(title, fontsize=16)
    plt.xlabel("Family Status")
    plt.ylabel("Count")

    total = df['values'].sum()
    for i, row in df.iterrows():
        pct = row['values'] / total * 100
        label = f"{row['values']}\n({pct:.1f}%)"
        plt.text(i, row['values'] + total*0.01, label, ha='center', fontsize=11)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Usage
temp = application_train["NAME_FAMILY_STATUS"].value_counts()
df = pd.DataFrame({'labels': temp.index, 'values': temp.values})
family_status_bar_chart(df)



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Prepare age in years, remove infinite or NaN values if any
age_0 = (-application_train.loc[application_train['TARGET'] == 0, 'DAYS_BIRTH']) / 365.25
age_0 = age_0.replace([np.inf, -np.inf], np.nan).dropna()

age_1 = (-application_train.loc[application_train['TARGET'] == 1, 'DAYS_BIRTH']) / 365.25
age_1 = age_1.replace([np.inf, -np.inf], np.nan).dropna()

plt.figure(figsize=(10,8))

sns.kdeplot(age_0, label='TARGET = 0 (No default)', fill=True, color='green')
sns.kdeplot(age_1, label='TARGET = 1 (Default)', fill=True, color='red')

plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Age Distribution by Loan Repayment Status')
plt.legend()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Filter out placeholder value
valid_days = application_train['DAYS_EMPLOYED'] != 365243

# Convert to years
emp_years_0 = (-application_train.loc[(application_train['TARGET'] == 0) & valid_days, 'DAYS_EMPLOYED']) / 365.25
emp_years_0 = emp_years_0.replace([np.inf, -np.inf], np.nan).dropna()

emp_years_1 = (-application_train.loc[(application_train['TARGET'] == 1) & valid_days, 'DAYS_EMPLOYED']) / 365.25
emp_years_1 = emp_years_1.replace([np.inf, -np.inf], np.nan).dropna()

# Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(emp_years_0, label='TARGET = 0 (No default)', fill=True, color='green', alpha=0.5)
sns.kdeplot(emp_years_1, label='TARGET = 1 (Default)', fill=True, color='red', alpha=0.5)

plt.xlabel('Years of Employment')
plt.ylabel('Density')
plt.title('Employment Duration Distribution by Repayment Status')
plt.legend()

# Show more x-axis ticks
max_years = max(emp_years_0.max(), emp_years_1.max())
ticks = np.arange(0, int(max_years) + 5, 2)
plt.xticks(ticks)

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
from scipy.stats import pearsonr, norm

# Feature engineering (add only if columns exist)
if 'AMT_ANNUITY' in application_train.columns and 'AMT_CREDIT' in application_train.columns:
    application_train['PAYMENT_RATE'] = application_train['AMT_ANNUITY'] / application_train['AMT_CREDIT']

if 'AMT_INCOME_TOTAL' in application_train.columns and 'AMT_CREDIT' in application_train.columns:
    application_train['INCOME_CREDIT_PERC'] = application_train['AMT_INCOME_TOTAL'] / application_train['AMT_CREDIT']

if 'AMT_INCOME_TOTAL' in application_train.columns and 'AMT_ANNUITY' in application_train.columns:
    application_train['INCOME_TO_ANNUITY_RATIO'] = application_train['AMT_INCOME_TOTAL'] / application_train['AMT_ANNUITY']

if 'AMT_CREDIT' in application_train.columns and 'AMT_GOODS_PRICE' in application_train.columns:
    application_train['CREDIT_TO_GOODS_RATIO'] = application_train['AMT_CREDIT'] / application_train['AMT_GOODS_PRICE']

features = [
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CREDIT_TO_GOODS_RATIO',
    'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT',
    'DAYS_LAST_PHONE_CHANGE', 'OWN_CAR_AGE', 'DEF_30/60_CNT_SOCIAL_CIRCLE',
    'PAYMENT_RATE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'INCOME_CREDIT_PERC', 'INCOME_TO_ANNUITY_RATIO'
]

# Filter features to those existing in your dataframe
features = [f for f in features if f in application_train.columns]

def correlation_with_ci(data, target_col, features, confidence=0.95):
    results = []
    for feature in features:
        df = data[[target_col, feature]].dropna()
        if df.shape[0] < 3:
            continue
        x = df[target_col]
        y = df[feature]

        r, p_value = pearsonr(x, y)

        n = len(df)
        stderr = 1.0 / np.sqrt(n - 3)
        delta = norm.ppf((1 + confidence) / 2) * stderr
        z = np.arctanh(r)
        lo_z, hi_z = z - delta, z + delta
        ci_low, ci_high = np.tanh((lo_z, hi_z))

        results.append({
            'Feature': feature,
            'Correlation': r,
            'P-Value': p_value,
            '95% CI Lower': ci_low,
            '95% CI Upper': ci_high,
            'N Obs': n
        })
    return pd.DataFrame(results).sort_values(by='Correlation', ascending=False)

correlation_results = correlation_with_ci(application_train, 'TARGET', features)
print(correlation_results)


