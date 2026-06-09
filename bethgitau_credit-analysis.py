# Import the libraries
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 


# Load Data
application_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
# 1. Preview the data
application_train.head()


application_train.info()


application_train.describe()


# 2. Check for missing values
missing = application_train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_ratio = (missing / len(application_train))

# Display top 10
missing_df = pd.DataFrame({'Missing Values': missing, 'Missing Ratio': missing_ratio})
missing_df.head(10)


# 3. Draw a graph showing class ratios
sns.set_style('whitegrid')

plt.figure(figsize=(6,4))
sns.countplot(x='TARGET', data=application_train, palette='viridis')
plt.title('Home Credit Repayment', fontsize=14)
plt.xlabel('Target (0 = Repaid, 1 = Defaulted)')
plt.ylabel('Count')

# Add percentages
total = len(application_train)
for p in plt.gca().patches:
    plt.text(p.get_x() + 0.3, p.get_height() + 2000, f'{100 * p.get_height() / total:.2f}%', fontsize=11)

plt.tight_layout()
plt.show()


# 1. Drop columns with over 70% missing values
high_missing_cols = missing_ratio[missing_ratio > 0.70].index.tolist()

application_train_cleaned = application_train.drop(columns=high_missing_cols)

# Keep only numeric columns
numeric_df = application_train_cleaned.select_dtypes(include='number')


# 2.. Correlation
correlations = numeric_df.corr()['TARGET'].sort_values(key=abs, ascending=False)
correlations.head(10)


# Now calculate correlations with the target
correlations = numeric_df.corr()['TARGET'].sort_values(key=abs, ascending=False)
positive_corr = correlations[correlations > 0].drop('TARGET').sort_values(ascending=False)
negative_corr = correlations[correlations < 0].drop('TARGET', errors='ignore').sort_values()
# Show top 10 features most correlated with TARGET
print(positive_corr.head(10))
print(negative_corr.head(10))


# 3. Visualization
# Combine top positive and negative correlations
top_positive = positive_corr.head(10)
top_negative = negative_corr.head(10)

# Concatenate them for visualization
top_corr_combined = pd.concat([top_positive, top_negative])

# Plot the top correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=top_corr_combined.values, y=top_corr_combined.index, palette='coolwarm')
plt.title('Top Features Correlated with Default (TARGET)', fontsize=14)
plt.xlabel('Correlation with TARGET')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# Visualization using heatmap for the overall matrix
corr_matrix = numeric_df.corr(numeric_only=True)
# Create a heatmap to find relationships with high correlation coefficient
plt.figure(figsize=(16,12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of All Features')
plt.show()


# Sort features based on correlation with TARGET
target_corr = corr_matrix['TARGET'].sort_values(key=abs, ascending=False)

# Select top 15 most correlated features (including TARGET itself)
top_corr_features = target_corr.head(15).index

# Draw a heatmap of just those
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df[top_corr_features].corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Heatmap of Top Correlated Features with TARGET', fontsize=16)
plt.show()


# Group data for total applications and default rates by gender
gender_total = application_train['CODE_GENDER'].value_counts().reset_index()
gender_total.columns = ['Gender', 'Count']

gender_stats = application_train.groupby('CODE_GENDER')['TARGET'].agg(['count', 'sum']).reset_index()
gender_stats.columns = ['Gender', 'Total', 'Defaults']
gender_stats['Default Rate (%)'] = 100 * gender_stats['Defaults'] / gender_stats['Total']

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Barplot: Total number of males and females
sns.barplot(x='Gender', y='Count', data=gender_total, palette='pastel', ax=axes[0])
axes[0].set_title('Total Number of Applicants by Gender')
axes[0].set_ylabel('Number of Applicants')
axes[0].set_xlabel('Gender')

# Annotate bars
for i, row in gender_total.iterrows():
    axes[0].text(i, row['Count'] + 2000, f"{row['Count']:,}", ha='center', fontsize=11)

# 2. Barplot: Default rate by gender
sns.barplot(x='Gender', y='Default Rate (%)', data=gender_stats, palette='Set2', ax=axes[1])
axes[1].set_title('Default Rate by Gender')
axes[1].set_ylabel('Default Rate (%)')
axes[1].set_xlabel('Gender')

# Annotate bars
for i, row in gender_stats.iterrows():
    axes[1].text(i, row['Default Rate (%)'] + 0.2, f"{row['Default Rate (%)']:.2f}%", ha='center', fontsize=11)

plt.tight_layout()
plt.show()


# Group by Income Type
income_total = application_train['NAME_INCOME_TYPE'].value_counts().reset_index()
income_total.columns = ['Income Type', 'Total Applicants']

income_stats = application_train.groupby('NAME_INCOME_TYPE')['TARGET'].agg(['count', 'sum']).reset_index()
income_stats.columns = ['Income Type', 'Total', 'Defaults']
income_stats['Default Rate (%)'] = 100 * income_stats['Defaults'] / income_stats['Total']

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 1. Plot: Total number of applicants by income type
sns.barplot(x='Total Applicants', y='Income Type', data=income_total, palette='pastel', ax=axes[0])
axes[0].set_title('Total Number of Applicants by Income Type')
axes[0].set_xlabel('Number of Applicants')
axes[0].set_ylabel('Income Type')

# Annotate bars
for i, row in income_total.iterrows():
    axes[0].text(row['Total Applicants'] + 500, i, f"{row['Total Applicants']:,}", va='center', fontsize=11)

# 2. Plot: Default Rate by income type
sns.barplot(x='Default Rate (%)', y='Income Type', data=income_stats.sort_values('Default Rate (%)', ascending=False),
            palette='Set2', ax=axes[1])
axes[1].set_title('Default Rate by Income Type')
axes[1].set_xlabel('Default Rate (%)')
axes[1].set_ylabel('Income Type')

# Annotate bars
for bar in axes[1].patches:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    axes[1].text(width + 0.5, y, f'{width:.2f}%', va='center', fontsize=11)

plt.tight_layout()
plt.show()


# Count and default stats by education level
education_stats = application_train.groupby('NAME_EDUCATION_TYPE')['TARGET'].agg(['count', 'sum']).reset_index()
education_stats.columns = ['Education Level', 'Total', 'Defaults']
education_stats['Default Rate (%)'] = 100 * education_stats['Defaults'] / education_stats['Total']
education_total = application_train['NAME_EDUCATION_TYPE'].value_counts().reset_index()
education_total.columns = ['Education Level', 'Total Applicants']

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 1. Total applicants
sns.barplot(x='Total Applicants', y='Education Level', data=education_total, palette='Blues_d', ax=axes[0])
axes[0].set_title('Total Applicants by Education Level')
axes[0].set_xlabel('Total Applicants')

for bar in axes[0].patches:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    axes[0].text(width + 500, y, f'{int(width):,}', va='center', fontsize=11)

# 2. Default rate
sorted_edu = education_stats.sort_values('Default Rate (%)', ascending=False)
sns.barplot(x='Default Rate (%)', y='Education Level', data=sorted_edu, palette='coolwarm', ax=axes[1])
axes[1].set_title('Default Rate by Education Level')
axes[1].set_xlabel('Default Rate (%)')

for bar in axes[1].patches:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    axes[1].text(width + 0.5, y, f'{width:.2f}%', va='center', fontsize=11)

plt.tight_layout()
plt.show()


# Create Age feature from DAYS_BIRTH
application_train['AGE_YEARS'] = (-application_train['DAYS_BIRTH']) // 365
application_train['AGE_GROUP'] = pd.cut(application_train['AGE_YEARS'],
                                        bins=[20, 30, 40, 50, 60, 70],
                                        labels=['20s', '30s', '40s', '50s', '60s'])

# Count and default rate by age group
age_stats = application_train.groupby('AGE_GROUP')['TARGET'].agg(['count', 'sum']).reset_index()
age_stats.columns = ['Age Group', 'Total', 'Defaults']
age_stats['Default Rate (%)'] = 100 * age_stats['Defaults'] / age_stats['Total']

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Total applicants
sns.barplot(x='Total', y='Age Group', data=age_stats, palette='Purples', ax=axes[0])
axes[0].set_title('Total Applicants by Age Group')
axes[0].set_xlabel('Total Applicants')

for bar in axes[0].patches:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    axes[0].text(width + 200, y, f'{int(width):,}', va='center', fontsize=11)

# 2. Default rate
sns.barplot(x='Default Rate (%)', y='Age Group', data=age_stats, palette='coolwarm', ax=axes[1])
axes[1].set_title('Default Rate by Age Group')
axes[1].set_xlabel('Default Rate (%)')

for bar in axes[1].patches:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    axes[1].text(width + 0.5, y, f'{width:.2f}%', va='center', fontsize=11)

plt.tight_layout()
plt.show()


# Drop missing occupation entries
occupation_data = application_train.dropna(subset=['OCCUPATION_TYPE'])

# Count and default rate by occupation
occ_stats = occupation_data.groupby('OCCUPATION_TYPE')['TARGET'].agg(['count', 'sum']).reset_index()
occ_stats.columns = ['Occupation', 'Total', 'Defaults']
occ_stats['Default Rate (%)'] = 100 * occ_stats['Defaults'] / occ_stats['Total']

# Sort for plotting
occ_stats_sorted = occ_stats.sort_values('Total', ascending=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 1. Total applicants
sns.barplot(x='Total', y='Occupation', data=occ_stats_sorted, palette='Greens', ax=axes[0])
axes[0].set_title('Total Applicants by Occupation')
axes[0].set_xlabel('Total Applicants')

for bar in axes[0].patches:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    axes[0].text(width + 300, y, f'{int(width):,}', va='center', fontsize=10)

# 2. Default rate
occ_stats_sorted_rate = occ_stats.sort_values('Default Rate (%)', ascending=False)
sns.barplot(x='Default Rate (%)', y='Occupation', data=occ_stats_sorted_rate, palette='coolwarm', ax=axes[1])
axes[1].set_title('Default Rate by Occupation')
axes[1].set_xlabel('Default Rate (%)')

for bar in axes[1].patches:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    axes[1].text(width + 0.5, y, f'{width:.2f}%', va='center', fontsize=10)

plt.tight_layout()
plt.show()

