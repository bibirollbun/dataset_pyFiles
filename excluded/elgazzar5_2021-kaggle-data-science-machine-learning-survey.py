
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
sns.set(style='whitegrid', palette='pastel', font_scale=1.1)

df = pd.read_csv('/kaggle/input/kaggle-survey-2021/kaggle_survey_2021_responses.csv', low_memory=False)
print("Original Shape:", df.shape)
df.head()

print("\n--- PREPROCESSING STARTED ---")

#Remove the first row (question text row)
df = df.iloc[1:]
df.reset_index(drop=True, inplace=True)
print("After removing question row:", df.shape)

# Check missing values
missing_summary = df.isnull().sum().sort_values(ascending=False)
print("\nTop 10 Columns with Missing Values:")
print(missing_summary.head(10))

#Drop columns with more than 60% missing data
df = df.loc[:, df.isnull().mean() < 0.6]
print("\nAfter dropping columns with >60% missing data:", df.shape)

# Fill missing values in remaining columns with 'Unknown'
df.fillna('Unknown', inplace=True)

#Remove duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

# 3.6 Check data types and preview
print("\nData types of sample columns:")
print(df.dtypes.head(10))

print("\n--- PREPROCESSING COMPLETED ---")

education_map = {
    'Doctoral degree': 4,
    'Masterâ€™s degree': 3,
    'Bachelorâ€™s degree': 2,
    'Some college/university study without earning a bachelorâ€™s degree': 1,
    'No formal education past high school': 0,
    'Unknown': -1
}
df['Q4'] = df['Q4'].map(education_map)

gender_map = {
    'Man': 0,
    'Woman': 1,
    'Nonbinary': 2,
    'Prefer not to say': 3,
    'Unknown': -1
}
df['Q2'] = df['Q2'].map(gender_map)

print("\nEncoded columns:")
print("Q2 â†’ Gender | Q4 â†’ Education Level")

#  Insight 1: Top 5 Countries by Respondents
top_countries = df['Q3'].value_counts().head(5)
plt.figure(figsize=(8,5))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top 5 Countries by Number of Respondents')
plt.xlabel('Respondents')
plt.ylabel('Country')
plt.show()

# Insight 2: Most Common Programming Languages
language_cols = [col for col in df.columns if 'Q7_Part_' in col]
langs = df[language_cols].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False).head(5)
langs.plot(kind='barh', figsize=(8,5), color='skyblue', title='Top 5 Programming Languages')
plt.xlabel('Count')
plt.show()

#  Insight 3: Education Level Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Q4', data=df)
plt.title('Distribution of Education Levels (Encoded)')
plt.xlabel('Education Level Code')
plt.ylabel('Respondents')
plt.show()

# Insight 4: Salary vs Age (Numeric Conversion Fix)
def convert_salary(value):
    """Convert salary range strings like '$0-999' to numeric midpoints"""
    if isinstance(value, str):
        value = value.replace(',', '').replace('$', '')
        if '-' in value:
            low, high = value.split('-')
            try:
                return (int(low) + int(high)) / 2
            except:
                return np.nan
        elif value.startswith('>'):
            try:
                return int(value[1:])
            except:
                return np.nan
    return np.nan

df['Salary_Numeric'] = df['Q25'].apply(convert_salary)

age_order = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44',
             '45-49', '50-54', '55-59', '60-69', '70+', 'Unknown']

plt.figure(figsize=(10,5))
sns.boxplot(x='Q1', y='Salary_Numeric', data=df, order=age_order)
plt.title('Average Salary Distribution by Age Group')
plt.xticks(rotation=45)
plt.ylabel('Salary (USD, Midpoint)')
plt.show()

ml_tools = [col for col in df.columns if 'Q16_Part_' in col]
tools = df[ml_tools].apply(pd.Series.value_counts).sum(axis=1).sort_values(ascending=False).head(5)
tools.plot(kind='bar', figsize=(8,5), color='orange', title='Top 5 Machine Learning Tools')
plt.ylabel('Count')
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(18,10))
fig.suptitle('ðŸ“Š Kaggle Data Science Survey 2021 â€” Summary Dashboard', fontsize=18, weight='bold')

#  Top Countries
sns.barplot(x=top_countries.values, y=top_countries.index, ax=axes[0,0])
axes[0,0].set_title('Top 5 Countries')

#  Programming Languages
langs.plot(kind='barh', ax=axes[0,1], color='teal')
axes[0,1].set_title('Top 5 Programming Languages')

#  Education
sns.countplot(x='Q4', data=df, ax=axes[0,2])
axes[0,2].set_title('Education Level Distribution')

#  Salary vs Age
sns.boxplot(x='Q1', y='Salary_Numeric', data=df, order=age_order, ax=axes[1,0])
axes[1,0].set_title('Salary vs Age Group')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].set_ylabel('Salary (USD)')

tools.plot(kind='bar', ax=axes[1,1], color='orange')
axes[1,1].set_title('Top 5 ML Tools')

axes[1,2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

df.to_csv('cleaned_kaggle_survey.csv', index=False)
print("\n Cleaned dataset saved as cleaned_kaggle_survey.csv")


print("\n--- SUMMARY OF PREPROCESSING ---")
print(" Removed first question row from dataset.")
print(" Dropped columns with >60% missing values.")
print(" Filled remaining missing cells with 'Unknown'.")
print(" Removed duplicate rows.")
print(" Encoded Gender (Q2) and Education (Q4) columns.")
print(" Created numeric salary column (Salary_Numeric).")
print(" Preprocessing completed successfully!")


