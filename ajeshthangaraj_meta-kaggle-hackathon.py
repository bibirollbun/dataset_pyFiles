import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting a clean style
sns.set(style="whitegrid")


import kagglehub
path = kagglehub.dataset_download("kaggle/meta-kaggle")
print("Path to dataset files:", path)


# Loading the competitions data
df = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')

# Displaying the shape and the first few rows
print("Shape of the dataset:", df.shape)
df.head()


# Convert EnabledDate to datetime format
df['EnabledDate'] = pd.to_datetime(df['EnabledDate'], errors='coerce')

# Extract the year
df['Year'] = df['EnabledDate'].dt.year

# Count competitions per year
comp_per_year = df['Year'].value_counts().sort_index()
print(comp_per_year)


# Plot the number of competitions per year
plt.figure(figsize=(10, 6))
sns.barplot(x=comp_per_year.index, y=comp_per_year.values, palette='viridis')

plt.title('Number of Kaggle Competitions per Year')
plt.xlabel('Year')
plt.ylabel('Number of Competitions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


users_columns = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv', nrows=0)

# Print the column names
print("Columns in Users.csv:")
print(users_columns.columns.tolist())


users_df = pd.read_csv(
    '/kaggle/input/meta-kaggle/Users.csv',
    usecols=['Id', 'RegisterDate'],
    parse_dates=['RegisterDate']
)


# Extract year from RegisterDate
users_df['Year'] = users_df['RegisterDate'].dt.year

# Count users per year
users_per_year = users_df['Year'].value_counts().sort_index()
print(users_per_year)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(x=users_per_year.index, y=users_per_year.values, marker='o', color='blue')

plt.title('New Kaggle Users Registered Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


users_df = pd.read_csv(
    '/kaggle/input/meta-kaggle/Users.csv',
    usecols=['Id', 'RegisterDate', 'PerformanceTier'],
    parse_dates=['RegisterDate']
)

# Extract year from registration date
users_df['Year'] = users_df['RegisterDate'].dt.year


# Drop missing values
users_df = users_df.dropna(subset=['PerformanceTier', 'Year'])


# Group and count users by year and tier
tier_counts = users_df.groupby(['Year', 'PerformanceTier']).size().reset_index(name='Count')

# Replacing numeric tiers with labels
tier_labels = {
    1: 'Novice',
    2: 'Contributor',
    3: 'Expert',
    4: 'Master',
    5: 'Grandmaster'
}
tier_counts['Tier'] = tier_counts['PerformanceTier'].map(tier_labels)


# Remove Novice tier
filtered_df = users_df[users_df['PerformanceTier'] > 1]

# Group again
tier_counts_filtered = filtered_df.groupby(['Year', 'PerformanceTier']).size().reset_index(name='Count')
tier_counts_filtered['Tier'] = tier_counts_filtered['PerformanceTier'].map(tier_labels)

# Plot
plt.figure(figsize=(12, 7))
sns.lineplot(data=tier_counts_filtered, x='Year', y='Count', hue='Tier', marker='o')

plt.title('Growth of Contributor, Expert, Master & Grandmaster Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Users')
plt.legend(title='Performance Tier')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Filter only Masters and Grandmasters
elite_df = filtered_df[filtered_df['PerformanceTier'].isin([4, 5])]

# Group by year and tier
elite_counts = elite_df.groupby(['Year', 'PerformanceTier']).size().reset_index(name='Count')
elite_counts['Tier'] = elite_counts['PerformanceTier'].map(tier_labels)

# Plot only Masters and Grandmasters
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(data=elite_counts, x='Year', y='Count', hue='Tier', marker='o')

plt.title('Master & Grandmaster Growth Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)
plt.legend(title='Tier')
plt.tight_layout()
plt.show()


users_df = pd.read_csv(
    '/kaggle/input/meta-kaggle/Users.csv', usecols=['Id', 'Country']
)


# Drop rows with missing countries
users_df = users_df.dropna(subset=['Country'])

# Count users per country
country_counts = users_df['Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'UserCount']

# See top 10 countries
country_counts.head(10)


plt.figure(figsize=(12, 8))
sns.barplot(data=country_counts.head(20), x='UserCount', y='Country', palette='mako')

plt.title('Top 20 Countries by Number of Kaggle Users')
plt.xlabel('Number of Users')
plt.ylabel('Country')
plt.tight_layout()
plt.show()


users_df = pd.read_csv(
    '/kaggle/input/meta-kaggle/Users.csv',
    usecols=['Id', 'RegisterDate', 'Country'],
    parse_dates=['RegisterDate']
)

# Drop rows with missing country or date
users_df = users_df.dropna(subset=['RegisterDate', 'Country'])

# Extract year
users_df['Year'] = users_df['RegisterDate'].dt.year


country_year_df = users_df.groupby(['Country', 'Year']).size().reset_index(name='UserCount')


# Find top 5 countries with highest total users
top_countries = users_df['Country'].value_counts().head(5).index.tolist()

# Filter data for top countries
top_country_years = country_year_df[country_year_df['Country'].isin(top_countries)]


plt.figure(figsize=(12, 7))
sns.lineplot(data=top_country_years, x='Year', y='UserCount', hue='Country', marker='o')

plt.title('Yearly New User Growth for Top 5 Countries')
plt.xlabel('Year')
plt.ylabel('New Users Registered')
plt.legend(title='Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

