# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Essential Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import ipywidgets as widgets
from IPython.display import display
import warnings
from tabulate import tabulate

import matplotlib.pyplot as plt
import seaborn as sns

# Set global plot style for consistency
sns.set_style('darkgrid')
sns.set_palette('Set2')


# Niche Imports
import chardet


# Tools
def load_csv(csv_file):
    with open(csv_file, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
    
    # print(result)  # Check detected encoding
    df = pd.read_csv(csv_file, encoding=result["encoding"])
    return df


# Load the datasets
mens_df = load_csv(os.path.join(dirname, 'MTeams.csv'))
womens_df = load_csv(os.path.join(dirname, 'WTeams.csv'))

# Iterate through both dataframes for EDA
for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*30}\nInvestigating {gender} Teams\n{'='*30}\n")
    
    # Display basic information
    print("Basic Information:")
    display(df.info())
    
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    display(df.describe())
    
    # Display first few rows of the dataframe
    print("\nFirst 5 Rows of the Data:")
    display(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    display(df.isnull().sum())
    
    # Display unique values in categorical columns
    print(f"\nNumber of Unique TeamIDs: {df['TeamID'].nunique()}")
    print(f"Number of Unique TeamNames: {df['TeamName'].nunique()}")



for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*50}\nMono Variance Analysis: {gender} Teams\n{'='*50}\n")

    # Select only numerical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Initialize a dictionary to store results
    analysis_results = []

    for col in numeric_cols:
        data = df[col]
        variance = data.var()
        std_dev = data.std()
        mean = data.mean()
        cv = (std_dev / mean) * 100 if mean != 0 else 0
        iqr = data.quantile(0.75) - data.quantile(0.25)
        skewness = data.skew()
        kurtosis = data.kurt()

        analysis_results.append([
            col,
            round(mean, 2),
            round(data.median(), 2),
            round(variance, 2),
            round(std_dev, 2),
            round(cv, 2),
            data.min(),
            data.max(),
            data.max() - data.min(),
            round(iqr, 2),
            round(skewness, 2),
            round(kurtosis, 2)
        ])
    
    # Display the analysis results using tabulate for a formatted table
    headers = [
        "Feature", "Mean", "Median", "Variance", "Std Dev",
        "CV (%)", "Min", "Max", "Range", "IQR", "Skewness", "Kurtosis"
    ]
    print(tabulate(analysis_results, headers=headers, tablefmt="fancy_grid"))


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global plot style for consistency
plt.style.use('seaborn-darkgrid')
sns.set_palette('Set2')

# Assuming mens_df and womens_df are already loaded
for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*50}")
    print(f"Graphical Mono Variance Analysis: {gender} Teams")
    print(f"{'='*50}\n")
    
    # Select only numerical columns and replace inf values with NaN
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Create plots for each numerical column
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(2, 2, 1)
        sns.histplot(df[col].dropna(), kde=True, color='skyblue', bins=20)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Box Plot
        plt.subplot(2, 2, 2)
        sns.boxplot(x=df[col].dropna(), color='lightgreen')
        plt.title(f'Box Plot of {col}')
        plt.xlabel(col)

        # Density Plot
        plt.subplot(2, 2, 3)
        sns.kdeplot(df[col].dropna(), fill=True, color='salmon')
        plt.title(f'Density Plot of {col}')
        plt.xlabel(col)

        # Bar Plot for Standard Deviation and Variance
        plt.subplot(2, 2, 4)
        variance = df[col].var()
        std_dev = df[col].std()
        cv = (std_dev / df[col].mean()) * 100 if df[col].mean() != 0 else 0
        
        metrics = ['Variance', 'Standard Deviation', 'CV (%)']
        values = [variance, std_dev, cv]
        
        sns.barplot(x=metrics, y=values, palette='viridis')
        plt.title(f'Variance Metrics of {col}')
        plt.xlabel('Metric')
        plt.ylabel('Value')

        plt.tight_layout()
        plt.show()



for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*50}")
    print(f"Non-Graphical Multivariate Analysis: {gender} Teams")
    print(f"{'='*50}\n")

    # Select numerical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # 1. Correlation Matrix
    print("\nğŸ“ˆ Correlation Matrix:")
    correlation_matrix = df[numeric_cols].corr()
    print(correlation_matrix)

    # 2. Covariance Matrix
    print("\nğŸ”— Covariance Matrix:")
    covariance_matrix = df[numeric_cols].cov()
    print(covariance_matrix)

    # 3. Grouped Summary Statistics (e.g., by TeamName)
    # This is more illustrative; not typically useful for TeamName
    print("\nğŸ“Š Summary Statistics Grouped by TeamName:")
    grouped_stats = df.groupby('TeamName')[numeric_cols].mean().head()
    print(grouped_stats)


for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*50}")
    print(f"ğŸŒŸ Graphical Multivariate Analysis: {gender} Teams ğŸŒŸ")
    print(f"{'='*50}\n")
    
    # Select numerical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # 1. Heatmap of Correlation Matrix
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'Correlation Heatmap: {gender} Teams')
    plt.show()

    # 2. Pairplot to visualize pairwise relationships
    sns.pairplot(df[numeric_cols])
    plt.suptitle(f'Pairplot of Numerical Features: {gender} Teams', y=1.02)
    plt.show()
    
    # 3. Scatter Plot (e.g., FirstD1Season vs LastD1Season)
    if 'FirstD1Season' in numeric_cols and 'LastD1Season' in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='FirstD1Season', y='LastD1Season', hue='TeamName', palette='tab10', legend=False)
        plt.title(f'FirstD1Season vs LastD1Season: {gender} Teams')
        plt.xlabel('FirstD1Season')
        plt.ylabel('LastD1Season')
        plt.show()


# Load the datasets
mens_df = load_csv(os.path.join(dirname, 'MSeasons.csv'))
womens_df = load_csv(os.path.join(dirname, 'WSeasons.csv'))

# Iterate through both dataframes for EDA
for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*30}\nInvestigating {gender} Seasons\n{'='*30}\n")
    
    # Display basic information
    print("Basic Information:")
    display(df.info())
    
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    display(df.describe())
    
    # Display first few rows of the dataframe
    print("\nFirst 5 Rows of the Data:")
    display(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    display(df.isnull().sum())
    
    # Display unique values in categorical columns
    print(f"\nNumber of Unique Seasons: {df['Season'].nunique()}")
    print(f"Number of Unique DayZeros: {df['DayZero'].nunique()}")



for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*50}")
    print(f"ğŸŒŸ Univariate Non-Graphical Analysis: {gender} Seasons ğŸŒŸ")
    print(f"{'='*50}\n")

    # Numerical Analysis - Season
    print("\nğŸ“ˆ Numerical Analysis for 'Season':")
    season_stats = {
        'Count': df['Season'].count(),
        'Mean': df['Season'].mean(),
        'Median': df['Season'].median(),
        'Variance': df['Season'].var(),
        'Standard Deviation': df['Season'].std(),
        'Min': df['Season'].min(),
        'Max': df['Season'].max(),
        'Range': df['Season'].max() - df['Season'].min(),
        'IQR': df['Season'].quantile(0.75) - df['Season'].quantile(0.25),
        'Skewness': df['Season'].skew(),
        'Kurtosis': df['Season'].kurt()
    }
    for stat, value in season_stats.items():
        print(f"{stat}: {value:.2f}")

    # Categorical Analysis
    categorical_cols = ['DayZero', 'RegionW', 'RegionX', 'RegionY', 'RegionZ']
    print("\nğŸ“Š Categorical Analysis:")
    for col in categorical_cols:
        print(f"\nTop 5 Most Frequent Values in '{col}':")
        print(df[col].value_counts().head())

    # Unique Values in Categorical Columns
    print("\nğŸ”� Unique Values in Categorical Columns:")
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")


for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*50}")
    print(f"ğŸŒŸ Graphical Univariate Analysis: {gender} Seasons ğŸŒŸ")
    print(f"{'='*50}\n")
    
    # 1. Numerical Data: Season Column
    plt.figure(figsize=(15, 5))

    # Histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df['Season'], kde=True, color='skyblue', bins=15)
    plt.title(f'{gender} Seasons - Histogram')
    plt.xlabel('Season')
    plt.ylabel('Frequency')

    # Box Plot
    plt.subplot(1, 3, 2)
    sns.boxplot(x=df['Season'], color='lightgreen')
    plt.title(f'{gender} Seasons - Box Plot')
    plt.xlabel('Season')

    # Density Plot
    plt.subplot(1, 3, 3)
    sns.kdeplot(df['Season'], fill=True, color='salmon')
    plt.title(f'{gender} Seasons - Density Plot')
    plt.xlabel('Season')

    plt.tight_layout()
    plt.show()
    
    # 2. Categorical Data: DayZero and Regions
    categorical_cols = ['DayZero', 'RegionW', 'RegionX', 'RegionY', 'RegionZ']
    
    for col in categorical_cols:
        plt.figure(figsize=(8, 6))
        top_values = df[col].value_counts().head(10)  # Show top 10 most frequent values
        sns.barplot(x=top_values.values, y=top_values.index, palette='viridis')
        plt.title(f'{gender} Seasons - Top 10 Values in {col}')
        plt.xlabel('Frequency')
        plt.ylabel(col)
        plt.show()


for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*50}")
    print(f"ğŸŒŸ Non-Graphical Multivariate Analysis: {gender} Seasons ğŸŒŸ")
    print(f"{'='*50}\n")

    # 1. Correlation Matrix (Only Numerical Columns)
    print("\nğŸ“ˆ Correlation Matrix:")
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numerical_df.corr()
    print(correlation_matrix)

    # 2. Crosstab Analysis - Showing Region distribution per Season
    print("\nğŸ“Š Crosstab Analysis for 'RegionW' and 'Season':")
    crosstab_regionW = pd.crosstab(df['Season'], df['RegionW'])
    print(crosstab_regionW.head())

    # 3. Pivot Table - Frequency of Regions over Seasons
    print("\nğŸ“… Pivot Table: Frequency of 'RegionX' by 'Season':")
    pivot_table = df.pivot_table(index='Season', columns='RegionX', aggfunc='size', fill_value=0)
    print(pivot_table.head())


for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*50}")
    print(f"ğŸŒŸ Graphical Multivariate Analysis: {gender} Seasons ğŸŒŸ")
    print(f"{'='*50}\n")

    # 1. Heatmap of the Correlation Matrix (Numerical Columns Only)
    plt.figure(figsize=(5, 3))
    numerical_df = df.select_dtypes(include=['int64', 'float64'])  # Filter only numerical columns
    correlation_matrix = numerical_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'Correlation Heatmap: {gender} Seasons')
    plt.show()
    
    # 2. Clustered Bar Chart: Frequency of Regions by Season
    categorical_cols = ['RegionW', 'RegionX', 'RegionY', 'RegionZ']
    
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        region_by_season = pd.crosstab(df['Season'], df[col])
        region_by_season.plot(kind='bar', stacked=False, figsize=(12, 6), colormap='tab20')
        plt.title(f'{gender} Seasons - Frequency of {col} by Season')
        plt.xlabel('Season')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
    # 3. Stacked Bar Chart: Distribution of Regions by Season
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        region_by_season.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
        plt.title(f'{gender} Seasons - Stacked Distribution of {col} by Season')
        plt.xlabel('Season')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    
    # 4. Line Plot: Trend of a Specific Region's Occurrences over Seasons
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        top_regions = df[col].value_counts().head(5).index
        trend_data = df[df[col].isin(top_regions)].pivot_table(index='Season', columns=col, aggfunc='size', fill_value=0)
        sns.lineplot(data=trend_data, palette='tab10', marker='o')
        plt.title(f'Trend of Top {col} Regions Over Seasons: {gender}')
        plt.xlabel('Season')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


# Load the datasets
mens_df = load_csv(os.path.join(dirname, 'MNCAATourneySeeds.csv'))
womens_df = load_csv(os.path.join(dirname, 'WNCAATourneySeeds.csv'))

# Iterate through both dataframes for EDA
for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*30}\nInvestigating {gender} Tourney\n{'='*30}\n")
    
    # Display basic information
    print("Basic Information:")
    display(df.info())
    
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    display(df.describe())
    
    # Display first few rows of the dataframe
    print("\nFirst 5 Rows of the Data:")
    display(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    display(df.isnull().sum())
    
    # Display unique values in categorical columns
    print(f"\nNumber of Unique Team IDS: {df['TeamID'].nunique()}")
    print(f"Number of Unique Seasons: {df['Season'].nunique()}")


for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*50}")
    print(f"ğŸŒŸ Non-Graphical Univariate Analysis: {gender} Tourney ğŸŒŸ")
    print(f"{'='*50}\n")
    
    # Numerical Analysis for 'Season' and 'TeamID'
    for col in ['Season', 'TeamID']:
        print(f"\n--- {col} Statistics ---")
        stats = {
            'Count': df[col].count(),
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Variance': df[col].var(),
            'Standard Deviation': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Range': df[col].max() - df[col].min(),
            'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
            'Skewness': df[col].skew(),
            'Kurtosis': df[col].kurt()
        }
        for k, v in stats.items():
            print(f"{k}: {v:.2f}")
    
    # Categorical Analysis for 'Seed'
    print("\n--- Seed Frequency Analysis ---")
    print(df['Seed'].value_counts().head(10))
    
    # Extract region from Seed (first character) for further insight
    df['Region'] = df['Seed'].str[0]
    print("\n--- Seed Region Frequency Analysis ---")
    print(df['Region'].value_counts())


# Graphical Analysis for each dataset
for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*50}")
    print(f"ğŸŒŸ Graphical Univariate Analysis: {gender} Tourney ğŸŒŸ")
    print(f"{'='*50}\n")
    
    # 1. Numerical Analysis for 'Season'
    plt.figure(figsize=(18,5))
    
    # Histogram with KDE
    plt.subplot(1, 3, 1)
    sns.histplot(df['Season'], kde=True, color='skyblue', bins=15)
    plt.title(f'{gender} Tourney - Season Histogram')
    plt.xlabel('Season')
    
    # Box Plot
    plt.subplot(1, 3, 2)
    sns.boxplot(x=df['Season'], color='lightgreen')
    plt.title(f'{gender} Tourney - Season Box Plot')
    plt.xlabel('Season')
    
    # Density Plot
    plt.subplot(1, 3, 3)
    sns.kdeplot(df['Season'], fill=True, color='salmon')
    plt.title(f'{gender} Tourney - Season Density Plot')
    plt.xlabel('Season')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Categorical Analysis for 'Seed'
    plt.figure(figsize=(10,6))
    sns.countplot(y='Seed', data=df, order=df['Seed'].value_counts().index)
    plt.title(f'{gender} Tourney - Seed Frequency')
    plt.xlabel('Count')
    plt.ylabel('Seed')
    plt.tight_layout()
    plt.show()
    
    # 3. Categorical Analysis for Extracted 'Region'
    # Ensure the Region column exists (if not already added in non-graphical section)
    if 'Region' not in df.columns:
        df['Region'] = df['Seed'].str[0]
    plt.figure(figsize=(8,6))
    sns.countplot(x='Region', data=df, order=df['Region'].value_counts().index)
    plt.title(f'{gender} Tourney - Seed Region Frequency')
    plt.xlabel('Region')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print("\n" + "="*50)
    print(f"ğŸŒŸ Non-Graphical Multivariate Analysis: {gender} Tourney ğŸŒŸ")
    print("="*50 + "\n")
    
    # Ensure that a Region column exists by extracting the first character from Seed
    df['Region'] = df['Seed'].str[0]
    
    # 1. Correlation Matrix for numerical features
    print("ğŸ“ˆ Correlation Matrix (Numerical Features):")
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numerical_df.corr()
    print(correlation_matrix, "\n")
    
    # 2. Pivot Table: Frequency of Regions by Season
    print("ğŸ“… Pivot Table: Frequency of Regions by Season:")
    pivot_table = df.pivot_table(index='Season', columns='Region', aggfunc='size', fill_value=0)
    print(pivot_table.head(), "\n")
    
    # 3. Crosstab Analysis: Region vs. Seed frequency
    print("ğŸ“Š Crosstab: Region vs. Seed:")
    crosstab_region_seed = pd.crosstab(df['Region'], df['Seed'])
    print(crosstab_region_seed.head(), "\n")
    
    # 4. Grouped Summary Statistics by Region
    print("ğŸ“‹ Grouped Summary Statistics by Region:")
    grouped_stats = df.groupby('Region')[['Season', 'TeamID']].agg(['mean', 'median', 'std', 'min', 'max'])
    print(grouped_stats, "\n")


for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    # Ensure the Region column is present
    if 'Region' not in df.columns:
        df['Region'] = df['Seed'].str[0]
    
    print("\n" + "="*50)
    print(f"ğŸŒŸ Graphical Multivariate Analysis: {gender} Tourney ğŸŒŸ")
    print("="*50 + "\n")
    
    # 1. Heatmap of the Correlation Matrix (Numerical Only)
    plt.figure(figsize=(5, 3))
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numerical_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"{gender} Tourney: Correlation Heatmap")
    plt.show()
    
    # 2. Clustered Bar Chart: Frequency of Regions by Season
    pivot_table = df.pivot_table(index='Season', columns='Region', aggfunc='size', fill_value=0)
    pivot_table.plot(kind='bar', figsize=(12, 6))
    plt.title(f"{gender} Tourney: Frequency of Regions by Season (Clustered Bar)")
    plt.xlabel("Season")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # 3. Stacked Bar Chart: Frequency of Regions by Season
    pivot_table.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title(f"{gender} Tourney: Frequency of Regions by Season (Stacked Bar)")
    plt.xlabel("Season")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # 4. Line Plot: Trend of Region Frequency Over Seasons
    plt.figure(figsize=(10, 6))
    for region in pivot_table.columns:
        plt.plot(pivot_table.index, pivot_table[region], marker='o', label=region)
    plt.title(f"{gender} Tourney: Trend of Region Frequency Over Seasons")
    plt.xlabel("Season")
    plt.ylabel("Count")
    plt.legend(title="Region")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Load the datasets
mens_df = load_csv(os.path.join(dirname, 'MRegularSeasonCompactResults.csv'))
womens_df = load_csv(os.path.join(dirname, 'WRegularSeasonCompactResults.csv'))

# Iterate through both dataframes for EDA
for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*30}\nInvestigating {gender} Compact Results\n{'='*30}\n")
    
    # Display basic information
    print("Basic Information:")
    display(df.info())
    
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    display(df.describe())
    
    # Display first few rows of the dataframe
    print("\nFirst 5 Rows of the Data:")
    display(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    display(df.isnull().sum())
    
    # Display unique values in categorical columns
    print(f"\nNumber of Unique Wining Team IDS: {df['WTeamID'].nunique()}")
    print(f"Number of Unique Losing Team IDS: {df['LTeamID'].nunique()}")


numerical_cols = ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT']

for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print("\n" + "="*50)
    print(f"ğŸŒŸ Non-Graphical Univariate Analysis: {gender} Regular Season Compact Results ğŸŒŸ")
    print("="*50 + "\n")
    
    # Numerical Analysis
    for col in numerical_cols:
        print(f"\n--- {col} Statistics ---")
        stats = {
            'Count': df[col].count(),
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Variance': df[col].var(),
            'Std Dev': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Range': df[col].max() - df[col].min(),
            'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
            'Skewness': df[col].skew(),
            'Kurtosis': df[col].kurt()
        }
        for k, v in stats.items():
            print(f"{k}: {v:.2f}")
    
    # Categorical Analysis for Winning Location
    print("\n--- WLoc Frequency Analysis ---")
    print(df['WLoc'].value_counts())


for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print("\n" + "="*50)
    print(f"ğŸŒŸ Graphical Univariate Analysis: {gender} Regular Season Compact Results ğŸŒŸ")
    print("="*50 + "\n")
    
    # Graphical analysis for each numerical column
    for col in numerical_cols:
        plt.figure(figsize=(15, 4))
        
        # Histogram with KDE
        plt.subplot(1, 3, 1)
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f"{gender} {col} Histogram & KDE")
        plt.xlabel(col)
        
        # Box Plot
        plt.subplot(1, 3, 2)
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f"{gender} {col} Box Plot")
        plt.xlabel(col)
        
        # Density Plot
        plt.subplot(1, 3, 3)
        sns.kdeplot(df[col], fill=True, color='salmon')
        plt.title(f"{gender} {col} Density Plot")
        plt.xlabel(col)
        
        plt.tight_layout()
        plt.show()
    
    # Graphical analysis for categorical variable WLoc
    plt.figure(figsize=(8, 6))
    sns.countplot(x='WLoc', data=df, order=df['WLoc'].value_counts().index)
    plt.title(f"{gender} WLoc Count Plot")
    plt.xlabel("WLoc")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# Define numerical columns
numerical_cols = ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT']

for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print("\n" + "="*50)
    print(f"ğŸŒŸ Non-Graphical Multivariate Analysis: {gender} Regular Season Compact Results ğŸŒŸ")
    print("="*50 + "\n")
    
    # 1. Correlation Matrix for numerical features
    num_df = df[numerical_cols]
    print("ğŸ“ˆ Correlation Matrix:")
    print(num_df.corr())
    print("\n")
    
    # 2. Pivot Table: Average Winning and Losing Scores by Season and WLoc
    print("ğŸ“… Pivot Table: Average Scores by Season and WLoc:")
    pivot_scores = df.pivot_table(index='Season', columns='WLoc', values=['WScore', 'LScore'], aggfunc='mean')
    print(pivot_scores.head())
    print("\n")
    
    # 3. Grouped Summary Statistics: Average Number of Overtime Periods by Season and WLoc
    print("ğŸ“‹ Grouped Summary: Average NumOT by Season and WLoc:")
    group_ot = df.groupby(['Season', 'WLoc'])['NumOT'].mean().unstack()
    print(group_ot.head())
    print("\n")
    
    # 4. Crosstab: Frequency Count of WLoc vs. Binned DayNum
    # Bin DayNum into intervals (e.g., [0, 30, 60, 90, 120, 132])
    df['DayNum_bin'] = pd.cut(df['DayNum'], bins=[0, 30, 60, 90, 120, 132], include_lowest=True)
    print("ğŸ“Š Crosstab: WLoc vs. DayNum bins:")
    crosstab_daynum = pd.crosstab(df['WLoc'], df['DayNum_bin'])
    print(crosstab_daynum)


for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    # Ensure a fresh copy for plotting if modifications were made earlier
    df_plot = df.copy()
    
    print("\n" + "="*50)
    print(f"ğŸŒŸ Graphical Multivariate Analysis: {gender} Regular Season Compact Results ğŸŒŸ")
    print("="*50 + "\n")
    
    # 1. Heatmap of Correlation Matrix (Numerical Variables Only)
    numerical_cols = ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT']
    num_df = df_plot[numerical_cols]
    plt.figure(figsize=(8, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"{gender} Regular Season: Correlation Heatmap")
    plt.show()
    
    # 2. Pairplot for a Sample of the Data
    # Sampling if the dataset is very large
    sample_df = df_plot.sample(n=1000, random_state=42) if len(df_plot) > 1000 else df_plot
    sns.pairplot(sample_df[numerical_cols])
    plt.suptitle(f"{gender} Regular Season: Pairplot of Numerical Variables", y=1.02)
    plt.show()
    
    # 3. Scatter Plot: Winning Score vs. Losing Score Colored by WLoc
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x='WScore', y='LScore', hue='WLoc', alpha=0.5)
    plt.title(f"{gender} Regular Season: WScore vs. LScore by WLoc")
    plt.xlabel("Winning Score")
    plt.ylabel("Losing Score")
    plt.tight_layout()
    plt.show()
    
    # 4. Line Plot: Trend of Average Scores by Season
    scores_by_season = df_plot.groupby('Season').agg({'WScore':'mean', 'LScore':'mean'}).reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(scores_by_season['Season'], scores_by_season['WScore'], marker='o', label='Average WScore')
    plt.plot(scores_by_season['Season'], scores_by_season['LScore'], marker='s', label='Average LScore')
    plt.title(f"{gender} Regular Season: Average Scores by Season")
    plt.xlabel("Season")
    plt.ylabel("Average Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Load the datasets
mens_df = load_csv(os.path.join(dirname, 'MNCAATourneyCompactResults.csv'))
womens_df = load_csv(os.path.join(dirname, 'WNCAATourneyCompactResults.csv'))

# Iterate through both dataframes for EDA
for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*30}\nInvestigating {gender} Tourney Results\n{'='*30}\n")
    
    # Display basic information
    print("Basic Information:")
    display(df.info())
    
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    display(df.describe())
    
    # Display first few rows of the dataframe
    print("\nFirst 5 Rows of the Data:")
    display(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    display(df.isnull().sum())
    
    # Display unique values in categorical columns
    print(f"\nNumber of Unique Wining Team IDS: {df['WTeamID'].nunique()}")
    print(f"Number of Unique Losing Team IDS: {df['LTeamID'].nunique()}")    
    print(f"Number of Unique Days: {df['DayNum'].nunique()}")


# Define the numerical columns for analysis
numerical_cols = ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT']

for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print("\n" + "="*50)
    print(f"ğŸŒŸ Non-Graphical Univariate Analysis: {gender} Tourney Results ğŸŒŸ")
    print("="*50 + "\n")
    
    # Numerical Analysis for each column
    for col in numerical_cols:
        print(f"\n--- {col} Statistics ---")
        stats = {
            'Count': df[col].count(),
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Variance': df[col].var(),
            'Std Dev': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Range': df[col].max() - df[col].min(),
            'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
            'Skewness': df[col].skew(),
            'Kurtosis': df[col].kurt()
        }
        for k, v in stats.items():
            print(f"{k}: {v:.2f}")
    
    # Categorical Analysis for WLoc
    print("\n--- WLoc Frequency Analysis ---")
    print(df['WLoc'].value_counts())
    
    # Unique Days Analysis
    print("\n--- Unique DayNum Values ---")
    print(f"Number of Unique Days: {df['DayNum'].nunique()}")



for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print("\n" + "="*50)
    print(f"ğŸŒŸ Graphical Univariate Analysis: {gender} Tourney Results ğŸŒŸ")
    print("="*50 + "\n")
    
    # Graphical analysis for each numerical column
    for col in numerical_cols:
        plt.figure(figsize=(15, 4))
        
        # Histogram with KDE overlay
        plt.subplot(1, 3, 1)
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f"{gender} {col} Histogram & KDE")
        plt.xlabel(col)
        
        # Box Plot
        plt.subplot(1, 3, 2)
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f"{gender} {col} Box Plot")
        plt.xlabel(col)
        
        # Density Plot
        plt.subplot(1, 3, 3)
        sns.kdeplot(df[col], fill=True, color='salmon')
        plt.title(f"{gender} {col} Density Plot")
        plt.xlabel(col)
        
        plt.tight_layout()
        plt.show()
    
    # Graphical analysis for the categorical variable WLoc
    plt.figure(figsize=(8, 6))
    sns.countplot(x='WLoc', data=df, order=df['WLoc'].value_counts().index)
    plt.title(f"{gender} WLoc Count Plot")
    plt.xlabel("WLoc")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# Analyze both Mens and Womens Tournament Results
for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print("\n" + "="*50)
    print(f"ğŸŒŸ Non-Graphical Multivariate Analysis: {gender} Tourney Results ğŸŒŸ")
    print("="*50 + "\n")
    
    # -------------------------------
    # 1. Early vs. Late Wins Analysis
    # Group wins by Season and Winning Team
    team_wins = df.groupby(['Season', 'WTeamID']).agg(
        first_win_day=('DayNum', 'min'),
        last_win_day=('DayNum', 'max'),
        win_count=('DayNum', 'count')
    ).reset_index()
    
    print("Team Wins Summary (first win, last win, win count):")
    print(team_wins.head(), "\n")
    
    # Calculate correlation between first_win_day and last_win_day
    corr = team_wins[['first_win_day', 'last_win_day']].corr().iloc[0,1]
    print(f"Correlation between first win day and last win day: {corr:.2f}\n")
    
    # Summary statistics of aggregated win days
    print("Summary statistics for first win day:")
    print(team_wins['first_win_day'].describe(), "\n")
    print("Summary statistics for last win day:")
    print(team_wins['last_win_day'].describe(), "\n")
    
    # -------------------------------
    # 2. Wins by DayNum and WLoc
    # Pivot table: average winning and losing scores by DayNum and WLoc
    pivot_scores = df.pivot_table(index='DayNum', columns='WLoc', values=['WScore','LScore'], aggfunc='mean')
    print("Pivot Table: Average Scores by DayNum and WLoc:")
    print(pivot_scores.head(), "\n")
    
    # Crosstab: Count of wins by WLoc for each DayNum bin (let's bin DayNum for clarity)
    # For tournaments, DayNum has a limited set of distinct values (rounds)
    day_bins = pd.cut(df['DayNum'], bins=5)
    crosstab = pd.crosstab(df['WLoc'], day_bins)
    print("Crosstab: Wins by WLoc vs. DayNum bins:")
    print(crosstab, "\n")



for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    # Make a copy for safe plotting
    df_plot = df.copy()
    
    print("\n" + "="*50)
    print(f"ğŸŒŸ Graphical Multivariate Analysis: {gender} Tourney Results ğŸŒŸ")
    print("="*50 + "\n")
    
    # --------
    # 1. Scatter Plot: First Win Day vs. Last Win Day
    team_wins = df_plot.groupby(['Season', 'WTeamID']).agg(
        first_win_day=('DayNum', 'min'),
        last_win_day=('DayNum', 'max'),
        win_count=('DayNum', 'count')
    ).reset_index()
    
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(data=team_wins, x='first_win_day', y='last_win_day',
                              size='win_count', sizes=(20, 200),
                              alpha=0.6, palette='viridis')
    plt.title(f"{gender} Team Progression: First vs. Last Win Day")
    plt.xlabel("First Win Day")
    plt.ylabel("Last Win Day")
    plt.legend(title='Win Count', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # --------
    # 2. Heatmap / Clustered Bar Chart: Wins by DayNum and WLoc
    # Pivot table: count of wins by DayNum and WLoc
    pivot_wins = df_plot.pivot_table(index='DayNum', columns='WLoc', aggfunc='size', fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_wins, annot=True, fmt="d", cmap='YlGnBu', linewidths=0.5)
    plt.title(f"{gender} Wins Count by DayNum and WLoc")
    plt.xlabel("WLoc")
    plt.ylabel("DayNum")
    plt.tight_layout()
    plt.show()
    
    # Alternatively, a clustered bar chart by DayNum:
    pivot_wins.plot(kind='bar', figsize=(12, 6))
    plt.title(f"{gender} Wins Count by DayNum and WLoc (Clustered Bar)")
    plt.xlabel("DayNum")
    plt.ylabel("Count of Wins")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # --------
    # 3. Line Plot: Trend of Average Scores by DayNum
    scores_by_day = df_plot.groupby('DayNum').agg({'WScore':'mean', 'LScore':'mean'}).reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(scores_by_day['DayNum'], scores_by_day['WScore'], marker='o', label='Average WScore')
    plt.plot(scores_by_day['DayNum'], scores_by_day['LScore'], marker='s', label='Average LScore')
    plt.title(f"{gender} Average Scores Trend by DayNum")
    plt.xlabel("DayNum")
    plt.ylabel("Average Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --------
    # 4. (Optional for Women's Data) Line Plot: Trend of WLoc Distribution Over DayNum
    # If there is variation in WLoc for women's data
    if gender == 'Womens':
        wloc_by_day = df_plot.groupby(['DayNum', 'WLoc']).size().reset_index(name='count')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=wloc_by_day, x='DayNum', y='count', hue='WLoc', marker='o')
        plt.title(f"{gender} Wins Count Trend by DayNum and WLoc")
        plt.xlabel("DayNum")
        plt.ylabel("Count")
        plt.legend(title="WLoc")
        plt.tight_layout()
        plt.show()


# Load the datasets
mens_df = load_csv(os.path.join(dirname, 'MRegularSeasonDetailedResults.csv'))
womens_df = load_csv(os.path.join(dirname, 'WRegularSeasonDetailedResults.csv'))

# Iterate through both dataframes for EDA
for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*30}\nInvestigating {gender} Regular Season Results\n{'='*30}\n")
    
    # Display basic information
    print("Basic Information:")
    display(df.info())
    
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    display(df.describe())
    
    # Display first few rows of the dataframe
    print("\nFirst 5 Rows of the Data:")
    display(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    display(df.isnull().sum())
    
    # Display unique values in categorical columns
    print(f"\nNumber of Unique Wining Team IDS: {df['WTeamID'].nunique()}")
    print(f"Number of Unique Losing Team IDS: {df['LTeamID'].nunique()}")    
    print(f"Number of Unique Days: {df['DayNum'].nunique()}")


def advanced_non_graphical_analysis(df, label):
    print(f"\n{'='*60}\nğŸŒŸAdvanced Non-Graphical Analysis for {label}ğŸŒŸ\n{'='*60}\n")
    
    # Basic Information
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nDescriptive Statistics (numeric & object):")
    print(df.describe(include='all').T)
    
    # Skewness and Kurtosis for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print("\nSkewness and Kurtosis:")
    for col in numeric_cols:
        skew_val = df[col].skew()
        kurt_val = df[col].kurtosis()
        print(f"  {col}: Skewness = {skew_val:.2f}, Kurtosis = {kurt_val:.2f}")
    
    # Missing Values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Unique Value Counts for Categorical Columns
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print("\nUnique Value Counts for Categorical Columns:")
        for col in object_cols:
            print(f"\n  Column: {col}")
            print(df[col].value_counts())
    
    # Derived Metric: Win Margin (WScore - LScore)
    if 'WScore' in df.columns and 'LScore' in df.columns:
        df['WinMargin'] = df['WScore'] - df['LScore']
        print("\nWin Margin Descriptive Statistics:")
        print(df['WinMargin'].describe())
    
    # Correlation Matrix among key metrics
    key_cols = ['WScore', 'LScore', 'WinMargin', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
                'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    key_cols = [col for col in key_cols if col in df.columns]
    print("\nCorrelation Matrix for Key Metrics:")
    print(df[key_cols].corr())
    
    # Games per Season and Average Win Margin per Season
    if 'Season' in df.columns:
        print("\nGames per Season:")
        season_counts = df['Season'].value_counts().sort_index()
        print(season_counts)
        
        if 'WinMargin' in df.columns:
            print("\nAverage Win Margin per Season:")
            avg_win_margin = df.groupby('Season')['WinMargin'].mean()
            print(avg_win_margin)
    
    # Outlier Detection for Win Margin using the IQR method
    if 'WinMargin' in df.columns:
        Q1 = df['WinMargin'].quantile(0.25)
        Q3 = df['WinMargin'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df['WinMargin'] < lower_bound) | (df['WinMargin'] > upper_bound)]
        print("\nWin Margin Outlier Detection:")
        print(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        print(f"  Number of outlier games based on win margin: {outliers.shape[0]}")


# Apply advanced non-graphical analysis for both datasets
for label, df in zip(["Mens Regular Season Results", "Womens Regular Season Results"], [mens_df, womens_df]):
    advanced_non_graphical_analysis(df, label)


# Create a derived metric: Win Margin (winning score minus losing score)
mens_df['WinMargin'] = mens_df['WScore'] - mens_df['LScore']
womens_df['WinMargin'] = womens_df['WScore'] - womens_df['LScore']

def graphical_analysis(df, label):
    sns.set(style="whitegrid")
    
    # --- Win Margin Distribution ---
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['WinMargin'], kde=True, bins=30, color='skyblue')
    plt.title(f'{label}: Win Margin Distribution')
    plt.xlabel('Win Margin')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df['WinMargin'], color='lightgreen')
    plt.title(f'{label}: Win Margin Boxplot')
    plt.xlabel('Win Margin')
    plt.tight_layout()
    plt.show()
    
    # --- Scatter Plot: Winning Score vs Losing Score ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='WScore', y='LScore', hue='WLoc', data=df, palette='Set1', alpha=0.6)
    plt.title(f'{label}: Winning Score vs Losing Score')
    plt.xlabel('Winning Score')
    plt.ylabel('Losing Score')
    plt.show()
    
    # --- Correlation Heatmap for Key Metrics ---
    key_cols = ['WScore', 'LScore', 'WinMargin', 'WFGM', 'WFGA', 'WFTM', 'WFTA', 'WFGM3', 'WFGA3']
    key_cols = [col for col in key_cols if col in df.columns]
    plt.figure(figsize=(10, 8))
    corr = df[key_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'{label}: Correlation Heatmap of Key Metrics')
    plt.show()
    
    # --- Pairplot for Selected Metrics ---
    selected_cols = ['WScore', 'LScore', 'WinMargin', 'WFGM', 'WFGA']
    selected_cols = [col for col in selected_cols if col in df.columns]
    sns.pairplot(df[selected_cols], diag_kind='kde', height=2.5)
    plt.suptitle(f'{label}: Pairplot of Selected Metrics', y=1.02)
    plt.show()
    
    # --- Count Plot for Categorical Variable: WLoc ---
    if 'WLoc' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='WLoc', data=df, palette='Set2')
        plt.title(f'{label}: Count Plot for Game Location (WLoc)')
        plt.xlabel('WLoc')
        plt.ylabel('Count')
        plt.show()

# Run the graphical analysis for both datasets
for label, df in zip(["Mens Regular Season Results", "Womens Regular Season Results"], [mens_df, womens_df]):
    graphical_analysis(df, label)


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create a derived metric: Win Margin (WScore minus LScore)
for df in [mens_df, womens_df]:
    df['WinMargin'] = df['WScore'] - df['LScore']

def multivariate_analysis_non_graphical(df, label):
    print(f"\n{'='*60}\nğŸŒŸMultivariate Non-Graphical Analysis for {label}ğŸŒŸ\n{'='*40}\n")
    
    # --- Multiple Linear Regression ---
    # We use selected offensive metrics to predict WinMargin.
    predictors = ['WFGM', 'WFGA', 'WFTM', 'WFTA', 'WFGM3', 'WFGA3', 'WAst']
    if not all(col in df.columns for col in predictors + ['WinMargin']):
        print("Required columns not present in the dataframe.")
        return
    
    X = df[predictors]
    y = df['WinMargin']
    
    # Add constant (intercept) to the predictors
    X_const = sm.add_constant(X)
    
    # Fit OLS regression model
    model = sm.OLS(y, X_const).fit()
    print("OLS Regression Model Summary (Predicting WinMargin):")
    print(model.summary())
    
    # --- Variance Inflation Factor (VIF) ---
    print("\nVariance Inflation Factors (VIF):")
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_const.columns
    vif_data['VIF'] = [variance_inflation_factor(X_const.values, i) 
                        for i in range(X_const.shape[1])]
    print(vif_data)
    
    # --- Principal Component Analysis (PCA) ---
    # Standardize the predictors before PCA.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    print("\nPCA Explained Variance Ratios for Predictors:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.4f}")
    
    # Cumulative explained variance
    cum_explained = pca.explained_variance_ratio_.cumsum()
    print("\nCumulative Explained Variance:")
    for i, cum in enumerate(cum_explained):
        print(f"  PC1 to PC{i+1}: {cum:.4f}")

# Run multivariate analysis for both datasets
for label, df in zip(["Mens Regular Season Results", "Womens Regular Season Results"], [mens_df, womens_df]):
    multivariate_analysis_non_graphical(df, label)


import statsmodels.graphics.regressionplots as smg

# Create a derived metric: WinMargin (WScore minus LScore)
for df in [mens_df, womens_df]:
    df['WinMargin'] = df['WScore'] - df['LScore']

def graphical_multivariate_analysis(df, label):
    sns.set(style="whitegrid")
    print(f"\n{'='*60}\nğŸŒŸGraphical Multivariate Analysis for {label}ğŸŒŸ\n{'='*60}\n")
    
    # Select predictors and response variable
    predictors = ['WFGM', 'WFGA', 'WFTM', 'WFTA', 'WFGM3', 'WFGA3', 'WAst']
    X = df[predictors]
    y = df['WinMargin']
    
    # Add a constant to the predictors and fit the OLS model
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    
    # -------------------------------
    # 1. Regression Diagnostic Plots
    # -------------------------------
    fitted_vals = model.fittedvalues
    residuals = model.resid

    # Residual vs. Fitted Plot
    plt.figure(figsize=(8, 6))
    sns.residplot(x=fitted_vals, y=y, lowess=True, 
                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plt.title(f"{label} - Residuals vs. Fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.show()

    # Q-Q Plot
    fig = sm.qqplot(residuals, line='45', fit=True)
    plt.title(f"{label} - Normal Qâ€“Q Plot")
    plt.show()

    # Partial Regression (Added Variable) Plots
    fig = plt.figure(figsize=(12, 8))
    sm.graphics.plot_partregress_grid(model, fig=fig)
    plt.suptitle(f"{label} - Partial Regression Plots", y=0.92)
    plt.show()
    
    # -------------------------------
    # 2. PCA Biplot for Predictors
    # -------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pcs[:, 0], pcs[:, 1], 
                          alpha=0.3, c=fitted_vals, cmap='viridis')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title(f"{label} - PCA Biplot")
    plt.colorbar(scatter, label="Fitted WinMargin")
    
    # Plot loading vectors for each predictor
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, var in enumerate(predictors):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='red', width=0.005)
        plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, var, color='red')
    plt.grid()
    plt.show()

    # -------------------------------
    # 3. Correlation Cluster Map of Predictors
    # -------------------------------
    plt.figure(figsize=(8, 6))
    corr = X.corr()
    # Using clustermap for an interactive view of clustered correlations.
    cluster = sns.clustermap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f"{label} - Correlation Cluster Map of Predictors", pad=100)
    plt.show()

# Run graphical multivariate analysis for both datasets
for label, df in zip(["Mens Regular Season Results", "Womens Regular Season Results"],
                     [mens_df, womens_df]):
    graphical_multivariate_analysis(df, label)


# Load the datasets
mens_df = load_csv(os.path.join(dirname, 'MNCAATourneyDetailedResults.csv'))
womens_df = load_csv(os.path.join(dirname, 'WNCAATourneyDetailedResults.csv'))

# Iterate through both dataframes for EDA
for gender, df in zip(['Mens', 'Womens'], [mens_df, womens_df]):
    print(f"\n{'='*30}\nInvestigating {gender} Tourney Detailed Results\n{'='*30}\n")
    
    # Display basic information
    print("Basic Information:")
    display(df.info())
    
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    display(df.describe())
    
    # Display first few rows of the dataframe
    print("\nFirst 5 Rows of the Data:")
    display(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    display(df.isnull().sum())
    
    # Display unique values in categorical columns
    print(f"\nNumber of Unique Wining Team IDS: {df['WTeamID'].nunique()}")
    print(f"Number of Unique Losing Team IDS: {df['LTeamID'].nunique()}")    
    print(f"Number of Unique Days: {df['DayNum'].nunique()}")


def advanced_non_graphical_analysis_tourney(df, label):
    print(f"\n{'='*40}\nAdvanced Non-Graphical Analysis for {label}\n{'='*40}\n")
    
    # Basic Information
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nDescriptive Statistics (numeric & categorical):")
    print(df.describe(include='all').T)
    
    # Skewness and Kurtosis for Numeric Columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print("\nSkewness and Kurtosis:")
    for col in numeric_cols:
        skew_val = df[col].skew()
        kurt_val = df[col].kurtosis()
        print(f"  {col}: Skewness = {skew_val:.2f}, Kurtosis = {kurt_val:.2f}")
    
    # Missing Values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Unique Value Counts for Categorical Columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\nUnique Value Counts for Categorical Columns:")
        for col in categorical_cols:
            print(f"  {col}:")
            print(df[col].value_counts())
    
    # Derived Metric: WinMargin
    if 'WScore' in df.columns and 'LScore' in df.columns:
        df['WinMargin'] = df['WScore'] - df['LScore']
        print("\nWin Margin Descriptive Statistics:")
        print(df['WinMargin'].describe())
    
    # Correlation Matrix for Key Metrics
    key_cols = ['WScore', 'LScore', 'WinMargin', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
                'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    key_cols = [col for col in key_cols if col in df.columns]
    print("\nCorrelation Matrix for Key Metrics:")
    print(df[key_cols].corr())
    
    # Seasonal Analysis: Games per Season and Average Win Margin per Season
    if 'Season' in df.columns:
        season_counts = df['Season'].value_counts().sort_index()
        print("\nGames per Season:")
        print(season_counts)
        
        if 'WinMargin' in df.columns:
            avg_win_margin = df.groupby('Season')['WinMargin'].mean()
            print("\nAverage Win Margin per Season:")
            print(avg_win_margin)
    
    # Outlier Detection for WinMargin using the IQR method
    if 'WinMargin' in df.columns:
        Q1 = df['WinMargin'].quantile(0.25)
        Q3 = df['WinMargin'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df['WinMargin'] < lower_bound) | (df['WinMargin'] > upper_bound)]
        print("\nWin Margin Outlier Detection:")
        print(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        print(f"  Number of outlier games based on win margin: {outliers.shape[0]}")

# Run Advanced Non-Graphical Analysis for both tournament datasets
for label, df in zip(["Mens Tourney Detailed Results", "Womens Tourney Detailed Results"],
                     [mens_df, womens_df]):
    advanced_non_graphical_analysis_tourney(df, label)


def univariate_graphical_analysis(df, label):
    sns.set(style="whitegrid")
    print(f"\n{'='*30}\n{label} Univariate Graphical Analysis\n{'='*30}\n")
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Plot distributions for numeric columns
    for col in numeric_cols:
        plt.figure(figsize=(12, 4))
        
        # Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f"{label} - Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f"{label} - Boxplot of {col}")
        plt.xlabel(col)
        
        plt.tight_layout()
        plt.show()
    
    # Plot count plots for categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=df[col], palette='pastel')
        plt.title(f"{label} - Count Plot of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Run univariate graphical analysis for both datasets
for label, df in zip(["Mens Tourney Detailed Results", "Womens Tourney Detailed Results"],
                     [mens_df, womens_df]):
    univariate_graphical_analysis(df, label)



from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a derived metric: WinMargin = WScore - LScore
for df in [mens_df, womens_df]:
    df['WinMargin'] = df['WScore'] - df['LScore']

def multivariate_analysis_tourney(df, label):
    print(f"\n{'='*40}\nMultivariate Non-Graphical Analysis for {label}\n{'='*40}\n")
    
    # Select predictors for the model
    predictors = ['WFGM', 'WFGA', 'WFTM', 'WFTA', 'WFGM3', 'WFGA3', 'WAst']
    if not all(col in df.columns for col in predictors + ['WinMargin']):
        print("Some required columns are missing in the dataset.")
        return
    
    X = df[predictors]
    y = df['WinMargin']
    
    # Add constant term for the intercept
    X_const = sm.add_constant(X)
    
    # Fit OLS regression model
    model = sm.OLS(y, X_const).fit()
    print("OLS Regression Model Summary (Predicting WinMargin):")
    print(model.summary())
    
    # Compute Variance Inflation Factors (VIF)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_const.columns
    vif_data['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data)
    
    # Principal Component Analysis (PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    
    print("\nPCA Explained Variance Ratios for Predictors:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.4f}")
    
    # Cumulative explained variance
    cum_explained = pca.explained_variance_ratio_.cumsum()
    print("\nCumulative Explained Variance:")
    for i, cum in enumerate(cum_explained):
        print(f"  PC1 to PC{i+1}: {cum:.4f}")

# Run the multivariate non-graphical analysis for both datasets
for label, df in zip(["Mens Tourney Detailed Results", "Womens Tourney Detailed Results"],
                     [mens_df, womens_df]):
    multivariate_analysis_tourney(df, label)


# Create a derived metric: WinMargin = WScore - LScore
for df in [mens_df, womens_df]:
    df['WinMargin'] = df['WScore'] - df['LScore']

def graphical_multivariate_analysis(df, label):
    sns.set(style="whitegrid")
    print(f"\nGraphical Multivariate Analysis for {label}\n")
    
    # Select predictors and the response variable
    predictors = ['WFGM', 'WFGA', 'WFTM', 'WFTA', 'WFGM3', 'WFGA3', 'WAst']
    X = df[predictors]
    y = df['WinMargin']
    
    # Fit an OLS model
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    
    # --- 1. Regression Diagnostic Plots ---
    fitted_vals = model.fittedvalues
    residuals = model.resid

    # Residual vs. Fitted Plot
    plt.figure(figsize=(8,6))
    sns.residplot(x=fitted_vals, y=y, lowess=True,
                  line_kws={'color':'red', 'lw':1, 'alpha':0.8})
    plt.title(f"{label} - Residuals vs. Fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.show()

    # Normal Qâ€“Q Plot
    fig = sm.qqplot(residuals, line='45', fit=True)
    plt.title(f"{label} - Normal Qâ€“Q Plot")
    plt.show()

    # Partial Regression (Added Variable) Plots
    fig = plt.figure(figsize=(12,8))
    sm.graphics.plot_partregress_grid(model, fig=fig)
    plt.suptitle(f"{label} - Partial Regression Plots", y=0.92)
    plt.show()

    # --- 2. PCA Biplot for Predictors ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(pcs[:, 0], pcs[:, 1],
                          alpha=0.3, c=fitted_vals, cmap='viridis')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title(f"{label} - PCA Biplot")
    plt.colorbar(scatter, label="Fitted WinMargin")
    
    # Plot loading vectors for each predictor
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, var in enumerate(predictors):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  color='red', width=0.005)
        plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, var, color='red')
    plt.grid()
    plt.show()
    
    # --- 3. Correlation Cluster Map ---
    plt.figure(figsize=(8,6))
    corr = X.corr()
    cluster = sns.clustermap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f"{label} - Correlation Cluster Map of Predictors", pad=100)
    plt.show()

# Run graphical multivariate analysis for both datasets
for label, df in zip(["Mens Tourney Detailed Results", "Womens Tourney Detailed Results"],
                     [mens_df, womens_df]):
    graphical_multivariate_analysis(df, label)

