import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Plotting Style ---
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14

# --- File Path ---
BASE_PATH = "/kaggle/input/meta-kaggle"

# --- Data Loading Function ---
def load_data(file_name, date_cols=[]):
    path = os.path.join(BASE_PATH, file_name)
    try:
        df = pd.read_csv(path, parse_dates=date_cols)
        print(f"Successfully loaded {file_name} ({df.shape[0]:,} rows, {df.shape[1]} cols)")
        return df
    except FileNotFoundError:
        print(f"Error: {file_name} not found at {path}")
        return None

# --- Load All Datasets ---
competitions_df = load_data("Competitions.csv", date_cols=['EnabledDate'])
users_df = load_data("Users.csv", date_cols=['RegisterDate'])
kernels_df = load_data("Kernels.csv", date_cols=['CreationDate'])



if competitions_df is not None:
    competitions_df['LaunchYear'] = competitions_df['EnabledDate'].dt.year
    
    # --- Visualization 1: Competition Types Breakdown ---
    plt.figure(figsize=(16, 8))
    # Group by year and type, then unstack for plotting
    comp_types = competitions_df.groupby(['LaunchYear', 'HostSegmentTitle']).size().unstack().fillna(0)
    
    # Use a clear color palette
    comp_types.plot(kind='bar', stacked=True, figsize=(16, 8), colormap='viridis')
    
    plt.title('Kaggle Competition Growth by Type', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Competitions Launched')
    plt.legend(title='Competition Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if users_df is not None:
    users_df['JoinYear'] = users_df['RegisterDate'].dt.year

    # --- Visualization 2: Top 15 Countries by User Count ---
    plt.figure(figsize=(16, 8))
    top_countries = users_df['Country'].value_counts().nlargest(15)
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='plasma')
    
    plt.title('Top 15 Countries of Kaggle Users', fontweight='bold')
    plt.xlabel('Number of Registered Users')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.show()


# Prepare data for the final combined plot
competitions_growth = competitions_df.groupby('LaunchYear').size().reset_index(name='Competitions') if competitions_df is not None else None
users_growth = users_df.groupby('JoinYear').size().reset_index(name='NewUsers') if users_df is not None else None
kernels_growth = kernels_df.groupby(kernels_df['CreationDate'].dt.year).size().reset_index(name='Kernels') if kernels_df is not None else None
if kernels_growth is not None:
    kernels_growth.rename(columns={'CreationDate': 'Year'}, inplace=True)

# Merge the datasets for plotting
if all(df is not None for df in [competitions_growth, users_growth, kernels_growth]):
    growth_df = pd.merge(competitions_growth, users_growth, left_on='LaunchYear', right_on='JoinYear', how='outer')
    growth_df = pd.merge(growth_df, kernels_growth, left_on='LaunchYear', right_on='Year', how='outer')
    growth_df['Year'] = growth_df['LaunchYear'].fillna(growth_df['JoinYear']).fillna(growth_df['Year'])
    growth_df = growth_df[['Year', 'Competitions', 'NewUsers', 'Kernels']].fillna(0)
    growth_df = growth_df[growth_df['Year'] >= 2010].sort_values('Year').set_index('Year')

    # --- Visualization 3: The Combined Ecosystem Growth ---
    # Normalize the data to plot on the same scale (0 to 1)
    normalized_df = (growth_df - growth_df.min()) / (growth_df.max() - growth_df.min())

    normalized_df.plot(figsize=(16, 9), lw=3, marker='o', markersize=8)

    plt.title('The Kaggle Flywheel: Growth of Users, Competitions, and Kernels', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Normalized Growth (0 to 1)')
    plt.legend(title='Ecosystem Pillar')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


