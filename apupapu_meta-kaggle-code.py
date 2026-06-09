# Part 1: Introduction & The Foundation of Meta Kaggle
# Goal: Set the stage, introduce the Meta Kaggle and Meta Kaggle Code datasets,
# and provide a high-level overview of Kaggle's growth. This section aims to
# hook the judges by demonstrating the vastness and richness of the data.

# --- Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import kagglehub # Import kagglehub for dataset downloading

# --- Configuration & Styling ---
# Set plotting style for Matplotlib/Seaborn
plt.style.use('ggplot')
sns.set_palette('viridis')

# Custom colors for visualizations (feel free to adjust)
kaggle_blue = '#20BEFF'
kaggle_dark = '#222222'
kaggle_light_grey = '#CCCCCC'
colors_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Vivid

# --- Data Loading ---
print("Downloading Meta Kaggle datasets using kagglehub...")

# Use kagglehub to download the datasets
try:
    MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
    MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")

    print(f"Path to Meta-Kaggle dataset files: {MK_PATH}")
    print(f"Path to Meta-Kaggle-Code dataset files: {MKC_PATH}")

    print("\nLoading Meta Kaggle datasets from downloaded paths...")

    # Attempt to load key datasets using the downloaded paths
    # Users data - contains information about Kaggle users
    users_df = pd.read_csv(f"{MK_PATH}/Users.csv", low_memory=False)
    # Competitions data - contains information about Kaggle competitions
    competitions_df = pd.read_csv(f"{MK_PATH}/Competitions.csv", low_memory=False)
    # Kernels data (Notebooks) - contains metadata about public notebooks
    kernels_df = pd.read_csv(f"{MK_PATH}/Kernels.csv", low_memory=False)
    # Datasets data - contains information about public datasets
    datasets_df = pd.read_csv(f"{MK_PATH}/Datasets.csv", low_memory=False)
    # Discussions data (Topics) - for forum activity
    topics_df = pd.read_csv(f"{MK_PATH}/ForumTopics.csv", low_memory=False)
    # Submissions data - required for participation trends in Part 1.8
    submissions_df = pd.read_csv(f"{MK_PATH}/Submissions.csv", low_memory=False)
    # Teams data - useful for approximating participation if Submissions are problematic
    teams_df = pd.read_csv(f"{MK_PATH}/Teams.csv", low_memory=False)

    print("Datasets loaded successfully!")

except Exception as e: # Catch a broader exception for download or read errors
    print(f"Error during data loading: {e}. Please ensure you have Kaggle API configured or check permissions.")
    # Create empty DataFrames to avoid errors in subsequent code
    users_df = pd.DataFrame()
    competitions_df = pd.DataFrame()
    kernels_df = pd.DataFrame()
    datasets_df = pd.DataFrame()
    topics_df = pd.DataFrame()
    submissions_df = pd.DataFrame()
    teams_df = pd.DataFrame()


# --- Data Preprocessing for Part 1 ---
# Convert relevant date columns to datetime objects
# Assuming 'RegisterDate' for Users, 'EnabledDate' for Competitions,
# 'CreationDate' for Kernels and Datasets, 'CreationDate' for ForumTopics,
# 'SubmissionDate' for Submissions.

# Safely apply conversion only if DataFrame is not empty and column exists
for df, date_col in [(users_df, 'RegisterDate'),
                      (competitions_df, 'EnabledDate'),
                      (kernels_df, 'CreationDate'),
                      (datasets_df, 'CreationDate'),
                      (topics_df, 'CreationDate'),
                      (submissions_df, 'SubmissionDate')]: # Add submissions_df to the loop
    if not df.empty and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        # Drop rows where date conversion failed (if any)
        df.dropna(subset=[date_col], inplace=True)



# --- PART 1: Results & Visualizations ---

### **Result 1.1: Overall Scale of Kaggle Operations**
print("\n--- 1.1 Overall Scale of Kaggle Operations ---")
total_users = users_df['Id'].nunique() if not users_df.empty else 0
total_competitions = competitions_df['Id'].nunique() if not competitions_df.empty else 0
total_notebooks = kernels_df['Id'].nunique() if not kernels_df.empty else 0
total_datasets = datasets_df['Id'].nunique() if not datasets_df.empty else 0
total_discussions = topics_df['Id'].nunique() if not topics_df.empty else 0

print(f"Total Registered Users: {total_users:,}")
print(f"Total Competitions Launched: {total_competitions:,}")
print(f"Total Public Notebooks: {total_notebooks:,}")
print(f"Total Public Datasets: {total_datasets:,}")
print(f"Total Discussion Topics: {total_discussions:,}")

# Visualization 1.1: Key Metrics Overview
metrics_data = {
    'Metric': ['Users', 'Competitions', 'Notebooks', 'Datasets', 'Discussions'],
    'Count': [total_users, total_competitions, total_notebooks, total_datasets, total_discussions]
}
metrics_df = pd.DataFrame(metrics_data).sort_values(by='Count', ascending=False)

fig = px.bar(metrics_df, x='Metric', y='Count',
             title='Overall Scale of Kaggle Operations (Cumulative)',
             labels={'Count': 'Count', 'Metric': 'Kaggle Entity'},
             color='Metric', color_discrete_sequence=colors_palette,
             text='Count')
fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
fig.update_layout(showlegend=False, title_x=0.5, font_size=12)
fig.show()


### **Result 1.2 & Visualization 1.2: Total Registered Users Over Time**
print("\n--- 1.2 Total Registered Users Over Time ---")
if not users_df.empty and 'RegisterDate' in users_df.columns:
    users_df['RegisterYearMonth'] = users_df['RegisterDate'].dt.to_period('M')
    users_over_time = users_df.groupby('RegisterYearMonth').size().sort_index().cumsum().reset_index(name='CumulativeUsers')
    users_over_time['RegisterYearMonth'] = users_over_time['RegisterYearMonth'].astype(str)

    fig = px.line(users_over_time, x='RegisterYearMonth', y='CumulativeUsers',
                  title='Cumulative Registered Users Over Time',
                  labels={'RegisterYearMonth': 'Year-Month', 'CumulativeUsers': 'Cumulative Users'},
                  markers=False)
    fig.update_layout(title_x=0.5, xaxis_rangeslider_visible=True, font_size=12)
    fig.show()
    print(f"Average monthly user sign-ups (last 12 months, if data available): {users_over_time['CumulativeUsers'].diff().tail(12).mean():,.0f}")
else:
    print("Users data or 'RegisterDate' column not available for this analysis.")



### **Result 1.3 & Visualization 1.3: Number of Competitions Launched Annually**
print("\n--- 1.3 Number of Competitions Launched Annually ---")
if not competitions_df.empty and 'EnabledDate' in competitions_df.columns:
    competitions_df['LaunchYear'] = competitions_df['EnabledDate'].dt.year
    competitions_per_year = competitions_df.groupby('LaunchYear').size().reset_index(name='NumCompetitions')

    fig = px.bar(competitions_per_year, x='LaunchYear', y='NumCompetitions',
                 title='Number of Competitions Launched Per Year',
                 labels={'LaunchYear': 'Year', 'NumCompetitions': 'Number of Competitions'},
                 color='NumCompetitions', color_continuous_scale=px.colors.sequential.Plasma,
                 text='NumCompetitions')
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(title_x=0.5, font_size=12)
    fig.show()
    if not competitions_per_year.empty:
        print(f"Year with most competition launches: {competitions_per_year.loc[competitions_per_year['NumCompetitions'].idxmax(), 'LaunchYear']} ({competitions_per_year['NumCompetitions'].max():,.0f} competitions)")
    else:
        print("No competition data available to determine the year with most launches.")
else:
    print("Competitions data or 'EnabledDate' column not available for this analysis.")




### **Result 1.4 & Visualization 1.4: Growth of Public Notebooks/Kernels Over Time**
print("\n--- 1.4 Growth of Public Notebooks/Kernels Over Time ---")
if not kernels_df.empty and 'CreationDate' in kernels_df.columns:
    kernels_df['CreationYearMonth'] = kernels_df['CreationDate'].dt.to_period('M')
    kernels_over_time = kernels_df.groupby('CreationYearMonth').size().sort_index().cumsum().reset_index(name='CumulativeKernels')
    kernels_over_time['CreationYearMonth'] = kernels_over_time['CreationYearMonth'].astype(str)

    fig = px.area(kernels_over_time, x='CreationYearMonth', y='CumulativeKernels',
                  title='Cumulative Public Notebooks (Kernels) Over Time',
                  labels={'CreationYearMonth': 'Year-Month', 'CumulativeKernels': 'Cumulative Notebooks'},
                  color_discrete_sequence=[kaggle_blue])
    fig.update_layout(title_x=0.5, xaxis_rangeslider_visible=True, font_size=12)
    fig.show()
    if not kernels_over_time.empty:
        print(f"Total public notebooks created: {kernels_over_time['CumulativeKernels'].iloc[-1]:,.0f}")
    else:
        print("No kernel data available to count total public notebooks.")
else:
    print("Kernels data or 'CreationDate' column not available for this analysis.")



### **Result 1.5 & Visualization 1.5: Growth of Datasets Hosted on Kaggle Over Time**
print("\n--- 1.5 Growth of Datasets Hosted on Kaggle Over Time ---")
if not datasets_df.empty and 'CreationDate' in datasets_df.columns:
    datasets_df['CreationYearMonth'] = datasets_df['CreationDate'].dt.to_period('M')
    datasets_over_time = datasets_df.groupby('CreationYearMonth').size().sort_index().cumsum().reset_index(name='CumulativeDatasets')
    datasets_over_time['CreationYearMonth'] = datasets_over_time['CreationYearMonth'].astype(str)

    fig = px.line(datasets_over_time, x='CreationYearMonth', y='CumulativeDatasets',
                  title='Cumulative Public Datasets Hosted on Kaggle Over Time',
                  labels={'CreationYearMonth': 'Year-Month', 'CumulativeDatasets': 'Cumulative Datasets'},
                  markers=False, color_discrete_sequence=[colors_palette[1]])
    fig.update_layout(title_x=0.5, xaxis_rangeslider_visible=True, font_size=12)
    fig.show()
    if not datasets_over_time.empty:
        print(f"Total public datasets hosted: {datasets_over_time['CumulativeDatasets'].iloc[-1]:,.0f}")
    else:
        print("No dataset data available to count total public datasets.")
else:
    print("Datasets data or 'CreationDate' column not available for this analysis.")



import pandas as pd
import plotly.express as px
import numpy as np

# --- Sample Data Generation (Replace with your actual data loading) ---
# Create a sample DataFrame similar to what 'users_df' might look like
data = {
    'RegisterDate': pd.to_datetime(['2022-01-15', '2022-03-20', '2022-06-10', '2023-01-05', 
                                    '2023-04-22', '2023-07-11', '2024-02-14', '2024-05-01',
                                    '2024-09-30', '2022-02-01', '2023-03-15', '2024-06-20']),
    'PerformanceTier': np.random.randint(0, 5, size=12) # Tiers 0-4
}
users_df = pd.DataFrame(data)

# Ensure 'RegisterDate' is datetime type
users_df['RegisterDate'] = pd.to_datetime(users_df['RegisterDate'])

# --- 1.6 Distribution of User Tiers Over Time ---
print("\n--- 1.6 Distribution of User Tiers Over Time ---")

if not users_df.empty and 'PerformanceTier' in users_df.columns and 'RegisterDate' in users_df.columns:
    # Map tier numbers to descriptive names
    tier_mapping = {
        0: 'Novice',
        1: 'Contributor',
        2: 'Expert',
        3: 'Master',
        4: 'Grandmaster'
    }
    users_df['PerformanceTierName'] = users_df['PerformanceTier'].map(tier_mapping)

    # Aggregate by year and tier
    users_by_year_tier = users_df.groupby([users_df['RegisterDate'].dt.year.rename('Year'), 'PerformanceTierName']).size().unstack(fill_value=0)
    
    # Ensure all tier names are present in columns, fill missing with 0
    for tier_name in tier_mapping.values():
        if tier_name not in users_by_year_tier.columns:
            users_by_year_tier[tier_name] = 0
            
    # Ensure consistent order and drop NA years
    users_by_year_tier = users_by_year_tier[tier_mapping.values()].loc[users_by_year_tier.index.dropna()]

    # Create the area chart
    # Using a built-in discrete color sequence (px.colors.qualitative.Plotly or px.colors.qualitative.Dark24)
    # You can also define a custom list of colors or use color_discrete_map for specific color-tier associations.
    fig = px.area(users_by_year_tier,
                  title='Evolution of User Tier Distribution Over Time',
                  labels={'value': 'Number of Users', 'variable': 'User Tier', 'Year': 'Year'},
                  color_discrete_sequence=px.colors.qualitative.Plotly) # Corrected line
    fig.update_layout(title_x=0.5, font_size=12)
    fig.show()

    # Percentage distribution for the latest full year
    latest_year = users_by_year_tier.index.max()
    if latest_year and not users_by_year_tier.empty:
        latest_year_data = users_by_year_tier.loc[latest_year]
        tier_percentage = latest_year_data / latest_year_data.sum() * 100
        print(f"\nUser Tier Distribution for {latest_year}:")
        print(tier_percentage.sort_values(ascending=False).round(2).to_string())
    else:
        print("Not enough data to show latest year tier distribution.")
else:
    print("Users data, 'PerformanceTier', or 'RegisterDate' column not available for this analysis.")


### **Result 1.11 & Visualization 1.11: Average Daily/Weekly Activity Metrics**
print("\n--- 1.11 Average Daily/Weekly Activity Metrics ---")
# Combine creation dates from multiple sources
all_dates = pd.Series(dtype='datetime64[ns]')
if not users_df.empty and 'RegisterDate' in users_df.columns: all_dates = pd.concat([all_dates, users_df['RegisterDate']])
if not competitions_df.empty and 'EnabledDate' in competitions_df.columns: all_dates = pd.concat([all_dates, competitions_df['EnabledDate']])
if not kernels_df.empty and 'CreationDate' in kernels_df.columns: all_dates = pd.concat([all_dates, kernels_df['CreationDate']])
if not datasets_df.empty and 'CreationDate' in datasets_df.columns: all_dates = pd.concat([all_dates, datasets_df['CreationDate']])
if not topics_df.empty and 'CreationDate' in topics_df.columns: all_dates = pd.concat([all_dates, topics_df['CreationDate']])

if not all_dates.empty:
    activity_daily = all_dates.value_counts().sort_index()
    activity_weekly = activity_daily.resample('W').sum().fillna(0)
    
    # Calculate rolling average for smoother trend
    activity_weekly_rolling_avg = activity_weekly.rolling(window=4, min_periods=1).mean() # 4-week rolling average

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=activity_weekly.index, y=activity_weekly.values,
                             mode='lines', name='Weekly Activity',
                             line=dict(color=kaggle_light_grey, width=1.5)))
    fig.add_trace(go.Scatter(x=activity_weekly_rolling_avg.index, y=activity_weekly_rolling_avg.values,
                             mode='lines', name='4-Week Rolling Average',
                             line=dict(color=kaggle_blue, width=3)))
    
    fig.update_layout(title='Average Weekly Activity on Kaggle (New Users, Competitions, Notebooks, Datasets, Discussions)',
                      xaxis_title='Date',
                      yaxis_title='Count of New Entries',
                      title_x=0.5, font_size=12,
                      xaxis_rangeslider_visible=True)
    fig.show()

    print(f"Average new entries per week (overall): {activity_weekly.mean():,.2f}")
    if not activity_weekly.empty:
        print(f"Peak weekly activity occurred around: {activity_weekly.idxmax().strftime('%Y-%m-%d')} with {activity_weekly.max():,.0f} entries.")
    else:
        print("No weekly activity data available.")
else:
    print("No date columns found across primary dataframes for overall activity analysis.")



### **Result 1.12 & Visualization 1.12: Initial Snapshot of Key Features**
print("\n--- 1.12 Initial Snapshot of Key Features (Min/Max Dates, Data Completeness) ---")

snapshot_data = []
# Assuming date columns determined previously
dataframes_to_check = {
    "Users": users_df,
    "Competitions": competitions_df,
    "Notebooks": kernels_df,
    "Datasets": datasets_df,
    "Discussions": topics_df,
    "Submissions": submissions_df, # Added submissions_df
    "Teams": teams_df # Added teams_df
}
date_cols_mapping = {
    "Users": "RegisterDate",
    "Competitions": "EnabledDate",
    "Notebooks": "CreationDate",
    "Datasets": "CreationDate",
    "Discussions": "CreationDate",
    "Submissions": "SubmissionDate",
    # Teams doesn't have a direct creation date for the team itself relevant here,
    # but could be linked via competition EnabledDate if needed, skipping for this direct check.
}

for name, df in dataframes_to_check.items():
    if not df.empty:
        total_rows = len(df)
        id_col = 'Id' if 'Id' in df.columns else df.columns[0] # Try 'Id' or first column
        unique_ids = df[id_col].nunique() if id_col in df.columns else 'N/A'

        date_col = date_cols_mapping.get(name)
        min_date = df[date_col].min().strftime('%Y-%m-%d') if date_col and date_col in df.columns and not df[date_col].isnull().all() else 'N/A'
        max_date = df[date_col].max().strftime('%Y-%m-%d') if date_col and date_col in df.columns and not df[date_col].isnull().all() else 'N/A'

        num_cols = len(df.columns)
        num_null_values = df.isnull().sum().sum() # Total missing values in the entire DataFrame

        snapshot_data.append({
            'Entity': name,
            'Total Rows': total_rows,
            'Unique IDs': unique_ids,
            'First Record Date': min_date,
            'Last Record Date': max_date,
            'Number of Columns': num_cols,
            'Total Missing Values': num_null_values
        })
    else:
        snapshot_data.append({
            'Entity': name,
            'Total Rows': 0, 'Unique IDs': 0,
            'First Record Date': 'N/A', 'Last Record Date': 'N/A',
            'Number of Columns': 0, 'Total Missing Values': 0
        })

snapshot_df = pd.DataFrame(snapshot_data)
print(snapshot_df.to_string(index=False))

# Visualization 1.12: Completeness/Metadata Overview (Heatmap of nulls if many columns, else simple bar)
null_counts = []
for name, df in dataframes_to_check.items():
    if not df.empty:
        for col in df.columns:
            null_percent = df[col].isnull().sum() / len(df) * 100
            if null_percent > 0: # Only include columns with missing values
                null_counts.append({'DataFrame': name, 'Column': col, 'Null_Percentage': null_percent})

if null_counts:
    null_df = pd.DataFrame(null_counts)
    fig = px.bar(null_df, x='Column', y='Null_Percentage', color='DataFrame',
                 title='Percentage of Missing Values in Key Columns Across DataFrames',
                 labels={'Null_Percentage': 'Missing (%)', 'Column': 'Column Name'},
                 facet_col='DataFrame', facet_col_wrap=3,
                 color_discrete_sequence=colors_palette)
    fig.update_layout(title_x=0.5, font_size=10)
    fig.show()
else:
    print("No significant missing values detected in loaded dataframes for visualization, or dataframes are empty.")






import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Set aesthetic styles for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

# --- 1. Environment Setup & Data Loading ---
print("--- Kaggle Data Path Setup ---")
MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code") # Although not directly used in Part 1, good to have it setup

print(f"Path to Meta-Kaggle dataset files: {MK_PATH}")
print(f"Path to Meta-Kaggle-Code dataset files: {MKC_PATH}")

print("\n--- Loading Core DataFrames ---")
try:
    df_users = pd.read_csv(f"{MK_PATH}/Users.csv")
    print(f"Loaded Users.csv: {df_users.shape[0]} rows, {df_users.shape[1]} columns")

    competitions_df = pd.read_csv(f"{MK_PATH}/Competitions.csv")
    print(f"Loaded Competitions.csv: {competitions_df.shape[0]} rows, {competitions_df.shape[1]} columns")

except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure the dataset is properly downloaded and path is correct.")
    # Exit or handle gracefully if files aren't found
    exit()

# --- 2. Initial Data Inspection (competitions_df) ---
print("\n--- Initial Inspection: competitions_df ---")
print("Shape:", competitions_df.shape)
print("\nFirst 5 rows:")
print(competitions_df.head())
print("\nColumn Information (Data Types & Non-Null Counts):")
competitions_df.info()
print("\nMissing Values (Top 10):")
print(competitions_df.isnull().sum().sort_values(ascending=False).head(10))

# --- Initial Data Inspection (df_users) ---
print("\n--- Initial Inspection: df_users ---")
print("Shape:", df_users.shape)
print("\nFirst 5 rows:")
print(df_users.head())
print("\nColumn Information (Data Types & Non-Null Counts):")
df_users.info()
print("\nMissing Values (Top 10):")
print(df_users.isnull().sum().sort_values(ascending=False).head(10))

# --- 3. Date Feature Engineering ---
print("\n--- Date Feature Engineering ---")

# Convert 'EnabledDate' and 'DeadlineDate' in competitions_df to datetime
# 'EnabledDate' seems like the best candidate for competition start date
competitions_df['EnabledDate'] = pd.to_datetime(competitions_df['EnabledDate'], errors='coerce')
competitions_df['DeadlineDate'] = pd.to_datetime(competitions_df['DeadlineDate'], errors='coerce')

# Drop rows where EnabledDate is null after conversion, as it's crucial for time-series
initial_competitions_count = competitions_df.shape[0]
competitions_df.dropna(subset=['EnabledDate'], inplace=True)
print(f"Dropped {initial_competitions_count - competitions_df.shape[0]} rows from competitions_df due to invalid EnabledDate.")

# Extract year and month for time series analysis from EnabledDate
competitions_df['Competition_Year'] = competitions_df['EnabledDate'].dt.year
competitions_df['Competition_Month'] = competitions_df['EnabledDate'].dt.to_period('M') # For monthly aggregation

# Convert 'RegisterDate' in df_users to datetime
if 'RegisterDate' in df_users.columns:
    df_users['RegisterDate'] = pd.to_datetime(df_users['RegisterDate'], errors='coerce')
    # Drop rows with invalid RegisterDate, crucial for user trends
    initial_users_count = df_users.shape[0]
    df_users.dropna(subset=['RegisterDate'], inplace=True)
    print(f"Dropped {initial_users_count - df_users.shape[0]} rows from df_users due to invalid RegisterDate.")

    df_users['User_Register_Year'] = df_users['RegisterDate'].dt.year
    df_users['User_Register_Month'] = df_users['RegisterDate'].dt.to_period('M')
else:
    print("Warning: 'RegisterDate' column not found in df_users. Skipping user date feature engineering.")

print("\nDate columns processed. Sample 'EnabledDate' and 'RegisterDate' years:")
print("Competitions EnabledYear sample:", competitions_df['Competition_Year'].value_counts().sort_index().head(5).index.tolist())
if 'User_Register_Year' in df_users.columns:
    print("Users RegisterYear sample:", df_users['User_Register_Year'].value_counts().sort_index().head(5).index.tolist())



import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Set aesthetic styles for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

# --- 1. Environment Setup & Data Loading ---
print("--- Kaggle Data Path Setup ---")
MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code") # Although not directly used in Part 1, good to have it setup

print(f"Path to Meta-Kaggle dataset files: {MK_PATH}")
print(f"Path to Meta-Kaggle-Code dataset files: {MKC_PATH}")

print("\n--- Loading Core DataFrames ---")
try:
    df_users = pd.read_csv(f"{MK_PATH}/Users.csv")
    print(f"Loaded Users.csv: {df_users.shape[0]} rows, {df_users.shape[1]} columns")

    competitions_df = pd.read_csv(f"{MK_PATH}/Competitions.csv")
    print(f"Loaded Competitions.csv: {competitions_df.shape[0]} rows, {competitions_df.shape[1]} columns")

except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure the dataset is properly downloaded and path is correct.")
    # Exit or handle gracefully if files aren't found
    exit()



# --- 2. Initial Data Inspection (competitions_df) ---
print("\n--- Initial Inspection: competitions_df ---")
print("Shape:", competitions_df.shape)
print("\nFirst 5 rows:")
print(competitions_df.head())
print("\nColumn Information (Data Types & Non-Null Counts):")
competitions_df.info()
print("\nMissing Values (Top 10):")
print(competitions_df.isnull().sum().sort_values(ascending=False).head(10))



# --- Initial Data Inspection (df_users) ---
print("\n--- Initial Inspection: df_users ---")
print("Shape:", df_users.shape)
print("\nFirst 5 rows:")
print(df_users.head())
print("\nColumn Information (Data Types & Non-Null Counts):")
df_users.info()
print("\nMissing Values (Top 10):")
print(df_users.isnull().sum().sort_values(ascending=False).head(10))


# --- 3. Date Feature Engineering ---
print("\n--- Date Feature Engineering ---")

# Convert 'EnabledDate' and 'DeadlineDate' in competitions_df to datetime
# 'EnabledDate' seems like the best candidate for competition start date
competitions_df['EnabledDate'] = pd.to_datetime(competitions_df['EnabledDate'], errors='coerce')
competitions_df['DeadlineDate'] = pd.to_datetime(competitions_df['DeadlineDate'], errors='coerce')

# Drop rows where EnabledDate is null after conversion, as it's crucial for time-series
initial_competitions_count = competitions_df.shape[0]
competitions_df.dropna(subset=['EnabledDate'], inplace=True)
print(f"Dropped {initial_competitions_count - competitions_df.shape[0]} rows from competitions_df due to invalid EnabledDate.")

# Extract year and month for time series analysis from EnabledDate
competitions_df['Competition_Year'] = competitions_df['EnabledDate'].dt.year
competitions_df['Competition_Month'] = competitions_df['EnabledDate'].dt.to_period('M') # For monthly aggregation

# Convert 'RegisterDate' in df_users to datetime
if 'RegisterDate' in df_users.columns:
    df_users['RegisterDate'] = pd.to_datetime(df_users['RegisterDate'], errors='coerce')
    # Drop rows with invalid RegisterDate, crucial for user trends
    initial_users_count = df_users.shape[0]
    df_users.dropna(subset=['RegisterDate'], inplace=True)
    print(f"Dropped {initial_users_count - df_users.shape[0]} rows from df_users due to invalid RegisterDate.")

    df_users['User_Register_Year'] = df_users['RegisterDate'].dt.year
    df_users['User_Register_Month'] = df_users['RegisterDate'].dt.to_period('M')
else:
    print("Warning: 'RegisterDate' column not found in df_users. Skipping user date feature engineering.")

print("\nDate columns processed. Sample 'EnabledDate' and 'RegisterDate' years:")
print("Competitions EnabledYear sample:", competitions_df['Competition_Year'].value_counts().sort_index().head(5).index.tolist())
if 'User_Register_Year' in df_users.columns:
    print("Users RegisterYear sample:", df_users['User_Register_Year'].value_counts().sort_index().head(5).index.tolist())



# --- 4. High-Level Competition Activity Trends ---
print("\n--- Competition Activity Trends ---")

# Result 1: Total Number of Competitions
total_competitions = competitions_df.shape[0]
print(f"1. Total number of competitions recorded: {total_competitions}")

# Result 2: Time range of competitions
min_comp_date = competitions_df['EnabledDate'].min()
max_comp_date = competitions_df['EnabledDate'].max()
print(f"2. Competition timeframe: From {min_comp_date.strftime('%Y-%m-%d')} to {max_comp_date.strftime('%Y-%m-%d')}")



# Result 3 & Visualization 1: Number of Competitions Launched per Year
competitions_per_year = competitions_df['Competition_Year'].value_counts().sort_index().reset_index()
competitions_per_year.columns = ['Year', 'Num_Competitions']
print("\n3. Number of competitions launched per year:")
print(competitions_per_year.tail()) # Show recent years

fig = px.line(competitions_per_year, x='Year', y='Num_Competitions',
              title='1.1: Number of Kaggle Competitions Launched Per Year (15-Year Trend)',
              labels={'Num_Competitions': 'Number of Competitions', 'Year': 'Year'},
              markers=True)
fig.update_traces(marker_symbol='circle', marker_size=8, line_width=2)
fig.update_layout(hovermode="x unified")
fig.show()
print("Visualization 1: Line plot showing the yearly trend of new competition launches. This clearly depicts the growth and peak periods of competition activity on Kaggle.")



# Result 4 & Visualization 2: Monthly Trend of Competitions (more granular)
competitions_per_month = competitions_df['Competition_Month'].value_counts().sort_index().reset_index()
competitions_per_month.columns = ['Month', 'Num_Competitions']
competitions_per_month['Month'] = competitions_per_month['Month'].astype(str) # Convert Period to string for plotting
print("\n4. Number of competitions launched per month (last 12 months in data):")
print(competitions_per_month.tail(12))

fig = px.line(competitions_per_month, x='Month', y='Num_Competitions',
              title='1.2: Number of Kaggle Competitions Launched Per Month',
              labels={'Num_Competitions': 'Number of Competitions', 'Month': 'Month'},
              markers=True)
fig.update_xaxes(dtick="M3", tickformat="%b\n%Y") # Show every 3rd month
fig.update_traces(marker_symbol='circle', marker_size=5, line_width=1)
fig.update_layout(hovermode="x unified")
fig.show()
print("Visualization 2: Line plot showing the monthly trend of new competition launches, revealing more granular patterns and potential seasonality.")



# Result 5 & Visualization 3: Top 10 Host Segment Titles (Competition Categories)
top_host_segments = competitions_df['HostSegmentTitle'].value_counts().head(10).reset_index()
top_host_segments.columns = ['HostSegmentTitle', 'Count']
print("\n5. Top 10 Most Frequent Competition Host Segment Titles:")
print(top_host_segments)

fig = px.bar(top_host_segments, x='HostSegmentTitle', y='Count',
             title='1.3: Top 10 Most Frequent Competition Host Segment Titles',
             labels={'HostSegmentTitle': 'Competition Category', 'Count': 'Number of Competitions'},
             color='Count', color_continuous_scale=px.colors.sequential.Plasma)
fig.update_xaxes(tickangle=45)
fig.show()
print("Visualization 3: Bar chart displaying the dominant categories of competitions on Kaggle, providing an initial understanding of the platform's historical focus.")

# Result 6 & Visualization 4: HasKernels vs. OnlyAllowKernelSubmissions Over Time (Yearly)
# This shows the shift towards Kaggle Notebooks being used for submissions
kernels_data = competitions_df.groupby('Competition_Year')[['HasKernels', 'OnlyAllowKernelSubmissions']].sum().reset_index()
print("\n6. Competitions with Kernel requirements over time:")
print(kernels_data.tail())

fig = px.line(kernels_data, x='Competition_Year', y=['HasKernels', 'OnlyAllowKernelSubmissions'],
              title='1.4: Evolution of Kernel Usage in Competitions',
              labels={'value': 'Number of Competitions', 'Competition_Year': 'Year', 'variable': 'Requirement Type'},
              markers=True)
fig.update_traces(marker_symbol='circle', marker_size=8, line_width=2)
fig.update_layout(hovermode="x unified", legend_title_text='Requirement Type')
fig.show()
print("Visualization 4: Line plot showing the yearly count of competitions that either 'Have Kernels' or 'Only Allow Kernel Submissions'. This is a key indicator of Kaggle's platform evolution towards integrated notebook environments.")



# --- 5. User Registration Trends ---
if 'User_Register_Year' in df_users.columns:
    print("\n--- User Registration Trends ---")

    # Result 7: Total Number of Users
    total_users = df_users.shape[0]
    print(f"7. Total number of users recorded: {total_users}")

    # Result 8: Time range of user registrations
    min_user_date = df_users['RegisterDate'].min()
    max_user_date = df_users['RegisterDate'].max()
    print(f"8. User registration timeframe: From {min_user_date.strftime('%Y-%m-%d')} to {max_user_date.strftime('%Y-%m-%d')}")

    # Result 9 & Visualization 5: Number of New Users Registered per Year
    users_per_year = df_users['User_Register_Year'].value_counts().sort_index().reset_index()
    users_per_year.columns = ['Year', 'Num_Users']
    print("\n9. Number of new users registered per year:")
    print(users_per_year.tail())

    fig = px.bar(users_per_year, x='Year', y='Num_Users',
                 title='1.5: New Kaggle User Registrations Per Year',
                 labels={'Num_Users': 'Number of New Users', 'Year': 'Year'},
                 color='Num_Users', color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()
    print("Visualization 5: Bar chart illustrating the annual growth of the Kaggle user base, highlighting periods of rapid user acquisition.")

    # Result 10 & Visualization 6: Cumulative User Growth Over Time
    cumulative_users = users_per_year.sort_values('Year').copy()
    cumulative_users['Cumulative_Users'] = cumulative_users['Num_Users'].cumsum()
    print("\n10. Cumulative User Growth (last 5 years):")
    print(cumulative_users.tail())

    fig = px.area(cumulative_users, x='Year', y='Cumulative_Users',
                  title='1.6: Cumulative Kaggle User Growth Over Time',
                  labels={'Cumulative_Users': 'Cumulative Number of Users', 'Year': 'Year'},
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(hovermode="x unified")
    fig.show()
    print("Visualization 6: Area chart showing the total cumulative number of users on Kaggle, demonstrating the platform's overall expansion.")

    # Result 11 & Visualization 7: User Registrations per Month (Smoothed/Aggregated)
    # Using 'User_Register_Month'
    users_per_month = df_users['User_Register_Month'].value_counts().sort_index().reset_index()
    users_per_month.columns = ['Month', 'Num_Users']
    users_per_month['Month'] = users_per_month['Month'].astype(str) # Convert Period to string for plotting
    print("\n11. User registrations per month (last 12 months in data):")
    print(users_per_month.tail(12))

    fig = px.line(users_per_month, x='Month', y='Num_Users',
                  title='1.7: New Kaggle User Registrations Per Month',
                  labels={'Num_Users': 'Number of New Users', 'Month': 'Month'},
                  markers=True)
    fig.update_xaxes(dtick="M3", tickformat="%b\n%Y") # Show every 3rd month
    fig.update_traces(marker_symbol='circle', marker_size=5, line_width=1, line_color='orange')
    fig.update_layout(hovermode="x unified")
    fig.show()
    print("Visualization 7: Line plot detailing monthly user registration trends, which can reveal short-term fluctuations or impacts of specific events.")

    # Result 12 & Visualization 8: Competition to User Growth Ratio
    # Join competitions_per_year and users_per_year on 'Year'
    comp_user_growth = pd.merge(competitions_per_year, users_per_year, on='Year', how='inner')
    comp_user_growth['Competitions_Per_1000_Users'] = (comp_user_growth['Num_Competitions'] / comp_user_growth['Num_Users']) * 1000
    print("\n12. Ratio of Competitions to New Users per Year (per 1000 users):")
    print(comp_user_growth.tail())

    fig = px.line(comp_user_growth, x='Year', y='Competitions_Per_1000_Users',
                  title='1.8: Ratio of Competitions Launched Per 1000 New Users Per Year',
                  labels={'Competitions_Per_1000_Users': 'Competitions per 1000 New Users'},
                  markers=True, color_discrete_sequence=['purple'])
    fig.update_traces(marker_symbol='diamond', marker_size=10, line_width=2)
    fig.update_layout(hovermode="x unified")
    fig.show()
    print("Visualization 8: Line plot showing the ratio of new competitions launched relative to new user registrations. This metric indicates whether competition supply is keeping pace with community growth or if the platform is becoming more/less saturated with competitions per new user.")

else:
    print("\nSkipping user-related trend analysis as 'RegisterDate' column was not found in df_users.")




# --- 1. Evolution of HostSegmentTitle Categories ---
print("\n--- Evolution of Competition Categories (HostSegmentTitle) ---")

# Result 1: Unique HostSegmentTitles and their counts
unique_host_segments = competitions_df['HostSegmentTitle'].value_counts()
print("\n1. Unique Competition Host Segment Titles and their total counts:")
print(unique_host_segments)
print(f"Total unique HostSegmentTitles: {len(unique_host_segments)}")

# Result 2 & Visualization 1: Trend of Top N HostSegmentTitles over time (Stacked Area Chart)
# Identify top N categories for clear visualization. Let's pick top 7-10.
# For simplicity, we'll start with filtering categories with at least 50 occurrences
# and then take the top N from the remaining ones.
# Or, directly use top N from value_counts().
top_n = 10 # Number of top categories to visualize
top_host_segments_list = competitions_df['HostSegmentTitle'].value_counts().nlargest(top_n).index.tolist()

# Filter DataFrame to include only these top categories
filtered_competitions_df = competitions_df[competitions_df['HostSegmentTitle'].isin(top_host_segments_list)]

# Group by year and HostSegmentTitle
category_trends_yearly = filtered_competitions_df.groupby(['Competition_Year', 'HostSegmentTitle']).size().unstack(fill_value=0)
category_trends_yearly_melted = category_trends_yearly.reset_index().melt(id_vars='Competition_Year', var_name='HostSegmentTitle', value_name='Count')
print(f"\n2. Yearly counts for Top {top_n} Host Segment Titles (sample last 5 years):")
print(category_trends_yearly.tail())


fig = px.area(category_trends_yearly_melted, x='Competition_Year', y='Count', color='HostSegmentTitle',
              title=f'2.1: Evolution of Top {top_n} Competition Host Segment Titles Over Time',
              labels={'Count': 'Number of Competitions', 'Competition_Year': 'Year', 'HostSegmentTitle': 'Category'},
              template='plotly_white')
fig.update_layout(hovermode="x unified", legend_title_text='Competition Category')
fig.update_xaxes(dtick=1) # Ensure yearly ticks
fig.show()
print(f"Visualization 1: Stacked area chart showing the evolution of the top {top_n} competition categories by 'HostSegmentTitle' over the years. This reveals the rise and fall of different ML/AI focus areas on Kaggle.")

# Result 3 & Visualization 2: Percentage share of HostSegmentTitles over time
# Normalize the counts to get proportions
category_trends_yearly_percentage = category_trends_yearly.apply(lambda x: x / x.sum(), axis=1).fillna(0)
category_trends_yearly_percentage_melted = category_trends_yearly_percentage.reset_index().melt(id_vars='Competition_Year', var_name='HostSegmentTitle', value_name='Percentage')
print(f"\n3. Percentage share of Top {top_n} Host Segment Titles per year (sample last 5 years):")
print(category_trends_yearly_percentage.tail())

fig = px.area(category_trends_yearly_percentage_melted, x='Competition_Year', y='Percentage', color='HostSegmentTitle',
              title=f'2.2: Percentage Share of Top {top_n} Competition Host Segment Titles Over Time',
              labels={'Percentage': 'Percentage of Competitions', 'Competition_Year': 'Year', 'HostSegmentTitle': 'Category'},
              template='plotly_white', groupnorm='percent') # Use groupnorm='percent' for 100% stacked area
fig.update_layout(hovermode="x unified", legend_title_text='Competition Category')
fig.update_xaxes(dtick=1)
fig.show()
print(f"Visualization 2: 100% Stacked area chart illustrating the proportional shift in the top {top_n} competition categories over time. This highlights changes in Kaggle's strategic focus, for instance, from traditional ML to deep learning applications.")



# --- 2. Analysis of CompetitionTypeId ---
print("\n--- Analysis of CompetitionTypeId ---")

# Result 4: Unique CompetitionTypeIds and their counts
unique_comp_types = competitions_df['CompetitionTypeId'].value_counts()
print("\n4. Unique CompetitionTypeIds and their total counts:")
print(unique_comp_types)
print(f"Total unique CompetitionTypeIds: {len(unique_comp_types)}")

# Result 5 & Visualization 3: Trend of CompetitionTypeId over time
comp_type_trends_yearly = competitions_df.groupby(['Competition_Year', 'CompetitionTypeId']).size().unstack(fill_value=0)
comp_type_trends_yearly_melted = comp_type_trends_yearly.reset_index().melt(id_vars='Competition_Year', var_name='CompetitionTypeId', value_name='Count')
print("\n5. Yearly counts for each CompetitionTypeId (sample last 5 years):")
print(comp_type_trends_yearly.tail())

fig = px.bar(comp_type_trends_yearly_melted, x='Competition_Year', y='Count', color='CompetitionTypeId',
             title='2.3: Evolution of CompetitionTypeIds Over Time',
             labels={'Count': 'Number of Competitions', 'Competition_Year': 'Year', 'CompetitionTypeId': 'Competition Type ID'},
             template='plotly_white')
fig.update_layout(hovermode="x unified", legend_title_text='Competition Type ID')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 3: Stacked bar chart showing the yearly distribution of different 'CompetitionTypeIds'. This can reveal the introduction of new competition formats or the deprecation of older ones.")

# Result 6 & Visualization 4: CompetitionTypeId popularity by TotalCompetitors
# Aggregate total competitors by CompetitionTypeId
comp_type_competitors = competitions_df.groupby('CompetitionTypeId')['TotalCompetitors'].sum().sort_values(ascending=False).reset_index()
print("\n6. Total Competitors by CompetitionTypeId:")
print(comp_type_competitors)

fig = px.bar(comp_type_competitors, x='CompetitionTypeId', y='TotalCompetitors',
             title='2.4: Total Competitors by Competition Type ID',
             labels={'TotalCompetitors': 'Total Number of Competitors', 'CompetitionTypeId': 'Competition Type ID'},
             color='TotalCompetitors', color_continuous_scale=px.colors.sequential.Sunset)
fig.show()
print("Visualization 4: Bar chart displaying the total number of competitors for each 'CompetitionTypeId'. This helps identify which competition formats attract the most participants.")




# --- 3. Relationship between Competition Type and Engagement Metrics ---
print("\n--- Relationship between Competition Type and Engagement Metrics ---")

# Result 7 & Visualization 5: Average TotalTeams per HostSegmentTitle per Year
avg_teams_per_category_yearly = filtered_competitions_df.groupby(['Competition_Year', 'HostSegmentTitle'])['TotalTeams'].mean().unstack(fill_value=0)
avg_teams_per_category_yearly_melted = avg_teams_per_category_yearly.reset_index().melt(id_vars='Competition_Year', var_name='HostSegmentTitle', value_name='Average_Teams')
print("\n7. Average number of teams per competition category per year (sample last 5 years):")
print(avg_teams_per_category_yearly.tail())

fig = px.line(avg_teams_per_category_yearly_melted, x='Competition_Year', y='Average_Teams', color='HostSegmentTitle',
              title=f'2.5: Average Teams per Top {top_n} Competition Category Over Time',
              labels={'Average_Teams': 'Average Number of Teams', 'Competition_Year': 'Year', 'HostSegmentTitle': 'Category'},
              markers=True, template='plotly_dark')
fig.update_layout(hovermode="x unified", legend_title_text='Competition Category')
fig.update_xaxes(dtick=1)
fig.show()
print(f"Visualization 5: Line plot showing the average number of teams participating in competitions within the top {top_n} categories each year. This helps understand the sustained or declining interest in specific ML/AI domains.")

# Result 8 & Visualization 6: Average TotalSubmissions per HostSegmentTitle per Year
avg_submissions_per_category_yearly = filtered_competitions_df.groupby(['Competition_Year', 'HostSegmentTitle'])['TotalSubmissions'].mean().unstack(fill_value=0)
avg_submissions_per_category_yearly_melted = avg_submissions_per_category_yearly.reset_index().melt(id_vars='Competition_Year', var_name='HostSegmentTitle', value_name='Average_Submissions')
print("\n8. Average number of submissions per competition category per year (sample last 5 years):")
print(avg_submissions_per_category_yearly.tail())

fig = px.line(avg_submissions_per_category_yearly_melted, x='Competition_Year', y='Average_Submissions', color='HostSegmentTitle',
              title=f'2.6: Average Submissions per Top {top_n} Competition Category Over Time',
              labels={'Average_Submissions': 'Average Number of Submissions', 'Competition_Year': 'Year', 'HostSegmentTitle': 'Category'},
              markers=True, template='plotly_dark')
fig.update_layout(hovermode="x unified", legend_title_text='Competition Category')
fig.update_xaxes(dtick=1)
fig.show()
print(f"Visualization 6: Line plot depicting the average number of submissions made in competitions within the top {top_n} categories annually. This indicates the effort and activity levels within different ML/AI problem types.")

# Result 9 & Visualization 7: Box Plot of TotalCompetitors by HostSegmentTitle (Overall)
print("\n9. Distribution of Total Competitors across top competition categories (overall):")
# Use a box plot to show the spread (median, quartiles, outliers)
fig = px.box(filtered_competitions_df, x='HostSegmentTitle', y='TotalCompetitors',
             title=f'2.7: Distribution of Total Competitors by Top {top_n} Competition Category',
             labels={'HostSegmentTitle': 'Competition Category', 'TotalCompetitors': 'Total Competitors'},
             points="outliers", # Show individual outliers
             height=600)
fig.update_xaxes(tickangle=45)
fig.show()
print(f"Visualization 7: Box plot showing the distribution of 'TotalCompetitors' for each of the top {top_n} 'HostSegmentTitle' categories. This helps identify categories that consistently attract a high number of participants vs. those with more variance or lower engagement.")

# Result 10 & Visualization 8: Correlation between HostSegmentTitle and Dataset Size (if data size is meaningful)
# Average TotalCompressedBytes per HostSegmentTitle per Year
avg_data_size_per_category_yearly = filtered_competitions_df.groupby(['Competition_Year', 'HostSegmentTitle'])['TotalCompressedBytes'].mean().unstack(fill_value=0)
avg_data_size_per_category_yearly_melted = avg_data_size_per_category_yearly.reset_index().melt(id_vars='Competition_Year', var_name='HostSegmentTitle', value_name='Average_Data_Size_Bytes')
avg_data_size_per_category_yearly_melted['Average_Data_Size_GB'] = avg_data_size_per_category_yearly_melted['Average_Data_Size_Bytes'] / (1024**3) # Convert to GB
print("\n10. Average data size (GB) per competition category per year (sample last 5 years):")
print(avg_data_size_per_category_yearly_melted.tail())

fig = px.line(avg_data_size_per_category_yearly_melted, x='Competition_Year', y='Average_Data_Size_GB', color='HostSegmentTitle',
              title=f'2.8: Average Competition Data Size (GB) by Top {top_n} Category Over Time',
              labels={'Average_Data_Size_GB': 'Average Data Size (GB)', 'Competition_Year': 'Year', 'HostSegmentTitle': 'Category'},
              markers=True, template='plotly_dark')
fig.update_layout(hovermode="x unified", legend_title_text='Competition Category')
fig.update_xaxes(dtick=1)
fig.show()
print(f"Visualization 8: Line plot illustrating the trend of average dataset size (in GB) for competitions within the top {top_n} categories. This is crucial for understanding the increasing demand for processing larger datasets in specific ML/AI domains.")


# Result 11 & Visualization 9: Top Host Segment Title by number of total submissions
top_submission_categories = competitions_df.groupby('HostSegmentTitle')['TotalSubmissions'].sum().sort_values(ascending=False).head(10).reset_index()
print("\n11. Top 10 Host Segment Titles by total number of submissions:")
print(top_submission_categories)

fig = px.bar(top_submission_categories, x='HostSegmentTitle', y='TotalSubmissions',
             title='2.9: Top 10 Competition Categories by Total Submissions',
             labels={'HostSegmentTitle': 'Competition Category', 'TotalSubmissions': 'Total Submissions'},
             color='TotalSubmissions', color_continuous_scale=px.colors.sequential.Magenta)
fig.update_xaxes(tickangle=45)
fig.show()
print("Visualization 9: Bar chart identifying which competition categories have generated the most total submissions, indicating areas of high activity and iterative problem-solving.")

# Result 12 & Visualization 10: CompetitionTypeId distribution by year, normalized to show proportion
comp_type_trends_yearly_normalized = competitions_df.groupby('Competition_Year')['CompetitionTypeId'].value_counts(normalize=True).unstack(fill_value=0)
comp_type_trends_yearly_normalized_melted = comp_type_trends_yearly_normalized.reset_index().melt(id_vars='Competition_Year', var_name='CompetitionTypeId', value_name='Proportion')
print("\n12. Proportional distribution of CompetitionTypeIds per year (sample last 5 years):")
print(comp_type_trends_yearly_normalized.tail())

fig = px.area(comp_type_trends_yearly_normalized_melted, x='Competition_Year', y='Proportion', color='CompetitionTypeId',
              title='2.10: Proportional Evolution of CompetitionTypeIds Over Time',
              labels={'Proportion': 'Proportion of Competitions', 'Competition_Year': 'Year', 'CompetitionTypeId': 'Competition Type ID'},
              template='plotly_white', groupnorm='percent') # Ensures 100% stacked area
fig.update_layout(hovermode="x unified", legend_title_text='Competition Type ID')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 10: 100% Stacked area chart showing the proportional changes in 'CompetitionTypeIds' over time, highlighting shifts in preferred competition formats.")



print("--- Part 3: Understanding Competition Engagement & Rewards ---")

# --- 1. Trends in TotalCompetitors and TotalTeams ---
print("\n--- Trends in Competition Engagement ---")

# Convert 'RewardQuantity' to numeric, handling non-numeric values
# We'll assume 'RewardQuantity' represents monetary value if numeric, or count of non-monetary prizes
# For this analysis, let's treat it as monetary for now and convert non-numeric to 0
competitions_df['RewardQuantity_Numeric'] = pd.to_numeric(competitions_df['RewardQuantity'], errors='coerce').fillna(0)

# Result 1: Overall statistics for TotalCompetitors and TotalTeams
print("\n1. Overall Statistics for Total Competitors and Total Teams:")
print(competitions_df[['TotalCompetitors', 'TotalTeams', 'TotalSubmissions']].describe())

# Result 2 & Visualization 1: Average TotalCompetitors and TotalTeams per Year
engagement_yearly = competitions_df.groupby('Competition_Year')[['TotalCompetitors', 'TotalTeams']].mean().reset_index()
print("\n2. Average Total Competitors and Total Teams per Year (sample last 5 years):")
print(engagement_yearly.tail())

fig = px.line(engagement_yearly, x='Competition_Year', y=['TotalCompetitors', 'TotalTeams'],
              title='3.1: Average Total Competitors and Teams Per Competition Over Time',
              labels={'value': 'Average Count', 'Competition_Year': 'Year', 'variable': 'Metric'},
              markers=True, template='plotly_white')
fig.update_layout(hovermode="x unified", legend_title_text='Engagement Metric')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 1: Line plot showing the average number of competitors and teams per competition over the years. This helps to understand if competitions are generally attracting more or fewer participants/teams over time.")

# Result 3 & Visualization 2: Total Competitors and Total Teams per Year (Sum)
total_engagement_yearly = competitions_df.groupby('Competition_Year')[['TotalCompetitors', 'TotalTeams']].sum().reset_index()
print("\n3. Total Competitors and Total Teams per Year (Sum, sample last 5 years):")
print(total_engagement_yearly.tail())

fig = px.bar(total_engagement_yearly, x='Competition_Year', y=['TotalCompetitors', 'TotalTeams'],
             title='3.2: Total Competitors and Teams Across All Competitions Per Year',
             labels={'value': 'Total Count', 'Competition_Year': 'Year', 'variable': 'Metric'},
             template='plotly_white', barmode='group')
fig.update_layout(hovermode="x unified", legend_title_text='Engagement Metric')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 2: Bar chart showing the total number of competitors and teams across *all* competitions in a given year. This captures the overall scale of community participation, accounting for the number of competitions as well.")


# Result 4 & Visualization 3: Distribution of TotalCompetitors (Histogram)
print("\n4. Distribution of Total Competitors:")
print(competitions_df['TotalCompetitors'].describe())

fig = px.histogram(competitions_df, x='TotalCompetitors', nbins=50,
                   title='3.3: Distribution of Total Competitors per Competition',
                   labels={'TotalCompetitors': 'Number of Competitors'},
                   log_y=True, # Log scale for y-axis due to skewed distribution
                   template='plotly_white')
fig.show()
print("Visualization 3: Histogram showing the distribution of total competitors, often revealing a long-tail distribution where many competitions have few participants and a few have very many.")

# Result 5 & Visualization 4: Distribution of TotalTeams (Histogram)
print("\n5. Distribution of Total Teams:")
print(competitions_df['TotalTeams'].describe())

fig = px.histogram(competitions_df, x='TotalTeams', nbins=50,
                   title='3.4: Distribution of Total Teams per Competition',
                   labels={'TotalTeams': 'Number of Teams'},
                   log_y=True, # Log scale for y-axis due to skewed distribution
                   template='plotly_white')
fig.show()
print("Visualization 4: Histogram showing the distribution of total teams, similar to competitors, indicating team-based engagement patterns.")



# --- 2. Relationship with RewardQuantity (Prize Money) ---
print("\n--- Impact of Reward Quantity (Prize Money) ---")

# Filter for competitions with non-zero RewardQuantity_Numeric (monetary prizes)
monetary_competitions = competitions_df[competitions_df['RewardQuantity_Numeric'] > 0].copy()

# Result 6 & Visualization 5: Total Prize Money Awarded per Year
total_prize_money_yearly = monetary_competitions.groupby('Competition_Year')['RewardQuantity_Numeric'].sum().reset_index()
print("\n6. Total Prize Money Awarded Per Year (USD, sample last 5 years):")
print(total_prize_money_yearly.tail())

fig = px.bar(total_prize_money_yearly, x='Competition_Year', y='RewardQuantity_Numeric',
             title='3.5: Total Prize Money Awarded in Kaggle Competitions Per Year',
             labels={'RewardQuantity_Numeric': 'Total Prize Money (USD)', 'Competition_Year': 'Year'},
             color='RewardQuantity_Numeric', color_continuous_scale=px.colors.sequential.Plotly3,
             template='plotly_white')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 5: Bar chart illustrating the total prize money distributed across all competitions each year. This highlights Kaggle's and its sponsors' investment in the community.")

# Result 7 & Visualization 6: Average Prize Money per Competition per Year
avg_prize_money_yearly = monetary_competitions.groupby('Competition_Year')['RewardQuantity_Numeric'].mean().reset_index()
print("\n7. Average Prize Money Per Competition Per Year (USD, sample last 5 years):")
print(avg_prize_money_yearly.tail())

fig = px.line(avg_prize_money_yearly, x='Competition_Year', y='RewardQuantity_Numeric',
              title='3.6: Average Prize Money Per Kaggle Competition Per Year',
              labels={'RewardQuantity_Numeric': 'Average Prize Money (USD)', 'Competition_Year': 'Year'},
              markers=True, template='plotly_dark')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 6: Line plot showing the average prize money offered per competition annually. This can indicate whether competition sponsors are increasing or decreasing the typical reward.")

# Result 8 & Visualization 7: Scatter plot: RewardQuantity_Numeric vs. TotalCompetitors
# Use log scale for better visualization due to large spread
fig = px.scatter(monetary_competitions, x='RewardQuantity_Numeric', y='TotalCompetitors',
                 size='TotalSubmissions', color='Competition_Year',
                 hover_name='Title',
                 log_x=True, log_y=True, # Log scales for both axes
                 title='3.7: Prize Money vs. Total Competitors (Size=Total Submissions)',
                 labels={'RewardQuantity_Numeric': 'Prize Money (USD, Log Scale)', 'TotalCompetitors': 'Total Competitors (Log Scale)'},
                 template='plotly_white')
fig.update_layout(showlegend=True, legend_title_text='Competition Year',
                  xaxis_title='Prize Money (USD, Log Scale)',
                  yaxis_title='Total Competitors (Log Scale)')
fig.show()
print("Visualization 7: Scatter plot exploring the relationship between prize money and the number of competitors, with submission count indicating activity level. This helps to discern if higher prizes consistently lead to more engagement.")




# --- 3. Analysis of NumPrizes and RewardType ---
print("\n--- Analysis of Prize Structure ---")

# Result 9: Unique RewardTypes and their counts
unique_reward_types = competitions_df['RewardType'].value_counts(dropna=False) # Include NaN for competitions with no specified reward type
print("\n9. Unique Reward Types and their counts:")
print(unique_reward_types)

# Result 10 & Visualization 8: Trend of NumPrizes over time (average)
# Fill NaN NumPrizes with 0 before calculating mean
competitions_df['NumPrizes'].fillna(0, inplace=True)
avg_num_prizes_yearly = competitions_df.groupby('Competition_Year')['NumPrizes'].mean().reset_index()
print("\n10. Average Number of Prizes Per Competition Per Year (sample last 5 years):")
print(avg_num_prizes_yearly.tail())

fig = px.line(avg_num_prizes_yearly, x='Competition_Year', y='NumPrizes',
              title='3.8: Average Number of Prizes Per Competition Per Year',
              labels={'NumPrizes': 'Average Number of Prizes', 'Competition_Year': 'Year'},
              markers=True, template='plotly_white')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 8: Line plot showing the average number of prize tiers offered per competition annually. This indicates whether Kaggle/sponsors are spreading rewards among more winners or concentrating them.")

# Result 11 & Visualization 9: Correlation between NumPrizes and TotalCompetitors
fig = px.scatter(competitions_df, x='NumPrizes', y='TotalCompetitors',
                 size='RewardQuantity_Numeric', color='Competition_Year',
                 hover_name='Title',
                 log_y=True, # Log scale for y-axis
                 title='3.9: Number of Prizes vs. Total Competitors (Size=Prize Money)',
                 labels={'NumPrizes': 'Number of Prizes', 'TotalCompetitors': 'Total Competitors (Log Scale)'},
                 template='plotly_white')
fig.update_layout(showlegend=True, legend_title_text='Competition Year')
fig.show()
print("Visualization 9: Scatter plot examining if offering more prize tiers correlates with higher overall participation. The size of bubbles indicates the prize money.")

# Result 12 & Visualization 10: Percentage of Competitions with Monetary Prizes vs. Non-Monetary/No Prizes
# Define a 'Has_Monetary_Prize' column
competitions_df['Has_Monetary_Prize'] = competitions_df['RewardQuantity_Numeric'] > 0

monetary_prize_yearly_counts = competitions_df.groupby('Competition_Year')['Has_Monetary_Prize'].value_counts(normalize=True).unstack(fill_value=0)
monetary_prize_yearly_counts.columns = ['No_Monetary_Prize', 'Has_Monetary_Prize'] # Rename columns for clarity
monetary_prize_yearly_melted = monetary_prize_yearly_counts.reset_index().melt(id_vars='Competition_Year', var_name='Prize_Status', value_name='Proportion')

# To ensure 'No Monetary Prize' comes first in legend, if desired, sort by it.
# Or just let plotly order them, usually alphabetically if not specified.
print("\n12. Proportion of Competitions with and without Monetary Prizes per Year (sample last 5 years):")
print(monetary_prize_yearly_counts.tail())

fig = px.area(monetary_prize_yearly_melted, x='Competition_Year', y='Proportion', color='Prize_Status',
              title='3.10: Proportion of Competitions with Monetary vs. Non-Monetary/No Prizes Per Year',
              labels={'Proportion': 'Proportion of Competitions', 'Competition_Year': 'Year', 'Prize_Status': 'Prize Status'},
              template='plotly_white', groupnorm='percent') # 100% stacked area
fig.update_layout(hovermode="x unified", legend_title_text='Prize Status')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 10: 100% Stacked area chart showing the proportion of competitions offering monetary prizes compared to those offering non-monetary (e.g., swag, bragging rights) or no prizes. This reveals the strategic shift in incentive structures.")



for col in ['TotalCompressedBytes', 'TotalUncompressedBytes']:
    if col in competitions_df.columns:
        competitions_df[col] = pd.to_numeric(competitions_df[col], errors='coerce')
    else:
        print(f"Warning: Column '{col}' not found in competitions_df. Some analyses might be skipped.")

# Drop rows where these key columns are NaN for this specific analysis to avoid issues
initial_count = competitions_df.shape[0]
competitions_df_data_size = competitions_df.dropna(subset=['TotalCompressedBytes', 'TotalUncompressedBytes']).copy()
print(f"Dropped {initial_count - competitions_df_data_size.shape[0]} rows from competitions_df for data size analysis due to missing values.")






print("--- Part 4: Data Characteristics and Complexity: The Evolving Datasets ---")

# --- 1. Trends in Dataset Size ---
print("\n--- Trends in Competition Dataset Size ---")

# Convert bytes to GB for easier interpretation
competitions_df_data_size['Compressed_Size_GB'] = competitions_df_data_size['TotalCompressedBytes'] / (1024**3)
competitions_df_data_size['Uncompressed_Size_GB'] = competitions_df_data_size['TotalUncompressedBytes'] / (1024**3)


# Result 1: Overall statistics for dataset sizes
print("\n1. Overall Statistics for Compressed and Uncompressed Dataset Sizes (GB):")
print(competitions_df_data_size[['Compressed_Size_GB', 'Uncompressed_Size_GB']].describe())


# Result 2 & Visualization 1: Average Compressed Dataset Size per Year
avg_compressed_size_yearly = competitions_df_data_size.groupby('Competition_Year')['Compressed_Size_GB'].mean().reset_index()
print("\n2. Average Compressed Dataset Size (GB) Per Competition Per Year (sample last 5 years):")
print(avg_compressed_size_yearly.tail())

fig = px.line(avg_compressed_size_yearly, x='Competition_Year', y='Compressed_Size_GB',
              title='4.1: Average Compressed Dataset Size (GB) Per Competition Over Time',
              labels={'Compressed_Size_GB': 'Average Compressed Size (GB)', 'Competition_Year': 'Year'},
              markers=True, template='plotly_white')
fig.update_layout(hovermode="x unified")
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 1: Line plot showing the average compressed size of datasets used in Kaggle competitions over time. This indicates the increasing scale of data challenges.")

# Result 3 & Visualization 2: Average Uncompressed Dataset Size per Year
avg_uncompressed_size_yearly = competitions_df_data_size.groupby('Competition_Year')['Uncompressed_Size_GB'].mean().reset_index()
print("\n3. Average Uncompressed Dataset Size (GB) Per Competition Per Year (sample last 5 years):")
print(avg_uncompressed_size_yearly.tail())

fig = px.line(avg_uncompressed_size_yearly, x='Competition_Year', y='Uncompressed_Size_GB',
              title='4.2: Average Uncompressed Dataset Size (GB) Per Competition Over Time',
              labels={'Uncompressed_Size_GB': 'Average Uncompressed Size (GB)', 'Competition_Year': 'Year'},
              markers=True, template='plotly_dark')
fig.update_layout(hovermode="x unified")
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 2: Line plot displaying the average uncompressed size of competition datasets over time. This metric better reflects the actual memory and processing demands.")

# Result 4 & Visualization 3: Total Compressed Bytes per Year (Sum)
total_compressed_size_yearly = competitions_df_data_size.groupby('Competition_Year')['TotalCompressedBytes'].sum().reset_index()
total_compressed_size_yearly['Total_Compressed_Size_TB'] = total_compressed_size_yearly['TotalCompressedBytes'] / (1024**4) # Convert to TB
print("\n4. Total Compressed Data (TB) across all competitions per Year (sample last 5 years):")
print(total_compressed_size_yearly.tail())

fig = px.bar(total_compressed_size_yearly, x='Competition_Year', y='Total_Compressed_Size_TB',
             title='4.3: Total Compressed Dataset Volume (TB) Across All Competitions Per Year',
             labels={'Total_Compressed_Size_TB': 'Total Compressed Size (TB)', 'Competition_Year': 'Year'},
             color='Total_Compressed_Size_TB', color_continuous_scale=px.colors.sequential.Bluyl,
             template='plotly_white')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 3: Bar chart showing the cumulative compressed data volume across all competitions each year. This highlights the sheer scale of data being handled on the platform annually.")



# --- 2. Relationship between Data Size and Engagement/Complexity ---
print("\n--- Data Size vs. Engagement/Complexity ---")

# Result 5 & Visualization 4: Scatter Plot: Compressed Size vs. TotalCompetitors
fig = px.scatter(competitions_df_data_size, x='Compressed_Size_GB', y='TotalCompetitors',
                 size='TotalSubmissions', color='Competition_Year',
                 hover_name='Title',
                 log_x=True, log_y=True, # Log scales for both axes
                 title='4.4: Dataset Compressed Size vs. Total Competitors (Size=Total Submissions)',
                 labels={'Compressed_Size_GB': 'Compressed Size (GB, Log Scale)', 'TotalCompetitors': 'Total Competitors (Log Scale)'},
                 template='plotly_white')
fig.update_layout(showlegend=True, legend_title_text='Competition Year')
fig.show()
print("Visualization 4: Scatter plot exploring if larger datasets lead to more competitors. Log scales are used to handle wide value ranges.")

# Result 6 & Visualization 5: Scatter Plot: Compressed Size vs. TotalSubmissions
fig = px.scatter(competitions_df_data_size, x='Compressed_Size_GB', y='TotalSubmissions',
                 size='TotalCompetitors', color='Competition_Year',
                 hover_name='Title',
                 log_x=True, log_y=True, # Log scales for both axes
                 title='4.5: Dataset Compressed Size vs. Total Submissions (Size=Total Competitors)',
                 labels={'Compressed_Size_GB': 'Compressed Size (GB, Log Scale)', 'TotalSubmissions': 'Total Submissions (Log Scale)'},
                 template='plotly_dark')
fig.update_layout(showlegend=True, legend_title_text='Competition Year')
fig.show()
print("Visualization 5: Scatter plot showing the relationship between dataset size and the total number of submissions, indicating the effort required for larger data challenges.")

# Result 7 & Visualization 6: Average Data Size per HostSegmentTitle per Year
# Join competitions_df_data_size with filtered_competitions_df (from Part 2, for top categories)
# For simplicity, let's re-filter based on top_host_segments_list if it's available, otherwise just use HostSegmentTitle directly
# Re-identify top categories from the data size filtered DF
top_n = 10 # Number of top categories to visualize, consistent with Part 2
if 'HostSegmentTitle' in competitions_df_data_size.columns:
    top_host_segments_list_p4 = competitions_df_data_size['HostSegmentTitle'].value_counts().nlargest(top_n).index.tolist()
    filtered_competitions_df_p4 = competitions_df_data_size[competitions_df_data_size['HostSegmentTitle'].isin(top_host_segments_list_p4)].copy()

    avg_data_size_by_category_yearly = filtered_competitions_df_p4.groupby(['Competition_Year', 'HostSegmentTitle'])['Compressed_Size_GB'].mean().unstack(fill_value=0)
    avg_data_size_by_category_yearly_melted = avg_data_size_by_category_yearly.reset_index().melt(id_vars='Competition_Year', var_name='HostSegmentTitle', value_name='Average_Compressed_Size_GB')
    print(f"\n7. Average Compressed Data Size (GB) by Top {top_n} Competition Category per Year (sample last 5 years):")
    print(avg_data_size_by_category_yearly.tail())

    fig = px.line(avg_data_size_by_category_yearly_melted, x='Competition_Year', y='Average_Compressed_Size_GB', color='HostSegmentTitle',
                  title=f'4.6: Average Compressed Data Size (GB) by Top {top_n} Competition Category Over Time',
                  labels={'Average_Compressed_Size_GB': 'Avg. Compressed Size (GB)', 'Competition_Year': 'Year', 'HostSegmentTitle': 'Category'},
                  markers=True, template='plotly_white')
    fig.update_layout(hovermode="x unified", legend_title_text='Competition Category')
    fig.update_xaxes(dtick=1)
    fig.show()
    print(f"Visualization 6: Line plot showing how the average dataset size has evolved for the top {top_n} competition categories. This reveals which domains are increasingly data-heavy.")
else:
    print("\nWarning: HostSegmentTitle not available for data size analysis. Skipping category-specific data size trend.")



# --- 3. Data Compression and Efficiency ---
print("\n--- Data Compression and Efficiency ---")

# Calculate compression ratio
competitions_df_data_size['Compression_Ratio'] = competitions_df_data_size['TotalUncompressedBytes'] / competitions_df_data_size['TotalCompressedBytes']
# Handle cases where TotalCompressedBytes might be 0, leading to infinite ratio
competitions_df_data_size['Compression_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
competitions_df_data_size.dropna(subset=['Compression_Ratio'], inplace=True)


# Result 8: Overall statistics for compression ratio
print("\n8. Overall Statistics for Compression Ratio (Uncompressed / Compressed):")
print(competitions_df_data_size['Compression_Ratio'].describe())


# Result 9 & Visualization 7: Average Compression Ratio per Year
avg_compression_ratio_yearly = competitions_df_data_size.groupby('Competition_Year')['Compression_Ratio'].mean().reset_index()
print("\n9. Average Compression Ratio Per Competition Per Year (sample last 5 years):")
print(avg_compression_ratio_yearly.tail())

fig = px.line(avg_compression_ratio_yearly, x='Competition_Year', y='Compression_Ratio',
              title='4.7: Average Data Compression Ratio Per Competition Over Time',
              labels={'Compression_Ratio': 'Average Compression Ratio', 'Competition_Year': 'Year'},
              markers=True, template='plotly_white')
fig.update_layout(hovermode="x unified")
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 7: Line plot showing the average data compression ratio over time. This indicates if the data types or storage practices are changing towards more compressible formats.")

# Result 10 & Visualization 8: Top 10 competitions by Uncompressed Size
top_10_largest_uncompressed = competitions_df_data_size.sort_values('Uncompressed_Size_GB', ascending=False).head(10)
print("\n10. Top 10 Competitions by Uncompressed Data Size (GB):")
print(top_10_largest_uncompressed[['Title', 'Uncompressed_Size_GB', 'Competition_Year']])

fig = px.bar(top_10_largest_uncompressed, x='Title', y='Uncompressed_Size_GB',
             title='4.8: Top 10 Competitions by Uncompressed Data Size (GB)',
             labels={'Uncompressed_Size_GB': 'Uncompressed Size (GB)', 'Title': 'Competition Title'},
             color='Uncompressed_Size_GB', color_continuous_scale=px.colors.sequential.OrRd,
             template='plotly_white')
fig.update_xaxes(tickangle=45)
fig.show()
print("Visualization 8: Bar chart highlighting the ten competitions that involved the largest uncompressed datasets, showcasing extreme data scale challenges.")


# Result 11 & Visualization 9: Distribution of Compressed Data Size (Histogram)
print("\n11. Distribution of Compressed Data Size (GB):")
print(competitions_df_data_size['Compressed_Size_GB'].describe())

fig = px.histogram(competitions_df_data_size, x='Compressed_Size_GB', nbins=50,
                   title='4.9: Distribution of Compressed Data Size per Competition (GB)',
                   labels={'Compressed_Size_GB': 'Compressed Size (GB)'},
                   log_y=True, # Log scale for y-axis due to skewed distribution
                   template='plotly_white')
fig.show()
print("Visualization 9: Histogram showing the distribution of compressed data sizes. Most competitions have small datasets, but a few have very large ones.")

# Result 12 & Visualization 10: Comparison of Compressed vs. Uncompressed Sizes (Scatter Plot)
fig = px.scatter(competitions_df_data_size, x='TotalCompressedBytes', y='TotalUncompressedBytes',
                 color='Competition_Year',
                 hover_name='Title',
                 log_x=True, log_y=True, # Log scales for both axes
                 title='4.10: Compressed vs. Uncompressed Dataset Size (Bytes) by Competition Year',
                 labels={'TotalCompressedBytes': 'Compressed Size (Bytes, Log Scale)', 'TotalUncompressedBytes': 'Uncompressed Size (Bytes, Log Scale)'},
                 template='plotly_white')
fig.update_layout(showlegend=True, legend_title_text='Competition Year')
fig.add_shape(type='line', line=dict(dash='dash', color='red'), x0=1e3, y0=1e3, x1=competitions_df_data_size['TotalUncompressedBytes'].max(), y1=competitions_df_data_size['TotalUncompressedBytes'].max())
fig.show()
print("Visualization 10: Scatter plot comparing compressed and uncompressed dataset sizes. The diagonal red dashed line represents a 1:1 ratio (no compression). Points below the line indicate effective compression. This plot helps understand the typical compression achieved and how it might vary over time.")


print("\n--- Part 4: Completed ---")


print("--- Part 5: Advanced Competition Features and Platform Evolution ---")

# --- 1. Evolution of Kernel-Based Submissions ---
print("\n--- Trends in Kernel-Based Submissions ---")

# Convert boolean columns to numeric (0/1) for easier aggregation
competitions_df['HasKernels_int'] = competitions_df['HasKernels'].astype(int)
competitions_df['OnlyAllowKernelSubmissions_int'] = competitions_df['OnlyAllowKernelSubmissions'].astype(int)

# Result 1 & Visualization 1: Proportion of competitions requiring/allowing kernels over time
kernel_trends_yearly = competitions_df.groupby('Competition_Year')[['HasKernels_int', 'OnlyAllowKernelSubmissions_int']].mean().reset_index()
kernel_trends_yearly.columns = ['Competition_Year', 'Proportion_HasKernels', 'Proportion_OnlyAllowKernelSubmissions']
print("\n1. Proportion of competitions with 'HasKernels' and 'OnlyAllowKernelSubmissions' per year (sample last 5 years):")
print(kernel_trends_yearly.tail())

fig = px.line(kernel_trends_yearly, x='Competition_Year', y=['Proportion_HasKernels', 'Proportion_OnlyAllowKernelSubmissions'],
              title='5.1: Proportion of Competitions with Kernel Features Over Time',
              labels={'value': 'Proportion', 'Competition_Year': 'Year', 'variable': 'Feature'},
              markers=True, template='plotly_white')
fig.update_layout(hovermode="x unified", legend_title_text='Feature')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 1: Line plot showing the increasing proportion of competitions that either allow or *only* allow kernel-based submissions. This indicates Kaggle's push towards reproducible research and cloud-based environments.")


# Result 2 & Visualization 2: Total Competitors vs. OnlyAllowKernelSubmissions
# Analyze engagement for kernel-only competitions vs others
kernel_only_engagement = competitions_df.groupby('OnlyAllowKernelSubmissions')[['TotalCompetitors', 'TotalTeams', 'TotalSubmissions']].mean().reset_index()
kernel_only_engagement['OnlyAllowKernelSubmissions_Label'] = kernel_only_engagement['OnlyAllowKernelSubmissions'].map({True: 'Kernel-Only Submissions', False: 'Other Submission Methods'})
print("\n2. Average Engagement Metrics for Kernel-Only vs. Other Competitions:")
print(kernel_only_engagement)

fig = px.bar(kernel_only_engagement, x='OnlyAllowKernelSubmissions_Label', y=['TotalCompetitors', 'TotalTeams', 'TotalSubmissions'],
             title='5.2: Average Engagement for Kernel-Only Competitions',
             labels={'value': 'Average Count', 'OnlyAllowKernelSubmissions_Label': 'Submission Type', 'variable': 'Engagement Metric'},
             barmode='group', template='plotly_white')
fig.update_layout(hovermode="x unified", legend_title_text='Metric')
fig.show()
print("Visualization 2: Bar chart comparing average total competitors, teams, and submissions for competitions that strictly require kernel-only submissions versus those that allow other methods. This helps assess the impact of this rule change on participation.")



# --- 2. Leaderboard Dynamics ---
print("\n--- Leaderboard Dynamics ---")

# Result 3: Overall statistics for LeaderboardPercentage
print("\n3. Overall Statistics for Leaderboard Percentage:")
print(competitions_df['LeaderboardPercentage'].describe())

# Result 4 & Visualization 3: Trend of LeaderboardPercentage over time (average)
# Filter out competitions without a leaderboard (HasLeaderboard == False) if desired, or assume NaN means no leaderboard
leaderboard_pct_yearly = competitions_df[competitions_df['HasLeaderboard'] == True].groupby('Competition_Year')['LeaderboardPercentage'].mean().reset_index()
print("\n4. Average Leaderboard Percentage Per Year (sample last 5 years):")
print(leaderboard_pct_yearly.tail())

fig = px.line(leaderboard_pct_yearly, x='Competition_Year', y='LeaderboardPercentage',
              title='5.3: Average Public Leaderboard Percentage Over Time',
              labels={'LeaderboardPercentage': 'Average Public Leaderboard Percentage', 'Competition_Year': 'Year'},
              markers=True, template='plotly_dark')
fig.update_layout(hovermode="x unified")
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 3: Line plot showing the average percentage of test data used for the public leaderboard over time. This indicates a shift towards larger private test sets, promoting robust model generalization.")


# Result 5 & Visualization 4: HasLeaderboard trend
has_leaderboard_yearly = competitions_df.groupby('Competition_Year')['HasLeaderboard'].value_counts(normalize=True).unstack(fill_value=0)
if True in has_leaderboard_yearly.columns: # Check if 'True' column exists
    has_leaderboard_yearly['Proportion_HasLeaderboard'] = has_leaderboard_yearly[True]
    if False in has_leaderboard_yearly.columns: # Check if 'False' column exists
        has_leaderboard_yearly['Proportion_NoLeaderboard'] = has_leaderboard_yearly[False]
    else:
        has_leaderboard_yearly['Proportion_NoLeaderboard'] = 0 # No competitions without leaderboard
else: # If no True column, assume all are False or no data
    has_leaderboard_yearly['Proportion_HasLeaderboard'] = 0
    has_leaderboard_yearly['Proportion_NoLeaderboard'] = 1 # Assuming all are without leaderboard if no 'True' is found

has_leaderboard_yearly_melted = has_leaderboard_yearly.reset_index().melt(id_vars='Competition_Year', var_name='Leaderboard_Status', value_name='Proportion')

print("\n5. Proportion of Competitions with and without a Leaderboard per Year (sample last 5 years):")
print(has_leaderboard_yearly[['Proportion_HasLeaderboard', 'Proportion_NoLeaderboard']].tail())


fig = px.area(has_leaderboard_yearly_melted, x='Competition_Year', y='Proportion', color='Leaderboard_Status',
              title='5.4: Proportion of Competitions with and without a Public Leaderboard Over Time',
              labels={'Proportion': 'Proportion of Competitions', 'Competition_Year': 'Year', 'Leaderboard_Status': 'Leaderboard Status'},
              template='plotly_white', groupnorm='percent')
fig.update_layout(hovermode="x unified", legend_title_text='Leaderboard Status')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 4: 100% stacked area chart showing the proportion of competitions that have a public leaderboard vs. those that don't (e.g., code competitions, hidden test sets).")




# --- 3. Evaluation Metrics Evolution ---
print("\n--- Evaluation Metrics Evolution ---")

# Result 6: Top 10 most common EvaluationAlgorithmAbbreviations
top_eval_metrics = competitions_df['EvaluationAlgorithmAbbreviation'].value_counts().head(10)
print("\n6. Top 10 Most Common Evaluation Algorithm Abbreviations:")
print(top_eval_metrics)

# Result 7 & Visualization 5: Trend of top N EvaluationAlgorithmAbbreviations over time
top_n_eval_metrics = 5 # Number of top metrics to visualize
top_eval_metrics_list = competitions_df['EvaluationAlgorithmAbbreviation'].value_counts().nlargest(top_n_eval_metrics).index.tolist()
filtered_eval_metrics_df = competitions_df[competitions_df['EvaluationAlgorithmAbbreviation'].isin(top_eval_metrics_list)]

eval_metric_trends_yearly = filtered_eval_metrics_df.groupby(['Competition_Year', 'EvaluationAlgorithmAbbreviation']).size().unstack(fill_value=0)
eval_metric_trends_yearly_melted = eval_metric_trends_yearly.reset_index().melt(id_vars='Competition_Year', var_name='EvaluationAlgorithmAbbreviation', value_name='Count')
print(f"\n7. Yearly counts for Top {top_n_eval_metrics} Evaluation Algorithm Abbreviations (sample last 5 years):")
print(eval_metric_trends_yearly.tail())

fig = px.area(eval_metric_trends_yearly_melted, x='Competition_Year', y='Count', color='EvaluationAlgorithmAbbreviation',
              title=f'5.5: Evolution of Top {top_n_eval_metrics} Evaluation Metrics Over Time',
              labels={'Count': 'Number of Competitions', 'Competition_Year': 'Year', 'EvaluationAlgorithmAbbreviation': 'Metric'},
              template='plotly_white')
fig.update_layout(hovermode="x unified", legend_title_text='Evaluation Metric')
fig.update_xaxes(dtick=1)
fig.show()
print(f"Visualization 5: Stacked area chart showing the prevalence of the top {top_n_eval_metrics} evaluation metrics used in competitions over time. This highlights which metrics are gaining or losing favor in the ML community.")


# Result 8 & Visualization 6: Competition duration trends
# Assuming 'EnabledDate' and 'DeadlineDate' are available and in datetime format
if 'EnabledDate' in competitions_df.columns and 'DeadlineDate' in competitions_df.columns:
    competitions_df['Duration_Days'] = (competitions_df['DeadlineDate'] - competitions_df['EnabledDate']).dt.days
    avg_duration_yearly = competitions_df.groupby('Competition_Year')['Duration_Days'].mean().reset_index()
    print("\n8. Average Competition Duration (Days) Per Year (sample last 5 years):")
    print(avg_duration_yearly.tail())

    fig = px.line(avg_duration_yearly, x='Competition_Year', y='Duration_Days',
                  title='5.6: Average Competition Duration (Days) Over Time',
                  labels={'Duration_Days': 'Average Duration (Days)', 'Competition_Year': 'Year'},
                  markers=True, template='plotly_white')
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(dtick=1)
    fig.show()
    print("Visualization 6: Line plot illustrating the average duration of Kaggle competitions over time. This might reflect changes in problem complexity or platform strategy.")
else:
    print("\nWarning: 'EnabledDate' or 'DeadlineDate' not available. Skipping competition duration analysis.")


# Result 9 & Visualization 7: HasLeaderboard vs. HasKernels relationship
has_feat_relationship = competitions_df.groupby(['HasKernels', 'HasLeaderboard']).size().reset_index(name='Count')
has_feat_relationship['HasKernels_Label'] = has_feat_relationship['HasKernels'].map({True: 'Has Kernels', False: 'No Kernels'})
has_feat_relationship['HasLeaderboard_Label'] = has_feat_relationship['HasLeaderboard'].map({True: 'Has Leaderboard', False: 'No Leaderboard'})
print("\n9. Relationship between having Kernels and having a Leaderboard:")
print(has_feat_relationship)

fig = px.bar(has_feat_relationship, x='HasKernels_Label', y='Count', color='HasLeaderboard_Label',
             title='5.7: Competitions with Kernels vs. Competitions with Leaderboards',
             labels={'Count': 'Number of Competitions', 'HasKernels_Label': 'Has Kernels', 'HasLeaderboard_Label': 'Has Leaderboard'},
             template='plotly_white', barmode='group')
fig.show()
print("Visualization 7: Bar chart showing how many competitions have kernels, split by whether they also have a public leaderboard. This illustrates the interplay of these features.")


# Result 10 & Visualization 8: Correlation matrix of key numerical features
# Select relevant numerical columns for correlation
numerical_cols = ['TotalCompetitors', 'TotalTeams', 'TotalSubmissions', 'RewardQuantity_Numeric',
                  'TotalCompressedBytes', 'TotalUncompressedBytes', 'NumPrizes',
                  'LeaderboardPercentage', 'Duration_Days'] # Add Duration_Days if calculated

# Filter out NaNs for correlation calculation
corr_df = competitions_df[numerical_cols].dropna()
correlation_matrix = corr_df.corr()
print("\n10. Correlation Matrix of Key Numerical Features:")
print(correlation_matrix)

fig = px.imshow(correlation_matrix,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title='5.8: Correlation Matrix of Key Competition Features',
                labels=dict(color="Correlation"),
                height=700, width=700)
fig.update_xaxes(side="top")
fig.show()
print("Visualization 8: Heatmap showing the correlation between various numerical competition features. This helps identify strong relationships, e.g., between prize money and participation, or data size and submissions.")


# Result 11 & Visualization 9: Evolution of MaxDailySubmissions
# Fill NaN MaxDailySubmissions with a reasonable default (e.g., 5, common for older comps) or drop them
competitions_df['MaxDailySubmissions'].fillna(5, inplace=True) # Assuming 5 is a common default or a sensible fill value
avg_daily_submissions_yearly = competitions_df.groupby('Competition_Year')['MaxDailySubmissions'].mean().reset_index()
print("\n11. Average Max Daily Submissions Per Competition Per Year (sample last 5 years):")
print(avg_daily_submissions_yearly.tail())

fig = px.line(avg_daily_submissions_yearly, x='Competition_Year', y='MaxDailySubmissions',
              title='5.9: Average Max Daily Submissions Per Competition Over Time',
              labels={'MaxDailySubmissions': 'Average Max Daily Submissions', 'Competition_Year': 'Year'},
              markers=True, template='plotly_white')
fig.update_layout(hovermode="x unified")
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 9: Line plot showing the trend in the average maximum number of daily submissions allowed per competition. This indicates how frequently participants are allowed to iterate and submit.")

# Result 12 & Visualization 10: HasKernels vs. CompetitionTypeId
# To see if certain types of competitions are more likely to require/allow kernels
if 'CompetitionTypeId' in competitions_df.columns:
    kernel_by_comp_type = competitions_df.groupby('CompetitionTypeId')['HasKernels_int'].mean().sort_values(ascending=False).reset_index()
    kernel_by_comp_type.columns = ['CompetitionTypeId', 'Proportion_HasKernels']
    print("\n12. Proportion of Competitions with Kernels by Competition Type ID:")
    print(kernel_by_comp_type)

    fig = px.bar(kernel_by_comp_type, x='CompetitionTypeId', y='Proportion_HasKernels',
                 title='5.10: Proportion of Competitions with Kernels by Competition Type',
                 labels={'Proportion_HasKernels': 'Proportion Having Kernels', 'CompetitionTypeId': 'Competition Type ID'},
                 color='Proportion_HasKernels', color_continuous_scale=px.colors.sequential.Teal,
                 template='plotly_white')
    fig.update_xaxes(tickangle=45)
    fig.show()
    print("Visualization 10: Bar chart showing the proportion of competitions that have kernel support, grouped by Competition Type ID. This can highlight if certain competition formats are more suited or have adopted kernel-based workflows more readily.")
else:
    print("\nWarning: 'CompetitionTypeId' not available. Skipping kernel feature analysis by competition type.")


print("\n--- Part 5: Completed ---")
print("This section has provided insights into the evolution of Kaggle's platform features, particularly the increasing emphasis on kernel-based submissions and the dynamics of public/private leaderboards. We've also explored trends in evaluation metrics, competition duration, and the interplay of these features, all of which shape the competitive data science experience on Kaggle.")



print("--- Part 6: User Skill Progression and Community Dynamics ---")

# --- 1. User Participation Trends (Individual vs. Team) ---
print("\n--- User Participation Trends ---")

# Result 1 & Visualization 1: Ratio of TotalCompetitors to TotalTeams over time
# A higher ratio might indicate more individual participation or larger teams
competitions_df['Competitors_Per_Team'] = competitions_df['TotalCompetitors'] / competitions_df['TotalTeams']
# Handle cases where TotalTeams is 0 to avoid division by zero (fillna with 0 for safety)
competitions_df['Competitors_Per_Team'].replace([np.inf, -np.inf], np.nan, inplace=True)
competitions_df['Competitors_Per_Team'].fillna(0, inplace=True) # Replace NaN if TotalTeams was 0
avg_competitors_per_team_yearly = competitions_df.groupby('Competition_Year')['Competitors_Per_Team'].mean().reset_index()
print("\n1. Average Competitors Per Team Per Competition Per Year (sample last 5 years):")
print(avg_competitors_per_team_yearly.tail())

fig = px.line(avg_competitors_per_team_yearly, x='Competition_Year', y='Competitors_Per_Team',
              title='6.1: Average Competitors Per Team Per Competition Over Time',
              labels={'Competitors_Per_Team': 'Average Competitors Per Team', 'Competition_Year': 'Year'},
              markers=True, template='plotly_white')
fig.update_layout(hovermode="x unified")
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 1: Line plot showing the average number of competitors per team in competitions over time. This can indicate whether team sizes are growing or shrinking, reflecting collaboration patterns.")

# Result 2 & Visualization 2: Distribution of MaxTeamSize
# Fill NaN MaxTeamSize with a common default (e.g., 5 or a max of some competitions, or average)
# For this analysis, let's fill with mode or a sensible max
mode_max_team_size = competitions_df['MaxTeamSize'].mode()[0] if not competitions_df['MaxTeamSize'].mode().empty else 5
competitions_df['MaxTeamSize'].fillna(mode_max_team_size, inplace=True)
print("\n2. Distribution of Max Team Size (Overall):")
print(competitions_df['MaxTeamSize'].value_counts().sort_index())

fig = px.histogram(competitions_df, x='MaxTeamSize', nbins=10,
                   title='6.2: Distribution of Maximum Allowed Team Size per Competition',
                   labels={'MaxTeamSize': 'Maximum Team Size Allowed'},
                   template='plotly_white')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 2: Histogram showing the distribution of the maximum allowed team size in competitions. This indicates Kaggle's typical policies regarding team formation.")



# --- 2. Activity and Engagement over Time (Revisiting from user perspective) ---
print("\n--- Overall Activity and Engagement ---")

# Result 3 & Visualization 3: Total Unique Competitors and Teams (Approximate)
# Since we don't have UserIDs, we can approximate total unique participants by summing max of competitors/teams per year
# This is a rough proxy as users participate in multiple competitions.
# More accurate unique user analysis would require the Users.csv and Teams.csv datasets.
# For now, let's focus on the competition counts as a proxy for community activity.
total_competitors_yearly_sum = competitions_df.groupby('Competition_Year')['TotalCompetitors'].sum().reset_index()
total_teams_yearly_sum = competitions_df.groupby('Competition_Year')['TotalTeams'].sum().reset_index()
total_submissions_yearly_sum = competitions_df.groupby('Competition_Year')['TotalSubmissions'].sum().reset_index()

print("\n3. Total Competitors, Teams, and Submissions Aggregated Per Year (sample last 5 years):")
print(total_competitors_yearly_sum.merge(total_teams_yearly_sum, on='Competition_Year')
                                 .merge(total_submissions_yearly_sum, on='Competition_Year').tail())

fig = px.line(total_competitors_yearly_sum.merge(total_teams_yearly_sum, on='Competition_Year')
                                 .merge(total_submissions_yearly_sum, on='Competition_Year'),
              x='Competition_Year', y=['TotalCompetitors', 'TotalTeams', 'TotalSubmissions'],
              title='6.3: Aggregate Participation Metrics Over Time',
              labels={'value': 'Total Count', 'Competition_Year': 'Year', 'variable': 'Metric'},
              markers=True, template='plotly_white')
fig.update_layout(hovermode="x unified", legend_title_text='Metric')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 3: Line plot showing the sum of TotalCompetitors, TotalTeams, and TotalSubmissions across all competitions per year. This provides an overall view of the platform's activity growth.")




# --- 3. Success Metrics and Difficulty (Indirect) ---
print("\n--- Success Metrics and Difficulty (Indirect) ---")

# Result 4: Competitions with very high vs. very low participation
high_participation_threshold = competitions_df['TotalCompetitors'].quantile(0.95)
low_participation_threshold = competitions_df['TotalCompetitors'].quantile(0.05)
print(f"\n4. Number of Competitions with Very High (> {int(high_participation_threshold)} competitors) vs. Very Low (< {int(low_participation_threshold)} competitors) Participation:")
high_part_comps = competitions_df[competitions_df['TotalCompetitors'] >= high_participation_threshold]
low_part_comps = competitions_df[competitions_df['TotalCompetitors'] <= low_participation_threshold]
print(f"  High participation competitions: {len(high_part_comps)} (top 5%)")
print(f"  Low participation competitions: {len(low_part_comps)} (bottom 5%)")

# Result 5 & Visualization 4: Distribution of LeaderboardPercentage (as an indirect measure of 'difficulty' of winning due to smaller public test sets)
print("\n5. Distribution of Leaderboard Percentage:")
print(competitions_df['LeaderboardPercentage'].describe())

fig = px.histogram(competitions_df, x='LeaderboardPercentage', nbins=20,
                   title='6.4: Distribution of Public Leaderboard Percentage',
                   labels={'LeaderboardPercentage': 'Public Leaderboard Percentage'},
                   template='plotly_white')
fig.show()
print("Visualization 4: Histogram showing the distribution of the public leaderboard percentage. A smaller percentage implies a larger private test set, potentially increasing the challenge of avoiding overfitting.")




# --- 4. User Impact / Community Contributions (Inferred) ---
print("\n--- User Impact / Community Contributions (Inferred) ---")

# Result 6 & Visualization 5: Trend in Average TotalSubmissions per Competitor
# This can indicate how much effort on average each competitor is putting in
competitions_df['Submissions_Per_Competitor'] = competitions_df['TotalSubmissions'] / competitions_df['TotalCompetitors']
competitions_df['Submissions_Per_Competitor'].replace([np.inf, -np.inf], np.nan, inplace=True)
competitions_df['Submissions_Per_Competitor'].fillna(0, inplace=True) # Fill NaN where TotalCompetitors might be 0
avg_subs_per_comp_yearly = competitions_df.groupby('Competition_Year')['Submissions_Per_Competitor'].mean().reset_index()
print("\n6. Average Submissions Per Competitor Per Competition Per Year (sample last 5 years):")
print(avg_subs_per_comp_yearly.tail())

fig = px.line(avg_subs_per_comp_yearly, x='Competition_Year', y='Submissions_Per_Competitor',
              title='6.5: Average Submissions Per Competitor Per Competition Over Time',
              labels={'Submissions_Per_Competitor': 'Average Submissions Per Competitor', 'Competition_Year': 'Year'},
              markers=True, template='plotly_white')
fig.update_layout(hovermode="x unified")
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 5: Line plot showing the average number of submissions made per competitor per competition over time. This could indicate changes in iteration speed, model complexity, or submission strategies.")


# Result 7 & Visualization 6: Relationship between NumPrizes and TotalSubmissions (as a proxy for community value from prizes)
# Filter for competitions with prizes only
prizes_competitions = competitions_df[competitions_df['NumPrizes'] > 0].copy()
fig = px.scatter(prizes_competitions, x='NumPrizes', y='TotalSubmissions',
                 size='RewardQuantity_Numeric', color='Competition_Year',
                 hover_name='Title',
                 log_y=True, # Log scale for y-axis
                 title='6.6: Number of Prizes vs. Total Submissions (Size=Prize Money)',
                 labels={'NumPrizes': 'Number of Prizes', 'TotalSubmissions': 'Total Submissions (Log Scale)'},
                 template='plotly_white')
fig.update_layout(showlegend=True, legend_title_text='Competition Year')
fig.show()
print("Visualization 6: Scatter plot exploring if more prize tiers (NumPrizes) correlate with a higher number of submissions (indicating more iterative effort), with bubble size representing prize money.")


# Result 8 & Visualization 7: Host Segment Titles vs. Avg. Competitors per Team
# Re-identify top categories from the main DF
top_n_segments_p6 = 10
if 'HostSegmentTitle' in competitions_df.columns:
    top_host_segments_list_p6 = competitions_df['HostSegmentTitle'].value_counts().nlargest(top_n_segments_p6).index.tolist()
    filtered_competitions_df_p6 = competitions_df[competitions_df['HostSegmentTitle'].isin(top_host_segments_list_p6)].copy()

    avg_comp_per_team_by_category = filtered_competitions_df_p6.groupby('HostSegmentTitle')['Competitors_Per_Team'].mean().sort_values(ascending=False).reset_index()
    print(f"\n8. Average Competitors Per Team by Top {top_n_segments_p6} Host Segment Title:")
    print(avg_comp_per_team_by_category)

    fig = px.bar(avg_comp_per_team_by_category, x='HostSegmentTitle', y='Competitors_Per_Team',
                 title=f'6.7: Average Competitors Per Team by Top {top_n_segments_p6} Host Segment Title',
                 labels={'Competitors_Per_Team': 'Average Competitors Per Team', 'HostSegmentTitle': 'Host Segment Title'},
                 color='Competitors_Per_Team', color_continuous_scale=px.colors.sequential.Viridis,
                 template='plotly_white')
    fig.update_xaxes(tickangle=45)
    fig.show()
    print(f"Visualization 7: Bar chart showing the average number of competitors per team for different competition categories. This can highlight if certain domains encourage larger or smaller team collaborations.")
else:
    print("\nWarning: 'HostSegmentTitle' not available for category-specific team size analysis.")


# Result 9 & Visualization 8: Correlation between Duration_Days and TotalSubmissions
# If Duration_Days was calculated in Part 5, use it. Otherwise, calculate here.
if 'Duration_Days' not in competitions_df.columns and 'EnabledDate' in competitions_df.columns and 'DeadlineDate' in competitions_df.columns:
    competitions_df['Duration_Days'] = (competitions_df['DeadlineDate'] - competitions_df['EnabledDate']).dt.days

if 'Duration_Days' in competitions_df.columns:
    corr_duration_submissions = competitions_df[['Duration_Days', 'TotalSubmissions']].corr().loc['Duration_Days', 'TotalSubmissions']
    print(f"\n9. Correlation between Competition Duration (Days) and Total Submissions: {corr_duration_submissions:.2f}")

    fig = px.scatter(competitions_df.dropna(subset=['Duration_Days', 'TotalSubmissions']), x='Duration_Days', y='TotalSubmissions',
                     size='TotalCompetitors', color='Competition_Year',
                     hover_name='Title',
                     log_y=True,
                     title='6.8: Competition Duration vs. Total Submissions (Size=Total Competitors)',
                     labels={'Duration_Days': 'Competition Duration (Days)', 'TotalSubmissions': 'Total Submissions (Log Scale)'},
                     template='plotly_white')
    fig.update_layout(showlegend=True, legend_title_text='Competition Year')
    fig.show()
    print("Visualization 8: Scatter plot showing if longer competitions tend to receive more submissions, with bubble size representing the number of competitors. This helps understand the relationship between time and effort.")
else:
    print("\nWarning: 'Duration_Days' not available. Skipping correlation with TotalSubmissions.")


# Result 10 & Visualization 9: Average TotalSubmissions per competition type over time
if 'CompetitionTypeId' in competitions_df.columns:
    avg_submissions_by_type_yearly = competitions_df.groupby(['Competition_Year', 'CompetitionTypeId'])['TotalSubmissions'].mean().unstack(fill_value=0)
    avg_submissions_by_type_yearly_melted = avg_submissions_by_type_yearly.reset_index().melt(id_vars='Competition_Year', var_name='CompetitionTypeId', value_name='Average_Submissions')
    print("\n10. Average Total Submissions by Competition Type ID per Year (sample last 5 years):")
    print(avg_submissions_by_type_yearly.tail())

    fig = px.line(avg_submissions_by_type_yearly_melted, x='Competition_Year', y='Average_Submissions', color='CompetitionTypeId',
                  title='6.9: Average Total Submissions by Competition Type ID Over Time',
                  labels={'Average_Submissions': 'Average Total Submissions', 'Competition_Year': 'Year', 'CompetitionTypeId': 'Competition Type ID'},
                  markers=True, template='plotly_white')
    fig.update_layout(hovermode="x unified", legend_title_text='Competition Type ID')
    fig.update_xaxes(dtick=1)
    fig.show()
    print("Visualization 9: Line plot showing how the average number of submissions has evolved for different competition types, highlighting which formats encourage more iterative work.")
else:
    print("\nWarning: 'CompetitionTypeId' not available. Skipping average submissions by type.")


# Result 11: Top 10 competitions by TotalSubmissions
top_10_submissions = competitions_df.sort_values('TotalSubmissions', ascending=False).head(10)
print("\n11. Top 10 Competitions by Total Submissions:")
print(top_10_submissions[['Title', 'TotalSubmissions', 'Competition_Year', 'TotalCompetitors']])

# Result 12 & Visualization 10: Relationship between RewardQuantity and Submissions_Per_Competitor
# Filter for competitions with monetary prizes
monetary_comps = competitions_df[competitions_df['RewardQuantity_Numeric'] > 0].copy()
fig = px.scatter(monetary_comps.dropna(subset=['Submissions_Per_Competitor', 'RewardQuantity_Numeric']),
                 x='RewardQuantity_Numeric', y='Submissions_Per_Competitor',
                 color='Competition_Year', size='TotalCompetitors',
                 hover_name='Title',
                 log_x=True, log_y=True,
                 title='6.10: Prize Money vs. Submissions Per Competitor (Size=Total Competitors)',
                 labels={'RewardQuantity_Numeric': 'Prize Money (Log Scale)', 'Submissions_Per_Competitor': 'Submissions Per Competitor (Log Scale)'},
                 template='plotly_white')
fig.update_layout(showlegend=True, legend_title_text='Competition Year')
fig.show()
print("Visualization 10: Scatter plot exploring if higher prize money leads to more intense effort per competitor (higher submissions per competitor), with bubble size indicating the number of competitors.")


print("\n--- Part 6: Completed ---")
print("This final analytical section has delved into user dynamics and community engagement on Kaggle. We've explored trends in team formation, the overall activity on the platform, and indirect indicators of competition difficulty and user effort. These insights collectively contribute to a holistic understanding of the Kaggle ecosystem, from competition design to participant behavior.")



#  Overall Growth: Competitions & User Registrations Per Year ---
print("\n--- 1. Overall Growth: Competitions & User Registrations Per Year ---")
competitions_per_year = competitions_df['Competition_Year'].value_counts().sort_index().reset_index()
competitions_per_year.columns = ['Year', 'Num_Competitions']

if 'User_Register_Year' in df_users.columns:
    users_per_year = df_users['User_Register_Year'].value_counts().sort_index().reset_index()
    users_per_year.columns = ['Year', 'Num_Users']
    combined_growth = pd.merge(competitions_per_year, users_per_year, on='Year', how='outer').fillna(0)
    combined_growth_melted = combined_growth.melt(id_vars='Year', var_name='Metric', value_name='Count')

    fig = px.line(combined_growth_melted, x='Year', y='Count', color='Metric',
                  title='1. Key Trend: Annual Growth of Kaggle Competitions and New Users',
                  labels={'Count': 'Count', 'Year': 'Year', 'Metric': 'Growth Metric'},
                  markers=True, template='plotly_white')
    fig.update_traces(marker_symbol='circle', marker_size=8, line_width=2)
    fig.update_layout(hovermode="x unified")
    fig.show()
    print("Visualization 1: This plot clearly shows the parallel growth trajectories of new competitions and new user registrations, indicating the overall expansion of the Kaggle platform.")
else:
    print("Skipping combined user/competition growth plot due to missing 'User_Register_Year'.")
    fig = px.line(competitions_per_year, x='Year', y='Num_Competitions',
                  title='1. Key Trend: Number of Kaggle Competitions Launched Per Year',
                  labels={'Num_Competitions': 'Number of Competitions', 'Year': 'Year'},
                  markers=True)
    fig.update_traces(marker_symbol='circle', marker_size=8, line_width=2)
    fig.update_layout(hovermode="x unified")
    fig.show()
    print("Visualization 1 (Alternative): Line plot showing only the yearly trend of new competition launches.")




#  Domain Shift: Percentage Share of Top Host Segment Titles Over Time ---
print("\n--- 2. Domain Shift: Percentage Share of Top Host Segment Titles Over Time ---")
top_n_domains = 7 # Adjust as needed for clarity
top_host_segments_list = competitions_df['HostSegmentTitle'].value_counts().nlargest(top_n_domains).index.tolist()
filtered_competitions_df_domains = competitions_df[competitions_df['HostSegmentTitle'].isin(top_host_segments_list)].copy()

category_trends_yearly = filtered_competitions_df_domains.groupby(['Competition_Year', 'HostSegmentTitle']).size().unstack(fill_value=0)
category_trends_yearly_percentage = category_trends_yearly.apply(lambda x: x / x.sum(), axis=1).fillna(0)
category_trends_yearly_percentage_melted = category_trends_yearly_percentage.reset_index().melt(id_vars='Competition_Year', var_name='HostSegmentTitle', value_name='Percentage')

fig = px.area(category_trends_yearly_percentage_melted, x='Competition_Year', y='Percentage', color='HostSegmentTitle',
              title=f'2. Key Trend: Percentage Share of Top {top_n_domains} Competition Categories Over Time',
              labels={'Percentage': 'Percentage of Competitions', 'Competition_Year': 'Year', 'HostSegmentTitle': 'Category'},
              template='plotly_white', groupnorm='percent')
fig.update_layout(hovermode="x unified", legend_title_text='Competition Category')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 2: This 100% stacked area chart vividly illustrates the proportional shift in Kaggle's focus, highlighting the rise of domains like Computer Vision and NLP, reflecting the broader advancements in ML/AI.")



#  Data Scale: Average Uncompressed Dataset Size (GB) Per Competition Over Time ---
print("\n--- 3. Data Scale: Average Uncompressed Dataset Size (GB) Per Competition Over Time ---")
competitions_df_data_size = competitions_df.dropna(subset=['TotalUncompressedBytes']).copy()
competitions_df_data_size['Uncompressed_Size_GB'] = competitions_df_data_size['TotalUncompressedBytes'] / (1024**3)

avg_uncompressed_size_yearly = competitions_df_data_size.groupby('Competition_Year')['Uncompressed_Size_GB'].mean().reset_index()

fig = px.line(avg_uncompressed_size_yearly, x='Competition_Year', y='Uncompressed_Size_GB',
              title='3. Key Trend: Average Uncompressed Dataset Size (GB) Per Competition Over Time',
              labels={'Uncompressed_Size_GB': 'Average Uncompressed Size (GB)', 'Competition_Year': 'Year'},
              markers=True, template='plotly_dark')
fig.update_layout(hovermode="x unified")
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 3: This line plot demonstrates the increasing scale of data challenges on Kaggle, directly reflecting the growing computational demands and complexity in modern ML/AI problems.")



#  Platform Evolution: Proportion of Competitions with Kernel Features Over Time ---
print("\n--- 4. Platform Evolution: Proportion of Competitions with Kernel Features Over Time ---")
competitions_df['HasKernels_int'] = competitions_df['HasKernels'].astype(int)
competitions_df['OnlyAllowKernelSubmissions_int'] = competitions_df['OnlyAllowKernelSubmissions'].astype(int)

kernel_trends_yearly = competitions_df.groupby('Competition_Year')[['HasKernels_int', 'OnlyAllowKernelSubmissions_int']].mean().reset_index()
kernel_trends_yearly.columns = ['Competition_Year', 'Proportion_HasKernels', 'Proportion_OnlyAllowKernelSubmissions']
kernel_trends_yearly_melted = kernel_trends_yearly.melt(id_vars='Competition_Year', var_name='Feature', value_name='Proportion')

fig = px.line(kernel_trends_yearly_melted, x='Competition_Year', y='Proportion', color='Feature',
              title='4. Key Trend: Proportion of Competitions with Kernel Features Over Time',
              labels={'Proportion': 'Proportion', 'Competition_Year': 'Year', 'Feature': 'Kernel Feature'},
              markers=True, template='plotly_white')
fig.update_layout(hovermode="x unified", legend_title_text='Kernel Feature')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 4: This plot highlights Kaggle's strategic shift towards promoting reproducible research and cloud-based environments through the increasing adoption of kernel-based submissions.")



#  Engagement & Incentives: Total Prize Money Awarded Per Year ---
print("\n--- 5. Engagement & Incentives: Total Prize Money Awarded Per Year ---")
monetary_competitions = competitions_df[competitions_df['RewardQuantity_Numeric'] > 0].copy()
total_prize_money_yearly = monetary_competitions.groupby('Competition_Year')['RewardQuantity_Numeric'].sum().reset_index()

fig = px.bar(total_prize_money_yearly, x='Competition_Year', y='RewardQuantity_Numeric',
             title='5. Key Trend: Total Prize Money Awarded in Kaggle Competitions Per Year',
             labels={'RewardQuantity_Numeric': 'Total Prize Money (USD)', 'Competition_Year': 'Year'},
             color='RewardQuantity_Numeric', color_continuous_scale=px.colors.sequential.Plotly3,
             template='plotly_white')
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 5: This bar chart showcases the significant financial investment by Kaggle and its sponsors, indicating the growing value and stakes in the competitive data science landscape.")


#  Community Collaboration: Average Competitors Per Team Per Competition Over Time ---
print("\n--- 6. Community Collaboration: Average Competitors Per Team Per Competition Over Time ---")
competitions_df['Competitors_Per_Team'] = competitions_df['TotalCompetitors'] / competitions_df['TotalTeams']
competitions_df['Competitors_Per_Team'].replace([np.inf, -np.inf], np.nan, inplace=True)
competitions_df['Competitors_Per_Team'].fillna(0, inplace=True) # Fill NaN where TotalTeams might be 0

avg_competitors_per_team_yearly = competitions_df.groupby('Competition_Year')['Competitors_Per_Team'].mean().reset_index()

fig = px.line(avg_competitors_per_team_yearly, x='Competition_Year', y='Competitors_Per_Team',
              title='6. Key Trend: Average Competitors Per Team Per Competition Over Time',
              labels={'Competitors_Per_Team': 'Average Competitors Per Team', 'Competition_Year': 'Year'},
              markers=True, template='plotly_white')
fig.update_layout(hovermode="x unified")
fig.update_xaxes(dtick=1)
fig.show()
print("Visualization 6: This line plot reveals trends in team formation and collaboration, indicating whether the competitive environment is fostering larger or smaller collaborative groups over time.")

print("\n--- Consolidated Key Trends Visualizations Completed ---")
print("These visualizations provide a powerful narrative of Kaggle's evolution and its impact on the ML/AI landscape, covering growth, domain shifts, data scale, platform features, incentives, and collaboration.")










