# imports
import pandas as pd


# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load all the datasets
df_forest_area = pd.read_csv("/kaggle/input/forest-area-epochhackathon/Forest_Area_Final.csv")
df_gdp = pd.read_csv("/kaggle/input/gdp-epochhakcathon/gdp.csv")
df_palmer_index = pd.read_csv("/kaggle/input/palmerindex-epochhakcathon/palmer_index.csv")
df_merged_state_data = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/merged_state_data.csv")
df_weather = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/weather_monthly_state_aggregates.csv")
df_wildfire_sizes = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/wildfire_sizes_before_2010.csv")
df_zero_submission = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/zero_submission.csv")

# Check the dataframes loaded
print(df_forest_area.head())
print(df_gdp.head())
print(df_palmer_index.head())
print(df_merged_state_data.head())
print(df_weather.head())
print(df_wildfire_sizes.head())
print(df_zero_submission.head())



# Check for missing values in all datasets
dfs = [df_forest_area, df_gdp, df_palmer_index, df_merged_state_data, df_weather, df_wildfire_sizes, df_zero_submission]
for i, df in enumerate(dfs):
    print(f"Dataset {i+1} Missing Values:")
    print(df.isnull().sum())
    print("\n")



# Print column names, first few rows, and data types of all datasets
datasets = [df_forest_area, df_gdp, df_palmer_index, df_merged_state_data, df_weather, df_wildfire_sizes, df_zero_submission]
dataset_names = [
    "Forest Area", "GDP", "Palmer Index", "Merged State Data", "Weather", "Wildfire Sizes", "Zero Submission"
]

for i, df in enumerate(datasets):
    print(f"Dataset: {dataset_names[i]}")
    print("Columns:")
    print(df.columns)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
    print("="*80)



# Plotting total forest area per state
plt.figure(figsize=(12, 8))
sns.barplot(x='State', y='Forest_Area', data=df_forest_area, palette='viridis')
plt.title("Total Forest Area by State")
plt.xticks(rotation=90)
plt.xlabel("State")
plt.ylabel("Forest Area (sq mi)")
plt.show()



# Box plot to compare GDP across states
plt.figure(figsize=(12, 8))
sns.boxplot(x='state', y='GDP', data=df_gdp, palette='coolwarm')
plt.title("GDP Distribution by State")
plt.xticks(rotation=90)
plt.xlabel("State")
plt.ylabel("GDP ($ Million)")
plt.show()



# Let's take a look at monthly temperature trends for a specific state (e.g., California)
state_weather = df_weather[df_weather['State'] == 'CA']

plt.figure(figsize=(12, 6))
sns.lineplot(x='year_month', y='TMAX', data=state_weather)
plt.title("Monthly Maximum Temperature Trend for California (1992-2015)")
plt.xticks(rotation=45)
plt.xlabel("Year-Month")
plt.ylabel("Max Temperature (°F)")
plt.show()



# # Merge the Palmer Index and Wildfire Sizes data on state and month columns
# df_palmer_wildfire = pd.merge(df_palmer_index, df_wildfire_sizes, left_on=['state_abbrev', 'year'], right_on=['STATE', 'month'])

# # Scatter plot of Palmer Drought Index vs. fire size
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='jan', y='total_fire_size', data=df_palmer_wildfire, color='r')
# plt.title("Correlation between Palmer Drought Index (January) and Total Fire Size")
# plt.xlabel("Palmer Drought Index (Jan)")
# plt.ylabel("Total Fire Size (sq mi)")
# plt.show()



# # Merge forest area and wildfire size data on state
# df_forest_fire = pd.merge(df_forest_area, df_wildfire_sizes, left_on='State', right_on='STATE')

# # Calculate correlation matrix
# corr = df_forest_fire[['Forest_Area', 'total_fire_size']].corr()

# # Plot the correlation matrix as a heatmap
# plt.figure(figsize=(6, 4))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap: Forest Area vs Total Fire Size")
# plt.show()



# # Violin plot for Fire Size Distribution in Different Forest Area Ranges
# plt.figure(figsize=(12, 6))

# # Create bins for the forest area to group them into ranges
# forest_area_bins = pd.cut(df_forest_fire['Forest_Area'], bins=10)

# # Violin plot by the forest area bins
# sns.violinplot(x=forest_area_bins, y='total_fire_size', data=df_forest_fire, palette='viridis')
# plt.title("Violin Plot: Fire Size Distribution by Forest Area Ranges")
# plt.xlabel("Forest Area Ranges (sq mi)")
# plt.ylabel("Total Fire Size (sq mi)")
# plt.xticks(rotation=45)
# plt.show()






# Merge forest area and wildfire size data on state without modifying original datasets
df_forest_fire = pd.merge(df_forest_area[['State', 'Forest_Area']], 
                           df_wildfire_sizes[['STATE', 'month', 'total_fire_size']], 
                           left_on='State', right_on='STATE', how='inner')

# Calculate correlation between forest area and total fire size
correlation = df_forest_fire[['Forest_Area', 'total_fire_size']].corr()

# Print correlation coefficient
print("Correlation between Forest Area and Total Fire Size:")
print(correlation)

# Scatter plot of Forest Area vs. Total Fire Size
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Forest_Area', y='total_fire_size', data=df_forest_fire, color='b')
plt.title("Correlation between Forest Area and Total Fire Size")
plt.xlabel("Forest Area (sq mi)")
plt.ylabel("Total Fire Size (sq mi)")
plt.show()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: Forest Area vs Total Fire Size")
plt.show()



# Distribution of total fire sizes
plt.figure(figsize=(10, 6))
sns.histplot(df_wildfire_sizes['total_fire_size'], kde=True, bins=200, color='skyblue')
plt.title("Distribution of Total Fire Size")
plt.xlabel("Total Fire Size (sq mi)")
plt.ylabel("Frequency")
plt.show()



# #Set the device to GPU if available, otherwise use CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using:", device)


# #Empty cache files
# torch.cuda.empty_cache()


import matplotlib.pyplot as plt

# Load the training data
df = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/wildfire_sizes_before_2010.csv")

plt.figure(figsize=(10, 6))
plt.hist(np.log10(df['total_fire_size']), bins=200, edgecolor='black')
plt.xlabel("Total Fire Size (log scale)")
plt.ylabel("Frequency")
plt.title("Histogram of Forest Fires (Log Scale)")
plt.show()


# get the predictions for 2010
last_year = df[df['month'].str[:4] == '2010']

last_year.head()


import pandas as pd

# Load datasets
weather_path = "/kaggle/input/forest-fire-prediction-epoch-hackathon/weather_monthly_state_aggregates.csv"
palmer_path = "/kaggle/input/palmerindex-epochhakcathon/palmer_index.csv"

weather_df = pd.read_csv(weather_path)
palmer_df = pd.read_csv(palmer_path)

# Standardize the state abbreviation columns
weather_df['State'] = weather_df['State'].str.strip().str.upper()
palmer_df['state_abbrev'] = palmer_df['state_abbrev'].str.strip().str.upper()

# Define the month columns that need to be melted
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# Melt the Palmer index data to convert it from wide to long format
palmer_long = palmer_df.melt(id_vars=['state_abbrev', 'year'],
                             value_vars=months,
                             var_name='month',
                             value_name='palmer_value')

# Map month abbreviations to two-digit numbers
month_map = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
}
palmer_long['month_num'] = palmer_long['month'].map(month_map)
palmer_long['year_month'] = palmer_long['year'].astype(str) + '-' + palmer_long['month_num']

# (Optional) Debug: Check matching keys for a particular state, e.g., Alaska
print("Weather year_month for AK:", weather_df[weather_df['State'] == 'AK']['year_month'].unique())
print("Palmer year_month for AK:", palmer_long[palmer_long['state_abbrev'] == 'AK']['year_month'].unique())

# Merge the weather data with the Palmer index data on state and year_month
merged_df = weather_df.merge(
    palmer_long[['state_abbrev', 'year_month', 'palmer_value']],
    left_on=['State', 'year_month'],
    right_on=['state_abbrev', 'year_month'],
    how='left'
)

# Optionally, drop the extra state column from palmer_long if no longer needed
merged_df.drop(columns=['state_abbrev'], inplace=True)

merged_df.fillna(0, inplace=True)

input_df = merged_df.copy()
input_df.head()


import pandas as pd
import numpy as np

# Load the data (assumed to span 1992 to 2010)
df = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/wildfire_sizes_before_2010.csv")
print(df.head())

# Extract all unique years from the data based on the 'month' column
# (Assuming 'month' is in a format like 'YYYY-MM')
all_years = sorted(df['month'].str[:4].unique())  # e.g., ['1992', '1993', ..., '2010']

# Create a dictionary to store a dataframe for each year.
# For each year, we filter the data, shorten the month to just the two-digit portion,
# and rename the 'total_fire_size' column to indicate the year.
dfs_by_year = {}
for year in all_years:
    df_year = df[df['month'].str.startswith(year)].copy()
    # Keep only the two-digit month (e.g., "01", "12")
    df_year['month'] = df_year['month'].str[-2:]
    df_year = df_year.rename(columns={'total_fire_size': f'{year}_area'})
    # Only keep the relevant columns for merging
    dfs_by_year[year] = df_year[['STATE', 'month', f'{year}_area']]

# Merge the yearly dataframes on 'STATE' and 'month' using an outer join
merged = None
for year in all_years:
    if merged is None:
        merged = dfs_by_year[year]
    else:
        merged = pd.merge(merged, dfs_by_year[year], on=['STATE', 'month'], how='outer')

# Do not fill NaNs with zeros
merged = merged.sort_values(by='month').reset_index(drop=True)

# ----- Forecast for 2011 using exponentially weighted geometric mean -----

# Define the target forecast year
forecast_year = 2011

# Compute weights for each training year based on how old the data is relative to the forecast year.
# Weight for a given year = 0.5^((forecast_year - data_year) / 3)
weights = {}
for year in all_years:
    age = forecast_year - int(year)
    weights[year] = np.power(0.5, age / 6)

# Define a function to compute the weighted geometric mean for a row,
# ignoring NaN values (i.e., only using the years with data)
def compute_weighted_geometric_mean(row):
    log_sum = 0.0
    effective_weight = 0.0
    for year in all_years:
        col = f'{year}_area'
        if pd.notna(row[col]):
            weight = weights[year]
            log_sum += np.log(row[col] + 0.1) * weight
            effective_weight += weight
    if effective_weight == 0:
        return np.nan  # or set to a default value if desired
    else:
        return np.exp(log_sum / effective_weight) - 0.1

# Compute the weighted geometric mean forecast for 2011 on a row-by-row basis
merged['p2011'] = merged.apply(compute_weighted_geometric_mean, axis=1)

# ----- Forecast subsequent years (2012 to 2015) by applying a growth factor -----
growth_factor = 1.05
for year in range(2012, 2016):
    prev_year = year - 1
    merged[f'p{year}'] = merged[f'p{prev_year}'] * growth_factor

print(merged)

# ----- Build the submission DataFrame -----
submission_rows = []
forecast_years = range(2011, 2016)

# For each forecast year, state, and month, sum up the predictions.
for year in forecast_years:
    col = f'p{year}'
    for state in merged['STATE'].unique():
        for month in merged['month'].unique():
            total_fire_size = merged[(merged['STATE'] == state) &
                                       (merged['month'] == month)][col].sum()
            # Ensure a minimum value of 0.1 for each submission row
            submission_rows.append({
                "STATE": state,
                "month": f"{year}-{month}",
                "total_fire_size": max(total_fire_size, 0.1)
            })

submission = pd.DataFrame(submission_rows)
print(submission)



# Add the ID column required by Kaggle
submission['ID'] = range(len(submission))

# Order columns as needed
submission = submission[['ID', 'STATE', 'month', 'total_fire_size']]
submission.to_csv('submission.csv', index=False)

submission.head()




