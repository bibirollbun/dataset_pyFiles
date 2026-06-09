# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation and analysis
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import seaborn as sns

# Display data in Jupyter notebooks
from IPython.display import display

# Machine learning libraries
import scipy as sp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv',index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv',index_col='id')
submission_data = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
original_data = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')

#verify shapes
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)
print("Original Data Shape:", original_data.shape)


print("Train Data Preview:")
display(train_data.tail())

print("\nTest Data Preview")
display(test_data.head())


print("\nOriginal Data Preview:")
display(original_data.tail())


# Display information about the DataFrames
print("\nTrain Data Info:")
train_data.info()

print("\nOriginal Data Info:")
original_data.info()

print("\nTest Data Info:")
test_data.info()


# Fill missing value in 'winddirection' column using linear interpolation
test_data["winddirection"] = test_data["winddirection"].interpolate(method='linear')

# Check if there are any missing values after interpolation
print("Missing values after interpolation:", test_data["winddirection"].isnull().sum())


# Remove spaces from column names
original_data.columns = original_data.columns.str.strip()
print("\nUpdated Column Names")
print(original_data.columns)


# Correct Spelling Inconsistencies
train_data = train_data.rename(columns={'temparature':'temperature'})
test_data = test_data.rename(columns={'temparature':'temperature'})
original_data = original_data.rename(columns={'temparature':'temperature'})


# Reorder columns in original_data to match train_data
original_data = original_data.reindex(columns=train_data.columns)
print("Original Data Columns After Reordering:")
print(original_data.columns)


# Descriptive statistics for numerical columns
print("\nTrain Data Describe:")
display(train_data.describe().T.style.background_gradient(cmap='Greens'))

print("\nTest Data Describe:")
display(test_data.describe().T.style.background_gradient(cmap='Greens'))

print("\nOriginal Data Describe:")
display(original_data.describe().T.style.background_gradient(cmap='Greens'))


#Check for duplicated rows
print("\nDuplicate Rows in Train Data:", train_data.duplicated().sum())
print("\nDuplicate Rows in Original Data:", original_data.duplicated().sum())
print("\nDuplicate Rows in Test Data:", test_data.duplicated().sum())


numerical_variables = ['day','pressure', 'maxtemp', 'temperature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall' 
categorical_variables = ['winddirection']


custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'
original_data['Dataset'] = 'Original'

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train_data, test_data,original_data.dropna()]), x=variable, y="Dataset", palette=custom_palette)
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test_data, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    sns.histplot(data=original_data.dropna(), x=variable, color=custom_palette[2], kde=True, bins=30, label="Original")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN, TEST & ORIGINAL]")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each variable
for variable in numerical_variables:
    create_variable_plots(variable)

# Drop the 'Dataset' column after analysis
train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)
original_data.drop('Dataset', axis=1, inplace=True)


# Define custom color palette for Train, Test, and Original datasets
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

# Function to create Wind Rose plot in a subplot
def create_wind_rose(ax, data, dataset_name, color):
    # Convert wind direction to radians
    wind_direction_radians = np.radians(data['winddirection'].dropna())

    # Create histogram bins (every 10Â°)
    bins = np.linspace(0, 2*np.pi, 37)  # 36 bins (every 10Â°)
    counts, bin_edges = np.histogram(wind_direction_radians, bins=bins)

    # Plot on the polar axis with improved style
    bars = ax.bar(bin_edges[:-1], counts, width=np.radians(10), color=color, edgecolor='black', alpha=0.8)

    # Formatting for professional appearance
    ax.set_theta_zero_location("N")  # North is at 0Â°
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))  # Tick labels every 45Â°
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10, fontweight='bold')

    # Add grid and labels for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_yticklabels([])  # Remove radial labels to avoid clutter
    ax.set_title(f"Wind Direction ({dataset_name})", fontsize=12, fontweight='bold', pad=10)

# Create a single row with three wind rose plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'})

# Generate wind rose plots for Train, Test, and Original datasets
create_wind_rose(axes[0], train_data, "Train Data", custom_palette[0])  # Blue
create_wind_rose(axes[1], test_data, "Test Data", custom_palette[1])    # Red
create_wind_rose(axes[2], original_data.dropna(), "Original Data", custom_palette[2])  # Green

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


print("Before Conversion:")
print(original_data['rainfall'].value_counts())  # ë³€í™˜ ì „ ê°œìˆ˜ í™•ì�¸



# 'rainfall' ê°’ì�´ ë¬¸ì��ì—´ì�¸ì§€ í™•ì�¸í•˜ê³ , ê³µë°± ë°� ëŒ€ì†Œë¬¸ì�� ì •ë¦¬
original_data['rainfall'] = original_data['rainfall'].astype(str).str.lower().str.strip()

# 'yes'ëŠ” 1, 'no'ëŠ” 0ìœ¼ë¡œ ë³€í™˜
original_data['rainfall'] = (original_data['rainfall'] == 'yes').astype(int)

# ë³€í™˜ í›„ ê°’ í™•ì�¸
print("\nAfter Conversion:")
print(original_data['rainfall'].value_counts())


# Set target variable
target_variable = 'rainfall'


# Create a figure with 2 subplots (1 row, 2 columns) for train_data
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1ï¸�âƒ£ Bar Chart (Horizontal) for rainfall distribution in train_data
sns.countplot(y=train_data[target_variable], palette="Blues", ax=axes[0])
axes[0].set_title("Rainfall Distribution in Train Data", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Count", fontsize=12)
axes[0].set_ylabel("Rainfall", fontsize=12)

# 2ï¸�âƒ£ Pie Chart for rainfall proportion in train_data
train_rainfall_counts = train_data[target_variable].value_counts()
axes[1].pie(train_rainfall_counts, labels=train_rainfall_counts.index, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90)
axes[1].set_title("Rainfall Proportion in Train Data", fontsize=14, fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.show()

# Create a figure with 2 subplots (1 row, 2 columns) for original_data
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3ï¸�âƒ£ Bar Chart (Horizontal) for rainfall distribution in original_data
sns.countplot(y=original_data[target_variable], palette="Blues", ax=axes[0])
axes[0].set_title("Rainfall Distribution in Original Data", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Count", fontsize=12)
axes[0].set_ylabel("Rainfall", fontsize=12)

# 4ï¸�âƒ£ Pie Chart for rainfall proportion in original_data
original_rainfall_counts = original_data[target_variable].value_counts()
axes[1].pie(original_rainfall_counts, labels=original_rainfall_counts.index, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90)
axes[1].set_title("Rainfall Proportion in Original Data", fontsize=14, fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.show()



#Heatmap
variables = [col for col in train_data.columns if col in numerical_variables]

# Adding variables to the existing list
test_variables = variables
train_variables = variables+ ['rainfall']

# Calculate correlation matrices for train_data and test_data
corr_train = train_data[train_variables].corr()
corr_test = test_data[test_variables].corr()

# Create masks for the upper triangle
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

# Set the text size and rotation
annot_kws = {"size": 8, "rotation": 45}

# Generate heatmaps for train_data
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
ax_train = sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
                      square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data')

# Generate heatmaps for test_data
plt.subplot(1, 2, 2)
ax_test = sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
                     square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# ê²°ê³¼


# 'day' ì»¬ëŸ¼ì—�ì„œ íŒ¨í„´ ë¶„ì„�: 'day' ì»¬ëŸ¼ì—� ì–´ë–¤ ê°’ì�´ ì�ˆëŠ”ê°€?
print("Unique Values in 'day' Column (Sorted) - Train Data:")
print(sorted(train_data['day'].unique()))

print("\nUnique Values in 'day' Column (Sorted) - Test Data:")
print(sorted(test_data['day'].unique()))

print("\nUnique Values in 'day' Column (Sorted) - Original Data:")
print(sorted(original_data['day'].unique()))

# ê°™ì�€ 'day'ê°’ì�´ ì—¬ëŸ¬ ë²ˆ ë‚˜íƒ€ë‚˜ëŠ”ê°€?
print("\nCount of Each Unique 'day' Value - Train Data:")
print(train_data['day'].value_counts().sort_index())

print("\nCount of Each Unique 'day' Value - Test Data:")
print(test_data['day'].value_counts().sort_index())

print("\nCount of Each Unique 'day' Value - Original Data:")
print(original_data['day'].value_counts().sort_index())

# 'day'ê°’ ê¸°ì¤€ìœ¼ë¡œ ì—°ë�„ ì¶”ì •
train_data['year'] = (train_data.index // 365) + 1
print("\nTrain Data - Sample Year Assignment:")
print(train_data[['day', 'year']].head(10))

test_data['year'] = (test_data.index // 365) + 1
print("\nTest Data - Sample Year Assignment:")
print(test_data[['day', 'year']].head(10))

original_data['year'] = (original_data.index // 365) + 1
print("\nOriginal Data - Sample Year Assignment:")
print(original_data[['day', 'year']].head(10))



# drop'year' column
train_data = train_data.drop(columns=['year'])
test_data = test_data.drop(columns=['year'])
original_data = original_data.drop(columns=['year'])


# ê°� ë�°ì�´í„°í”„ë ˆì�„ì�˜ ì»¬ëŸ¼ ëª©ë¡� í™•ì�¸
print("Columns in Train Data:")
print(train_data.columns.tolist())

print("\nColumns in Test Data:")
print(test_data.columns.tolist())

print("\nColumns in Original Data:")
print(original_data.columns.tolist())



# ì�¸ë�±ìŠ¤ë¥¼ 'id' ì»¬ëŸ¼ìœ¼ë¡œ ì¶”ê°€
train_data = train_data.reset_index(names='id')
test_data = test_data.reset_index(names='id')
original_data = original_data.reset_index(names='id')


import matplotlib.pyplot as plt

# Define colors for Train and Test data
train_color = '#3498db'  # Blue
test_color = '#e74c3c'   # Red

# Create the plot
plt.figure(figsize=(12, 5))

# Plot Train Data (Using 'id' column)
plt.plot(train_data['id'], train_data['day'], linestyle='-', color=train_color, label='Train Data', alpha=0.7)

# Plot Test Data (Using 'id' column)
plt.plot(test_data['id'], test_data['day'], linestyle='-', color=test_color, label='Test Data', alpha=0.7)

# Formatting
plt.xlabel('ID')  # ì�´ì œ Xì¶•ì�„ 'ID'ë¡œ ì„¤ì •
plt.ylabel('Day')
plt.title('Trend Plot: Day vs ID')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Show plot
plt.show()



# Generate the expected repeating pattern (1-365 for 6 years)
expected_pattern = np.tile(np.arange(1, 366), 6)  # Repeats 1-365 exactly 6 times

# Check for incorrect labels
train_data['expected_day'] = expected_pattern[:len(train_data)]  # Assign expected pattern
train_data['day_mismatch'] = train_data['day'] != train_data['expected_day']  # Flag mismatches


flag_color = '#8B0000'   # Dark Red (for mismatched days)

# Generate expected repeating pattern (1-365 for 6 years)
expected_pattern = np.tile(np.arange(1, 366), 6)  # Repeats 1-365 exactly 6 times

# Assign expected pattern and flag mismatches
train_data['expected_day'] = expected_pattern[:len(train_data)]
train_data['day_mismatch'] = train_data['day'] != train_data['expected_day']  # Boolean flag

# Create the plot
plt.figure(figsize=(12, 5))

# Plot Train Data
plt.plot(train_data['id'], train_data['day'], linestyle='-', color=train_color, label='Train Data', alpha=0.7)

# Plot Test Data
plt.plot(test_data['id'], test_data['day'], linestyle='-', color=test_color, label='Test Data', alpha=0.7)

# Flag mismatched days using red markers
plt.scatter(
    train_data.loc[train_data['day_mismatch'], 'id'],  # X-axis: IDs of mismatched days
    train_data.loc[train_data['day_mismatch'], 'day'], # Y-axis: Corresponding incorrect days
    color=flag_color, marker='X', s=80, label='Mismatched Days', alpha=0.9
)

# Formatting
plt.xlabel('ID')
plt.ylabel('Day')
plt.title('Trend Plot: Day vs ID (Flagging Mismatches)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Show plot
plt.show()


train_data['day'] = train_data['expected_day']

# Get the last day value from train data
last_train_day = train_data['day'].iloc[-1]

# Generate sequential day numbers for the test dataset
test_data['day'] = np.arange(last_train_day + 1, last_train_day + 1 + len(test_data))

train_data.drop(columns=['expected_day', 'day_mismatch'], errors='ignore', inplace=True)  # Drop 'expected_day' if it exists


train_data.columns


# Define colors
train_color = '#3498db'  # Blue
test_color = '#e74c3c'   # Red
rainfall_colors = {0: '#f1c40f', 1: '#2980b9'}  # Dark Yellow (no rainfall), Blue (rainfall)

# Numerical columns to plot
numerical_columns = test_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in ['id', 'day', 'rainfall']:
    if col in numerical_columns:
        numerical_columns.remove(col)

# Plotting loop for each numerical variable
for column in numerical_columns:
    # Create figure with specific layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # ---- Trend Plot (ID vs Variable) ----
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(train_data['id'], train_data[column], linestyle='-', color=train_color, label='Train Data', alpha=0.7)
    ax0.plot(test_data['id'], test_data[column], linestyle='-', color=test_color, label='Test Data', alpha=0.7)

    ax0.set_xlabel('ID', fontsize=14)
    ax0.set_ylabel(column, fontsize=14)
    ax0.set_title(f'Trend Plot: {column} vs ID', fontsize=16, fontweight='bold')  # âœ… Fix applied
    ax0.legend(fontsize=12)
    ax0.grid(True, linestyle='--', alpha=0.5)

    # ---- Scatter Plot (Day vs Variable) ----
    ax1 = fig.add_subplot(gs[1, 0])
    scatter = ax1.scatter(
        train_data['day'], train_data[column],
        c=train_data['rainfall'].map(rainfall_colors), alpha=0.7
    )
    ax1.set_xlabel('Day', fontsize=14)
    ax1.set_ylabel(column, fontsize=14)
    ax1.set_title(f'Scatter Plot: {column} vs Day (by Rainfall)', fontsize=16, fontweight='bold')  # âœ… Fix applied

    # Custom legend for rainfall
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='No Rainfall',
               markersize=10, markerfacecolor=rainfall_colors[0]),
        Line2D([0], [0], marker='o', color='w', label='Rainfall',
               markersize=10, markerfacecolor=rainfall_colors[1])
    ]
    ax1.legend(handles=legend_elements, title="Rainfall", fontsize=12, title_fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---- KDE Plot (Variable distribution by Rainfall) ----
    ax2 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(data=train_data, x=column, hue='rainfall', palette=rainfall_colors, ax=ax2, fill=True, common_norm=False, alpha=0.6)

    ax2.set_xlabel(column, fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.set_title(f'Distribution (KDE) of {column} by Rainfall', fontsize=16, fontweight='bold')  
    ax2.legend(title='Rainfall', fontsize=12, title_fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout spacing
    plt.tight_layout(pad=3.0)
    plt.show()

    # ---- Add clear separation after each variable ----
    plt.figure(figsize=(16, 0.3))  # Adjust spacing
    plt.axhline(y=0, color='gray', linewidth=5, linestyle='-') 
    plt.axis('off')
    plt.show()


# Define function to categorize wind direction into sectors
def wind_sector(direction):
    if pd.isna(direction):
        return np.nan  # Preserve missing values for later handling
    direction = float(direction)
    if direction >= 315 or direction < 45:
        return 'North'
    elif direction >= 45 and direction < 135:
        return 'East'
    elif direction >= 135 and direction < 225:
        return 'South'
    else:
        return 'West'

def perform_feature_engineering(df):
    """
    Applies feature engineering to the dataframe, creating new features for weather prediction.
    """

    # 1. Seasonal Features using 'day' (cyclical representation of the year)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    # 2. Lagged Features (previous day's values for key predictors)
    #    Shift by 1, then fill any remaining NaNs with 0 (or a median if desired).
    df['cloud_lag1'] = df['cloud'].shift(1).fillna(0)
    df['sunshine_lag1'] = df['sunshine'].shift(1).fillna(0)
    df['humidity_lag1'] = df['humidity'].shift(1).fillna(0)

    # 3. Rolling Statistics (3-day trends for key predictors)
    #    Use rolling(window=3, min_periods=1) so the first 1-2 rows won't be NaN. Fill with bfill if still needed.
    df['cloud_roll3_mean'] = (df['cloud']
                              .rolling(window=3, min_periods=1)
                              .mean()
                              .fillna(method='bfill'))
    df['sunshine_roll3_mean'] = (df['sunshine']
                                 .rolling(window=3, min_periods=1)
                                 .mean()
                                 .fillna(method='bfill'))
    df['humidity_roll3_mean'] = (df['humidity']
                                 .rolling(window=3, min_periods=1)
                                 .mean()
                                 .fillna(method='bfill'))

    # 4. Interaction Features (combinations of highly correlated features)
    df['cloud_humidity'] = (df['cloud'] * df['humidity']).fillna(0)  # Replace missing with 0
    df['sunshine_cloud_ratio'] = (df['sunshine'] / (df['cloud'] + 1e-5)).fillna(0)

    # 5. Meteorological Features
    #    Use median fill for safety, or 0 if that makes more sense in your domain
    df['temp_range'] = (df['maxtemp'] - df['mintemp']).fillna(df['maxtemp'].median())
    df['pressure_diff'] = df['pressure'].diff().fillna(0)

    # 6. Additional Time-Based Interactions with 'day'
    df['cloud_day_sin'] = (df['cloud'] * df['day_sin']).fillna(0)
    df['sunshine_day_cos'] = (df['sunshine'] * df['day_cos']).fillna(0)
    df['humidity_roll3_day_sin'] = (df['humidity_roll3_mean'] * df['day_sin']).fillna(0)

    # 7. Categorical Feature
    #    Map wind direction to bins and replace missing with 'Unknown'
    df['wind_sector'] = df['winddirection'].apply(wind_sector).fillna('Unknown')

    return df
    
# Apply feature engineering to both train and test data
# Note: For lagged/rolling features to work correctly across train/test boundary,
# concatenate train and test, apply this function, then split back in the main script:
id_test = test_data['id']
full_data = pd.concat([train_data, test_data], axis=0).sort_values('id')
full_data = perform_feature_engineering(full_data)
train_data = full_data[full_data['rainfall'].notna()]
test_data = full_data[full_data['rainfall'].isna()]

# Numeric new features
newly_created_vars = [
    'day_sin',              # Sine of day for seasonality
    'day_cos',              # Cosine of day for seasonality
    'cloud_lag1',           # Previous day's cloud cover
    'sunshine_lag1',        # Previous day's sunshine duration
    'humidity_lag1',        # Previous day's humidity
    'cloud_roll3_mean',     # 3-day average cloud cover
    'sunshine_roll3_mean',  # 3-day average sunshine duration
    'humidity_roll3_mean',  # 3-day average humidity
    'cloud_humidity',       # Interaction of cloud and humidity
    'sunshine_cloud_ratio', # Ratio of sunshine to cloud cover
    'temp_range',           # Daily temperature range
    'pressure_diff',        # Change in pressure from previous day
    'cloud_day_sin',        # Cloud cover with seasonal sine component
    'sunshine_day_cos',     # Sunshine with seasonal cosine component
    'humidity_roll3_day_sin', # 3-day humidity trend with seasonal sine
]

# Categorical new features
categorical_new_feats = [
    'wind_sector'           # Wind direction binned into North, East, South, West
]


import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation matrix only for newly created features
corr_train = train_data[newly_created_vars + ['rainfall']].corr()[['rainfall']]

# Heatmap visualization without color bar, displaying values vertically
plt.figure(figsize=(10, 2))
ax = sns.heatmap(
    corr_train.T,  # Transposing so features are on x-axis
    annot=True, 
    cmap='viridis', 
    linewidths=0.5, 
    cbar=False, 
    fmt=".2f", 
    annot_kws={"rotation": 90}  # Rotate annotations to be vertical
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title('Correlation Heatmap - New Engineered Features vs Rainfall')
plt.show()


# Define common list of columns to drop
columns_to_drop = [
    'day_cos',
    'pressure_diff',
    'cloud_day_sin',
    'sunshine_day_cos',
    'humidity_roll3_day_sin'
]

# Drop columns from both train and test data
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop+['rainfall'], inplace=True)


# Identify numerical variables
columns_to_check = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove unwanted variables
columns_to_check = [col for col in columns_to_check if col not in ['rainfall', 'id']]

# Function to remove outliers using IQR and visualize
def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.05)
    Q3 = data[column].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Calculate the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)
    
    # Create a 1x2 plot for before & after visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Original Data Boxplot
    sns.boxplot(x=data[column], color='lightblue', ax=axes[0], 
                flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
    axes[0].set_title(f'Before Outlier Removal: {column}')
    
    # Highlight Q1, Q3, and Bounds in the first plot
    axes[0].axvline(Q1, color='green', linestyle='--', label='Q1 (15th Percentile)')
    axes[0].axvline(Q3, color='blue', linestyle='--', label='Q3 (85th Percentile)')
    axes[0].axvline(lower_bound, color='red', linestyle='-', label='Lower Bound')
    axes[0].axvline(upper_bound, color='red', linestyle='-', label='Upper Bound')
    axes[0].legend()

    # Boxplot after outlier removal
    sns.boxplot(x=filtered_data[column], color='lightgreen', ax=axes[1], 
                flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
    axes[1].set_title(f'After Outlier Removal: {column}')

    plt.suptitle(f'Outlier Detection & Removal for {column}')
    plt.tight_layout()
    plt.show()
    
    return filtered_data, rows_deleted

# Apply function to each numerical column and visualize
rows_deleted_total = 0

for column in columns_to_check:
    train_data, rows_deleted = remove_outliers_iqr_with_plot(train_data, column)
    rows_deleted_total += rows_deleted
    print(f"Rows deleted for {column}: {rows_deleted}")

print(f"Total rows deleted: {rows_deleted_total}")


y = train_data['rainfall']


# Identify numerical variables
numerical_variables = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove unwanted variables
numerical_variables = [col for col in numerical_variables if col not in ['rainfall', 'id']]

# [FOR TRAIN]
# Identify features with skewness greater than 0.75
skewed_features = train_data[numerical_variables].skew()[train_data[numerical_variables].skew() > 0.75].index.values

# Print the list of variables to be transformed
print("Features to be transformed (skewness > 0.75):")
display(skewed_features)

# Plot skewed features before transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_data[feature], bins=50, kde=True, color='blue')
    plt.title(f'Distribution of {feature} before log transformation')
    plt.show()

# Apply log1p transformation to skewed features
train_data[skewed_features] = np.log1p(train_data[skewed_features])

# Plot skewed features after transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_data[feature], bins=50, kde=True, color='green')
    plt.title(f'Distribution of {feature} after log transformation')
    plt.show()


#[FOR TRAIN]
# Identify features with skewness greater than 0.75
skewed_features = train_data[numerical_variables].skew()[train_data[numerical_variables].skew() > 0.75].index.values

# Print the list of variables to be transformed
print("Features to be transformed (skewness > 0.75):")
display(skewed_features)

# Plot skewed features before transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_data[feature], bins=50, kde=True, color='blue')
    plt.title(f'Distribution of {feature} before log transformation')
    plt.show()

# Apply log1p transformation to skewed features
train_data[skewed_features] = np.log1p(train_data[skewed_features])

# Plot skewed features after transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_data[feature], bins=50, kde=True, color='green')
    plt.title(f'Distribution of {feature} after log transformation')
    plt.show()


#[FOR TEST]
# Identify features with skewness greater than 0.75
skewed_features = test_data[numerical_variables].skew()[test_data[numerical_variables].skew() > 0.75].index.values

# Print the list of variables to be transformed
print("Features to be transformed (skewness > 0.75):")
display(skewed_features)

# Plot skewed features before transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(test_data[feature], bins=50, kde=True, color='blue')
    plt.title(f'Distribution of {feature} before log transformation')
    plt.show()

# Apply log1p transformation to skewed features
test_data[skewed_features] = np.log1p(test_data[skewed_features])

# Plot skewed features after transformation
for feature in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(test_data[feature], bins=50, kde=True, color='green')
    plt.title(f'Distribution of {feature} after log transformation')
    plt.show()


# Selecting specific columns for encoding
columns_to_encode = [
    'wind_sector'
]

train_data_to_encode = train_data[columns_to_encode]
test_data_to_encode = test_data[columns_to_encode]

# Dropping selected columns for scaling
train_data_to_scale = train_data.drop(columns_to_encode, axis=1)
test_data_to_scale = test_data.drop(columns_to_encode, axis=1)

train_data_encoded = pd.get_dummies(train_data_to_encode, columns=columns_to_encode, drop_first=True)
test_data_encoded = pd.get_dummies(test_data_to_encode, columns=columns_to_encode, drop_first=True)


train_data_encoded.sample(3)


test_data_encoded.sample(3)


train_data_encoded.columns


train_data_to_scale.columns


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# âœ… VIF ê³„ì‚° í•¨ìˆ˜ ì •ì�˜
def calculate_vif(df):
    """
    ì£¼ì–´ì§„ ë�°ì�´í„°í”„ë ˆì�„ì—�ì„œ VIF(ë‹¤ì¤‘ê³µì„ ì„±) ê°’ì�„ ê³„ì‚°í•˜ì—¬ ë°˜í™˜
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    return vif_data.sort_values(by="VIF", ascending=False)

# âœ… 'train_data_to_scale'ì—�ì„œ ë‹¤ì¤‘ê³µì„ ì„± í™•ì�¸
print("ğŸ“Œ ë‹¤ì¤‘ê³µì„ ì„± í™•ì�¸ (Train Data)")
vif_result_train = calculate_vif(train_data_to_scale)

# âœ… VIF ê°’ì�´ 10 ì�´ìƒ�ì�¸ ë³€ìˆ˜ ì¶œë ¥ (ë‹¤ì¤‘ê³µì„ ì„±ì�´ ì‹¬í•œ ë³€ìˆ˜)
high_vif_features_train = vif_result_train[vif_result_train["VIF"] > 10]
print("\nğŸš¨ VIF 10 ì�´ìƒ�ì�¸ ë³€ìˆ˜ (Train Data - ì œê±° ê³ ë ¤ ëŒ€ìƒ�)")
print(high_vif_features_train)



import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# âœ… VIF ê³„ì‚° í•¨ìˆ˜ ì •ì�˜
def calculate_vif(df):
    """
    ì£¼ì–´ì§„ ë�°ì�´í„°í”„ë ˆì�„ì—�ì„œ VIF(ë‹¤ì¤‘ê³µì„ ì„±) ê°’ì�„ ê³„ì‚°í•˜ì—¬ ë°˜í™˜
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    return vif_data.sort_values(by="VIF", ascending=False)

# âœ… 1ï¸�âƒ£ ë‹¤ì¤‘ê³µì„ ì„± í™•ì�¸ (train_data_to_scaleì—�ì„œ VIF ê°’ ê³„ì‚°)
print("ğŸ“Œ ë‹¤ì¤‘ê³µì„ ì„± í™•ì�¸ (Train Data)")

vif_result_train = calculate_vif(train_data_to_scale)

# âœ… VIF 10 ì�´ìƒ�ì�¸ Feature ì°¾ê¸°
high_vif_features = vif_result_train[vif_result_train["VIF"] > 10]["Feature"].tolist()

print("\nğŸš¨ VIF 10 ì�´ìƒ�ì�¸ Feature ëª©ë¡� (ì œê±° ê³ ë ¤ ëŒ€ìƒ�)")
print(high_vif_features)

# âœ… 2ï¸�âƒ£ ì œê±°í•  Feature ì„ ì • (ì‚¬ìš©ì��ê°€ ì„ íƒ�í•œ Feature)
features_to_remove = ["maxtemp", "cloud", "humidity"]

# âœ… ì‹¤ì œë¡œ ì œê±°í•  Feature ë¦¬ìŠ¤íŠ¸ ìƒ�ì„± (VIFê°€ ë†’ì�€ Feature ì¤‘ ìœ„ Feature í�¬í•¨)
final_features_to_remove = [feat for feat in high_vif_features if feat in features_to_remove]

print("\nğŸ“Œ ìµœì¢… ì œê±°í•  Feature ëª©ë¡�")
print(final_features_to_remove)

# âœ… 3ï¸�âƒ£ Train & Test ë�°ì�´í„°ì—�ì„œ Feature ì œê±°
train_data_to_scale.drop(columns=final_features_to_remove, inplace=True)
test_data_to_scale.drop(columns=final_features_to_remove, inplace=True)

print("\nğŸš€ Feature ì œê±° ì™„ë£Œ! ìµœì¢… Feature ê°œìˆ˜:", train_data_to_scale.shape[1])



train_data_to_scale.columns


test_data_to_scale.columns


y


train_data_to_scale


train_data_to_scale['rainfall']


from sklearn.preprocessing import MinMaxScaler, RobustScaler

# âœ… 1ï¸�âƒ£ 'rainfall'ì�„ ë³„ë�„ë¡œ ì €ì�¥ (ë‚˜ì¤‘ì—� ì‚¬ìš©)
train_target = train_data_to_scale['rainfall'].copy()  # 'rainfall' ë”°ë¡œ ì €ì�¥

# âœ… 2ï¸�âƒ£ 'rainfall'ì�„ ì œì™¸í•œ í›ˆë ¨ ë�°ì�´í„° ì¤€ë¹„
train_features = train_data_to_scale.drop(columns=['rainfall'], errors='ignore')  # rainfall ì œê±°
test_features = test_data_to_scale.copy()  # í…ŒìŠ¤íŠ¸ ë�°ì�´í„° (ì›�ë³¸ ìœ ì§€)

# âœ… 3ï¸�âƒ£ ë¡œì§€ìŠ¤í‹± íšŒê·€ìš© (RobustScaler) ì �ìš©
robust_scaler = RobustScaler()
train_data_scaled_robust = pd.DataFrame(robust_scaler.fit_transform(train_features), columns=train_features.columns)
test_data_scaled_robust = pd.DataFrame(robust_scaler.transform(test_features), columns=test_features.columns)

# âœ… 4ï¸�âƒ£ íŠ¸ë¦¬ ê¸°ë°˜ ëª¨ë�¸ìš© (MinMaxScaler) ì �ìš©
minmax_scaler = MinMaxScaler()
train_data_scaled_minmax = pd.DataFrame(minmax_scaler.fit_transform(train_features), columns=train_features.columns)
test_data_scaled_minmax = pd.DataFrame(minmax_scaler.transform(test_features), columns=test_features.columns)

# âœ… 5ï¸�âƒ£ ìŠ¤ì¼€ì�¼ë§� ì™„ë£Œ í›„ ë�°ì�´í„° í�¬ê¸° í™•ì�¸
print("\nğŸš€ ìŠ¤ì¼€ì�¼ë§� ì™„ë£Œ!")
print("ğŸ”¹ Train Data (RobustScaler) í�¬ê¸°:", train_data_scaled_robust.shape)
print("ğŸ”¹ Test Data (RobustScaler) í�¬ê¸°:", test_data_scaled_robust.shape)
print("ğŸ”¹ Train Data (MinMaxScaler) í�¬ê¸°:", train_data_scaled_minmax.shape)
print("ğŸ”¹ Test Data (MinMaxScaler) í�¬ê¸°:", test_data_scaled_minmax.shape)

# âœ… 6ï¸�âƒ£ 'rainfall'ê³¼ ê²°í•©í•˜ì§€ ì•Šê³  ìœ ì§€ (í›„ì—� train_targetìœ¼ë¡œ í™œìš©)



# âœ… 1ï¸�âƒ£ ë¡œì§€ìŠ¤í‹± íšŒê·€ìš© (RobustScaler ì �ìš©ë�œ ë�°ì�´í„°) -> ë²”ì£¼í˜• ì�¸ì½”ë”© ë�°ì�´í„° ì—†ì�´ ê²°í•©
train_data_combined_robust = train_data_scaled_robust.copy()
test_data_combined_robust = test_data_scaled_robust.copy()

# âœ… 2ï¸�âƒ£ íŠ¸ë¦¬ ê¸°ë°˜ ëª¨ë�¸ìš© (MinMaxScaler ì �ìš©ë�œ ë�°ì�´í„°) -> ë²”ì£¼í˜• ì�¸ì½”ë”© ë�°ì�´í„° ì—†ì�´ ê²°í•©
train_data_combined_minmax = train_data_scaled_minmax.copy()
test_data_combined_minmax = test_data_scaled_minmax.copy()

# âœ… 3ï¸�âƒ£ ê²°í•©ë�œ ë�°ì�´í„° í�¬ê¸° í™•ì�¸
print("\nğŸš€ ë²”ì£¼í˜• ì�¸ì½”ë”© ë�°ì�´í„° ì œê±° ì™„ë£Œ!")
print("ğŸ”¹ Train Data (ë¡œì§€ìŠ¤í‹± íšŒê·€ìš©):", train_data_combined_robust.shape)
print("ğŸ”¹ Test Data (ë¡œì§€ìŠ¤í‹± íšŒê·€ìš©):", test_data_combined_robust.shape)
print("ğŸ”¹ Train Data (íŠ¸ë¦¬ ëª¨ë�¸ìš©):", train_data_combined_minmax.shape)
print("ğŸ”¹ Test Data (íŠ¸ë¦¬ ëª¨ë�¸ìš©):", test_data_combined_minmax.shape)



from sklearn.model_selection import TimeSeriesSplit

# âœ… 1ï¸�âƒ£ í›ˆë ¨/ê²€ì¦� ë�°ì�´í„° ë¶„í•  (ì‹œê°„ ìˆœì„œ ìœ ì§€)
split_index = int(len(train_data_combined_robust) * 0.8)  # 80% í›ˆë ¨ ë�°ì�´í„°, 20% ê²€ì¦� ë�°ì�´í„°

# ë¡œì§€ìŠ¤í‹± íšŒê·€ìš© ë�°ì�´í„° ë¶„í• 
X_train_robust, X_val_robust = train_data_combined_robust.iloc[:split_index], train_data_combined_robust.iloc[split_index:]

# íŠ¸ë¦¬ ê¸°ë°˜ ëª¨ë�¸ìš© ë�°ì�´í„° ë¶„í• 
X_train_minmax, X_val_minmax = train_data_combined_minmax.iloc[:split_index], train_data_combined_minmax.iloc[split_index:]

# íƒ€ê²Ÿ ë³€ìˆ˜ (rainfall) ë¶„í• 
y_train, y_val = train_target.iloc[:split_index], train_target.iloc[split_index:]

# âœ… 2ï¸�âƒ£ TimeSeriesSplit ì„¤ì • (5ê°œ í�´ë“œ ì‚¬ìš©)
tscv = TimeSeriesSplit(n_splits=5)

# âœ… 3ï¸�âƒ£ ë�°ì�´í„° í�¬ê¸° í™•ì�¸
print("\nğŸš€ Chronological Train-Validation Split ì �ìš© ì™„ë£Œ!")
print("ğŸ”¹ Train Data (ë¡œì§€ìŠ¤í‹± íšŒê·€ìš©):", X_train_robust.shape, "ğŸ”¹ Validation Data:", X_val_robust.shape)
print("ğŸ”¹ Train Data (íŠ¸ë¦¬ ëª¨ë�¸ìš©):", X_train_minmax.shape, "ğŸ”¹ Validation Data:", X_val_minmax.shape)
print("ğŸ”¹ Target Train Data í�¬ê¸°:", y_train.shape, "ğŸ”¹ Target Validation Data í�¬ê¸°:", y_val.shape)



from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# âœ… 1ï¸�âƒ£ ë¡œì§€ìŠ¤í‹± íšŒê·€ ëª¨ë�¸ (Logistic Regression)
logistic_model = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    class_weight='balanced',
    solver='liblinear'
)

# âœ… 2ï¸�âƒ£ íŠ¸ë¦¬ ê¸°ë°˜ ëª¨ë�¸ë“¤ (LGBM, XGBoost, CatBoost)
lgbm_model = LGBMClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
catboost_model = CatBoostClassifier(verbose=False, random_state=42)



def evaluate_model(model, X_train, y_train, model_name):
    """
    TimeSeriesSplitì�„ í™œìš©í•˜ì—¬ ëª¨ë�¸ì�„ í�‰ê°€í•˜ëŠ” í•¨ìˆ˜
    """
    tscv = TimeSeriesSplit(n_splits=5)
    y_true, y_pred, y_proba = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"ğŸ“Œ Fold {fold+1} training...")

        # í›ˆë ¨/ê²€ì¦� ë�°ì�´í„° ë¶„í• 
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # ëª¨ë�¸ í•™ìŠµ
        model.fit(X_train_fold, y_train_fold)

        # ì˜ˆì¸¡ ìˆ˜í–‰
        preds = model.predict(X_val_fold)
        pred_proba = model.predict_proba(X_val_fold)[:, 1]  # í™•ë¥  ì˜ˆì¸¡

        # ê²°ê³¼ ì €ì�¥
        y_true.extend(y_val_fold)
        y_pred.extend(preds)
        y_proba.extend(pred_proba)

    # ìµœì¢… í�‰ê°€
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"\nğŸš€ {model_name} TimeSeriesSplit í�‰ê°€ ê²°ê³¼")
    print(f"Accuracy: {acc:.5f}, AUC: {auc:.5f}")

    return acc, auc



# âœ… 1ï¸�âƒ£ ë¡œì§€ìŠ¤í‹± íšŒê·€ í�‰ê°€
logistic_acc, logistic_auc = evaluate_model(logistic_model, X_train_robust, y_train, "Logistic Regression")

# âœ… 2ï¸�âƒ£ íŠ¸ë¦¬ ê¸°ë°˜ ëª¨ë�¸ í�‰ê°€
lgbm_acc, lgbm_auc = evaluate_model(lgbm_model, X_train_minmax, y_train, "LGBM Classifier")
xgb_acc, xgb_auc = evaluate_model(xgb_model, X_train_minmax, y_train, "XGBoost Classifier")
catboost_acc, catboost_auc = evaluate_model(catboost_model, X_train_minmax, y_train, "CatBoost Classifier")

# âœ… 3ï¸�âƒ£ ê²°ê³¼ ë¹„êµ� ì¶œë ¥
print("\nğŸš€ 1ì°¨ ëª¨ë�¸ ì„±ëŠ¥ ë¹„êµ�:")
print(f"ğŸ”¹ Logistic Regression -> Accuracy: {logistic_acc:.5f}, AUC: {logistic_auc:.5f}")
print(f"ğŸ”¹ LGBM -> Accuracy: {lgbm_acc:.5f}, AUC: {lgbm_auc:.5f}")
print(f"ğŸ”¹ XGBoost -> Accuracy: {xgb_acc:.5f}, AUC: {xgb_auc:.5f}")
print(f"ğŸ”¹ CatBoost -> Accuracy: {catboost_acc:.5f}, AUC: {catboost_auc:.5f}")



import warnings
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# ë¶ˆí•„ìš”í•œ ê²½ê³  ì œê±°
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_lgbm(trial):
    """Optunaë¥¼ ì‚¬ìš©í•˜ì—¬ LGBM í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° íŠœë‹�"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
    }
    
    model = LGBMClassifier(**params, random_state=42, verbose=-1)  # LightGBM ê²½ê³  ìµœì†Œí™”
    tscv = TimeSeriesSplit(n_splits=5)
    
    auc_scores = []
    for train_idx, val_idx in tscv.split(X_train_minmax):
        model.fit(X_train_minmax.iloc[train_idx], y_train.iloc[train_idx])
        preds_proba = model.predict_proba(X_train_minmax.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y_train.iloc[val_idx], preds_proba)
        auc_scores.append(auc)
    
    return sum(auc_scores) / len(auc_scores)  # í�‰ê·  AUC ë°˜í™˜

# Optuna í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° ìµœì �í™”
study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=50)

# ìµœì � íŒŒë�¼ë¯¸í„° ì¶œë ¥
lgbm_best_params = study_lgbm.best_params
print("\nğŸš€ ìµœì � LGBM íŒŒë�¼ë¯¸í„°:", lgbm_best_params)



from xgboost import XGBClassifier

def objective_xgb(trial):
    """Optunaë¥¼ ì‚¬ìš©í•˜ì—¬ XGBoost í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° íŠœë‹�"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }

    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)

    auc_scores = []
    for train_idx, val_idx in tscv.split(X_train_minmax):
        model.fit(X_train_minmax.iloc[train_idx], y_train.iloc[train_idx])
        preds_proba = model.predict_proba(X_train_minmax.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y_train.iloc[val_idx], preds_proba)
        auc_scores.append(auc)

    return sum(auc_scores) / len(auc_scores)  # í�‰ê·  AUC ë°˜í™˜

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=50)

# ìµœì � íŒŒë�¼ë¯¸í„° ì¶œë ¥
xgb_best_params = study_xgb.best_params
print("\nğŸš€ ìµœì � XGBoost íŒŒë�¼ë¯¸í„°:", xgb_best_params)



from catboost import CatBoostClassifier

def objective_cat(trial):
    """Optunaë¥¼ ì‚¬ìš©í•˜ì—¬ CatBoost í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° íŠœë‹�"""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 3, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'random_strength': trial.suggest_uniform('random_strength', 1, 10),
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0)
    }

    model = CatBoostClassifier(**params, verbose=False, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)

    auc_scores = []
    for train_idx, val_idx in tscv.split(X_train_minmax):
        model.fit(X_train_minmax.iloc[train_idx], y_train.iloc[train_idx])
        preds_proba = model.predict_proba(X_train_minmax.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y_train.iloc[val_idx], preds_proba)
        auc_scores.append(auc)

    return sum(auc_scores) / len(auc_scores)  # í�‰ê·  AUC ë°˜í™˜

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=50)

# ìµœì � íŒŒë�¼ë¯¸í„° ì¶œë ¥
cat_best_params = study_cat.best_params
print("\nğŸš€ ìµœì � CatBoost íŒŒë�¼ë¯¸í„°:", cat_best_params)



from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# âœ… 1ï¸�âƒ£ ê¸°ë³¸ 1ì°¨ ëª¨ë�¸ (íŠœë‹� X)
base_lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
base_lgbm = LGBMClassifier(random_state=42, verbose=-1)
base_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
base_cat = CatBoostClassifier(verbose=False, random_state=42)

# âœ… 2ï¸�âƒ£ íŠœë‹�ë�œ 1ì°¨ ëª¨ë�¸ (Optuna ìµœì �í™” ì �ìš©)
tuned_lgbm = LGBMClassifier(**lgbm_best_params, verbose=-1)
tuned_xgb = XGBClassifier(**xgb_best_params, use_label_encoder=False, eval_metric='logloss')
tuned_cat = CatBoostClassifier(**cat_best_params, verbose=False)

# âœ… 3ï¸�âƒ£ ê¸°ë³¸ ìŠ¤íƒœí‚¹ ëª¨ë�¸
basic_stacking = StackingClassifier(
    estimators=[
        ('lr', base_lr),
        ('lgbm', base_lgbm),
        ('xgb', base_xgb),
        ('cat', base_cat)
    ],
    final_estimator=LGBMClassifier(random_state=42, verbose=-1),  # ìµœì¢… ëª¨ë�¸ì�€ LGBM ì‚¬ìš©
    n_jobs=-1
)

# âœ… 4ï¸�âƒ£ íŠœë‹�ë�œ ìŠ¤íƒœí‚¹ ëª¨ë�¸
tuned_stacking = StackingClassifier(
    estimators=[
        ('lr', base_lr),
        ('lgbm', tuned_lgbm),
        ('xgb', tuned_xgb),
        ('cat', tuned_cat)
    ],
    final_estimator=LGBMClassifier(**lgbm_best_params, verbose=-1),  # ìµœì �í™”ë�œ LGBM ì‚¬ìš©
    n_jobs=-1
)

# âœ… 5ï¸�âƒ£ ëª¨ë�¸ í•™ìŠµ
basic_stacking.fit(X_train_minmax, y_train)
tuned_stacking.fit(X_train_minmax, y_train)

# âœ… 6ï¸�âƒ£ í�‰ê°€
basic_preds = basic_stacking.predict(X_val_minmax)
basic_preds_proba = basic_stacking.predict_proba(X_val_minmax)[:, 1]

tuned_preds = tuned_stacking.predict(X_val_minmax)
tuned_preds_proba = tuned_stacking.predict_proba(X_val_minmax)[:, 1]

# âœ… 7ï¸�âƒ£ ì„±ëŠ¥ ë¹„êµ�
basic_acc = accuracy_score(y_val, basic_preds)
basic_auc = roc_auc_score(y_val, basic_preds_proba)

tuned_acc = accuracy_score(y_val, tuned_preds)
tuned_auc = roc_auc_score(y_val, tuned_preds_proba)

print("\nğŸš€ **ìŠ¤íƒœí‚¹ ëª¨ë�¸ ì„±ëŠ¥ ë¹„êµ�**")
print(f"ğŸ”¹ ê¸°ë³¸ ìŠ¤íƒœí‚¹ ëª¨ë�¸ -> Accuracy: {basic_acc:.5f}, AUC: {basic_auc:.5f}")
print(f"ğŸ”¹ íŠœë‹�ë�œ ìŠ¤íƒœí‚¹ ëª¨ë�¸ -> Accuracy: {tuned_acc:.5f}, AUC: {tuned_auc:.5f}")



# âœ… í…ŒìŠ¤íŠ¸ ë�°ì�´í„°ì—�ì„œ ID ì»¬ëŸ¼ ê°€ì ¸ì˜¤ê¸°
test_ids = test_data['id']  # 'id' ì»¬ëŸ¼ì�´ ì›�ë³¸ test_dataì—� ì�ˆë‹¤ê³  ê°€ì •

# âœ… í…ŒìŠ¤íŠ¸ ë�°ì�´í„° ì˜ˆì¸¡ (í™•ë¥ ê°’)
test_preds_proba = tuned_stacking.predict_proba(test_data_combined_minmax)[:, 1]

# âœ… ì œì¶œ ë�°ì�´í„°í”„ë ˆì�„ ìƒ�ì„±
submission = pd.DataFrame({
    'id': test_ids,  # í…ŒìŠ¤íŠ¸ ë�°ì�´í„°ì�˜ ID
    'rainfall': test_preds_proba  # ì˜ˆì¸¡ë�œ í™•ë¥ ê°’
})

# âœ… CSV íŒŒì�¼ ì €ì�¥
submission.to_csv('submission.csv', index=False)

print("\nğŸš€ **Submission íŒŒì�¼ ìƒ�ì„± ì™„ë£Œ!**")
print(submission.head(10))  # ìƒ�ìœ„ 10ê°œ í™•ì�¸
































