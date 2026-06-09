import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv, pd.read_parquet )
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter

import os, gc
from tqdm.auto import tqdm
import pickle # module to serialize and deserialize objects
import re # for Regular expression operations 

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data  import Dataset, DataLoader
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor

import lightgbm as lgb
from lightgbm import LGBMRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

import kaggle_evaluation.jane_street_inference_server
gridColor = 'lightgrey'


%%time
path = "/kaggle/input/jane-street-real-time-market-data-forecasting"
samples = [] 

# Load a data from each file:
r = range(2)
for i in r:
    file_path = f"{path}/train.parquet/partition_id={i}/part-0.parquet"
    part = pd.read_parquet(file_path)
    samples.append(part)
    
sample_df = pd.concat(samples, ignore_index=True) # Concatenate all samples into one DataFrame if needed

sample_df.round(1)


train = sample_df
train['N'] = train.index.values
train['id'] = train.index.values

xx = sample_df[(sample_df.symbol_id == 1)]['id']
yy = sample_df[(sample_df.symbol_id == 1)]['responder_6']

plt.figure(figsize=(16, 5))
plt.plot(xx, yy, color='green', linewidth=0.05)
plt.suptitle('Returns of responder_6 for Financial Instrument (symbol_id = 1)', weight='bold', fontsize=16)
plt.xlabel("Time (Sequential ID)", fontsize=12)
plt.ylabel("Returns (responder_6)", fontsize=12)
plt.grid(color='lightgray', linewidth=0.8)
plt.axhline(0, color='red', linestyle='-', linewidth=1.2)




# Plotting cumulative responder_6 for symbol_id=1
plt.figure(figsize=(14, 4))
plt.plot(xx, yy.cumsum(), color='green', linewidth=0.6)
plt.suptitle('Cumulative Responder_6 (for Symbol ID = 1)', weight='bold', fontsize=16)
plt.xlabel("Time (which we got from Sequential ID column)", fontsize=12)
plt.ylabel("Cumulative Returns", fontsize=12)
plt.yticks(np.arange(-500, 1000, 250))
plt.grid(color='lightgray', linewidth=0.7)
plt.axhline(0, color='red', linestyle='-', linewidth=0.7)  # Zero baseline
plt.show()



# for symbol_id == 0
plt.figure(figsize=(18, 7))
predictor_cols = [col for col in sample_df.columns if 'responder' in col]
for i in predictor_cols: 
    if i == 'responder_6': 
        c='red'
        lw=2.5
        plt.plot((sample_df[sample_df.symbol_id == 0].groupby(['date_id'])[i].mean()).cumsum(), linewidth = lw, color = c)
    else: 
        lw=1
        plt.plot((sample_df[sample_df.symbol_id == 0].groupby(['date_id'])[i].mean()).cumsum(), linewidth = lw)

plt.xlabel('Trade days (from date_id column)')
plt.ylabel('Cumulative response (from responder values)')
plt.title('Response time series for symbol_id 0 over trade days  \n (Responder 6 (red) and other responders in diff colors)', weight='bold')
plt.grid(visible=True, color = gridColor, linewidth = 0.7)
plt.axhline(0, color='green', linestyle='-', linewidth=2)
plt.legend(predictor_cols)
sns.despine()
#plt.show()


import matplotlib.pyplot as plt
import re

# Parameters
df_train = sample_df
s_id = 0  # Change params to take a look at other symbols
res_columns = [col for col in df_train.columns if re.match("responder_", col)]
n_rows = len(res_columns)  # Number of responders

# Plot configuration
fig, axs = plt.subplots(figsize=(18, 4 * n_rows))
grid_color = '#cccccc'  # Define gridline color

# Loop through responders to plot
for j, responder in enumerate(res_columns):
    xx = df_train[df_train.symbol_id == s_id]['N']
    yy = df_train[df_train.symbol_id == s_id][responder]
    color = 'red' if j == 6 else 'black'  # Highlight responder_6 in red
    
    # Cumulative sum plot
    ax1 = plt.subplot(n_rows, 3, j * 3 + 1)
    ax1.plot(xx, yy.cumsum(), color=color, linewidth=0.8)
    ax1.axhline(0, color='blue', linestyle='-', linewidth=0.9)
    ax1.grid(color=grid_color)
    ax1.set_title(f"Cumulative Sum: {responder}", fontsize=12)
    
    # Line plot
    ax2 = plt.subplot(n_rows, 3, j * 3 + 2)
    ax2.plot(xx, yy, color=color, linewidth=0.5)
    ax2.axhline(0, color='blue', linestyle='-', linewidth=0.9)
    ax2.grid(color=grid_color)
    ax2.set_title(f"Raw Values: {responder}", fontsize=12)
    
    # Histogram plot
    ax3 = plt.subplot(n_rows, 3, j * 3 + 3)
    bins = 100  # Adjusted for better clarity
    ax3.hist(yy, bins=bins, color=color, density=True, histtype="step", linewidth=1.2)
    ax3.hist(yy, bins=bins, color='lightgrey', density=True, alpha=0.7)
    ax3.grid(color=grid_color)
    ax3.set_title(f"Histogram: {responder}", fontsize=12)
    ax3.set_xlim([-2.5, 2.5])
    ax3.set_ylim([0, 3.5])

# Global styling
fig.suptitle(f"Responder Analysis for Symbol ID {s_id}", fontsize=16, weight='bold')
fig.tight_layout(pad=3)
fig.patch.set_linewidth(2)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#f9f9f9')
plt.show()




#responder_6 for all different symbol ID's 

res_columns = [col for col in df_train.columns if re.match("responder_", col)]
row=10
fig, axs = plt.subplots(figsize=(18, 5*row))
b=300
j = 0
for i in range(1, 3 * row + 1, 3):
    xx= sample_df[(sample_df.symbol_id==j)] ['N']
    yy= sample_df[(sample_df.symbol_id==j)]['responder_6']
    c='black'
        
    ax1 = plt.subplot(row, 3, i)
    ax1.plot(   xx,yy.cumsum()   , color = c, linewidth =0.8 )
    plt.axhline(0, color='red', linestyle='-', linewidth=0.7)
    plt.grid(color = gridColor)
    plt.xlabel('Time')
    
    ax2 = plt.subplot(row, 3, i+1)
    ax2.plot(xx,yy   , color = c, linewidth =0.05)
    plt.axhline(0, color='red', linestyle='-', linewidth=0.7)
    ax2.set_title(f"symbol_id={j}", fontsize = '14')
    plt.grid(color = gridColor)
    plt.xlabel('Time')
    
    ax3 = plt.subplot(row, 3, i+2)
    ax3.hist(yy, bins=b, color = c, density=True, histtype="step" )
    ax3.hist(yy, bins=b, color = 'lightgrey',density=True)
    plt.grid(color = gridColor)
    ax3.set_xlim([-2.5, 2.5])
    ax3.set_ylim([0, 1.5])
    plt.xlabel('Time')
    
    j = j + 1
    
fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee') 
plt.show()


#missing values

df_train = sample_df
plt.figure(figsize=(20, 3))    # Plot missing values
plt.bar(x=df_train.isna().sum().index, height=df_train.isna().sum().values, color="red", label='missing')   # analog: using missingno
plt.xticks(rotation=90)
plt.title(f'Missing values over the {len(df_train)} samples which have a target')
plt.grid()
plt.legend()
plt.show()


#features

features = pd.read_csv(f"{path}/features.csv")
features


#Which features have which all tags

plt.figure(figsize=(18, 6))
plt.imshow(features.iloc[:, 1:].T.values, cmap="gray_r")
plt.xlabel("feature_00 - feature_78")
plt.ylabel("tag_0 - tag_16")
plt.yticks(np.arange(17))
plt.xticks(np.arange(79))
plt.grid(color = 'lightgrey')
plt.show()


#correlation bw all features based on their tags

plt.figure(figsize=(11, 11))
matrix = features[[ f"tag_{no}" for no in range(0,17,1) ] ].T.corr()
sns.heatmap(matrix, square=True, cmap="coolwarm", alpha =0.9, vmin=-1, vmax=1, center= 0, linewidths=0.5, linecolor='white')
plt.show()


#weights

sample_df['weight'].describe().round(1)


plt.figure(figsize=(8,3))
plt.hist(sample_df['weight'], bins=30, color='grey', edgecolor = 'white',density=True )
plt.title('Distribution of weights')
plt.grid(color = 'lightgrey', linewidth=0.5)
plt.axvline(1.7, color='red', linestyle='-', linewidth=0.7)
plt.show()


#responders

responders = pd.read_csv(f"{path}/responders.csv")
responders


#correlation matrix for all responders based on their tags
plt.figure(figsize=(6, 6))
responders = pd.read_csv(f"{path}/responders.csv")
matrix = responders[[ f"tag_{no}" for no in range(0,5,1) ] ].T.corr()
sns.heatmap(matrix, square=True, cmap="coolwarm", alpha =0.9, vmin=-1, vmax=1, center= 0, linewidths=0.5, 
            linecolor='white', annot=True, fmt='.2f')
plt.xlabel("Responder_0 - Responder_8")
plt.ylabel("Responder_0 - Responder_8")
plt.show()


col =[]
for i in range(9):
    col.append(f"responder_{i}") 

sample_df[col].describe().round(1)


#responders distribution against other responders

numerical_features = []
numerical_features = sample_df.filter(regex='^responder_').columns.tolist()  # Separate responders
numerical_features.remove('responder_6')

gs = 600  # Grid size for hexbin
k = 1
col = 3
row = 3

fig, axs = plt.subplots(row, col, figsize=(6*col, 5*row))

for i in numerical_features:
    ax = plt.subplot(col, row, k)
    
    # Hexbin plot
    hb = ax.hexbin(sample_df[i], sample_df['responder_6'], 
                   gridsize=gs, cmap='CMRmap', bins='log', alpha=0.8)
    
    # Labels and title for each subplot
    ax.set_xlabel(f'{i}', fontsize=12)
    ax.set_ylabel('responder_6', fontsize=12)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # Adding a colorbar for the hexbin
    cb = fig.colorbar(hb, ax=ax, orientation='vertical')
    cb.set_label('Log Density', fontsize=10)
    
    k += 1

# Add a main title for the figure
fig.suptitle('Responder Relationships: Hexbin Plot with Responder_6', 
             fontsize=18, weight='bold', color='black')

# Beautify the figure's border and background
fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee')   

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
plt.show()





#Each subplot is a hexagonal binning plot showing the density of data points where values of responder_6 and another responder (responder_i) overlap.



## Responder_6 Distribution Against Selected Features

# Responder_6 Distribution Against Selected Features
numerical_features = [f'feature_{i}' for i in ['05', '06', '07', '08', '12', '15', '19', '32', '38', '39', '50', '51', '65', '66', '67']]

gs = 600  # Grid size for hexbin
k = 1
col = 3
row = int(np.ceil(len(numerical_features) / col))  # Dynamically calculate rows
sz = 5
w = sz * col
h = w / col * row

fig, axs = plt.subplots(row, col, figsize=(w, h))

for i in numerical_features:
    ax = plt.subplot(row, col, k)
    
    # Hexbin plot
    hb = ax.hexbin(sample_df['responder_6'], sample_df[i], gridsize=gs, cmap='CMRmap', bins='log', alpha=0.5)
    ax.set_xlabel(f'{i}', fontsize=10)
    ax.set_ylabel('responder_6', fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # Add colorbar
    cb = fig.colorbar(hb, ax=ax, orientation='vertical', shrink=0.8)
    cb.set_label('Log Density', fontsize=8)
    
    k += 1

# Add a main title
fig.suptitle('Responder_6 Distribution Against Selected Features', fontsize=16, weight='bold')

# Beautify figure
fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee')   

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
plt.show()
  


#Relationship between selected features from the dataset.

numerical_features = [f'feature_0{i}' for i in range(5, 9)] + [f'feature_{i}' for i in range(15, 20)]

a = 0
k = 1
n = 3  # Number of subplots per row

fig_num = 1  # Track the figure number
plt.figure(figsize=(15, 4))  # Initialize first figure

for i in numerical_features[:-1]:
    a += 1
    for j in numerical_features[a:]:
        plt.subplot(1, n, k)
        
        # Create hexbin plot
        hb = plt.hexbin(sample_df[i], sample_df[j], gridsize=200, cmap='CMRmap', bins='log', alpha=0.8)
        plt.grid()
        plt.xlabel(f'{i}', fontsize=10)
        plt.ylabel(f'{j}', fontsize=10)
        plt.tick_params(axis='x', labelsize=6)
        plt.tick_params(axis='y', labelsize=6)
        plt.title(f'{i} vs {j}', fontsize=12)
        
        # Add colorbar
        if k == n:
            cb = plt.colorbar(hb, ax=plt.gca(), shrink=0.8)
            cb.set_label('Log Density', fontsize=8)
        
        k += 1
        
        # If row limit is reached, reset the figure
        if k > n:
            plt.suptitle(f'Pairwise Hexbin Plots - Figure {fig_num}', fontsize=14, weight='bold')
            plt.show()
            
            k = 1
            fig_num += 1
            plt.figure(figsize=(15, 4))  # Start new figure

# Handle the final figure
if k != 1:
    plt.tight_layout()
    plt.suptitle(f'Pairwise Hexbin Plots - Figure {fig_num}', fontsize=14, weight='bold')
    plt.show()




# Assuming df_train is your DataFrame (e.g., sample_df)
df_train = sample_df

# Define feature columns
feature_cols = [f'feature_{i:02d}' for i in range(79)]

# Compute statistical summaries for feature columns
feature_stats = df_train[feature_cols].describe().transpose()

# Reset index and rename columns
feature_stats = feature_stats.reset_index().rename(columns={'index': 'Feature'})

# Select the desired statistics
desired_stats = ['Feature', 'mean', 'std', 'min', 'max']

# Display the DataFrame
print(feature_stats[desired_stats])


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

def analyze_missing_data(df):
    """
    Analyze missing data patterns and categorize features based on missing percentages
    """
    # Calculate missing percentages for each column
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    
    # Categorize features based on missing percentages
    completely_missing = missing_percentages[missing_percentages == 100].index.tolist()
    high_missing = missing_percentages[(missing_percentages >= 50) & (missing_percentages < 100)].index.tolist()
    moderate_missing = missing_percentages[(missing_percentages >= 25) & (missing_percentages < 50)].index.tolist()
    low_missing = missing_percentages[(missing_percentages > 0) & (missing_percentages < 25)].index.tolist()
    
    print("\nMissing Data Analysis:")
    print(f"Completely Missing (100%): {len(completely_missing)} features")
    print(f"High Missing (50-99%): {len(high_missing)} features")
    print(f"Moderate Missing (25-49%): {len(moderate_missing)} features")
    print(f"Low Missing (1-24%): {len(low_missing)} features")
    
    return {
        'completely_missing': completely_missing,
        'high_missing': high_missing,
        'moderate_missing': moderate_missing,
        'low_missing': low_missing,
        'missing_percentages': missing_percentages
    }

def handle_missing_data(df, missing_analysis, target_col='responder_6'):
    """
    Handle missing data based on different categories and their relationship with target
    """
    print("\nHandling Missing Data:")
    
    # 1. Remove completely missing columns
    df.drop(columns=missing_analysis['completely_missing'], inplace=True)
    print(f"Removed {len(missing_analysis['completely_missing'])} completely missing columns")
    
    # 2. Handle high missing features
    high_missing_correlations = {}
    for feature in missing_analysis['high_missing']:
        if feature.startswith('feature_'):  # Only process feature columns
            # Calculate correlation with target using available data
            valid_data = df[[feature, target_col]].dropna()
            if len(valid_data) > 0:
                corr = valid_data[feature].corr(valid_data[target_col])
                high_missing_correlations[feature] = corr
    
    # Keep high-missing features with significant correlation (|corr| >= 0.01)
    high_missing_to_keep = [f for f, c in high_missing_correlations.items() if abs(c) >= 0.01]
    high_missing_to_drop = [f for f in missing_analysis['high_missing'] 
                           if f.startswith('feature_') and f not in high_missing_to_keep]
    
    # Drop high-missing features with low correlation
    df.drop(columns=high_missing_to_drop, inplace=True)
    print(f"Dropped {len(high_missing_to_drop)} high-missing features with low correlation")
    
    # 3. Handle moderate missing features
    moderate_missing_correlations = {}
    for feature in missing_analysis['moderate_missing']:
        if feature.startswith('feature_'):
            valid_data = df[[feature, target_col]].dropna()
            if len(valid_data) > 0:
                corr = valid_data[feature].corr(valid_data[target_col])
                moderate_missing_correlations[feature] = corr
    
    # For moderate-missing features, use more sophisticated imputation based on correlation
    for feature in missing_analysis['moderate_missing']:
        if feature.startswith('feature_'):
            if abs(moderate_missing_correlations.get(feature, 0)) >= 0.05:
                # For features with higher correlation, use more sophisticated imputation
                # Here we'll use interpolation with forward and backward fill
                df[feature].interpolate(method='linear', limit_direction='both', inplace=True)
                # Fill any remaining missing values with median
                df[feature].fillna(df[feature].median(), inplace=True)
            else:
                # For features with lower correlation, use median imputation
                df[feature].fillna(df[feature].median(), inplace=True)
    
    print(f"Handled {len(missing_analysis['moderate_missing'])} moderate-missing features")
    
    # 4. Handle low missing features
    for feature in missing_analysis['low_missing']:
        if feature.startswith('feature_'):
            # For low missing values, use interpolation
            df[feature].interpolate(method='linear', limit_direction='both', inplace=True)
            # Fill any remaining missing values with median
            df[feature].fillna(df[feature].median(), inplace=True)
    
    print(f"Handled {len(missing_analysis['low_missing'])} low-missing features")
    
    return df

def load_and_clean_data(path, num_partitions=2):
    """
    Load and clean data from specified partitions
    """
    print("Loading data...")
    samples = []
    
    # Load data from specified partitions
    for i in range(num_partitions):
        file_path = f"{path}/train.parquet/partition_id={i}/part-0.parquet"
        part = pd.read_parquet(file_path)
        samples.append(part)
    
    # Combine samples
    df = pd.concat(samples, ignore_index=True)
    print(f"Loaded {len(df)} rows from {num_partitions} partitions")
    
    # Analyze missing data
    missing_analysis = analyze_missing_data(df)
    
    # Handle missing data
    df = handle_missing_data(df, missing_analysis)
    
    # Step 3: Analyze responder relationships
    responder_cols = [f'responder_{i}' for i in range(9) if i != 6]
    responder_corrs = {col: df[col].corr(df['responder_6']) for col in responder_cols}
    significant_responders = [r for r, c in responder_corrs.items() if abs(c) >= 0.1]
    
    # Keep only significant responders and target
    responders_to_drop = [r for r in responder_cols if r not in significant_responders]
    df.drop(columns=responders_to_drop, inplace=True)
    print("\nKept significant responders:", significant_responders)
    
    # Step 4: Feature Engineering
    # Get remaining feature columns
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # Calculate skewness for features
    skewed_features = [col for col in feature_cols if abs(skew(df[col].dropna())) > 0.5]
    
    # Log transform skewed features (adding 1 to handle zeros/negative values)
    for col in skewed_features:
        min_val = df[col].min()
        if min_val <= 0:
            df[col] = np.log1p(df[col] - min_val + 1)
    print("\nApplied log transformation to skewed features:", len(skewed_features))
    
    # Standardize features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("\nStandardized all features")
    
    return df, scaler, missing_analysis

def save_processed_files(df, scaler, missing_analysis, output_path):
    """
    Save all necessary processed files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save cleaned dataset
    df.to_parquet(f"{output_path}/cleaned_train.parquet", index=False)
    
    # Save missing data analysis as separate files or pad the lists to make them equal length
    missing_info = {}
    max_length = max(len(v) for v in [
        missing_analysis['completely_missing'],
        missing_analysis['high_missing'],
        missing_analysis['moderate_missing'],
        missing_analysis['low_missing']
    ])
    
    # Pad each list with None to make them equal length
    missing_info = {
        'completely_missing': missing_analysis['completely_missing'] + [None] * (max_length - len(missing_analysis['completely_missing'])),
        'high_missing': missing_analysis['high_missing'] + [None] * (max_length - len(missing_analysis['high_missing'])),
        'moderate_missing': missing_analysis['moderate_missing'] + [None] * (max_length - len(missing_analysis['moderate_missing'])),
        'low_missing': missing_analysis['low_missing'] + [None] * (max_length - len(missing_analysis['low_missing']))
    }
    
    # Save as DataFrame
    pd.DataFrame(missing_info).to_csv(f"{output_path}/missing_data_info.csv", index=False)
    
    # Save scaler parameters
    scaler_params = {
        'mean': scaler.mean_,
        'scale': scaler.scale_
    }
    pd.DataFrame(scaler_params).to_csv(f"{output_path}/scaler_params.csv", index=False)
    
    # Also save a summary of missing data analysis
    missing_summary = {
        'category': ['Completely Missing', 'High Missing', 'Moderate Missing', 'Low Missing'],
        'count': [
            len(missing_analysis['completely_missing']),
            len(missing_analysis['high_missing']),
            len(missing_analysis['moderate_missing']),
            len(missing_analysis['low_missing'])
        ]
    }
    pd.DataFrame(missing_summary).to_csv(f"{output_path}/missing_data_summary.csv", index=False)
    
    print(f"\nSaved processed files to {output_path}")
    
    # Print summary of missing data
    print("\nMissing Data Summary:")
    print(pd.DataFrame(missing_summary))

# Main execution
if __name__ == "__main__":
    # Set paths
    input_path = "/kaggle/input/jane-street-real-time-market-data-forecasting"
    output_path = "/kaggle/working/processed_data"
    
    # Process data
    print("Starting data processing...")
    df_cleaned, scaler, missing_analysis = load_and_clean_data(input_path, num_partitions=2)
    
    # Save processed files
    save_processed_files(df_cleaned, scaler, missing_analysis, output_path)
    
    # Display sample of cleaned data
    print("\nSample of cleaned data:")
    print(df_cleaned.round(1).head())
    
    # Display basic information about the cleaned dataset
    print("\nCleaned dataset info:")
    print(f"Shape: {df_cleaned.shape}")
    print(f"Features: {len([col for col in df_cleaned.columns if col.startswith('feature_')])}")
    print(f"Responders: {len([col for col in df_cleaned.columns if col.startswith('responder_')])}")
    print(f"Missing values: {df_cleaned.isnull().sum().sum()}")


%%time
# Load the cleaned data
cleaned_df = pd.read_parquet("/kaggle/working/processed_data/cleaned_train.parquet")

df_train = cleaned_df



# Calculate missing values percentage
missing_percentages = (df_train.isna().sum() / len(df_train) * 100).round(2)

# Create a DataFrame with the percentages
missing_df = pd.DataFrame({
    'Column': missing_percentages.index,
    'Missing_Percentage': missing_percentages.values
})

# Sort by percentage in descending order
missing_df = missing_df.sort_values('Missing_Percentage', ascending=False)

# Display only columns with missing values (optional)
missing_df = missing_df[missing_df['Missing_Percentage'] > 0]

print(f"Missing values percentage over {len(df_train)} samples:")
print(missing_df.to_string(index=False))


# Load the current cleaned dataset
df = pd.read_parquet("/kaggle/working/processed_data/cleaned_train.parquet")

# Handle missing values in feature_04
median_value = df['feature_04'].median()
df['feature_04'] = df['feature_04'].fillna(median_value)

# Verify no missing values remain
missing_check = (df.isna().sum() / len(df) * 100).round(2)
print("Missing values percentage after cleaning:")
print(missing_check[missing_check > 0])

# Save the updated dataset, replacing the old one
df.to_parquet("/kaggle/working/processed_data/cleaned_train.parquet")

print("\nDataset updated and saved. No missing values remain.")


import pandas as pd
import numpy as np
from scipy.stats import skew

# Load the cleaned data
cleaned_df = pd.read_parquet("/kaggle/working/processed_data/cleaned_train.parquet")

# Step 1: Check if columns with 100% missing values are removed
removed_columns = ['feature_21', 'feature_26', 'feature_27', 'feature_31']
missing_columns = [col for col in removed_columns if col in cleaned_df.columns]
if not missing_columns:
    print("âœ… Columns with 100% missing values were successfully removed.")
else:
    print(f"â�Œ These columns were not removed: {missing_columns}")

# Step 2: Check high-missing features for imputation or removal
high_missing_features = ['feature_00', 'feature_01', 'feature_02', 'feature_03', 'feature_04']
for feature in high_missing_features:
    if feature in cleaned_df.columns:
        missing_percentage = cleaned_df[feature].isna().mean() * 100
        if missing_percentage == 0:
            print(f"âœ… Missing values in {feature} were handled correctly.")
        else:
            print(f"â�Œ {feature} still has {missing_percentage:.2f}% missing values.")
    else:
        print(f"âœ… {feature} was removed from the dataset.")

# Step 3: Check responder columns and their correlation with the target

target_column = "responder_6"
responder_columns = [col for col in cleaned_df.columns if "responder" in col]

if responder_columns:
    for col in responder_columns:
        correlation = cleaned_df[col].corr(cleaned_df[target_column])
        print(f"Correlation of {col} with {target_column}: {correlation:.4f}")
else:
    print("âœ… No responder columns are present or they were already validated.")

# Step 4: Check skewness and normalization/scaling of specific features
# Replace with the actual list of skewed features
skewed_features = ['feature_39', 'feature_42']
for feature in skewed_features:
    if feature in cleaned_df.columns:
        skewness = skew(cleaned_df[feature].dropna())
        print(f"Skewness of {feature}: {skewness:.4f}")
        if abs(skewness) < 1:
            print(f"âœ… {feature} has been normalized/scaled successfully.")
        else:
            print(f"â�Œ {feature} might still be skewed (Skewness: {skewness:.4f}).")
    else:
        print(f"âœ… {feature} was removed from the dataset.")

# Final missing value check (overall validation)
missing_percentages = (cleaned_df.isna().sum() / len(cleaned_df) * 100).round(2)
remaining_missing = missing_percentages[missing_percentages > 0]

if remaining_missing.empty:
    print("âœ… No missing values found in the cleaned dataset.")
else:
    print(f"â�Œ Missing values still exist in these columns:\n{remaining_missing}")



%%time
# Load the cleaned data
cleaned_df = pd.read_parquet("/kaggle/working/processed_data/cleaned_train.parquet")

df_train = cleaned_df



# Calculate missing values percentage
missing_percentages = (df_train.isna().sum() / len(df_train) * 100).round(2)

# Create a DataFrame with the percentages
missing_df = pd.DataFrame({
    'Column': missing_percentages.index,
    'Missing_Percentage': missing_percentages.values
})

# Sort by percentage in descending order
missing_df = missing_df.sort_values('Missing_Percentage', ascending=False)

# Display only columns with missing values (optional)
missing_df = missing_df[missing_df['Missing_Percentage'] > 0]

print(f"Missing values percentage over {len(df_train)} samples:")
print(missing_df.to_string(index=False))


import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(cleaned_df['feature_39'], label="Feature 39", fill=True)
sns.kdeplot(cleaned_df['feature_42'], label="Feature 42", fill=True)
plt.legend()
plt.show()



# Compute correlation of all features with responder_6
correlations_with_target = cleaned_df.corr()['responder_6'].sort_values(ascending=False)

# Display correlations (excluding responder_6 itself)
correlations_with_target = correlations_with_target.drop('responder_6')

# Thresholds for strong, moderate, and weak correlations
strong_threshold = 0.5
moderate_threshold = 0.3

# Identify strong, moderate, and weak correlations
strong_corr = correlations_with_target[correlations_with_target >= strong_threshold]
moderate_corr = correlations_with_target[(correlations_with_target < strong_threshold) & (correlations_with_target >= moderate_threshold)]
weak_corr = correlations_with_target[correlations_with_target < moderate_threshold]

print("Strong Correlations with responder_6:")
print(strong_corr)

print("\nModerate Correlations with responder_6:")
print(moderate_corr)

print("\nWeak Correlations with responder_6:")
print(weak_corr)



# Correlation of each responder column with responder_6
target_column = 'responder_6'
correlation_with_target = {}

for col in responder_columns:
    if col != target_column:
        correlation_with_target[col] = cleaned_df[col].corr(cleaned_df[target_column])

# Sort by correlation strength
sorted_correlation = sorted(correlation_with_target.items(), key=lambda x: abs(x[1]), reverse=True)

print("Correlation of responders with responder_6:")
for feature, corr in sorted_correlation:
    print(f"{feature}: {corr:.4f}")









'''from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Define features (excluding responder_6) and target
X = cleaned_df[responder_columns].drop(columns=['responder_6'])
y = cleaned_df['responder_6']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Feature Importance for predicting responder_6:")
print(feature_importance)
'''





ENSEMBLE_SOLUTIONS = ['SOLUTION_14','SOLUTION_5']
OPTION,__WTS = 'option 91',[0.899, 0.28]


def predict(test:pl.DataFrame, lags:pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:    
    pdB = predict_14(test,lags).to_pandas()
    pdC = predict_5 (test,lags).to_pandas()

    pdB = pdB.rename(columns={'responder_6':'responder_B'})
    pdC = pdC.rename(columns={'responder_6':'responder_C'})
    pds = pd.merge(pdB,pdC, on=['row_id'])
    pds['responder_6'] =\
        pds['responder_B'] *__WTS[0] +\
        pds['responder_C'] *__WTS[1] 

    display(pds)
    predictions = test.select('row_id', pl.lit(0.0).alias('responder_6'))
    pred = pds['responder_6'].to_numpy()
    predictions = predictions.with_columns(pl.Series('responder_6', pred.ravel()))
    return predictions


if 'SOLUTION_14' in ENSEMBLE_SOLUTIONS:    
    
    lags_ : pl.DataFrame | None = None

    def predict_14(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
        global lags_
        if lags is not None:
            lags_ = lags

        predictions_14 = test.select(
            'row_id',
            pl.lit(0.0).alias('responder_6'),
        )
        symbol_ids = test.select('symbol_id').to_numpy()[:, 0]

        if not lags is None:
            lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last() # pick up last record of previous date
            test = test.join(lags, on=["date_id", "symbol_id"],  how="left")
        else:
            test = test.with_columns(
                ( pl.lit(0.0).alias(f'responder_{idx}_lag_1') for idx in range(9) )
            )

        preds = np.zeros((test.shape[0],))
        preds += xgb_model.predict(test[xgb_feature_cols].to_pandas()) / 2
        test_input = test[CONFIG.feature_cols].to_pandas()
        test_input = test_input.fillna(method = 'ffill').fillna(0)
        test_input = torch.FloatTensor(test_input.values).to("cuda:0")
        with torch.no_grad():
            for i, nn_model in enumerate(tqdm(models)):
                nn_model.eval()
                preds += nn_model(test_input).cpu().numpy() / 10
        print(f"predict> preds.shape =", preds.shape)

        predictions_14 = \
        test.select('row_id').\
        with_columns(
            pl.Series(
                name   = 'responder_6', 
                values = np.clip(preds, a_min = -5, a_max = 5),
                dtype  = pl.Float64,
            )
        )

        # The predict function must return a DataFrame
        #assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
        # with columns 'row_id', 'responer_6'
        #assert list(predictions.columns) == ['row_id', 'responder_6']
        # and as many rows as the test data.
        #assert len(predictions) == len(test)

        return predictions_14

