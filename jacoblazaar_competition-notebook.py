import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import random
import os
import gc
import warnings
import time
from typing import List
from math import sqrt
import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


import optuna

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from catboost import CatBoost, CatBoostRegressor
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from copy import deepcopy
from sklearn.metrics import mean_squared_error

rc = {
    "axes.facecolor": "#243139",
    "figure.facecolor": "#243139",
    "axes.edgecolor": "#000000",
    "grid.color": "#000000",
    "font.family": "arial",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#FFFFFF",
    "ytick.color": "#FFFFFF",
    "grid.alpha": 0.4,
}
sns.set(rc=rc)
#sns.set_palette("YlOrRd")

# Useful line of code to set the display option so we could see all the columns in pd dataframe
pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def print_sl():
    print("=" * 50)
    print()


train_PATH    = '/kaggle/input/playground-series-s5e2/train.csv'
train_ex_PATH = '/kaggle/input/playground-series-s5e2/training_extra.csv'
test_PATH     = '/kaggle/input/playground-series-s5e2/test.csv'
sub_PATH      = '/kaggle/input/playground-series-s5e2/sample_submission.csv'

train_df      = pd.read_csv(train_PATH)
train_ex_df   = pd.read_csv(train_ex_PATH)
test_df       = pd.read_csv(test_PATH)
sub_df        = pd.read_csv(sub_PATH)

train_df.drop('id',axis=1,inplace=True)
train_ex_df.drop('id',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)

print('Data Loaded Succesfully!')
print_sl()

print(f'Train Data Shape: {train_df.shape}')
print(f'Are there any null values in train? - {train_df.isnull().any().any()}\n')

print(f'Train Data Shape: {train_ex_df.shape}')
print(f'Are there any null values in train? - {train_ex_df.isnull().any().any()}\n')

print(f'Test Data Shape:  {test_df.shape}')
print(f'Are there any null values in test? - {test_df.isnull().any().any()}\n')
print_sl()

target = 'Price'

train_df.head()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df[target], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Prices in train_df', color='white')
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df[target], ax=axes[1])
axes[1].set_title('Box plot of Prices in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_ex_df[target], bins=7, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Prices in train_ex_df', color='white')
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_ex_df[target], ax=axes[1])
axes[1].set_title('Box plot of Prices in train_ex_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


train_ex_df['Compartments'].unique()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['Compartments'], bins=10, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Compartments in train_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['Compartments'], ax=axes[1])
axes[1].set_title('Box plot of Compartments in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_ex_df['Compartments'], bins=10, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Compartments in train_ex_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_ex_df['Compartments'], ax=axes[1])
axes[1].set_title('Box plot of Compartments in train_ex_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(test_df['Compartments'], bins=10, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Compartments in test_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=test_df['Compartments'], ax=axes[1])
axes[1].set_title('Box plot of Compartments in test_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_df['Weight Capacity (kg)'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Weight Capacity (kg) Distribution in train_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_df['Weight Capacity (kg)'], ax=axes[1])
axes[1].set_title('Box plot of Weight Capacity (kg) in train_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(train_ex_df['Weight Capacity (kg)'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Weight Capacity (kg) Distribution in train_ex_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=train_ex_df['Weight Capacity (kg)'], ax=axes[1])
axes[1].set_title('Box plot of Weight Capacity (kg) in train_ex_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Adjust the figure and axes creation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram plot
sns.histplot(test_df['Weight Capacity (kg)'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Weight Capacity (kg) Distribution in test_df', color='white')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot
sns.boxplot(x=test_df['Weight Capacity (kg)'], ax=axes[1])
axes[1].set_title('Box plot of Weight Capacity (kg) in test_df', color='white')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Create a copy of the dataframe
df_encoded = pd.concat([train_ex_df, train_df], axis=0).reset_index(drop=True).copy()

# Assuming these are your categorical variables, including 'outcome'
categorical_vars = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof', 'Style', 'Color']

# Label encode categorical columns
label_encoders = {}
for column in categorical_vars:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le

def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str = 'Train correlation') -> None:
    excluded_columns = ['id']
    columns_without_excluded = [col for col in df.columns if col not in excluded_columns]
    corr = df[columns_without_excluded].corr()
    
    fig, axes = plt.subplots(figsize=(14, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='mako', annot=True, annot_kws={"size": 6})
    plt.title(title_name, color='white')
    plt.show()

# Plot correlation heatmap for encoded dataframe
plot_correlation_heatmap(df_encoded, 'Encoded Dataset Correlation')


def plot_count(df: pd.core.frame.DataFrame, col: str, title_name: str='Train') -> None:
    # Set background color
    f, ax = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.2)

    s1 = df[col].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    colors = sns.color_palette("mako")
    # hex_colors = [matplotlib.colors.to_hex(color) for color in colors]
    # print(hex_colors)
    
    outer_colors = ['#2e1e3b', '#413d7b', '#37659e', '#348fa7', '#40b7ad', '#8bdab2']
    inner_colors = ['#2e1e3b', '#413d7b', '#37659e', '#348fa7', '#40b7ad', '#8bdab2']
    #inner_colors = ['#59b3a3',] #'#433C64']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90, frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={'linewidth' : 1, 'edgecolor' : 'black'}, 
        textprops={'fontsize': 12, 'weight': 'bold', 'color': 'white'}
    )

    textprops = {
        'size': 13, 
        'weight': 'bold', 
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%', explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', fc='#243139', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = s1.index.tolist()
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette=colors, orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,       
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='white', fontweight='bold', fontsize=12)

    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col, fontweight="bold", color='white')
    ax[1].set_ylabel('count', fontweight="bold", color='white')

    f.suptitle(f'{title_name}', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout() 
    plt.show()


#train_tg = pd.concat([train_ex_df, train_df], axis=0).reset_index(drop=True).copy()
train_tg = train_df.reset_index(drop=True).copy()
train_tg.head()


plot_count(train_tg, 'Brand', 'Brand Distribution of Train Data')


plot_count(train_tg, 'Material', 'Material Distribution of Train Data')


plot_count(train_tg, 'Size', 'Size Distribution of Train Data')


plot_count(train_tg, 'Laptop Compartment', 'Laptop Compartment Distribution of Train Data')


plot_count(train_tg, 'Waterproof', 'Waterproof Distribution of Train Data')


plot_count(train_tg, 'Style', 'Style Distribution of Train Data')


plot_count(train_tg, 'Color', 'Color Distribution of Train Data')

