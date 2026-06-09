import numpy as np
import pandas as pd
import math
import tensorflow as tf


# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Getting rid of warnings, tip from @broccolibeef

import warnings

msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less',
    'use_inf_as_na option'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)

train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
train.head()





train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# Number of rows / columns

print(f"Shape of Train: {train.shape}")
print(f"Shape of Test: {test.shape}")


# Data types 

train.info()


# Check numerical features
# (transpose with T for better readability)

train.describe().T


# check the first 5 rows

train.head()


train.isna().sum().sort_values(ascending=False)


# determine all numerical columns

columns_numerical = [col for col in train.columns if pd.api.types.is_numeric_dtype(train[col])]


# Dynamically calculate number of rows & columns @adrien97

num_vars = len(columns_numerical)

num_cols = 2  # Keep 2 columns for readability
num_rows = math.ceil(num_vars / num_cols)  # Calculate rows dynamically


# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4.5))
axes = axes.flatten()

# Color palette (color-blind friendly)
palette = sns.color_palette("Set3", n_colors=num_vars)

# Plotting each variable
for i, var in enumerate(columns_numerical):
    sns.histplot(
        data=train,
        x=var,
        kde=True,
        color=palette[i],
        bins=50,
        edgecolor="white",
        linewidth=1.3,
        ax=axes[i]
    )
    axes[i].set_title(f"{var}", fontsize=14, weight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].tick_params(axis='x', labelrotation=15)

# Remove unused axes
#for j in range(i + 1, len(axes)):
#    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout(h_pad=2.5)
plt.show()


def handle_na(df):
    """
    fill all NAs with median value
    """
    
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median())
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].median())
    
    return df



def handle_sentiments(df):
    """
    make sentiment into categorical va
    """
    
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(
        {'Negative': -1, 
         'Neutral': 0, 
         'Positive': 1 
         })

    return df


# change dataframes 

train = handle_na(handle_sentiments(train))
test = handle_na(handle_sentiments(test))


train['Episode_Title'] = train['Episode_Title'].astype(str).str.replace('Episode ', '').astype(int)

test['Episode_Title'] = test['Episode_Title'].astype(str).str.replace('Episode ', '').astype(int)





train['Genre'].unique()


# let's count how many of those we have
genre_counts = pd.DataFrame(train['Genre'].value_counts())

genre_counts = genre_counts.reset_index()
genre_counts.columns = ['Genre', 'counts']


sns.barplot(
    x = genre_counts['Genre'],
    y = genre_counts['counts']
)

plt.title(f"Distribution of Genre")
plt.xticks(rotation=45)


train['Publication_Day'].unique()


days_ordered = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# let's count how many of those we have
publication_day_counts = pd.DataFrame(train['Publication_Day'].value_counts())

publication_day_counts = publication_day_counts.reset_index()
publication_day_counts.columns = ['Publication_Day', 'counts']

# order days 
publication_day_counts['Publication_Day'] = pd.Categorical(
    publication_day_counts['Publication_Day'],
    categories=days_ordered,
    ordered=True
)


sns.barplot(
    x = publication_day_counts['Publication_Day'],
    y = publication_day_counts['counts']
)

plt.title(f"Distribution of Publication_Day")
plt.xticks(rotation=45)


train['Publication_Time'].unique()


time_ordered = ['Morning', 'Afternoon', 'Evening', 'Night']


# let's count them
publication_time_counts = pd.DataFrame(train['Publication_Time'].value_counts())

publication_time_counts = publication_time_counts.reset_index()
publication_time_counts.columns = ['Publication_Time', 'counts']

# order days 
publication_time_counts['Publication_Time'] = pd.Categorical(
    publication_time_counts['Publication_Time'],
    categories=time_ordered,
    ordered=True
)


sns.barplot(
    x = publication_time_counts['Publication_Time'],
    y = publication_time_counts['counts']
)

plt.title(f"Distribution of Publication_Time")
plt.xticks(rotation=45)


# scatterplot

sns.scatterplot(
    x = train['Episode_Length_minutes'], 
    y=train["Listening_Time_minutes"], 
    alpha=0.25,
    color = 'darkorchid'
    )


plt.title(f"Episode_Length_minutes vs. Listening_Time_minutes")
plt.xlabel('Episode_Length_minutes')
plt.ylabel("Listening_Time_minutes")
plt.show()


sns.scatterplot(
    x = train[train['Episode_Length_minutes'] < 150]['Episode_Length_minutes'], 
    y=train["Listening_Time_minutes"], 
    alpha=0.25,
    color = 'darkorchid'
    )


plt.title(f"Episode_Length_minutes vs. Listening_Time_minutes")
plt.xlabel('Episode_Length_minutes')
plt.ylabel("Listening_Time_minutes")
plt.show()


# scatterplot

sns.scatterplot(
    x = train['Host_Popularity_percentage'], 
    y=train["Listening_Time_minutes"], 
    alpha=0.25,
    color = 'darkorchid'
    )


plt.title(f"Host_Popularity_percentage vs. Listening_Time_minutes")
plt.xlabel('Host_Popularity_percentage')
plt.ylabel("Listening_Time_minutes")
plt.show()

