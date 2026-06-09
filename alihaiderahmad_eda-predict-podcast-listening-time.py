import numpy as np  
import pandas as pd   
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv(os.path.join(dirname, "train.csv"), index_col='id')


train_df.head()


train_df.info()


# non-numeric columns 
non_num_df = train_df.select_dtypes(exclude=['number']).columns.to_list()
train_df[non_num_df].nunique()


# numerical columns (int & float)
num_df = train_df.select_dtypes(include=['number']).columns.to_list()
num_df


train_df[num_df].describe().T


train_df.isnull().sum()


missing_percentage = (train_df.isnull().sum() / len(train_df)) * 100
missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
print(missing_percentage.to_string())


# Visualizing missing values
plt.figure(figsize=(12,6))
sns.heatmap(train_df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()


nan_rows = train_df[train_df.isnull().any(axis=1)]
train_df.dropna(inplace=True)


print(f"Total rows with missing values: {nan_rows.shape[0]}")


ep_len_missing = nan_rows[nan_rows['Episode_Length_minutes'].isnull()]
guest_pop_missing = nan_rows[nan_rows['Guest_Popularity_percentage'].isnull()]


# Check if Listening_Time_minutes is also zero when Episode_Length_minutes is Nan
ep_len_missing['Listening_Time_minutes'].describe()


# Check if Missingness is Related to Categorical Features
for cat in ["Podcast_Name", 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']:
    if cat in nan_rows.columns:
        plt.figure(figsize=(12, 4))
        df_grouped = nan_rows.groupby(cat)["Episode_Length_minutes"].apply(lambda x: x.isnull().mean())
        df_grouped.sort_values().plot(kind="bar", color="teal")
        plt.title(f"Missing Rate of Episode_Length_minutes by {cat}")
        plt.xlabel(cat)
        plt.ylabel("Missing Percentage")
        plt.show()


# Check If Missingness Affects Listening Time
plt.figure(figsize=(8, 6))
sns.boxplot(x=nan_rows["Episode_Length_minutes"].isnull(), y=nan_rows["Listening_Time_minutes"], palette=["lightblue", "salmon"])
plt.title("Effect of Missing Episode Length on Listening Time")
plt.xlabel("Episode Length Missing")
plt.ylabel("Listening Time (minutes)")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=nan_rows["Guest_Popularity_percentage"].isnull(), y=nan_rows["Listening_Time_minutes"], palette=["lightblue", "salmon"])
plt.title("Effect of Missing Guest Popularity on Listening Time")
plt.xlabel("Guest Popularity Missing")
plt.ylabel("Listening Time (minutes)")
plt.show()


numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']
categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
text_cols = ['Episode_Title']


plt.figure(figsize=(12, 4))
n_bins = int(np.ceil((train_df['Listening_Time_minutes'].max() + 5))/5)
sns.histplot(train_df['Listening_Time_minutes'], kde=True, bins= n_bins)
plt.title('Distribution of Listening Time')
plt.xlabel('Listening Time (minutes)')
plt.xticks(np.arange(0, n_bins * 5, 5))
plt.ylabel('Count')
print("Skewness:", skew(train_df['Listening_Time_minutes']))


plt.figure(figsize=(12, 4))
sns.histplot(train_df['Episode_Length_minutes'], kde=True, bins= 50)
plt.title('Distribution of Episode Length Minutes')
plt.xlabel('Episode Length (minutes)')
plt.xticks(np.arange(0, 125, 15))
plt.ylabel('Count')
print("Skewness:", skew(train_df['Episode_Length_minutes']))


plt.figure(figsize=(12, 4))
sns.histplot(train_df[train_df['Host_Popularity_percentage']<=100]['Host_Popularity_percentage'], kde=True, bins= 20)
plt.title('Distribution of Host Popularity Percentage')
plt.xlabel('Host_Popularity (percentage)')
plt.ylabel('Count')
plt.xticks(np.arange(0, 105, 5))
print("Skewness:", skew(train_df[train_df['Host_Popularity_percentage']<=100]['Host_Popularity_percentage']))


plt.figure(figsize=(12, 4))
sns.histplot(train_df[train_df['Guest_Popularity_percentage']<=100]['Guest_Popularity_percentage'], kde=True, bins= 20)
plt.title('Distribution of Guest Popularity Percentage')
plt.xlabel('Guest__Popularity (percentage)')
plt.ylabel('Count')
plt.xticks(np.arange(0, 105, 5))
print("Skewness:", skew(train_df['Guest_Popularity_percentage']))


plt.figure(figsize=(12, 6))
sns.kdeplot(data=train_df[train_df['Number_of_Ads']<=3]['Number_of_Ads'],shade=True)
plt.title('Distribution of Number_of_Ads')
plt.xlabel('Number_of_Ads')
plt.ylabel('Count')
print("Skewness:", skew(train_df[train_df['Number_of_Ads']<=3]['Number_of_Ads']))


plt.figure(figsize=(10, 6))
sns.heatmap(train_df[numerical_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")


for col in numerical_cols[:-2]:  
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=train_df[col], y=train_df['Listening_Time_minutes'])
    plt.title(f'{col} vs Listening Time')


train_df[train_df['Episode_Length_minutes'] == 0].shape[0]


nan_rows[nan_rows['Episode_Length_minutes'] == 0].shape[0]


sns.regplot(x=train_df['Episode_Length_minutes'], y=train_df['Listening_Time_minutes'])


# Filter cases where listening time exceeds the episode length
temp_df = train_df[train_df['Listening_Time_minutes'] > train_df['Episode_Length_minutes']]
print(f'The percentage of Filtered rows: {temp_df.shape[0]/train_df.shape[0] * 100}%')
temp_df[['Listening_Time_minutes', 'Episode_Length_minutes']].describe().T


sns.histplot(temp_df['Episode_Length_minutes'], bins=30, kde=True)
plt.title("Episode Lengths for Invalid Rows")
plt.xticks(np.arange(0, 125, 15))
plt.show()


sns.scatterplot(x=temp_df['Episode_Length_minutes'], y=temp_df['Listening_Time_minutes'])
plt.title('Episode_Length_minutes vs Listening Time')


num_cols = len(categorical_cols)
fig, axes = plt.subplots(nrows=num_cols, ncols=2, figsize=(16, 5 * num_cols))
axes = axes.flatten()
 
for i, cat in enumerate(categorical_cols):
    sns.countplot(data=train_df, x=cat, ax=axes[2 * i])
    axes[2 * i].set_title(f'Count of Episodes by {cat}')
    axes[2 * i].tick_params(axis='x', rotation=90)
    
    sns.boxplot(x=train_df[cat], y=train_df['Listening_Time_minutes'], ax=axes[2 * i + 1])
    axes[2 * i + 1].set_title(f'{cat} vs Listening Time')
    axes[2 * i + 1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
axes = axes.flatten()
sns.countplot(data=train_df, x='Number_of_Ads', ax=axes[0])
sns.boxplot(x=train_df[train_df['Number_of_Ads']<=4]['Number_of_Ads'], y=train_df['Listening_Time_minutes'], ax=axes[1])


train_df['Episode_Title'].nunique(), train_df['Episode_Title'].shape[0]


train_df['Episode_Title'].str.len().describe()


episode_stats = train_df.groupby('Episode_Title')['Listening_Time_minutes'].mean().sort_values()
episode_stats.plot(kind='barh', figsize=(8, 20), title='Avg Listening Time per Episode')


print(train_df['Episode_Title'].value_counts().head(10))

