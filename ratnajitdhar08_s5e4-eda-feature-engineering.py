import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import os
import math
import warnings
warnings.filterwarnings('ignore')

from tabulate import tabulate
from scipy.stats import f_oneway


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.head(5)


train[train['Listening_Time_minutes'] > train['Episode_Length_minutes']]


train = train[(train['Listening_Time_minutes'] <= train['Episode_Length_minutes']) | (train['Episode_Length_minutes'].isna())]



print("Train Dataset")

info = []
for col in train.columns:
    info.append([col, train[col].count(), train[col].dtype])

headers = ["Column", "Non-Null Count", "Datatype"]

print(tabulate(info, headers=headers, tablefmt="fancy_grid"))
print()
print()
print("Test Dataset")

info = []
for col in test.columns:
    info.append([col, test[col].count(), test[col].dtype])

headers = ["Column", "Non-Null Count", "Datatype"]

print(tabulate(info, headers=headers, tablefmt="fancy_grid"))


print("Train Dataset")
info = []
for col in train.columns:
    if train[col].isnull().sum() > 0:
        info.append([col, train[col].isnull().sum()])

headers = ["Column", "Null Value Counts"]

print(tabulate(info, headers=headers, tablefmt="fancy_grid"))
print()
print()
print("Test Dataset")
info = []
for col in test.columns:
    if test[col].isnull().sum() > 0:
        info.append([col, test[col].isnull().sum()])

headers = ["Column", "Null Value Counts"]

print(tabulate(info, headers=headers, tablefmt="fancy_grid"))


print("Train Dataset")
info = []
for col in train.columns:
    info.append([col, train[col].nunique()])

headers = ["Column", "Unique Values"]
print(tabulate(info, headers=headers, tablefmt="fancy_grid"))
print("\n\n")

print("Test Dataset")
info = []
for col in test.columns:
    info.append([col, test[col].nunique()])

print(tabulate(info, headers=headers, tablefmt="fancy_grid"))



train = train.drop(columns=['id'], axis=1)
num_col = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Listening_Time_minutes"
]

cat_col = [col for col in train.columns if col not in num_col]

print(f"Numerical Columns:")
print(num_col)
print()
print()
print(f"Categorical Columns:")
print(cat_col)


num_cat_col = len(cat_col)
cols = 2
rows = math.ceil((2*num_cat_col)/cols)

plt.figure(figsize=(20, rows*6))

for i, col in enumerate(cat_col):
    plt.subplot(rows, cols, 2*i+1)
    sns.countplot(x=col, data=train)
    plt.title(f"Countplot of {col} - Train")
    plt.xlabel(col)
    plt.xticks(rotation=90)
    plt.ylabel('Count')

    plt.subplot(rows, cols, 2*i+2)
    sns.countplot(x=col, data=test)
    plt.title(f"Countplot of {col} - Train")
    plt.xlabel(col)
    plt.xticks(rotation=90)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()


train['Number_of_Ads'] = train['Number_of_Ads'].apply(lambda x: 1 if x > 3.0 else x)
test['Number_of_Ads'] = test['Number_of_Ads'].apply(lambda x: 1 if x > 3.0 else x)


num_num_col = len(num_col)
cols = 4  # 4 plots per row (Hist Train, Hist Test, Box Train, Box Test)
rows = math.ceil(num_num_col)

plt.figure(figsize=(20, rows * 4))

for i, col in enumerate(num_col):
    # Histogram for Train
    plt.subplot(rows, cols, 4 * i + 1)
    sns.histplot(train[col], kde=True, bins=30, color='blue')
    plt.title(f"Histogram of {col} - Train")
    plt.xlabel(col)
    plt.ylabel("Density")

    # Histogram for Test (Skip Listening_Time_minutes)
    if col != "Listening_Time_minutes":
        plt.subplot(rows, cols, 4 * i + 2)
        sns.histplot(test[col], kde=True, bins=30, color='green')
        plt.title(f"Histogram of {col} - Test")
        plt.xlabel(col)
        plt.ylabel("Density")

    # Boxplot for Train
    plt.subplot(rows, cols, 4 * i + 3)
    sns.boxplot(x=train[col], color='blue')
    plt.title(f"Boxplot of {col} - Train")
    plt.xlabel(col)

    # Boxplot for Test (Skip Listening_Time_minutes)
    if col != "Listening_Time_minutes":
        plt.subplot(rows, cols, 4 * i + 4)
        sns.boxplot(x=test[col], color='green')
        plt.title(f"Boxplot of {col} - Test")
        plt.xlabel(col)

plt.tight_layout()
plt.show()


test[test['Episode_Length_minutes'] > 150]


test.loc[test['Episode_Length_minutes'] > 150, 'Episode_Length_minutes'] = np.nan


train.loc[train['Episode_Length_minutes'] > 150, 'Episode_Length_minutes'] = np.nan


genres = train['Genre'].unique()

num_genres = len(genres)
cols = 3 
rows = (num_genres // cols) + (num_genres % cols > 0)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows * 3), constrained_layout=True)

axes = axes.flatten()

for i, genre in enumerate(genres):
    ax = axes[i]
    genre_data = train[train['Genre'] == genre]['Listening_Time_minutes']

    sns.histplot(genre_data, kde=True, bins=30, color='skyblue', ax=ax)

    mean_val = np.mean(genre_data)
    median_val = np.median(genre_data)
    std_val = np.std(genre_data)

    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    ax.axvline(mean_val + std_val, color='purple', linestyle='dotted', linewidth=2, label=f'Std Dev: {std_val:.2f}')
    ax.axvline(mean_val - std_val, color='purple', linestyle='dotted', linewidth=2)

    ax.set_title(f'Listening Time - {genre}', fontsize=12)
    ax.set_xlabel('Listening Time (Minutes)')
    ax.set_ylabel('Density')
    ax.legend()

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()



genres = train['Genre'].unique()

num_genres = len(genres)
cols = 3
rows = (num_genres // cols) + (num_genres % cols > 0)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows * 3), constrained_layout=True)
axes = axes.flatten()

for i, genre in enumerate(genres):
    ax = axes[i]
    genre_data = train[train['Genre'] == genre]

    sns.boxplot(data=genre_data, x='Episode_Sentiment', y='Listening_Time_minutes', ax=ax, palette='Set2', order = ['Positive', 'Neutral', 'Negative'])

    ax.set_title(f'Listening Time by Sentiment - {genre}', fontsize=12)
    ax.set_xlabel('Episode Sentiment')
    ax.set_ylabel('Listening Time (Minutes)')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()



# Get unique genres
genres = train['Genre'].unique()

# Loop through each genre and perform one-way ANOVA for Episode_Sentiment and Listening_Time_minutes
for genre in genres:
    # Filter data for the current genre
    genre_data = train[train['Genre'] == genre]
    
    # Split the data based on sentiment
    positive_data = genre_data[genre_data['Episode_Sentiment'] == 'Positive']['Listening_Time_minutes'].dropna()
    neutral_data = genre_data[genre_data['Episode_Sentiment'] == 'Neutral']['Listening_Time_minutes'].dropna()
    negative_data = genre_data[genre_data['Episode_Sentiment'] == 'Negative']['Listening_Time_minutes'].dropna()
    
    # Perform one-way ANOVA
    f_stat, p_value = f_oneway(positive_data, neutral_data, negative_data)
    
    # Check if the p-value is less than a significance level (typically 0.05)
    if p_value < 0.05:
        print(f"For genre '{genre}', listening time is significantly different across sentiment categories (p-value = {p_value:.4f})")
    else:
        print(f"For genre '{genre}', listening time is not significantly different across sentiment categories (p-value = {p_value:.4f})")



plt.figure(figsize=(10, 6))
sns.boxplot(data=train, x='Publication_Day', y='Listening_Time_minutes', palette='Set2', order= ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])

plt.title('Listening Time by Day of the Week', fontsize=14)
plt.xlabel('Day of the Week')
plt.ylabel('Listening Time (Minutes)')
plt.show()



time = train['Publication_Time'].unique()

num_time = len(time)
cols = 2
rows = (num_genres // cols) + (num_genres % cols > 0)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows*3), constrained_layout=True)

axes = axes.flatten()

for i, t in enumerate(time):
    ax = axes[i]
    time_data = train[train['Publication_Time']==t]['Listening_Time_minutes']
    sns.histplot(time_data, kde=True, bins=30, color='skyblue', ax=ax)

    mean_val = np.mean(time_data)
    med_val = np.median(time_data)
    std_val = np.std(time_data)

    ax.axvline(mean_val, color='red', linestyle='dashed', label=f'Mean: {mean_val:.2f}')
    ax.axvline(med_val, color='blue', linestyle='dashed', label=f'Median: {med_val:.2f}')
    ax.axvline(std_val, color='blue', linestyle='dashed', label=f'Standard Deviation: {std_val:.2f}')

    ax.set_title(f'Listening Time - {t}', fontsize=12)
    ax.set_xlabel('Listening Time (Minutes)')
    ax.set_ylabel('Density')
    ax.legend()

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()



plt.figure(figsize=(8,5))
sns.regplot(x=train['Host_Popularity_percentage'], y=train['Listening_Time_minutes'], 
            scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
plt.title("Host Popularity vs Listening Time")
plt.xlabel("Host Popularity (%)")
plt.ylabel("Listening Time (minutes)")
plt.show()



correlation = train['Host_Popularity_percentage'].corr(train['Listening_Time_minutes'])
print(f"Correlation coefficient: {correlation:.3f}")



plt.figure(figsize=(8,5))
sns.regplot(x=train['Guest_Popularity_percentage'], y=train['Listening_Time_minutes'], 
            scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
plt.title("Guest Popularity vs Listening Time")
plt.xlabel("Guest Popularity (%)")
plt.ylabel("Listening Time (minutes)")
plt.show()



correlation = train['Guest_Popularity_percentage'].corr(train['Listening_Time_minutes'])
print(f"Correlation coefficient: {correlation:.3f}")



combined_popularity = train['Host_Popularity_percentage'] + train['Guest_Popularity_percentage']

plt.figure(figsize=(8,5))
sns.regplot(x=combined_popularity, y=train['Listening_Time_minutes'], 
            scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
plt.title("Combined Popularity (Host + Guest) vs Listening Time")
plt.xlabel("Combined Popularity (%)")
plt.ylabel("Listening Time (minutes)")
plt.show()



ad_density = train['Number_of_Ads'] / train['Episode_Length_minutes']

ad_density_bins = pd.cut(ad_density, bins=5)


plt.figure(figsize=(10,5))
sns.boxplot(x=ad_density_bins, y=train['Listening_Time_minutes'])
plt.xticks(rotation=45)
plt.title("Listening Time Distribution Across Ad Density Levels")
plt.xlabel("Ad Density (Binned)")
plt.ylabel("Listening Time (minutes)")
plt.show()



train = train[train['Number_of_Ads'].notna()]
train.isnull().sum()


train['Guest_Popularity_percentage'] = train.groupby(['Podcast_Name'])['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.mean().round(2)))
print(train.isnull().sum())



test['Guest_Popularity_percentage'] = test.groupby(['Podcast_Name'])['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.mean().round(2)))
print(test.isnull().sum())



train['Episode_Length_minutes'] = train.groupby(['Podcast_Name', 'Episode_Title'])['Episode_Length_minutes'].transform(lambda x: x.fillna(x.mean().round(2)))
print(train.isnull().sum())



test['Episode_Length_minutes'] = test.groupby(['Podcast_Name', 'Episode_Title'])['Episode_Length_minutes'].transform(lambda x: x.fillna(x.mean().round(2)))
print(test.isnull().sum())


bins = [-1, 30, 50, 80, np.inf]
labels = [0,1,2, 3]

train['Combined_Popularity'] = (train['Host_Popularity_percentage'] + train['Guest_Popularity_percentage'])/2
test['Combined_Popularity'] = (test['Host_Popularity_percentage'] + test['Guest_Popularity_percentage'])/2


train['Combined_Popularity_Level'] = pd.cut(train['Combined_Popularity'], bins, labels=labels)
test['Combined_Popularity_Level'] = pd.cut(test['Combined_Popularity'], bins, labels=labels)

train['Host_Popularity_Level'] = pd.cut(train['Host_Popularity_percentage'], bins, labels=labels)
test['Host_Popularity_Level'] = pd.cut(test['Host_Popularity_percentage'], bins, labels=labels)

train['Guest_Popularity_Level'] = pd.cut(train['Guest_Popularity_percentage'], bins, labels=labels)
test['Guest_Popularity_Level'] = pd.cut(test['Guest_Popularity_percentage'], bins, labels=labels)



train


ad_density = train['Number_of_Ads'] / train['Episode_Length_minutes']
labels = [0,1,2,3,4]
train['ad_density'] = pd.cut(ad_density, bins=5, labels=labels)
test['ad_density'] = pd.cut(ad_density, bins=5, labels=labels)



weekday_mapping = {
    "Sunday": 0, "Monday" : 1, "Tuesday" : 2, "Wednesday" : 3,
    "Thursday" : 4, "Friday" : 5, "Saturday" : 6
}

train['Publication_Day'] = train['Publication_Day'].map(weekday_mapping)
train['Publication_Day_sin'] = np.sin(2 * np.pi * train['Publication_Day']/7)
train['Publication_Day_cos'] = np.cos(2 * np.pi * train['Publication_Day']/7)
train.drop(columns = ['Publication_Day'], inplace = True)

test['Publication_Day'] = test['Publication_Day'].map(weekday_mapping)
test['Publication_Day_sin'] = np.sin(2 * np.pi * test['Publication_Day']/7)
test['Publication_Day_cos'] = np.cos(2 * np.pi * test['Publication_Day']/7)
test.drop(columns = ['Publication_Day'], inplace = True)


time_mapping = {
    "Morning":0, "Afternoon":1, "Evening":2, "Night":3
}

train['Publication_Time'] = train['Publication_Time'].map(time_mapping)
train['Publication_Time_sin'] = np.sin(2 * np.pi * train['Publication_Time']/4)
train['Publication_Time_cos'] = np.cos(2 * np.pi * train['Publication_Time']/4)
train.drop(columns=['Publication_Time'], inplace=True)

test['Publication_Time'] = test['Publication_Time'].map(time_mapping)
test['Publication_Time_sin'] = np.sin(2 * np.pi * test['Publication_Time']/4)
test['Publication_Time_cos'] = np.cos(2 * np.pi * test['Publication_Time']/4)
test.drop(columns=['Publication_Time'], inplace=True)


train.columns


train['Episode_Title'] = train['Episode_Title'].str.split(" ", expand=True)[1].astype(np.uint16)
test['Episode_Title'] = test['Episode_Title'].str.split(" ", expand=True)[1].astype(np.uint16)


train['Episode_Length_sin'] = np.sin(2 * np.pi * train['Episode_Length_minutes']/60)
test['Episode_Length_sin'] = np.sin(2 * np.pi * test['Episode_Length_minutes']/60)


train


cat_col = ['Podcast_Name', 'Genre', 'Episode_Sentiment']

for col in cat_col:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])


train


num_col = ['Podcast_Name','Episode_Title', 'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Combined_Popularity']

for col in num_col:
    scaler = StandardScaler()
    train[col] = scaler.fit_transform(train[col].values.reshape(-1,1))
    test[col] = scaler.fit_transform(test[col].values.reshape(-1,1))



train


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# LightGBM Parameters
params = {
    "n_estimators": 3500,
    "random_state": 42,
    "max_bin": 1024,
    "colsample_bytree": 0.6,
    "reg_lambda": 80,
    "verbosity": -1,
    "num_leaves": 64,  
    "max_depth": 15,  
    "learning_rate": 0.05,  
    "feature_fraction": 0.8,  
    "bagging_fraction": 0.8,  
    "lambda_l1": 0.1, 
    "lambda_l2": 0.1
}

# Features and target
X = train.drop(columns=["Listening_Time_minutes"])
y = train["Listening_Time_minutes"]
X_test = test.drop(columns=['id'])

# Initialize storage for test predictions
test_predictions = np.zeros(len(X_test))

# Initialize LightGBM model
model_lgbm = lgb.LGBMRegressor(**params)

# Train with tqdm progress tracking
with tqdm(total=params['n_estimators'], desc="Training", unit="iter") as pbar:
    model_lgbm.fit(
        X, y,
        eval_metric='rmse',
        callbacks=[lgb.callback.log_evaluation(100), lgb.callback.record_evaluation({})]
    )
    pbar.update(params['n_estimators'])

# Predict on test data
test_predictions = model_lgbm.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    "id": test["id"],  # Ensure "id" column exists in test
    "target": test_predictions  # Replace "target" with actual column name
})

# Save submission
submission.to_csv("submission.csv", index=False)
print("\nâœ… Submission file saved as 'submission.csv'")





