import pandas as pd
import seaborn as sns
import numpy as np

podcast = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
podcast.head()


podcast.shape


podcast.info()


# podcast = podcast.dropna()



# podcast['Guest_Popularity_percentage'] = podcast['Guest_Popularity_percentage'].fillna(method='ffill')
# podcast['Guest_Popularity_percentage'] = podcast['Guest_Popularity_percentage'].fillna(method='bfill')



# podcast['Episode_Length_minutes'] = podcast['Episode_Length_minutes'].fillna(method='ffill')
# podcast['Episode_Length_minutes'] = podcast['Episode_Length_minutes'].fillna(method='bfill')



# podcast['Number_of_Ads'] = podcast['Number_of_Ads'].fillna(method='ffill')
# podcast['Number_of_Ads'] = podcast['Number_of_Ads'].fillna(method='bfill')





podcast.drop(columns=["id",'Podcast_Name','Episode_Title'], inplace=True)


podcast = pd.get_dummies(podcast, drop_first=False, dtype=int)


!pip install miceforest


from miceforest import ImputationKernel

podcast = podcast.reset_index(drop=True)

mice_kernel = ImputationKernel(
data = podcast,
)

mice_kernel.mice(2)
podcast = mice_kernel.complete_data()


podcast.head()


podcast.info()


podcast.describe()


sns.boxenplot(x=podcast["Episode_Length_minutes"])


podcast[podcast["Episode_Length_minutes"]>121]


podcast.drop(podcast[podcast["Episode_Length_minutes"] > 121].index, inplace=True)


sns.boxenplot(x=podcast["Episode_Length_minutes"])


podcast[podcast["Episode_Length_minutes"]<5]


podcast.drop(podcast[podcast["Episode_Length_minutes"] < 5].index, inplace=True)


sns.boxenplot(x=podcast["Host_Popularity_percentage"])


podcast[podcast["Host_Popularity_percentage"]>100]


podcast[podcast["Host_Popularity_percentage"]<20]


podcast.drop(podcast[podcast["Host_Popularity_percentage"] > 100].index, inplace=True)


podcast.drop(podcast[podcast["Host_Popularity_percentage"] < 20].index, inplace=True)


sns.boxenplot(x=podcast["Host_Popularity_percentage"])


podcast[podcast["Guest_Popularity_percentage"]>100]


podcast[podcast["Guest_Popularity_percentage"]<0.01]


podcast.drop(podcast[podcast["Guest_Popularity_percentage"] > 100].index, inplace=True)


podcast.drop(podcast[podcast["Guest_Popularity_percentage"] < 0.01].index, inplace=True)


podcast[podcast["Number_of_Ads"]>3]


podcast.drop(podcast[podcast["Number_of_Ads"] > 3].index, inplace=True)


sns.boxenplot(x=podcast["Listening_Time_minutes"])


podcast[podcast["Listening_Time_minutes"]>119]


podcast.describe()


podcast.info()


import seaborn as sns
import matplotlib.pyplot as plt


# Sum dummy genre columns
genre_counts = podcast.filter(like='Genre_').sum()

# Manually add the dropped first genre
dropped_genre = 'Other'
dropped_count = len(podcast) - genre_counts.sum()  # Remaining rows must be that genre
genre_counts[dropped_genre] = dropped_count

# Clean labels
genre_counts.index = genre_counts.index.str.replace('Genre_', '')
genre_counts = genre_counts.sort_values(ascending=False)

# Plot
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Genre Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Sum dummy genre columns
genre_counts = podcast.filter(like='Publication_Day_').sum()

# Manually add the dropped first genre
dropped_genre = 'Other'
dropped_count = len(podcast) - genre_counts.sum()  # Remaining rows must be that genre
genre_counts[dropped_genre] = dropped_count

# Clean labels
genre_counts.index = genre_counts.index.str.replace('Publication_Day_', '')
genre_counts = genre_counts.sort_values(ascending=False)

# Plot
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Publication_Day Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Sum dummy genre columns
genre_counts = podcast.filter(like='Publication_Time_').sum()

# Manually add the dropped first genre
dropped_genre = 'Other'
dropped_count = len(podcast) - genre_counts.sum()  # Remaining rows must be that genre
genre_counts[dropped_genre] = dropped_count

# Clean labels
genre_counts.index = genre_counts.index.str.replace('Publication_Time_', '')
genre_counts = genre_counts.sort_values(ascending=False)

# Plot
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Publication_Time Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Sum dummy genre columns
genre_counts = podcast.filter(like='Episode_Sentiment_').sum()

# Manually add the dropped first genre
dropped_genre = 'Other'
dropped_count = len(podcast) - genre_counts.sum()  # Remaining rows must be that genre
genre_counts[dropped_genre] = dropped_count

# Clean labels
genre_counts.index = genre_counts.index.str.replace('Episode_Sentiment_', '')
genre_counts = genre_counts.sort_values(ascending=False)

# Plot
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Episode_Sentiment Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


podcast.corr()


podcast = podcast.assign(
    host_gest_popularity = (podcast['Host_Popularity_percentage'] + podcast['Guest_Popularity_percentage'])/2,
    episode_Length_over_ads = podcast['Episode_Length_minutes'] / (podcast['Number_of_Ads'] + 1),
    
)


sns.lmplot(podcast, x='host_gest_popularity', y='Listening_Time_minutes')


sns.lmplot(podcast, x='episode_Length_over_ads', y='Listening_Time_minutes')


podcast.corr()


from sklearn.model_selection import train_test_split

X = podcast.drop(['Listening_Time_minutes'], axis=1)
y = podcast["Listening_Time_minutes"]

# Test Split
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X.head()


from sklearn.preprocessing import StandardScaler

std = StandardScaler()
X_tr = std.fit_transform(X)
X_te = std.transform(X_test)


from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_error as mae

n_alphas = 200
alphas = 10 ** np.linspace(-3, 3, n_alphas)

ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_tr, y)

print(f"Best alpha: {ridge.alpha_}")
print(f"Training R2: {r2(y, ridge.predict(X_tr))}")
print(f"Training MAE: {mae(y, ridge.predict(X_tr))}")
print(f"Test R2: {r2(y_test, ridge.predict(X_te))}")
print(f"Test MAE: {mae(y_test, ridge.predict(X_te))}")


from sklearn.linear_model import LassoCV

n_alphas = 5000
alphas = 10 ** np.linspace(-4, 4, n_alphas)

lasso = LassoCV(alphas=alphas, cv=5)
lasso.fit(X_tr, y)

print(f"Best alpha: {lasso.alpha_}")
print(f"Training R2: {r2(y, lasso.predict(X_tr))}")
print(f"Training MAE: {mae(y, lasso.predict(X_tr))}")
print(f"Test R2: {r2(y_test, lasso.predict(X_te))}")
print(f"Test MAE: {mae(y_test, lasso.predict(X_te))}")


list(zip(X.columns, lasso.coef_))

