#Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import  make_scorer,  mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from lightgbm  import LGBMRegressor
from category_encoders import MEstimateEncoder

import warnings 
warnings.filterwarnings('ignore')


test_x = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
train_x = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')

train_x["dataset"] = 0
test_x["dataset"] = 1

train = pd.concat([train_x,test_x], axis=0).copy()

train.head()


train_x.describe()


test_x.describe()


abc = train[train["Episode_Length_minutes"] <150]["Episode_Length_minutes"].median()
train.loc[train["Episode_Length_minutes"] > 150, "Episode_Length_minutes"] = abc
train.loc[train["Episode_Length_minutes"] == 0, "Episode_Length_minutes"] = 1
train["Episode_Length_minutes"].mean()

train.loc[train["Number_of_Ads"] > 15, "Number_of_Ads"] = train["Number_of_Ads"].median()
train.loc[train["Guest_Popularity_percentage"] > 100, "Guest_Popularity_percentage"] = 100
train.loc[train["Host_Popularity_percentage"] > 100, "Host_Popularity_percentage"] = 100


train_grouped = train.groupby(['Podcast_Name']).agg({
    'Episode_Length_minutes': 'median',
    'Guest_Popularity_percentage': 'median',
    'Number_of_Ads': 'median'
}).rename(columns={
    'Episode_Length_minutes': 'Median_Episode_Length',
    'Guest_Popularity_percentage': 'Median_Guest_Popularity',
    'Number_of_Ads': 'Median_Number_of_Ads'
}).reset_index()

train_grouped1 = train.groupby(['Genre']).agg({
    'Episode_Length_minutes': 'median',
    'Guest_Popularity_percentage': 'median',
    'Number_of_Ads': 'median'
}).rename(columns={
    'Episode_Length_minutes': 'Median_Episode_Length',
    'Guest_Popularity_percentage': 'Median_Guest_Popularity',
    'Number_of_Ads': 'Median_Number_of_Ads'
}).reset_index()



podcast_to_length = dict(zip(train_grouped['Podcast_Name'], train_grouped['Median_Episode_Length']))
podcast_to_popularity = dict(zip(train_grouped['Podcast_Name'], train_grouped['Median_Guest_Popularity']))
podcast_to_ads = dict(zip(train_grouped['Podcast_Name'], train_grouped['Median_Number_of_Ads']))

genre_to_length = dict(zip(train_grouped1['Genre'], train_grouped1['Median_Episode_Length']))
genre_to_popularity = dict(zip(train_grouped1['Genre'], train_grouped1['Median_Guest_Popularity']))
genre_to_ads = dict(zip(train_grouped1['Genre'], train_grouped1['Median_Number_of_Ads']))



train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Podcast_Name'].map(podcast_to_length))
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(train['Podcast_Name'].map(podcast_to_popularity))
train['Number_of_Ads'] = train['Number_of_Ads'].fillna(train['Podcast_Name'].map(podcast_to_ads))
train["length_median_dev_podcast"] = (train["Episode_Length_minutes"] - train["Podcast_Name"].map(podcast_to_length)).abs()
train["ads_median_dev_podcast"] = (train['Number_of_Ads'] - train["Podcast_Name"].map(podcast_to_ads)).abs()
train["length_median_dev_genre"] = (train["Episode_Length_minutes"] - train["Genre"].map(genre_to_length)).abs()
train["ads_median_dev_genre"] = (train['Number_of_Ads'] - train["Genre"].map(genre_to_ads)).abs()





length_mean_per_podcast = train.groupby("Podcast_Name")["Episode_Length_minutes"].transform("mean")
length_mean_per_genre = train.groupby("Genre")["Episode_Length_minutes"].transform("mean")
train["length_normalized_per_podcast"] = train["Episode_Length_minutes"] / length_mean_per_podcast
train["length_normalized_per_genre"] = train["Episode_Length_minutes"] / length_mean_per_genre



train["podcast_median_length_between_ads"] =  (train["Episode_Length_minutes"] / train["Number_of_Ads"].replace(0, np.nan)).fillna(0)
train["host_popularity_episode_lenght"] = train["Episode_Length_minutes"] * (train["Host_Popularity_percentage"]/100)
#train["Length_bin"] = pd.cut(train["Episode_Length_minutes"], bins=[0, 30, 60, 90, 150], labels=["short", "medium", "long", "very_long"],include_lowest=True)
train["length_percentile"] = train["Episode_Length_minutes"].rank(pct=True)

train["length_percentile_podcast"] = train.groupby("Podcast_Name")["Episode_Length_minutes"].rank(pct=True)
train["length_percentile_genre"] = train.groupby("Genre")["Episode_Length_minutes"].rank(pct=True)




train["Episode_Length_minutes"].describe()


train["host_guest_popularity_gap"] = train["Host_Popularity_percentage"] - train["Guest_Popularity_percentage"]
train["ads_per_minute"] = (train["Number_of_Ads"] / train["Episode_Length_minutes"].replace(0, np.nan)).fillna(0)



train['is_weekend'] = train['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
day_map = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

train['Publication_Day'] = train["Publication_Day"].map(day_map)



time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
train['Episode_Sentiment'] = train['Episode_Sentiment'].replace(sent_dict)
train['Publication_Time'] = train["Publication_Time"].map(time_dict)


genre_dict = {'True Crime':0, 'Comedy':1, 'Education':2, 'Technology':3, 'Health':4,
       'News':5, 'Music':6, 'Sports':7, 'Business':8, 'Lifestyle':9}
train['Genre'] = train['Genre'].replace(genre_dict)


length_order = {"short": 0, "medium": 1, "long": 2, "very_long": 3}
#train["Length_bin"] = train["Length_bin"].map(length_order).astype(int)


episode_order = {}
for x in np.arange(1,101):
    a = f"Episode {x}"
    episode_order[a] = x

train["Episode_Title"] = train["Episode_Title"].map(episode_order).astype(int)


#Lag median listening time based on podcast and episode number
train["list"] = train.groupby(["Podcast_Name","Episode_Title"])["Listening_Time_minutes"].transform('median').shift(1)
train["list"].fillna(0, inplace=True)


train.describe()


train_x = train[train["dataset"]==0].copy()

test_x = train[train["dataset"]==1].copy()
train_explore = train_x.copy()


categorical_cols = train_x.select_dtypes(include=["category","object"]).columns.tolist()

print("Categorical columns:train_x", categorical_cols,)


train_x1 = train_x.copy()

train_y1 =train_x1.pop("Listening_Time_minutes")
for colname in train_x1[categorical_cols].select_dtypes(["object","category"]):
    train_x1[colname], _ = train_x1[colname].factorize()


# All discrete features should now have integer dtypes 
discrete_features = train_x1.dtypes == np.int64
discrete_features


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(train_x1, train_y1, discrete_features=discrete_features)
print(mi_scores[::3])  # show a few features with their MI scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)


mi_scores1 = pd.Series(mi_scores, index=train_x.columns)
selected_columns = train_x.columns[(mi_scores1 > 0.01)].tolist() 

# Remove the 'transactions' column if it exists as will cause data leakeage in future predictions given its unknown
selected_columns = [col for col in selected_columns if col not in ['transactions','Podcast_Name'] ]



a = train_explore.select_dtypes(include=["integer","float"]).columns.tolist()
len(a)


fig, axes = plt.subplots((len(a)//2),2,constrained_layout=True, figsize=(40,60))
axes = axes.flatten()
for xy,ax in zip(a,axes):
    sns.histplot(x=xy,data=train_explore, ax=ax)
    ax.set_title(f"Histogram Distribution for {xy}", fontsize=25)
    ax.tick_params(axis='both', labelsize=20)



sns.histplot(x="Episode_Length_minutes", data=train_explore)
plt.tick_params(axis='both', labelsize=8)
plt.show()


def custom_agg(series):
    if series.dtype == 'O' or str(series.dtype).startswith("category"):
        mode = series.mode()
        return mode.iloc[0] if not mode.empty else None
    else:
        return series.median()

podcast_df = train[train['dataset'] == 0] #[train_x["Podcast_Name"] == "Mystery Matters"].copy()
summary = podcast_df.groupby(["Podcast_Name","Episode_Title"]).agg(custom_agg).reset_index()



fig,axes = plt.subplots(12,4,figsize=(20,40),constrained_layout=True)
axes = axes.flatten()
for a,b in zip(axes, train_x["Podcast_Name"].unique()):
    sns.scatterplot(
    data=summary[summary["Podcast_Name"]==b],
    x="Episode_Title",
    y="Episode_Length_minutes",
    hue="Episode_Sentiment",
    palette="Set1",
    ax=a
    )
    a.set_title(f"Sentiment by {b}")
    a.set_xlabel("Episode Number")
    a.set_ylabel("Episode Length (minutes)")
    a.legend(title="Sentiment")
    a.grid(True)
plt.tight_layout()
plt.show()



fig,axes = plt.subplots(12,4,figsize=(20,40)#,constrained_layout=True
                        )
axes = axes.flatten()
for a,b in zip(axes, train_x["Podcast_Name"].unique()):
    sns.scatterplot(
    data=summary[summary["Podcast_Name"]==b],
    x="Episode_Title",
    y="Listening_Time_minutes",
    hue="Episode_Sentiment",
    palette="dark",
    ax=a
    )
    a.set_title(f"Sentiment by {b}")
    a.set_xlabel("Episode Number")
    a.set_ylabel("Episode Listening Time (minutes)")
    a.legend(title="Sentiment")
    a.grid(True)
plt.tight_layout()
plt.show()


train_y = train_x.pop("Listening_Time_minutes")
train_x = train_x[selected_columns]
test_x.pop("Listening_Time_minutes")
test_x = test_x[selected_columns]



x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, train_size=0.2)


categorical_cols = x_train.select_dtypes(include=["category","object"]).columns.tolist()
categorical_cols


lg = LGBMRegressor(n_estimators=1200,  learning_rate=0.01, num_leaves=64,
                   verbose=-1,metric="rmse",objective="regression",
                   max_depth=6)
lg.fit(x_train,y_train,
       eval_set=[(x_test, y_test)])
y_pred = lg.predict(x_train)
y_pred1 = lg.predict(x_test)

rmsle_score = np.sqrt(mean_squared_error(y_train, y_pred))
rmsle_score1 = np.sqrt(mean_squared_error(y_test, y_pred1))
print("Train RMSE: ",rmsle_score, " Test RMSE: ",rmsle_score1)
scores = cross_val_score(lg, x_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
print("Avg CV RMSE:", -scores.mean())





# Plot feature importances
importance_xgb = lg.feature_importances_
sorted_idx = np.argsort(importance_xgb)[::-1]
features1 = x_train.columns

plt.figure(figsize=(10, 6))
plt.barh([features1[i] for i in sorted_idx], importance_xgb[sorted_idx], color="greenyellow")
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Light GBM Regression Feature Importance')
plt.gca().invert_yaxis()  
plt.show()


test = lg.predict(test_x)
res = pd.DataFrame({"id":test_x.index,
                    "Listening_Time_minutes": test})

res = res.sort_values("id")
res.to_csv('submission.csv', index=False)
res.head(10)

