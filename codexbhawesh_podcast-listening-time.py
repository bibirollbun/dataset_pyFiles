import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import RandomizedSearchCV
import joblib
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
data.head()


data.info()


data.describe()


data.isna().sum()


unq_podcast = data["Podcast_Name"].unique().tolist()
print(f"Total unique podcast are {len(unq_podcast)}")
for i in unq_podcast:
    print(i)


total_no_of_podcast_for_each_podcast = data["Podcast_Name"].value_counts().reset_index()
total_no_of_podcast_for_each_podcast
plt.figure(figsize = (10,10))
plt.plot(
    total_no_of_podcast_for_each_podcast["Podcast_Name"], 
    total_no_of_podcast_for_each_podcast["count"], 
    marker = "o", 
    c = "r", 
    label = "total_no_of_podcast_for_each_podcast")
plt.xticks(rotation=90)
plt.grid()
plt.title("Total no of podcast for each podcast")
plt.show


avg_episode_len_each_podcast = data.groupby("Podcast_Name")["Episode_Length_minutes"].mean().reset_index()
avg_episode_len_each_podcast
plt.figure(figsize = (10, 10))
plt.plot(avg_episode_len_each_podcast["Podcast_Name"], 
         avg_episode_len_each_podcast["Episode_Length_minutes"],
        marker = "v",
        c = "g",
        label = "avg_episode_len_each_podcast")
plt.xticks(rotation=90)
plt.grid()
plt.title("Average Episodes length for Each Podcast")
plt.show


bad_vals = data[~np.isfinite(data["Episode_Length_minutes"])]
podcast_missing_episode_len = bad_vals["Podcast_Name"].value_counts().reset_index()
plt.figure(figsize = (10, 8))
plt.plot(podcast_missing_episode_len["Podcast_Name"], 
         podcast_missing_episode_len["count"],
        marker = ">",
        c= "b",
        label = "Podcast with nan values")
plt.xticks(rotation = 90)
plt.show()


color = ["r", "g", "b", "y", "k"]
data["Genre"].value_counts().plot(kind = "bar", color = color)


podcast_vs_genre = pd.pivot_table(
    data,
    index="Podcast_Name",
    columns="Genre",
    values="Episode_Length_minutes",
    aggfunc="count",
    fill_value=0
)

podcast_vs_genre.plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title("Number of Episodes per Podcast by Genre")
plt.ylabel("Episode Count")
plt.xlabel("Podcast Name")
plt.xticks(rotation=90)
plt.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



avg_podcast_host = data.groupby("Podcast_Name")["Host_Popularity_percentage"].mean()
plt.figure(figsize = (12, 6))
avg_podcast_host.plot(kind= "bar", color = color)
plt.show()


data.columns


df = data.copy()
df.dropna(inplace=True)


corr = df[["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads", "Listening_Time_minutes"]].corr()
sns.heatmap(corr, annot = True , cmap= "coolwarm")


avg_listening_len_each_podcast = data.groupby("Podcast_Name")["Listening_Time_minutes"].mean().reset_index()
avg_listening_len_each_podcast
plt.figure(figsize = (10, 10))
plt.plot(avg_listening_len_each_podcast["Podcast_Name"], 
         avg_listening_len_each_podcast["Listening_Time_minutes"],
        marker = "v",
        c = "g",
        label = "avg_episode_len_each_podcast")
plt.xticks(rotation=90)
plt.grid()
plt.title("Average Listening length for Each Podcast")
plt.show


fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(avg_episode_len_each_podcast["Podcast_Name"], 
         avg_episode_len_each_podcast["Episode_Length_minutes"],
         marker="v", color="g", label="Avg Episode Length")
ax1.set_ylabel("Episode Length (minutes)", color="g")
ax1.tick_params(axis='y', labelcolor='g')
ax1.set_xlabel("Podcast Name")
ax1.set_xticklabels(avg_episode_len_each_podcast["Podcast_Name"], rotation=90)

ax2 = ax1.twinx()
ax2.plot(avg_listening_len_each_podcast["Podcast_Name"], 
         avg_listening_len_each_podcast["Listening_Time_minutes"],
         marker="o", color="b", label="Avg Listening Time")
ax2.set_ylabel("Listening Time (minutes)", color="b")
ax2.tick_params(axis='y', labelcolor='b')

# Title and legends
fig.suptitle("Avg Episode Length vs Listening Time per Podcast")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()


data.groupby("Genre")["Listening_Time_minutes"].mean().plot(kind= "bar", color = color)


weekedays = data["Publication_Day"].value_counts()
weekedays.plot(kind="bar", color = color)


weekedays = data["Publication_Time"].value_counts()
weekedays.plot(kind="bar", color = color)


podcast_sentiment = pd.pivot_table(data, 
                                   index = "Podcast_Name", 
                                   columns = ["Episode_Sentiment"], 
                                   values = ["Episode_Length_minutes"], 
                                   aggfunc = "count",
                                  fill_value = 0)
plt.figure(figsize = (12, 8))
podcast_sentiment.plot(kind="bar", cmap = "coolwarm", stacked = True, figsize = (12, 8))
plt.show()


publication_length = data.groupby("Publication_Time")["Episode_Length_minutes"].mean()
plt.figure(figsize=(12, 6)) 
plt.plot(publication_length.index, publication_length, label="Average Length", color="blue")
plt.title("Average Length  by Podcast", fontsize=16)
plt.xlabel("Podcast Name", fontsize=14)
plt.ylabel("Time (Minutes)", fontsize=14)
plt.legend(fontsize=12)
plt.xticks(rotation=90, fontsize=12)
plt.show()


df_2 = data.copy()
df_2["category"] = df_2["Guest_Popularity_percentage"].map(lambda x: 1 if x <= 25 else (2 if x > 25 and x <= 75 else 3))
df_2.head()


category_listening_time = df_2.groupby("category")["Listening_Time_minutes"].mean()
category_listening_time.plot(kind = "bar", cmap = "coolwarm", figsize = (12, 8))
plt.show()


df_2["category"] = df_2["Host_Popularity_percentage"].map(lambda x: 1 if x <= 25 else (2 if x > 25 and x <= 75 else 3))


category_listening_time = df_2.groupby("category")["Listening_Time_minutes"].mean()
category_listening_time.plot(kind = "bar", cmap = "coolwarm", figsize = (12, 8))
plt.show()


df_4 = data.copy()


def fill_value(data, group_col, target_col):
    data[target_col] = data.groupby(group_col)[target_col].transform(lambda x: x.fillna(x.mean()))
    return data
dataset = fill_value(df_4, "Podcast_Name", "Episode_Length_minutes")
dataset = fill_value(dataset, "Podcast_Name", "Host_Popularity_percentage")
dataset = fill_value(dataset, "Podcast_Name", "Guest_Popularity_percentage")
dataset = fill_value(dataset, "Podcast_Name", "Number_of_Ads")
dataset.info()


def get_average(data):
    host = data["Host_Popularity_percentage"]
    guest = data["Guest_Popularity_percentage"]
    avg = ( host + guest ) / 2
    data["average"] = avg

    return data


dataset = get_average(dataset)
dataset.head()


categorical_columns = ["Podcast_Name", "Genre", "Publication_Day", "Episode_Sentiment", "Publication_Time"]
OHE_data = pd.get_dummies(
    dataset,
    prefix ="OHE",
    prefix_sep = "_",
    columns = categorical_columns,
    drop_first = True,
    dtype = 'int8'
)

OHE_data.head()


OHE_data.drop(["Episode_Title", "id", "Host_Popularity_percentage"], inplace = True, axis = 1)


OHE_data.head()


X = OHE_data.drop("Listening_Time_minutes", axis = 1)
y = OHE_data["Listening_Time_minutes"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
print(f"{X_train.shape}\n{X_test.shape}\n{y_train.shape}\n{y_test.shape}")


model = RandomForestRegressor(max_depth = 20, n_estimators = 300, random_state = 42)


model.fit(X, y)
#model = joblib.load("/kaggle/working/podcast_listening_time.pkl")


y_pred = model.predict(X)


MSE = mean_squared_error(y, y_pred)


MSE


#joblib.dump(model, "podcast_listening_time.pkl")


test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
test_data.head()


test = test_data.copy()


test_dataset = fill_value(test, "Podcast_Name", "Guest_Popularity_percentage")
test_dataset = fill_value(test, "Podcast_Name", "Episode_Length_minutes")


test_dataset.info()


test_dataset = get_average(test_dataset)
test_dataset.head()


categorical_columns = ["Podcast_Name", "Genre", "Publication_Day", "Episode_Sentiment", "Publication_Time"]
OHE_test_data = pd.get_dummies(
    test_dataset,
    prefix ="OHE",
    prefix_sep = "_",
    columns = categorical_columns,
    drop_first = True,
    dtype = 'int8'
)

OHE_test_data.head()


OHE_test_data.drop(["Episode_Title", "id", "Host_Popularity_percentage"], inplace = True, axis = 1)


X_test = OHE_test_data


y_test_pred = model.predict(X_test)


submission = pd.DataFrame(test_data["id"])


submission["Listening_Time_minutes"] = y_test_pred


submission.to_csv("submission.csv", index = False)




