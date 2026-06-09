import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler
)
from sklearn.feature_selection import (
    mutual_info_regression
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score, KFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    r2_score, 
    mean_squared_error
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# We load the data

listening_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col = "id")


listening_train.shape


listening_train.head()


listening_train.describe(exclude = np.number)


listening_train.describe().style.background_gradient(cmap='Greens')


listening_train.info()


# Function to view the data of each variable in detail

def detail_columns(data, colum):

    print(
        "Variable: ", colum,
        "\nFormat: ", data[colum].dtype,
        "\nNumber of null values: ", data[colum].isnull().sum(),
        "\nUnique values: ", data[colum].nunique(),
        "\nDistribution of values: \n", data[colum].value_counts()
    )


# Establishing the seaborn aesthetic

sns.set_style("dark")


detail_columns(listening_train, "Listening_Time_minutes")


# We analyze the distribution of the data

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=listening_train, 
    x="Listening_Time_minutes", 
    color="green",
    edgecolor="k",
    kde=True
)

plt.title(label="Distribution of target variable values")
plt.tight_layout()
plt.show()



detail_columns(listening_train, "Podcast_Name")
print("-" * 50)
detail_columns(listening_train, "Episode_Title")


# We analyze the distribution of the data

fig, axes = plt.subplots(nrows=2, figsize=(22, 8))

sns.histplot(
    data=listening_train, 
    x="Podcast_Name", 
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[0]
).tick_params(axis='x', labelrotation=90)

sns.histplot(
    data=listening_train, 
    x="Episode_Title",
    color="green",
     edgecolor="k",
    kde=True,
    ax=axes[1]
).tick_params(axis='x', labelrotation=90)

plt.suptitle(t="Distribution of Podcast values by Names & Episodes")
plt.tight_layout()
plt.show()


detail_columns(listening_train, "Episode_Length_minutes")
print("-" * 50)
detail_columns(listening_train, "Episode_Sentiment")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.histplot(
    data=listening_train, 
    x="Episode_Length_minutes", 
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[0]
)

sns.histplot(
    data=listening_train, 
    x="Episode_Sentiment",
    color="green",
    edgecolor="k",
    ax=axes[1]
)

plt.suptitle(t="Distribution of values by Episodes length & sentiment")
plt.tight_layout()
plt.show()


detail_columns(listening_train, "Genre")
print("-" * 50)
detail_columns(listening_train, "Number_of_Ads")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.histplot(
    data=listening_train, 
    x="Genre", 
    color="green",
    edgecolor="k",
    ax=axes[0]
).tick_params(axis='x', labelrotation=45)

sns.histplot(
    data=listening_train, 
    x="Number_of_Ads",
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[1]
)

plt.suptitle(t="Distribution of values by Podcast Genre & Number of Ads")
plt.xlim(-1, 4)
plt.tight_layout()
plt.show()



# We obtain the rows that we are interested in analyzing

null_ads = listening_train.loc[listening_train["Number_of_Ads"] > 3]

null_ads


detail_columns(listening_train, "Host_Popularity_percentage")
print("-" * 50)
detail_columns(listening_train, "Guest_Popularity_percentage")



# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.histplot(
    data=listening_train, 
    x="Host_Popularity_percentage", 
    color="green",
    edgecolor="k",
    kde=True,
    stat="percent",
    ax=axes[0]
)

sns.histplot(
    data=listening_train, 
    x="Guest_Popularity_percentage",
    color="green",
    edgecolor="k",
    kde=True,
    stat="percent",
    ax=axes[1]
)
plt.suptitle(t="Distribution of values by Host & Guest Popularity(%)")
plt.tight_layout()
plt.show()



# We obtain the rows that we are interested in analyzing

null_popularity = listening_train.loc[
    (listening_train["Host_Popularity_percentage"] > 100) | 
    (listening_train["Guest_Popularity_percentage"] > 100)
]

print(f"Total samples with a percentage greater than 100: {len(null_popularity)} \n")


detail_columns(listening_train, "Publication_Day")
print("-" * 50)
detail_columns(listening_train, "Publication_Time")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.histplot(
    data=listening_train, 
    x="Publication_Day", 
    color="green",
    edgecolor="k",
    ax=axes[0]
).tick_params(axis='x', labelrotation=45)

sns.histplot(
    data=listening_train, 
    x="Publication_Time",
    color="green",
    edgecolor="k",
    ax=axes[1]
).tick_params(axis='x', labelrotation=45)

plt.suptitle(t="Distribution of values by Publication Day & Time")
plt.tight_layout()
plt.show()


# We make a copy of the original dataset

listening_new = listening_train.copy()


# We check that no duplicate data is found

print(f"Length: {len(listening_new.duplicated())}")
print(f"Duplicates: {listening_new.duplicated().sum()}")


# We check for null values

null_values = (
    pd.DataFrame(
        {f"Amount of Null Data": listening_new.isnull().sum(), 
         "Percentage of Null Data" : (
             listening_new.isnull().sum()) / (len(listening_new)) * (100)
        }))

null_values.style.background_gradient(cmap="Greens")


# We fill null values with the mean groupby Podcast

listening_new["Episode_Length_minutes"] = (
    listening_new["Episode_Length_minutes"].fillna(
        listening_new.groupby("Podcast_Name")["Episode_Length_minutes"].transform("mean")
    )
)

print(
    "Number of null values: ", listening_new["Episode_Length_minutes"].isnull().sum(), "\n\n",
    "Distribution of values: \n", listening_new["Episode_Length_minutes"].value_counts(normalize=True)
)


# We replace the erroneous values with the mode of the Podcasts

listening_new["Number_of_Ads"] = (
    listening_new["Number_of_Ads"].apply(lambda x: np.NaN if x>3 else x)
)

listening_new["Number_of_Ads"] = (
    listening_new["Number_of_Ads"].fillna(
        listening_new.groupby("Podcast_Name")["Number_of_Ads"].transform(lambda v: v.mode()[0])
    )
)

print(
    "Number of null values: ", listening_new["Number_of_Ads"].isnull().sum(), "\n\n",
    "Distribution of values: \n", listening_new["Number_of_Ads"].value_counts()
)


# We replace the erroneous values

listening_new["Host_Popularity_percentage"] = np.where(
    listening_new["Host_Popularity_percentage"] > 100, 100, listening_new["Host_Popularity_percentage"]
).round(decimals=2)

print(
    "Number of Unique values: ", listening_new["Host_Popularity_percentage"].nunique(), "\n",
    "Distribution of values: \n", listening_new["Host_Popularity_percentage"].value_counts(normalize=True)
)



# We replace the erroneous values

listening_new["Guest_Popularity_percentage"] = np.where(
    listening_new["Guest_Popularity_percentage"] > 100, 100, listening_new["Guest_Popularity_percentage"]
)

# We fill null values with the mean groupby Podcast

listening_new["Guest_Popularity_percentage"] = (
    listening_new["Guest_Popularity_percentage"].fillna(
        listening_new.groupby("Podcast_Name")["Guest_Popularity_percentage"].transform("mean")
    )
).round(decimals=2)

print(
    "Number of Unique values: ", listening_new["Guest_Popularity_percentage"].nunique(), "\n",
    "Number of null values: ", listening_new["Guest_Popularity_percentage"].isnull().sum(), "\n\n",
    "Distribution of values: \n", listening_new["Guest_Popularity_percentage"].value_counts(normalize=True)
)


# We changed the format for more efficient memory usage

listening_new[listening_new.select_dtypes(["object"]).columns] = (
    listening_new.select_dtypes(["object"]).apply(
        lambda x: x.astype("category"))
)


listening_new.info()


# Establishing the seaborn aesthetic

sns.set_style("darkgrid")


popularity_ltm = listening_new.pivot(
    columns="Podcast_Name", values="Listening_Time_minutes"
)


popularity = popularity_ltm.sum().sort_values(ascending=False)

print(popularity)


print(
    " Most listened Podcast: {0:0.3f}\n".format(max(popularity)),
    "Least listened Podcast: {0:0.3f}\n".format(min(popularity))
)


# We analyze the popularity

fig, axes = plt.subplots(figsize=(14, 6))

sns.barplot(
    data=listening_new, 
    x="Podcast_Name",
    y="Listening_Time_minutes",
    estimator="sum",
    hue="Episode_Sentiment",
    edgecolor="k"
).tick_params(axis='x', labelrotation=90)

plt.title(label="Most and Least popular Podcasts")
plt.tight_layout()
plt.show()



popularity_ltm.describe()


# We analyze the popularity

fig, axes = plt.subplots(figsize=(12, 4))

sns.boxplot(data=popularity_ltm).tick_params(axis='x', labelrotation=90)


genre_ltm = listening_new.pivot(
    columns="Genre", values="Listening_Time_minutes"
)

genre_ltm.describe()


popularity_genre = genre_ltm.sum().sort_values(ascending=False)

print(popularity_genre)


 #We analyze the Genre popularity

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
sns.set_theme()

sns.barplot(
    data=listening_new, 
    x="Genre",
    y="Listening_Time_minutes",
    estimator="sum",
    edgecolor="k",
    ax=axes[0]
).tick_params(axis='x', labelrotation=45)

sns.boxenplot(
    data=genre_ltm,
    linewidth=.5,
    line_kws=dict(linewidth=1.5, color="#cde"),
    flier_kws=dict(facecolor=".7", linewidth=.5),
    ax=axes[1]
).tick_params(axis='x', labelrotation=45)

plt.suptitle(t="Most and Least popular Genre")
plt.tight_layout()
plt.show()



df_host_guest = listening_new[[
    "Host_Popularity_percentage", 
    "Guest_Popularity_percentage"
]]

df_guest_old_stat = listening_train[["Guest_Popularity_percentage"]].rename(
    columns={"Guest_Popularity_percentage" : "Guest_old_stats"}
)

df_host_guest_stats = pd.concat([df_host_guest, df_guest_old_stat], axis=1)

df_host_guest_stats.describe()


# We analyze the Guests popularity data

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.violinplot(
    data=df_host_guest_stats, 
    x="Guest_Popularity_percentage",
    inner="quart",
    color="g",
    ax=axes[0]
)

sns.violinplot(
    data=df_host_guest_stats, 
    x="Guest_old_stats",
    inner="quart",
    color="r",
    ax=axes[1]
)

plt.suptitle(t="New and Previous Guests Popularity Data")
plt.tight_layout()
plt.show()



df_host_guest["Host_round"] = df_host_guest["Host_Popularity_percentage"].round(decimals=0).astype('Int64')
df_host_guest["Guest_round"] = df_host_guest["Guest_Popularity_percentage"].round(decimals=0).astype('Int64')
df_host_guest["Listening_time"] = listening_new["Listening_Time_minutes"].round(decimals=0).astype('Int64')


# We analyze the Host & Guests popularity

fig, axes = plt.subplots(figsize=(12, 4))

sns.lineplot(
    data=df_host_guest, 
    x="Host_round",
    y="Listening_time",
    color="g"
)

sns.lineplot(
    data=df_host_guest, 
    x="Guest_round",
    y="Listening_time",
    color="r"
)

plt.title(label="Host & Guests popularity impact on listening time")
plt.tight_layout()
plt.show()



ads_ltm = listening_new.pivot(
    columns="Number_of_Ads", values="Listening_Time_minutes"
)

ads_ltm.describe()


popularity_ads = ads_ltm.sum().sort_values(ascending=False)

print(popularity_ads)


# We analyze the impact of the number of ads

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.barplot(
    data=listening_new, 
    x="Number_of_Ads",
    y="Listening_Time_minutes",
    estimator="sum",
    palette="Set3",
    edgecolor="k",
    ax=axes[0]
)

sns.violinplot(
    data=listening_new, 
    x="Number_of_Ads",
    y="Listening_Time_minutes",
    #estimator="sum",
    palette="Set3",
    edgecolor="k",
    ax=axes[1]
)

plt.suptitle(t="Amount of Advertising vs. Listening Time")
plt.tight_layout()
plt.show()

