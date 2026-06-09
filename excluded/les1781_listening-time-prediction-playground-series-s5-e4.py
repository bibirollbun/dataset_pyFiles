# We load the competition data

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler
)
from sklearn.feature_selection import (
    mutual_info_regression
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
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

listening_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col="id")


listening_train.shape


listening_train.head()


listening_train.describe(exclude = np.number)


listening_train.describe().style.background_gradient(cmap="Greens")


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


genre_ltm = listening_new.pivot(columns="Genre", values="Listening_Time_minutes")

genre_ltm.describe()


popularity_genre = genre_ltm.sum().sort_values(ascending=False)

print(popularity_genre)


# We analyze the Genre popularity

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


ads_ltm = listening_new.pivot(columns="Number_of_Ads", values="Listening_Time_minutes")

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


# We analyze the podcast advertising

fig, axes = plt.subplots(figsize=(14, 6))

sns.lineplot(
    data=listening_new, 
    x="Podcast_Name",
    y="Listening_Time_minutes",
    estimator="sum",
    hue="Number_of_Ads",
    palette="Paired",
).tick_params(axis='x', labelrotation=90)

plt.title(label="Most and Least Ads by Podcasts")
plt.tight_layout()
plt.show()


day_ltm = listening_new.pivot(columns="Publication_Day", values="Listening_Time_minutes")

day_ltm.describe()


time_ltm = listening_new.pivot(columns="Publication_Time", values="Listening_Time_minutes")

time_ltm.describe()


# We analyze the impact of day and time

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.boxenplot(
    data=day_ltm,
    linewidth=.5,
    line_kws=dict(linewidth=1.5, color="#cde"),
    flier_kws=dict(facecolor=".7", linewidth=.5),
    ax=axes[0]
).tick_params(axis='x', labelrotation=45)

sns.boxenplot(
    data=time_ltm,
    linewidth=.5,
    line_kws=dict(linewidth=1.5, color="#cde"),
    flier_kws=dict(facecolor=".7", linewidth=.5),
    ax=axes[1]
).tick_params(axis='x', labelrotation=45)

plt.suptitle(t="Distribution of Day & Time publication on Listening Time")
plt.tight_layout()
plt.show()


# We analyze the impact of the number of ads

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.barplot(
    data=listening_new, 
    x="Publication_Day",
    y="Listening_Time_minutes",
    estimator="sum",
    palette="Set3",
    edgecolor="k",
    ax=axes[0]
).tick_params(axis='x', labelrotation=45)

sns.barplot(
    data=listening_new, 
    x="Publication_Time",
    y="Listening_Time_minutes",
    estimator="sum",
    palette="Set3",
    edgecolor="k",
    ax=axes[1]
).tick_params(axis='x', labelrotation=45)

plt.suptitle(t="Impact of Day & Time publication on Listening Time")
plt.tight_layout()
plt.show()


listening_end = listening_new.copy()


listening_end.info()


listening_end.describe().T


listening_end.describe(exclude = np.number).T


# We map the variables and change the format

eps_order = {"Negative" : 0, "Neutral" : 1, "Positive" : 2}
listening_end["Episode_Sentiment"] = listening_end["Episode_Sentiment"].map(eps_order)
listening_end["Episode_Sentiment"] = listening_end["Episode_Sentiment"].astype("float64")


# We separate the categorical variables from the numerical ones

df_numerical = listening_end.select_dtypes(include="number")
df_categorical = listening_end.select_dtypes(include="category")


# We apply OrdinalEncoder to the remaining categorical variables

enc = OrdinalEncoder(categories="auto")

enc_data = enc.fit_transform(df_categorical)

# Creating a DataFrame with the encoded data and then join them

df_encoded = pd.DataFrame(enc_data, columns=enc.get_feature_names_out(df_categorical.columns))

df_listening = pd.concat([df_encoded, df_numerical], axis=1)


df_listening.info()


df_listening.describe().style.background_gradient(cmap="Greens")


df_listening.corr().style.background_gradient(cmap="Greens")


# We graph the correlation between the variables

matrix_listening = df_listening.corr(numeric_only=True).round(2)

plt.figure(figsize=(10, 4))

sns.heatmap(
    matrix_listening, 
    annot=True,
    cmap=sns.cubehelix_palette(
        start=2, rot=0, 
        dark=0, light=.95, 
        reverse=True, as_cmap=True
    )
)


df_listening.describe().T


# We separate the target variable from the features and data to scale

x_listening = df_listening.drop(columns="Listening_Time_minutes")
y_listening = df_listening["Listening_Time_minutes"]


# Numerical variables to scale

df_numeric = x_listening[[
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage"
]]

# We transform the data

scaler = StandardScaler()

scale_num = scaler.fit_transform(df_numeric)

df_scale = pd.DataFrame(
    scale_num, columns=scaler.get_feature_names_out(df_numeric.columns)
)


# We create a df with the remaining variables

df_rest = x_listening.drop(
    columns=["Episode_Length_minutes",
             "Host_Popularity_percentage",
             "Guest_Popularity_percentage"]
)

# We concatenate the dataframes

x_end = pd.concat([df_rest, df_scale], axis=1)


x_end.describe().T


mi_scores = mutual_info_regression(x_end, y_listening)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=x_end.columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores


scores = mi_scores.sort_values(ascending=True)
width = np.arange(len(mi_scores))
ticks = list(mi_scores.index)
plt.barh(width, mi_scores)
plt.yticks(width, ticks)
plt.title("Mutual Information Scores")
plt.figure(dpi=100, figsize=(8, 5))
plt.show()


# We separate the data into training and validation sets

x_train, x_val, y_train, y_val = (
    train_test_split(
        x_end, y_listening, test_size=0.3, random_state=42
    )
)


# We create the model instance

lr = LinearRegression()

# Train the model with the data

lr.fit(x_train, y_train)


# We evaluate the initial performance of the model

y_pred_lr = lr.predict(x_val)

r2_lr = r2_score(y_val, y_pred_lr)

rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))

print(f"LinearRegression\n\nR-squared score: {r2_lr}\nRMSE: {rmse_lr}")


lr.get_params()


# Evaluate the model using cross-validation

cv_scores_lr = cross_val_score(
    lr, x_train, y_train, 
    scoring="neg_root_mean_squared_error", 
    cv=5
)

print(f"Cross-validation scores: {cv_scores_lr}")
print(f"Mean CV accuracy: {np.mean(cv_scores_lr):.2f}")


# Create the KFold object

kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# We evaluate the model with the KFold method

kfold_scores_lr = cross_val_score(
    lr, x_train, y_train, 
    scoring="neg_root_mean_squared_error", 
    cv=kfold
)

print(f"Cross-validation Kfold scores: {kfold_scores_lr}")
print(f"Mean CV-kfold accuracy: {np.mean(kfold_scores_lr):.2f}")


# We create the model instance

rfr = RandomForestRegressor()

# Train the model with the data

rfr.fit(x_train, y_train)


# We evaluate the initial performance of the model

y_pred_rfr = rfr.predict(x_val)

r2_rfr = r2_score(y_val, y_pred_rfr)

rmse_rfr = np.sqrt(mean_squared_error(y_val, y_pred_rfr))

print(f"RandomForestRegressor\n\nR-squared score: {r2_rfr}\nRMSE: {rmse_rfr}")


# We review the parameters

rfr.get_params()


# We establish the parameters to test

rfr_param_grid  = {
    "max_depth": [3, 5],
    "n_estimators" : [50, 100],
    "max_features" : [1.0, 0.2],
    "min_samples_split" : [2, 4],
    "min_samples_leaf" : [1, 2]
}

rfr_grid = RandomizedSearchCV(
    rfr,
    rfr_param_grid,
    cv=kfold,
    scoring="neg_root_mean_squared_error",
    return_train_score=True
)

rfr_search = rfr_grid.fit(x_train, y_train)

print(
    f'Parameters: {rfr_search.best_params_}\nScore: {rfr_search.best_score_}'
)


# We save the results within a dataframe

rfr_cv_results = pd.DataFrame(rfr_search.cv_results_)

rfr_cv_results.head(10).sort_values(by="rank_test_score", ascending=True)


# We evaluate the performance after an initial optimization

y_pred_rfr_search = rfr_search.best_estimator_.predict(x_val)

r2_rfr_search = r2_score(y_val, y_pred_rfr_search)

rmse_rfr_search = np.sqrt(mean_squared_error(y_val, y_pred_rfr_search))

print(f"RandomForestRegressor optimization\n\nR-squared score: {r2_rfr_search}\nRMSE: {rmse_rfr_search}")


# We create the model instance

xgbr = XGBRegressor()

# Train the model with the data

xgbr.fit(x_train, y_train)


# We evaluate the initial performance of the model

y_pred_xgbr = xgbr.predict(x_val)

r2_xgbr = r2_score(y_val, y_pred_xgbr)

rmse_xgbr = np.sqrt(mean_squared_error(y_val, y_pred_xgbr))

print(f"XGBRegressor\n\nR-squared score: {r2_xgbr}\nRMSE: {rmse_xgbr}")


xgbr.get_params()


# We establish the parameters to test

xgbr_param_grid  = {
    "gamma" : [1, 2, 3],
    "max_depth" : [3, 4, 5],
    "learning_rate" : [0.1, 0.01, 0.001],
    "subsample" : [0.5, 0.7, 1],
    "n_estimators" : [50, 100, 150]
}

xgbr_grid = RandomizedSearchCV(
    xgbr,
    xgbr_param_grid,
    cv=kfold,
    scoring="neg_root_mean_squared_error",
    return_train_score=True
)

xgbr_search = xgbr_grid.fit(x_train, y_train)

print(
    f'Parameters: {xgbr_search.best_params_}\nScore: {xgbr_search.best_score_}'
)


# We save the results within a dataframe

xgbr_cv_results = pd.DataFrame(xgbr_search.cv_results_)

xgbr_cv_results.head(10).sort_values(by="rank_test_score", ascending=True)


# We evaluate the performance after an initial optimization

y_pred_xgbr_search = xgbr_search.best_estimator_.predict(x_val)

r2_xgbr_search = r2_score(y_val, y_pred_xgbr_search)

rmse_xgbr_search = np.sqrt(mean_squared_error(y_val, y_pred_xgbr_search))

print(f"XGBRegressor optimization\n\nR-squared score: {r2_xgbr_search}\nRMSE: {rmse_xgbr_search}")


# We compare the models results

print(
    "LinearRegression RMSE Score: {0:0.3f}\n".format(rmse_lr),
    "\nRandomForestRegressor RMSE Score: {0:0.3f}\n".format(rmse_rfr_search),
    "\nXGBRegressor RMSE Score: {0:0.3f}\n".format(rmse_xgbr_search)
)


# Create the KFold object

kfold = KFold(n_splits=10, shuffle=True, random_state=42)


# We establish the parameters to test

xgbr_param_grid_end = {
    "gamma" : [0.1, 0.5, 0.8, 0, 1],
    "max_depth" : [3, 4, 5, 6, 7],
    "learning_rate" : [0.2, 0.1, 0.01, 0.001],
    "subsample" : [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "n_estimators" : [50, 100, 150, 200]
}

xgbr_grid_end = RandomizedSearchCV(
    xgbr,
    xgbr_param_grid_end,
    cv=kfold,
    scoring="neg_root_mean_squared_error",
    return_train_score=True
)

xgbr_search_end = xgbr_grid_end.fit(x_train, y_train)

print(
    f'Parameters: {xgbr_search_end.best_params_}\nScore: {xgbr_search_end.best_score_}'
)


# We save the final results within a dataframe

xgbr_cv_results_end = pd.DataFrame(xgbr_search_end.cv_results_)

xgbr_cv_results_end.head().sort_values(by="rank_test_score", ascending=True)


# We evaluate the performance of the model

y_pred_xgbr_search_end = xgbr_search_end.best_estimator_.predict(x_val)

r2_xgbr_search_end = r2_score(y_val, y_pred_xgbr_search_end)

rmse_xgbr_search_end = np.sqrt(mean_squared_error(y_val, y_pred_xgbr_search_end))

print(f"XGBRegressor optimization\n\nR-squared score: {r2_xgbr_search_end}\nRMSE: {rmse_xgbr_search_end}")


final_model = xgbr_search_end.best_estimator_

final_model.get_params()


# We fit the best model

final_model.fit(x_train, y_train)


# We evaluate the performance of the final model

final_model_ypred = final_model.predict(x_val)

final_model_rmse = np.sqrt(mean_squared_error(y_val, final_model_ypred))

print("Final Model RMSE Score: %.3f" % final_model_rmse)


# We create an explainer for the best estimator

explainer = shap.Explainer(final_model)
shap_values = explainer.shap_values(x_val)

# we visualize the importance

fig = shap.summary_plot(
    shap_values,
    x_val,
    show=False
)
plt.title("Feature Importance", fontsize=20, color='g', loc='left')
plt.xlabel("Mean SHAP Values", fontsize=20)
plt.ylabel("Features", fontsize=20)
plt.show()


# We load the test data

df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# We check the shape and that no duplicate data is found

print(f"Length: {len(df_test.duplicated())}")

print(f"Duplicates: {df_test.duplicated().sum()}")

print(f"Shape: {df_test.shape}")


df_test.info()


df_test.describe()


df_test.describe(exclude = np.number)


# We check the null values

null_values_test = (
    pd.DataFrame(
        {f"Amount of Null Data" : df_test.isnull().sum(), 
         "Percentage of Null Data" : (
             df_test.isnull().sum()) / (len(df_test)) * (100)
        }
    ))

null_values_test.style.background_gradient(cmap="Greens")


# We start by removing the variables that we will not use

df_test_new = df_test.drop(columns=["id"])


# We filled in null values

df_test_new["Episode_Length_minutes"] = (df_test_new["Episode_Length_minutes"].fillna(
    df_test_new.groupby("Podcast_Name")["Episode_Length_minutes"].transform("mean")
))
df_test_new["Guest_Popularity_percentage"] = (df_test_new["Guest_Popularity_percentage"].fillna(
    df_test_new.groupby("Podcast_Name")["Guest_Popularity_percentage"].transform("mean")
))

# We replace the erroneous values

df_test_new["Number_of_Ads"] = (df_test_new["Number_of_Ads"].apply(lambda x: np.NaN if x>3 else x))
df_test_new["Number_of_Ads"] = (df_test_new["Number_of_Ads"].fillna(
    df_test_new.groupby("Podcast_Name")["Number_of_Ads"].transform(lambda v: v.mode()[0])
))
df_test_new["Host_Popularity_percentage"] = np.where(
    df_test_new["Host_Popularity_percentage"] > 100, 100, df_test_new["Host_Popularity_percentage"]
).round(decimals=2)
df_test_new["Guest_Popularity_percentage"] = np.where(
    df_test_new["Guest_Popularity_percentage"] > 100, 100, df_test_new["Guest_Popularity_percentage"]
).round(decimals=2)


# We encode the categorical variables

df_test_new["Episode_Sentiment"] = df_test_new["Episode_Sentiment"].map(eps_order)
df_test_new["Episode_Sentiment"] = df_test_new["Episode_Sentiment"].astype("float64")

test_num = df_test_new.select_dtypes(include="number")
test_cat = df_test_new.select_dtypes(include="object")
test_enc = enc.fit_transform(test_cat)
test_cat_enc = pd.DataFrame(test_enc, columns=enc.get_feature_names_out(test_cat.columns))
test_encoded = pd.concat([test_cat_enc, test_num[["Number_of_Ads", "Episode_Sentiment"]]], axis=1)

# We transform the data

test_num = test_num.drop(columns=["Number_of_Ads", "Episode_Sentiment"])
test_sca = scaler.transform(test_num)
test_scale = pd.DataFrame(test_sca, columns=scaler.get_feature_names_out(test_num.columns))

# We concatenate the dataframes

test_end = pd.concat([test_encoded, test_scale], axis=1)


test_end.info()


test_end.describe().T


# We apply the trained model

listening_predictions = final_model.predict(test_end)


# We review the result

print("Total predictions: ", len(listening_predictions), "\n")


# We create the dataframe

listening_submission = pd.DataFrame({
    "id" : df_test["id"], 
    "Listening_Time_minutes" : listening_predictions
})

listening_submission.head()


# We load the submission sample data

listening_sample = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


# We compare the results with the sample

print(
    f"Shape Sample Submission: {listening_sample.shape}",
    f"\nShape Listening Submission: {listening_submission.shape}"
)
print("\n", listening_sample.head())


# We convert the dataframe to a csv file

listening_submission.to_csv("submission.csv", index=False)

