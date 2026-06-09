# imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import make_scorer, mean_squared_error # earlier root_mean_squared_error

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV



warnings.simplefilter(action='ignore', category=FutureWarning)



# reading data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
display(train.head())
display(train.describe())
display(train.info())
display(train.isna().sum())


df = train
# reduce memory usage: 
start_memory = df.memory_usage().sum()/1024**2

float_datatypes = ['float16', 'float32', "float64"]
float_datatypes = ['float32', "float64"]

for col in df.select_dtypes('float').columns[:-1]: # i do not want to covert the target data
    for dtype in float_datatypes:
        if abs(df[col].max()) <= np.finfo(dtype).max:
            df[col] = df[col].astype(dtype)
            print(f"{col} converted to {dtype}")
            break

int_datatypes = ["int16", "int32", "int64"]
for col in df.select_dtypes("int").columns: 
    for dtype in int_datatypes:
        if abs(df[col].max()) <= np.iinfo(dtype).max:
            df[col] = df[col].astype(dtype)
            print(f"{col} converted to {dtype}.")
            break
end_memory = df.memory_usage().sum()/1024**2
print(f"Memory reduced by {(end_memory - start_memory)/start_memory*100:.2f}% from {start_memory} -> {end_memory}. ")



g = sns.pairplot(
    train.drop(columns = ["id"]),
    plot_kws = {"s":0.2,
               "marker":"."})
g.fig.suptitle("Relation of features to each other")
plt.show()


train.Podcast_Name.value_counts().reset_index()



train.Podcast_Name.value_counts()


fig = px.bar(
    train.Podcast_Name.value_counts().reset_index(drop=False),
    y = "Podcast_Name", 
    x = "count",
    title="Frequency of podcasts in the train dataset",
    labels={"count": "Number of episodes", "Podcast_Name": "Podcast title"}, 
    text="count"
    
)    

fig.update_layout(template = "plotly_white")
fig.show()


fig, ax = plt.subplots(figsize=(15, 10))
g = sns.boxplot(
    train, 
    y = "Podcast_Name",
    x = "Listening_Time_minutes",
    ax=ax, 

)
g.set(
    xlabel= "Listening time (mins)", 
    ylabel="Podcasts",
    title="Distribution of listening time of various podcasts"
)
sns.despine()
plt.show()


import plotly.graph_objects as go
import pandas as pd


fig = go.Figure()

for podcast in train['Podcast_Name'].unique():
    fig.add_trace(go.Box(
        y=train[train['Podcast_Name'] == podcast]['Listening_Time_minutes'],
        name=podcast
    ))

fig.update_layout(
    title="Distribution of listening time of various podcasts",
    xaxis_title="Listening time (mins)",
    yaxis_title="Podcasts",
    boxmode='group', # Ez csoportosítja a boxplotokat a Podcast_Name alapján
    template="plotly_white"
)

fig.show()



display(train.Number_of_Ads.unique())

train[train.Number_of_Ads.notna() & (train.Number_of_Ads >3)]



train.loc[train.Number_of_Ads > 3,"Number_of_Ads"] = train.Number_of_Ads.median()
train["Number_of_Ads"] = train.Number_of_Ads.fillna(train.Number_of_Ads.median()).astype(int)

display(train.Number_of_Ads.unique())


train.Number_of_Ads.median()


train.head()


train["Number_of_Ads"] = pd.Categorical(train.Number_of_Ads, ordered=True, categories = [0, 1,2,3])

for col in ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Number_of_Ads"]:
    fig, ax = plt.subplots(figsize=(9,6))
    g = sns.boxplot(
        train,
        y=col,
        x= "Listening_Time_minutes",
        ax=ax,
    
    )
    g.set(
        title=f"Distribution of listening time for {col}",
        xlabel="Listening Time (mins)")
    sns.despine()
    plt.show()



# Number of ads
fig, ax = plt.subplots(figsize=(9,6))
g=sns.boxplot(
    train, 
    x= "Number_of_Ads",
    y="Listening_Time_minutes",
    ax=ax
)
sns.despine()




for col in ["Host_Popularity_percentage", "Guest_Popularity_percentage", "Episode_Length_minutes"]: 
    fig, ax = plt.subplots(figsize=(20,8))
    g = sns.scatterplot(
        train,
        x= col,
        y="Listening_Time_minutes",
        ax=ax,
        s=0.5,
        marker="."
    )
    g.set(
        title=f"{col.replace('_', ' ')} and listening time",
        xlabel=f"{col.replace('_', ' ')}",
        ylabel= "Listening time (mins)",
    )
    sns.despine()
    plt.show()


# more than 100% -> 100%, 0 % -> to median
train.loc[train.Host_Popularity_percentage > 100,"Host_Popularity_percentage"] = 100
train.loc[train.Host_Popularity_percentage == 0,"Host_Popularity_percentage"] = train.Host_Popularity_percentage.median()

train.loc[train.Guest_Popularity_percentage > 100,"Guest_Popularity_percentage"] = 100
train.loc[train.Guest_Popularity_percentage == 0,"Guest_Popularity_percentage"] = train.Guest_Popularity_percentage.median()

#Episode_Length
train.loc[train.Episode_Length_minutes > 120,"Episode_Length_minutes"] = 120
train.loc[train.Episode_Length_minutes == 0,"Episode_Length_minutes"] = train.Episode_Length_minutes.median()








# Host popularity and guest popularity
plt.subplots(figsize=(9,6))
g = sns.scatterplot(
    train,
    x="Guest_Popularity_percentage",
    y="Host_Popularity_percentage",
    s=0.2,
    marker='.'
)
g.set(title="Guest and Host popularity percentage",
      xlabel="Guest Popularity",
      ylabel="Host Popularity")
sns.despine()


train.dtypes


train.isna().sum()


# handling datatypes and missing values
catcols = train.select_dtypes("object").columns
for col in catcols: 
    train[col] = pd.Categorical(train[col])

train["Episode_Sentiment"] = pd.Categorical(train.Episode_Sentiment, ordered =True, categories = ["Negative", "Neutral", "Positive"])
#train["Publication_Time"] = pd.Categorical(train.Publication_Time, ordered=True, categories=["Morning", "Afternoon", "Evening", "Night"])

# Let's fillna with the mean values first: 
for col in ["Episode_Length_minutes", "Guest_Popularity_percentage"]:
    train[col] = train[col].fillna(train[col].median())
for col in ["Number_of_Ads"]:
    train[col] = train[col].fillna(train[col].mode()[0])

train.isna().sum()


train.dtypes


# let's change categorical values to numeric ones with one hot encoding
# for the correlation, i change back Number of Ads to numeric

train["Number_of_Ads"] = pd.to_numeric(train.Number_of_Ads, errors="coerce")
display(train.head())




# checking pairplot again

ax = sns.pairplot(
     data = train.drop(columns =  ["id"]),
     plot_kws={"s": 0.2, "marker":"."})
ax.fig.suptitle("Relation of features to each other") 

plt.show()


train[train.Host_Popularity_percentage > 100]





train_encoded = pd.get_dummies(train.drop(columns = ["Episode_Title", "Podcast_Name", "id"]))
train_encoded.head()



fig, ax = plt.subplots(figsize = (15, 10))
g = sns.heatmap(
    train_encoded.corr(),
    linewidth = 1,
    annot=True,
    fmt = ".2f",
    ax=ax,
    
)
ax.set(title= "Correlation of features and target variable")





pip show lightgbm



from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
X = train_encoded.drop(columns = ["Listening_Time_minutes"])
y = train_encoded.Listening_Time_minutes

# let's use it with a simple LinearRegression

model=LinearRegression()
model = Lasso()
#model = KNeighborsRegressor()
model = LGBMRegressor()
model = LGBMRegressor(max_depth=8, n_estimators=1000)
# we do not have rmse so I need to make it

# I use a negativ sign, as the scorer will want to change the values to negative in order to maximize the value
scores = - cross_val_score(model, X, y, cv=5, scoring=make_scorer(mean_squared_error, squared=False, greater_is_better = False)) 
display(scores)
# I want to penalize variance 
expected_scores = np.mean(scores) + np.std(scores)
display(f"{model} expected_scores: {expected_scores}")


from sklearn.neighbors import KNeighborsRegressor
scaler = StandardScaler()

X_scaled =scaler.fit_transform(train_encoded.drop(columns = ["Listening_Time_minutes"]))
y = train_encoded.Listening_Time_minutes

model = LinearRegression()
model = Lasso()
#model = KNeighborsRegressor()
model = LGBMRegressor()


# I use a negativ sign, as the scorer will want to change the values to negative in order to maximize the value
scores = - cross_val_score(model, X_scaled, y, cv=5, scoring=make_scorer(mean_squared_error, squared=False, greater_is_better = False)) 
display(scores)
# I want to penalize variance 
expected_scores = np.mean(scores) + np.std(scores)
expected_scores                      


from sklearn.model_selection import RandomizedSearchCV

param_grid = { 'n_estimators': [100, 200, 500, 1000], 
              'learning_rate': [0.01, 0.05, 0.1, 0.2],  
              'max_depth': [4, 6, 8, 10], 
              #'num_leaves': [15, 31, 63, 127],  
              #'min_child_samples': [10, 20, 50, 100],  
              
             } 
estimator = LGBMRegressor()
"""
search = RandomizedSearchCV(estimator = estimator,
                        param_distributions = param_grid,
                        cv = 5,
                        scoring = 'neg_root_mean_squared_error',
                        random_state=0, verbose =4)

search.fit(X, y)

search_best_model = simple_gs.best_estimator_
display(f"Best model: {search_best_model}: {search.best_params_} : {search.best_score_}")
"""


#display(search.best_estimator_)
#display(search.best_score_)
model = LGBMRegressor(max_depth=8, n_estimators=1000)


test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
display(test.head())
display(test.isna().sum())
display(test.describe())


train.Number_of_Ads.value_counts().index[0]


# Number of Ads
# this is categorical data, so we want to fill it with the most frequent value

test.loc[test.Number_of_Ads.isna() | (test["Number_of_Ads"] > 3), "Number_of_Ads"] = train.Number_of_Ads.value_counts().index[0]
test["Number_of_Ads"] = test.Number_of_Ads.fillna(train.Number_of_Ads.median()).astype(int)

test.Number_of_Ads.unique()



test.loc[test.Episode_Length_minutes.isna() , "Episode_Length_minutes"] = train.Episode_Length_minutes.median()
test.loc[test.Episode_Length_minutes == 0,"Episode_Length_minutes"] = train.Episode_Length_minutes.median()

test.loc[(test.Episode_Length_minutes > 120), "Episode_Length_minutes"] = 120

test.loc[test.Guest_Popularity_percentage.isna(), "Guest_Popularity_percentage"] = train.Guest_Popularity_percentage.median()
test.loc[(test.Guest_Popularity_percentage > 100), "Guest_Popularity_percentage"] = 100
test.loc[test.Guest_Popularity_percentage == 0,"Guest_Popularity_percentage"] = train.Guest_Popularity_percentage.median()



test.loc[test.Host_Popularity_percentage.isna(), "Host_Popularity_percentage"] = train.Host_Popularity_percentage.median()
test.loc[test.Host_Popularity_percentage == 0,"Host_Popularity_percentage"] = train.Host_Popularity_percentage.median()

test.loc[(test.Host_Popularity_percentage > 100), "Host_Popularity_percentage"] = 100








test_encoded = pd.get_dummies(test.drop(columns = ["Episode_Title", "Podcast_Name", "id"]))


test_encoded.head()
test_encoded.isna().sum()
test_scaled = scaler.transform(test_encoded)


test_encoded.describe()



model.fit(X, y)
predictions = model.predict(test_encoded)
predictions[:5]


submission_df = pd.DataFrame({"Listening_Time_minutes": predictions, "id":test.id})
#submission_df.set_index= test.index
submission_df.to_csv("submission.csv", index=False)


submission_df


test.head()

