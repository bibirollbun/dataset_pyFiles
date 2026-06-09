 # imports

import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import plotly.express as px
from xgboost import XGBRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col="id")
display(train.head())
display(train.info())
display(train.describe())
display(train.isna().sum())



# reading test data
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col="id")
display(test.head())
display(test.isna().sum())


# filling missing values with train data mode
test["winddirection"] = test["winddirection"].fillna(train.winddirection.mode()[0])
test.isna().sum()


# Let's create some deltas from previous days data
train["d_pressure"] = (train.pressure - train.pressure.shift()).bfill()
train["d_temparature"] = (train.temparature - train.temparature.shift()).bfill()
train["d_dewpoint"] = (train.dewpoint - train.dewpoint.shift()).bfill()
train["d_winddirection"] = (train.winddirection - train.winddirection.shift()).bfill()
train["d_windspeed"] = (train.windspeed - train.windspeed.shift()).bfill()

test["d_pressure"] = (test.pressure - test.pressure.shift()).bfill()
test["d_temparature"] = (test.temparature - test.temparature.shift()).bfill()
test["d_dewpoint"] = (test.dewpoint - test.dewpoint.shift()).bfill()
test["d_winddirection"] = (test.winddirection - test.winddirection.shift()).bfill()
test["d_windspeed"] = (test.windspeed - test.windspeed.shift()).bfill()


display(train.isna().sum())
train.head()


# let's transfrom days to month: 

train["month"] = train.day.apply(lambda day: datetime.datetime.strptime(str(day), "%j").month)
test["month"] = test.day.apply(lambda day: datetime.datetime.strptime(str(day), "%j").month)



# let's transform winddirection to categories: 
"""
train["winddirection_cat"] = pd.cut(
    train.winddirection,
    bins=[0,90,180,270,360],
    labels=["North", "East", "South", "West"],
    right=False,
    include_lowest=True)
    
test["winddirection_cat"] = pd.cut(
    test.winddirection,
    bins=[0,90,180,270,360],
    labels=["North", "East", "South", "West"],
    right=False,
    include_lowest =True
)
"""
display(train.head())
display(test.head())
# transforming to long format
df_melt = train.melt(id_vars="rainfall", var_name= "property", value_name="value")
display(df_melt.head())
(df_melt.shape)

# Now we can make the 


# Let's check the max values again - a boxplot cannot infer much information if the features are on very different scale
display()
maxes = train.describe().loc["max"].sort_values(ascending=False)
display(type(maxes))
ax = sns.barplot(y=maxes.index, x=maxes.values)
ax.set_title("Max of various features")
sns.despine()
plt.show()
#plt.xticks(rotation=90)





props = ["pressure"]
for prop in props: 
    
    # checking min, max and rough distribution
    pressure_df = df_melt[df_melt.property== prop]
    
    ax = sns.boxplot(
        data = pressure_df,
        x = "property",
        y= "value",
        hue="rainfall")
    
    ax.set_title(f"Distribution of {prop} features for rainy and dry days")
    sns.despine()
    plt.show()
    
    ax2 = sns.displot(data = pressure_df,
                       x = "value",
                       row = "rainfall", 
                       facet_kws = {"sharey": True})
    ax2.fig.suptitle(f"Distribution of {prop} for dry and rainy days")
    
    plt.show()




df = df_melt[df_melt.property== "winddirection"]

ax = sns.boxplot(
    data = df,
    x = "property",
    y= "value",
    hue="rainfall")

ax.set_title("Distribution of winddirection features for rainy and dry days")
sns.despine()
plt.show()

ax2 = sns.displot(data = df,
                   x = "value",
                   row = "rainfall", 
                   facet_kws = {"sharey": True},
                   kde = True)
ax2.fig.suptitle("Distribution of winddirection for dry and rainy days")

plt.show()


train.head()





# Let's try and balance the rainy and dry days: 
balanced = df_melt.copy()
for i in range(3):
    balanced = pd.concat([balanced, df_melt[df_melt.rainfall ==0]])
balanced.shape
# Now let's check the winddirection: 
df = balanced[balanced.property == "winddirection"].copy()

ax3 = sns.displot(data = df,
                   x = "value",
                   row = "rainfall", 
                   facet_kws = {"sharey": True},
                   kde = True)
ax3.fig.suptitle("Distribution of winddirection for dry and rainy days")

plt.show()


train.head()


# winddirection - wind seems to be in buckets already. let's see if there are winddirections, where it is more likey to have rain

ax6 = train.groupby("winddirection").agg(avg_rain=("rainfall", "mean")).plot(kind="barh", title="Average ratio of rainy days per winddirection")
sns.despine()


ax7 = train.groupby("winddirection").agg(count=("rainfall", "count")).plot(kind="barh", stacked=True, title="Distribution of winddirection")



# Let's check, whether we have a similar datapoints for each day: 

days_count = train.groupby("day").agg(counts = ("month", "count")).reset_index(drop=False).groupby("counts").agg(days=("day", list), amount=("day", lambda day: len(set(list(day)))))
days_count


# calculating observations for each month
months = train.groupby("month").agg(counts = ("day", "count")).reset_index(drop=False)
#display(months)
display(f"Average number of observations per month: {months.counts.mean()}")


ax8 = sns.countplot(data=train, y="month")
ax8.set_title("Number of observations per month")
ax8.bar_label(ax8.containers[0])
plt.show()


df = df_melt[df_melt.property == "month"]
ax9 = train.groupby("month").agg(avg_rainy_days = ("rainfall", "mean")).plot(kind="barh", title = "Ratio of rainy days per month" )
ax9.bar_label(ax9.containers[0], fmt="%.2f")
sns.despine()
plt.show()
#df = balanced[balanced.property == "month"].copy()



train.head()


###### fig, ax8 = plt.subplots(figsize = (13,6))
features = ["maxtemp", "temparature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "d_pressure", "d_temparature", "d_dewpoint"]
sns.boxplot(
    data=df_melt[df_melt.property.isin(features)].copy(),
    y="property",    
    x = "value",
    hue='rainfall',
    ax=ax8)
sns.despine()
plt.show()


train.d_winddirection.unique()


for feature in ["humidity", "cloud", "sunshine","d_pressure", "d_temparature", "d_dewpoint", "d_winddirection", "d_windspeed"]: 
    df = df_melt[df_melt.property == feature].copy()
    #display(df.head())
    ax7 = sns.displot(
        data=df,
        x="value", 
        hue="rainfall"
    )
    ax7.fig.suptitle(f"Distribution of {feature} for rainy and dry days")
    
    plt.show()


corr = train.corr()
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(
    corr, 
    annot =True, 
    fmt = ".2f", 
    linewidth=.1,
    ax=ax
)
ax.set_title("Correlation of features")
plt.show()



# Let's drop maxtemp and mintemp

train = train.drop(columns=["maxtemp", "mintemp"])
test = test.drop(columns = ["maxtemp", "mintemp"])


# let's make a categorical data from month
train["m"] =train.month.apply(lambda month: datetime.datetime.strftime(datetime.datetime(1, month, 1), "%b"))
test["m"] = test.month.apply(lambda month: datetime.datetime.strftime(datetime.datetime(1, month, 1), "%b"))
train = train.drop(columns = ["month", "day"])
test = test.drop(columns = ["month", "day"])

#train = train.drop(columns = ["month"])
train = pd.get_dummies(train)
test = pd.get_dummies(test)
train.head()


# let's check the correlations again:

corr = train.corr()
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(
    corr, 
    annot =True, 
    fmt = ".2f", 
    linewidth=.1,
    ax=ax
)
ax.set_title("Correlation of features")
plt.show()








from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, XGBClassifier

# scaling: 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#scaler = MinMaxScaler()

scaler = StandardScaler()
cols_to_drop = ["winddirection", "d_temparature", "m_Dec", "m_Feb", "m_Mar","m_May", "m_Jun", "m_Aug","m_Oct"] # we leave Jan April July Sep, Nov, 
test = test.drop(columns = cols_to_drop)
X = train.drop(columns = ["rainfall"] + cols_to_drop)
y = train.rainfall


X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)





#model=ExtraTreesRegressor()
#

#model = GradientBoostingRegressor()
model=RandomForestRegressor()
model = LinearRegression()
model = LogisticRegression(max_iter=1000)

model=KNeighborsRegressor()
model=KNeighborsRegressor(n_neighbors=5)


model = RandomForestRegressor(max_depth=10, min_samples_split=10, n_estimators=200)

model = Lasso(alpha=0.001, tol=0.001)
model=Ridge(**{'alpha': 100} )
model = GradientBoostingRegressor(**{'learning_rate': 0.01,
 'max_depth': 3,
 'min_samples_split': 10,
 'n_estimators': 500})


#model = XGBRegressor(**{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100} )

model= RandomForestClassifier(**{'max_depth': 7, 'min_samples_split': 5, 'n_estimators': 500})

scores = cross_val_score(model, X, y, cv=5, scoring = make_scorer(roc_auc_score), verbose = 3)
display(scores)

print(np.mean(scores))
print(np.std(scores))
final_score = np.mean(scores) - np.std(scores)                        
display(f"Final score of this model {model}: {final_score}.")


# hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier

model = Lasso()
param_grid = {"alpha": [0.001, 0.01, 0.1, 0.5], 
             "tol": [0.001, 0.0001, 0.01]}





model = XGBRegressor()
param_grid = {"n_estimators": [100, 150, 200, 300, 500],
              "learning_rate": [0.01, 0.05, 0.1, 0.5],
              "max_depth": [3, 6, 9]}



model = Ridge()
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

model = Lasso()
param_grid = {"alpha": [0.001, 0.01, 0.1, 0.5], 
             "tol": [0.001, 0.0001, 0.01]}

model = GradientBoostingRegressor()
param_grid = {
    "learning_rate":[0.01, 0.1],
    "n_estimators":[100, 200, 500], 
    "max_depth": [3,7,],
    "min_samples_split": [5,10]
}

model = RandomForestClassifier() # let's try a classifier
param_grid = {
    "n_estimators":[100, 200, 500], 
    "max_depth": [3,7,9],
    "min_samples_split": [5,7, 10]
}


simple_gs = GridSearchCV(
    estimator =model,
    param_grid = param_grid,

    verbose=4,
    n_jobs=-1,
    scoring="roc_auc"
)

simple_gs.fit(X, y)
simple_gs_best_model = simple_gs.best_estimator_
display(f"Best model: {simple_gs_best_model}: {simple_gs.best_params_} : {simple_gs.best_score_}")


simple_gs.best_params_


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# splitting train data és training the first and the second model
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75, random_state=0)

model_1=RandomForestRegressor()
model_1.fit(X_train, y_train)
predictions_1_valid = model_1.predict(X_valid)
roc_auc_1 = roc_auc_score(y_valid, predictions_1_valid)
display(f"{model_1} roc_auc score: {roc_auc_1}")

#display(predictions_1[:10])
#display(y_valid[:10])

predictions_1_valid_reshaped = predictions_1_valid.reshape(-1, 1)

model_2 = LinearRegression()
model_2.fit(predictions_1_valid_reshaped, y_valid)

# applying to the whole train set to predict on test dataset: 

test_predictions_1 = model_1.predict(test)
test_predictions_1_reshaped = test_predictions_1.reshape(-1, 1)
test_predictions_2 = model_2.predict(test_predictions_1_reshaped)
test_predictions_2[:10]


# stacking with 2 hyperparameter tuning
model_1 = Lasso()
param_grid = {"alpha": [0.001, 0.01, 0.1, 0.5], 
             "tol": [0.001, 0.0001, 0.01]}




grid = GridSearchCV(
    estimator= model_1,
    cv=5,
    param_grid=param_grid,
    scoring="roc_auc",
    n_jobs =-1,
    verbose=5
)

grid.fit(X_train, y_train)
stacked_best_model_1 = grid.best_estimator_

#model_1.fit(X_train_1, y_train_1)
#predictions_1_valid = model_1.predict(X_valid_1)
predictions_1_train = stacked_best_model_1.predict(X_train)

roc_auc_1 = roc_auc_score(y_train, predictions_1_train)
display(f"{stacked_best_model_1} roc_auc score: {roc_auc_1}")

#display(predictions_1[:10])
#display(y_valid[:10])

# model 2 learning
predictions_1_train_reshaped = predictions_1_train.reshape(-1, 1)

#model_2 = LinearRegression()

model_2= RandomForestRegressor()

param_grid_2 = {
    "n_estimators":[100,200,500],
    "max_depth" : [None, 10, 20],
    "min_samples_split":[2,5,10]
}

gs2 = GridSearchCV(
    estimator = model_2,
    scoring="roc_auc",
    param_grid=param_grid_2,
    cv=5,
    n_jobs=-1,
    verbose=3,
)

gs2.fit(predictions_1_train_reshaped, y_train)

stacked_best_model_2 = gs2.best_estimator_
display(f"best model 2: {stacked_best_model_2}, {gs2.best_score_}")

# Predictions on the valid data: 
final_predictions_1 = stacked_best_model_1.predict(X_valid)
final_predictions_1_reshaped = final_predictions_1.reshape(-1,1)

final_predictions_valid = stacked_best_model_2.predict(final_predictions_1_reshaped)

# measureing ROC AUC: 

roc_auc_final = roc_auc_score(y_valid, final_predictions_valid)
display(f"After {stacked_best_model_2}, the final roc_auc score: {roc_auc_final}")


# fitting the model to the whole train dataset
if False: 
    model = simple_gs_best_model

    model.fit(X, y)
    predictions = model.predict(test)
    #display(predictions)
    prediction_df = pd.DataFrame({"rainfall":predictions}, index=test.index)
    prediction_df
    
    prediction_df.to_csv("submission.csv")



# Stacked model: fitting the model to the whole train dataset

predictions_1 =stacked_best_model_1.predict(test)

predictions_1_reshaped = predictions_1.reshape(-1, 1)

predictions_2 = stacked_best_model_2.predict(predictions_1_reshaped)
#display(predictions)
prediction_df = pd.DataFrame({"rainfall":predictions_2}, index=test.index)
prediction_df

prediction_df.to_csv("submission.csv")

