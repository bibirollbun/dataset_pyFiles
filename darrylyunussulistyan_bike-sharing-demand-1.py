import numpy as np
import pandas as pd


import seaborn as sns


import matplotlib.pyplot as plt


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor


df_test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
df_train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')


df_train


df_test


df_train['datetime'] = pd.to_datetime(df_train['datetime'])
df_test['datetime'] = pd.to_datetime(df_test['datetime'])

def split_date(df):
    df['date_year'] = df['datetime'].dt.year
    df['date_month'] = df['datetime'].dt.month
    df['date_day_of_month'] = df['datetime'].dt.day
    df['date_day_of_week'] = df['datetime'].dt.dayofweek
    df['date_week'] = df['datetime'].dt.isocalendar().week
    df['date_hour'] = df['datetime'].dt.hour



split_date(df_train)
split_date(df_test)


df_train


def plot_by_sum_count(cols, figsize=(15,7)):
    for col in cols:
        df_train_season_count = df_train[[col, 'count']].groupby(col).agg(['sum'])
        df_train_season_avg = df_train[[col, 'count']].groupby(col).agg(['mean'])
        
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        df_train_season_count.plot(kind="bar", title = "%s count sum" % col, ylabel = "sum", ax=ax1)
        df_train_season_avg.plot(kind="bar", title = "%s count mean" % col, ylabel = "mean", ax=ax2)

        df_col_count = df_train[col].value_counts().reset_index()
        df_col_count.plot(x=col, y='count', kind = 'bar', ax=ax3)


plot_by_sum_count(['season', 'weather', 'date_year', 'date_month', 'date_day_of_month', 'date_day_of_week', 'date_week', 'date_hour', 'holiday', 'workingday'])


df_train['weather'].value_counts()


df_test['weather'].value_counts()


season_winter_2_fall = {
    1 : 1,
    2 : 2,
    3 : 3,
    4 : 3
}
df_train['weather'] = df_train['weather'].map(season_winter_2_fall)
df_test['weather'] = df_test['weather'].map(season_winter_2_fall)


df_train['weather'].value_counts(), df_test['weather'].value_counts()


#df_train.plot.scatter(x="temp", y="atemp")
sns.regplot(data=df_train, x="temp", y="atemp", line_kws=dict(color="r"))


#df_train.plot.scatter(x="temp", y="count")
sns.regplot(data=df_train, x="temp", y="count", line_kws=dict(color="r"))


sns.regplot(data=df_train, x="humidity", y="count", line_kws=dict(color="r"))


sns.regplot(data=df_train, x="windspeed", y="count", line_kws=dict(color="r"))


df_numeric_corr = df_train[['temp','atemp','humidity','windspeed','count']].corr()
sns.heatmap(df_numeric_corr, annot=True)


df_train['count'].plot.hist()


df_train['count_log'] = np.log1p(df_train['count'])


df_train['count_log'].plot.hist()


sns.regplot(data=df_train, x="temp", y="count_log", line_kws=dict(color="r"))


sns.regplot(data=df_train, x="humidity", y="count_log", line_kws=dict(color="r"))


sns.regplot(data=df_train, x="windspeed", y="count_log", line_kws=dict(color="r"))


df_train['holiday'].value_counts()


df_train['workingday'].value_counts()


fig, axs = plt.subplots(nrows = 2)
sns.pointplot(data = df_train, x = 'date_hour', y = 'count', hue = 'workingday', ax = axs[0])
sns.pointplot(data = df_train, x = 'date_hour', y = 'count', hue = 'holiday', ax = axs[1])


df_train['is_midday'] = df_train['date_hour'].apply(lambda x : 1 if x >= 10 and x <= 16 else 0)
df_test['is_midday'] = df_test['date_hour'].apply(lambda x : 1 if x >= 10 and x <= 16 else 0)

df_train['is_night'] = df_train['date_hour'].apply(lambda x : 1 if x >= 20 or x <= 6 else 0)
df_test['is_night'] = df_test['date_hour'].apply(lambda x : 1 if x >= 20 or x <= 6 else 0)

df_train['is_commute'] = df_train['date_hour'].apply(lambda x : 1 if (x >= 7 and x <= 9) or (x >= 17 and x <= 19) else 0)
df_test['is_commute'] = df_test['date_hour'].apply(lambda x : 1 if (x >= 7 and x <= 9) or (x >= 17 and x <= 19) else 0)


df_train


df_test


df_train.columns


cols = [
   'season', 'holiday', 'workingday', 'weather',
   'date_year', 'date_month', 'date_day_of_week', 'date_hour', 
    'is_midday', 'is_night', 'is_commute'
]
fig, axs = plt.subplots(nrows = 4, ncols = 3, figsize = (15, 15))
#print(axs)
for i in range(0, len(cols)):
    row = i // 3
    col = i % 3
    sns.boxplot(data = df_train, x = cols[i], y = 'count', ax = axs[row][col])


def plot_distr(cols, figsize=(12,7)):
    for col in cols:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        df_train[col].plot.hist(title = "train %s distr" % col, ylabel = col, ax=ax1)
        df_test[col].plot.hist(title = "test %s distr" % col, ylabel = col, ax=ax2)


def plot_box(cols, figsize=(12,7)):
    for col in cols:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        sns.boxplot(data = df_train, x = col, ax = ax1).set_title("train %s" % col)
        sns.boxplot(data = df_test, x = col, ax = ax2).set_title("test %s" % col)


df_train['windspeed_log'] = np.log1p(df_train['windspeed'])
df_test['windspeed_log'] = np.log1p(df_test['windspeed'])

df_train['temp_log'] = np.log1p(df_train['temp'])
df_test['temp_log'] = np.log1p(df_test['temp'])

df_train['humidity_log'] = np.log1p(df_train['humidity'])
df_test['humidity_log'] = np.log1p(df_test['humidity'])


cols = [
    'temp',
    'temp_log',
    'humidity',
    'humidity_log',
    'windspeed',
    'windspeed_log',
]
plot_box(cols)


sns.regplot(data=df_train, x="windspeed_log", y="count_log", line_kws=dict(color="r"))



sns.regplot(data=df_train, x="temp_log", y="count_log", line_kws=dict(color="r"))



sns.regplot(data=df_train, x="humidity_log", y="count_log", line_kws=dict(color="r"))


#import shap
#shap.initjs()


model = RandomForestRegressor()
cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    'date_day_of_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]
X_train = df_train[cols]
y_train = df_train['count_log']
model.fit(X_train, y_train)


# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();


#shap_explainer = shap.Explainer(model)
## too long
#shap_values = shap_explainer.shap_values(X_train)
#shap.summary_plot(shap_values, X_train)


#df_train_rb = df_train.copy()


def strat_k_fold(X_df, y_df, model, metrics = [mean_squared_error]):
    skf = KFold(n_splits=5, shuffle=True, random_state=4761)
    metrics_stat = { fn.__name__ : [] for fn in metrics }
      
    for train_index, test_index in skf.split(X_df, y_df):
        x_train_fold, x_test_fold = X_df.iloc[train_index, :], X_df.iloc[test_index, :]
        y_train_fold, y_test_fold = y_df[train_index], y_df[test_index]
        model.fit(x_train_fold, y_train_fold)
        y_hat_fold = model.predict(x_test_fold)

        for metric in metrics:
            metrics_stat[metric.__name__].append(metric(y_test_fold, y_hat_fold))

    return metrics_stat


def try_train_models(X_df, y_df, models=[], metrics=[mean_squared_error]):
    model_names = []
    model_scores = []
    model_df = {}

    def list_get(name):
        if name not in model_df:
            model_df[name] = []
        return model_df[name]
    
    for pm in models:
        if type(pm) is tuple:
            model = pm[0]
            name = pm[1]
        else:
            model = pm
            name = model.__class__.__name__
            
        metrics = strat_k_fold(X_df, y_df, model, metrics=metrics)

        list_get('Model').append(name)
        for avgm, sc in metrics.items():
            list_get(avgm).append(np.mean(sc))
    
    model_df_real = pd.DataFrame(model_df)
    
    return model_df_real


df_train.head()


df_test.head()


df_train['train'] = 1
df_test['train'] = 0
df_all = pd.concat([df_train, df_test])


df_all


df_all.reset_index(inplace=True)


classgm = RandomForestClassifier()
cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    'atemp',
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    'date_day_of_month',
    'date_day_of_week',
    'date_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]
try_train_models(df_all[cols], df_all['train'], [classgm], metrics=[roc_auc_score])


classgm.fit(df_all[cols], df_all['train'])


# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(classgm.feature_importances_, index=cols).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();


classgm = RandomForestClassifier()
cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    'atemp',
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    #'date_day_of_month',
    'date_day_of_week',
    'date_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]
try_train_models(df_all[cols], df_all['train'], [classgm], metrics=[roc_auc_score])


# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(classgm.feature_importances_, index=cols).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();


df_train['date_week'].plot.hist()


df_test['date_week'].plot.hist()


classgm = RandomForestClassifier()
cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    'atemp',
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    #'date_day_of_month',
    'date_day_of_week',
    #'date_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]
try_train_models(df_all[cols], df_all['train'], [classgm], metrics=[roc_auc_score])


# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(classgm.feature_importances_, index=cols).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();


plt, (ax1, ax2) = plt.subplots(1, 2)
df_train['date_month'].plot.hist(ax=ax1)
df_test['date_month'].plot.hist(ax=ax2)


classgm = RandomForestClassifier()
cols = [
    'season',
    'holiday',
    'workingday',
    #'weather',
    'temp',
    #'atemp',
    #'humidity',
    #'windspeed',
    'date_year',
    #'date_month',
    #'date_day_of_month',
    'date_day_of_week',
    #'date_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]
try_train_models(df_all[cols], df_all['train'], [classgm], metrics=[roc_auc_score])


## Create a series containing feature importances from the model and feature names from the training data
#feature_importances = pd.Series(classgm.feature_importances_, index=cols).sort_values(ascending=False)
#
## Plot a simple bar chart
#feature_importances.plot.bar();


# for unknown reason, plt.subplots decided that it doesn't like figsize and throw error. Wtf? Reimport the plt and try again.
import matplotlib.pyplot as plt
plt.subplots(1, 3, figsize=(1,1))


cols =  [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    'atemp',
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    'date_day_of_month',
    'date_day_of_week',
    'date_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]

plot_distr(cols)


df_train.columns


cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    'date_day_of_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
    
]

X_train = df_train[cols]
y_train = df_train['count']
try_train_models(X_train, y_train, [RandomForestRegressor()], metrics=[mean_squared_error, mean_squared_log_error])


cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    'date_day_of_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]

X_train = df_train[cols]
y_train = df_train['count_log']
try_train_models(X_train, y_train, [RandomForestRegressor()], metrics=[mean_squared_error])


cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    'date_day_of_week',
    'date_hour',
    #"is_midday",
    #"is_night",
    #"is_commute",
]

X_train = df_train[cols]
y_train = df_train['count_log']
try_train_models(X_train, y_train, [RandomForestRegressor()], metrics=[mean_squared_error])


cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    "atemp",
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    'date_day_of_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]

X_train = df_train[cols]
y_train = df_train['count_log']
try_train_models(X_train, y_train, [RandomForestRegressor()], metrics=[mean_squared_error])


cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    #'temp',
    "atemp",
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    'date_day_of_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]

X_train = df_train[cols]
y_train = df_train['count_log']
try_train_models(X_train, y_train, [RandomForestRegressor()], metrics=[mean_squared_error])


#cols = [
#    #'season',
#    #'holiday',
#    'workingday',
#    #'weather',
#    'temp',
#    #'atemp',
#    #'humidity',
#    #'windspeed',
#    'date_year',
#    'date_month',
#    #'date_day_of_week',
#    'date_hour'
#]
#
#X_train = df_train[cols]
#y_train = df_train['count_log']
#try_train_models(X_train, y_train, [RandomForestRegressor()], metrics=[mean_squared_error])


cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    #"atemp",
    'humidity',
    #'windspeed',
    'date_year',
    'date_month',
    'date_day_of_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]

X_train = df_train[cols]
y_train = df_train['count_log']
try_train_models(X_train, y_train, [RandomForestRegressor()], metrics=[mean_squared_error])


#cols = [
#    'season',
#    'holiday',
#    'workingday',
#    'weather',
#    'temp',
#    #'atemp',
#    'humidity',
#    #'windspeed',
#    'date_year',
#    'date_month',
#    'date_day_of_week',
#    'date_hour'
#]
#
#X_train = df_train[cols]
#y_train = df_train['count_log']
#try_train_models(X_train, y_train, [RandomForestRegressor()], metrics=[mean_squared_error])


#cols = [
#    'season',
#    'holiday',
#    'workingday',
#    'weather',
#    'temp',
#    'atemp',
#    'humidity',
#    #'windspeed',
#    'date_year',
#    'date_month',
#    'date_day_of_week',
#    'date_hour'
#]
#
#X_train = df_train[cols]
#y_train = df_train['count_log']
#try_train_models(X_train, y_train, [RandomForestRegressor()], metrics=[mean_squared_error])


cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    #"atemp",
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    'date_day_of_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]

X_train = df_train[cols]
y_train = df_train['count_log']
try_train_models(X_train, y_train, [lgb.LGBMRegressor()], metrics=[mean_squared_error])


model = lgb.LGBMRegressor()
cols = [
    'season',
    'holiday',
    'workingday',
    'weather',
    'temp',
    #"atemp",
    'humidity',
    'windspeed',
    'date_year',
    'date_month',
    'date_day_of_week',
    'date_hour',
    "is_midday",
    "is_night",
    "is_commute",
]
X_train = df_train[cols]
y_train = df_train['count_log']
model.fit(X_train, y_train)


df_test


y_hat = model.predict(df_test[cols])


y_hat


np.exp(y_hat)-1


df_submission = pd.DataFrame({
    "datetime": df_test['datetime'],
    "count": np.exp(y_hat)-1
})


!head /kaggle/input/bike-sharing-demand/sampleSubmission.csv


df_submission.to_csv("./submission.csv", index=False)


!head submission.csv

