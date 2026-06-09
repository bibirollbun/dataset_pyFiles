import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, mean_absolute_error


train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')
df = train.copy()
df_test = test.copy()


display(df.head())
display(df_test.head())


df.info()


df.describe(include='all').T


df_test.info()


df_test.describe(include='all').T


df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour

# drop the original feature.
df = df.drop(['datetime'], axis=1)

df.describe().T


train.describe().T


plt.figure(figsize=(8, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='rainbow')


print(round(correlation_matrix, 2) >= 0.97)


print(round(correlation_matrix['count'], 2))


df = df.drop(['casual', 'registered', 'holiday', 'atemp'], axis=1)
df.info()


sns.pairplot(df, corner=True)


fix, ax = plt.subplots(1, 3, figsize=(15, 4))

# uni-variate analysis
count = sns.countplot(data=df, x=df['season'], ax=ax[0])
count.set_title('Distribution of season')
count.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])

# bi-variate analysis
bar = sns.barplot(data=df, x=df['season'], y=df['count'], ax=ax[1])
bar.set_title('Season vs Count')
bar.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])

# multi-variate analysis
violin = sns.violinplot(data=df, x=df['season'], y=df['count'], hue=df['workingday'], ax=ax[2])
violin.set_title('Season vs Count by working day')
violin.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])


fix, ax = plt.subplots(1, 3, figsize=(15, 4))

# uni-variate analysis
count = sns.countplot(data=df, x=df['workingday'], ax=ax[0])
count.set_title('Distribution of workingday')
count.set_xticklabels(['Weekend/holiday', 'Workingday'])

# bi-variate analysis
bar = sns.barplot(data=df, x=df['workingday'], y=df['count'], ax=ax[1])
bar.set_title('Working day vs. Count')
bar.set_xticklabels(['Weekend/holiday', 'Workingday'])

# multi-variate analysis
bar = sns.barplot(data=df, x=df['workingday'], y=df['count'], hue=df['year'])
bar.set_title('Working day vs. Count by Year')
bar.set_xticklabels(['Weekend/holiday', 'Workingday'])


df.groupby('workingday').mean()


fix, ax = plt.subplots(1, 3, figsize=(15, 4))

# uni-variate analysis
count = sns.countplot(data=df, x=df['weather'], ax=ax[0])
count.set_title('Distribution of weather')
count.set_xticklabels(['Sunny', 'Cloudy', 'Minor rain/snow', 'rain/snow storm'], rotation=45)

# bi-variate analysis
bar = sns.barplot(data=df, x=df['weather'], y=df['count'], ax=ax[1])
bar.set_title('Weather vs. Count')
bar.set_xticklabels(['Sunny', 'Cloudy', 'rain/snow', 'rain/snow storm'], rotation=45)

# multi-variate analysis
bar = sns.barplot(data=df, x=df['weather'], y=df['count'], hue=df['season'])
bar.set_title('Weather vs. Count by Season')
bar.set_xticklabels(['Sunny', 'Cloudy', 'rain/snow', 'rain/snow storm'], rotation=45)


fix, ax = plt.subplots(1, 3, figsize=(15, 4))

# uni-variate analysis
hist = sns.histplot(df['temp'], ax=ax[0], bins=50)
hist.set(xlabel='Actual Temperature', title='Distribution of temperature')
hist.axvline(df['temp'].median(), color='black', linestyle='dashed', linewidth=2)
hist.axvline(df['temp'].mean(), color='red', linestyle='dashed', linewidth=2)

# bi-variate analysis
scatter = sns.scatterplot(data=df, x=df['temp'], y=df['count'], ax=ax[1])
scatter.set(xlabel='Actual Temperature', ylabel = 'Rented Count', title='Bike Rented Vs Actual Temperature')

# line plot
line = sns.lineplot(data=df, x='temp', y='count')
line.set(xlabel='Actual Temperature', ylabel = 'Rented Count', title='Average Bike Rented Vs Actual Temperature')


fix, ax = plt.subplots(1, 3, figsize=(15, 4))

# uni-variate analysis
hist = sns.histplot(df['humidity'], ax=ax[0])
hist.set(xlabel='Humidity', title='Distribution of humidity')
hist.axvline(df['humidity'].mean(), color='red', linestyle='dashed', linewidth=2)
hist.axvline(df['humidity'].median(), color='black', linestyle='dashed', linewidth=2)

# bi-variate analysis
scatter = sns.scatterplot(data=df, x=df['humidity'], y=df['count'], ax=ax[1])
scatter.set(xlabel='Humidity', ylabel = 'Rented Count', title='Bike Rented Vs Humidity')

# line plot
line = sns.lineplot(data=df, x='humidity', y='count')
line.set(xlabel='Humidity', ylabel = 'Rented Count', title='Average Bike Rented Vs Humidity')


fix, ax = plt.subplots(1, 3, figsize=(15, 4))

# uni-variate analysis
hist = sns.histplot(df['windspeed'], ax=ax[0], bins=15)
hist.set(xlabel='windspeed', title='Distribution of windspeed')
hist.axvline(df['windspeed'].mean(), color='red', linestyle='dashed', linewidth=2)
hist.axvline(df['windspeed'].median(), color='black', linestyle='dashed', linewidth=2)

# bi-variate analysis
scatter = sns.scatterplot(data=df, x=df['windspeed'], y=df['count'], ax=ax[1])
scatter.set(xlabel='windspeed', ylabel = 'Rented Count', title='Bike Rented Vs windspeed')

# line plot
line = sns.lineplot(data=df, x='windspeed', y='count')
line.set(xlabel='windspeed', ylabel = 'Rented Count', title='Average Bike Rented Vs windspeed')


fix, ax = plt.subplots(1, 3, figsize=(15, 4))

# uni-variate analysis
hist = sns.histplot(df['year'], ax=ax[0])
hist.set(xlabel='year', title='Distribution of year')

# bi-variate analysis
scatter = sns.scatterplot(data=df, x=df['year'], y=df['count'], ax=ax[1])
scatter.set(xlabel='year', ylabel = 'Rented Count', title='Bike Rented Vs year')

# line plot
line = sns.lineplot(data=df, x='year', y='count')
line.set(xlabel='year', ylabel = 'Rented Count', title='Average Bike Rented Vs year')


fix, ax = plt.subplots(1, 3, figsize=(15, 4))

# uni-variate analysis
hist = sns.histplot(df['month'], ax=ax[0])
hist.set(xlabel='month', title='Distribution of month')

# bi-variate analysis
scatter = sns.scatterplot(data=df, x=df['month'], y=df['count'], ax=ax[1])
scatter.set(xlabel='month', ylabel = 'Rented Count', title='Bike Rented Vs month')

# line plot
line = sns.lineplot(data=df, x='month', y='count')
line.set(xlabel='month', ylabel = 'Rented Count', title='Average Bike Rented Vs month')


year_month_group_df = df.groupby(['year', 'month'])['count'].mean().reset_index()
line = sns.lineplot(data=year_month_group_df, x='month', y='count', hue='year')


fix, ax = plt.subplots(1, 3, figsize=(15, 4))

# uni-variate analysis
hist = sns.histplot(df['hour'], ax=ax[0])
hist.set(xlabel='hour', title='Distribution of hour')

# bi-variate analysis
scatter = sns.scatterplot(data=df, x=df['hour'], y=df['count'], ax=ax[1])
scatter.set(xlabel='hour', ylabel = 'Rented Count', title='Bike Rented Vs hour')

# line plot
line = sns.lineplot(data=df, x='hour', y='count')
line.set(xlabel='hour', ylabel = 'Rented Count', title='Average Bike Rented Vs hour')


fix, ax = plt.subplots(1, 2, figsize=(15, 4))

# uni-variate analysis
hist = sns.histplot(df['count'], ax=ax[0])
hist.set(xlabel='count', title='Distribution of count')
hist.axvline(df['count'].mean(), color='red', linestyle='dashed', linewidth=2)
hist.axvline(df['count'].median(), color='black', linestyle='dashed', linewidth=2)

# box-plot
box = sns.boxplot(df['count'], orient='h')
box.set(xlabel='count', title='Outlier')


# Seterate the train data(df) to target feature and the data features
X = df.drop('count', axis=1)
y = df['count']


# Split the datsets into train dataset, validation dataset, since test datasets were given
kf = KFold(n_splits=5, shuffle=True, random_state=18)

best_rmse = float('inf')
best_train_index = None
best_val_index = None

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model with validation datasets
    y_val_pred = model.predict(X_val)
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val,y_val)
    print(f"Train Model score: {train_score}")
    print(f"Validation Model score: {val_score}")

    mse = mean_squared_error(y_val, y_val_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2_score: {r2}")

    # store the best result indexes
    if rmse < best_rmse:
        best_rmse = rmse
        best_train_index = train_index
        best_val_index = val_index

X_train, X_val = X.iloc[best_train_index], X.iloc[best_val_index]
y_train, y_val = y.iloc[best_train_index], y.iloc[best_val_index]

X_train.shape, y_train.shape, X_val.shape, y_val.shape


# null value check **train, validation, test data is derived from df data**
check_list = [X_train, y_train, X_val, y_val]
item_names = ['X_train', 'y_train', 'X_val', 'y_val']

for i, name in zip(check_list, item_names):
    print(f'---{name}---')
    print(f"{i.isna().sum()}\n")


# duplicated value check
# the target data y (including train, validation, test data) is numerical vector which counts the total rent of registered and casual user -- no need for duplication check
check_list = [X_train, X_val]
name_slice = [item_names[i] for i in [0, 2]]

for i, name in zip(check_list, name_slice):
    print(f"---{name}---")
    print(f"duplicated values: {i.duplicated().sum()}\n")


# concatenate the input variables and target variables to diagnose the duplicates
train_data = pd.concat([X_train, y_train.rename('target')], axis=1)

duplicated_rows = train_data[train_data.duplicated(keep=False)]

print(duplicated_rows)


# apply it to validation dataset as well.
val_data = pd.concat([X_val, y_val.rename('target')], axis=1)

val_data.duplicated(keep=False).sum()


# Split the training datasets into input and target
train_data = train_data.drop_duplicates(keep='first')

X_train = train_data.drop('target', axis=1)
y_train = train_data['target']


# outlier check
X_train.describe(include='all'), X_val.describe(include='all')


num_cols = ['temp', 'humidity', 'windspeed']
for i in num_cols:
    plt.boxplot(X_train[i])
    plt.title(f"{i}")
    plt.show()


def outlier_clip(data_input):
    for feature in data_input[num_cols]:
        q1, q3 = np.percentile(X_train[feature], [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr

        data_input[feature] = data_input[feature].clip(upper_bound, lower_bound)

    return data_input

print(X_train.describe().T)
X_train = outlier_clip(X_train)
print(X_train.describe().T)


for i in num_cols:
    plt.boxplot(X_train[i])
    plt.title(f"{i}")
    plt.show()


for i in num_cols:
    plt.boxplot(X_val[i])
    plt.title(f"{i}")
    plt.show()


print(X_val.describe().T)
X_val = outlier_clip(X_val)
print(X_val.describe().T)


for i in num_cols:
    plt.boxplot(X_val[i])
    plt.title(f"{i}")
    plt.show()


fig,ax = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(num_cols):
    hist = sns.histplot(X_train[col], ax=ax[i], bins=15)
    hist.set_title(f"{col}")


scaler = MinMaxScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])


X_train.describe(), X_val.describe()


cat_cols = ['season', 'workingday', 'weather', 'year', 'month', 'hour']

[print(f'{i} unique:\n{X_train[i].unique()}\n') for i in cat_cols]


seasons = pd.get_dummies(X_train['season'], prefix='season').astype(int)

X_train = pd.concat([X_train, seasons], axis=1)

X_train = X_train.drop('season', axis=1)
X_train.info()


seasons = pd.get_dummies(X_val['season'], prefix='season').astype(int)
seasons
X_val = pd.concat([X_val, seasons], axis=1)
X_val = X_val.drop('season', axis=1)
X_val.info()


binary_encode = {2011: 0, 2012: 1}

X_train['year'] = X_train['year'].map(binary_encode)
X_train.describe().T


X_val['year'] = X_val['year'].map(binary_encode)
X_val.describe().T


l_model = LinearRegression()
r_model = Ridge()
lr_model = Lasso()
log_model = LogisticRegression()
l_model.fit(X_train, y_train)
r_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
log_model.fit(X_train, y_train)


models = {
    'Linear Regression': l_model,
    'Ridge Regression': r_model,
    'Lasso Regression': lr_model,
    'Logistic Regression': log_model
}

results={}

for name,model in models.items():
    y_pred = model.predict(X_train)
    rmse = mean_squared_error(y_train, y_pred) ** 0.5
    r2 = r2_score(y_train, y_pred)
    MAE = mean_absolute_error(y_train, y_pred)
    model_score = model.score(X_train,y_train)
    results[name] = {'RMSE': rmse, 'R²': r2, 'MAE': MAE,'Model Score' : model_score}

results_df = pd.DataFrame(results).T
results_df


fig, ax = plt.subplots(1, 3, figsize=(15, 4))

sns.distplot(np.log1p(y_train), kde=True, ax=ax[0])
sns.distplot(np.sqrt(y_train), kde=True, ax=ax[1])
sns.distplot(np.cbrt(y_train), kde=True, ax=ax[2])


fig, ax = plt.subplots(1,2,figsize=(15,4))

dist = sns.distplot(np.sqrt(y_train), kde=True, ax=ax[0])
dist.axvline(np.sqrt(y_train).mean(), color='red', linestyle='dashed', linewidth=2)
dist.axvline(np.sqrt(y_train).median(), color='black', linestyle='dashed', linewidth=2)

box = sns.boxplot(np.sqrt(y_train), ax=ax[1])


y_train = np.cbrt(y_train)
y_val = np.cbrt(y_val)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0, 1, 10, 20, 30, 40],
    'max_iter': [1000, 2000, 3000, 3500],
    'solver': ['saga']
}

grid_search = GridSearchCV(r_model, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)


ridge_model = grid_search.best_estimator_

ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred) ** 0.5

r2 = r2_score(y_val, y_pred)
r2, rmse


log_y_val = np.log1p(y_val)
log_pred = np.log1p(y_pred)
rmsle = np.sqrt(mean_squared_error(log_y_val, log_pred))
rmsle


plt.figure(figsize=(6,5))
plt.plot((y_pred)[:20])
plt.plot(np.array((y_val)[:20]))
plt.legend(["Predicted","Actual"])
plt.xlabel('Test Data on last 20 points')
plt.show()
print('-'*150)


param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0, 1, 10, 20, 30, 40],
    'max_iter': [1000, 2000, 3000, 3500]
    }

grid_search = GridSearchCV(lr_model, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)


lasso_model = grid_search.best_estimator_
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred) ** 0.5

r2 = r2_score(y_val, y_pred)
r2, rmse


log_y_val = np.log1p(y_val)
log_pred = np.log1p(y_pred)
rmsle = np.sqrt(mean_squared_error(log_y_val, log_pred))
rmsle


plt.figure(figsize=(6,5))
plt.plot((y_pred)[:20])
plt.plot(np.array((y_val)[:20]))
plt.legend(["Predicted","Actual"])
plt.xlabel('Test Data on last 20 points')
plt.show()
print('-'*150)


X_train = X_train.drop('humidity', axis=1)
X_val = X_val.drop('humidity', axis=1)


l_model.fit(X_train, y_train)
y_pred = l_model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred) ** 0.5

r2 = r2_score(y_val, y_pred)
r2, rmse


log_y_val = np.log1p(y_val)
log_pred = np.log1p(y_pred)
rmsle = np.sqrt(mean_squared_error(log_y_val, log_pred))
rmsle


plt.figure(figsize=(6,5))
plt.plot((y_pred)[:20])
plt.plot(np.array((y_val)[:20]))
plt.legend(["Predicted","Actual"])
plt.xlabel('Test Data on last 20 points')
plt.show()
print('-'*150)


from sklearn.ensemble import RandomForestRegressor
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'criterion': ['squared_error'],
    'max_depth': [2, 5, 7, 9, 10, 11, 12],
    }

forest_model = RandomForestRegressor()

grid_search = GridSearchCV(forest_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


forest_model = grid_search.best_estimator_
forest_model.fit(X_train, y_train)
forest_model.score(X_train, y_train)


y_pred = forest_model.predict(X_val)

rmse = mean_squared_error(y_val, y_pred) ** 0.5

r2 = r2_score(y_val, y_pred)

log_y_val = np.log1p(y_val)
log_pred = np.log1p(y_pred)
rmsle = np.sqrt(mean_squared_error(log_y_val, log_pred))
r2, rmse, rmsle


plt.figure(figsize=(6,5))
plt.plot((y_pred)[:20])
plt.plot(np.array((y_val)[:20]))
plt.legend(["Predicted","Actual"])
plt.xlabel('Test Data on last 20 points')
plt.show()
print('-'*150)


df_test['datetime'] = pd.to_datetime(df_test['datetime'])
df_test['year'] = df_test['datetime'].dt.year
df_test['month'] = df_test['datetime'].dt.month
df_test['hour'] = df_test['datetime'].dt.hour

# drop the original feature.
df_test = df_test.drop(['datetime'], axis=1)

df_test.describe().T

df_test = df_test.drop(['holiday', 'atemp'], axis=1)
df_test.info()


X_test = df_test
X_test.describe().T


X_test[num_cols] = scaler.transform(X_test[num_cols])


seasons = pd.get_dummies(X_test['season'], prefix='season').astype(int)

X_test = pd.concat([X_test, seasons], axis=1)

X_test = X_test.drop('season', axis=1)
X_test = X_test.drop('humidity', axis=1)


X_test['year'] = X_test['year'].map(binary_encode)



X_test.describe().T


X_train.describe().T


y_test_pred = forest_model.predict(X_test)
y_test_pred


plt.figure(figsize=(6,5))
plt.plot((y_test_pred)[:20])
plt.legend(["Predicted"])
plt.xlabel('Test Data on last 20 points')
plt.show()
print('-'*150)

