# Basic libraries
import numpy as np
import pandas as pd
import os

# data visualizaion libraries
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# model training libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error , mean_squared_error, r2_score


print("Libraries are imported!!!")


filePath_train = "/kaggle/input/playground-series-s5e1/train.csv"
filePath_test = "/kaggle/input/playground-series-s5e1/test.csv"
filePath_sample_submission = "/kaggle/input/playground-series-s5e1/sample_submission.csv"


train_df = pd.read_csv(filePath_train)
test_df = pd.read_csv(filePath_test)
sample_submission_df = pd.read_csv(filePath_sample_submission)
print("DataFrame created!")


print("Top 5 data of train_df:")
print(train_df.head(5))
print("\n\nTop 5 data of test_df:")
print(test_df.head(5))


train_df.info()


columnsName = list(train_df.columns)
print(columnsName)


print("Unique items:\n")
train_df.nunique()


print("Emptry Fields:")
train_df.isnull().sum()


train_df.describe()


train_df.describe(include='object')


# Examining the Distribution of Sales (num_sold)
plt.figure(figsize=(10,6))
sns.histplot(train_df['num_sold'], bins=60, kde=True)
plt.title("Distribution of Stricker Sales")
plt.xlabel("Number of Stricker Sold")
plt.ylabel("Frequency")
plt.show()


# Ploting num_sold over time to observe trends.
plt.figure(figsize=(12,6))
train_df.groupby('date')['num_sold'].sum().plot()
plt.title('Total Stricker Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Stricker Sold')
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(x='country', y='num_sold', data=train_df)
plt.title('Stricker Sales Distribution by Coutnry')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='store', y='num_sold', data=train_df)
plt.title('Stricker Sales Distribution by Store')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='product', y='num_sold', data=train_df)
plt.title('Stricker Sales Distribution by Product')
plt.show()


def featureEngineering(data):
    data['date'] = pd.to_datetime(data['date'])
    data['day_of_week'] = data['date'].dt.dayofweek
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

    # Handling missing values
    if 'num_sold' in list(data.columns):
        data['num_sold'] = data.groupby(['country', 'store', 'product'])['num_sold'].transform(
            lambda x: x.fillna(x.median())
        )
        overall_median = data['num_sold'].median()
        data['num_sold'].fillna(overall_median, inplace=True)

    data.fillna(0.1, inplace=True)

    # Hotencoding
    data = pd.get_dummies(data, columns=['country', 'store', 'product'])

    return data
    


train_df = featureEngineering(train_df)
test_df = featureEngineering(test_df)


plt.figure(figsize=(10,6))
sns.boxplot(x='day_of_week', y='num_sold', data=train_df)
plt.title('Stricker Sales Distribution by Day of the Week')
plt.show()


print("Top 5 data of train_df:")
train_df.head(5)


print("\n\nTop 5 data of test_df:")
test_df.head(5)


train_df.isnull().sum()


train_set, val_set = train_test_split(train_df, test_size=0.2)

X_train = train_set.drop(['id', 'date', 'num_sold'], axis=1)
y_train = train_set['num_sold']
X_val = val_set.drop(['id', 'date', 'num_sold'], axis=1)
y_val = val_set['num_sold']


model = RandomForestRegressor(n_estimators=100, random_state=42)


model.fit(X_train, y_train)


def printError(val, pred):
    mse = mean_squared_error(val, pred)
    r2 = r2_score(val, pred)
    mape = mean_absolute_percentage_error(val, pred)
    
    print(f'MAPE on validation set: {mape:}')
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")


y_pred = model.predict(X_val)
printError(y_val, y_pred)


# # Define parameter grid
# param_grid = {
#     'n_estimators': [100, 150, 200, 250],
#     'max_depth': [None, 10, 20, 30],
#     # 'min_samples_split': [2, 5, 10],
# }

# # GridSearchCV
# grid_search = GridSearchCV(
#     estimator=RandomForestRegressor(random_state=42),
#     param_grid=param_grid,
#     scoring='neg_mean_squared_error',
#     cv=5,
#     verbose=3,
#     n_jobs=-1
# )
# grid_search.fit(X_train, y_train)

# print(f"Best Parameters: {grid_search.best_params_}")



# best_model = grid_search.best_estimator_
# print(f"Best Parameters: {grid_search.best_params_}")


# y_pred = best_model.predict(X_val)
# printError(y_val, y_pred)


test_df.isnull().sum()


X_test = test_df.drop(['id', 'date'], axis=1)
predictions = model.predict(X_test)


submission = pd.DataFrame({'id': test_df['id'], 'num_sold': predictions})
submission.to_csv('submission.csv', index=False)

