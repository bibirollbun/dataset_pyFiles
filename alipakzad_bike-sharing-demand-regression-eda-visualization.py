import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# turn off warnings for final notebook
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline


df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")


df.head()


df.shape


df.columns


df.info()


df["datetime"]= pd.to_datetime(df["datetime"])


df_clean = df.copy()

df_clean['year'] = df_clean.datetime.apply(lambda x: x.year)

df_clean['month'] = df_clean.datetime.apply(lambda x: x.month)

df_clean['weekday'] = df_clean.datetime.apply(lambda x: x.weekday())

#0: Monday 1:Tuesday 2:Wednesday
# 3:Thursday 4:Friday 5:Saturday 6:Sunday

df_clean['hour'] = df_clean.datetime.apply(lambda x: x.hour)


df_clean.tail()


df_clean.duplicated().sum()


print(df_clean.isna().sum())


df_clean.describe()


df.workingday.value_counts()


df.season.value_counts()


df.weather.value_counts()


#sns.heatmap(df.corr(), annot=True,cmap='RdYlGn')

fig = plt.figure(figsize=(20,20))
sns.heatmap(df_clean.corr().round(3), annot=True, cmap='RdYlGn')
plt.show()



#Total Bike Rentals Per Season
count_season = df.groupby('season')['count'].sum()


#plt.figure(figsize=(8,6))

chart = sns.barplot(x=count_season.index , y= count_season.values, estimator=sum, ci=None)

plt.title('Number of Total Rental Bikes Per Season', fontsize=14)

chart.set_xticklabels(['spring', 'summer', 'fall', 'winter'])

plt.show()


count_workingday = df.groupby('workingday')['count'].sum()


chart = sns.barplot(x=count_workingday.index , y= count_workingday.values, estimator=sum, ci=None)

plt.title('Number of Total Rental Bikes According to Working Day', fontsize=14)

plt.show()


count_holiday = df.groupby('holiday')['count'].sum()


chart = sns.barplot(x=count_holiday.index , y= count_holiday.values, estimator=sum, ci=None)

plt.title('Number of Total Bikes Rented on holiday VS not holiday', fontsize=14)

plt.show()


count_weather = df.groupby('weather')['count'].sum()


chart = sns.barplot(x=count_weather.index , y= count_weather.values, estimator=sum, ci=None)

plt.title('Number of Total Rental Bikes Per Weather Conditions', fontsize=14)

chart.set_xticklabels(['Clear', 'Mist' ,'Light Snow/Rain', 'Heavy Rain'])

plt.show()


sns.histplot(df_clean['temp'])  
plt.show()


sns.histplot(df_clean['humidity'])  
plt.show()


sns.histplot(df_clean['windspeed'])  
plt.show()


sns.regplot(data = df_clean
                ,x = 'temp'
                ,y = 'count'
                ,line_kws={"color": "red"}
                ,scatter_kws ={'alpha':0.2}
           )
plt.show()


sns.regplot(data = df_clean
                ,x = 'humidity'
                ,y = 'count'
                ,line_kws={"color": "red"}
                ,scatter_kws ={'alpha':0.3}
           )
plt.show()


plt.figure(figsize=(12,6))

sns.lineplot(data = df_clean
                ,x = 'hour'
                ,y = 'count'
                ,ci = None
           )

plt.title('The Number of Rented Bikes According to Hour of the Day')

plt.xticks(ticks=range(24), labels=range(24))
#plt.axvline(x=17, color ='red')

plt.show()


plt.figure(figsize=(12,6))

plt.title('The Number of Rented Bikes According to Hour of the Day')

sns.lineplot(data = df_clean
                ,x = 'hour'
                ,y = 'count'
                ,hue ='season'
                ,palette= ["Red", "Gold", "Blue", "Green"]
                ,ci = None
           )

plt.xticks(ticks=range(24), labels=range(24))

plt.legend(["Spring", "Summer", "Fall", "Winter"])

plt.show()


plt.figure(figsize=(12,6))

plt.title('The Number of Rented Bikes According to Hour of the Day for Different Weathers')

sns.lineplot(data = df_clean
                ,x = 'hour'
                ,y = 'count'
                ,hue ='weather'
                ,palette= ["Red", "Gold", "Blue", "Green"]
                ,ci = None
           )

plt.xticks(ticks=range(24), labels=range(24))

plt.legend(['Clear', 'Mist' ,'Light Snow/Rain', 'Heavy Rain'])

plt.show()


plt.figure(figsize=(12,6))

sns.lineplot(data = df_clean
                ,x = 'weekday'
                ,y = 'count'
                ,ci = None
             ,color ='DarkOrange'
           )

plt.title('The Number of Rented Bikes According to Weekday')

plt.xticks(ticks=range(7), labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
#plt.axvline(x=4, color ='red')

plt.show()


plt.figure(figsize=(12,6))

sns.lineplot(data = df_clean
                ,x = 'month'
                ,y = 'count'
                ,ci = None
             ,color ='Red'
           )

plt.title('The Number of Rented Bikes According to month')

plt.xticks(ticks=range(1,13), labels=['January', 'February', 'March', 'April', 'May', 'June', \
                                            'July', 'August', 'September', 'October', 'November', 'December'])
#plt.axvline(x=6, color ='blue')

plt.show()


plt.figure(figsize=(12,6))

plt.title('The Number of Bikes Rented by Casual users According to Hour of the Day')

sns.lineplot(data = df_clean
                ,x = 'hour'
                ,y = 'casual'
                ,hue ='season'
                ,palette= ["Red", "Gold", "Blue", "Green"]
                ,ci = None
           )

plt.xticks(ticks=range(24), labels=range(24))

plt.legend(["Spring", "Summer", "Fall", "Winter"])
#plt.axvline(x=15, color ='blue')

plt.show()


s = df_clean[["casual", "registered"]].sum()

chart = sns.barplot(x=s.index , y= s.values, estimator=sum, ci=None)

plt.title('Number of Casual VS Registered Users Rentals Initiated', fontsize=14)

plt.show()


plt.figure(figsize=(8,6))

plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)


plt.subplot(1, 2, 1)
sns.regplot(data = df_clean
                ,x = 'temp'
                ,y = 'casual'
                ,line_kws={"color": "red"}
                ,scatter_kws ={'alpha':0.1}
           )

plt.subplot(1, 2, 2)
sns.regplot(data = df_clean
                ,x = 'temp'
                ,y = 'registered'
                ,line_kws={"color": "red"}
                ,scatter_kws ={'alpha':0.1}
           )
plt.show()


plt.figure(figsize=(8,6))

plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)


plt.subplot(1, 2, 1)
sns.regplot(data = df_clean
                ,x = 'humidity'
                ,y = 'casual'
                ,line_kws={"color": "red"}
                ,scatter_kws ={'alpha':0.1}
           )

plt.subplot(1, 2, 2)
sns.regplot(data = df_clean
                ,x = 'humidity'
                ,y = 'registered'
                ,line_kws={"color": "red"}
                ,scatter_kws ={'alpha':0.1}
           )
plt.show()


season_codes ={1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'}

weather_codes = {1: 'Clear', 2: 'Mist', 3: 'Light_Snow', 4: 'Heavy_Rain'}

#month_codes = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June',
#                                            7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}

df_clean['season'] = df_clean['season'].map(season_codes)

df_clean['weather'] = df_clean['weather'].map(weather_codes)

#df_clean['month'] = df_clean['month'].map(month_codes)


df_clean.sample(5)


df_clean = pd.get_dummies(df_clean, columns= ['season', 'weather'], drop_first =True)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#Y is the target column, X has the features
X = df_clean[['temp']]
y = df_clean['count']


#Split the data into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

#Scaler = StandardScaler() MinMaxScaler()

Scaler = StandardScaler()

X_train  = Scaler.fit_transform(X_train)
X_test  = Scaler.transform(X_test)


lr = LinearRegression()
lr.fit(X_train, y_train)

# make predictions
y_pred = lr.predict(X_test)

# evaluate predictions
acc = r2_score(y_test, y_pred)
#acc = mlr.score(X_test, y_test)
print('R-squared: %.3f' % acc)

mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

mse = mean_squared_error(y_test, y_pred)
print('MSE: %.3f' % mse)

rmse =mse**(0.5)

print('RMSE: %.3f' % rmse)


selected_features = ['season_spring', 'season_summer', 'season_winter', 'weather_Heavy_Rain', 'weather_Light_Snow', 
                     'weather_Mist', 'workingday', 'temp',  'humidity', 'year', 'hour']

#Y is the target column, X has the features
X = df_clean[selected_features]
y = df_clean['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

Scaler = StandardScaler()

X_train  = Scaler.fit_transform(X_train)
X_test  = Scaler.transform(X_test)


mlr = LinearRegression()
mlr.fit(X_train, y_train)

# make predictions
y_pred = mlr.predict(X_test)

# evaluate predictions
acc = r2_score(y_test, y_pred)
#acc = mlr.score(X_test, y_test)
print('R-squared: %.3f' % acc)

mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

mse = mean_squared_error(y_test, y_pred)
print('MSE: %.3f' % mse)

rmse =mse**(0.5)

print('RMSE: %.3f' % rmse)


print(f"Our Regression Coefficients: {mlr.coef_}")
print(f"Our intercept is: {mlr.intercept_}")


Coefficients = pd.DataFrame([X.columns, mlr.coef_]).T

Coefficients = Coefficients.rename(columns ={0:'Attributes', 1:'coef'})

Coefficients


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree = 4)
X_train_poly = poly_features.fit_transform(X_train)

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)

X_test_poly = poly_features.transform(X_test)
y_pred = lin_reg.predict(X_test_poly)

# evaluate predictions
acc = r2_score(y_test, y_pred)

#acc = mlr2.score(X_test, y_test)
print('R-squared: %.3f' % acc)

mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

mse = mean_squared_error(y_test, y_pred)
print('MSE: %.3f' % mse)

rmse =mse**(0.5)

print('RMSE: %.3f' % rmse)


from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=3)

knn_model.fit(X_train, y_train)


y_pred = knn_model.predict(X_test)

# evaluate predictions
acc = r2_score(y_test, y_pred)

print('R-squared: %.3f' % acc)

mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

mse = mean_squared_error(y_test, y_pred)
print('MSE: %.3f' % mse)

rmse =mse**(0.5)
#rmse = sqrt(mse)

print('RMSE: %.3f' % rmse)


# import the regressor
from sklearn.tree import DecisionTreeRegressor 
  
# create a regressor object
DTmodel = DecisionTreeRegressor(random_state = 42) 
  
# fit the regressor with X and Y data
DTmodel.fit(X_train, y_train)

y_pred = DTmodel.predict(X_test)

# evaluate predictions
acc = r2_score(y_test, y_pred)

print('R-squared: %.3f' % acc)

mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

mse = mean_squared_error(y_test, y_pred)
print('MSE: %.3f' % mse)

rmse =mse**(0.5)
#rmse = sqrt(mse)

print('RMSE: %.3f' % rmse)


# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor
 
# create regressor object
RF_model = RandomForestRegressor(n_estimators=100,
                                  random_state=0)
 
# fit the regressor with x and y data
RF_model.fit(X_train, y_train)

y_pred = RF_model.predict(X_test)

# evaluate predictions
acc = r2_score(y_test, y_pred)

print('R-squared: %.3f' % acc)

mae = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' % mae)

mse = mean_squared_error(y_test, y_pred)
print('MSE: %.3f' % mse)

rmse =mse**(0.5)
#rmse = sqrt(mse)

print('RMSE: %.3f' % rmse)


test_df = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

test_df.head()


test_df["dtime"]= pd.to_datetime(test_df["datetime"])

test_df['year'] = test_df.dtime.apply(lambda x: x.year)

test_df['month'] = test_df.dtime.apply(lambda x: x.month)

test_df['weekday'] = test_df.dtime.apply(lambda x: x.weekday())

#0: Monday 1:Tuesday 2:Wednesday
# 3:Thursday 4:Friday 5:Saturday 6:Sunday

test_df['hour'] = test_df.dtime.apply(lambda x: x.hour)


season_codes ={1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'}

weather_codes = {1: 'Clear', 2: 'Mist', 3: 'Light_Snow', 4: 'Heavy_Rain'}

test_df['season'] = test_df['season'].map(season_codes)

test_df['weather'] = test_df['weather'].map(weather_codes)


test_df = pd.get_dummies(test_df, columns= ['season', 'weather'], drop_first =True)


selected_features = ['season_spring', 'season_summer', 'season_winter', 'weather_Heavy_Rain', 'weather_Light_Snow', 
                     'weather_Mist', 'workingday', 'temp',  'humidity', 'year', 'hour']


final_test_df = test_df[selected_features]

final_test_df.index = test_df.datetime

final_test_df

#scale the test dataframe
final_test_df_scaled  = Scaler.transform(final_test_df)

final_test_df


test_pred = RF_model.predict(final_test_df_scaled)


AnswerDF = pd.DataFrame({
    'datetime': test_df['datetime'], 
    'count': test_pred
})

AnswerDF


# Checking if submission rows is equal to test set rows
print("Test rows:", len(test_df))
print("Submission rows:", len(AnswerDF))

# checking if dates of test set is identical to submission dates
print("Missing datetimes:", set(test_df['datetime']) - set(AnswerDF['datetime']))


# saving submission file
AnswerDF.to_csv('submission.csv', index=False)

