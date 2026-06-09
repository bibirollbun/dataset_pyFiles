# import library
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


#read the csv file
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")


df.head()


# chcek total dataframe infromation
df.info()


# Check the total statistics dataframe
df.describe()


# chceck duplicated data
df.duplicated().sum()


#check the Nan value
df.isnull().sum()


# convert datetime data type
df.datetime = pd.to_datetime(df.datetime)


# show datetime describe
df.describe(include = "datetime64[ns]")


# check oulier datetime
df["datetime"].describe()


# set the numeric columns
num_cols = df._get_numeric_data().columns.tolist()
len(num_cols)
# set size of subplots
fig, axes = plt.subplots(nrows = 3, ncols = 4,figsize = (15,10))

# draw each column boxpolt
for i,col in enumerate(num_cols):
    row , col_idx = divmod(i,4)
    sns.boxplot(data =df, x = col, ax = axes[row,col_idx])


# see humidity and windspeed oulier

def see_outlier(df,col):
    # copy the df
    dff = df
    # set q1 and q3 and iqr
    q1 = dff[col].quantile(0.25)
    q3 = dff[col].quantile(0.75)
    iqr = q3 - q1

    # filter outlier
    condition = (df[col] < q1 - iqr * 1.5) | (df[col] > q3 + iqr * 1.5)
    return df[condition][col]

# filter outlier
outlier_hdf = see_outlier(df,"humidity")
outlier_wdf = see_outlier(df,"windspeed")


#filter outlier_hdf
len(np.array(outlier_hdf))


#check the humidity zero data
df[df.humidity == 0]


# show windspeed outlier data
outlier_wdf


# Convert the unit from km/h to m/s and determine the maximum value to check if it is reasonable.
# becasue paper say the windspeed measure km/h
np.array(outlier_wdf / 3.6).max()


#show humidity value zero data
df[df.humidity == 0].head()


#remove humidity zero data
df.drop(df[df.humidity == 0].index, inplace=True)


# copy the dateframe
dff = df.copy()


#The 'count' column is the result of adding the 'casual' column and the 'registered' column. So, check if it matches the count and verify the sum column
dff['check_sum'] = dff['casual'] + dff['registered']
result = dff[dff['check_sum'] != dff['count']]
result


df["month"] = df.datetime.dt.month


df["year"] = df.datetime.dt.year


# use datetime column and convert datetime to hour
df["hour"] = df.datetime.dt.hour
# rearange the colums position
cols = [col for col in df.columns if col != "count"] + ["count"]
df = df[cols]


df.hist(figsize = (15,10))


# show the year distribution how many customer use it
# The business started in 2011, so its usage is relatively lower than in 2012
year_df =df.groupby(df.datetime.dt.year)["count"].sum().to_frame().reset_index()
sns.barplot(data = year_df, x = "datetime", y = "count")
plt.xlabel("year")
plt.show()


# show the month distribution how many customer use it
month_df =df.groupby(df.datetime.dt.month)["count"].sum().to_frame().reset_index()
sns.barplot(data = month_df, x = "datetime", y = "count")
plt.xlabel("month")
plt.show()


# show the days distribution how many customer use it
days_df =df.groupby(df.datetime.dt.day)["count"].sum().to_frame().reset_index()
sns.barplot(data = days_df, x = "datetime", y = "count")
plt.xlabel("days")
plt.show()


# get numeric data
num_data = df._get_numeric_data()
# set figure size
plt.figure(figsize = (15,10))
# show heatmap
sns.heatmap(data = num_data.corr(), annot = True, cmap = "coolwarm")


#show weather frequency graph
sns.barplot(data = df.groupby(["weather"])["count"].sum().to_frame().reset_index(), x = "weather", y = 'count')


 df.groupby(["season"])["count"].sum()


#show weather frequency graph
sns.barplot(data = df.groupby(["season"])["count"].sum().to_frame().reset_index(), x = "season", y = 'count')


#show the workingday frequency
sns.barplot(data = df.groupby(["workingday"])["count"].sum().to_frame().reset_index(), x = "workingday", y = 'count')


#show holiday frequency
sns.barplot(data = df.groupby(["holiday"])["count"].sum().to_frame().reset_index(), x = "holiday", y = 'count')


sns.barplot(data = df.groupby(["hour"])["count"].sum().to_frame().reset_index(), x = "hour", y = 'count')


# year was operating year and month be able to replace season data so I drop the datetime column
X = df.drop(["datetime", "count","casual","registered"], axis=1)
y = df["count"]



X_temp = pd.get_dummies(data =X,columns = ['year','month','hour'],dtype = int)


from sklearn.model_selection import train_test_split

X_train , X_val, y_train, y_val = train_test_split(X_temp,y,test_size = 0.2, shuffle = True, random_state =42)


from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_log_error

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=True)


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_log_error



parm_linear_grid1 = {
    'fit_intercept': [True, False]
}

lr_model = LinearRegression()

n_split_list = [2,3,4,5,6,7,8,9,10]


for n_split in n_split_list:
    search_grid1 = GridSearchCV(
            estimator= lr_model,
            param_grid=parm_linear_grid1,
            cv=n_split,
            refit=True,
            scoring=rmsle_scorer,  # 예: MSlE 지표 사용
            verbose=0,
            n_jobs=-1
        )
    search_grid1.fit(X_train, y_train)

    print(f"n_split : {n_split}, params: {search_grid1.best_params_}, score: {search_grid1.best_score_}")




gird1_pred = search_grid1.predict(X_val)

# 산점도
plt.figure(figsize=(8, 6))
plt.scatter(y_val, gird1_pred, alpha=0.6, edgecolor='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_val_pred)")
plt.title("Actual vs Predicted Values : Grid1")
plt.show()



from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import PolynomialFeatures


parm_linear_grid2 = {
    'fit_intercept': [True, False]
}

model = LinearRegression()

n_split_list = [4,6,7]

poly_features = PolynomialFeatures(degree=2)

for n_split in n_split_list:
    search_grid2 = GridSearchCV(
            estimator= model,
            param_grid=parm_linear_grid2,
            cv=n_split,
            refit=True,
            scoring=rmsle_scorer,  # 예: MSlE 지표 사용
            verbose=0,
            n_jobs=-1
        )
    search_grid2.fit(poly_features.fit_transform(X_train), y_train)

    print(f"n_split : {n_split}, params: {search_grid2.best_params_}, score: {search_grid2.best_score_}")




gird2_pred = search_grid2.predict(poly_features.transform(X_val))
# scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val, gird2_pred, alpha=0.6, edgecolor='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_val_pred)")
plt.title("Actual vs Predicted Values : Grid2")
plt.show()



from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer
from sklearn.preprocessing import PolynomialFeatures

param_grid = {
    'regressor__fit_intercept': [True, False]
}

poly_features = PolynomialFeatures(degree=2)

tt = TransformedTargetRegressor(regressor=LinearRegression(),
                                func=np.log, inverse_func=np.exp)
n_split_list = [4,6,7]

for n_split in n_split_list:
  grid_search3 = GridSearchCV(estimator=tt, param_grid=param_grid, cv=n_split, scoring=rmsle_scorer, refit=True, verbose=0, n_jobs=-1)
  grid_search3.fit(poly_features.fit_transform(X_train), y_train)
  print(f"n_split: {n_split}, best_params: {grid_search3.best_params_}, best_score: {grid_search3.best_score_}")


gird3_pred = grid_search3.predict(poly_features.transform(X_val))
# scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val, gird3_pred, alpha=0.6, edgecolor='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_val_pred)")
plt.title("Actual vs Predicted Values : Grid3")
plt.show()



from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import make_scorer
from sklearn.compose import TransformedTargetRegressor
import numpy as np

def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    log_diff = np.log1p(y_true) - np.log1p(y_pred)
    return np.sqrt(np.mean(np.square(log_diff)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=True)


poly_features = PolynomialFeatures(degree=2)

lf = LinearRegression()
rfr = RandomForestRegressor()
svr = SVR()

vo_reg = VotingRegressor(estimators=[
    ('lf', lf),
    ('rfr', rfr),
    ('svr', svr)
])

tt = TransformedTargetRegressor(regressor=vo_reg,
                                func=np.log, inverse_func=np.exp)


tt.fit(X_train, y_train)
vo_pred = tt.predict(X_val)
rmsle_value = rmsle(y_val, vo_pred)
print(f"RMSLE: {rmsle_value:.4f}")







# scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val, vo_pred, alpha=0.6, edgecolor='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_val_pred)")
plt.title("Actual vs Predicted Values : Grid3")
plt.show()



import numpy as np
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor


y_train_log = np.log1p(y_train)


xg_model = XGBRegressor(n_estimators=200, learning_rate=0.02, random_state=42)
xg_model.fit(X_train, y_train_log)


y_val_pred_log = xg_model.predict(X_val)

# 로그 변환된 예측값을 원래 값으로 복원
xg_val_pred = np.expm1(y_val_pred_log)


rmsle_value = rmsle(y_val, xg_val_pred)

print(f"XGBRegressor RMSLE: {rmsle_value:.4f}")



# scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val, xg_val_pred , alpha=0.6, edgecolor='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_val_pred)")
plt.title("Actual vs Predicted Values : XGBOOST")
plt.show()



from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor(n_estimators=300,max_depth=50,random_state=42)

lgbm_model.fit(X_train, y_train_log)

y_val_pred_log = lgbm_model.predict(X_val)
lg_val_pred = np.expm1(y_val_pred_log)


rmsle = rmsle(y_val, lg_val_pred)

print(f"LightGBM Validation RMSLE: {rmsle:}")


# scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_val, lg_val_pred , alpha=0.6, edgecolor='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
plt.xlabel("Actual Values (y_val)")
plt.ylabel("Predicted Values (y_val_pred)")
plt.title("Actual vs Predicted Values : LGBM")
plt.show()



best_model = grid_search3.best_estimator_
coefficients = best_model.regressor_.coef_

len(coefficients)


import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Save the names of features before polynomial transformation
original_feature_names = X_temp.columns  # Feature names from the original dataset before transformation

# Create a PolynomialFeatures object and fit it to the data
poly_features = PolynomialFeatures(degree=2)
poly_features.fit(X_temp)  # Fit the PolynomialFeatures object to the X_temp data

# Generate new feature names after polynomial transformation
poly_feature_names = poly_features.get_feature_names_out(input_features=original_feature_names)

# Retrieve the best model from GridSearchCV
best_model = grid_search3.best_estimator_
regressor = best_model.regressor_  # Optimized LinearRegression model

# Get the coefficients of the regression model
coefficients = regressor.coef_

# Organize feature importance into a DataFrame
importance_df = pd.DataFrame({
    "Feature": poly_feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

# Extract the top 10 features with the highest coefficients
top_10_features = importance_df.head(10)

# Print the top 10 features with the highest coefficients
print("Top 10 features with the highest coefficients:")
print(top_10_features)



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(top_10_features["Feature"], top_10_features["Coefficient"], color='skyblue')
plt.xlabel("Coefficient", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.title("Top 10 Feature Importances", fontsize=16)
plt.gca().invert_yaxis()  
plt.show()


