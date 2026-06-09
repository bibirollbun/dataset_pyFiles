import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from datetime import datetime
import calendar

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_log_error as msle
#from sklearn.metrics import root_mean_squared_log_error as rmsle

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def rmsle(y_true, y_pred):
    return np.sqrt(msle(y_true, y_pred))


train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
train


# Checking out the data types of our variables.
train.info()


# Checking the number of diferent data types.
types = train.dtypes.value_counts()
types


# Plotting the number of each types of variables.
fig, ax = plt.subplots()
fig.set_size_inches(12, 5)
sns.barplot(x = types.index, y = types.values, hue = types.index, ax = ax)
ax.set(xlabel = "Variable Types", ylabel = "Counts", title = "Types of Variables by Count")
plt.show()


# Look for missing values.
train.isna().sum()


# Cheeck for duplicated samples.
print(f"Number of duplicate samples: {(train.duplicated() > 0).sum()}.")


# Number of outliers.
outliers = (np.abs(sp.stats.zscore(train.drop(columns = ["datetime"]))) > 3).sum()
outliers


# Contains the names of variables that have outliers in them.
indices = outliers[ outliers > 0 ].index


# Dropping numerical variables, "registered", and "casual".
# It does not make sense to include "casual" and "registered" to observe outliers with respect to "count".
indices = indices.drop(["casual", "registered", "count", "windspeed", "humidity"])
indices


# Plotting the variables that have outliers.
fig, axes = plt.subplots(2, 1, figsize = (20, 15))
for i, name in enumerate(indices, start = 0):
  sns.boxplot(data = train, x = f"{name}", y = "count", ax = axes[i % 2])
  #ax.set(xlabel = f"{name}", ylabel = "Count", title = f"Box-Plot of Count Grouped by {name}.")
fig.suptitle("Two Categorical Variables with Outliers Grouped by Count")
plt.show()


# Takes the "datetime" variable and returns year, month, day, and time.
def get_date(datetime):
  x = datetime.split("-")
  y = x[2].split(" ")
  time = y[1].split(":")
  return int(x[0]), int(x[1]), int(y[0]), int(time[0])


# Get the day out of the date.
#train["day"] = train["datetime"].apply(lambda day : day.split("-")[2])
#train["month"], train["day"], train["hour"]
dates = train["datetime"].apply(get_date)
train["year"], train["month"], train["day"], train["hour"] = [i[0] for i in dates], [i[1] for i in dates], [i[2] for i in dates], [i[3] for i in dates]
#train["weekend"] = ((train["holiday"] == 0) & (train["workingday"] == 0)).apply(lambda is_weekend : 1 if is_weekend else 0)
train["xhr"] = np.sin(2 * np.pi * train["hour"] / 24)
train["yhr"] = np.cos(2 * np.pi * train["hour"] / 24)
date = train["datetime"].apply(lambda x : x.split()[0])
train["weekday"] = date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
train["weekday"] = train["weekday"].map( {"Sunday" : 0, "Monday" : 1, "Tuesday" : 2, "Wednesday" : 3, "Thursday" : 4, "Friday" : 5, "Saturday" : 6} )
train = train[ [ "datetime",	"season",	"year",	"month", "day", "hour", "xhr", "yhr", "holiday", "weekday", "workingday",	"weather", "temp",	"atemp",	"humidity",	"windspeed",	"casual",	"registered",	"count"] ]
train


# Get the day out of the date for test set.
dates = test["datetime"].apply(get_date)
test["year"], test["month"], test["day"], test["hour"] = [i[0] for i in dates], [i[1] for i in dates], [i[2] for i in dates], [i[3] for i in dates]
#test["weekend"] = ((test["holiday"] == 0) & (test["workingday"] == 0)).apply(lambda is_weekend : 1 if is_weekend else 0)
test["xhr"] = np.sin(2 * np.pi * test["hour"] / 24)
test["yhr"] = np.cos(2 * np.pi * test["hour"] / 24)
date = test["datetime"].apply(lambda x : x.split()[0])
test["weekday"] = date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
test["weekday"] = test["weekday"].map( {"Sunday" : 0, "Monday" : 1, "Tuesday" : 2, "Wednesday" : 3, "Thursday" : 4, "Friday" : 5, "Saturday" : 6} )
test = test[ [ "datetime",	"season",	"year",	"month", "day", "hour", "xhr", "yhr", "holiday", "weekday", "workingday",	"weather", "temp",	"atemp",	"humidity",	"windspeed"] ]
test


# Plot all numerical variables.
sns.pairplot(train).fig.suptitle("Distribution of All Variables", y = 1)
plt.show()


# Checking the correlation between variables.
plt.figure(figsize = (12, 5))
sns.heatmap(train.drop(columns = ["datetime"]).corr(), annot = True, fmt = ".2f").figure.suptitle("Heat Map of All Variables", x = 0.5, y = 1)
plt.show()


# Checking out the number of casual users by season.
#a = pd.crosstab(hour["season"], hour["casual"])
#a


# Plot of the number of casual users grouped by season.
'''
for i in range(0, 4):
    plt.plot(a.values[i], marker='.', linestyle='none', markersize=7, label=f'Season {i + 1}')

plt.ylabel("Counts")
plt.xlabel(a.columns.names[0])
plt.legend()
plt.show()
'''


# Box-plot of casual users groupd by season.
a = train["casual"].value_counts()
plt.figure(figsize = (12, 5))
sns.boxplot(data = train, x = "season", y = "casual")
plt.title("Number of Casual Riders by Season")
plt.ylabel("Number of Casual Riders")
plt.show()


# Correlation between casual and other numerical variables.
# We ignore "cnt" and "registered".
np.abs(train.drop(columns = ["datetime"]).corr()["casual"]).sort_values(ascending = False)


# Correlation between registered and other numerical variables.
# We ignore "cnt" and "casual".
np.abs(train.drop(columns = ["datetime"]).corr()["registered"]).sort_values(ascending = False)


# Correlation between "cnt" and other numerical variables.
# We ignore "registered" and "casual".
np.abs(train.drop(columns = ["datetime"]).corr()["count"]).sort_values(ascending = False)


# One-hot encode "season", "year", "month", and "day"
encode = OneHotEncoder(sparse_output = False)
encode.fit(train[["season", "year", "month", "weekday"]])
print(encode.transform(train[["season", "year", "month", "weekday"]]))


encode.get_feature_names_out()


# Appending the one-hot-encoded variables.
train = pd.concat([train, pd.DataFrame(encode.transform(train[["season", "year", "month", "weekday"]]), columns = encode.get_feature_names_out())], axis = 1)
# We drop columns that we will not need.
train.drop(columns = ["datetime", "season", "year", "month", "day", "hour", "weekday"], axis = 1, inplace = True)
train


# We are going to make three target variables.
X = train.drop(columns = ["casual", "registered", "count"])
y_casual = np.log1p(train["casual"]) # Logarithmic transformation.
y_registered = np.log1p(train["registered"]) # Logarithmic transformation.
y_cnt = np.log1p(train["count"]) # Logarithmic transformation.


# Splitting into training set and temporary set.
X_train, X_temp, y_cas_train, y_cas_temp = train_test_split(X, y_casual, test_size = 0.2, random_state = 1)
X_train.shape, X_temp.shape, y_cas_train.shape, y_cas_temp.shape


# Splitting temporary set into validation and test sets.
X_val, X_test, y_cas_val, y_cas_test = train_test_split(X_temp, y_cas_temp, test_size = 0.5, random_state = 1)
X_val.shape, X_test.shape, y_cas_val.shape, y_cas_test.shape


# Cross-validation linear regression.
lasso = Lasso()
ridge = Ridge()
elastic = ElasticNet()
parameters = {
    "alpha": [0.0001, 0.001, 0.01, 1, 10, 100, 1000],
    "l1_ratio" : [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 0.9],
    "max_iter" : [20000]
}
lasso_params = {"alpha" : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
ridge_params = { "alpha" : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
score = list() # Apppend the train, validation, and test scores.

# Casual elastic net.
cross_validate = GridSearchCV(elastic, param_grid = parameters, cv = 28, refit = True)
cross_validate.fit(X_train, y_cas_train)

y_pred_en = cross_validate.predict(X_train)
y_pred_en[ y_pred_en < 0 ] = 0

y_pred_en_v = cross_validate.predict(X_val)
y_pred_en_v[ y_pred_en_v < 0 ] = 0

y_pred_en_te = cross_validate.predict(X_test)
y_pred_en_te[ y_pred_en_te < 0 ] = 0

score.append( {"Casual Elastic Train" : rmsle(y_cas_train, y_pred_en), "Casual Elastic Validate" : rmsle(y_cas_val, y_pred_en_v), "Casual Elastic Test" : rmsle(y_cas_test, y_pred_en_te)} )

# Casual lasso.
x_val_lasso = GridSearchCV(lasso, param_grid = lasso_params, cv = 28, refit = True)
x_val_lasso.fit(X_train, y_cas_train)

y_pred_la = x_val_lasso.predict(X_train)
y_pred_la[ y_pred_la < 0 ] = 0

y_pred_la_v = x_val_lasso.predict(X_val)
y_pred_la_v[ y_pred_la_v < 0 ] = 0

y_pred_la_te = x_val_lasso.predict(X_test)
y_pred_la_te[ y_pred_la_te < 0 ] = 0

score.append( {"Casual Lasso Train" : rmsle(y_cas_train, y_pred_la), "Casual Lasso Validate" : rmsle(y_cas_val, y_pred_la_v), "Casual Lasso Test" : rmsle(y_cas_test, y_pred_la_te)} )

# Casual ridge.
x_val_ridge = GridSearchCV(ridge, param_grid = ridge_params, cv = 28, refit = True)
x_val_ridge.fit(X_train, y_cas_train)

y_pred_ri = x_val_ridge.predict(X_train)
y_pred_ri[ y_pred_ri < 0 ] = 0

y_pred_ri_v = x_val_ridge.predict(X_val)
y_pred_ri_v[ y_pred_ri_v < 0 ] = 0

y_pred_ri_te = x_val_ridge.predict(X_test)
y_pred_ri_te[ y_pred_ri_te < 0 ] = 0

score.append( {"Casual Ridge Train" : rmsle(y_cas_train, y_pred_ri), "Casual Ridge Validate" : rmsle(y_cas_val, y_pred_ri_v), "Casual Ridge Test" : rmsle(y_cas_test, y_pred_ri_te)} )


score


# Splitting into training set and temporary set.
# Registered variable
X_train, X_temp, y_reg_train, y_reg_temp = train_test_split(X, y_registered, test_size = 0.2, random_state = 1)
X_train.shape, X_temp.shape, y_reg_train.shape, y_reg_temp.shape


# Splitting temporary set into validation and test sets.
# Registered variable.
X_val, X_test, y_reg_val, y_reg_test = train_test_split(X_temp, y_reg_temp, test_size = 0.5, random_state = 1)
X_val.shape, X_test.shape, y_reg_val.shape, y_reg_test.shape


# Registered variable.
la_reg = Lasso()
ri_reg = Ridge()
en_reg = ElasticNet()

# Elastic Net.
x_val_reg = GridSearchCV(en_reg, param_grid = parameters, cv = 28, refit = True)
x_val_reg.fit(X_train, y_reg_train)

y_pred_reg = x_val_reg.predict(X_train)
y_pred_reg[ y_pred_reg < 0 ] = 0

y_pred_reg_v = x_val_reg.predict(X_val)
y_pred_reg_v[ y_pred_reg_v < 0 ] = 0

y_pred_reg_te = x_val_reg.predict(X_test)
y_pred_reg_te[ y_pred_reg_te < 0 ] = 0

score.append( {"Registered Elastic Train" : rmsle(y_reg_train, y_pred_reg), "Registered Elastic Validate" : rmsle(y_reg_val, y_pred_reg_v), "Registered Elastic Test" : rmsle(y_reg_test, y_pred_reg_te)} )

# Lasso.
x_val_reg_la = GridSearchCV(la_reg, param_grid = lasso_params, cv = 28, refit = True)
x_val_reg_la.fit(X_train, y_reg_train)

y_pred_reg_la = x_val_reg_la.predict(X_train)
y_pred_reg_la[ y_pred_reg_la < 0 ] = 0

y_pred_reg_la_v = x_val_reg_la.predict(X_val)
y_pred_reg_la_v[ y_pred_reg_la_v < 0 ] = 0

y_pred_reg_la_te = x_val_reg_la.predict(X_test)
y_pred_reg_la_te[ y_pred_reg_la_te < 0 ] = 0

score.append( {"Registered Lasso Train" : rmsle(y_reg_train, y_pred_reg), "Registered Lasso Validate" : rmsle(y_reg_val, y_pred_reg_la_v), "Registered Lasso Test" : rmsle(y_reg_test, y_pred_reg_la_te)} )

# Ridge.
x_val_reg_ri = GridSearchCV(ri_reg, param_grid = ridge_params, cv = 28, refit = True)
x_val_reg_ri.fit(X_train, y_reg_train)

y_pred_reg_ri = x_val_reg_ri.predict(X_train)
y_pred_reg_ri[ y_pred_reg_ri < 0 ] = 0

y_pred_reg_ri_v = x_val_reg_ri.predict(X_val)
y_pred_reg_ri_v[ y_pred_reg_ri_v < 0 ] = 0

y_pred_reg_ri_te = x_val_reg_ri.predict(X_test)
y_pred_reg_ri_te[ y_pred_reg_ri_te < 0 ] = 0

score.append( {"Registered Ridge Train" : rmsle(y_reg_train, y_pred_reg), "Registered Ridge Validate" : rmsle(y_reg_val, y_pred_reg_ri_v), "Registered Ridge Test" : rmsle(y_reg_test, y_pred_reg_ri_te)} )


score


# Splitting into training set and temporary set.
# Count variable
X_train, X_temp, y_cnt_train, y_cnt_temp = train_test_split(X, y_cnt, test_size = 0.2, random_state = 1)
X_train.shape, X_temp.shape, y_cnt_train.shape, y_cnt_temp.shape


# Splitting temporary set into validation and test sets.
# Count variable.
X_val, X_test, y_cnt_val, y_cnt_test = train_test_split(X_temp, y_cnt_temp, test_size = 0.5, random_state = 1)
X_val.shape, X_test.shape, y_cnt_val.shape, y_cnt_test.shape


# Count variable.
la_cnt = Lasso()
ri_cnt = Ridge()
en_cnt = ElasticNet()

# Count Elastic Net.
x_val_cnt = GridSearchCV(en_cnt, param_grid = parameters, cv = 28, refit = True)
x_val_cnt.fit(X_train, y_cnt_train)

y_pred_cnt = x_val_cnt.predict(X_train)
y_pred_cnt[ y_pred_cnt < 0 ] = 0

y_pred_cnt_v = x_val_cnt.predict(X_val)
y_pred_cnt_v[ y_pred_cnt_v < 0 ] = 0

y_pred_cnt_te = x_val_cnt.predict(X_test)
y_pred_cnt_te[ y_pred_cnt_te < 0 ] = 0

score.append( {"Count Elastic Train" : rmsle(y_cnt_train, y_pred_cnt), "Count Elastic Validate" : rmsle(y_cnt_val, y_pred_cnt_v), "Count Elastic Test" : rmsle(y_cnt_test, y_pred_cnt_te)} )

# Count Lasso.
x_val_cnt_la = GridSearchCV(la_cnt, param_grid = lasso_params, cv = 28, refit = True)
x_val_cnt_la.fit(X_train, y_cnt_train)

y_pred_cnt_la = x_val_cnt_la.predict(X_train)
y_pred_cnt_la[ y_pred_cnt_la < 0 ] = 0

y_pred_cnt_la_v = x_val_cnt_la.predict(X_val)
y_pred_cnt_la_v[ y_pred_cnt_la_v < 0 ] = 0

y_pred_cnt_la_te = x_val_cnt_la.predict(X_test)
y_pred_cnt_la_te[ y_pred_cnt_la_te < 0 ] = 0

score.append( {"Count Lasso Train" : rmsle(y_cnt_train, y_pred_cnt_la), "Count Lasso Validate" : rmsle(y_cnt_val, y_pred_cnt_la_v), "Count Lasso Test" : rmsle(y_cnt_test, y_pred_cnt_la_te)} )

# Count Ridge.
x_val_cnt_ri = GridSearchCV(ri_cnt, param_grid = ridge_params, cv = 28, refit = True)
x_val_cnt_ri.fit(X_train, y_cnt_train)

y_pred_cnt_ri = x_val_cnt_ri.predict(X_train)
y_pred_cnt_ri[ y_pred_cnt_ri < 0 ] = 0

y_pred_cnt_ri_v = x_val_cnt_ri.predict(X_val)
y_pred_cnt_ri_v[ y_pred_cnt_ri_v < 0 ] = 0

y_pred_cnt_ri_te = x_val_cnt_ri.predict(X_test)
y_pred_cnt_ri_te[ y_pred_cnt_ri_te < 0 ] = 0

score.append( {"Count Ridge Train" : rmsle(y_cnt_train, y_pred_cnt_ri), "Count Ridge Validate" : rmsle(y_cnt_val, y_pred_cnt_ri_v), "Count Ridge Test" : rmsle(y_cnt_test, y_pred_cnt_ri_te)} )


score


# Checking out the coefficients estimated by the three best linear models on "casual", "registered", and "count".
plt.figure(figsize = (15, 3))
plt.bar(X.columns, x_val_cnt_ri.best_estimator_.coef_)
plt.title("Ridge Regularization on 'Count'")
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize = (15, 3))
plt.bar(X.columns, x_val_reg_la.best_estimator_.coef_)
plt.title("Lasso Regularization on 'Registered'")
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize = (15, 3))
plt.bar(X.columns, cross_validate.best_estimator_.coef_)
plt.title("Elastic Net Regularization on 'Casual'")
plt.xticks(rotation = 45)
plt.show()


test


# One-hot encode "season", "year", "month", and "day"
encode_test = OneHotEncoder(sparse_output = False)
encode_test.fit(test[["season", "year", "month", "weekday"]])
print(encode_test.transform(test[["season", "year", "month", "weekday"]]))


# Appending the one-hot-encoded variables to test set.
test = pd.concat([test, pd.DataFrame(encode_test.transform(test[["season", "year", "month", "weekday"]]), columns = encode_test.get_feature_names_out())], axis = 1)
# We drop columns that we will not need.
test.drop(columns = ["datetime", "season", "year", "month", "day", "hour", "weekday"], axis = 1, inplace = True)
test


# Predicting "count" for the test set.
out = pd.DataFrame( columns = ["count"])
out["count"] = x_val_cnt_ri.predict(test)
out


# Writing a .csv file with the predicted test "count" values.
out.to_csv("bike_predictions_linear_regression_ridge_logarithmically_normalized.csv", index = False)


real_out = (np.rint(np.expm1(out))).astype(int)
real_out


# Writing a .csv file with the predicted test "count" values.
real_out.to_csv("bike_predictions_linear_regression_ridge.csv", index = False)


poly = PolynomialFeatures(degree = 2, include_bias = False) # include_bias 0차항 출력 여부

poly.fit(X_train)

X_poly = pd.DataFrame(poly.transform(X_train), columns=poly.get_feature_names_out())


X_poly


lr_poly = LinearRegression()

lr_poly.fit(X_poly, y_cnt_train)


poly_pred = lr_poly.predict(X_poly)


rmsle(y_cnt_train, poly_pred)


from sklearn.pipeline import Pipeline
model = Pipeline([
    ('polynomial_features', PolynomialFeatures(degree = 3)),
    ('standard_scaler', StandardScaler()),
    ('ridge_regression', Ridge())
])

# Train the model
model.fit(X_train, y_cnt_train)

# Predict and evaluate the model
y_pred = model.predict(X_train)
rmsle(y_cnt_train, y_pred)


asd = model.predict(X_val)
asd


rmsle(y_cnt_val, asd)


zxc = model.predict(X_test)
zxc


rmsle(y_cnt_test, zxc)

