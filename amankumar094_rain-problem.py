# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
train.info(),test.info()


column_of_interest = [
    "pressure", "maxtemp", "temparature", "mintemp",
    "dewpoint", "humidity", "cloud", "sunshine",
    "winddirection", "windspeed", "rainfall"
]

correlation_matrix = train[column_of_interest].corr(method="spearman")
sns.set_theme(style="white")
plt.figure(figsize=(10,8))
heatmap = sns.heatmap(correlation_matrix,annot=True ,fmt=".2f",
                     cmap="coolwarm",cbar_kws={"label":"Spearman Corrleation"})
heatmap.set_title("Correlation Heatmap")
plt.show()

#Darker colors (depending on your cmap) indicate stronger correlations.
#Positive correlations are often shown in warm colors (reds/oranges).
#Negative correlations are usually displayed in cooler colors (blues/purples).
#Values close to 0 appear as neutral colors, suggesting no correlation.


#Positive Correlation: If one feature increases, the other feature also increases.
#Example: Height and weight. Taller people usually tend to weigh more, so there's a positive correlation.
#Correlation value: Close to +1.

#Negative Correlation: If one feature increases, the other feature decreases.
#Example: Exercise time and body fat percentage. The more time spent exercising, the lower the body fat percentage tends to be.
#Correlation value: Close to -1.

#No Correlation: If there's no relationship between features.
#Example: Shoe size and exam scores have no meaningful connection.
#Correlation value: Close to 0.

#So i can conclude that 
# Cloud and humidty are strongly correlated to rainfall to each other 
#


# column_of_interest = column_of_interest.T

# # Plotting the heatmap
# plt.figure(figsize=(10, 8))
# heatmap = sns.heatmap(column_of_interest, annot=True, fmt=".1f", cbar=True, square=False, cmap="coolwarm")

# # Adding title for clarity
# plt.title("Heatmap of Weather Variables")
# plt.show()


#Calculate the difference between maximum and miimum temperature
train['temp_diff'] = train['maxtemp'] - train['mintemp']
test['temp_diff'] = test['maxtemp'] - test['mintemp']


#Convert day into month and season
train['date'] = pd.to_datetime(train['day'])
train['month'] = train['date'].dt.month
test['date'] = pd.to_datetime(test['day'])
test['month'] = test['date'].dt.month



train.info(),
test.info()



# #We will create feature to see if there has been rainfall from past 2 days or not
# train['rainfall_pre1'] = train['rainfall'].shift(1)# for previous day
# train['rainfall_pre2'] = train['rainfall'].shift(2)# For past 2 days
# test['rainfall_pre1'] = test['rainfall'].shift(1)# for previous day
# test['rainfall_pre2'] = test['rainfall'].shift(2)


#Now we will se if there has been rainfall for past 7 days and we will sum it.
#Why we do it: It help us find the trendover a defined period of time.
# train['rainfall_7day_sum'] = train['rainfall'].rolling(window=7).sum()
# test['rainfall_7day_sum'] = test['rainfall'].rolling(window=7).sum()


#Now we will calculate the interactive feature between Humidity and cloud
#Why?:The interaction between these two can help capture non-linear relationships,
#Including interaction terms provides the model with additional context, enabling it to learn relationships that individual features can't explain alone

train['humididty_cloud_ineraction'] = train['humidity'] * train['cloud']
test['humididty_cloud_ineraction'] = test['humidity'] * test['cloud']


#Now we will calculate Vapor Pressure Deficit (Advanced Feature)
#why?:It is an important feature that give us insights into how much moisture the air can hold and how close the sir is to being saturated with water vapor.
#A low VPD means the air is nearly saturated and rain is more likely to occur.
#A high VPD means the air is dry, making rainfall less likely.

#Saturation Vapor Pressure (SVP): The maximum amount of moisture the air can hold at a given temperature (air is fully saturated here).
#Actual Vapor Pressure (AVP): The amount of moisture currently in the air, determined by the dew point temperature.

#VPD is calculated as: VPD = SVP - AVP

train['svp'] = 0.6108 * np.exp((17.27 * train['temparature'])/(train['temparature'] + 273.3))
train['avp'] = 0.6108 * np.exp((17.27 * train['dewpoint']) / (train['dewpoint'] + 237.3))
train['vpd'] = train['svp'] - train['avp']
test['svp'] = 0.6108 * np.exp((17.27 * test['temparature'])/(test['temparature'] + 273.3))
test['avp'] = 0.6108 * np.exp((17.27 * test['dewpoint']) / (test['dewpoint'] + 237.3))
test['vpd'] = test['svp'] - test['avp']


train.info(),test.info()


#Display missing value
missing_data = train.isnull().sum()
missing_data2 = test.isnull().sum()
print("Missing Data Count: ")
print(missing_data[missing_data>0])
print("Missing Data Count: ")
print(missing_data2[missing_data2>0])


#For rainfall_pre1: backward Fill
# train['rainfall_pre1'] = train['rainfall_pre1'].bfill()
test.fillna(test.median(), inplace=True)
#For rainfall_pre2: backward fill
# train['rainfall_pre2'] = train['rainfall_pre2'].bfill()

#For rainfall_7day_sum: Rolling Average
#Fill using the rolling sum of the rainfall column (7-day window)
# train['rainfall_7day_sum'] = train['rainfall_7day_sum'].fillna(
#     train['rainfall'].rolling(window=7, min_periods=1).sum()
# )

#Verify that all missing values have been filled
print("Missing Data Count After Imputation:")
print(train.isnull().sum())
print(test.isnull().sum())


plt.figure(figsize=(22,20))
sns.heatmap(train.corr(),annot=True,cmap="coolwarm")
plt.show()


#Temperature different ratio
train["temp_ratio"] = train["temp_diff"]/(train["maxtemp"] + 1e-6)
test["temp_ratio"] = test["temp_diff"]/(test["maxtemp"] + 1e-6)
#why we calculate temp_ratio? : It help understand temperature varialibity.
#A high temperature means big flucation and low temperature means stable temparature.

#test transformation
train["Humidty_cloud_log"] = np.log1p(train["humididty_cloud_ineraction"])
test["Humidty_cloud_log"] = np.log1p(test["humididty_cloud_ineraction"])
#why we did log? :To reduced Skewness of the data and to make data normally distributed.


plt.figure(figsize=(22,20))
sns.heatmap(train.corr(),annot=True,cmap="coolwarm")
plt.show()


train["day_of_week"] = train["date"].dt.dayofweek
train["quarter"] = train["date"].dt.quarter
train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)
test["day_of_week"] = test["date"].dt.dayofweek
test["quarter"] = test["date"].dt.quarter
test["is_weekend"] = (test["day_of_week"] >= 5).astype(int)
test_ids = test["id"]


train=train.drop(columns="date")
test= test.drop(columns="date")
train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)


# Select only numeric columns
numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns

# Determine the number of rows needed
num_features = len(numeric_cols)
num_cols = 5  # Keep 5 columns per row
num_rows = int(np.ceil(num_features / num_cols))  # Adjust rows dynamically

# Plot histograms
plt.figure(figsize=(15, num_rows * 3)) 
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(num_rows, num_cols, i)  # Dynamically adjust grid size
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()



#Should You Normalize First or Make the Data Normally Distributed?

#1.)If using models sensitive to skewness (e.g., Linear Regression, Logistic Regression, PCA, K-Means, etc.)
#irst, make the data normally distributed (use transformations like Log, Box-Cox, or Yeo-Johnson).
#Then, normalize (or standardize) the data if required.

#2.)If using models like Tree-Based Models (Random Forest, XGBoost, etc.)
#Normalization is usually not necessary, as these models are robust to skewed data.
#You can apply transformations only if they improve performance.


#Why do we have outliers
#âœ… Natural variations (e.g., extreme weather events like storms)
#âœ… Data entry errors (e.g., wrong decimal placement)
#âœ… Rare events (e.g., unexpected sensor readings)

plt.figure(figsize=(14, 8))
sns.boxplot(data=train, orient="h", palette="coolwarm")
plt.title("Boxplot of Numerical Features to Identify Outliers")
plt.show()



y = train['rainfall']
X = train.drop(columns="rainfall")

X_test = test


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

kf = KFold(n_splits=5, shuffle=True,random_state=42)

lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
lgb_cv_rmse = np.sqrt(-cross_val_score(lgb_model, X, y, cv=kf, scoring="neg_mean_squared_error")).mean()
print(f"LightGBM CV RMSE: {lgb_cv_rmse:.4f}")

# ---- XGBoost with Cross-Validation ----
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_cv_rmse = np.sqrt(-cross_val_score(xgb_model, X, y, cv=kf, scoring="neg_mean_squared_error")).mean()
print(f"XGBoost CV RMSE: {xgb_cv_rmse:.4f}")

# ---- CatBoost with Cross-Validation ----
cat_model = CatBoostRegressor(n_estimators=100, learning_rate=0.1, verbose=0)
cat_cv_rmse = np.sqrt(-cross_val_score(cat_model, X, y, cv=kf, scoring="neg_mean_squared_error")).mean()
print(f"CatBoost CV RMSE: {cat_cv_rmse:.4f}")


from sklearn.model_selection import train_test_split, GridSearchCV
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic", 
    eval_metric="auc",
    use_label_encoder=False
)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
}
grid_search = GridSearchCV(xgb_model, param_grid, scoring="roc_auc", cv=3, verbose=2)
grid_search.fit(X,y)


best_xgb = grid_search.best_estimator_


final_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
final_model.fit(X, y)


test_predictions = best_xgb.predict_proba(X_test)[:, 1]


#test_predictions = final_model.predict(X_test)test_predictions




submission = pd.DataFrame({"id": test_ids, "rainfall": test_predictions})
submission.to_csv("submission.csv", index=False)

print("\nâœ… Submission file created: submission.csv")


test_predictions




