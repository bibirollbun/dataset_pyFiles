# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train


test


train.info()


train.shape


"""from the below summary,we can find almost mean and median almost equal, which explains there are no outliers
except one where the max in number of ads columns has got exreme value from the mean which we can confirm in the 
further process"""
train.describe()


train.isna().sum()


#Univariate Analysis
print(train["Podcast_Name"].value_counts(normalize=True)*100)


train["Genre"].value_counts(normalize=True)*100


train["Publication_Day"].value_counts(normalize=True)*100


train["Publication_Time"].value_counts(normalize=True)*100


train.groupby(["Genre"])[["Episode_Length_minutes"]].mean().reset_index().sort_values(by="Episode_Length_minutes",ascending=False).head(5)


train.groupby(["Genre"])[["Guest_Popularity_percentage"]].mean().reset_index().sort_values(by="Guest_Popularity_percentage",ascending=False).head(5)


train.groupby(["Genre"])[["Host_Popularity_percentage"]].mean().reset_index().sort_values(by="Host_Popularity_percentage",ascending=False).head(5)


train.groupby(["Genre"])[["Number_of_Ads"]].mean().reset_index().sort_values(by="Number_of_Ads",ascending=False).head(5)


train.groupby(["Podcast_Name"])[["Episode_Length_minutes"]].mean().reset_index().sort_values(by="Episode_Length_minutes",ascending=False).head(5)



train.groupby(["Podcast_Name"])[["Episode_Sentiment"]].count().reset_index().sort_values(by="Episode_Sentiment",ascending=False).head(5)


# the box plot below showcases the outliers present in the Number of ads column
cols_to_plot = ['Episode_Length_minutes', 'Guest_Popularity_percentage','Host_Popularity_percentage' ,'Number_of_Ads','Listening_Time_minutes']
for col in cols_to_plot:
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


# Bivariate Analysis 

"""Sunday is the prime day of the week followed by Monday & Friday"""

GP=round(pd.crosstab(train["Genre"],train["Publication_Day"], normalize='index') *100,2)
GP


"""Evening & Night are the prime time for almost all genres"""

GT=round(pd.crosstab(train["Genre"],train["Publication_Time"], normalize='index') *100,2)
GT


"""Thought Process, the Episode length will vary based both on time and genre , so took a mean of that, 
which can also be used to fillnull values"""

train.groupby(["Podcast_Name","Genre","Publication_Time"])[["Episode_Length_minutes"]].mean().reset_index()


"""Thought Process, the Guest pop % will vary based both on time and genre , so took a mean of that, 
which can also be used to fillnull values"""

train.groupby(["Podcast_Name","Genre","Publication_Time"])[["Guest_Popularity_percentage"]].mean().reset_index()


train.columns


GE=round(pd.crosstab(train["Genre"],train["Episode_Sentiment"], normalize='index') *100,2)
GE


#  this shows that there is no correlation between the dependant variables
Train_num=train[["Episode_Length_minutes","Host_Popularity_percentage","Guest_Popularity_percentage","Number_of_Ads","Listening_Time_minutes"]]
sns.heatmap(data=Train_num.corr(),annot=True)
plt.show()


test.isna().sum()


# Null values and outlier Treatment in training set 

train["Episode_Length_minutes"]=train["Episode_Length_minutes"].fillna(train.groupby(["Genre","Publication_Time"])["Episode_Length_minutes"].transform("mean"))
train["Guest_Popularity_percentage"]=train["Guest_Popularity_percentage"].fillna(train.groupby(["Genre","Publication_Time"])["Guest_Popularity_percentage"].transform("mean"))
train["Number_of_Ads"] = train["Number_of_Ads"].fillna(train["Number_of_Ads"].mean())

test["Episode_Length_minutes"]=test["Episode_Length_minutes"].fillna(test.groupby(["Genre","Publication_Time"])["Episode_Length_minutes"].transform("mean"))
test["Guest_Popularity_percentage"]=test["Guest_Popularity_percentage"].fillna(test.groupby(["Genre","Publication_Time"])["Guest_Popularity_percentage"].transform("mean"))



train_Cleaned = train[~(train['Number_of_Ads'] >= 10)]
train_Cleaned.shape


train_Cleaned['Episode_Title'] = train_Cleaned['Episode_Title'].str.replace(r'[^0-9]', '', regex=True)
test['Episode_Title'] = test['Episode_Title'].str.replace(r'[^0-9]', '', regex=True)


train_Cleaned


from category_encoders import TargetEncoder
cat_cols = ['Podcast_Name','Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
target_col = 'Listening_Time_minutes'

encoder = TargetEncoder(cols=cat_cols)

X_Encoded = train_Cleaned.copy()
X_Encoded=X_Encoded.drop(["Episode_Title"],axis=1)
X_Encoded[cat_cols] = encoder.fit_transform(X_Encoded[cat_cols], X_Encoded[target_col])


X_Encoded



#for test set
X_test_Encoded = test.copy()
X_test_Encoded = X_test_Encoded.drop(["Episode_Title"], axis=1)
X_test_Encoded[cat_cols] = encoder.transform(X_test_Encoded[cat_cols])

X_test_Encoded


X=X_Encoded.drop("Listening_Time_minutes",axis=1)
y=X_Encoded["Listening_Time_minutes"]

print(X.shape)
print(y.shape)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_Std = scaler.fit_transform(X_train)
X_test_Std =  scaler.transform(X_test)

print(X_train.shape)


X_train=pd.DataFrame(X_train_Std,columns=X_train.columns)
X_test=pd.DataFrame(X_test_Std,columns=X_test.columns)


#Linear Regression Model 
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import accuracy_score ,r2_score,mean_squared_error

lr=LinearRegression()
lr.fit(X_train,y_train)
print("train Score:",lr.score(X_train,y_train))
y_pred=lr.predict(X_test)


r2score=r2_score(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print('R2Score',r2score)
print("RMSE",rmse)
print("AdjustedR2:",1 - ( 1-r2_score(y_test,y_pred)) * ( len(y_train) - 1 ) / ( len(y_train) - X_train.shape[1] - 1 ))

print("coeff:",lr.coef_)
print("intercept;",lr.intercept_)


#Lasso Regression  
la=Lasso(alpha=0.01)
la.fit(X_train,y_train)
print("train Score:",la.score(X_train,y_train))
y_pred=la.predict(X_test)

r2score=r2_score(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print('R2Score',r2score)
print("RMSE",rmse)
print("AdjustedR2:",1 - ( 1-r2_score(y_test,y_pred)) * ( len(y_train) - 1 ) / ( len(y_train) - X_train.shape[1] - 1 ))

print("coeff:",lr.coef_)
print("intercept;",lr.intercept_)


#Ridge regression 
le=Ridge(alpha=0.01)
le.fit(X_train,y_train)
print("Train Score:",le.score(X_train,y_train))
y_pred=le.predict(X_test)

r2score=r2_score(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print('R2Score',r2score)
print("RMSE",rmse)
print("AdjustedR2:",1 - ( 1-r2_score(y_test,y_pred)) * ( len(y_train) - 1 ) / ( len(y_train) - X_train.shape[1] - 1 ))

print("coeff:",lr.coef_)
print("intercept;",lr.intercept_)


from sklearn.linear_model import ElasticNet

LEN=ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
LEN.fit(X_train,y_train)
print("Train Score:",LEN.score(X_train,y_train))
y_pred=LEN.predict(X_test)

r2score=r2_score(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print('R2Score',r2score)
print("RMSE",rmse)
print("AdjustedR2:",1 - ( 1-r2_score(y_test,y_pred)) * ( len(y_train) - 1 ) / ( len(y_train) - X_train.shape[1] - 1 ))

print("coeff:",lr.coef_)
print("intercept;",lr.intercept_)



# Random Forest Model 
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=3,
    criterion='squared_error',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)

print(f'Train R2 Score: {r2_score(y_train, pred_train)}')
print(f'Test R2 Score: {r2_score(y_test, pred_test)}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, pred_test, squared=False))}')





# tuning the RF Model takes a lot of time , hence skippped these lines and proeeded with other models
# check for your reference code below
"""# Random Forest Model 
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Model
rf = RandomForestRegressor(criterion='squared_error', random_state=42)

# Define parameter grid (same as you had)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,cv=3,     
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='r2'
)

# Fit model
random_search.fit(X_train, y_train)

# Best model and evaluation
print("Best Parameters:", random_search.best_params_)
print("Best R2 Score on Validation:", random_search.best_score_)

best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

print(f'Test R2 Score: {r2_score(y_test, y_pred)}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred, squared=False))}')"""


from sklearn.ensemble import GradientBoostingRegressor

# Model
gbdt = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Fit
gbdt.fit(X_train, y_train)

# Predict
y_pred = gbdt.predict(X_test)

# Evaluate
print(f"Test R2 Score: {r2_score(y_test, y_pred)}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")


import lightgbm as lgb

# Model
lgb_reg = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,  # -1 means no limit
    random_state=42
)

# Fit
lgb_reg.fit(X_train, y_train)

# Predict
y_pred = lgb_reg.predict(X_test)

# Evaluate
print(f"Test R2 Score: {r2_score(y_test, y_pred)}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")


import xgboost as xgb

# Model
xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    objective='reg:squarederror'  # Important for regression
)

# Fit
xgb_reg.fit(X_train, y_train)

# Predict
y_pred = xgb_reg.predict(X_test)

# Evaluate
print(f"Test R2 Score: {r2_score(y_test, y_pred)}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# Model
ada_reg = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Fit
ada_reg.fit(X_train, y_train)

# Predict
y_pred = ada_reg.predict(X_test)

# Evaluate
print(f"Test R2 Score: {r2_score(y_test, y_pred)}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")


X_test_Encoded


# from the above model, RF has the least RMSE 
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=3,
    criterion='squared_error',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)

print(f'Train R2 Score: {r2_score(y_train, pred_train)}')
print(f'Test R2 Score: {r2_score(y_test, pred_test)}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, pred_test, squared=False))}')



kaggle_predictions = rf.predict(X_test_Encoded)
submission = pd.DataFrame({
    "id": X_test_Encoded["id"],
    "Listening_Time_minutes": kaggle_predictions
})
submission.to_csv('submission.csv', index=False)
print(submission)

