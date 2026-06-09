import pandas as pd 
import numpy as np
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor 
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error

#To supress any warning messages
import warnings 
warnings.filterwarnings("ignore")


#Loading the training data
train_df=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")

#Loading the testing data
test_df=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


#Display the first 5 rows of the training dataset
train_df.head()


#Display the first 5 rows of the testing dataset
test_df.head()


#Show informations about the training dataset
train_df.info()


#Display the statistical summary of the training dataset
train_df.describe()


#Show informations about the training dataset
test_df.info()


#Display the statistical summary of the training dataset
test_df.describe()


#Checking for missing values in the training dataset
train_df.isna().sum()


#Checking for missing values in the training datset
test_df.isna().sum()


# Checking for duplicates in the training dataset
print("The number of duplicated observations in the train dataset is equal to", train_df.duplicated().sum())

# Checking for duplicates in the testing dataset
print("The number of duplicated observations in the test dataset is equal to", test_df.duplicated().sum())


columns=list(train_df.columns[1:11])
print(columns)


plt.figure(figsize=(13,6))
sns.histplot(data=train_df, x="BeatsPerMinute", bins=30, kde=True, label=f"Skewness = {train_df['BeatsPerMinute'].skew():.2f}")
plt.title("Distribution of BeatsPerMinute")
plt.legend()
plt.show()


for col in columns[0:9]:
    plt.figure(figsize=(13,6))
    sns.histplot(data=train_df, x=col, bins=30, kde=True, label=f"Skewness = {train_df[col].skew():.2f}")
    plt.title(f"Distribution of {col}")
    plt.legend()
    plt.show()


for col in columns[0:9]:
    plt.figure(figsize=(13,6))
    sns.relplot(data=train_df, x="BeatsPerMinute", y=col, kind="scatter")
    plt.title(f"BeatsPerMinute VS {col}")
    plt.show()


for col in columns:
    sns.boxplot(x=col, data=train_df)
    plt.title(f"Outliers in {col}")
    plt.show()


for col in columns[0:9]:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    train_df = train_df[(train_df[col] >= lower) & (train_df[col] <= upper)]


plt.figure(figsize=(18,9))
sns.heatmap(train_df.drop(columns=["id"]).corr(),annot=True,fmt='.2f')


#Splitting data into explanatory variables X and response variable y
X=train_df.drop(columns=["BeatsPerMinute","id"])
y=train_df["BeatsPerMinute"]


#Splitting data into training and testing (20%) dataset
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)


lr=LinearRegression()
lr.fit(X_train,y_train)


y_pred=lr.predict(X_test)


lr_rmse=mean_squared_error(y_test, y_pred,squared=False)
print("Mean Squared Error value for Linear Regression model :",lr_rmse)


rg = Ridge()  
rg.fit(X_train, y_train)


y_pred = rg.predict(X_test)


rg_rmse=mean_squared_error(y_test, y_pred,squared=False)
print("Mean Squared Error value for Ridge model :",rg_rmse)


xgb = XGBRegressor()
xgb.fit(X_train, y_train)


y_pred = xgb.predict(X_test)


xgb_rmse=mean_squared_error(y_test, y_pred,squared=False)
print("Mean Squared Error value for Ridge model :",xgb_rmse)


cat = CatBoostRegressor()
cat.fit(X_train, y_train)


y_pred = cat.predict(X_test)


cat_rmse=mean_squared_error(y_test, y_pred,squared=False)
print("Mean Squared Error value for Ridge model :",cat_rmse)


grad = GradientBoostingRegressor()
grad.fit(X_train, y_train)


y_pred = grad.predict(X_test)


grad_rmse=mean_squared_error(y_test, y_pred,squared=False)
print("Mean Squared Error value for Ridge model :",grad_rmse)


models = ["LinearRegression","Ridge","XGBRegressor","CatBoostRegressor","GradientBoostingRegressor"]
rmse   = [lr_rmse, rg_rmse, xgb_rmse, cat_rmse, grad_rmse]


model_rmse = pd.DataFrame({"Model": models, "RMSE": rmse})
model_rmse.sort_values("RMSE",ignore_index=True)


sns.barplot(x="RMSE",y="Model",data=model_rmse.sort_values(by="RMSE"))
plt.xlim(min(rmse)-0.01, max(rmse)+0.01) 
plt.ylabel("Model")
plt.title("RMSE Scores")
plt.show()


# from scipy.stats import randint, uniform

# param_dist = {# smaller upper bound
#     'learning_rate': uniform(0.05, 0.15),    
#     'max_depth': randint(3, 8),              
#     'subsample': [0.8, 1.0],                 
#     'max_features': ['sqrt', 'log2']         
# }

# random_search = RandomizedSearchCV(
#     estimator=grad,
#     param_distributions=param_dist,
#     n_iter=100,                  
#     scoring='neg_root_mean_squared_error',
#     cv=5,                      
#     n_jobs=-1,
#     verbose=2,
#     random_state=42
# )

# random_search.fit(X_train, y_train)



# best_model = random_search.best_estimator_
# # Best hyperparameters: {'learning_rate': 0.052345461011179095, 'max_depth': 3, 'max_features': 'log2', 'n_estimators': 228, 'subsample': 1.0}

# # Best hyperparameters
# print("Best hyperparameters:", random_search.best_params_)


# y_pred = best_model.predict(X_test)

# # Compute RMSE
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("Test RMSE:", rmse)


# y_pred = best_model.predict(test_df.drop(columns=['id']))


best_grad_model = GradientBoostingRegressor(
    learning_rate=0.052345461011179095,
    max_depth=3,
    max_features='log2',
    n_estimators=228,
    subsample=1.0,
    random_state=42
)


best_grad_model.fit(X_train, y_train)
y_pred = best_grad_model.predict(X_test)


mean_squared_error(y_test, y_pred,squared=False)


y_pred=best_grad_model.predict(test_df.drop(columns=['id']))


submission = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': y_pred})
submission.to_csv('submission.csv', index=False)
display(submission.head())




