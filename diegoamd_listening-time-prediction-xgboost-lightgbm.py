import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold


import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from catboost import Pool
import optuna


data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
data.head()


data.info()


data.isna().sum() / len(data)


data["Number_of_Ads"].value_counts()


ads_filtered = data[data["Number_of_Ads"].isin([0, 1, 2, 3])]
sns.boxplot(x = "Number_of_Ads", y = "Listening_Time_minutes", data = ads_filtered)
plt.show()


mean_ads = ads_filtered["Number_of_Ads"].mean()
median_ads = ads_filtered["Number_of_Ads"].median()
print(f"Mean: {mean_ads}")
print(f"Median: {median_ads}")


data_fill_ads = data.copy()
data_fill_ads["Number_of_Ads"] = data["Number_of_Ads"].fillna(median_ads)
data_fill_ads.isna().sum()


sns.histplot(x = "Episode_Length_minutes", data = data_fill_ads, bins = 40)
plt.show()


sns.boxplot(y = "Episode_Length_minutes", data = data_fill_ads)
plt.show()


data_fill_ads[data_fill_ads["Episode_Length_minutes"] > 300]


episodes_len = data_fill_ads[data_fill_ads["Podcast_Name"] == "Home & Living"]["Episode_Length_minutes"]
episodes_len.describe()


sns.histplot(data = episodes_len)
plt.show()


ep_len_mean = episodes_len.mean()
data_fill_ep_len = data_fill_ads.copy()
data_fill_ep_len.loc[data_fill_ep_len["id"] == 101637, "Episode_Length_minutes"] = ep_len_mean


data_fill_ep_len.loc[data_fill_ep_len["id"] == 101637, "Episode_Length_minutes"]


ep_len_mean2 = data_fill_ep_len["Episode_Length_minutes"].mean()
ep_len_mean2


data_fill_ep_len["Episode_Length_minutes"] = data_fill_ep_len["Episode_Length_minutes"].fillna(ep_len_mean2)
data_fill_ep_len.isna().sum()


sns.boxplot(y = "Guest_Popularity_percentage", data = data_fill_ep_len)
plt.show()


data_fill_ep_len["Guest_Popularity_percentage"].describe()


data_fill_ep_len.loc[data_fill_ep_len["Guest_Popularity_percentage"] > 100, "Guest_Popularity_percentage"] = 100.00
data_fill_ep_len["Guest_Popularity_percentage"].describe()


data_fill_guest_pop = data_fill_ep_len.copy()
mean_guest_pop = data_fill_guest_pop["Guest_Popularity_percentage"].mean()
print(mean_guest_pop)


data_fill_guest_pop["Guest_Popularity_percentage"] = data_fill_guest_pop["Guest_Popularity_percentage"].fillna(mean_guest_pop)
data_fill_guest_pop.isna().sum()


data_fill_guest_pop["Guest_Popularity_percentage"].describe()


data_fill_guest_pop.sample(20)


sns.countplot(x = "Episode_Sentiment", data = data_fill_guest_pop)
plt.show()


data_fill_guest_pop["Episode_Sentiment_Number"] = data_fill_guest_pop["Episode_Sentiment"].map({"Positive": 1, "Neutral": 0, "Negative": -1})


sns.countplot(x = "Publication_Time", data = data_fill_guest_pop)
plt.show()


data_fill_guest_pop["Publication_Time_number"] = data_fill_guest_pop["Publication_Time"].map({"Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4})


sns.countplot(x = "Publication_Day", data = data_fill_guest_pop)
plt.show()


data_fill_guest_pop["Publication_Day_number"] = data_fill_guest_pop["Publication_Day"].map({"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
                                                                                           "Friday": 5, "Saturday": 6, "Sunday": 7})


sns.countplot(y = "Genre", data = data_fill_guest_pop)
plt.show()


data_fill_guest_pop["Episode_Title"].str.contains("Episode").sum()


data_episode_num = data_fill_guest_pop.copy()
data_episode_num["Episode_Number"] = data_episode_num["Episode_Title"].apply(lambda x: x.split()[1])
data_episode_num["Episode_Number"] = data_episode_num["Episode_Number"].astype("int")


sns.histplot(data = data_episode_num, x = "Episode_Number")
plt.show()


data_episode_num["Podcast_Name"].value_counts()


sns.histplot(x = "Listening_Time_minutes", data = data_episode_num, bins = 20)
plt.show()


sns.boxplot(x = "Listening_Time_minutes", data = data_episode_num)
plt.show()


train_data = data_episode_num.copy()
train_data.info()


X = train_data.drop(columns = ["id", "Listening_Time_minutes", "Episode_Title"])
y = train_data["Listening_Time_minutes"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 19)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


X_train.info()


cat_cols = ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]
num_cols = ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads", "Episode_Sentiment_Number",
           "Publication_Time_number", "Publication_Day_number", "Episode_Number"]


for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')


ohe = OneHotEncoder(handle_unknown = "ignore")


cat_data = X_train[cat_cols]
cat_data_encoded = ohe.fit_transform(cat_data)


num_data = X_train[num_cols].values


X_train_processed = np.hstack([cat_data_encoded.toarray(), num_data])


cat_data_test = X_test[cat_cols]
cat_data_test_encoded = ohe.transform(cat_data_test)


num_data_test = X_test[num_cols].values


X_test_processed = np.hstack([cat_data_test_encoded.toarray(), num_data_test])


def rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


lin_reg = LinearRegression()
lin_reg.fit(X_train_processed, y_train)
lin_reg_pred = lin_reg.predict(X_test_processed)
rmse_lin_reg = rmse(y_test, lin_reg_pred)
print(rmse_lin_reg)


ridge_reg = Ridge()
ridge_reg.fit(X_train_processed, y_train)
ridge_reg_pred = ridge_reg.predict(X_test_processed)
rmse_ridge_reg = rmse(y_test, ridge_reg_pred)
print(rmse_ridge_reg)


lasso_reg = Lasso()
lasso_reg.fit(X_train_processed, y_train)
lasso_reg_pred = lasso_reg.predict(X_test_processed)
rmse_lasso_reg = rmse(y_test, lasso_reg_pred)
print(rmse_lasso_reg)


tree = DecisionTreeRegressor(random_state = 19)
tree.fit(X_train_processed, y_train)
tree_pred = tree.predict(X_test_processed)
rmse_tree = rmse(y_test, tree_pred)
print(rmse_tree)


xgb = XGBRegressor(random_state = 19)
xgb.fit(X_train_processed, y_train)
xgb_pred = xgb.predict(X_test_processed)
rmse_xgb = rmse(y_test, xgb_pred)
print(rmse_xgb)


forest = RandomForestRegressor(random_state = 19, n_estimators = 50, max_depth = 10, n_jobs = -1)
forest.fit(X_train_processed, y_train)
forest_pred = forest.predict(X_test_processed)
rmse_forest = rmse(y_test, forest_pred)
print(rmse_forest)


lightGBM = LGBMRegressor(random_state = 19, n_jobs = -1)
lightGBM.fit(
    X_train, y_train,
    eval_set = [(X_test, y_test)],
    categorical_feature = cat_cols
)
lightGBM_pred = lightGBM.predict(X_test)
rmse_lightGBM = rmse(y_test, lightGBM_pred)
print(rmse_lightGBM)


cat_model = CatBoostRegressor(
    iterations = 5000,
    learning_rate = 0.05,
    depth = 6,
    od_type = 'Iter',
    task_type = "GPU",
    random_state = 19,
    verbose = 100,
    early_stopping_rounds = 50
)

cat_model.fit(
    X_train, y_train,
    eval_set = (X_test, y_test),
    cat_features = cat_cols
)

cat_model_pred = cat_model.predict(X_test)
rmse_cat_model = rmse(y_test, cat_model_pred)
print(rmse_cat_model)


param_dist = {
    'n_estimators': [100, 300, 500, 800],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1, 5],
    'reg_lambda': [0.5, 1, 5, 10]
}


xgb = XGBRegressor(random_state = 19, n_jobs = -1)


search = RandomizedSearchCV(
    estimator = xgb,
    param_distributions = param_dist,
    n_iter = 30,
    scoring='neg_root_mean_squared_error',
    cv = 3,
    verbose = 2,
    random_state = 19,
    n_jobs = -1
)


#search.fit(X_train_processed, y_train)


#best_xgb = search.best_estimator_


#search.best_params_


#y_pred = best_xgb.predict(X_test_processed)
#rmse_xgb_tune = rmse(y_test, y_pred)
#print("RMSE:", rmse_xgb_tune)


best_xgb_ = XGBRegressor(random_state = 19, n_jobs = -1,
                       subsample = 1.0,
                       reg_lambda = 10,
                       reg_alpha = 0.1,
                       n_estimators = 800,
                       max_depth = 10,
                       learning_rate = 0.1,
                       gamma = 0,
                       colsample_bytree = 0.6)

best_xgb_.fit(X_train_processed, y_train)
y_pred = best_xgb_.predict(X_test_processed)
rmse_xgb_tune_ = rmse(y_test, y_pred)
print("RMSE:", rmse_xgb_tune_)


param_dist = {
    'num_leaves': np.arange(10, 300, 10),
    'max_depth': np.arange(3, 15, 1),
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}


model = lgb.LGBMRegressor()
random_search = RandomizedSearchCV(model, 
                                   param_distributions = param_dist, 
                                   n_iter = 30, cv = 3, verbose = 1, 
                                   random_state = 19, n_jobs = -1)


random_search.fit(
    X_train, y_train,
    categorical_feature = cat_cols
)


print("Best hyperparameters:", random_search.best_params_)


best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)


rmse_lgbm_tune = rmse(y_test, y_pred)
print("RMSE:", rmse_lgbm_tune)


test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = test_data.copy()
test_data.head()


test_data.isna().sum()


test_data["Episode_Length_minutes"] = test_data["Episode_Length_minutes"].fillna(ep_len_mean2)
test_data["Guest_Popularity_percentage"] = test_data["Guest_Popularity_percentage"].fillna(mean_guest_pop)
test_data.isna().sum()


test_data.loc[test_data["Host_Popularity_percentage"] > 100, "Host_Popularity_percentage"] = 100
test_data.loc[test_data["Guest_Popularity_percentage"] > 100, "Guest_Popularity_percentage"] = 100


test_data["Episode_Sentiment_Number"] = test_data["Episode_Sentiment"].map({"Positive": 1, "Neutral": 0, "Negative": -1})
test_data["Publication_Time_number"] = test_data["Publication_Time"].map({"Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4})
test_data["Publication_Day_number"] = test_data["Publication_Day"].map({"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
                                                                        "Friday": 5, "Saturday": 6, "Sunday": 7})
test_data["Episode_Number"] = test_data["Episode_Title"].apply(lambda x: x.split()[1])
test_data["Episode_Number"] = test_data["Episode_Number"].astype("int")


test_data = test_data.drop(columns = ["id", "Episode_Title"])


#cat_data_submit = test_data[cat_cols]
#cat_data_submit_encoded = ohe.transform(cat_data_submit)


#num_data_submit = test_data[num_cols].values


#test_data_processed = np.hstack([cat_data_submit_encoded.toarray(), num_data_submit])


#y_submit = best_xgb_.predict(test_data_processed)
#y_submit


for col in cat_cols:
    test_data[col] = test_data[col].astype("category")


y_submit = best_model.predict(test_data)
y_submit


submit_example = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submit_example.head()


submission = submission[["id"]]
submission["Listening_Time_minutes"] = y_submit
submission.head()


submission.to_csv("sub3.csv", index = False)




