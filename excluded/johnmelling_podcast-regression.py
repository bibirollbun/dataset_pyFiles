import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor


df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

all_df = [df, df_test]
df


sample_submission


df.dtypes


list(df.select_dtypes(include = ["object"]).columns)


#Most distributions are normal, but our target variable is skewed! We might want to consider using the ln of our target or at the very least our episode length input variable
#note, the log did not help 

df[list(df.select_dtypes(include = ["float"]).columns)].hist(layout=(5, 1), figsize=(8, 15)) # Display subplots
plt.show()


# df["Episode_Length_minutes"] = np.log1p(df["Episode_Length_minutes"])

# print("After the log transformation")
# sns.histplot(df["Episode_Length_minutes"], kde = True, bins = 50)


#Some serious outliers for number of ads and episode length minutes
df[list(df.select_dtypes(include = ["float"]).columns)].plot(
    kind='box',
    subplots = True,
    sharey = False,
    figsize = (15,10)
)

plt.subplots_adjust(wspace=0.5)
plt.show()

#removing outliers
df = df[df["Number_of_Ads"] < 4]


df[list(df.select_dtypes(include = ["float"]).columns)].plot(
    kind='box',
    subplots = True,
    sharey = False,
    figsize = (15,10)
)

plt.subplots_adjust(wspace=0.5)
plt.show()


df.isna().sum()

#episode_length_minutes - might be important to remove before training?
#guest_popularity_percentage - missing value might indicate no guest? we can add a seperate column for guest vs no guest
#number of ads - replace with 0


#Looks like we just have an error here so we can quickly fill it with the median

print(df[df["Number_of_Ads"].isna()])

df["Number_of_Ads"] = df["Number_of_Ads"].fillna(df["Number_of_Ads"].median())


print("Corr with Guest Pop Percentage without taking into account NAs :", df["Guest_Popularity_percentage"].corr(df["Listening_Time_minutes"]))

temp_x = df["Guest_Popularity_percentage"].fillna(0)
print("Corr with Guest Pop Percentage with replacing Guest Pop Percentage with 0 :", df["Listening_Time_minutes"].corr(temp_x))

temp_x = df["Guest_Popularity_percentage"].fillna(df["Guest_Popularity_percentage"].median())
print("Corr with Guest Pop Percentage with replacing Guest Pop Percentage with 0 :", df["Listening_Time_minutes"].corr(temp_x))

df["Guest_Binary"] = [1 if pd.notna(x) else 0 for x in df["Guest_Popularity_percentage"]]
print("Corr with a Guest Binary that considers whether or not a guest was present :", df["Listening_Time_minutes"].corr(df["Guest_Binary"]))

for col in ["Guest_Popularity_percentage", "Episode_Length_minutes"]:
    for d in [df, df_test]:
        d[col] = d[col].fillna(d[col].median())

df_test["Guest_Binary"] = [1 if pd.notna(x) else 0 for x in df_test["Guest_Popularity_percentage"]]


df.isna().sum()


print(df[list(df.select_dtypes(include = ["float"]).columns)].corrwith(df["Listening_Time_minutes"]))

df["add_length_ratio"] = df["Number_of_Ads"] / df["Episode_Length_minutes"]
df_test["add_length_ratio"] = df_test["Number_of_Ads"] / df_test["Episode_Length_minutes"]


mapping_dict = {"current_events" : ["Sports", "Business", "News", "Technology"], "entertainment": ["True Crime", "Comedy", "Music"], "wellness": ["Lifestyle", "Health", "Education"]}
#Possible Groups: [Sports,Business,News,Technology] [True Crime,Comedy,Music] [Lifestyle,Health,Education]
# Current Events
# Entertainment
# Wellness

print(df["Genre"].value_counts())

print("\n -------------------------------- \n")

#Invert so the mapping function works!
inverse_map = {v: k for k, values in mapping_dict.items() for v in values}
print(inverse_map)

print("\n -------------------------------- \n")

df["Genre"] = df["Genre"].map(inverse_map)
df_test["Genre"] = df_test["Genre"].map(inverse_map)
print(df["Genre"].head())


df["avg_popularity"] = (df["Host_Popularity_percentage"] + df["Guest_Popularity_percentage"]) / 2
df_test["avg_popularity"] = (df_test["Host_Popularity_percentage"] + df_test["Guest_Popularity_percentage"]) / 2


df


sample_df = df.sample(n = 3000)

X = sample_df.drop(['id', 'Podcast_Name', 'Episode_Title', 'Listening_Time_minutes'], axis = 1)
y = sample_df['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


ohcat_df = ['Genre', 'Publication_Day', 'Publication_Time']
ocat_df = ['Episode_Sentiment']
int_df = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Guest_Binary', 'add_length_ratio', 'avg_popularity']

oh = OneHotEncoder(sparse_output = False)
o = OrdinalEncoder()
mms = MinMaxScaler()
cat_imp = SimpleImputer(strategy = "most_frequent")
int_imp = SimpleImputer(strategy = "median")

int_transformer = Pipeline(steps = [('impute', int_imp), ('scale', mms)])
ohcat_transformer = Pipeline(steps = [('impute', cat_imp), ('encode', oh), ('scale', mms)])
ocat_transformer = Pipeline(steps = [('impute', cat_imp), ('encode', o), ('scale', mms)])

preprocessor = ColumnTransformer(transformers = [("ohcat", ohcat_transformer, ohcat_df), ("ocat", ocat_transformer, ocat_df), ("int", int_transformer, int_df)])


preprocessor.set_output(transform = 'pandas')

df_transformed = preprocessor.fit_transform(sample_df)

df_transformed


# Just to test a baseline theory, lets take a simple linear regression between the length in minutes and listeners as these had a VERY Linear relationship
#we need to figure out how to add missing values for episode_length_minutes accurately

sns.regplot(data=sample_df, x="Episode_Length_minutes", y="Listening_Time_minutes")


df_transformed.corrwith(df["Listening_Time_minutes"]).abs().sort_values(ascending = False)


lr = LinearRegression()
lr_pipe = Pipeline(steps = [('preprocessor', preprocessor), ('model', lr)])

lr_pipe.fit(X_train, y_train)

lr_pred = lr_pipe.predict(X_test)

print("R2: ", r2_score(y_test, lr_pred))
print("MSE: ", mean_squared_error(y_test, lr_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, lr_pred)))


rf = RandomForestRegressor()

rf_pipe = Pipeline(steps = [('preprocessor', preprocessor), ('model', rf)])
rf_pipe.fit(X_train, y_train)

rf_pred = rf_pipe.predict(X_test)

print("R2: ", r2_score(y_test, rf_pred))
print("MSE: ", mean_squared_error(y_test, rf_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, rf_pred)))


log_rf = TransformedTargetRegressor(rf, func=np.log1p, inverse_func = np.expm1)
log_rf_pipe = Pipeline(steps = [('preprocessor', preprocessor), ('model', log_rf)])

log_rf_pipe.fit(X_train, y_train)

log_rf_pred = log_rf_pipe.predict(X_test)

print("R2: ", r2_score(y_test, log_rf_pred))
print("MSE: ", mean_squared_error(y_test, log_rf_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, log_rf_pred)))


xgb = XGBRegressor(random_state = 0 )
xgb_pipe = Pipeline(steps = [('preprocessor', preprocessor), ('model', xgb)])

xgb_pipe.fit(X_train, y_train)

xgb_pred = xgb_pipe.predict(X_test)

print("R2: ", r2_score(y_test, xgb_pred))
print("MSE: ", mean_squared_error(y_test, xgb_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, xgb_pred)))


lgb = LGBMRegressor(n_estimators = 300, learning_rate = 0.01, reg_alpha = 0.05, verbose = -1, random_state = 0)
lgb_pipe = Pipeline(steps = [('preprocessor', preprocessor), ('model', lgb)])

lgb_pipe.fit(X_train, y_train)

lgb_pred = lgb_pipe.predict(X_test)

print("R2: ", r2_score(y_test, lgb_pred))
print("MSE: ", mean_squared_error(y_test, lgb_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, lgb_pred)))


# param_grid = {
#     'model__learning_rate': [0.01, 0.05, 0.1],
#     'model__n_estimators': [100, 500, 1000],
# }

# kf = KFold(n_splits=5, shuffle=True, random_state=0)

# grid = GridSearchCV(lgb_pipe, param_grid, cv=kf, scoring='neg_root_mean_squared_error')
# grid.fit(X, y)

# print("Best params:", grid.best_params_)
# print("Best CV score:", grid.best_score_)


X = df.drop(['id', 'Podcast_Name', 'Episode_Title', 'Listening_Time_minutes'], axis = 1)
y = df['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


lgb_pipe.fit(X_train, y_train)

lgb_full_pred = lgb_pipe.predict(X_test)

print("R2: ", r2_score(y_test, lgb_full_pred))
print("MSE: ", mean_squared_error(y_test, lgb_full_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, lgb_full_pred)))


fi_df = pd.DataFrame({'Feature': list(lgb_pipe[:-1].get_feature_names_out()), 'Importances': lgb.feature_importances_}).sort_values(by = 'Importances', ascending = False)

plt.figure(figsize = (10,10))
sns.barplot(x = 'Importances', y = 'Feature', data = fi_df)


final_pred = lgb_pipe.predict(df_test).round(3)
sample_submission["Listening_Time_minutes"] = final_pred

sample_submission


sample_submission.to_csv('submission.csv', index = False)

