import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gc

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
train.columns


# Can treat this as one hot later
# train['Publication_Time'].unique()


# See graph to figure out how to work with null values

# train['Guest_Popularity_percentage'].mean()


# Make all 103. ...'s as 103
# train['Number_of_Ads'].unique()


# Can make all large numbers (>12) to be `large`

# train['Number_of_Ads'].value_counts().sort_index()


train['deriv_Number_of_Ads'] = train['Number_of_Ads'].apply(lambda x: x if x < 4 else 10)
train['deriv_Number_of_Ads'].value_counts().sort_index()


# Can treat this as numeric (1, 2, 3)
# train['Episode_Sentiment'].unique()


train['deriv_Episode_Sentiment'] = train['Episode_Sentiment'].apply(lambda x: 3 if x == 'Positive' else 1 if x == 'Negative' else 2)
train['deriv_Episode_Sentiment'].value_counts().sort_index()


# Final output
# train['Listening_Time_minutes']


train['deriv_Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(
    train.groupby('Podcast_Name')['Episode_Length_minutes'].transform('mean')
)


train['deriv_Episode_Title'] = train['Episode_Title'].apply(lambda x: int(x[8:]))
train['deriv_Episode_Title']


# train.to_csv('train.csv')


train["deriv_Guest_Popularity_percentage"] = train["Guest_Popularity_percentage"].fillna(0)


df = train[["Podcast_Name", "deriv_Episode_Title", "deriv_Episode_Length_minutes", 
            "Genre", "Host_Popularity_percentage", "Publication_Day", "Publication_Time",
            "deriv_Guest_Popularity_percentage", "deriv_Number_of_Ads", "deriv_Episode_Sentiment",
            "Listening_Time_minutes"
           ]].copy(deep=True)


days = set(df["Publication_Day"])
time = set(df["Publication_Time"])

averages = dict()

for d in days:
    for t in time:
        averages[f"{d}_{t}"] = df[(df["Publication_Day"] == d) & (df["Publication_Time"] == t)]["Listening_Time_minutes"].mean()

df["deriv_Avg_Listening_Time_On_Same_Time_of_Day"] = df.apply(lambda row: averages[f"{row['Publication_Day']}_{row['Publication_Time']}"], axis = 1)
df


df = df.drop(["Publication_Day", "Publication_Time"], axis=1)
df


df = pd.get_dummies(df, columns=["Genre"], dtype=int).drop(["Genre_Education"], axis = 1)


df = df.drop(['Podcast_Name', 'deriv_Episode_Title'], axis = 1)
df


in_cols = ['deriv_Episode_Sentiment',
 'Genre_News',
 'deriv_Episode_Length_minutes',
 'deriv_Number_of_Ads',
 'Genre_Comedy',
 'Genre_Music',
 'Genre_Lifestyle',
 'Genre_Technology',
 'Genre_Business',
 'Host_Popularity_percentage',
 'deriv_Guest_Popularity_percentage',
 'Genre_Health',
 'deriv_Avg_Listening_Time_On_Same_Time_of_Day',
 'Genre_True Crime',
 'Genre_Sports']
out_col = 'Listening_Time_minutes'

X = df[in_cols].values
y = df[out_col].values


df = df[in_cols + [out_col]]
df


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train.shape


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RMSE:", root_mean_squared_error(y_test, y_pred))


y_pred = model.predict(X_train)

print("RMSE:", root_mean_squared_error(y_pred, y_train))


pca = PCA(n_components=15)
X_pca = pca.fit_transform(X_train)


plt.scatter(X_pca[:,0], X_pca[:,4], c = y_train, cmap='viridis')
plt.colorbar(label='Value of listening time')


print(pca.singular_values_)


model = DecisionTreeRegressor(random_state=0, max_depth=9)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RMSE:", root_mean_squared_error(y_test, y_pred))


# #max_depth = 9 :: RMSE = 13.122988529321258
# #max_depth = 13 :: RMSE = 13.019193381838358
# model = RandomForestRegressor(random_state=0, max_depth=13)

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print("RMSE:", root_mean_squared_error(y_test, y_pred))


# model = RandomForestRegressor(random_state=0, max_depth=13)

# model.fit(X_train, y_train)
# y_pred = model.predict(X_train)

# print("RMSE:", root_mean_squared_error(y_train, y_pred))


# min_so_far_max_depth = None
# min_so_far_n_estimators = None
# min_so_far_rmse = 40

# for n_estimator in range(50,301,50):
#     for max_depth in range(2,20):
#         model = RandomForestRegressor(random_state=0, max_depth=max_depth, n_estimators=n_estimator)
#         print(f"max_depth:{max_depth} n_estimators: {n_estimator}")
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         rmse = root_mean_squared_error(y_test, y_pred)  
#         if rmse < min_so_far_rmse:
#             min_so_far_rmse = rmse
#             min_so_far_max_depth = max_depth
#             min_so_far_n_estimators = n_estimator
#         print("RMSE:", rmse)
#         gc.collect()


# max_depth:19 n_estimators: 300
# RMSE: 12.839115881984613


model = RandomForestRegressor(random_state=0, max_depth=25, n_estimators=300)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
root_mean_squared_error(y_test, y_pred)


# model = GradientBoostingRegressor()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# root_mean_squared_error(y_test, y_pred)

