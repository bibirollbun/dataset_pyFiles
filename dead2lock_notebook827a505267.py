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


train_file = "/kaggle/input/playground-series-s5e4/train.csv"
test_file = "/kaggle/input/playground-series-s5e4/test.csv"


from matplotlib import pyplot as plt


train_df = pd.read_csv(train_file)
train_df.head(4)


# train_df.dropna(inplace=True)
train_df.info()


import re


def extract_episode_number(title):
    match = re.search(r'Episode (\d+)', title)
    if match:
        return int(match.group(1))
    return None


train_df['EP_no'] = train_df['Episode_Title'].apply(extract_episode_number)


# print("Guset_mean", train_df['Guest_Popularity_percentage'].mean())
# print("Guset_median", train_df['Guest_Popularity_percentage'].median())
# print("Guset_mode", train_df['Guest_Popularity_percentage'].mode()[0])
# print("Episode_L_mean", train_df['Episode_Length_minutes'].mean())
# print("Episode_L_median", train_df['Episode_Length_minutes'].median())
# print("Episode_L_mode", train_df['Episode_Length_minutes'].mode()[0])


train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].mean(), inplace=True)
train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mode()[0], inplace=True)


train_df.columns


# sd_tf = train_df[:3000]


# columns = [
#     'Episode_Length_minutes', 'Genre', 'Host_Popularity_percentage', 
#     'Publication_Day', 'Publication_Time', 'Number_of_Ads', 
#     'Episode_Sentiment', 'Guest_Popularity_percentage'
# ]
# fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(24, 24))

# for i, col in enumerate(columns):
#     axes[i].bar(sd_tf['Listening_Time_minutes'], sd_tf[col], label=col, color='b')
#     axes[i].set_title(f"{col} vs Length_of_show")
#     axes[i].set_xlabel("Listening_Time_minutes")
#     axes[i].set_ylabel(col)
#     axes[i].legend()
#     axes[i].grid(True)

# plt.tight_layout()

# plt.show()


train_df = train_df.drop(columns=['id'])  #, 'Episode_Length_minutes'


train_df.info()


train_df.describe()


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


encoders = {}
for col in ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']:
    encoders[col] = LabelEncoder()
    train_df[col] = encoders[col].fit_transform(train_df[col])


train_df.describe()


X = train_df.drop(columns=['Podcast_Name', 'Episode_Title', 'Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
scale1 = MinMaxScaler()
scale2 = RobustScaler()


# train_df['Number_of_Ads'] = scale.fit_transform(train_df[['Number_of_Ads']])


import seaborn as sns
corr_mat = X.corr()
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense


# model = Sequential([
#     Dense(10, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#     Dense(5, activation='relu'),
#     Dense(1, activation='relu'),
# ])


# model.compile(optimizer="adam", loss='mse', metrics=['mae'])
# model.fit(X_train_scaled, Y_train, epochs=10, batch_size=4, validation_data=(X_test_scaled, Y_test), verbose=0)
# preds = model.predict(X_test_scaled).flatten()


# def evaluate_model(name, y_test, y_pred):
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mae)
#     print(f"{name}: MAE = {mae:.2f}, Rmse = {rmse:.2f}")

# evaluate_model("Neural Networks", Y_test, preds)


# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


# Podcast_Name                 750000 non-null  object 
#  1   Episode_Title                750000 non-null  object 
#  2   Episode_Length_minutes       750000 non-null  float64
#  3   Genre                        750000 non-null  object 
#  4   Host_Popularity_percentage   750000 non-null  float64
#  5   Publication_Day              750000 non-null  object 
#  6   Publication_Time             750000 non-null  object 
#  7   Guest_Popularity_percentage  750000 non-null  float64
#  8   Number_of_Ads                750000 non-null  float64
#  9   Episode_Sentiment            750000 non-null  object 
#  10  Listening_Time_minutes       750000 non-null  float64
#  11  EP_no                


X['Episode_Length_minutes'] = scale2.fit_transform(X[['Episode_Length_minutes']])
X['Host_Popularity_percentage'] = scale1.fit_transform(X[['Host_Popularity_percentage']])
X['Guest_Popularity_percentage'] = scale1.fit_transform(X[['Guest_Popularity_percentage']])
X['Number_of_Ads'] = scale1.fit_transform(X[['Number_of_Ads']])


# X.drop(columns=['Episode_Length_minutes'])
X.dropna()


# X_scaled = X
X.describe()


from sklearn.model_selection import train_test_split, KFold


# X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.15, random_state=42)
# X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size = 0.5, random_state=42)


# X_train.shape


# X_train.dropna(inplace=True)
# Y_train.dropna(inplace=True)


import sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, ARDRegression, Lasso


# models = {'Linear Regression': LinearRegression(),
#          'SVM': SVR(), 
#           'ARD Regression': ARDRegression(),
#          'Decision Tree': DecisionTreeRegressor(random_state=42),
#          'Random Forest': RandomForestRegressor(n_estimators=60, random_state = 42),
#          'Gradient Boost': GradientBoostingRegressor(loss='absolute_error', learning_rate=0.1, random_state = 42),
#          'Ada Boost': AdaBoostRegressor(n_estimators=50, random_state=42)
#          }


from sklearn.metrics import mean_squared_error

# results = {}
# for name, model in models.items():
#     model.fit(X_train[:15000], Y_train.sample(15000))
#     Y_pred = model.predict(X_val)
#     val_rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
#     results[name] = {'Rmse Score': val_rmse}
    


# print(results)


test_df = pd.read_csv(test_file)
# test_df.dropna(inplace=True)
# test_df.drop(columns = ['id'], inplace = True)  #'Episode_Length_minutes'
test_df['EP_no'] = test_df['Episode_Title'].apply(extract_episode_number)
test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median(), inplace=True)
test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].mean(), inplace=True)
test_df['Number_of_Ads'].fillna(test_df['Number_of_Ads'].mode()[0], inplace=True)
encoders = {}
for col in ['Genre', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']:  #'Publication_Day', 'Publication_Time',
    encoders[col] = LabelEncoder()
    test_df[col] = encoders[col].fit_transform(test_df[col])


test_id = test_df['id']
test_df.drop(columns = ['id'], inplace = True)


test_df.info()


X_check = test_df.drop(columns=['Podcast_Name', 'Episode_Title'])   #'Episode_Length_minutes'
X_check['Episode_Length_minutes'] = scale2.fit_transform(X_check[['Episode_Length_minutes']])
X_check['Host_Popularity_percentage'] = scale1.fit_transform(X_check[['Host_Popularity_percentage']])
X_check['Guest_Popularity_percentage'] = scale1.fit_transform(X_check[['Guest_Popularity_percentage']])
X_check['Number_of_Ads'] = scale1.fit_transform(X_check[['Number_of_Ads']])


X_check.info()


X_check.describe()


# from sklearn.model_selection import GridSearchCV
# params = {
    
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [50, 100, 150],
#     'max_depth': [3, 5, 7]
# }


# par_model = GradientBoostingRegressor(random_state=42)
# grid_search = GridSearchCV(estimator = par_model, param_grid=params, cv=3, scoring='neg_root_mean_squared_error')


# grid_search.fit(X_train[:7000], Y_train[:7000])
# print("Best Parameters", grid_search.best_params_)
# print("Best Score", grid_search.best_score_)


import lightgbm as lgb
import xgboost as xgb
rfr = RandomForestRegressor(n_estimators=75, random_state = 42, verbose=1, n_jobs=-1, bootstrap = True, criterion='squared_error')
# rfr = RandomForestRegressor(n_estimators=20, random_state = 42, verbose=1, n_jobs=-1)
# rfr = ExtraTreeRegressor(random_state=42)
# rfr = DecisionTreeRegressor(random_state=42)
# xgb = Lasso(alpha = 0.01, random_state=42)
# rfr = LinearRegression()
# xgb = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.2, n_jobs=-1, random_state = 42)
xgb = xgb.XGBRegressor(n_estimators=300, eval_metric='rmse', early_stopping_rounds=20, random_state=42) #1st
# rfr = ARDRegression(verbose=True)
# rfr = AdaBoostRegressor(n_estimators=70, random_state=42)
gbr = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, random_state = 42)


kf = KFold(n_splits=5, shuffle = True, random_state=42)


fold_rmse = []
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"\n Fold {fold + 1}")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    Y_train, Y_val = y.iloc[train_index], y.iloc[val_index]
    
    rfr_w = 0.7
    xgb_w = 0.17
    gbr_w = 0.13
    
    xgb.fit(X_train, Y_train, eval_set=[(X_val, Y_val)])
    rfr.fit(X_train, Y_train)
    gbr.fit(X_train, Y_train)
    
    xgb_pred = xgb.predict(X_val)
    rfr_pred = rfr.predict(X_val)
    gbr_pred = gbr.predict(X_val)

    ensemble_preds = (xgb_pred * xgb_w + rfr_pred * rfr_w + gbr_pred * gbr_w)
    
    rmse = np.sqrt(mean_squared_error(Y_val, ensemble_preds))
    fold_rmse.append(rmse)
    print(f"Fold: {fold+1}; Rmse: {rmse}")

print(f"Average error: {np.mean(fold_rmse):.4f}")


# ensemble_pred = rfr.predict(X_check)

rfr_test_pred = rfr.predict(X_check)
gbr_test_pred = gbr.predict(X_check)
xgb_test_pred = xgb.predict(X_check)

ensemble_pred = (xgb_test_pred * xgb_w + rfr_test_pred * rfr_w + gbr_test_pred * gbr_w)


# rfr = ARDRegression(verbose=True)
# rfr = AdaBoostRegressor(n_estimators=70, random_state=42)
gbr = GradientBoostingRegressor(loss='absolute_error', learning_rate=0.1, random_state = 42)
rfr_w = 0.7
xgb_w = 0.17
gbr_w = 0.13

xgb.fit(X_train, Y_train, eval_set=[(X_val, Y_val)])
rfr.fit(X_train, Y_train)
gbr.fit(X_train, Y_train)
# model.fit(X_train, Y_train)  #eval_set=[(X_val, Y_val)]
# predictions = model.predict(X_check)
# Y_pred = model.predict(X_val)
xgb_pred = xgb.predict(X_val)
rfr_pred = rfr.predict(X_val)
gbr_pred = gbr.predict(X_val)

ensemble_preds = (xgb_pred * xgb_w + rfr_pred * rfr_w + gbr_pred * gbr_w)  #+ rfr_pred * rfr_w  #xgb_pred * xgb_w
val_rmse = np.sqrt(mean_squared_error(Y_val, ensemble_preds))

rfr_test_pred = rfr.predict(X_check)
gbr_test_pred = gbr.predict(X_check)
xgb_test_pred = xgb.predict(X_check)

ensemble_pred = (xgb_test_pred * xgb_w + rfr_test_pred * rfr_w + gbr_test_pred * gbr_w)  # + rfr_test_pred * rfr_w  #xgb_test_pred * xgb_w

# print(predictions)
print(val_rmse)
# print(ensemble_pred)


output_df = pd.DataFrame(test_id, index=None)


# output_df['id'] = test_id
output_df['Listening_Time_minutes'] = ensemble_pred
output_df.head()


output = output_df.to_csv('submission.csv',index=False)


output_df.info()


df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


df_sub.head()


df_sub['Listening_Time_minutes'] = ensemble_pred
df_sub.to_csv('submission.csv',index=False)


df_sub.head(5)


df_sub.describe()


# Version 11 #best 

