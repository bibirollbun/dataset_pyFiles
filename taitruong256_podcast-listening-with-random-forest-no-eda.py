import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

seed = 42 


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train_df.head(5)


train_df.info()


# column 0 
train_df.drop(columns=['id'], inplace=True)


# column 1
train_df.drop(columns=['Podcast_Name'], inplace=True)
test_df.drop(columns=['Podcast_Name'], inplace=True)


# column 2
train_df.drop(columns=['Episode_Title'], inplace=True)
test_df.drop(columns=['Episode_Title'], inplace=True)


# column 3
# train_df['Length_missing'] = train_df['Episode_Length_minutes'].isna().astype(int)
# test_df['Length_missing'] = test_df['Episode_Length_minutes'].isna().astype(int)

el_median = train_df['Episode_Length_minutes'].median()
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(el_median)
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(el_median)


# column 4 
targ_enc = train_df[['Genre','Listening_Time_minutes']].groupby(by='Genre').median()
train_df['Genre_encoded'] = targ_enc.loc[train_df['Genre']].values.squeeze()
test_df['Genre_encoded'] = targ_enc.loc[test_df['Genre']].values.squeeze()

train_df.drop(columns=['Genre'], inplace=True)
test_df.drop(columns=['Genre'], inplace=True)


# column 5 
train_df['Host_Popularity_percentage'] = train_df['Host_Popularity_percentage'] * train_df['Episode_Length_minutes']
test_df['Host_Popularity_percentage'] = test_df['Host_Popularity_percentage'] * test_df['Episode_Length_minutes']


# column 6 
day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6,
    
}

train_df['Publication_Day'] = train_df['Publication_Day'].map(day_mapping)
test_df['Publication_Day']  = test_df['Publication_Day'].map(day_mapping)


# column 7 
time_mapping = {
    'Morning': 0,
    'Afternoon': 1,
    'Evening': 2,
    'Night': 3
}

train_df['Publication_Time'] = train_df['Publication_Time'].map(time_mapping)
test_df['Publication_Time']  = test_df['Publication_Time'].map(time_mapping)


# column 8 
# train_df['Guest_Popularity_missing'] = train_df['Guest_Popularity_percentage'].isna().astype(int)
# test_df['Guest_Popularity_missing'] = test_df['Guest_Popularity_percentage'].isna().astype(int)

gp_median = train_df['Guest_Popularity_percentage'].median()
train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(gp_median)
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(gp_median)

train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'] * train_df['Episode_Length_minutes']
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'] * test_df['Episode_Length_minutes']


# column 9 
train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(0)
train_df['Number_of_Ads'] = train_df['Number_of_Ads'] / (train_df['Episode_Length_minutes']+0.00001)

test_df['Number_of_Ads'] = test_df['Number_of_Ads'].fillna(0)
test_df['Number_of_Ads'] = test_df['Number_of_Ads'] / (test_df['Episode_Length_minutes']+0.00001) 


# column 10 
sentiment_map = {"Negative": -1, "Neutral": 0, "Positive":1}
train_df['Episode_Sentiment'] = train_df['Episode_Sentiment'].map(sentiment_map)
test_df['Episode_Sentiment'] = test_df['Episode_Sentiment'].map(sentiment_map)


plt.figure(figsize=(12, 10))
corr_matrix = train_df.corr(numeric_only=True)

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of train_df (Lower Triangle)')
plt.show()


train_df.head(5)


train_df.info()


test_df.info()


features = train_df.columns.tolist()
target = 'Listening_Time_minutes'

if 'Listening_Time_minutes' in features:
    features.remove('Listening_Time_minutes')

X, y = train_df[features], train_df[target]


kf = KFold(n_splits=5, shuffle=True, random_state=42)


models = {
    "Random Forest": RandomForestRegressor(n_estimators=1000, min_samples_split=10, n_jobs=-1, random_state=seed),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=1000, min_samples_split=10, random_state=seed),
    "XGBoost": XGBRegressor(n_estimators=1000, random_state=seed),
    "LightGBM": LGBMRegressor(n_estimators=1000, random_state=seed),
    "CatBoost": CatBoostRegressor(n_estimators=1000, verbose=0, random_state=seed)
}


model_rmse_scores = {}

for model_name, model in models.items():
    rmse_scores = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse_scores.append(rmse)

    model_rmse_scores[model_name] = np.mean(rmse_scores)
    print(f"{model_name} - Mean RMSE: {np.mean(rmse_scores)}")


best_model_name = min(model_rmse_scores, key=model_rmse_scores.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with RMSE: {model_rmse_scores[best_model_name]}")


best_model.fit(X, y)


y_train_pred = best_model.predict(X)
rmse_train = np.sqrt(mean_squared_error(y, y_train_pred))

print("RMSE on training set:", rmse_train)


y_test_pred = best_model.predict(test_df[features])


submission = test_df[['id']].copy()
submission['Listening_Time_minutes'] = y_test_pred
submission.to_csv("submission.csv", index=False)
submission.head(5)




