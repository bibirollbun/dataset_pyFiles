podc_dict = {
    'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3,
    'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7,
    'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11,
    'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15,
    'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19,
    'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23,
    'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27,
    'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31,
    'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34,
    'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37,
    'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40,
    'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43,
    'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46,
    'Tune Time': 47
}

genr_dict = {
    'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4,
    'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9
}

week_dict = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

time_dict = {
    'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3
}

sent_dict = {
    'Negative': 0, 'Neutral': 1, 'Positive': 2
}
podc_dict = {
    'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3,
    'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7,
    'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11,
    'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15,
    'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19,
    'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23,
    'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27,
    'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31,
    'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34,
    'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37,
    'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40,
    'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43,
    'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46,
    'Tune Time': 47
}

genr_dict = {
    'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4,
    'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9
}

week_dict = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

time_dict = {
    'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3
}

sent_dict = {
    'Negative': 0, 'Neutral': 1, 'Positive': 2
}



import pandas as pd

df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
print(df_train.isnull().sum())
df_train['Episode_Title'] = df_train['Episode_Title'].str.extract(r'(\d+)').astype(int)
df_train['Genre'] = df_train['Genre'].replace(genr_dict).infer_objects(copy=False)
df_train['Podcast_Name'] = df_train['Podcast_Name'].replace(podc_dict).infer_objects(copy=False)
df_train['Publication_Day'] = df_train['Publication_Day'].replace(week_dict).infer_objects(copy=False)
df_train['Publication_Time'] = df_train['Publication_Time'].replace(time_dict).infer_objects(copy=False)
df_train['Episode_Sentiment'] = df_train['Episode_Sentiment'].replace(sent_dict).infer_objects(copy=False)

df_train['Episode_Length_is_NaN'] = df_train['Episode_Length_minutes'].isna()
df_train['Guest_Popularity_percentage_NaN'] = df_train['Guest_Popularity_percentage'].isna()
df_small = df_train.dropna(subset=['Episode_Length_minutes']).copy()
df_small.drop(columns=['Episode_Length_is_NaN', 'Guest_Popularity_percentage_NaN'], inplace=True)

#Remove outliers
#Remove outliers #if the value is greater than 3, set it to 3 If not, set it to 3
df_train['Number_of_Ads'] = df_train['Number_of_Ads'].apply(lambda x: 3 if x > 3 else x)
df_small['Number_of_Ads'] = df_small['Number_of_Ads'].apply(lambda x: 3 if x > 3 else x)

df_train['Number_of_Ads'] = df_train['Number_of_Ads'].fillna(1)
df_small['Number_of_Ads'] = df_small['Number_of_Ads'].fillna(1)
#df_train = pd.get_dummies(df_train, columns=['Episode_Sentiment','Genre','Publication_Day','Publication_Time','Podcast_Name'], drop_first=True)

#remove line
df_train.drop(columns=['id'], inplace=True)
df_small.drop(columns=['id'], inplace=True)
pd.set_option('future.no_silent_downcasting', True)


print(df_train['Episode_Length_minutes'].mean())
print(df_train['Guest_Popularity_percentage'].mean())


df_train['Episode_Length_minutes'] = df_train['Episode_Length_minutes'].fillna(64.50473835100325)
df_train['Guest_Popularity_percentage'] = df_train['Guest_Popularity_percentage'].fillna(52.23644893379307)
df_small['Guest_Popularity_percentage'] = df_small['Guest_Popularity_percentage'].fillna(52.23644893379307)


print(df_small.isnull().sum())


import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = df_train.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = df_small.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix")
plt.show()


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
# # Divide into features and objective variables
X = df_train.drop(columns=['Listening_Time_minutes'])
y = df_train['Listening_Time_minutes']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2,random_state=55)


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

X_small = df_small.drop(columns=['Listening_Time_minutes'])
y_small = df_small['Listening_Time_minutes']

X_train_small, X_valid_small, y_train_small, y_valid_small = train_test_split(X_small, y_small, test_size=0.2,random_state=55)


from sklearn.metrics import mean_squared_error

import numpy as np

# define model
# {'subsample': 0.6, 'num_leaves': 128, 'n_estimators': 1000, 'min_child_samples': 20, 'max_depth': -1, 'learning_rate': 0.05, 'colsample_bytree': 1.0}
lgb_model1 = lgb.LGBMRegressor(
    num_leaves=128,
    min_child_samples=20,
    colsample_bytree=1.0,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    n_jobs=2,
    metric='rmse')

lgb_model1.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

# Predict on the validation set
y_pred = lgb_model1.predict(X_valid)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("RMSE:", rmse)



from sklearn.metrics import mean_squared_error
import numpy as np

lgb_model2 = lgb.LGBMRegressor(
    num_leaves=128,
    min_child_samples=20,
    colsample_bytree=1.0,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    n_jobs=2,
    metric='rmse')

lgb_model2.fit(X_train_small, y_train_small,eval_set=[(X_valid_small, y_valid_small)])

# Predict on the validation set
y_pred_small = lgb_model2.predict(X_valid_small)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid_small, y_pred_small))
print("RMSE:", rmse)



df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
print(df_test.isnull().sum())
df_test['Episode_Title'] = df_test['Episode_Title'].str.extract(r'(\d+)').astype(int)


df_test['Genre'] = df_test['Genre'].replace(genr_dict).infer_objects(copy=False)
df_test['Podcast_Name'] = df_test['Podcast_Name'].replace(podc_dict).infer_objects(copy=False)
df_test['Publication_Day'] = df_test['Publication_Day'].replace(week_dict).infer_objects(copy=False)
df_test['Publication_Time'] = df_test['Publication_Time'].replace(time_dict).infer_objects(copy=False)
df_test['Episode_Sentiment'] = df_test['Episode_Sentiment'].replace(sent_dict).infer_objects(copy=False)



df_test['Episode_Length_is_NaN'] = df_test['Episode_Length_minutes'].isna()
df_test['Guest_Popularity_percentage_NaN'] = df_test['Guest_Popularity_percentage'].isna()



df_test['Number_of_Ads'] = df_test['Number_of_Ads'].apply(lambda x: 3 if x > 3 else x)

df_test['Episode_Length_minutes'] = df_test['Episode_Length_minutes'].fillna(64.50473835100325)
df_test['Guest_Popularity_percentage'] = df_test['Guest_Popularity_percentage'].fillna(52.23644893379307)
id = df_test['id']
#remove line
df_test.drop(columns=['id'], inplace=True)

pd.set_option('future.no_silent_downcasting', True)


mask_nan = df_test['Episode_Length_is_NaN'] == True
mask_not_nan = ~mask_nan  # False

# predict
y_pred_nan = lgb_model1.predict(df_test[mask_nan])
y_pred_not_nan = lgb_model2.predict(df_test[mask_not_nan].drop(columns=['Episode_Length_is_NaN', 'Guest_Popularity_percentage_NaN']))

y_test_pred = pd.Series(index=df_test.index, dtype='float64')
y_test_pred[mask_nan] = y_pred_nan
y_test_pred[mask_not_nan] = y_pred_not_nan

# save
submission = pd.DataFrame({'id': id, 'Listening_Time_minutes': y_test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)


