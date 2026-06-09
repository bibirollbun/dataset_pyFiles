import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from IPython.core.display import HTML
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


train_data_path = '/kaggle/input/playground-series-s5e4/train.csv'
test_data_path = '/kaggle/input/playground-series-s5e4/test.csv'


train_df = pd.read_csv(train_data_path)
display(HTML(train_df.head(5).to_html()))


test_df = pd.read_csv(test_data_path)
display(HTML(test_df.head(5).to_html()))


sns.heatmap(train_df[['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']].corr(), cmap='Blues', annot=True)
plt.show()


train_df = train_df.drop('id', axis=1)

podcast_ids = test_df[['id']]
test_df = test_df.drop('id', axis=1)


# find all columns with missing values
nan_columns_train = train_df.columns[train_df.isna().any()].tolist()
print(nan_columns_train)
nan_columns_test = test_df.columns[test_df.isna().any()].tolist()
print(nan_columns_test)


train_df['Episode_Num'] = train_df['Episode_Title'].str[8:].astype(int)
train_df['is_weekend']   = train_df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
test_df['Episode_Num'] = test_df['Episode_Title'].str[8:].astype(int)
test_df['is_weekend']   = test_df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
data_types = train_df.dtypes
print("Data types of all columns:")
print(data_types)


# label encode Genre, Publication_Day, Publication_Time, Episode_Sentiment train df
encoder = LabelEncoder()
all_genre_labels = list(set(train_df['Genre']).union(set(test_df['Genre'])))
print(all_genre_labels)
encoder.fit(all_genre_labels)
train_df['Genre'] = encoder.transform(train_df['Genre'])
display(HTML(train_df.head(5).to_html()))

# label encode Genre, Publication_Day, Publication_Time, Episode_Sentiment test df
test_df['Genre'] = encoder.transform(test_df['Genre'])
display(HTML(test_df.head(5).to_html()))


publication_day_encoder = LabelEncoder()
train_df['Publication_Day'] = publication_day_encoder.fit_transform(train_df['Publication_Day'])
test_df['Publication_Day'] = publication_day_encoder.transform(test_df['Publication_Day'])

publication_time_encoder = LabelEncoder()
train_df['Publication_Time'] = publication_time_encoder.fit_transform(train_df['Publication_Time'])
test_df['Publication_Time'] = publication_time_encoder.transform(test_df['Publication_Time'])

episode_sentiment_encoder = LabelEncoder()
train_df['Episode_Sentiment'] = episode_sentiment_encoder.fit_transform(train_df['Episode_Sentiment'])
test_df['Episode_Sentiment'] = episode_sentiment_encoder.transform(test_df['Episode_Sentiment'])

display(HTML(train_df.head(5).to_html()))
display(HTML(test_df.head(5).to_html()))


# select float columns
float_columns = train_df.select_dtypes(include=[np.number, 'float']).columns.to_list()
print(float_columns)


train_df = train_df.drop('Podcast_Name', axis=1)
train_df = train_df.drop('Episode_Title', axis=1)
test_df = test_df.drop('Podcast_Name', axis=1)
test_df = test_df.drop('Episode_Title', axis=1)


standard_scalar = StandardScaler()
train_df[['Episode_Length_minutes']] = standard_scalar.fit_transform(train_df[['Episode_Length_minutes']])
test_df[['Episode_Length_minutes']] = standard_scalar.transform(test_df[['Episode_Length_minutes']])
train_df[['Host_Popularity_percentage']] = train_df[['Host_Popularity_percentage']]/100
test_df[['Host_Popularity_percentage']] = test_df[['Host_Popularity_percentage']]/100
train_df[['Guest_Popularity_percentage']] = train_df[['Guest_Popularity_percentage']]/100
test_df[['Guest_Popularity_percentage']] = test_df[['Guest_Popularity_percentage']]/100
display(HTML(train_df.head(5).to_html()))
display(HTML(test_df.head(5).to_html()))


# fill with median
nan_columns = list(set(train_df.columns[train_df.isna().any()]) | set(test_df.columns[test_df.isna().any()]))
print("Columns with NaN:", nan_columns)
median_number_of_ads = train_df['Number_of_Ads'].median()
median_guest_popularity_percentage = train_df['Guest_Popularity_percentage'].median()
median_episode_length_minutes = train_df['Episode_Length_minutes'].median()
print(median_number_of_ads)
print(median_guest_popularity_percentage)
print(median_episode_length_minutes)
train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(median_number_of_ads)
train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(median_guest_popularity_percentage)
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(median_episode_length_minutes)
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].fillna(median_number_of_ads)
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(median_guest_popularity_percentage)
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(median_episode_length_minutes)
display(HTML(train_df.head(5).to_html()))
display(HTML(test_df.head(5).to_html()))


print(train_df.isna().sum())
print(test_df.isna().sum())


listening_time = train_df[['Listening_Time_minutes']]
train_df = train_df.drop('Listening_Time_minutes', axis=1)
listening_time = listening_time.squeeze()


print(train_df.isna().sum())
print(test_df.isna().sum())
print(train_df.dtypes)


X_train, X_cv, Y_train, Y_cv = train_test_split(train_df, listening_time, random_state=42, test_size=0.3)


assert not X_train.isnull().any().any(), "X_train has NaNs"
assert not pd.Series(Y_train).isnull().any(), "y_train has NaNs"
assert not X_cv.isnull().any().any(), "X_train has NaNs"
assert not pd.Series(Y_cv).isnull().any(), "y_train has NaNs"


eval_set = [(X_cv, Y_cv)]
grid = XGBRegressor(random_state=42,
                  n_estimators=2500,
                  learning_rate=0.01,
                  min_child_weight=10,
                  max_depth=15,
                  subsample=0.9,
                  colsample_bytree=0.9,
                  gamma=0.1,
                  n_jobs=-1,
                  early_stopping_rounds=50,
                  eval_metric='rmse') # or 'mse')
grid.fit(X_train, Y_train, eval_set=eval_set,
                  verbose=True) # See performance during training)

# grid = GridSearchCV(pipeline, param_grid=param_grid, scoring='neg_mean_squared_error',
#                    cv=3, verbose=2, n_jobs=-1)
# grid.fit(X_train, Y_train)


# print("Best Parameters:", grid.best_params_)
# best_rf = grid.best_estimator_
best_rf = grid
# Predict on test set
y_pred = best_rf.predict(X_cv)

# Compute Mean Squared Error (MSE)
mse = mean_squared_error(Y_cv, y_pred)
print("Test MSE:", mse)
print("Test RMSE:", np.sqrt(mse))  # Root Mean Squared Error


Y_test = best_rf.predict(test_df)


df = pd.DataFrame({
    'id': podcast_ids.squeeze().values,
    'Listening_Time_minutes': Y_test.flatten()  # Ensure it's a 1D array
})

# Save to CSV
csv_file_path = '/kaggle/working/predictions_xgboost.csv'
df.to_csv(csv_file_path, index=False)

# Output the path where the file is saved
csv_file_path




