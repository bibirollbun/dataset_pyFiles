# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

df = pd.concat([df_train, df_test], axis=0)
df.drop(columns=['id'], inplace=True)

df.head()


#categories in ascending order from least to greatest avg listening time

genre_mapping = {
    'News': 1,
    'Comedy': 2,
    'Sports': 3,
    'Lifestyle': 4,
    'Business': 5,
    'Technology': 6,
    'Education': 7,
    'Health': 8,
    'True Crime': 9,
    'Music': 10
}

day_mapping = {
    'Sunday': 1,
    'Thursday': 2,
    'Friday': 3,
    'Saturday': 4,
    'Wednesday': 5,
    'Monday': 6,
    'Tuesday': 7
}

time_mapping = {
    'Evening': 1,
    'Morning': 2,
    'Afternoon': 3,
    'Night': 4
}

sentiment_mapping = {
    'Negative': -1,      
    'Neutral': 0,       
    'Positive': 1      
}


def preprocess_data(data): 
    df = data.copy()
    #drop duplicates
    df.drop_duplicates(inplace=True)


    #fill missing values
    df.Episode_Length_minutes.fillna(df.Episode_Length_minutes.median(), inplace=True)
    df.Guest_Popularity_percentage.fillna(df.Guest_Popularity_percentage.mean(), inplace=True)
    df.Number_of_Ads.fillna(df.Number_of_Ads.mode()[0], inplace=True)

    #remove outliers
    for col in df.drop('Listening_Time_minutes', axis=1).columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if col == 'Number_of_Ads':
                df.loc[df[col] < lower_bound, col] = df[col].min()
                df.loc[df[col] > upper_bound, col] = df[col].quantile(0.99)
            else:
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound

    #map low cardinality categorical variables to numerical
    df['Genre'] = df['Genre'].map(genre_mapping)
    df.Publication_Day = df.Publication_Day.map(day_mapping)
    df.Publication_Time = df.Publication_Time.map(time_mapping)
    df.Episode_Sentiment = df.Episode_Sentiment.map(sentiment_mapping)

    df.Episode_Title = df.Episode_Title.str.split(' ').str[1].astype(int)
    df.Number_of_Ads = df.Number_of_Ads.astype(int)

    df['Podcast_Name'] = df['Podcast_Name'].astype('category').cat.codes

    df['Length_minutes_square'] = df['Episode_Length_minutes'].pow(2).astype(float)
    df['Length_minutes_cubed'] = df['Episode_Length_minutes'].pow(3).astype(float)
    df['Ads_to_length_ratio'] = (df['Number_of_Ads'] / df['Episode_Length_minutes']).astype(float) 
    df['Popularity_score'] = ((df['Guest_Popularity_percentage'] + df['Host_Popularity_percentage']) / 2).astype(float)
    df['Length_sentiment'] = (df['Episode_Length_minutes'] * df['Episode_Sentiment']).astype(float)

    df.Ads_to_length_ratio.fillna(0, inplace=True)
    return df
df = preprocess_data(df)

df.head()


from sklearn.model_selection import train_test_split

X = df[:len(df_train)].drop(columns=['Listening_Time_minutes'])
y = df[:len(df_train)]['Listening_Time_minutes']

X_test = df[len(df_train):].drop(columns=['Listening_Time_minutes'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xg = xgb.XGBRegressor(n_estimators=500, learning_rate=0.08, max_depth=15, verbose=-1)
xg.fit(X_train, y_train)
y_pred = xg.predict(X_val)
rmse_xg = np.sqrt(np.mean((y_val - y_pred) ** 2))

print(f"RMSE XGBoost: {rmse_xg}")


from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.08, max_depth=15)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_val)

rmse_gb = np.sqrt(np.mean((y_val - y_pred) ** 2))

print(f"RMSE Gradient Boosting: {rmse_gb}")


gb_preds = gb.predict(X)
xg_preds = xg.predict(X)

new_df = pd.DataFrame({
    'XG': xg_preds,
    'GB': gb_preds,
    'Label': y
})

new_df.head()


gb_preds = gb.predict(X_test)
xg_preds = xg.predict(X_test)

preds = (xg_preds + gb_preds) / 2
submission = pd.DataFrame({'id': df_test['id'].values, 'Listening_Time_minutes': preds})
submission.to_csv('submission.csv', index=False)
submission.head()




