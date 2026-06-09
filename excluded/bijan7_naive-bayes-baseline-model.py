import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error


dir_path = '/kaggle/input/playground-series-s5e4/'

df = pd.read_csv(os.path.join(dir_path,'train.csv'))

test_df = pd.read_csv(os.path.join(dir_path,'test.csv'))

df.info()

test_df.info()


# group-wise median imputation

def impute(df, column, group, imputer):
    df[column] = df[column].fillna(df[group].map(imputer))
    return df

def median_imputation(df, column, group):
    imputer = df.groupby(group)[column].median().to_dict()
    df = impute(df, column, group, imputer)
    return df, imputer



df.loc[df['Episode_Length_minutes'] == 0, 'Episode_Length_minutes'] = np.nan

df, episode_length_imputer = median_imputation(df, 'Episode_Length_minutes', 'Podcast_Name')
df, guest_popularity_imputer = median_imputation(df, 'Guest_Popularity_percentage', 'Podcast_Name')
df, num_ads_imputer = median_imputation(df, 'Number_of_Ads', 'Podcast_Name')
df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '').astype(int)

# remove outliers and invalid data

df = df[df['Number_of_Ads'] < 10]
df = df[df['Guest_Popularity_percentage'] <= 100.0]
df = df[df['Host_Popularity_percentage'] <= 100.0]

test_df = impute(test_df, 'Episode_Length_minutes', 'Podcast_Name', episode_length_imputer)
test_df = impute(test_df, 'Guest_Popularity_percentage', 'Podcast_Name', guest_popularity_imputer)
test_df = impute(test_df, 'Number_of_Ads', 'Podcast_Name', num_ads_imputer)
test_df['Episode_Title'] = test_df['Episode_Title'].str.replace('Episode ', '').astype(int)


print(df.info())
print(test_df.info())


listening_time_dict = df.groupby('Podcast_Name')['Listening_Time_minutes'].mean().to_dict()
df['predicted'] = df['Podcast_Name'].map(listening_time_dict)

listening_time_mean = df['Listening_Time_minutes'].mean()

rmse = mean_squared_error(df['Listening_Time_minutes'], df['predicted']) ** 0.5
print(f"RMSE: {rmse}")


def fit_nb(df):
    df["fraction_listened"] = df["Listening_Time_minutes"] / df["Episode_Length_minutes"]
    fractional_listening_time_dict = df.groupby('Podcast_Name')['fraction_listened'].mean().to_dict()
    fractional_listening_time_mean = df['fraction_listened'].mean()
    return fractional_listening_time_dict, fractional_listening_time_mean

def predict_nb(df, model):
    fractional_listening_time_dict, fractional_listening_time_mean = model
    df['Average_Fractional_Listening_time'] = df['Podcast_Name'].map(fractional_listening_time_dict)
    df['Average_Fractional_Listening_time'] = df['Average_Fractional_Listening_time'].fillna(fractional_listening_time_mean)
    df['predicted'] = df['Average_Fractional_Listening_time'] * df['Episode_Length_minutes']
    return df['predicted']

kf = KFold(n_splits=5, shuffle=True, random_state=7)

scores = []

nb_df = df.copy()

for train_i, val_i in kf.split(nb_df):
    train_df = nb_df.iloc[train_i].copy()
    val_df = nb_df.iloc[val_i].copy()

    model = fit_nb(train_df)
    preds = predict_nb(val_df, model)

    score = mean_squared_error(preds, val_df['Listening_Time_minutes']) ** 0.5
    scores.append(score)

print(f'CV scores')
print(f'Mean: {np.mean(scores)}')
print(f'stddev: {np.std(scores)}')


# predict the test set for submission

test_nb_df = test_df.copy()

full_model = fit_nb(nb_df)
test_preds = predict_nb(test_nb_df, full_model)
result = test_nb_df.copy()
result = result.rename(columns= {'predicted':'Listening_Time_minutes'})
result = result.filter(['id', 'Listening_Time_minutes'])
result.info()
result.to_csv('submission.csv', index=False)




