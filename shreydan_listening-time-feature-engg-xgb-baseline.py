import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import math
import cudf
pd.set_option('max_colwidth',None)
tqdm.pandas()


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_df.info()


train_df['Podcast_Name'].value_counts()


train_df.groupby('Podcast_Name')['Genre'].unique().reset_index()


def get_podcast_minutes(df, name):
    return np.median(df[df['Podcast_Name']==name]['Episode_Length_minutes'].dropna().values)
    
def get_popularity(df, name):
    return np.median(df[df['Podcast_Name']==name]['Guest_Popularity_percentage'].dropna().values)
    
def get_ads(df, name):
    return np.ceil(df[df['Podcast_Name']==name]['Number_of_Ads'].dropna().values.mean())


podcast_lens = {n:get_podcast_minutes(train_df,n) for n in set(train_df['Podcast_Name'])}
nan_mask = train_df['Episode_Length_minutes'].isna()
train_df.loc[nan_mask, 'Episode_Length_minutes'] = train_df.loc[nan_mask, 'Podcast_Name'].map(podcast_lens)
train_df['ELm_imputed'] = 0
train_df.loc[nan_mask,'ELm_imputed'] = 1

nan_mask = test_df['Episode_Length_minutes'].isna()
test_df.loc[nan_mask, 'Episode_Length_minutes'] = test_df.loc[nan_mask, 'Podcast_Name'].map(podcast_lens)
test_df['ELm_imputed'] = 0
test_df.loc[nan_mask,'ELm_imputed'] = 1


podcast_pops = {n:get_popularity(train_df,n) for n in set(train_df['Podcast_Name'])}
nan_mask = train_df['Guest_Popularity_percentage'].isna()
train_df.loc[nan_mask, 'Guest_Popularity_percentage'] = train_df.loc[nan_mask, 'Podcast_Name'].map(podcast_pops)
train_df['GPp_imputed'] = 0
train_df.loc[nan_mask,'Gpp_imputed'] = 1

nan_mask = test_df['Guest_Popularity_percentage'].isna()
test_df.loc[nan_mask, 'Guest_Popularity_percentage'] = test_df.loc[nan_mask, 'Podcast_Name'].map(podcast_pops)
test_df['GPp_imputed'] = 0
test_df.loc[nan_mask,'Gpp_imputed'] = 1


podcast_ads = {n:get_ads(train_df,n) for n in set(train_df['Podcast_Name'])}
nan_mask = train_df['Number_of_Ads'].isna()
train_df.loc[nan_mask, 'Number_of_Ads'] = train_df.loc[nan_mask, 'Podcast_Name'].map(podcast_ads)
train_df['NoA_imputed'] = 0
train_df.loc[nan_mask,'NoA_imputed'] = 1

nan_mask = test_df['Number_of_Ads'].isna()
test_df.loc[nan_mask, 'Number_of_Ads'] = test_df.loc[nan_mask, 'Podcast_Name'].map(podcast_ads)
test_df['NoA_imputed'] = 0
test_df.loc[nan_mask,'NoA_imputed'] = 1


# max episodes is 100 (this doesnt make sense as there are wayyy more episodes than 100 but based on Episode_Title this is what I came up with at best!)
train_df['Episode_Title'] = train_df['Episode_Title'].map(lambda e:int(e.replace('Episode',''))) / 100.
test_df['Episode_Title'] = test_df['Episode_Title'].map(lambda e:int(e.replace('Episode',''))) / 100.


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['host_guest_popularity'] = (df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']) / 100
    df['popularity_diff'] = (df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']).abs()
    df['popularity_ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1e-5)
    df['episode_length_per_ad'] = df['Episode_Length_minutes'] / (df['Number_of_Ads'] + 1e-5)
    df['ad_density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['ads_per_10min'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] / 10 + 1e-5)

    # --- Weighted Sentiment ---
    df['weighted_sentiment'] = df['Episode_Sentiment'] * (
        df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
    ) / 2

    df['sentiment_gap_host_pop'] = df['Episode_Sentiment'] - (df['Host_Popularity_percentage'] / 100)
    df['sentiment_gap_guest_pop'] = df['Episode_Sentiment'] - (df['Guest_Popularity_percentage'] / 100)


    # --- Binned Features ---
    df['host_pop_bin'] = pd.qcut(df['Host_Popularity_percentage'], q=4, labels=False)
    df['guest_pop_bin'] = pd.qcut(df['Guest_Popularity_percentage'], q=4, labels=False)
    df['ep_length_bin'] = pd.qcut(df['Episode_Length_minutes'], q=4, labels=False)

    # --- Day of Week Features ---
    df['is_weekend'] = df['Publication_Day'].isin(['Friday', 'Saturday', 'Sunday']).astype(int)
    
    return df


sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
train_df['Episode_Sentiment'] = train_df['Episode_Sentiment'].map(sentiment_map)
test_df['Episode_Sentiment'] = test_df['Episode_Sentiment'].map(sentiment_map)

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

day_map = {d:i for i,d in enumerate(['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])}
time_map = {t:i for i,t in enumerate(['Morning','Afternoon','Evening','Night'])}
train_df['Publication_Day'] = train_df['Publication_Day'].map(day_map)
train_df['Publication_Time'] = train_df['Publication_Time'].map(time_map)


test_df['Publication_Day'] = test_df['Publication_Day'].map(day_map)
test_df['Publication_Time'] = test_df['Publication_Time'].map(time_map)


podcast_stats = train_df.groupby('Podcast_Name').mean(numeric_only=True).T.to_dict()

new_fts = [('diff_ep_length','Episode_Length_minutes'),
           ('diff_host_pop','Host_Popularity_percentage'),
           ('diff_guest_pop','Guest_Popularity_percentage'),
           ('publication_day_diff','Publication_Day'),
           ('publication_time_diff','Publication_Time'),
           ('num_ads_diff','Number_of_Ads'),
           ('host_guest_diff','host_guest_popularity')
          ]
train_df[[f[0] for f in new_fts]] = 0.

for podcast, stats in podcast_stats.items():
    for new_ft_name, main_ft_name in new_fts:
        mask = train_df['Podcast_Name'] == podcast
        train_df.loc[mask, new_ft_name] = train_df.loc[mask, main_ft_name] - stats[main_ft_name]
        mask = test_df['Podcast_Name'] == podcast
        test_df.loc[mask, new_ft_name] = test_df.loc[mask, main_ft_name] - stats[main_ft_name]


train_df['Day_sin'] = np.sin(2 * np.pi * train_df['Publication_Day'] / 7)
train_df['Day_cos'] = np.cos(2 * np.pi * train_df['Publication_Day'] / 7)
train_df['Time_sin'] = np.sin(2 * np.pi * train_df['Publication_Time'] / 4)
train_df['Time_cos'] = np.cos(2 * np.pi * train_df['Publication_Time'] / 4)

test_df['Day_sin'] = np.sin(2 * np.pi * test_df['Publication_Day'] / 7)
test_df['Day_cos'] = np.cos(2 * np.pi * test_df['Publication_Day'] / 7)
test_df['Time_sin'] = np.sin(2 * np.pi * test_df['Publication_Time'] / 4)
test_df['Time_cos'] = np.cos(2 * np.pi * test_df['Publication_Time'] / 4)


text_model = SentenceTransformer('/kaggle/input/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2')
names = list(set(train_df['Podcast_Name']))
name_encodings_full = text_model.encode(names)
name_pca = PCA(n_components=0.9)
name_encodings = name_pca.fit_transform(name_encodings_full)
name_encodings = {n:e for n,e in zip(names, list(name_encodings))}
name_encodings_full = {n:e for n,e in zip(names, list(name_encodings_full))}

genres = list(set(train_df['Genre']))
genre_encodings_full = text_model.encode(genres)
genre_pca = PCA(n_components=0.9)
genre_encodings = genre_pca.fit_transform(genre_encodings_full)
genre_encodings = {n:e for n,e in zip(genres, list(genre_encodings))}
genre_encodings_full = {n:e for n,e in zip(genres, list(genre_encodings_full))}


train_df['Podcast_Embed'] = train_df['Podcast_Name'].map(name_encodings)
test_df['Podcast_Embed'] = test_df['Podcast_Name'].map(name_encodings)
# name_dims = [f"name_dim_{i}" for i in range(name_pca.n_components_)]
# train_df[name_dims] = train_df['Podcast_Embed'].to_list()
# test_df[name_dims] = test_df['Podcast_Embed'].to_list()

train_df['Genre_Embed'] = train_df['Genre'].map(genre_encodings)
test_df['Genre_Embed'] = test_df['Genre'].map(genre_encodings)
# genre_dims = [f"genre_dim_{i}" for i in range(genre_pca.n_components_)]
# train_df[genre_dims] = train_df['Genre_Embed'].to_list()
# test_df[genre_dims] = test_df['Genre_Embed'].to_list()

# ------------------------

train_df['Podcast_Embed_Full'] = train_df['Podcast_Name'].map(name_encodings_full)
test_df['Podcast_Embed_Full'] = test_df['Podcast_Name'].map(name_encodings_full)

train_df['Genre_Embed_Full'] = train_df['Genre'].map(genre_encodings_full)
test_df['Genre_Embed_Full'] = test_df['Genre'].map(genre_encodings_full)


name_map = {n:i for i,n in enumerate(set(train_df['Podcast_Name']))}
genre_map = {n:i for i,n in enumerate(set(train_df['Genre']))}

train_df['Podcast_Name'] = train_df['Podcast_Name'].map(name_map)
test_df['Podcast_Name'] = test_df['Podcast_Name'].map(name_map)

train_df['Genre'] = train_df['Genre'].map(genre_map)
test_df['Genre'] = test_df['Genre'].map(genre_map)


def cossim(df,col1='Podcast_Embed_Full',col2='Genre_Embed_Full'):
    col1_stack = np.stack(df[col1].values)
    col2_stack = np.stack(df[col2].values)
    
    dot_products = np.einsum('ij,ij->i', col1_stack, col2_stack)
    norm_col1 = np.linalg.norm(col1_stack, axis=1)
    norm_col2 = np.linalg.norm(col2_stack, axis=1)
    
    denominator = norm_col1 * norm_col2
    cosine_similarities = np.where(denominator != 0, dot_products / denominator, 0)
    return cosine_similarities


train_df['podcast-genre-similarity'] = cossim(train_df)
test_df['podcast-genre-similarity'] = cossim(test_df)


train_df.drop(columns=['id','Podcast_Embed','Genre_Embed','Podcast_Embed_Full','Genre_Embed_Full','Publication_Day','Publication_Time'],inplace=True)
test_ids = test_df['id'].to_list()
test_df.drop(columns=['id','Podcast_Embed','Genre_Embed','Podcast_Embed_Full','Genre_Embed_Full','Publication_Day','Publication_Time'],inplace=True)


train_df = train_df.astype(float)
test_df = test_df.astype(float)


test_df.head()


train_df.columns


target_col = 'Listening_Time_minutes'
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

# 7-Fold Cross Validation
kf = KFold(n_splits=7, shuffle=True)

models = []
mae_list, mse_list, rmse_list, r2_list = [], [], [], []

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1),total=7):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=10_000,
        learning_rate=0.05,
        max_depth=20,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.1,
        reg_alpha=0.15,  # L1 regularization
        reg_lambda=1.5,   # L2 regularization
        min_child_weight=3,
        tree_method='hist',
        device='gpu',
        sampling_method='gradient_based',
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=300, verbose=False)
    preds = model.predict(X_val)
    models.append(model)
    mae = mean_absolute_error(y_val, preds)
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, preds)
    r2_train = r2_score(y_train, model.predict(X_train))

    print(f"Fold {fold} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, R² train: {r2_train:.4f}")

    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

# Average metrics across folds
print("\nAverage Across Folds:")
print(f"MAE:  {np.mean(mae_list):.4f}")
print(f"MSE:  {np.mean(mse_list):.4f}")
print(f"RMSE: {np.mean(rmse_list):.4f}")
print(f"R²:   {np.mean(r2_list):.4f}")


X_test = test_df.copy()
test_preds = np.mean([m.predict(X_test) for m in models], axis=0)


submission = pd.DataFrame({'id':test_ids,target_col:test_preds})


submission.to_csv('submission.csv',index=False)


submission.head()

