!pip install -qq scikit-learn==1.6.1


import gc
import numpy as np
import os
import pandas as pd
import warnings
from catboost import CatBoostRegressor
from itertools import combinations
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import TargetEncoder


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
warnings.filterwarnings("ignore")


# training data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", sep=",")
print(df_train.shape)
df_train.head(1)


# test data
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv", sep=",")
print(df_test.shape)
df_test.head(1)


# sample submission
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv", sep=",")
print(sample_submission.shape)
sample_submission.head(1)


# fill number of ads
# Assumption: if ads none => maybe there is no ad right?
df_train.Number_of_Ads = df_train.Number_of_Ads.fillna(0)


# place a binary null indicator column for each feature that might get none value
def generate_null_indicators(df):
    """ create null indicator columns for given cols, or for all 
    """
    # check given cols for null values
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in numerical_cols:
        if df[df[col].isna()].shape[0] > 0:
            df[f"is_{col}_there"] = np.where(df[col].notna(), 1, 0)
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
    return df

df_train = generate_null_indicators(df_train)
df_test = generate_null_indicators(df_test)


# Now let's do imputation for the most important column: Episode_Length_minutes
# To do that we will use median value of every podcast name & genre groups 
# set group sizes
df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
group_counts = df_all.value_counts(['Podcast_Name', 'Genre']).reset_index(name='group_size')
df_all = df_all.merge(group_counts, on=['Podcast_Name', 'Genre'], how='left')

# global & group median
global_medians = {col: df_all[col].median() for col in ['Episode_Length_minutes']}
group_medians = (df_all[df_all['group_size'] >= 50].groupby(['Podcast_Name', 'Genre'])[['Episode_Length_minutes']].median())
group_medians_dict = group_medians.to_dict(orient='index')
group_sizes_dict = {(row['Podcast_Name'], row['Genre']): row['group_size'] for _, row in group_counts.iterrows()}


# apply imputation for each group: podcast name + genre
def apply_group_median_imputation(data):
    def impute_value(row, col):
        key = (row['Podcast_Name'], row['Genre'])
        if group_sizes_dict.get(key, 0) >= 50:
            return group_medians_dict.get(key, {}).get(col, global_medians[col])
        else:
            return global_medians[col]
    
    for col in global_medians.keys():
        data[col] = data.apply(lambda row: row[col] if pd.notnull(row[col]) else impute_value(row, col), axis=1)
    return data

df_train = apply_group_median_imputation(df_train)
df_test = apply_group_median_imputation(df_test)


# impute guest popularity
# Assumption: maybe there is no guest right?
df_train.Guest_Popularity_percentage = df_train.Guest_Popularity_percentage.fillna(0)
df_test.Guest_Popularity_percentage = df_test.Guest_Popularity_percentage.fillna(0)


# allocate some memory 
del df_all, group_counts, global_medians, group_medians, group_medians_dict, group_sizes_dict
gc.collect()


def create_bins(df, col, interval):
    min_val = df[col].min()
    max_val = df[col].max()
    bins = list(range(int(min_val // interval) * interval, int(max_val // interval + 2) * interval, interval))
    labels = [f"{bins[i]}â€“{bins[i+1]}" for i in range(len(bins)-1)]
    df[col.lower() + "_category"] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

def categorize_ad_row(ad):
    if ad == 0:
        return "ad_0"
    elif ad == 1:
        return "ad_1"
    elif ad == 2:
        return "ad_2"
    elif ad == 3:
        return "ad_3"
    elif (ad > 3) & (ad < 50):
        return "ad_4_50"
    elif (ad >= 50) & (ad < 100):
        return "ad_50_100"
    return "ad_100_or_more"

def reduce_memory_usage(data):
    """ reduce memory used by the dataframe by converting into 
        more memory friendly data types
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = data.memory_usage().sum() / 1024**2
    for col in data.columns:
        col_type = data[col].dtypes
        if col_type in numerics:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)
    end_mem = data.memory_usage().sum() / 1024**2
    print('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return data
        
def feature_engineering(data):
    # episode
    data['episode_number'] = data.Episode_Title.apply(lambda x: int(x.split(" ")[1]))
    # sentiment
    data['sentiment'] = data.Episode_Sentiment.map({"Negative": -1, "Neutral": 0, "Positive": 1})
    # popularity
    data['popularity_total_add'] = data.Host_Popularity_percentage + data.Guest_Popularity_percentage
    data['popularity_total_mul'] = data.Host_Popularity_percentage * (data.Guest_Popularity_percentage + 1)
    data['popularity_diff'] = data.Host_Popularity_percentage - data.Guest_Popularity_percentage
    data['popularity_guest_to_host'] = (data.Guest_Popularity_percentage / data.Host_Popularity_percentage + 0.01)
    data['popularity_add_per_minute'] = data.popularity_total_add / (data.Episode_Length_minutes + 0.01)
    data['popularity_mul_per_minute'] = data.popularity_total_mul / (data.Episode_Length_minutes + 0.01)
    data['popularity_add_per_ad'] = data.popularity_total_add / (data.Number_of_Ads + 1)
    data['popularity_mul_per_ad'] = data.popularity_total_mul / (data.Number_of_Ads + 1)
    # temporal features
    data['day_number'] = data.Publication_Day.map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
    data['day_time'] = data.Publication_Time.map({'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4})
    data['is_weekend'] = np.where((data['day_number'] == 6) | (data['day_number'] == 7) | ((data['day_number'] == 5) & ((data['day_time'] == 3) | (data['day_time'] == 4))), 1, 0)
    # ads 
    data['ads_per_minute'] = data.Number_of_Ads / (data.Episode_Length_minutes + 0.01)
    # categorize some float columns
    for col_i in ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage"]:
        data = create_bins(data, col=col_i, interval=1)
    data["number_of_ads_category"] = data.Number_of_Ads.apply(categorize_ad_row)
    # polynomial category features
    cols = ['episode_number', 'episode_length_minutes_category', 'host_popularity_percentage_category', 'guest_popularity_percentage_category', 'number_of_ads_category', 'Publication_Day', 'Publication_Time']
    all_combos = [comb for r in range(2, 5) for comb in combinations(cols, r)]
    for cols in all_combos:
        data['_'.join(list(cols))] = data[list(cols)].astype(str).agg('_'.join, axis=1)
    data.drop(columns=['episode_length_minutes_category', 'host_popularity_percentage_category', 'guest_popularity_percentage_category', 'number_of_ads_category'], inplace=True)
    # allocate some more memory
    data = reduce_memory_usage(data)
    gc.collect()
    return data

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)


# this is our metric function, for evaluating stuff!
def regression_metrics(y_true, y_pred):
    return {
        'R2': round(r2_score(y_true, y_pred), 5),
        'MAE': round(mean_absolute_error(y_true, y_pred), 5),
        'MAPE': round(mean_absolute_percentage_error(y_true, y_pred), 5),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 5),
        'sample_size': len(y_true)    
    }


# split data to train and validation
target = "Listening_Time_minutes"
features = sorted(list(set(df_train.columns.tolist()) - set(["id", target])))
cat_features = df_train.select_dtypes(include=['object']).columns.tolist()
X = df_train[features].copy()
y = df_train[target].copy() 
X.shape, y.shape, len(cat_features)


def cv_score(params={}, model_name="lgbm"):
    """ Runs CV folds sequentially with TE and returns OOF and test predictions """

    cv = KFold(5, random_state=42, shuffle=True)
    splits = list(cv.split(X, y))

    cv_preds = np.zeros(len(y))
    sub_pred = np.zeros(len(sample_submission))
    df_submission = df_test.copy().reset_index(drop=True)

    for fold_num, (idx_train, idx_valid) in enumerate(splits):
        # split train/val
        X_train, y_train = X.iloc[idx_train].copy(), y.iloc[idx_train]
        X_val, y_val = X.iloc[idx_valid].copy(), y.iloc[idx_valid]

        # target encoding
        encoder = TargetEncoder(random_state = 42)
        X_train[cat_features] = encoder.fit_transform(X_train[cat_features], y_train)
        X_val[cat_features] = encoder.transform(X_val[cat_features])

        df_sub_fold = df_submission.copy()
        df_sub_fold[cat_features] = encoder.transform(df_sub_fold[cat_features])

        gc.collect()

        # train
        if model_name == "lgbm":
            model = LGBMRegressor(verbose=-1, random_state=42, metric='rmse', **params)
        elif model_name == "catb":
            model = CatBoostRegressor(silent=True, random_state=42, **params)
        model.fit(X_train, y_train)

        # predict
        pred_val = model.predict(X_val)
        pred_test = model.predict(df_sub_fold[features])

        # score
        scores = regression_metrics(y_val, pred_val)
        print(f"Fold {fold_num} score:", scores)

        # store predictions
        cv_preds[idx_valid] = pred_val
        sub_pred += pred_test
        gc.collect()

    return cv_preds, sub_pred / 5


# cv 5 score, base (default) lgbm model (no hyperparameter tuning!)
cv_preds, sub_pred = cv_score(params={}, model_name="lgbm")
print("Overall CV score:", regression_metrics(y, cv_preds))


# create submission data
df_submission = df_test.copy().reset_index(drop=True)
all(df_submission.id == sample_submission.id)


# place predictions
df_submission["Listening_Time_minutes"] = sub_pred

# save submission
df_submission[["id", "Listening_Time_minutes"]].to_csv('/kaggle/working/df_submission.csv', index=False)
df_submission[["id", "Listening_Time_minutes"]].head()

