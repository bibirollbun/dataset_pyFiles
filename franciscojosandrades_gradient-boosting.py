fold = 0


import json
datamodels_path = '/kaggle/input/exp-20250801-142306-bf8ebb/'
with open(datamodels_path+'args.json', "r") as f:
        config = json.load(f)
config


import polars as pl
import pandas as pd
import numpy as np
import re
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from scipy.stats import skew, kurtosis
import warnings
import lightgbm as lgb
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


num_cols = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft',
          '.', ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']

text_changes = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']


lista_behavioral = ['activity_0_cnt','activity_1_cnt','activity_2_cnt','activity_3_cnt',
                    'activity_4_cnt','down_event_3_cnt','down_event_4_cnt','down_event_5_cnt',
                    'down_event_6_cnt','down_event_9_cnt','down_event_10_cnt','up_event_3_cnt',
                    'up_event_4_cnt','up_event_5_cnt','up_event_6_cnt','up_event_9_cnt',
                    'up_event_10_cnt','action_time_sum','down_time_mean','up_time_mean',
                    'action_time_mean','cursor_position_mean','down_time_std','up_time_std',
                    'action_time_std','cursor_position_std','down_time_median','up_time_median',
                    'action_time_median','cursor_position_median','down_time_min','up_time_min',
                    'action_time_min','cursor_position_min','down_time_max','up_time_max',
                    'action_time_max','cursor_position_max','down_time_quantile','up_time_quantile',
                    'action_time_quantile','cursor_position_quantile','activity',
                    'inter_key_largest_lantency','inter_key_median_lantency','mean_pause_time',
                    'std_pause_time','total_pause_time','pauses_half_sec','pauses_1_sec',
                    'pauses_1_half_sec','pauses_2_sec','pauses_3_sec','P-bursts_mean',
                    'P-bursts_std','P-bursts_count','P-bursts_median','P-bursts_max',
                    'P-bursts_first','P-bursts_last','R-bursts_mean','R-bursts_std',
                    'R-bursts_median','R-bursts_max','R-bursts_first','R-bursts_last',
                    'keys_per_second','product_to_keys']



def count_by_values(df, colname, values):
    fts = df.select(pl.col('id').unique(maintain_order=True))
    for i, value in enumerate(values):
        tmp_df = df.group_by('id').agg(pl.col(colname).is_in([value]).sum().alias(f'{colname}_{i}_cnt'))
        fts  = fts.join(tmp_df, on='id', how='left') 
    return fts


def dev_feats(df):
    
    print("< Count by values features >")
    
    feats = count_by_values(df, 'activity', activities)
    feats = feats.join(count_by_values(df, 'text_change', text_changes), on='id', how='left') 
    feats = feats.join(count_by_values(df, 'down_event', events), on='id', how='left') 
    feats = feats.join(count_by_values(df, 'up_event', events), on='id', how='left') 

    print("< Input words stats features >")

    temp = df.filter((~pl.col('text_change').str.contains('=>')) & (pl.col('text_change') != 'NoChange'))
    temp = temp.group_by('id').agg(pl.col('text_change').str.concat('').str.extract_all(r'q+'))
    temp = temp.with_columns(input_word_count = pl.col('text_change').list.lengths(),
                             input_word_length_mean = pl.col('text_change').apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_max = pl.col('text_change').apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_std = pl.col('text_change').apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_median = pl.col('text_change').apply(lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_skew = pl.col('text_change').apply(lambda x: skew([len(i) for i in x] if len(x) > 0 else 0)))
    temp = temp.drop('text_change')
    feats = feats.join(temp, on='id', how='left') 


    
    print("< Numerical columns features >")

    temp = df.group_by("id").agg(pl.sum('action_time').suffix('_sum'), pl.mean(num_cols).suffix('_mean'), pl.std(num_cols).suffix('_std'),
                                 pl.median(num_cols).suffix('_median'), pl.min(num_cols).suffix('_min'), pl.max(num_cols).suffix('_max'),
                                 pl.quantile(num_cols, 0.5).suffix('_quantile'))
    feats = feats.join(temp, on='id', how='left') 


    print("< Categorical columns features >")
    
    temp  = df.group_by("id").agg(pl.n_unique(['activity', 'down_event', 'up_event', 'text_change']))
    feats = feats.join(temp, on='id', how='left') 


    
    print("< Idle time features >")

    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.group_by("id").agg(inter_key_largest_lantency = pl.max('time_diff'),
                                   inter_key_median_lantency = pl.median('time_diff'),
                                   mean_pause_time = pl.mean('time_diff'),
                                   std_pause_time = pl.std('time_diff'),
                                   total_pause_time = pl.sum('time_diff'),
                                   pauses_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)).count(),
                                   pauses_1_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)).count(),
                                   pauses_1_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)).count(),
                                   pauses_2_sec = pl.col('time_diff').filter((pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)).count(),
                                   pauses_3_sec = pl.col('time_diff').filter(pl.col('time_diff') > 3).count(),)
    feats = feats.join(temp, on='id', how='left') 
    
    print("< P-bursts features >")

    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('time_diff')<2)
    temp = temp.with_columns(pl.when(pl.col("time_diff") & pl.col("time_diff").is_last_distinct()).then(pl.count()).over(pl.col("time_diff").rle_id()).alias('P-bursts'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(pl.mean('P-bursts').suffix('_mean'), pl.std('P-bursts').suffix('_std'), pl.count('P-bursts').suffix('_count'),
                                   pl.median('P-bursts').suffix('_median'), pl.max('P-bursts').suffix('_max'),
                                   pl.first('P-bursts').suffix('_first'), pl.last('P-bursts').suffix('_last'))
    feats = feats.join(temp, on='id', how='left') 


    print("< R-bursts features >")

    temp = df.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('activity').is_in(['Remove/Cut']))
    temp = temp.with_columns(pl.when(pl.col("activity") & pl.col("activity").is_last()).then(pl.count()).over(pl.col("activity").rle_id()).alias('R-bursts'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(pl.mean('R-bursts').suffix('_mean'), pl.std('R-bursts').suffix('_std'), 
                                pl.median('R-bursts').suffix('_median'), pl.max('R-bursts').suffix('_max'),
                                pl.first('R-bursts').suffix('_first'), pl.last('R-bursts').suffix('_last'))

    feats = feats.join(temp, on='id', how='left')
    
    return feats


def train_valid_split(data_x, data_y, train_idx, valid_idx):
    x_train = data_x.iloc[train_idx]
    y_train = data_y[train_idx]
    x_valid = data_x.iloc[valid_idx]
    y_valid = data_y[valid_idx]
    return x_train, y_train, x_valid, y_valid


def evaluate(data_x, data_y, model, random_state=42, n_splits=5, test_x=None):
    skf    = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    test_y = np.zeros(len(data_x)) if (test_x is None) else np.zeros((len(test_x), n_splits))
    for i, (train_index, valid_index) in enumerate(skf.split(data_x, data_y.astype(str))):
        train_x, train_y, valid_x, valid_y = train_valid_split(data_x, data_y, train_index, valid_index)
        model.fit(train_x, train_y)
        if test_x is None:
            test_y[valid_index] = model.predict(valid_x)
        else:
            test_y[:, i] = model.predict(test_x)
    return test_y if (test_x is None) else np.mean(test_y, axis=1)


def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']

def reconstruct_essay(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        if Input[0] == 'Replace':
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue
        if "M" in Input[0]:
            croppedTxt = Input[0][10:]
            splitTxt = croppedTxt.split(' To ')
            valueArr = [item.split(', ') for item in splitTxt]
            moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            if moveData[0] != moveData[2]:
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def get_essay_df(df):
    df       = df[df.activity != 'Nonproduction']
    temp     = df.groupby('id').apply(lambda x: reconstruct_essay(x[['activity', 'cursor_position', 'text_change']]))
    essay_df = pd.DataFrame({'id': df['id'].unique().tolist()})
    essay_df = essay_df.merge(temp.rename('essay'), on='id')
    return essay_df


def word_feats(df):
    essay_df = df
    df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    df = df.explode('word')
    df['word_len'] = df['word'].apply(lambda x: len(x))
    df = df[df['word_len'] != 0]

    word_agg_df = df[['id','word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df


def sent_feats(df):
    df['sent'] = df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    df['sent_len'] = df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    df['sent_word_count'] = df['sent'].apply(lambda x: len(x.split(' ')))
    df = df[df.sent_len!=0].reset_index(drop=True)

    sent_agg_df = pd.concat([df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS), 
                             df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df


def parag_feats(df):
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    # Number of characters in paragraphs
    df['paragraph_len'] = df['paragraph'].apply(lambda x: len(x)) 
    # Number of words in paragraphs
    df['paragraph_word_count'] = df['paragraph'].apply(lambda x: len(x.split(' ')))
    df = df[df.paragraph_len!=0].reset_index(drop=True)
    
    paragraph_agg_df = pd.concat([df[['id','paragraph_len']].groupby(['id']).agg(AGGREGATIONS), 
                                  df[['id','paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1) 
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df

def product_to_keys(logs, essays):
    essays['product_len'] = essays.essay.str.len()
    tmp_df = logs[logs.activity.isin(['Input', 'Remove/Cut'])].groupby(['id']).agg({'activity': 'count'}).reset_index().rename(columns={'activity': 'keys_pressed'})
    essays = essays.merge(tmp_df, on='id', how='left')
    essays['product_to_keys'] = essays['product_len'] / essays['keys_pressed']
    return essays[['id', 'product_to_keys']]

def get_keys_pressed_per_second(logs):
    temp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(['id']).agg(keys_pressed=('event_id', 'count')).reset_index()
    temp_df_2 = logs.groupby(['id']).agg(min_down_time=('down_time', 'min'), max_up_time=('up_time', 'max')).reset_index()
    temp_df = temp_df.merge(temp_df_2, on='id', how='left')
    temp_df['keys_per_second'] = temp_df['keys_pressed'] / ((temp_df['max_up_time'] - temp_df['min_down_time']) / 1000)
    return temp_df[['id', 'keys_per_second']]


def preprocessing_silverbullet(data_path, args):
    train_logs    = pl.scan_csv(data_path)
    train_feats   = dev_feats(train_logs)
    train_feats   = train_feats.collect().to_pandas()
    

    print('< Essay Reconstruction >')
    train_logs             = train_logs.collect().to_pandas()
    train_essays           = get_essay_df(train_logs)
    train_feats            = train_feats.merge(word_feats(train_essays), on='id', how='left')
    train_feats            = train_feats.merge(sent_feats(train_essays), on='id', how='left')
    train_feats            = train_feats.merge(parag_feats(train_essays), on='id', how='left')
    train_feats            = train_feats.merge(get_keys_pressed_per_second(train_logs), on='id', how='left')
    train_feats            = train_feats.merge(product_to_keys(train_logs, train_essays), on='id', how='left')

    print('< Mapping >')
    test_ids = train_feats['id'].values
    testin_x = train_feats.drop(['id'], axis=1)
    if args["exp_type"] == 'behavioral':
        testin_x = testin_x[lista_behavioral]
    
    print(f'Number of features: {len(testin_x.columns)}')

    return testin_x,test_ids


import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
from statsmodels.tsa.ar_model import AutoReg
import random
from tqdm import tqdm

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
seed_all(42)

def get_lin_reg_coef(data):
    x_values = np.array([x for x in range(1, len(data)+1)])
    model = LinearRegression()
    model.fit(x_values.reshape(-1, 1), data)
    return model.coef_[0]

def get_ar_params(data):
    data = data.values

    model = AutoReg(data, lags=1)
    trained_model = model.fit()
    return trained_model.params

def get_shannon_entropy(data):
    entropy_data = [x/sum(data) for x in data]

    s_entropy = 0

    for p in entropy_data:
        if p > 0:
            s_entropy += p * math.log(p, 2)

    return -s_entropy

def get_shannon_jensen_div(data):
    data = np.array([x/sum(data) for x in data])
    uniform_dist = np.array([1/len(data) for x in data])
    return distance.jensenshannon(data, uniform_dist)

def get_extremes(data):
    data = data.values
    diffs = [data[i+1] - data[i] for i in range(len(data)-1)]
    extreme_count = 0

    for i in range(len(diffs)-1):
        if (diffs[i] < 0 and diffs[i+1] < 0) or (diffs[i] > 0 and diffs[i+1] > 0):
            pass
        else:
            extreme_count += 1

    return extreme_count

def get_avg_recurrence(data):

    total = 0
    count = 0
    last_non_0 = -1

    for idx in range(0, len(data)):
        if data[idx] != 0:
            total += idx - last_non_0 - 1
            count += 1
            last_non_0 = idx

    return total / count

def get_stddev_recurrence(data):
    data = data.values

    recurrences = []
    last_non_0 = -1

    for idx in range(0, len(data)):
        if data[idx] != 0:
            recurrences.append(idx - last_non_0 - 1)
            last_non_0 = idx

    return np.std(recurrences)

def get_cursor_back(data):
    return len([x for x in data.values if x < 0])

def count_bursts(data):
    data = data.values

    burst_count = 0

    burst_start = 0

    pause_total = 0

    pause_count = 0

    bursts = []

    for i in range(1, len(data)):
        if data[i] > 0.01:
            pause_total += data[i]
            pause_count += 1
            if i - burst_start - 1 > 0:
                burst_count += 1
                bursts.append(data[i])

            burst_start = i

    return [burst_count, pause_total/pause_count]

def feature_engineer(df):
    new_df = pd.DataFrame({"id" : list(df.groupby("id").groups.keys())})
    df_grouped = df.groupby("id")

    a = np.array(df_grouped.apply(lambda x: count_bursts(x["diffs_seconds"])).values.tolist())

    new_df["mean_pause_duration"] = a[:,0] #keystrokes
    new_df["burst_count"] = a[:,1] #keystrokes

    new_df["verbosity"] = df_grouped.size().values #keystrokes
    backspace_df = df.groupby(["up_event", "id"]).size()["Backspace"] #keystrokes
    new_df = pd.merge(new_df, backspace_df.rename("backspaces"), on="id", how="left")

    new_df["word_count"] = df_grouped["word_count"].last().values #semantic

    period_df = df.groupby(["up_event", "id"]).size()["."] #semantic
    new_df = pd.merge(new_df, period_df.rename("sent_count"), on="id", how="left")

    enter_df = df.groupby(["up_event", "id"]).size()["Enter"] #semantic
    new_df = pd.merge(new_df, enter_df.rename("paragraph_count"), on="id", how="left")

    nonprod_df = df.groupby(["activity", "id"]).size()["Nonproduction"] #keystrokes
    new_df = pd.merge(new_df, nonprod_df.rename("Nonproduction"), on="id", how="left")

    new_df["avg_keystroke_speed"] = new_df["verbosity"] / df_grouped["time_elapsed"].tail(1).values #keystrokes

    ar_60 = np.array(df_grouped.apply(lambda x: get_ar_params(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values.tolist()) #keystrokes
    ar_60_1 = ar_60[:,0]
    ar_60_2 = ar_60[:,1]
    new_df["ar_60_1"] = ar_60_1
    new_df["ar_60_2"] = ar_60_2

    ar_30 = np.array(df_grouped.apply(lambda x: get_ar_params(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values.tolist())
    ar_30_1 = ar_30[:,0]
    ar_30_2 = ar_30[:,1]
    new_df["ar_30_1"] = ar_30_1
    new_df["ar_30_2"] = ar_30_2

    new_df["largest_insert"] = df_grouped["word_diffs"].max().values #semantic
    new_df["largest_delete"] = df_grouped["word_diffs"].min().values #semantic

    new_df["backspaces"].fillna(0, inplace=True)
    new_df["largest_latency"] = df_grouped["diffs"].max().values
    new_df["smallest_latency"] = df_grouped["diffs"].min().values
    new_df["median_latency"] = df_grouped["diffs"].median().values
    new_df["first_pause"] = df.groupby("id").diffs_seconds.first().values
    new_df["pause_0.5"] = df[(df["diffs_seconds"] > 0.5) & (df["diffs_seconds"] < 1)].groupby("id").size().values
    new_df["pause_1"] = df[(df["diffs_seconds"] > 1) & (df["diffs_seconds"] < 1.5)].groupby("id").size().values
    new_df["pause_1.5"] = df[(df["diffs_seconds"] > 1.5) & (df["diffs_seconds"] < 2)].groupby("id").size().values
    pause_2_df = df[(df["diffs_seconds"] > 2) & (df["diffs_seconds"] < 3)].groupby("id").size()
    pause_3_df = df[df["diffs_seconds"] > 3].groupby("id").size()
    new_df = pd.merge(new_df, pause_2_df.rename("pause_2"), on="id", how="left")
    new_df = pd.merge(new_df, pause_3_df.rename("pause_3"), on="id", how="left")
    new_df["pause_2"].fillna(0, inplace=True)
    new_df["pause_3"].fillna(0, inplace=True)

    new_df["Slope_Degree_60"] = df_grouped.apply(lambda x: get_lin_reg_coef(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values
    new_df["Entropy_60"] = df_grouped.apply(lambda x: get_shannon_entropy(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values
    new_df["Degree_Uniformity_60"] = df_grouped.apply(lambda x: get_shannon_jensen_div(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values
    new_df["Local_Extremes_60"] = df_grouped.apply(lambda x: get_extremes(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values
    new_df["Average_Recurrence_60"] = df_grouped.apply(lambda x: get_avg_recurrence(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values
    new_df["StdDev_Recurrence_60"] = df_grouped.apply(lambda x: get_stddev_recurrence(x["window_60_sec_idx"].value_counts().reindex(range(max(x["window_60_sec_idx"])+1), fill_value=0))).values

    new_df["Slope_Degree_30"] = df_grouped.apply(lambda x: get_lin_reg_coef(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values
    new_df["Entropy_30"] = df_grouped.apply(lambda x: get_shannon_entropy(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values
    new_df["Degree_Uniformity_30"] = df_grouped.apply(lambda x: get_shannon_jensen_div(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values
    new_df["Local_Extremes_30"] = df_grouped.apply(lambda x: get_extremes(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values
    new_df["Average_Recurrence_30"] = df_grouped.apply(lambda x: get_avg_recurrence(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values
    new_df["StdDev_Recurrence_30"] = df_grouped.apply(lambda x: get_stddev_recurrence(x["window_30_sec_idx"].value_counts().reindex(range(max(x["window_30_sec_idx"])+1), fill_value=0))).values

    new_df["StDev_Events_60"] = df_grouped.apply(lambda x: x["window_60_sec_idx"].value_counts().reindex(x["window_60_sec_idx"].unique(), fill_value=0).std()).values
    new_df["StDev_Events_30"] = df_grouped.apply(lambda x: x["window_30_sec_idx"].value_counts().reindex(x["window_30_sec_idx"].unique(), fill_value=0).std()).values

    new_df["Cursor_Back_Count"] = df_grouped.apply(lambda x: get_cursor_back(x["curpos_diffs"])).values
    new_df["Word_Back_Count"] = df_grouped.apply(lambda x: get_cursor_back(x["word_diffs"])).values #semantic

    return new_df


lista_semantic = ['word_count', 'sent_count', 'paragraph_count','largest_insert',
                   'largest_delete','Word_Back_Count']


def preprocessing_barreto(data_path, args):


    train_data = pd.read_csv(data_path)

    train_data.sort_values(['id', 'up_time'], inplace=True)

    train_data['diffs'] = train_data.groupby(['id'])['up_time'].transform(lambda x: x.diff())

    train_data.sort_index(inplace=True)
    train_data["diffs_seconds"] = train_data["diffs"] / 1000

    train_data["time_elapsed"] = train_data.groupby("id")["diffs_seconds"].cumsum()
    train_data["time_elapsed"].fillna(0, inplace=True)

    train_data['curpos_diffs'] = train_data.groupby(['id'])['cursor_position'].transform(lambda x: x.diff())
    train_data['word_diffs'] = train_data.groupby(['id'])['word_count'].transform(lambda x: x.diff()) #semantic

    train_data["window_60_sec_idx"] = train_data["time_elapsed"].apply(lambda x: math.floor(x / 60.0))
    train_data["window_30_sec_idx"] = train_data["time_elapsed"].apply(lambda x: math.floor(x / 30.0))

    full_df = feature_engineer(train_data)

    if args["exp_type"] == 'behavorial':
        full_df = full_df.drop(lista_semantic, axis=1)

    print(full_df.columns)

    return full_df.drop('id', axis=1), full_df['id'].values


#cargar el modelo
data_path     = '/kaggle/input/linking-writing-processes-to-writing-quality/'

if config['model_type'] == 'silver_bullet':
    testin_x,test_ids = preprocessing_silverbullet(data_path+'test_logs.csv', config)
    model_path = datamodels_path+f"fold_{fold}/"
    import lightgbm as lgb
    # Ruta al archivo del modelo
    model = lgb.Booster(model_file=model_path+"/model.txt")
    y_pred = model.predict(testin_x)
    
    sub = pd.DataFrame({'id': test_ids, 'score': y_pred})
    sub.to_csv('submission.csv', index=False)

if config['model_type'] == 'barreto':
    testin_x,test_ids = preprocessing_barreto(data_path+'test_logs.csv', config)
    model_path = datamodels_path+f"fold_{fold}/"
    import lightgbm as lgb
    # Ruta al archivo del modelo
    model = lgb.Booster(model_file=model_path+"/model.txt")
    y_pred = model.predict(testin_x)
    
    sub = pd.DataFrame({'id': test_ids, 'score': y_pred})
    sub.to_csv('submission.csv', index=False)



















