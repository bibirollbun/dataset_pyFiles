import gc
import os
import itertools
import pickle
import re
import time

import warnings
warnings.filterwarnings('ignore')

from random import choice, choices
from functools import reduce
from tqdm import tqdm
from itertools import cycle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
%matplotlib inline
import copy

from collections import Counter
from functools import reduce
from tqdm import tqdm
from itertools import cycle
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
from sklearn import decomposition
from sklearn import tree

import lightgbm as lgb
import xgboost as xgb

import optuna
import polars as pl



DATA_DIR = '/kaggle/input/linking-writing-processes-to-writing-quality/'
df_train_scores = pd.read_csv(DATA_DIR + "train_scores.csv")
df_train_logs = pd.read_csv(DATA_DIR + "train_logs.csv")
df_test_logs = pd.read_csv(DATA_DIR + "test_logs.csv")


import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re

num_cols = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
text_changes = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']


def count_by_values(df, colname, values):
    """Count occurrences of specific values in a column for each id"""
    fts = df[['id']].drop_duplicates().reset_index(drop=True)
    
    for i, value in enumerate(values):
        tmp_df = df[df[colname] == value].groupby('id').size().reset_index(name=f'{colname}_{i}_cnt')
        fts = fts.merge(tmp_df, on='id', how='left')
        fts[f'{colname}_{i}_cnt'] = fts[f'{colname}_{i}_cnt'].fillna(0)
    
    return fts


def dev_feats(df):
    
    print("< Count by values features >")
    
    feats = count_by_values(df, 'activity', activities)
    feats = feats.merge(count_by_values(df, 'text_change', text_changes), on='id', how='left')
    feats = feats.merge(count_by_values(df, 'down_event', events), on='id', how='left')
    feats = feats.merge(count_by_values(df, 'up_event', events), on='id', how='left')

    print("< Input words stats features >")

    temp = df[(~df['text_change'].str.contains('=>', na=False)) & (df['text_change'] != 'NoChange')].copy()
    temp = temp.groupby('id')['text_change'].apply(lambda x: ''.join(x.astype(str))).reset_index()
    temp['text_change'] = temp['text_change'].str.findall(r'q+')
    
    temp['input_word_count'] = temp['text_change'].apply(len)
    temp['input_word_length_mean'] = temp['text_change'].apply(lambda x: np.mean([len(i) for i in x]) if len(x) > 0 else 0)
    temp['input_word_length_max'] = temp['text_change'].apply(lambda x: np.max([len(i) for i in x]) if len(x) > 0 else 0)
    temp['input_word_length_std'] = temp['text_change'].apply(lambda x: np.std([len(i) for i in x]) if len(x) > 0 else 0)
    temp['input_word_length_median'] = temp['text_change'].apply(lambda x: np.median([len(i) for i in x]) if len(x) > 0 else 0)
    temp['input_word_length_skew'] = temp['text_change'].apply(lambda x: skew([len(i) for i in x]) if len(x) > 0 else 0)
    
    temp = temp.drop('text_change', axis=1)
    feats = feats.merge(temp, on='id', how='left')

    print("< Numerical columns features >")

    agg_dict = {col: ['mean', 'std', 'median', 'min', 'max'] for col in num_cols}
    agg_dict['action_time'].append('sum')
    
    temp = df.groupby('id').agg(agg_dict)
    temp.columns = ['_'.join(col).strip() for col in temp.columns.values]
    temp = temp.reset_index()
    
    # Add quantile features
    quantile_feats = df.groupby('id')[num_cols].quantile(0.5).reset_index()
    quantile_feats.columns = ['id'] + [f'{col}_quantile' for col in num_cols]
    temp = temp.merge(quantile_feats, on='id', how='left')
    
    feats = feats.merge(temp, on='id', how='left')

    print("< Categorical columns features >")
    
    temp = df.groupby('id')[['activity', 'down_event', 'up_event', 'text_change']].nunique().reset_index()
    feats = feats.merge(temp, on='id', how='left')

    print("< Idle time features >")

    temp = df.copy()
    temp['up_time_lagged'] = temp.groupby('id')['up_time'].shift(1)
    temp['time_diff'] = (abs(temp['down_time'] - temp['up_time_lagged']) / 1000).fillna(0)
    temp = temp[temp['activity'].isin(['Input', 'Remove/Cut'])]
    
    idle_agg = temp.groupby('id').agg(
        inter_key_largest_lantency=('time_diff', 'max'),
        inter_key_median_lantency=('time_diff', 'median'),
        mean_pause_time=('time_diff', 'mean'),
        std_pause_time=('time_diff', 'std'),
        total_pause_time=('time_diff', 'sum'),
        pauses_half_sec=('time_diff', lambda x: ((x > 0.5) & (x < 1)).sum()),
        pauses_1_sec=('time_diff', lambda x: ((x > 1) & (x < 1.5)).sum()),
        pauses_1_half_sec=('time_diff', lambda x: ((x > 1.5) & (x < 2)).sum()),
        pauses_2_sec=('time_diff', lambda x: ((x > 2) & (x < 3)).sum()),
        pauses_3_sec=('time_diff', lambda x: (x > 3).sum())
    ).reset_index()
    
    feats = feats.merge(idle_agg, on='id', how='left')
    
    print("< P-bursts features >")

    temp = df.copy()
    temp['up_time_lagged'] = temp.groupby('id')['up_time'].shift(1)
    temp['time_diff'] = (abs(temp['down_time'] - temp['up_time_lagged']) / 1000).fillna(0)
    temp = temp[temp['activity'].isin(['Input', 'Remove/Cut'])]
    temp['time_diff_bool'] = temp['time_diff'] < 2
    
    # Calculate P-bursts using run-length encoding
    temp['burst_group'] = (temp['time_diff_bool'] != temp.groupby('id')['time_diff_bool'].shift()).cumsum()
    temp['P-bursts'] = temp.groupby(['id', 'burst_group'])['time_diff_bool'].transform('size')
    temp = temp[temp['time_diff_bool']].dropna()
    
    pburst_agg = temp.groupby('id')['P-bursts'].agg(['mean', 'std', 'count', 'median', 'max', 'first', 'last']).reset_index()
    pburst_agg.columns = ['id', 'P-bursts_mean', 'P-bursts_std', 'P-bursts_count', 'P-bursts_median', 'P-bursts_max', 'P-bursts_first', 'P-bursts_last']
    
    feats = feats.merge(pburst_agg, on='id', how='left')

    print("< R-bursts features >")

    temp = df[df['activity'].isin(['Input', 'Remove/Cut'])].copy()
    temp['activity_bool'] = temp['activity'].isin(['Remove/Cut'])
    
    # Calculate R-bursts using run-length encoding
    temp['burst_group'] = (temp['activity_bool'] != temp.groupby('id')['activity_bool'].shift()).cumsum()
    temp['R-bursts'] = temp.groupby(['id', 'burst_group'])['activity_bool'].transform('size')
    temp = temp[temp['activity_bool']].dropna()
    
    rburst_agg = temp.groupby('id')['R-bursts'].agg(['mean', 'std', 'median', 'max', 'first', 'last']).reset_index()
    rburst_agg.columns = ['id', 'R-bursts_mean', 'R-bursts_std', 'R-bursts_median', 'R-bursts_max', 'R-bursts_first', 'R-bursts_last']
    
    feats = feats.merge(rburst_agg, on='id', how='left')
    
    return feats


def train_valid_split(data_x, data_y, train_idx, valid_idx):
    x_train = data_x.iloc[train_idx]
    y_train = data_y[train_idx]
    x_valid = data_x.iloc[valid_idx]
    y_valid = data_y[valid_idx]
    return x_train, y_train, x_valid, y_valid


def evaluate(data_x, data_y, model, random_state=42, n_splits=5, test_x=None):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    test_y = np.zeros(len(data_x)) if (test_x is None) else np.zeros((len(test_x), n_splits))
    for i, (train_index, valid_index) in enumerate(skf.split(data_x, data_y.astype(str))):
        train_x, train_y, valid_x, valid_y = train_valid_split(data_x, data_y, train_index, valid_index)
        model.fit(train_x, train_y)
        if test_x is None:
            test_y[valid_index] = model.predict(valid_x)
        else:
            test_y[:, i] = model.predict(test_x)
    return test_y if (test_x is None) else np.mean(test_y, axis=1)


# Pandas FE & Helper Functions
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
    df = df[df.activity != 'Nonproduction']
    temp = df.groupby('id').apply(lambda x: reconstruct_essay(x[['activity', 'cursor_position', 'text_change']]))
    essay_df = pd.DataFrame({'id': df['id'].unique().tolist()})
    essay_df = essay_df.merge(temp.rename('essay'), on='id')
    return essay_df


def word_feats(df):
    essay_df = df.copy()
    df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!', x))
    df = df.explode('word')
    df['word_len'] = df['word'].apply(lambda x: len(x))
    df = df[df['word_len'] != 0]

    word_agg_df = df[['id', 'word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df


def sent_feats(df):
    df = df.copy()
    df['sent'] = df['essay'].apply(lambda x: re.split('\\.|\\?|\\!', x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n', '').strip())
    # Number of characters in sentences
    df['sent_len'] = df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    df['sent_word_count'] = df['sent'].apply(lambda x: len(x.split(' ')))
    df = df[df.sent_len != 0].reset_index(drop=True)

    sent_agg_df = pd.concat([df[['id', 'sent_len']].groupby(['id']).agg(AGGREGATIONS), 
                             df[['id', 'sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count": "sent_count"})
    return sent_agg_df


def parag_feats(df):
    df = df.copy()
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    # Number of characters in paragraphs
    df['paragraph_len'] = df['paragraph'].apply(lambda x: len(x))
    # Number of words in paragraphs
    df['paragraph_word_count'] = df['paragraph'].apply(lambda x: len(x.split(' ')))
    df = df[df.paragraph_len != 0].reset_index(drop=True)
    
    paragraph_agg_df = pd.concat([df[['id', 'paragraph_len']].groupby(['id']).agg(AGGREGATIONS), 
                                  df[['id', 'paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1)
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count": "paragraph_count"})
    return paragraph_agg_df


def product_to_keys(logs, essays):
    essays = essays.copy()
    essays['product_len'] = essays.essay.str.len()
    tmp_df = logs[logs.activity.isin(['Input', 'Remove/Cut'])].groupby(['id']).agg({'activity': 'count'}).reset_index().rename(columns={'activity': 'keys_pressed'})
    essays = essays.merge(tmp_df, on='id', how='left')
    essays['product_to_keys'] = essays['product_len'] / essays['keys_pressed']
    return essays[['id', 'product_to_keys']]


def get_word_tfidf_features(train_essays, test_essays, ngram_range=(1, 1), max_features=500):
    """
    Word-level TF-IDF features
    
    Args:
        train_essays: DataFrame with 'essay' column
        test_essays: DataFrame with 'essay' column
        ngram_range: tuple, ngram range for TF-IDF (default: (1,1) for unigrams)
        max_features: int, maximum number of features to extract
    
    Returns:
        train_df, test_df: DataFrames with TF-IDF features
    """
    print(f"< Word-level TF-IDF features (ngram_range={ngram_range}, max_features={max_features}) >")
    
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2,
        sublinear_tf=True
    )
    
    # Fit on training data
    tfidf_matrix_train = vectorizer.fit_transform(train_essays['essay'].astype(str))
    tfidf_df_train = pd.DataFrame(
        tfidf_matrix_train.toarray(), 
        columns=[f'word_tfidf_{col}' for col in vectorizer.get_feature_names_out()]
    )
    tfidf_df_train['id'] = train_essays['id'].values
    
    # Transform test data
    tfidf_matrix_test = vectorizer.transform(test_essays['essay'].astype(str))
    tfidf_df_test = pd.DataFrame(
        tfidf_matrix_test.toarray(), 
        columns=[f'word_tfidf_{col}' for col in vectorizer.get_feature_names_out()]
    )
    tfidf_df_test['id'] = test_essays['id'].values
    
    print(f"   Generated {len(tfidf_df_train.columns)-1} word TF-IDF features")
    
    return tfidf_df_train, tfidf_df_test


def get_keys_pressed_per_second(logs):
    temp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(['id']).agg(keys_pressed=('event_id', 'count')).reset_index()
    temp_df_2 = logs.groupby(['id']).agg(min_down_time=('down_time', 'min'), max_up_time=('up_time', 'max')).reset_index()
    temp_df = temp_df.merge(temp_df_2, on='id', how='left')
    temp_df['keys_per_second'] = temp_df['keys_pressed'] / ((temp_df['max_up_time'] - temp_df['min_down_time']) / 1000)
    return temp_df[['id', 'keys_per_second']]


def get_word_tfidf_features(train_essays, test_essays, ngram_range=(1, 1), max_features=500):
    """
    Word-level TF-IDF features
    
    Args:
        train_essays: DataFrame with 'essay' column
        test_essays: DataFrame with 'essay' column
        ngram_range: tuple, ngram range for TF-IDF (default: (1,1) for unigrams)
        max_features: int, maximum number of features to extract
    
    Returns:
        train_df, test_df: DataFrames with TF-IDF features
    """
    print(f"< Word-level TF-IDF features (ngram_range={ngram_range}, max_features={max_features}) >")
    
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2,
        sublinear_tf=True
    )
    
    # Fit on training data
    tfidf_matrix_train = vectorizer.fit_transform(train_essays['essay'].astype(str))
    tfidf_df_train = pd.DataFrame(
        tfidf_matrix_train.toarray(), 
        columns=[f'word_tfidf_{col}' for col in vectorizer.get_feature_names_out()]
    )
    tfidf_df_train['id'] = train_essays['id'].values
    
    # Transform test data
    tfidf_matrix_test = vectorizer.transform(test_essays['essay'].astype(str))
    tfidf_df_test = pd.DataFrame(
        tfidf_matrix_test.toarray(), 
        columns=[f'word_tfidf_{col}' for col in vectorizer.get_feature_names_out()]
    )
    tfidf_df_test['id'] = test_essays['id'].values
    
    print(f"   Generated {len(tfidf_df_train.columns)-1} word TF-IDF features")
    
    return tfidf_df_train, tfidf_df_test


def get_char_tfidf_features(train_essays, test_essays, n_components=100, max_features=500):
    """
    Character-level TF-IDF features with dimensionality reduction using SVD
    
    Args:
        train_essays: DataFrame with 'essay' column
        test_essays: DataFrame with 'essay' column
        n_components: int, number of SVD components (default: 100)
        max_features: int, maximum number of features before SVD (default: 500)
    
    Returns:
        train_df, test_df: DataFrames with character TF-IDF features
    """
    print(f"< Character-level TF-IDF features (n_components={n_components}, max_features={max_features}) >")
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
        analyzer='char',
        sublinear_tf=True
    )
    
    # Fit and transform training data
    train_tfidf = vectorizer.fit_transform(train_essays['essay'].astype(str))
    test_tfidf = vectorizer.transform(test_essays['essay'].astype(str))
    
    # Apply SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)
    
    # Create DataFrames
    train_df = pd.DataFrame(
        train_svd, 
        columns=[f'char_tfidf_{i}' for i in range(n_components)]
    )
    train_df['id'] = train_essays['id'].values
    
    test_df = pd.DataFrame(
        test_svd, 
        columns=[f'char_tfidf_{i}' for i in range(n_components)]
    )
    test_df['id'] = test_essays['id'].values
    
    print(f"   Generated {n_components} character TF-IDF components (explained variance: {svd.explained_variance_ratio_.sum():.3f})")
    
    return train_df, test_df


def split_para(text):
    return re.split(r'\n', text)


def get_essay_features_pandas(df_essays, debug=""):
    """
    Extract advanced essay features using pure pandas operations.
    Converted from polars to pandas for compatibility.
    """
    
    df_essays = df_essays.copy()
    df_essays['essay'] = df_essays['essay'].str.replace(r'q\\q', 'q q', regex=True)  
    df_essays['essay'] = df_essays['essay'].str.replace(r'\n+\.', '.\n\n', regex=True)
    df_essays['essay'] = df_essays['essay'].str.replace(r'\n+[\.,]\s*', '.\n\n', regex=True)
    
    para_df = df_essays[["id", "essay"]].copy()
    para_df['essay'] = para_df['essay'].apply(lambda x: split_para(x))
    para_df = para_df.explode('essay')  ## split in paras
    
    if debug != "":
        display(para_df[para_df['id'] == debug].head(20))
        
    para_df['is_valid'] = para_df['essay'].apply(lambda x: 'q' in str(x))  ## check if character in para
    para_df = para_df[para_df['is_valid']].copy()
    
    para_df['para_len'] = para_df['essay'].apply(lambda x: len(str(x).split(' ')))
    para_df['nth_para'] = 1
    para_df['nth_para'] = (para_df.groupby('id')['nth_para'].cumsum() - 1).clip(0, np.inf).astype(int)
    
    if debug != "":
        display(para_df[para_df['id'] == debug].head(20))
        
    para_df['essay'] = para_df['essay'].apply(lambda x: str(x).split(" "))  ##split on the basis of spaces
    para_df = para_df.explode("essay")  ##explode word wise
    
    para_df["word_len"] = para_df["essay"].apply(lambda x: len(str(x)))
    para_df["is_valid_word"] = para_df["essay"].apply(lambda x: (len(str(x)) != 0) & ('q' in str(x)))
    para_df = para_df[para_df["is_valid_word"]].copy()

    if debug != "":
        display(para_df[para_df['id'] == debug].head(20))
    
    ## get factual instances
    para_df["is_question"] = para_df["essay"].apply(lambda x: ('?' in str(x)))
    para_df["is_exclamation"] = para_df["essay"].apply(lambda x: ('!' in str(x)))
    para_df["is_colon"] = para_df["essay"].apply(lambda x: (':' in str(x)) or (";" in str(x)))
    para_df["is_comma"] = para_df["essay"].apply(lambda x: (',' in str(x)))
    para_df["is_reference"] = para_df["essay"].apply(lambda x: ('"' in str(x)))
    para_df["is_fact"] = para_df["essay"].apply(lambda x: ('%' in str(x)) or ('-' in str(x)))
    
    if debug != "":
        display(para_df[para_df['id'] == debug].head(20))        
    
    ## to handle one length sentences
    para_df["is_valid_sentence"] = para_df["essay"].apply(lambda x: ('.' in str(x)) or ('?' in str(x)) or ('!' in str(x)))
    para_df["previous_valid"] = para_df.groupby("id")["is_valid_sentence"].shift()

    para_df["nth_sentence"] = para_df.groupby("id")["is_valid_sentence"].cumsum()
    para_df["nth_sentence"] = para_df.groupby('id')["nth_sentence"].shift().fillna(0).astype(int)
    
    ### create a column of two words together in a sentence
    para_df['two_words'] = para_df['essay'] + ' ' + para_df.groupby(['id', 'nth_sentence'])['essay'].shift(-1).fillna('')
    ### create a column of three words together in a sentence
    para_df['three_words'] = para_df['two_words'] + ' ' + para_df.groupby(['id', 'nth_sentence'])['essay'].shift(-2).fillna('')
    
    if debug != "":
        display(para_df[para_df['id'] == debug].head(20))
    
    # Calculate paragraph-level features using pandas
    para_sentence_feats = pd.DataFrame()
    
    grouped = para_df.groupby('id')
    
    # Basic counts
    para_sentence_feats['questions_count'] = grouped['is_question'].sum()
    para_sentence_feats['exclamations_count'] = grouped['is_exclamation'].sum()
    para_sentence_feats['colons_count'] = grouped['is_colon'].sum()
    para_sentence_feats['reference_count'] = grouped['is_reference'].sum()
    para_sentence_feats['comma_count'] = grouped['is_comma'].sum()
    para_sentence_feats['factual_count'] = grouped['is_fact'].sum()
    
    # First and last paragraph questions
    para_sentence_feats['first_paragraph_questions_count'] = grouped.apply(
        lambda x: x[x['nth_para'] == 0]['is_question'].sum()
    )
    para_sentence_feats['last_paragraph_questions_count'] = grouped.apply(
        lambda x: x[x['nth_para'] == x['nth_para'].max()]['is_question'].sum()
    )
    
    # Unique counts
    para_sentence_feats['valid_paragraphs_count'] = grouped['nth_para'].nunique()
    para_sentence_feats['valid_sentences_count'] = grouped['nth_sentence'].nunique()
    para_sentence_feats['valid_words_count'] = grouped['word_len'].count()
    
    # Para length features
    para_sentence_feats['one_length_paras'] = grouped.apply(lambda x: (x['para_len'] == 1).sum())
    para_sentence_feats['two_length_paras'] = grouped.apply(lambda x: (x['para_len'] == 2).sum())
    para_sentence_feats['three_length_paras'] = grouped.apply(lambda x: (x['para_len'] == 3).sum())
    para_sentence_feats['four_length_paras'] = grouped.apply(lambda x: (x['para_len'] == 4).sum())
    
    # Ratio features
    para_sentence_feats['sentences_per_paragraph'] = (
        grouped['nth_sentence'].nunique() / grouped['nth_para'].nunique()
    )
    para_sentence_feats['words_per_sentences'] = (
        grouped.size() / grouped['nth_sentence'].nunique()
    )
    para_sentence_feats['words_per_paragraphs'] = (
        grouped.size() / grouped['nth_para'].nunique()
    )
    para_sentence_feats['paragraph_per_sentences'] = (
        grouped['nth_para'].nunique() / grouped['nth_sentence'].nunique()
    )
    para_sentence_feats['sentences_per_words'] = (
        grouped['nth_sentence'].nunique() / grouped.size()
    )
    para_sentence_feats['paragraphs_per_words'] = (
        grouped['nth_para'].nunique() / grouped.size()
    )
    
    # Word length in first para
    for length in range(1, 10):
        para_sentence_feats[f'word_len{length}_first_para_count'] = grouped.apply(
            lambda x: ((x['nth_para'] == 1) & (x['word_len'] == length)).sum()
        )
    
    # Paragraph word counts
    for idx, name in enumerate(['first', 'second', 'third', 'fourth', 'fifth']):
        para_sentence_feats[f'{name}_paragraph_word_len'] = grouped.apply(
            lambda x: (x['nth_para'] == idx).sum()
        )
    para_sentence_feats['last_paragraph_word_len'] = grouped.apply(
        lambda x: (x['nth_para'] == x['nth_para'].max()).sum()
    )
    
    # Paragraph char counts
    for idx, name in enumerate(['first', 'second', 'third', 'fourth', 'fifth']):
        para_sentence_feats[f'{name}_paragraph_char_len'] = grouped.apply(
            lambda x: x[x['nth_para'] == idx]['word_len'].sum()
        )
    para_sentence_feats['last_paragraph_char_len'] = grouped.apply(
        lambda x: x[x['nth_para'] == x['nth_para'].max()]['word_len'].sum()
    )
    
    # Paragraph sentence counts
    for idx, name in enumerate(['first', 'second', 'third', 'fourth', 'fifth']):
        para_sentence_feats[f'{name}_paragraph_sentence_count'] = grouped.apply(
            lambda x: x[x['nth_para'] == idx]['nth_sentence'].nunique()
        )
    para_sentence_feats['last_paragraph_sentence_count'] = grouped.apply(
        lambda x: x[x['nth_para'] == x['nth_para'].max()]['nth_sentence'].count()
    )
    
    # Cumulative sentence counts
    para_sentence_feats['first_paragraph_sentence_count_cum'] = grouped.apply(
        lambda x: x[x['nth_para'] == 0]['nth_sentence'].max() if len(x[x['nth_para'] == 0]) > 0 else 0
    )
    para_sentence_feats['second_paragraph_sentence_count_cum'] = grouped.apply(
        lambda x: x[x['nth_para'] <= 1]['nth_sentence'].max() if len(x[x['nth_para'] <= 1]) > 0 else 0
    )
    para_sentence_feats['third_paragraph_sentence_count_cum'] = grouped.apply(
        lambda x: x[x['nth_para'] <= 2]['nth_sentence'].max() if len(x[x['nth_para'] <= 2]) > 0 else 0
    )
    para_sentence_feats['fourth_paragraph_sentence_count_cum'] = grouped.apply(
        lambda x: x[x['nth_para'] <= 3]['nth_sentence'].max() if len(x[x['nth_para'] <= 3]) > 0 else 0
    )
    para_sentence_feats['fifth_paragraph_sentence_count_cum'] = grouped.apply(
        lambda x: x[x['nth_para'] <= 4]['nth_sentence'].max() if len(x[x['nth_para'] <= 4]) > 0 else 0
    )
    
    # Sentence word counts
    for idx, name in enumerate(['first', 'second', 'third', 'fourth', 'fifth']):
        para_sentence_feats[f'{name}_sentence_word_count'] = grouped.apply(
            lambda x: (x['nth_sentence'] == idx).sum()
        )
    para_sentence_feats['last_sentence_word_count'] = grouped.apply(
        lambda x: (x['nth_sentence'] == x['nth_sentence'].max()).sum()
    )
    
    # Sentence char counts
    for idx, name in enumerate(['first', 'second', 'third', 'fourth', 'fifth']):
        para_sentence_feats[f'{name}_sentence_char_count'] = grouped.apply(
            lambda x: x[x['nth_sentence'] == idx]['word_len'].sum()
        )
    para_sentence_feats['last_sentence_char_count'] = grouped.apply(
        lambda x: x[x['nth_sentence'] == x['nth_sentence'].max()]['word_len'].sum()
    )
    
    para_sentence_feats = para_sentence_feats.reset_index()
    
    if debug != "":
        display(para_sentence_feats[para_sentence_feats['id'] == debug])
    
    # Calculate nth words features
    nth_words_df = para_df.groupby(['id', 'nth_sentence']).agg({
        'essay': 'first',
        'word_len': 'first',
        'nth_para': 'first',
        'two_words': 'first',
        'three_words': 'first'
    }).reset_index()
    
    nth_words_df.columns = ['id', 'nth_sentence', 'first_word', 'first_word_len', 
                            'nth_para', 'first_two_words', 'first_three_words']
    nth_words_df['sentence_length'] = para_df.groupby(['id', 'nth_sentence']).size().values
    
    # Calculate mode frequencies
    nth_words_features = pd.DataFrame()
    nth_grouped = nth_words_df.groupby('id')
    
    nth_words_features['max_freq_of_first_word'] = nth_grouped.apply(
        lambda x: (x['first_word'] == x['first_word'].mode().iloc[0] if len(x['first_word'].mode()) > 0 else None).sum() / len(x)
    )
    nth_words_features['max_freq_of_first_two_words'] = nth_grouped.apply(
        lambda x: (x['first_two_words'] == x['first_two_words'].mode().iloc[0] if len(x['first_two_words'].mode()) > 0 else None).sum() / len(x)
    )
    nth_words_features['max_freq_of_first_three_words'] = nth_grouped.apply(
        lambda x: (x['first_three_words'] == x['first_three_words'].mode().iloc[0] if len(x['first_three_words'].mode()) > 0 else None).sum() / len(x)
    )
    
    for word_len in range(1, 10):
        nth_words_features[f'count_first_len_{word_len}_words'] = nth_grouped.apply(
            lambda x: (x['first_word_len'] == word_len).sum() / len(x)
        )
    
    nth_words_features = nth_words_features.reset_index()
    
    if debug != "":
        display(nth_words_features[nth_words_features['id'] == debug])
    
    # Join features
    para_sentence_feats = para_sentence_feats.merge(nth_words_features, on='id', how='left')
    
    return para_sentence_feats, para_df, nth_words_features


# Data Processing
data_path = '/kaggle/input/linking-writing-processes-to-writing-quality/'

print("< Loading training data >")
train_logs = pd.read_csv(data_path + 'train_logs.csv')
train_feats = dev_feats(train_logs)

print('< Essay Reconstruction >')
train_essays = get_essay_df(train_logs)
train_feats = train_feats.merge(word_feats(train_essays), on='id', how='left')
train_feats = train_feats.merge(sent_feats(train_essays), on='id', how='left')
train_feats = train_feats.merge(parag_feats(train_essays), on='id', how='left')
train_feats = train_feats.merge(get_keys_pressed_per_second(train_logs), on='id', how='left')
train_feats = train_feats.merge(product_to_keys(train_logs, train_essays), on='id', how='left')

print('< Advanced Essay Features >')
para_sentence_feats, para_df, nth_words_features = get_essay_features_pandas(train_essays)
# Note: para_sentence_feats already includes numerical nth_words features merged inside the function
train_feats = train_feats.merge(para_sentence_feats, on='id', how='left')

print('< Word-level TF-IDF Features >')
# Load test essays first for TF-IDF
test_logs = pd.read_csv(data_path + 'test_logs.csv')
test_essays = get_essay_df(test_logs)

word_tfidf_train, word_tfidf_test = get_word_tfidf_features(
    train_essays, 
    test_essays,
    ngram_range=(1, 1),
    max_features=500
)

print('< Character-level TF-IDF Features >')
char_tfidf_train, char_tfidf_test = get_char_tfidf_features(
    train_essays,
    test_essays,
    n_components=100,
    max_features=500
)

train_feats = train_feats.merge(word_tfidf_train, on='id', how='left')
train_feats = train_feats.merge(char_tfidf_train, on='id', how='left')

print('< Mapping >')
train_scores = pd.read_csv(data_path + 'train_scores.csv')
data = train_feats.merge(train_scores, on='id', how='left')
x = data.drop(['id', 'score'], axis=1)
y = data['score'].values
print(f'Number of features: {len(x.columns)}')

print('< Testing Data >')
test_feats = dev_feats(test_logs)

test_feats = test_feats.merge(word_feats(test_essays), on='id', how='left')
test_feats = test_feats.merge(sent_feats(test_essays), on='id', how='left')
test_feats = test_feats.merge(parag_feats(test_essays), on='id', how='left')
test_feats = test_feats.merge(get_keys_pressed_per_second(test_logs), on='id', how='left')
test_feats = test_feats.merge(product_to_keys(test_logs, test_essays), on='id', how='left')

print('< Advanced Essay Features - Test >')
test_para_sentence_feats, test_para_df, test_nth_words_features = get_essay_features_pandas(test_essays)
test_feats = test_feats.merge(test_para_sentence_feats, on='id', how='left')
test_feats = test_feats.merge(word_tfidf_test, on='id', how='left')
test_feats = test_feats.merge(char_tfidf_test, on='id', how='left')

test_ids = test_feats['id'].values
test_x = test_feats.drop(['id'], axis=1)

print("\n< Feature Engineering Complete >")
print(f"Training features shape: {x.shape}")
print(f"Test features shape: {test_x.shape}")


len(test_x.columns)


# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD

# def get_char_tfidf_features(train_essays, test_essays, n_components=100):
#     """Character-level TF-IDF features"""
    
#     vectorizer = TfidfVectorizer(
#         ngram_range=(1, 2),
#         max_features=500,
#         min_df=1,
#         analyzer='char',
#         sublinear_tf=True
#     )
    
#     train_tfidf = vectorizer.fit_transform(train_essays['essay'])
#     test_tfidf = vectorizer.transform(test_essays['essay'])
    
#     svd = TruncatedSVD(n_components=n_components, random_state=42)
#     train_svd = svd.fit_transform(train_tfidf)
#     test_svd = svd.transform(test_tfidf)
    
#     train_df = pd.DataFrame(train_svd, columns=[f'char_tfidf_{i}' for i in range(n_compon_]()



# # Data Processing
# # data_path = '/kaggle/input/linking-writing-processes-to-writing-quality/'
# train_logs = pd.read_csv(data_path + 'train_logs.csv')
# train_feats = dev_feats(train_logs)
# test_logs = pd.read_csv(data_path + 'test_logs.csv')
# test_feats = dev_feats(test_logs)
# test_essays = get_essay_df(test_logs)

# print('< Essay Reconstruction >')
# train_essays = get_essay_df(train_logs)
# train_feats = train_feats.merge(word_feats(train_essays), on='id', how='left')
# train_feats = train_feats.merge(sent_feats(train_essays), on='id', how='left')
# train_feats = train_feats.merge(parag_feats(train_essays), on='id', how='left')
# train_feats = train_feats.merge(get_keys_pressed_per_second(train_logs), on='id', how='left')
# train_feats = train_feats.merge(product_to_keys(train_logs, train_essays), on='id', how='left')
# print('tf-idf vectors')
# train_char_tfidf, test_char_tfidf = get_char_tfidf_features(train_essays, test_essays)
# train_feats = train_feats.merge(train_char_tfidf, on='id', how='left')
# print('< Mapping >')
# train_scores = pd.read_csv(data_path + 'train_scores.csv')
# data = train_feats.me_



target_col = ['score']

drop_cols = ['id']

train_cols = x.columns

train_cols.__len__(), target_col.__len__()


kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)

oof_valid_preds = np.zeros(x.shape[0], )

X_test = test_feats[train_cols]
test_predict_list = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(x)):
    
    print("==-"* 50)
    print("Fold : ", fold)
    
    X_train, y_train = x.iloc[train_idx][train_cols], y[train_idx]
    X_valid, y_valid = x.iloc[valid_idx][train_cols], y[valid_idx]
    
    print("Trian :", X_train.shape, y_train.shape)
    print("Valid :", X_valid.shape, y_valid.shape)
    
    params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators" : 10000,
            "boosting_type": "gbdt",                
            "seed": 42
    }
    
    model = lgb.LGBMRegressor(**params)
        
    early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
    verbose_callback = lgb.log_evaluation(100)
        
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],  
              callbacks=[early_stopping_callback, verbose_callback],
    )
        
    valid_predict = model.predict(X_valid)
    oof_valid_preds[valid_idx] = valid_predict
    
    test_predict = model.predict(X_test)
    test_predict_list.append(test_predict)
    
    score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
    print("Fold RMSE Score : ", score)

    
oof_score = metrics.mean_squared_error(y, oof_valid_preds, squared=False)
print("OOF RMSE Score : ", oof_score)


def objective(trial):
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        'random_state': 48,
        "n_estimators" : 10000,
        "verbosity": -1,
        
        'max_depth': trial.suggest_categorical('max_depth', [5,10,20,40,100, -1]),
        'num_leaves' : trial.suggest_int('num_leaves', 2, 256),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        'reg_sqrt': trial.suggest_categorical('reg_sqrt', ['true', 'false']),
        
        'early_stopping_round' : 50,
        'n_jobs': -1,
    }
    
    kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    oof_valid_preds = np.zeros(x.shape[0], )
    X_test = test_feats[train_cols]
    test_predict_list = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(x)):
        print("==-"* 50)
        print("Fold : ", fold)
        X_train, y_train = x.iloc[train_idx][train_cols], y[train_idx]
        X_valid, y_valid = x.iloc[valid_idx][train_cols], y[valid_idx]
        print("Trian :", X_train.shape, y_train.shape)
        print("Valid :", X_valid.shape, y_valid.shape)
        model = lgb.LGBMRegressor(**params)
        early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
        verbose_callback = lgb.log_evaluation(100)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],  
              callbacks=[early_stopping_callback, verbose_callback],)
        valid_predict = model.predict(X_valid)
        oof_valid_preds[valid_idx] = valid_predict
        test_predict = model.predict(X_test)
        test_predict_list.append(test_predict)
        score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
        print("Fold RMSE Score : ", score)
    oof_score = metrics.mean_squared_error(y, oof_valid_preds, squared=False)
    return oof_score


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


params={'max_depth': 5, 'num_leaves': 95, 'reg_alpha': 0.005955763104435645, 'reg_lambda': 0.21455120855246107, 'colsample_bytree': 0.8213580994450626, 'subsample': 0.8650398638655841, 'reg_sqrt': 'true'}
kf = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)

oof_valid_preds = np.zeros(x.shape[0], )

X_test = test_feats[train_cols]
test_predict_list = []
models_dict={}

for fold, (train_idx, valid_idx) in enumerate(kf.split(x)):
    
    print("==-"* 50)
    print("Fold : ", fold)
    
    X_train, y_train = x.iloc[train_idx][train_cols], y[train_idx]
    X_valid, y_valid = x.iloc[valid_idx][train_cols], y[valid_idx]
    
    print("Trian :", X_train.shape, y_train.shape)
    print("Valid :", X_valid.shape, y_valid.shape)
    
    model = lgb.LGBMRegressor(**params)
        
    early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
    verbose_callback = lgb.log_evaluation(100)
        
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],  
              callbacks=[early_stopping_callback, verbose_callback],
    )
        
    valid_predict = model.predict(X_valid)
    oof_valid_preds[valid_idx] = valid_predict
    
    test_predict = model.predict(X_test)
    test_predict_list.append(test_predict)
    
    score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
    print("Fold RMSE Score : ", score)
    models_dict[fold]=model

    
oof_score = metrics.mean_squared_error(y, oof_valid_preds, squared=False)
print("OOF RMSE Score : ", oof_score)


feature_importances_values = np.asarray([model.feature_importances_ for model in models_dict.values()]).mean(axis=0)
feature_importance_df = pd.DataFrame({'name': train_cols, 'importance': feature_importances_values})

feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)


plt.figure(figsize=(15, 6))

ax = sns.barplot(data=feature_importance_df.head(50), x='name', y='importance')
ax.set_title(f"Mean feature importances")
ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)

plt.show()


test_feats['score'] = np.mean(test_predict_list, axis=0)


test_feats['score']


from catboost import CatBoostRegressor
import copy
params = {
            "iterations": 5000,
            "early_stopping_rounds": 50,
            "depth": 6,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose":100
        }
test_feats1=copy.deepcopy(test_feats)
kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)

oof_valid_preds = np.zeros(x.shape[0], )

X_test1 = test_feats1[train_cols]
test_predict_list1 = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(x)):
    
    print("==-"* 50)
    print("Fold : ", fold)
    
    X_train, y_train = x.iloc[train_idx][train_cols], y[train_idx]
    X_valid, y_valid = x.iloc[valid_idx][train_cols], y[valid_idx]
    
    print("Trian :", X_train.shape, y_train.shape)
    print("Valid :", X_valid.shape, y_valid.shape)
    
    
    model = CatBoostRegressor(**params)
        
    
        
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)]
    )
        
    valid_predict = model.predict(X_valid)
    oof_valid_preds[valid_idx] = valid_predict
    
    test_predict1 = model.predict(X_test1)
    test_predict_list1.append(test_predict1)
    
    score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
    print("Fold RMSE Score : ", score)

    
oof_score = metrics.mean_squared_error(y, oof_valid_preds, squared=False)
print("OOF RMSE Score : ", oof_score)


def objective1(trial):
    
    params = {
        "iterations": 5000,
        "early_stopping_rounds": 200,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": 0,  # keep quiet during CV

        # Optuna search space
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255)
    }
    
    kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    oof_valid_preds = np.zeros(x.shape[0], )
    X_test1 = test_feats1[train_cols]
    test_predict_list1 = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(x)):
        print("==-"* 50)
        print("Fold : ", fold)
        X_train, y_train = x.iloc[train_idx][train_cols], y[train_idx]
        X_valid, y_valid = x.iloc[valid_idx][train_cols], y[valid_idx]
        print("Trian :", X_train.shape, y_train.shape)
        print("Valid :", X_valid.shape, y_valid.shape)
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        valid_predict = model.predict(X_valid)
        oof_valid_preds[valid_idx] = valid_predict
        test_predict = model.predict(X_test1)
        test_predict_list1.append(test_predict1)
        score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
        print("Fold RMSE Score : ", score)
    oof_score = metrics.mean_squared_error(y, oof_valid_preds, squared=False)
    return oof_score


studyCatBoost = optuna.create_study(direction='minimize')
studyCatBoost.optimize(objective1, n_trials=5)
print('Number of finished trials:', len(studyCatBoost.trials))
print('Best trial:', studyCatBoost.best_trial.params)


params={'depth': 7, 'learning_rate': 0.0022953552509762673, 'l2_leaf_reg': 1.089443011371721, 'bagging_temperature': 2.7793514257896677, 'random_strength': 0.0031728427363960196, 'border_count': 143}
kf = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
oof_valid_preds = np.zeros(x.shape[0], )
X_test1 = test_feats1[train_cols]
test_predict_list1 = []
models_dict1={}
for fold, (train_idx, valid_idx) in enumerate(kf.split(x)):
    print("==-"* 50)
    print("Fold : ", fold)
    X_train, y_train = x.iloc[train_idx][train_cols], y[train_idx]
    X_valid, y_valid = x.iloc[valid_idx][train_cols], y[valid_idx]
    print("Trian :", X_train.shape, y_train.shape)
    print("Valid :", X_valid.shape, y_valid.shape)
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    valid_predict = model.predict(X_valid)
    oof_valid_preds[valid_idx] = valid_predict
    test_predict1 = model.predict(X_test1)
    test_predict_list1.append(test_predict1)
    score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
    print("Fold RMSE Score : ", score)
    models_dict1[fold]=model
oof_score = metrics.mean_squared_error(y, oof_valid_preds, squared=False)
print("OOF_Mean_score", oof_score)


feature_importances_values1 = np.asarray([model.feature_importances_ for model in models_dict1.values()]).mean(axis=0)
feature_importance_df1 = pd.DataFrame({'name': train_cols, 'importance': feature_importances_values})

feature_importance_df1 = feature_importance_df.sort_values('importance', ascending=False)


plt.figure(figsize=(15, 6))

ax = sns.barplot(data=feature_importance_df1.head(30), x='name', y='importance')
ax.set_title(f"Mean feature importances")
ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)

plt.show()


test_feats1['score'] = np.mean(test_predict_list1, axis=0)


test_feats1['score']


test_feats['score']


test_Catboost=np.array(test_predict_list1)


test_LGBM=np.array(test_predict_list)


blend=1.0*test_Catboost+0.0*test_LGBM


test_feats_2=copy.deepcopy(test_feats)


test_feats_2['score']=np.mean(blend, axis=0)


test_feats_2['score']


test_feats_2[['id','score']].to_csv("submission.csv",index=False)

