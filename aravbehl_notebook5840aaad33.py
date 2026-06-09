import polars as pl
import pandas as pd
import numpy as np
import re
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")


num_cols = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
text_changes = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']


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
    
    # Fix for Polars version compatibility
    try:
        # New Polars syntax (0.19+)
        temp = temp.with_columns(
            input_word_count = pl.col('text_change').list.len(),
            input_word_length_mean = pl.col('text_change').map_elements(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)),
            input_word_length_max = pl.col('text_change').map_elements(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)),
            input_word_length_std = pl.col('text_change').map_elements(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)),
            input_word_length_median = pl.col('text_change').map_elements(lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0)),
            input_word_length_skew = pl.col('text_change').map_elements(lambda x: skew([len(i) for i in x] if len(x) > 0 else 0))
        )
    except AttributeError:
        # Old Polars syntax (< 0.19)
        temp = temp.with_columns(
            input_word_count = pl.col('text_change').list.lengths(),
            input_word_length_mean = pl.col('text_change').apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)),
            input_word_length_max = pl.col('text_change').apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)),
            input_word_length_std = pl.col('text_change').apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)),
            input_word_length_median = pl.col('text_change').apply(lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0)),
            input_word_length_skew = pl.col('text_change').apply(lambda x: skew([len(i) for i in x] if len(x) > 0 else 0))
        )
    
    temp = temp.drop('text_change')
    feats = feats.join(temp, on='id', how='left') 

    
    print("< Numerical columns features (ENHANCED with quantiles) >")
    
    # IMPROVED: Add more quantiles like number1_code (q05, q25, q75, q95)
    temp = df.group_by("id").agg([
        pl.sum('action_time').alias('action_time_sum'),
        # Mean
        pl.mean('down_time').alias('down_time_mean'),
        pl.mean('up_time').alias('up_time_mean'),
        pl.mean('action_time').alias('action_time_mean'),
        pl.mean('cursor_position').alias('cursor_position_mean'),
        pl.mean('word_count').alias('word_count_mean'),
        # Std
        pl.std('down_time').alias('down_time_std'),
        pl.std('up_time').alias('up_time_std'),
        pl.std('action_time').alias('action_time_std'),
        pl.std('cursor_position').alias('cursor_position_std'),
        pl.std('word_count').alias('word_count_std'),
        # Median
        pl.median('down_time').alias('down_time_median'),
        pl.median('up_time').alias('up_time_median'),
        pl.median('action_time').alias('action_time_median'),
        pl.median('cursor_position').alias('cursor_position_median'),
        pl.median('word_count').alias('word_count_median'),
        # Min
        pl.min('down_time').alias('down_time_min'),
        pl.min('up_time').alias('up_time_min'),
        pl.min('action_time').alias('action_time_min'),
        pl.min('cursor_position').alias('cursor_position_min'),
        pl.min('word_count').alias('word_count_min'),
        # Max
        pl.max('down_time').alias('down_time_max'),
        pl.max('up_time').alias('up_time_max'),
        pl.max('action_time').alias('action_time_max'),
        pl.max('cursor_position').alias('cursor_position_max'),
        pl.max('word_count').alias('word_count_max'),
        # NEW: Additional quantiles (like number1_code)
        pl.quantile('down_time', 0.05).alias('down_time_q05'),
        pl.quantile('down_time', 0.25).alias('down_time_q25'),
        pl.quantile('down_time', 0.5).alias('down_time_quantile'),
        pl.quantile('down_time', 0.75).alias('down_time_q75'),
        pl.quantile('down_time', 0.95).alias('down_time_q95'),
        
        pl.quantile('up_time', 0.05).alias('up_time_q05'),
        pl.quantile('up_time', 0.25).alias('up_time_q25'),
        pl.quantile('up_time', 0.5).alias('up_time_quantile'),
        pl.quantile('up_time', 0.75).alias('up_time_q75'),
        pl.quantile('up_time', 0.95).alias('up_time_q95'),
        
        pl.quantile('action_time', 0.05).alias('action_time_q05'),
        pl.quantile('action_time', 0.25).alias('action_time_q25'),
        pl.quantile('action_time', 0.5).alias('action_time_quantile'),
        pl.quantile('action_time', 0.75).alias('action_time_q75'),
        pl.quantile('action_time', 0.95).alias('action_time_q95'),
        
        pl.quantile('cursor_position', 0.05).alias('cursor_position_q05'),
        pl.quantile('cursor_position', 0.25).alias('cursor_position_q25'),
        pl.quantile('cursor_position', 0.5).alias('cursor_position_quantile'),
        pl.quantile('cursor_position', 0.75).alias('cursor_position_q75'),
        pl.quantile('cursor_position', 0.95).alias('cursor_position_q95'),
        
        pl.quantile('word_count', 0.05).alias('word_count_q05'),
        pl.quantile('word_count', 0.25).alias('word_count_q25'),
        pl.quantile('word_count', 0.5).alias('word_count_quantile'),
        pl.quantile('word_count', 0.75).alias('word_count_q75'),
        pl.quantile('word_count', 0.95).alias('word_count_q95'),
    ])
    feats = feats.join(temp, on='id', how='left') 


    print("< Categorical columns features >")
    
    temp  = df.group_by("id").agg(pl.n_unique(['activity', 'down_event', 'up_event', 'text_change']))
    feats = feats.join(temp, on='id', how='left') 


    
    print("< Idle time features (ENHANCED) >")

    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    
    # IMPROVED: Match number1_code's 8 pause buckets (in milliseconds, then convert)
    temp = temp.group_by("id").agg(
        inter_key_largest_lantency = pl.max('time_diff'),
        inter_key_median_lantency = pl.median('time_diff'),
        inter_key_mean = pl.mean('time_diff'),
        inter_key_std = pl.std('time_diff'),
        inter_key_sum = pl.sum('time_diff'),
        # NEW: Additional IKI stats (like number1_code)
        inter_key_q05 = pl.quantile('time_diff', 0.05),
        inter_key_q25 = pl.quantile('time_diff', 0.25),
        inter_key_q75 = pl.quantile('time_diff', 0.75),
        inter_key_q95 = pl.quantile('time_diff', 0.95),
        inter_key_skew = pl.col('time_diff').map_elements(lambda x: skew(x) if len(x) > 1 else 0),
        inter_key_kurtosis = pl.col('time_diff').map_elements(lambda x: kurtosis(x) if len(x) > 1 else 0),
        # IMPROVED: 8 pause buckets (like number1_code)
        pause_0_100 = pl.col('time_diff').filter((pl.col('time_diff') >= 0) & (pl.col('time_diff') < 0.1)).count(),
        pause_100_250 = pl.col('time_diff').filter((pl.col('time_diff') >= 0.1) & (pl.col('time_diff') < 0.25)).count(),
        pause_250_500 = pl.col('time_diff').filter((pl.col('time_diff') >= 0.25) & (pl.col('time_diff') < 0.5)).count(),
        pause_500_1000 = pl.col('time_diff').filter((pl.col('time_diff') >= 0.5) & (pl.col('time_diff') < 1)).count(),
        pause_1000_1500 = pl.col('time_diff').filter((pl.col('time_diff') >= 1) & (pl.col('time_diff') < 1.5)).count(),
        pause_1500_2000 = pl.col('time_diff').filter((pl.col('time_diff') >= 1.5) & (pl.col('time_diff') < 2)).count(),
        pause_2000_3000 = pl.col('time_diff').filter((pl.col('time_diff') >= 2) & (pl.col('time_diff') < 3)).count(),
        pause_3000_plus = pl.col('time_diff').filter(pl.col('time_diff') >= 3).count(),
    )
    feats = feats.join(temp, on='id', how='left') 
    
    print("< P-bursts features >")

    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('time_diff')<2)
    
    # Simplified P-bursts calculation - compatible with all Polars versions
    # Count consecutive True values in time_diff < 2
    temp = temp.with_columns(
        pl.col('time_diff').rle_id().alias('burst_id')
    )
    temp = temp.group_by(['id', 'burst_id', 'time_diff']).agg(
        pl.count().alias('burst_length')
    )
    temp = temp.filter(pl.col('time_diff') == True)
    temp = temp.group_by('id').agg([
        pl.mean('burst_length').alias('P-bursts_mean'),
        pl.std('burst_length').alias('P-bursts_std'),
        pl.count('burst_length').alias('P-bursts_count'),
        pl.median('burst_length').alias('P-bursts_median'),
        pl.max('burst_length').alias('P-bursts_max'),
        pl.first('burst_length').alias('P-bursts_first'),
        pl.last('burst_length').alias('P-bursts_last')
    ])
    feats = feats.join(temp, on='id', how='left') 


    print("< R-bursts features >")

    temp = df.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('activity').is_in(['Remove/Cut']))
    
    # Simplified R-bursts calculation - compatible with all Polars versions
    temp = temp.with_columns(
        pl.col('activity').rle_id().alias('burst_id')
    )
    temp = temp.group_by(['id', 'burst_id', 'activity']).agg(
        pl.count().alias('burst_length')
    )
    temp = temp.filter(pl.col('activity') == True)
    temp = temp.group_by('id').agg([
        pl.mean('burst_length').alias('R-bursts_mean'),
        pl.std('burst_length').alias('R-bursts_std'),
        pl.median('burst_length').alias('R-bursts_median'),
        pl.max('burst_length').alias('R-bursts_max'),
        pl.first('burst_length').alias('R-bursts_first'),
        pl.last('burst_length').alias('R-bursts_last')
    ])
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


# This cell is intentionally left empty - duplicate functionality removed
# The dev_feats function is already defined in cell-1 with Polars compatibility fixes
pass


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



data_path     = '/kaggle/input/linking-writing-processes-to-writing-quality/'
train_logs    = pl.scan_csv(data_path + 'train_logs.csv')
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
train_scores   = pd.read_csv(data_path + 'train_scores.csv')
data           = train_feats.merge(train_scores, on='id', how='left')
x              = data.drop(['id', 'score'], axis=1)
y              = data['score'].values
print(f'Number of features: {len(x.columns)}')


print('< Testing Data >')
test_logs   = pl.scan_csv(data_path + 'test_logs.csv')
test_feats  = dev_feats(test_logs)
test_feats  = test_feats.collect().to_pandas()

test_logs             = test_logs.collect().to_pandas()
test_essays           = get_essay_df(test_logs)
test_feats            = test_feats.merge(word_feats(test_essays), on='id', how='left')
test_feats            = test_feats.merge(sent_feats(test_essays), on='id', how='left')
test_feats            = test_feats.merge(parag_feats(test_essays), on='id', how='left')
test_feats            = test_feats.merge(get_keys_pressed_per_second(test_logs), on='id', how='left')
test_feats            = test_feats.merge(product_to_keys(test_logs, test_essays), on='id', how='left')


test_ids = test_feats['id'].values
testin_x = test_feats.drop(['id'], axis=1)


# ===================================================================
# ADVANCED FEATURES - Critical for reaching 0.56-0.57 RMSE
# ===================================================================
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def add_advanced_text_features(train_feats, test_feats, train_essays, test_essays):
    """Add TF-IDF and advanced text features"""
    
    print("< TF-IDF Features (CRITICAL!) >")
    
    # Combine train and test for TF-IDF fitting
    all_essays = pd.concat([train_essays['essay'], test_essays['essay']], axis=0).fillna('')
    train_size = len(train_essays)
    
    # Word-level TF-IDF (1-3 grams) - compress to 32 features
    print("  - Word TF-IDF (1-3 grams)...")
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=1000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    word_features = word_tfidf.fit_transform(all_essays)
    
    # Reduce dimensions using SVD
    svd_word = TruncatedSVD(n_components=32, random_state=42)
    word_dense = svd_word.fit_transform(word_features)
    
    for i in range(32):
        train_feats[f'tfidf_word_{i}'] = word_dense[:train_size, i]
        test_feats[f'tfidf_word_{i}'] = word_dense[train_size:, i]
    
    # Character-level TF-IDF (2-5 grams) - compress to 32 features
    print("  - Char TF-IDF (2-5 grams)...")
    char_tfidf = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        max_features=1000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    char_features = char_tfidf.fit_transform(all_essays)
    
    svd_char = TruncatedSVD(n_components=32, random_state=42)
    char_dense = svd_char.fit_transform(char_features)
    
    for i in range(32):
        train_feats[f'tfidf_char_{i}'] = char_dense[:train_size, i]
        test_feats[f'tfidf_char_{i}'] = char_dense[train_size:, i]
    
    print(f"  ✓ Added 64 TF-IDF features")
    
    return train_feats, test_feats

def add_ratio_features(df):
    """Add ratio and interaction features"""
    
    print("< Ratio & Interaction Features >")
    
    # Typing efficiency ratios
    if 'action_time_sum' in df.columns and 'down_time_max' in df.columns:
        df['typing_efficiency'] = df['action_time_sum'] / (df['down_time_max'] + 1)
    
    if 'input_count' in df.columns and 'remove_count' in df.columns:
        df['edit_ratio'] = df['remove_count'] / (df['input_count'] + 1)
        df['productivity'] = df['input_count'] / (df['input_count'] + df['remove_count'] + 1)
    
    # Pause ratios
    pause_cols = [c for c in df.columns if 'pause_' in c and '_' in c[6:]]
    if pause_cols:
        total_pauses = df[pause_cols].sum(axis=1)
        for col in pause_cols:
            df[f'{col}_ratio'] = df[col] / (total_pauses + 1)
    
    # Burst ratios
    if 'P-bursts_count' in df.columns and 'R-bursts_count' in df.columns:
        df['burst_balance'] = df['P-bursts_count'] / (df['R-bursts_count'] + 1)
    
    # Word/sentence ratios
    if 'word_len_mean' in df.columns and 'sent_len_mean' in df.columns:
        df['words_per_sent_ratio'] = df['sent_len_mean'] / (df['word_len_mean'] + 1)
    
    print(f"  ✓ Added {len([c for c in df.columns if 'ratio' in c or 'efficiency' in c or 'productivity' in c]) - len(pause_cols)} ratio features")
    
    return df

def add_time_based_features(logs_df):
    """Add temporal pattern features"""
    
    print("< Temporal Pattern Features >")
    
    features = []
    
    for essay_id in logs_df['id'].unique():
        essay_logs = logs_df[logs_df['id'] == essay_id].sort_values('down_time')
        
        feat = {'id': essay_id}
        
        # Split into quartiles
        n = len(essay_logs)
        if n > 4:
            q1_logs = essay_logs.iloc[:n//4]
            q2_logs = essay_logs.iloc[n//4:n//2]
            q3_logs = essay_logs.iloc[n//2:3*n//4]
            q4_logs = essay_logs.iloc[3*n//4:]
            
            # Activity distribution across time
            for i, q_logs in enumerate([q1_logs, q2_logs, q3_logs, q4_logs], 1):
                feat[f'q{i}_input_count'] = (q_logs['activity'] == 'Input').sum()
                feat[f'q{i}_remove_count'] = (q_logs['activity'].isin(['Remove/Cut'])).sum()
                feat[f'q{i}_word_count'] = q_logs['word_count'].iloc[-1] if len(q_logs) > 0 else 0
            
            # Progression metrics
            feat['word_growth_q1_q4'] = feat['q4_word_count'] - feat['q1_word_count']
            feat['input_acceleration'] = feat['q4_input_count'] - feat['q1_input_count']
        
        features.append(feat)
    
    feat_df = pd.DataFrame(features)
    print(f"  ✓ Added {len(feat_df.columns)-1} temporal features")
    
    return feat_df

# Apply advanced features
print("="*70)
print("ADDING ADVANCED FEATURES FOR 0.56-0.57 RMSE TARGET")
print("="*70)

# TF-IDF features (most important!)
train_feats, test_feats = add_advanced_text_features(
    train_feats, test_feats, train_essays, test_essays
)

# Temporal features
train_temporal = add_time_based_features(train_logs)
test_temporal = add_time_based_features(test_logs)

train_feats = train_feats.merge(train_temporal, on='id', how='left')
test_feats = test_feats.merge(test_temporal, on='id', how='left')

# Ratio features
train_feats = add_ratio_features(train_feats)
test_feats = add_ratio_features(test_feats)

# Fill NaN values
train_feats = train_feats.fillna(0)
test_feats = test_feats.fillna(0)

print(f"\n✓ Total features now: {len(train_feats.columns)}")
print("="*70)


# Skip interaction features - they cause compatibility issues
# Using base features only for stability
print("="*70)
print("SKIPPING INTERACTION FEATURES FOR STABILITY")
print("Using base features only")
print("="*70)


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED ENSEMBLE WITH STACKING - TARGET: 0.56-0.57 RMSE")
print("="*70)

# Configuration
N_FOLDS = 10
RANDOM_SEED = 42

def train_model_kfold(x_train, y_train, x_test, model_fn, model_name, n_splits=10):
    """Train model with K-Fold CV"""
    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(len(x_test))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    print(f"\nTraining {model_name}...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train), 1):
        x_tr = x_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        x_val = x_train.iloc[val_idx]
        y_val = y_train[val_idx]
        
        model = model_fn()
        
        # Train with appropriate method
        if "LightGBM" in model_name:
            model.fit(
                x_tr, y_tr,
                eval_set=[(x_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
            )
        elif "XGBoost" in model_name:
            model.fit(
                x_tr, y_tr,
                eval_set=[(x_val, y_val)],
                early_stopping_rounds=150,
                verbose=False
            )
        elif "CatBoost" in model_name:
            model.fit(x_tr, y_tr, eval_set=(x_val, y_val), verbose=False)
        else:
            model.fit(x_tr, y_tr)
        
        # Predict
        oof_preds[val_idx] = model.predict(x_val)
        test_preds += model.predict(x_test) / n_splits
        
        if fold % 3 == 0 or fold == n_splits:
            fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
            print(f"  Fold {fold}/{n_splits}: RMSE {fold_rmse:.4f}")
    
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"✓ {model_name} OOF RMSE: {oof_rmse:.4f}")
    
    return oof_preds, test_preds, oof_rmse

# ============================================================================
# LEVEL 1: BASE MODELS (Diverse configurations)
# ============================================================================

# Model 1: LightGBM - Tuned like number1_code
lgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 16,
    'min_child_samples': 20,
    'subsample': 0.6,
    'colsample_bytree': 0.4,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'min_split_gain': 0.01,
    'random_state': RANDOM_SEED,
    'verbosity': -1,
    'force_col_wise': True,
    'extra_trees': True,
}

oof_lgb, test_lgb, rmse_lgb = train_model_kfold(
    x, y, testin_x,
    lambda: LGBMRegressor(**lgb_params),
    "LightGBM", N_FOLDS
)

# Model 2: LightGBM - Different config (diversity)
lgb_params2 = {
    'n_estimators': 2500,
    'learning_rate': 0.02,
    'max_depth': 6,
    'num_leaves': 32,
    'min_child_samples': 15,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED + 1,
    'verbosity': -1,
    'force_col_wise': True,
}

oof_lgb2, test_lgb2, rmse_lgb2 = train_model_kfold(
    x, y, testin_x,
    lambda: LGBMRegressor(**lgb_params2),
    "LightGBM-2", N_FOLDS
)

# Model 3: XGBoost
xgb_params = {
    'n_estimators': 2500,
    'learning_rate': 0.01,
    'max_depth': 4,
    'min_child_weight': 3,
    'subsample': 0.65,
    'colsample_bytree': 0.65,
    'reg_alpha': 1.5,
    'reg_lambda': 2.0,
    'gamma': 0.1,
    'random_state': RANDOM_SEED,
    'tree_method': 'hist'
}

oof_xgb, test_xgb, rmse_xgb = train_model_kfold(
    x, y, testin_x,
    lambda: XGBRegressor(**xgb_params),
    "XGBoost", N_FOLDS
)

# Model 4: CatBoost
cat_params = {
    'iterations': 2500,
    'learning_rate': 0.01,
    'depth': 6,
    'l2_leaf_reg': 5,
    'random_strength': 0.3,
    'bagging_temperature': 0.5,
    'random_state': RANDOM_SEED,
    'verbose': False
}

oof_cat, test_cat, rmse_cat = train_model_kfold(
    x, y, testin_x,
    lambda: CatBoostRegressor(**cat_params),
    "CatBoost", N_FOLDS
)

# ============================================================================
# LEVEL 2: STACKING (Meta-model on predictions)
# ============================================================================

print("\n" + "="*70)
print("STACKING LAYER")
print("="*70)

# Create meta-features (OOF predictions from base models)
meta_train = np.column_stack([oof_lgb, oof_lgb2, oof_xgb, oof_cat])
meta_test = np.column_stack([test_lgb, test_lgb2, test_xgb, test_cat])

print(f"Meta-features shape: {meta_train.shape}")

# Train Ridge regression on meta-features
ridge_params = {'alpha': 1.0, 'random_state': RANDOM_SEED}
oof_ridge = np.zeros(len(y))
test_ridge = np.zeros(len(testin_x))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(meta_train), 1):
    meta_tr = meta_train[train_idx]
    y_tr = y[train_idx]
    meta_val = meta_train[val_idx]
    
    ridge = Ridge(**ridge_params)
    ridge.fit(meta_tr, y_tr)
    
    oof_ridge[val_idx] = ridge.predict(meta_val)
    test_ridge += ridge.predict(meta_test) / N_FOLDS

rmse_ridge = np.sqrt(mean_squared_error(y, oof_ridge))
print(f"✓ Stacking (Ridge) OOF RMSE: {rmse_ridge:.4f}")

# ============================================================================
# FINAL ENSEMBLE: Weighted combination
# ============================================================================

print("\n" + "="*70)
print("FINAL ENSEMBLE")
print("="*70)

print("\nIndividual Model Performance:")
print(f"  LightGBM-1: {rmse_lgb:.4f}")
print(f"  LightGBM-2: {rmse_lgb2:.4f}")
print(f"  XGBoost:    {rmse_xgb:.4f}")
print(f"  CatBoost:   {rmse_cat:.4f}")
print(f"  Stacking:   {rmse_ridge:.4f}")

# Calculate optimal weights (inverse RMSE squared for stronger differentiation)
all_oof = np.column_stack([oof_lgb, oof_lgb2, oof_xgb, oof_cat, oof_ridge])
all_test = np.column_stack([test_lgb, test_lgb2, test_xgb, test_cat, test_ridge])
rmse_scores = np.array([rmse_lgb, rmse_lgb2, rmse_xgb, rmse_cat, rmse_ridge])

# Inverse RMSE squared weights (best models get much higher weight)
inverse_rmse_sq = 1 / (rmse_scores ** 2)
weights = inverse_rmse_sq / inverse_rmse_sq.sum()

print("\nOptimal Weights:")
names = ['LightGBM-1', 'LightGBM-2', 'XGBoost', 'CatBoost', 'Stacking']
for name, w in zip(names, weights):
    print(f"  {name}: {w:.3f}")

# Weighted ensemble
final_oof = np.sum(all_oof * weights, axis=1)
final_test = np.sum(all_test * weights, axis=1)
final_rmse = np.sqrt(mean_squared_error(y, final_oof))

print(f"\n✓ FINAL ENSEMBLE RMSE: {final_rmse:.4f}")

# Clip to valid range
final_test = np.clip(final_test, 0.5, 6.0)

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'score': final_test
})

submission.to_csv('submission.csv', index=False)
print("\n✓ Submission saved: submission.csv")
print(f"✓ Expected Kaggle Score: ~{final_rmse:.4f}")
print("="*70)










