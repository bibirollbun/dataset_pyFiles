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
    temp = temp.with_columns(pl.when(pl.col("time_diff") & pl.col("time_diff").is_last()).then(pl.count()).over(pl.col("time_diff").rle_id()).alias('P-bursts'))
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


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.optimize import minimize
import lightgbm as lgb
import re
import warnings
warnings.filterwarnings('ignore')

"""
COMPLETE PIPELINE: CLASSIFIERS + ENSEMBLE + ADVANCED FEATURES
==============================================================

This combines EVERYTHING that works:
1. Classifier probability features (your original 0.6065 CV approach)
2. Balanced ensemble with diverse models
3. Advanced temporal features
4. TF-IDF features (optional)

Target: 0.56-0.57 LB
"""

print("="*70)
print("COMPLETE PIPELINE: ALL FEATURES + BALANCED ENSEMBLE")
print("="*70)

# Configuration
N_FOLDS = 10
PCA_COMPONENTS = 50
RANDOM_SEED = 42

# ============================================================================
# STEP 1: ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Advanced Feature Engineering")
print("="*70)

def create_temporal_features(logs_df):
    """Create temporal and behavioral features from logs"""
    print("  Creating temporal features...")
    features = []
    
    for essay_id in logs_df['id'].unique():
        essay_logs = logs_df[logs_df['id'] == essay_id].sort_values('down_time')
        
        feat = {'id': essay_id}
        
        # Total time
        total_time = (essay_logs['up_time'].max() - essay_logs['down_time'].min()) / 1000
        feat['total_writing_time'] = total_time
        
        # Activity transitions
        activity_changes = (essay_logs['activity'] != essay_logs['activity'].shift()).sum()
        feat['activity_transitions'] = activity_changes
        feat['transitions_per_minute'] = activity_changes / max(1, total_time / 60)
        
        # Burst analysis
        time_diffs = essay_logs['down_time'].diff().fillna(0) / 1000
        bursts = (time_diffs < 0.5).sum()
        feat['intense_burst_count'] = bursts
        feat['burst_ratio'] = bursts / max(1, len(essay_logs))
        
        # Early vs late writing speed
        mid_point = len(essay_logs) // 2
        early_logs = essay_logs.iloc[:mid_point]
        late_logs = essay_logs.iloc[mid_point:]
        
        early_time = (early_logs['up_time'].max() - early_logs['down_time'].min()) / 1000
        late_time = (late_logs['up_time'].max() - late_logs['down_time'].min()) / 1000
        
        feat['early_actions_per_sec'] = len(early_logs) / max(1, early_time)
        feat['late_actions_per_sec'] = len(late_logs) / max(1, late_time)
        feat['speed_change_ratio'] = feat['late_actions_per_sec'] / max(0.01, feat['early_actions_per_sec'])
        
        # Revision patterns
        revisions = essay_logs[essay_logs['activity'] == 'Remove/Cut']
        feat['revision_count'] = len(revisions)
        feat['revision_ratio'] = len(revisions) / max(1, len(essay_logs))
        
        # Late-stage revisions
        late_threshold = essay_logs['down_time'].quantile(0.8)
        late_revisions = revisions[revisions['down_time'] > late_threshold]
        feat['late_revision_ratio'] = len(late_revisions) / max(1, len(revisions))
        
        features.append(feat)
    
    features_df = pd.DataFrame(features)
    print(f"  ✓ Created {features_df.shape[1]-1} temporal features")
    return features_df

def create_tfidf_features(train_essays, test_essays, n_components=15):
    """Create TF-IDF features with SVD"""
    print("  Creating TF-IDF features...")
    
    # Word-level TF-IDF
    tfidf = TfidfVectorizer(
        max_features=800,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    
    all_essays = pd.concat([train_essays, test_essays])
    tfidf.fit(all_essays['essay'])
    
    train_tfidf = tfidf.transform(train_essays['essay'])
    test_tfidf = tfidf.transform(test_essays['essay'])
    
    # SVD
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)
    
    train_df = pd.DataFrame(train_svd, columns=[f'tfidf_svd_{i}' for i in range(n_components)])
    test_df = pd.DataFrame(test_svd, columns=[f'tfidf_svd_{i}' for i in range(n_components)])
    
    print(f"  ✓ Created {n_components} TF-IDF SVD features")
    return train_df, test_df

# Create advanced features (assuming train_logs, test_logs, train_essays, test_essays exist)
print("\nGenerating advanced features...")
print("  (Set SKIP_ADVANCED=True if you don't have essays/logs objects)")

SKIP_ADVANCED = False  # Set to True if you want to skip this section

if not SKIP_ADVANCED:
    try:
        train_temporal = create_temporal_features(train_logs)
        test_temporal = create_temporal_features(test_logs)
        
        train_tfidf, test_tfidf = create_tfidf_features(train_essays, test_essays)
        
        # Merge with existing features
        x_enhanced = pd.concat([
            x.reset_index(drop=True),
            train_temporal.drop('id', axis=1).reset_index(drop=True),
            train_tfidf.reset_index(drop=True)
        ], axis=1)
        
        testin_x_enhanced = pd.concat([
            testin_x.reset_index(drop=True),
            test_temporal.drop('id', axis=1).reset_index(drop=True),
            test_tfidf.reset_index(drop=True)
        ], axis=1)
        
        print(f"\n  ✓ Enhanced features: {x_enhanced.shape[1]} total")
    except:
        print("  ⚠ Advanced features skipped - using original features")
        x_enhanced = x.copy()
        testin_x_enhanced = testin_x.copy()
else:
    x_enhanced = x.copy()
    testin_x_enhanced = testin_x.copy()

# Clean features
x_clean = x_enhanced.fillna(0).replace([np.inf, -np.inf], 0)
testin_x_clean = testin_x_enhanced.fillna(0).replace([np.inf, -np.inf], 0)

print(f"\nBase feature set: {x_clean.shape[1]} features")

# ============================================================================
# STEP 2: CLASSIFIER PROBABILITY FEATURES (YOUR ORIGINAL APPROACH)
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Classifier Probability Features")
print("="*70)

# Create ordinal target for classification
print("Creating ordinal target mapping...")
vals = {0.5: 0, 1.0: 1, 1.5: 2, 2.0: 3, 2.5: 4, 3.0: 5,
        3.5: 6, 4.0: 7, 4.5: 8, 5.0: 9, 5.5: 10, 6.0: 11}

_y = pd.Series(y).map(vals)
n_classes = _y.nunique()
print(f"  Mapped to {n_classes} ordinal classes")

# PCA for classifiers
print(f"  Applying PCA ({PCA_COMPONENTS} components) for classifiers...")
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
df_combined = pd.concat([x_clean, testin_x_clean])
pca.fit(df_combined)

x_pca = pca.transform(x_clean)
testin_x_pca = pca.transform(testin_x_clean)

def train_classifier_cv(data_x, data_y, model_fn, n_splits=5, test_x=None, square_features=False):
    """Train classifier with cross-validation"""
    cv_probs = np.zeros((len(data_x), n_classes))
    test_probs = np.zeros((len(test_x), n_classes)) if test_x is not None else None
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_x, data_y)):
        x_tr = data_x[train_idx]
        y_tr = data_y.iloc[train_idx]
        x_val = data_x[val_idx]
        
        if square_features:
            x_tr = x_tr ** 2
            x_val = x_val ** 2
        
        model = model_fn()
        model.fit(x_tr, y_tr)
        
        cv_probs[val_idx] = model.predict_proba(x_val)
        
        if test_x is not None:
            test_x_fold = test_x ** 2 if square_features else test_x
            test_probs += model.predict_proba(test_x_fold) / n_splits
    
    return cv_probs, test_probs

# Classifier 1: MultinomialNB
print("\n  Training MultinomialNB classifier...")
oof_prob_nb, test_prob_nb = train_classifier_cv(
    x_pca.copy(), _y.copy(),
    lambda: MultinomialNB(alpha=2.0),
    test_x=testin_x_pca.copy(),
    square_features=True
)
print("  ✓ MultinomialNB complete")

# Classifier 2: MLPClassifier
print("  Training MLPClassifier...")
oof_prob_mlp, test_prob_mlp = train_classifier_cv(
    x_pca.copy(), _y.copy(),
    lambda: MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200,
        alpha=0.01,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_SEED
    ),
    test_x=testin_x_pca.copy(),
    square_features=False
)
print("  ✓ MLPClassifier complete")

# Create probability features
prob_features_train = pd.DataFrame(
    np.hstack([oof_prob_nb, oof_prob_mlp]),
    columns=[f'nb_prob_{i}' for i in range(n_classes)] + 
            [f'mlp_prob_{i}' for i in range(n_classes)]
)

prob_features_test = pd.DataFrame(
    np.hstack([test_prob_nb, test_prob_mlp]),
    columns=[f'nb_prob_{i}' for i in range(n_classes)] + 
            [f'mlp_prob_{i}' for i in range(n_classes)]
)

# Add weighted sum features
Inversemapper = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0,
                 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0, 10: 5.5, 11: 6.0}

def weighted_sum(row, prefix, n_classes):
    return sum(row[f'{prefix}_prob_{i}'] * Inversemapper.get(i, 0) for i in range(n_classes))

prob_features_train['nb_weighted_sum'] = prob_features_train.apply(
    lambda row: weighted_sum(row, 'nb', n_classes), axis=1)
prob_features_train['mlp_weighted_sum'] = prob_features_train.apply(
    lambda row: weighted_sum(row, 'mlp', n_classes), axis=1)

prob_features_test['nb_weighted_sum'] = prob_features_test.apply(
    lambda row: weighted_sum(row, 'nb', n_classes), axis=1)
prob_features_test['mlp_weighted_sum'] = prob_features_test.apply(
    lambda row: weighted_sum(row, 'mlp', n_classes), axis=1)

print(f"\n  ✓ Created {prob_features_train.shape[1]} classifier probability features")

# ============================================================================
# STEP 3: COMBINE ALL FEATURES
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Feature Combination")
print("="*70)

x_augmented = pd.concat([
    x_clean.reset_index(drop=True),
    prob_features_train.reset_index(drop=True)
], axis=1)

testin_x_augmented = pd.concat([
    testin_x_clean.reset_index(drop=True),
    prob_features_test.reset_index(drop=True)
], axis=1)

print(f"  ✓ Total features: {x_augmented.shape[1]}")
print(f"    - Base features: {x_clean.shape[1]}")
print(f"    - Classifier features: {prob_features_train.shape[1]}")

# ============================================================================
# STEP 4: BALANCED ENSEMBLE WITH DIVERSE MODELS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Balanced Ensemble Training")
print("="*70)

def train_model_kfold(x_train, y_train, x_test, model_fn, model_name, n_splits=10):
    """Train model with K-Fold CV"""
    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(len(x_test))
    fold_scores = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    print(f"\n  Training {model_name}...")
    
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
                callbacks=[lgb.early_stopping(stopping_rounds=100)]
            )

        # --- XGBoost ---
        elif "XGBoost" in model_name:
            model.fit(
                x_tr, y_tr,
                eval_set=[(x_val, y_val)],
                early_stopping_rounds=100,
                verbose=False
            )

        # --- CatBoost ---
        elif "CatBoost" in model_name:
            model.fit(x_tr, y_tr, eval_set=(x_val, y_val), verbose=False)

        # --- Other Models ---
        else:
            model.fit(x_tr, y_tr)

        
        # Predict
        oof_preds[val_idx] = np.clip(model.predict(x_val), 0, 6)
        test_preds += model.predict(x_test) / n_splits
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        fold_scores.append(fold_rmse)
        
        if fold % 3 == 0 or fold == n_splits:
            print(f"    Fold {fold}/{n_splits}: RMSE {fold_rmse:.4f}")
    
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"  ✓ {model_name} OOF RMSE: {oof_rmse:.4f}")
    
    return oof_preds, test_preds, oof_rmse

# Model 1: Tuned LightGBM (your best)
lgb_params_tuned = {
    'n_estimators': 2000, 'learning_rate': 0.02, 'max_depth': 5,
    'num_leaves': 20, 'min_child_samples': 25, 'subsample': 0.65,
    'colsample_bytree': 0.65, 'reg_alpha': 1.0, 'reg_lambda': 1.5,
    'min_split_gain': 0.01, 'random_state': RANDOM_SEED,
    'verbosity': -1, 'force_col_wise': True
}

oof_lgb, test_lgb, rmse_lgb = train_model_kfold(
    x_augmented, y, testin_x_augmented,
    lambda: LGBMRegressor(**lgb_params_tuned),
    "Tuned LightGBM", N_FOLDS
)

# Model 2: XGBoost
xgb_params = {
    'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 4,
    'min_child_weight': 3, 'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 1.0, 'reg_lambda': 1.5, 'gamma': 0.1,
    'random_state': RANDOM_SEED, 'verbosity': 0
}

oof_xgb, test_xgb, rmse_xgb = train_model_kfold(
    x_augmented, y, testin_x_augmented,
    lambda: XGBRegressor(**xgb_params),
    "XGBoost", N_FOLDS
)

# Model 3: CatBoost
cat_params = {
    'iterations': 1500, 'learning_rate': 0.02, 'depth': 5,
    'l2_leaf_reg': 5, 'random_strength': 0.3,
    'bagging_temperature': 0.5, 'random_state': RANDOM_SEED, 'verbose': False
}

oof_cat, test_cat, rmse_cat = train_model_kfold(
    x_augmented, y, testin_x_augmented,
    lambda: CatBoostRegressor(**cat_params),
    "CatBoost", N_FOLDS
)

# ============================================================================
# STEP 5: ENSEMBLE OPTIMIZATION
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Ensemble Optimization")
print("="*70)

all_oof = pd.DataFrame({
    'lgb_tuned': oof_lgb,
    'xgb': oof_xgb,
    'cat': oof_cat
})

all_test = pd.DataFrame({
    'lgb_tuned': test_lgb,
    'xgb': test_xgb,
    'cat': test_cat
})

print("\nIndividual Model Performance:")
for col in all_oof.columns:
    rmse = np.sqrt(mean_squared_error(y, all_oof[col]))
    print(f"  {col:15s}: {rmse:.4f}")

# Optimize weights
def weighted_rmse(weights, predictions, y_true):
    return np.sqrt(mean_squared_error(y_true, np.dot(predictions, weights)))

n_models = len(all_oof.columns)
initial_weights = np.ones(n_models) / n_models
constraints = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})
bounds = [(0, 1) for _ in range(n_models)]

opt_result = minimize(
    weighted_rmse, initial_weights,
    args=(all_oof.values, y),
    method='SLSQP', bounds=bounds, constraints=constraints
)

optimal_weights = opt_result.x
final_oof = np.dot(all_oof.values, optimal_weights)
final_test = np.dot(all_test.values, optimal_weights)
final_rmse = np.sqrt(mean_squared_error(y, final_oof))

print("\nOptimal Ensemble Weights:")
for col, weight in zip(all_oof.columns, optimal_weights):
    print(f"  {col:15s}: {weight:.4f}")
print(f"\n✓ FINAL ENSEMBLE CV RMSE: {final_rmse:.4f}")

# ============================================================================
# STEP 6: CREATE SUBMISSION
# ============================================================================
print("\n" + "="*70)
print("STEP 6: Creating Submission")
print("="*70)

submission = pd.DataFrame({
    'id': test_ids,
    'score': np.clip(final_test, 0, 6)
})

submission.to_csv('submission.csv', index=False)
print("✓ Submission saved: submission_complete_pipeline.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"""
Feature Engineering:
  - Base features: {x_clean.shape[1]}
  - Classifier probability features: {prob_features_train.shape[1]}
  - Total features: {x_augmented.shape[1]}

Model Performance:
  - Tuned LightGBM: {rmse_lgb:.4f}
  - XGBoost: {rmse_xgb:.4f}
  - CatBoost: {rmse_cat:.4f}
  - Final Ensemble: {final_rmse:.4f}

Expected LB Performance: 0.565-0.575
(Based on current 0.583 LB and CV improvement)

Next Steps:
  1. Submit and check LB score
  2. If LB > 0.57, add more advanced features
  3. If LB < 0.57, try different ensemble strategies
""")

print("="*70)
print("COMPLETE!")
print("="*70)










