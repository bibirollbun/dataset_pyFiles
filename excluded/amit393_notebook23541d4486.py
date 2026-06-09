# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pickle
import time
from contextlib import contextmanager
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Configuration - Minimal changes for 95%
# -------------------------
PATH_TO_DATA = '/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2'
AUTHOR = 'Your_Name'
SEED = 42
N_JOBS = -1
NUM_TIME_SPLITS = 5
SITE_NGRAMS = (1, 6)      # Extended to capture longer patterns
MAX_FEATURES = 150000     # Slightly increased


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def prepare_sparse_features(path_to_train, path_to_test, path_to_site_dict, vectorizer_params):
    times = [f'time{i}' for i in range(1, 11)]
    sites = [f'site{i}' for i in range(1, 11)]
    
    train_df = pd.read_csv(path_to_train, index_col='session_id', parse_dates=times)
    test_df = pd.read_csv(path_to_test, index_col='session_id', parse_dates=times)
    
    train_df = train_df.sort_values(by='time1')
    
    with open(path_to_site_dict, 'rb') as f:
        site2id = pickle.load(f)
    
    id2site = {v: k for k, v in site2id.items()}
    id2site[0] = 'unknown'
    
    train_sessions = train_df[sites].fillna(0).astype('int').apply(
        lambda row: ' '.join([id2site[i] for i in row]), axis=1
    ).tolist()
    
    test_sessions = test_df[sites].fillna(0).astype('int').apply(
        lambda row: ' '.join([id2site[i] for i in row]), axis=1
    ).tolist()
    
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train_sparse = vectorizer.fit_transform(train_sessions)
    X_test_sparse = vectorizer.transform(test_sessions)
    
    y_train = train_df['target'].astype('int').values
    
    train_site_ids = train_df[sites]
    test_site_ids = test_df[sites]
    train_times = train_df[times]
    test_times = test_df[times]
    
    return (X_train_sparse, X_test_sparse, y_train, vectorizer, 
            train_times, test_times, train_site_ids, test_site_ids)


def add_time_features(times):
    features = {}
    
    hour = times['time1'].dt.hour
    features['morning'] = ((hour >= 7) & (hour <= 11)).astype('int')
    features['day'] = ((hour >= 12) & (hour <= 18)).astype('int')
    features['evening'] = ((hour >= 19) & (hour <= 23)).astype('int')
    features['night'] = ((hour >= 0) & (hour <= 6)).astype('int')
    
    features['sess_duration'] = (times.max(axis=1) - times.min(axis=1)).dt.total_seconds()
    features['day_of_week'] = times['time1'].dt.weekday
    features['month'] = times['time1'].dt.month
    features['day_of_month'] = times['time1'].dt.day
    features['is_weekend'] = (times['time1'].dt.weekday >= 5).astype('int')
    features['year_month'] = (times['time1'].dt.year * 100 + times['time1'].dt.month) / 1e5
    
    for i in range(1, 10):
        delta = (times[f'time{i+1}'] - times[f'time{i}']).dt.total_seconds()
        features[f'delta_{i}'] = delta
    
    delta_cols = [f'delta_{i}' for i in range(1, 10)]
    delta_df = pd.DataFrame({k: features[k] for k in delta_cols})
    features['delta_mean'] = delta_df.mean(axis=1)
    features['delta_std'] = delta_df.std(axis=1)
    features['delta_max'] = delta_df.max(axis=1)
    features['delta_min'] = delta_df.min(axis=1)
    features['delta_median'] = delta_df.median(axis=1)
    
    # NEW: Ratio features
    features['delta_range'] = features['delta_max'] - features['delta_min']
    features['delta_cv'] = features['delta_std'] / (features['delta_mean'] + 1)
    
    return pd.DataFrame(features)


def add_site_features(site_ids):
    features = {}
    
    features['unique_sites'] = site_ids.nunique(axis=1)
    features['total_sites'] = site_ids.notna().sum(axis=1)
    
    def count_revisits(row):
        valid_sites = row.dropna().values
        return len(valid_sites) - len(set(valid_sites))
    
    features['num_revisits'] = site_ids.apply(count_revisits, axis=1)
    features['has_revisit'] = (features['num_revisits'] > 0).astype('int')
    features['site_diversity'] = features['unique_sites'] / features['total_sites'].replace(0, 1)
    
    def max_site_frequency(row):
        valid_sites = row.dropna().values
        if len(valid_sites) == 0:
            return 0
        return max(pd.Series(valid_sites).value_counts().values)
    
    features['max_site_freq'] = site_ids.apply(max_site_frequency, axis=1)
    
    first_site = site_ids.iloc[:, 0]
    last_site = site_ids.apply(lambda row: row.dropna().iloc[-1] if len(row.dropna()) > 0 else np.nan, axis=1)
    features['same_first_last'] = (first_site == last_site).astype('int')
    
    # NEW: Entropy of site distribution
    def entropy(row):
        valid_sites = row.dropna().values
        if len(valid_sites) == 0:
            return 0
        _, counts = np.unique(valid_sites, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-10))
    
    features['site_entropy'] = site_ids.apply(entropy, axis=1)
    
    return pd.DataFrame(features)


def combine_features(X_sparse, time_features, site_features):
    scaler = StandardScaler()
    dense_features = pd.concat([time_features, site_features], axis=1)
    dense_scaled = scaler.fit_transform(dense_features.fillna(0))
    
    dense_sparse = csr_matrix(dense_scaled)
    X_combined = hstack([X_sparse, dense_sparse])
    
    return X_combined, scaler, dense_features.columns.tolist()


# -------------------------
# Main Pipeline
# -------------------------
with timer('Loading and preparing sparse features'):
    (X_train_sparse, X_test_sparse, y_train, vectorizer, 
     train_times, test_times, train_site_ids, test_site_ids) = prepare_sparse_features(
        path_to_train=os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
        path_to_test=os.path.join(PATH_TO_DATA, 'test_sessions.csv'),
        path_to_site_dict=os.path.join(PATH_TO_DATA, 'site_dic.pkl'),
        vectorizer_params={
            'ngram_range': SITE_NGRAMS,
            'max_features': MAX_FEATURES,
            'tokenizer': lambda s: s.split()
        }
    )
    print(f'Sparse features shape: {X_train_sparse.shape}')


with timer('Engineering additional features'):
    train_time_features = add_time_features(train_times)
    test_time_features = add_time_features(test_times)
    
    train_site_features = add_site_features(train_site_ids)
    test_site_features = add_site_features(test_site_ids)
    
    X_train_final, scaler, feature_names = combine_features(
        X_train_sparse, train_time_features, train_site_features
    )
    
    dense_test = pd.concat([test_time_features, test_site_features], axis=1)
    dense_test_scaled = scaler.transform(dense_test.fillna(0))
    X_test_final = hstack([X_test_sparse, csr_matrix(dense_test_scaled)])
    
    print(f'Final feature shape: {X_train_final.shape}')


with timer('Cross-validation with XGBoost'):
    # Fine-tuned parameters for better performance
    xgb_model = XGBClassifier(
        n_estimators=400,           # Increased from 300
        max_depth=7,                # Slightly deeper
        learning_rate=0.08,         # Lower learning rate with more trees
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,         # Less conservative
        gamma=0.05,                 # Less pruning
        reg_alpha=0.05,
        reg_lambda=1.5,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalance
        random_state=SEED,
        n_jobs=N_JOBS
    )
    
    time_split = TimeSeriesSplit(n_splits=NUM_TIME_SPLITS)
    cv_scores = []
    
    print(f'\nPerforming {NUM_TIME_SPLITS}-fold time-series cross-validation...')
    for fold, (train_idx, val_idx) in enumerate(time_split.split(X_train_final), 1):
        X_tr, X_val = X_train_final[train_idx], X_train_final[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        xgb_model.fit(X_tr, y_tr, verbose=False)
        val_pred = xgb_model.predict_proba(X_val)[:, 1]
        
        roc_auc = roc_auc_score(y_val, val_pred)
        cv_scores.append(roc_auc)
        print(f'Fold {fold} ROC-AUC: {roc_auc:.5f}')
    
    print(f'\nMean CV ROC-AUC: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})')


with timer('Training final model and prediction'):
    xgb_model.fit(X_train_final, y_train, verbose=False)
    test_pred = xgb_model.predict_proba(X_test_final)[:, 1]
    
    pred_df = pd.DataFrame(
        test_pred, 
        index=np.arange(1, test_pred.shape[0] + 1),
        columns=['target']
    )
    pred_df.to_csv(f'submission.csv', index_label='session_id')
    
    print(f'\n✅ Submission created: submission.csv')
    print(f'Predictions shape: {pred_df.shape}')
    print(f'\nPrediction statistics:')
    print(f'  Mean: {test_pred.mean():.4f}')
    print(f'  Std: {test_pred.std():.4f}')
    print(f'  Min: {test_pred.min():.4f}')
    print(f'  Max: {test_pred.max():.4f}')
    print(f'\nPreview:')
    print(pred_df.head(10))

