!pip install py7zr --quiet



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold, 
    train_test_split
)
from sklearn.metrics import (
    roc_auc_score
)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from typing import Callable, List, Optional
from tqdm import tqdm

import gc
from pathlib import Path
import py7zr
import joblib
import hashlib
import pickle
import json

sns.set_theme()


ROOT_DIR = Path('/kaggle/input/kkbox-music-recommendation-challenge')

file_paths = [
    ROOT_DIR / 'songs.csv.7z',
    ROOT_DIR / 'song_extra_info.csv.7z', 
    ROOT_DIR / 'members.csv.7z',
    ROOT_DIR / 'train.csv.7z',
    ROOT_DIR / 'test.csv.7z'
]

ROOT_DIR_EXTRACT = Path('/kaggle/working/extracted/')
ROOT_DIR_EXTRACT.mkdir(exist_ok=True)

for archive_path in file_paths:
    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(ROOT_DIR_EXTRACT)

SONG_PATH_EXTRACT = ROOT_DIR_EXTRACT / 'songs.csv'
SONG_EXTRA_PATH_EXTRACT = ROOT_DIR_EXTRACT /'song_extra_info.csv'
MEMBERS_PATH_EXTRACT = ROOT_DIR_EXTRACT / 'members.csv'
TRAIN_PATH_EXTRACT = ROOT_DIR_EXTRACT / 'train.csv'
TEST_PATH_EXTRACT = ROOT_DIR_EXTRACT / 'test.csv'


songs = pd.read_csv(SONG_PATH_EXTRACT)
song_extra_info = pd.read_csv(SONG_EXTRA_PATH_EXTRACT)
members = pd.read_csv(MEMBERS_PATH_EXTRACT)
train = pd.read_csv(TRAIN_PATH_EXTRACT)
test = pd.read_csv(TEST_PATH_EXTRACT)


for i, df in enumerate([songs, song_extra_info, members, train, test]):
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


del song_extra_info
gc.collect()


def optimize_dtypes(df, column_types=None):
    """Memory usage optimization"""
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype('category')
            
    if 'id' in df.columns:
        df['id'] = df['id'].astype('category')

    if 'genre_ids' in df.columns:
        df['genre_ids'] = df['genre_ids'].astype('object')
    
    for col in df.select_dtypes(['int64']).columns:
        c_min = df[col].min()
        c_max = df[col].max()
        
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
    
    for col in df.select_dtypes(['float64']).columns:
        df[col] = df[col].astype(np.float32)
    
    return df

for df in [songs, members, train, test]:
    df = optimize_dtypes(df)


for i, df in enumerate([songs, members, train, test]):
    DF_NAME = dict(zip(range(4), ['songs', 'members', 'train', 'test']))[i]
    print('=' * 60)
    print(f'TABLE {DF_NAME}:')
    print(f"SIZE: {df.shape}")
    desc = pd.concat([
                round(df.isna().sum() / len(df) * 100, 1), 
                df.dtypes, 
                df.nunique()
            ], axis=1).rename(
                columns={
                    0: 'NA%', 
                    1: 'DTYPE',
                    2: 'NUNIQUE'
                })
    print(desc)
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")



train['target'].value_counts()


MAIN_FEATURES = {
    'song_id': 'category',
    'id': 'category',
    'msno': 'category',
    'song_length': 'int32',
    'artist_name': 'category',
    'composer': 'category',
    'lyricist': 'category',
    'language': 'category',
    'city': 'category',
    'bd': 'int16',
    'gender': 'category',
    'genre_ids': 'object',
    'registration_init_time': 'int32',
    'registered_via': 'category',
    'expiration_date': 'int32',
    'source_system_tab': 'category',
    'source_screen_name': 'category',
    'source_type': 'category',
}


def interpolate_songs_features(songs_df: pd.DataFrame):
    '''Interpolate all song features'''
    print('Start interpolating Songs DF')
    # It is creating a big noise, after big leakages
    """ global_modes = {}
    columns_to_fill = ['genre_ids', 'language', 'composer', 'lyricist']
    for col in columns_to_fill:
        mode_series = songs_df[col].mode()
        global_modes[col] = mode_series[0] if not mode_series.empty else None
    
    for col in ['genre_ids', 'language', 'composer']:
        print(f'\tInterpolating {col}-column...')
        artist_modes = songs_df.groupby('artist_name')[col].apply(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        )
        songs_df[col] = songs_df[col].fillna(
            songs_df['artist_name'].map(artist_modes)
        ).fillna(global_modes[col])

    print('\tInterpolating lyricist-column...')
    com_modes = songs_df.groupby('composer')['lyricist'].apply(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )
    songs_df['lyricist'] = songs_df['lyricist'].fillna(
        songs_df['composer'].map(com_modes)
    ).fillna(global_modes['lyricist'])
    
    del global_modes, com_modes, artist_modes
    gc.collect()"""

    songs_df['language'] = songs_df['language'].cat.add_categories('Na').fillna('Na')
    songs_df['artist_name'] = songs_df['artist_name'].cat.add_categories('unknown_artist').fillna('unknown_artist')
    songs_df['composer'] = songs_df['composer'].cat.add_categories('unknown_composer').fillna('unknown_composer')
    songs_df['lyricist'] = songs_df['lyricist'].cat.add_categories('unknown_lyricist').fillna('unknown_lyricist')
    
    return songs_df



class GenreFeatureTransformer:
    """
    ĞšĞ¾Ğ¼Ğ±Ğ¸Ğ½Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğ¹ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€ Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ² Ğ´Ğ»Ñ� Ğ¼Ğ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğ³Ğ¾ ĞºĞ°Ñ‡ĞµÑ�Ñ‚Ğ²Ğ°
    Ñ� Ğ¿Ğ¾Ğ´Ğ´ĞµÑ€Ğ¶ĞºĞ¾Ğ¹ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ� Ğ¸ Ğ·Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ¸
    """

    def __init__(self, n_svd_components=15, n_top_genres=15):
        self.n_svd_components = n_svd_components
        self.n_top_genres = n_top_genres
        self.unique_genres = None
        self.svd = None
        self.scaler = None
        self.top_genres = None
        self.genre_importance = None
        self.genre_counter = None  

        self.rare_genres = None
        self.fitted = False
        self.version = "1.0"

    def fit(self, songs_df, target_series=None):
        """
        Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€Ğ°
        target_series: ĞµÑ�Ğ»Ğ¸ ĞµÑ�Ñ‚ÑŒ, Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼ Ğ´Ğ»Ñ� feature selection
        """
        print("ğŸ”� Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ· Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²...")

        unique_genres = set()
        self.genre_counter = Counter()

        for genres in songs_df['genre_ids'].fillna(''):
            if genres:
                genre_list = str(genres).split('|')
                unique_genres.update(genre_list)
                self.genre_counter.update(genre_list)

        self.unique_genres = list(unique_genres)
        print(f"   Ğ�Ğ°Ğ¹Ğ´ĞµĞ½Ğ¾ {len(self.unique_genres)} ÑƒĞ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²")

        self.top_genres = [g for g, _ in self.genre_counter.most_common(self.n_top_genres)]
        print(f"   Ğ’Ñ‹Ğ±Ñ€Ğ°Ğ½Ğ¾ {len(self.top_genres)} Ñ‚Ğ¾Ğ¿-Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²")

        if self.genre_counter:
            rare_threshold = np.percentile(list(self.genre_counter.values()), 20)
            self.rare_genres = {g for g, cnt in self.genre_counter.items() if cnt <= rare_threshold}

        genre_features = self._create_extended_features(songs_df)

        if target_series is not None:
            self._compute_genre_importance(genre_features, target_series)

        binary_matrix = self._create_binary_matrix(songs_df)

        print("   ĞŸÑ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµĞ¼ SVD...")
        self.svd = TruncatedSVD(
            n_components=self.n_svd_components,
            n_iter=10,
            random_state=42,
            algorithm='randomized'
        )

        svd_features = self.svd.fit_transform(binary_matrix)
        explained_variance = self.svd.explained_variance_ratio_.sum()
        print(f"   SVD Ğ¾Ğ±ÑŠÑ�Ñ�Ğ½Ñ�ĞµÑ‚ {explained_variance:.2%} Ğ´Ğ¸Ñ�Ğ¿ĞµÑ€Ñ�Ğ¸Ğ¸")

        self.scaler = StandardScaler()
        self.scaler.fit(svd_features)

        self.fitted = True
        print("âœ… Ğ¢Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½")
        return self

    def _create_binary_matrix(self, df):
        """Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµÑ‚ Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½ÑƒÑ� Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ñƒ Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²"""
        if not self.unique_genres:
            raise ValueError("Ğ¡Ğ½Ğ°Ñ‡Ğ°Ğ»Ğ° Ğ½ÑƒĞ¶Ğ½Ğ¾ Ğ¾Ğ±ÑƒÑ‡Ğ¸Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€ (Ğ²Ñ‹Ğ·Ğ¾Ğ²Ğ¸Ñ‚Ğµ fit)")

        n_samples = len(df)
        n_genres = len(self.unique_genres)
        genre_to_idx = {g: i for i, g in enumerate(self.unique_genres)}

        rows, cols, data = [], [], []

        for i, genres_str in enumerate(df['genre_ids'].fillna('')):
            if genres_str:
                for genre in str(genres_str).split('|'):
                    if genre in genre_to_idx:
                        rows.append(i)
                        cols.append(genre_to_idx[genre])
                        data.append(1.0)

        binary_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_samples, n_genres),
            dtype=np.float32
        )

        return binary_matrix

    def _create_extended_features(self, df):
        """Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµÑ‚ Ñ€Ğ°Ñ�ÑˆĞ¸Ñ€ĞµĞ½Ğ½Ñ‹Ğµ Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸"""
        features = pd.DataFrame(index=df.index)

        features['genre_count'] = df['genre_ids'].apply(
            lambda x: len(str(x).split('|')) if x else 0
        )

        def avg_popularity(genres_str):
            if not genres_str:
                return 0
            genres = str(genres_str).split('|')
            if not genres:
                return 0
            return np.mean([self.genre_counter.get(g, 0) for g in genres])

        features['genre_avg_popularity'] = df['genre_ids'].apply(avg_popularity)

        for i, genre in enumerate(self.top_genres[:30]):
            features[f'has_genre_{genre}'] = df['genre_ids'].apply(
                lambda x: 1 if genre in str(x) else 0
            )

        def genre_entropy(genres_str):
            if not genres_str:
                return 0
            genres = str(genres_str).split('|')
            if len(genres) <= 1:
                return 0

            counts = Counter(genres)
            probs = np.array(list(counts.values())) / len(genres)
            return -np.sum(probs * np.log(probs + 1e-10))

        features['genre_entropy'] = df['genre_ids'].apply(genre_entropy)

        if self.rare_genres:
            features['has_rare_genre'] = df['genre_ids'].apply(
                lambda x: 1 if any(g in self.rare_genres for g in str(x).split('|')) else 0
            )

        return features

    def _compute_genre_importance(self, features, target):
        """Ğ’Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµÑ‚ Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ² Ñ‡ĞµÑ€ĞµĞ· Ğ±Ñ‹Ñ�Ñ‚Ñ€Ñ‹Ğ¹ LightGBM"""
        print("   Ğ’Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµĞ¼ Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²...")

        importance_params = {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'num_leaves': 7,
            'verbose': -1,
            'random_state': 42
        }

        model = lgb.LGBMClassifier(**importance_params)
        model.fit(features, target, verbose=False)

        importance_df = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.genre_importance = importance_df

    def transform(self, df):
        """ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµÑ‚ Ğ¶Ğ°Ğ½Ñ€Ñ‹ Ğ² Ñ„Ğ¸Ñ‡Ğ¸"""
        if not self.fitted:
            raise ValueError("Ğ¡Ğ½Ğ°Ñ‡Ğ°Ğ»Ğ° Ğ½ÑƒĞ¶Ğ½Ğ¾ Ğ¾Ğ±ÑƒÑ‡Ğ¸Ñ‚ÑŒ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€ (Ğ²Ñ‹Ğ·Ğ¾Ğ²Ğ¸Ñ‚Ğµ fit)")

        print("ğŸ”„ Ğ¢Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¶Ğ°Ğ½Ñ€Ñ‹...")

        binary_matrix = self._create_binary_matrix(df)

        svd_features = self.svd.transform(binary_matrix)
        svd_features_scaled = self.scaler.transform(svd_features)

        extended_features = self._create_extended_features(df)

        svd_df = pd.DataFrame(
            svd_features_scaled,
            columns=[f'genre_svd_{i}' for i in range(self.n_svd_components)],
            index=df.index
        )

        if self.genre_importance is not None:
            top_extended = self.genre_importance['feature'].head(10).tolist()

            existing_cols = [col for col in top_extended if col in extended_features.columns]
            if existing_cols:
                extended_selected = extended_features[existing_cols]
            else:
                extended_selected = extended_features
        else:

            extended_selected = extended_features

        result = pd.concat([svd_df, extended_selected], axis=1)

        print(f"   Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¾ {result.shape[1]} Ñ„Ğ¸Ñ‡")
        return result

    def fit_transform(self, df, target=None):
        self.fit(df, target)
        return self.transform(df)

    def save(self, path, method='joblib'):
        """
        Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµÑ‚ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€ Ğ½Ğ° Ğ´Ğ¸Ñ�Ğº

        Parameters:
        -----------
        path : str Ğ¸Ğ»Ğ¸ Path
            ĞŸÑƒÑ‚ÑŒ Ğ´Ğ»Ñ� Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ�
        method : str
            ĞœĞµÑ‚Ğ¾Ğ´ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ�: 'joblib', 'pickle' Ğ¸Ğ»Ğ¸ 'json'
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        print(f"ğŸ’¾ Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�Ñ� Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€ Ğ² {path}")

        if method == 'joblib':

            joblib.dump(self, path, compress=3)

        elif method == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(self, f)

        elif method == 'json':

            self._save_to_json(path)

        else:
            raise ValueError(f"Ğ�ĞµĞ¸Ğ·Ğ²ĞµÑ�Ñ‚Ğ½Ñ‹Ğ¹ Ğ¼ĞµÑ‚Ğ¾Ğ´ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ�: {method}")

        print(f"âœ… Ğ¢Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½ ({method})")

        self._save_metadata(path)

    def _save_to_json(self, path):
        """Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ğµ Ğ² JSON (Ñ‡Ğ°Ñ�Ñ‚Ğ¸Ñ‡Ğ½Ğ°Ñ� Ñ�ĞµÑ€Ğ¸Ğ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�)"""
        import json

        data = {
            'version': self.version,
            'n_svd_components': self.n_svd_components,
            'n_top_genres': self.n_top_genres,
            'unique_genres': self.unique_genres,
            'top_genres': self.top_genres,
            'fitted': self.fitted,
        }

        if self.svd is not None:
            data['svd'] = {
                'components': self.svd.components_.tolist(),
                'explained_variance': self.svd.explained_variance_.tolist(),
                'explained_variance_ratio': self.svd.explained_variance_ratio_.tolist(),
                'singular_values': self.svd.singular_values_.tolist(),
            }

        if self.scaler is not None:
            data['scaler'] = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
            }

        if self.genre_counter is not None:
            data['genre_counter'] = dict(self.genre_counter.most_common())

        if self.rare_genres is not None:
            data['rare_genres'] = list(self.rare_genres)

        if self.genre_importance is not None:
            data['genre_importance'] = self.genre_importance.to_dict('records')

        with open(path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_metadata(self, path):
        """Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ğµ Ğ¼ĞµÑ‚Ğ°Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€Ğ°"""
        metadata = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'version': self.version,
            'n_unique_genres': len(self.unique_genres) if self.unique_genres else 0,
            'n_top_genres': len(self.top_genres) if self.top_genres else 0,
            'n_svd_components': self.n_svd_components,
            'fitted': self.fitted,
            'total_features': self._get_total_features_count(),
        }

        metadata_path = path.with_suffix('.meta.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _get_total_features_count(self):
        """Ğ’Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚ Ğ¾Ğ±Ñ‰ĞµĞµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ„Ğ¸Ñ‡, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ğ±ÑƒĞ´ĞµÑ‚ Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ²Ğ°Ñ‚ÑŒ transform"""
        if not self.fitted:
            return 0
        return self.n_svd_components + len(self.top_genres[:30]) + 4  

    @classmethod
    def load(cls, path, method='auto'):
        """
        Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµÑ‚ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€ Ñ� Ğ´Ğ¸Ñ�ĞºĞ°

        Parameters:
        -----------
        path : str Ğ¸Ğ»Ğ¸ Path
            ĞŸÑƒÑ‚ÑŒ Ğº Ñ„Ğ°Ğ¹Ğ»Ñƒ
        method : str
            ĞœĞµÑ‚Ğ¾Ğ´ Ğ·Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ¸: 'auto', 'joblib', 'pickle', 'json'
        """
        path = Path(path)

        if method == 'auto':

            if path.suffix == '.pkl':
                method = 'pickle'
            elif path.suffix == '.joblib':
                method = 'joblib'
            elif path.suffix == '.json':
                method = 'json'
            else:

                for ext in ['.joblib', '.pkl', '.json']:
                    if path.with_suffix(ext).exists():
                        path = path.with_suffix(ext)
                        if ext == '.joblib':
                            method = 'joblib'
                        elif ext == '.pkl':
                            method = 'pickle'
                        elif ext == '.json':
                            method = 'json'
                        break

        print(f"ğŸ“‚ Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°Ñ� Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€ Ğ¸Ğ· {path} ({method})")

        if method == 'joblib':
            if not path.exists():

                path = path.with_suffix('.joblib')
            transformer = joblib.load(path)

        elif method == 'pickle':
            if not path.exists():

                path = path.with_suffix('.pkl')
            with open(path, 'rb') as f:
                transformer = pickle.load(f)

        elif method == 'json':
            if not path.exists():

                path = path.with_suffix('.json')
            transformer = cls._load_from_json(path)

        else:
            raise ValueError(f"Ğ�ĞµĞ¸Ğ·Ğ²ĞµÑ�Ñ‚Ğ½Ñ‹Ğ¹ Ğ¼ĞµÑ‚Ğ¾Ğ´ Ğ·Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ¸: {method}")

        print(f"âœ… Ğ¢Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€ Ğ·Ğ°Ğ³Ñ€ÑƒĞ¶ĞµĞ½")
        print(f"   Ğ’ĞµÑ€Ñ�Ğ¸Ñ�: {transformer.version}")
        print(f"   Ğ£Ğ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²: {len(transformer.unique_genres) if transformer.unique_genres else 0}")
        print(f"   Ğ¢Ğ¾Ğ¿-Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²: {len(transformer.top_genres) if transformer.top_genres else 0}")
        print(f"   SVD ĞºĞ¾Ğ¼Ğ¿Ğ¾Ğ½ĞµĞ½Ñ‚: {transformer.n_svd_components}")

        return transformer

    @classmethod
    def _load_from_json(cls, path):
        """Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ¸Ğ· JSON"""
        import json
        import numpy as np

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        transformer = cls(
            n_svd_components=data.get('n_svd_components', 20),
            n_top_genres=data.get('n_top_genres', 50)
        )

        transformer.version = data.get('version', '1.0')
        transformer.unique_genres = data.get('unique_genres')
        transformer.top_genres = data.get('top_genres')
        transformer.fitted = data.get('fitted', False)

        if 'genre_counter' in data:
            transformer.genre_counter = Counter(data['genre_counter'])

        if 'rare_genres' in data:
            transformer.rare_genres = set(data['rare_genres'])

        if 'svd' in data:
            svd_data = data['svd']
            transformer.svd = TruncatedSVD(
                n_components=transformer.n_svd_components,
                random_state=42
            )

            transformer.svd.components_ = np.array(svd_data['components'])
            transformer.svd.explained_variance_ = np.array(svd_data['explained_variance'])
            transformer.svd.explained_variance_ratio_ = np.array(svd_data['explained_variance_ratio'])
            transformer.svd.singular_values_ = np.array(svd_data['singular_values'])

        if 'scaler' in data:
            scaler_data = data['scaler']
            transformer.scaler = StandardScaler()
            transformer.scaler.mean_ = np.array(scaler_data['mean'])
            transformer.scaler.scale_ = np.array(scaler_data['scale'])
            transformer.scaler.var_ = transformer.scaler.scale_ ** 2
            transformer.scaler.n_features_in_ = len(transformer.scaler.mean_)

        if 'genre_importance' in data:
            transformer.genre_importance = pd.DataFrame(data['genre_importance'])

        return transformer

    def get_feature_names(self):
        """
        Ğ’Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚ Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ¸Ğ¼ĞµĞ½ Ñ„Ğ¸Ñ‡, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ñ�Ğ¾Ğ·Ğ´Ğ°ĞµÑ‚ transform

        Returns:
        --------
        list: Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ¸Ğ¼ĞµĞ½ Ñ„Ğ¸Ñ‡
        """
        if not self.fitted:
            return []

        feature_names = []

        feature_names.extend([f'genre_svd_{i}' for i in range(self.n_svd_components)])

        feature_names.extend(['genre_count', 'genre_avg_popularity', 
                            'genre_entropy', 'has_rare_genre'])

        if self.top_genres:
            feature_names.extend([f'has_genre_{g}' for g in self.top_genres[:30]])

        return feature_names

    def get_info(self):
        """
        Ğ’Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚ Ğ¸Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€Ğµ

        Returns:
        --------
        dict: Ğ˜Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ñ� Ğ¾ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ¾Ñ€Ğµ
        """
        return {
            'version': self.version,
            'fitted': self.fitted,
            'n_unique_genres': len(self.unique_genres) if self.unique_genres else 0,
            'n_top_genres': len(self.top_genres) if self.top_genres else 0,
            'n_svd_components': self.n_svd_components,
            'total_features': len(self.get_feature_names()),
            'top_5_genres': self.top_genres[:5] if self.top_genres else [],
            'svd_explained_variance': self.svd.explained_variance_ratio_.sum() if self.svd else 0,
        }

def latent_genre_adding_songs(songs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµÑ‚ Ğ»Ğ°Ñ‚ĞµĞ½Ñ‚Ğ½Ñ‹Ğµ Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸ Ğº songs_df.
    Ğ•Ñ�Ğ»Ğ¸ Ñ„Ğ¸Ñ‡Ğ¸ ÑƒĞ¶Ğµ Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ñ‹ - Ğ·Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµÑ‚ Ğ¸Ñ…, Ğ¸Ğ½Ğ°Ñ‡Ğµ Ñ�Ğ¾Ğ·Ğ´Ğ°ĞµÑ‚ Ğ½Ğ¾Ğ²Ñ‹Ğµ.
    """

    if 'genre_ids' not in songs_df.columns:
        raise ValueError("DataFrame Ğ´Ğ¾Ğ»Ğ¶ĞµĞ½ Ñ�Ğ¾Ğ´ĞµÑ€Ğ¶Ğ°Ñ‚ÑŒ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºÑƒ 'genre_ids'")

    data_hash = hashlib.md5(
        songs_df['genre_ids'].fillna('').astype(str).values.tobytes()
    ).hexdigest()[:12]

    cache_dir = Path("cache/genre_features")
    cache_dir.mkdir(parents=True, exist_ok=True)

    features_path = cache_dir / f"features_{data_hash}.pkl"
    transformer_path = cache_dir / f"transformer_{data_hash}.pkl"

    if features_path.exists() and transformer_path.exists():
        print(f"ğŸ“‚ Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°Ñ� Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ½Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸ Ğ¸Ğ· ĞºĞµÑˆĞ°...")

        with open(features_path, 'rb') as f:
            genre_features = pickle.load(f)

        if len(genre_features) == len(songs_df):
            result_df = pd.concat([songs_df.reset_index(drop=True), 
                                 genre_features.reset_index(drop=True)], axis=1)
            print(f"âœ… Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶ĞµĞ½Ğ¾ {genre_features.shape[1]} Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²Ñ‹Ñ… Ñ„Ğ¸Ñ‡")
            return result_df

    print("ğŸ”„ Ğ¡Ğ¾Ğ·Ğ´Ğ°Ñ� Ğ½Ğ¾Ğ²Ñ‹Ğµ Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²Ñ‹Ğµ Ñ„Ğ¸Ñ‡Ğ¸...")

    transformer = GenreFeatureTransformer()
    genre_features = transformer.fit_transform(songs_df)

    with open(features_path, 'wb') as f:
        pickle.dump(genre_features, f)

    with open(transformer_path, 'wb') as f:
        pickle.dump(transformer, f)

    result_df = pd.concat([songs_df.reset_index(drop=True), 
                         genre_features.reset_index(drop=True)], axis=1)

    print(f"âœ… Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¾ {genre_features.shape[1]} Ğ¶Ğ°Ğ½Ñ€Ğ¾Ğ²Ñ‹Ñ… Ñ„Ğ¸Ñ‡")
    return result_df
    


def feature_adding_song(
    songs_df: pd.DataFrame, 
    train_df: pd.DataFrame,
    feature_blocks: List[Callable[..., None]]
):
    '''Creating and adding new song features'''
    print('Adding new features')
    for block in feature_blocks:
        try: 
            songs_df = block(songs_df, train_df)
        except:
            songs_df = block(songs_df)
    
    print('All new features has been added!')
    return songs_df

def prepare_songs(
    songs_df: pd.DataFrame, 
    train_df: pd.DataFrame,
    feature_blocks
):
    '''Preparing songs dataset'''
    print('='*60)
    print('Start Prepare Songs DF')
    print('='*60)
    cols = songs_df.columns
    songs_df = songs_df.astype({col: MAIN_FEATURES[col] for col in MAIN_FEATURES if col in cols})
    songs_df = interpolate_songs_features(songs_df)
    songs_df = feature_adding_song(songs_df, train_df, feature_blocks)
    songs_df = songs_df.drop(columns=['genre_ids'])
    print('='*60)
    print('Songs DF has been prepared success!')
    print('='*60)
    return songs_df
    


songs_prepared = prepare_songs(
    songs.copy(deep=True), 
    train, 
    [latent_genre_adding_songs])


songs_prepared.columns


def interpolate_members_features(members_df: pd.DataFrame):
    '''Interpolate all members features'''
    print('Start interpolating Members DF')
    print('\tInterpolating gender-column')
    members_df = interpolate_gender(members_df)
    return members_df

def interpolate_gender(members_df: pd.DataFrame):
    '''Filling NA-actions of gender column'''
    #members_df['gender'] = members_df['gender'].cat.add_categories('Na').fillna(members_df['gender'].mode()[0]) >57% - NA  
    members_df['gender'] = members_df['gender'].cat.add_categories('Na').fillna('Na')
    return members_df


def year_adding_members(members_df: pd.DataFrame):
    '''Creating and adding year features'''
    print('\tCreating year features')
    members_df['registration_year'] = pd.to_datetime(
    members_df['registration_init_time'].astype(str), format='%Y%m%d').dt.year

    members_df['expiration_year'] = pd.to_datetime( 
        members_df['expiration_date'].astype(str), format='%Y%m%d').dt.year

    members_df['delta_exp_reg_year'] = members_df['expiration_year'] - members_df['registration_year']

    return members_df



def feature_adding_members(
    members_df: pd.DataFrame, 
    train_df: pd.DataFrame,
    feature_blocks: List[Callable[..., None]]
):
    '''Creating and adding new member features'''
    for block in feature_blocks:
        try: 
            members_df = block(members_df, train_df)
        except:
            members_df = block(members_df)

    print('All new features has been added!')
    return members_df

def kill_outliers(members_df: pd.DataFrame):
    '''Killing outliers of all members features'''
    print('\tDropping age-columns')
    return members_df.drop(columns=['bd'])

def prepare_members(
    members_df: pd.DataFrame, 
    train_df: pd.DataFrame,
    feature_blocks: List[Callable[..., None]]
):
    '''Preparing members dataset'''
    print('='*60)
    print('Start Prepare Members DF')
    print('='*60)
    cols = members_df.columns
    members_df = members_df.astype({col: MAIN_FEATURES[col] for col in MAIN_FEATURES if col in cols})
    members_df = interpolate_members_features(members_df)
    members_df = feature_adding_members(members_df, train_df, feature_blocks)
    members_df = members_df.drop(columns=['registration_init_time', 'expiration_date']) 
    print('='*60)
    print('Members DF has been prepared success!')
    print('='*60)
    return members_df
    


members_prepared = prepare_members(
    members.copy(deep=True), 
    train,
    [kill_outliers, year_adding_members] 
)


members_prepared.columns


def interpolate_train_test_features(df: pd.DataFrame) -> pd.DataFrame:
    '''Interpolate all train features with minimal memory usage'''
    print('Start interpolating train DF')
    for cat in ['source_system_tab', 'source_screen_name', 'source_type']:
        print(f'\tInterpolating {cat}-column')
        df[cat] = df[cat].cat.add_categories('Na').fillna(df[cat].mode()[0])
    return df


def history_stats_adding_song_enchanted(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    songs_df: pd.DataFrame, 
    members_df: pd.DataFrame, 
    train: int
) -> pd.DataFrame:
    '''
    Adding historical features for train/test.
    In test mode, we use the last values from train data.
    '''
    history_columns = [
        'msno_total_interactions_before',
        'msno_artist_interactions_before', 
        'msno_lyricist_interactions_before',  
        'msno_composer_interactions_before',  
        'msno_language_interactions_before',    
        'msno_source_system_tab_interactions_before',
        'msno_source_screen_name_interactions_before',
        'msno_source_type_interactions_before',

        'gender_total_interactions_before',
        'gender_artist_interactions_before',
        'gender_composer_interactions_before',
        'gender_lyricist_interactions_before',

        'msno_target_mean_before',
        'artist_popularity_before',
        'composer_popularity_before',      
        'lyricist_popularity_before',  
        'language_popularity_before',          
        'source_system_tab_popularity_before',
        'source_screen_name_popularity_before',
        'source_type_popularity_before',

        'gender_target_mean_before',
        'gender_artist_popularity_before',
        'gender_composer_popularity_before',
        'gender_lyricist_popularity_before',

        'msno_artist_target_mean_before',  
        'msno_composer_target_mean_before',
        'msno_lyricist_target_mean_before',   
        'msno_language_target_mean_before',
        'msno_source_system_tab_target_mean_before',
        'msno_source_screen_name_target_mean_before',
        'msno_source_type_target_mean_before',

        'gender_artist_target_mean_before',
        'gender_composer_target_mean_before',
        'gender_lyricist_target_mean_before'
    ]

    INVALID_VALUES = {
        'composer': ['unknown_composer'],
        'lyricist': ['unknown_lyricist'],
        'artist_name': ['unknown_artist'],
        'language': ['Na', 'unknown_language'],
        'gender': ['Na', 'unknown_gender']
    }

    def is_valid_value(entity_col, value):
        """ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµÑ‚, Ñ�Ğ²Ğ»Ñ�ĞµÑ‚Ñ�Ñ� Ğ»Ğ¸ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ½Ñ‹Ğ¼ Ğ´Ğ»Ñ� Ğ²Ñ‹Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½Ğ¸Ñ� Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ¸"""
        if value is None or pd.isna(value):
            return False
        if entity_col in INVALID_VALUES and value in INVALID_VALUES[entity_col]:
            return False
        return True

    def create_valid_mask(series, entity_col):
        """Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµÑ‚ Ğ±ÑƒĞ»ĞµĞ²ÑƒÑ� Ğ¼Ğ°Ñ�ĞºÑƒ Ğ´Ğ»Ñ� Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ½Ñ‹Ñ… Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹ Ğ±ĞµĞ· NaN"""
        mask = series.apply(lambda x: is_valid_value(entity_col, x))

        return mask.fillna(False)

    if train == 1:
        print("TRAIN mode: creating historical features")

        song_info = songs_df[['song_id', 'artist_name', 'composer', 'lyricist', 'language']].set_index('song_id')
        df['artist_name'] = df['song_id'].map(song_info['artist_name'])
        df['composer'] = df['song_id'].map(song_info['composer'])
        df['lyricist'] = df['song_id'].map(song_info['lyricist'])
        df['language'] = df['song_id'].map(song_info['language'])
        del song_info
        gc.collect()

        user_gender = members_df[['msno', 'gender']].copy()
        user_gender = user_gender.drop_duplicates('msno')
        df = df.merge(user_gender, on='msno', how='left')
        del user_gender
        gc.collect()

        df = df.reset_index(drop=True)
        df['_temp_order'] = df.index

        df = df.sort_values(['msno', '_temp_order']).reset_index(drop=True)

        for col in history_columns:
            if col not in df.columns:
                if 'mean' in col or 'popularity' in col:
                    df[col] = 0.0
                else:
                    df[col] = 0

        print("Calculating user statistics...")

        df['msno_total_interactions_before'] = df.groupby('msno').cumcount()

        df['_shifted_target'] = df.groupby('msno')['target'].shift()
        df['_cumsum_target'] = df.groupby('msno')['_shifted_target'].cumsum()

        df['msno_target_mean_before'] = np.where(
            df['msno_total_interactions_before'] > 0,
            df['_cumsum_target'] / df['msno_total_interactions_before'],
            0.0
        )

        valid_gender_mask = create_valid_mask(df['gender'], 'gender')
        df['_gender_for_grouping'] = df['gender'].where(valid_gender_mask, None)

        df = df.sort_values(['_gender_for_grouping', '_temp_order']).reset_index(drop=True)

        df['_gender_cumcount'] = df.groupby('_gender_for_grouping').cumcount()

        df['gender_total_interactions_before'] = np.where(
            valid_gender_mask,
            df['_gender_cumcount'],
            0
        )

        df['_shifted_target_gender'] = df.groupby('_gender_for_grouping')['target'].shift()
        df['_cumsum_target_gender'] = df.groupby('_gender_for_grouping')['_shifted_target_gender'].cumsum()

        df['gender_target_mean_before'] = np.where(
            valid_gender_mask & (df['_gender_cumcount'] > 0),
            df['_cumsum_target_gender'] / df['_gender_cumcount'],
            0.0
        )

        df = df.drop(columns=['_gender_for_grouping', '_gender_cumcount'])

        df = df.sort_values(['msno', '_temp_order']).reset_index(drop=True)

        df = df.drop(columns=['_shifted_target', '_cumsum_target', '_shifted_target_gender', '_cumsum_target_gender'])

        entities = [
            ('artist_name', 'artist_popularity_before'),
            ('composer', 'composer_popularity_before'),
            ('lyricist', 'lyricist_popularity_before'),
            ('language', 'language_popularity_before'),
            ('source_system_tab', 'source_system_tab_popularity_before'),
            ('source_screen_name', 'source_screen_name_popularity_before'),
            ('source_type', 'source_type_popularity_before')
        ]

        for entity_col, popularity_col in entities:
            print(f"Calculating statistics for {entity_col}...")

            valid_mask = create_valid_mask(df[entity_col], entity_col)

            df['_temp_entity'] = df[entity_col].where(valid_mask, None)

            df = df.sort_values(['_temp_entity', '_temp_order']).reset_index(drop=True)

            df['_entity_cumcount'] = df.groupby('_temp_entity').cumcount()

            df['_shifted_target'] = df.groupby('_temp_entity')['target'].shift()
            df['_cumsum_target'] = df.groupby('_temp_entity')['_shifted_target'].cumsum()

            df['_temp_popularity'] = np.where(
                df['_entity_cumcount'] > 0,
                df['_cumsum_target'] / df['_entity_cumcount'],
                0.0
            )

            df.loc[valid_mask, popularity_col] = df.loc[valid_mask, '_temp_popularity']

            df = df.drop(columns=['_shifted_target', '_cumsum_target', '_temp_entity', '_entity_cumcount', '_temp_popularity'])

            df = df.sort_values(['msno', '_temp_order']).reset_index(drop=True)

            gc.collect()

        pairs = [
            ('artist_name', 'msno_artist_interactions_before', 'msno_artist_target_mean_before'),
            ('composer', 'msno_composer_interactions_before', 'msno_composer_target_mean_before'),
            ('lyricist', 'msno_lyricist_interactions_before', 'msno_lyricist_target_mean_before'),
            ('language', 'msno_language_interactions_before', 'msno_language_target_mean_before'),
            ('source_system_tab', 'msno_source_system_tab_interactions_before', 'msno_source_system_tab_target_mean_before'),
            ('source_screen_name', 'msno_source_screen_name_interactions_before', 'msno_source_screen_name_target_mean_before'),
            ('source_type', 'msno_source_type_interactions_before', 'msno_source_type_target_mean_before'),

            ('artist_name', 'gender_artist_interactions_before', 'gender_artist_target_mean_before'),
            ('composer', 'gender_composer_interactions_before', 'gender_composer_target_mean_before'),
            ('lyricist', 'gender_lyricist_interactions_before', 'gender_lyricist_target_mean_before')
        ]

        df = df.sort_values(['msno', '_temp_order']).reset_index(drop=True)

        for entity_col, count_col, mean_col in pairs:
            print(f"Calculating statistics for users and {entity_col}...")

            if count_col.startswith('msno_'):

                entity_mask = create_valid_mask(df[entity_col], entity_col)

                df['_temp_entity'] = df[entity_col].where(entity_mask, None)

                df = df.sort_values(['msno', '_temp_entity', '_temp_order']).reset_index(drop=True)

                df['_pair_cumcount'] = df.groupby(['msno', '_temp_entity']).cumcount()

                df[count_col] = np.where(
                    entity_mask,
                    df['_pair_cumcount'],
                    0
                )

                df['_shifted_target'] = df.groupby(['msno', '_temp_entity'])['target'].shift()
                df['_cumsum_target'] = df.groupby(['msno', '_temp_entity'])['_shifted_target'].cumsum()

                df[mean_col] = np.where(
                    entity_mask & (df[count_col] > 0),
                    df['_cumsum_target'] / df[count_col],
                    0.0
                )

                df = df.drop(columns=['_temp_entity', '_pair_cumcount', '_shifted_target', '_cumsum_target'])

            else:

                gender_mask = create_valid_mask(df['gender'], 'gender')
                entity_mask = create_valid_mask(df[entity_col], entity_col)
                valid_mask = gender_mask & entity_mask

                df['_temp_gender'] = df['gender'].where(gender_mask, None)
                df['_temp_entity'] = df[entity_col].where(entity_mask, None)

                df = df.sort_values(['_temp_gender', '_temp_entity', '_temp_order']).reset_index(drop=True)

                df['_pair_cumcount'] = df.groupby(['_temp_gender', '_temp_entity']).cumcount()

                df[count_col] = np.where(
                    valid_mask,
                    df['_pair_cumcount'],
                    0
                )

                df['_shifted_target'] = df.groupby(['_temp_gender', '_temp_entity'])['target'].shift()
                df['_cumsum_target'] = df.groupby(['_temp_gender', '_temp_entity'])['_shifted_target'].cumsum()

                df[mean_col] = np.where(
                    valid_mask & (df[count_col] > 0),
                    df['_cumsum_target'] / df[count_col],
                    0.0
                )

                df = df.drop(columns=['_temp_gender', '_temp_entity', '_pair_cumcount', '_shifted_target', '_cumsum_target'])

            df = df.sort_values(['msno', '_temp_order']).reset_index(drop=True)

            gc.collect()

        gender_entities = [
            ('artist_name', 'gender_artist_popularity_before'),
            ('composer', 'gender_composer_popularity_before'),
            ('lyricist', 'gender_lyricist_popularity_before')
        ]

        for entity_col, popularity_col in gender_entities:
            print(f"Calculating statistics for {entity_col} for each gender...")

            gender_mask = create_valid_mask(df['gender'], 'gender')
            entity_mask = create_valid_mask(df[entity_col], entity_col)
            valid_mask = gender_mask & entity_mask

            df['_temp_gender'] = df['gender'].where(gender_mask, None)
            df['_temp_entity'] = df[entity_col].where(entity_mask, None)

            df = df.sort_values(['_temp_gender', '_temp_entity', '_temp_order']).reset_index(drop=True)

            df['_pair_cumcount'] = df.groupby(['_temp_gender', '_temp_entity']).cumcount()

            df['_shifted_target'] = df.groupby(['_temp_gender', '_temp_entity'])['target'].shift()
            df['_cumsum_target'] = df.groupby(['_temp_gender', '_temp_entity'])['_shifted_target'].cumsum()

            df[popularity_col] = np.where(
                valid_mask & (df['_pair_cumcount'] > 0),
                df['_cumsum_target'] / df['_pair_cumcount'],
                0.0
            )

            df = df.drop(columns=['_temp_gender', '_temp_entity', '_pair_cumcount', '_shifted_target', '_cumsum_target'])

            df = df.sort_values(['msno', '_temp_order']).reset_index(drop=True)

            gc.collect()

        print("Filling remaining NaN values in historical features...")
        
        artist_mask = create_valid_mask(df['artist_name'], 'artist_name')
        composer_mask = create_valid_mask(df['composer'], 'composer')
        lyricist_mask = create_valid_mask(df['lyricist'], 'lyricist')
        language_mask = create_valid_mask(df['language'], 'language')
        gender_mask = create_valid_mask(df['gender'], 'gender')
        
        df.loc[~artist_mask, 'msno_artist_interactions_before'] = 0
        df.loc[~artist_mask, 'msno_artist_target_mean_before'] = 0.0
        df.loc[~artist_mask, 'artist_popularity_before'] = 0.0
        df.loc[~artist_mask, 'gender_artist_interactions_before'] = 0
        df.loc[~artist_mask, 'gender_artist_target_mean_before'] = 0.0
        df.loc[~artist_mask, 'gender_artist_popularity_before'] = 0.0
        
        df.loc[~composer_mask, 'msno_composer_interactions_before'] = 0
        df.loc[~composer_mask, 'msno_composer_target_mean_before'] = 0.0
        df.loc[~composer_mask, 'composer_popularity_before'] = 0.0
        df.loc[~composer_mask, 'gender_composer_interactions_before'] = 0
        df.loc[~composer_mask, 'gender_composer_target_mean_before'] = 0.0
        df.loc[~composer_mask, 'gender_composer_popularity_before'] = 0.0
        
        df.loc[~lyricist_mask, 'msno_lyricist_interactions_before'] = 0
        df.loc[~lyricist_mask, 'msno_lyricist_target_mean_before'] = 0.0
        df.loc[~lyricist_mask, 'lyricist_popularity_before'] = 0.0
        df.loc[~lyricist_mask, 'gender_lyricist_interactions_before'] = 0
        df.loc[~lyricist_mask, 'gender_lyricist_target_mean_before'] = 0.0
        df.loc[~lyricist_mask, 'gender_lyricist_popularity_before'] = 0.0
        
        df.loc[~language_mask, 'msno_language_interactions_before'] = 0
        df.loc[~language_mask, 'msno_language_target_mean_before'] = 0.0
        df.loc[~language_mask, 'language_popularity_before'] = 0.0
        
        df.loc[~gender_mask, 'gender_total_interactions_before'] = 0
        df.loc[~gender_mask, 'gender_target_mean_before'] = 0.0

        del artist_mask, gender_mask, language_mask, lyricist_mask, composer_mask
        gc.collect()

        df = df.sort_values('_temp_order').reset_index(drop=True)
        df = df.drop(columns=['_temp_order'])

        print(f"Created {len(history_columns)} historical features on {len(df)} records")
        return df

    else:
        print("TEST mode: SQL-style - minimal memory...")

        print("\tAdding artist_name, composer and lyricist...")
        song_info = songs_df[['song_id', 'artist_name', 'composer', 'lyricist', 'language']].copy()  
        song_info = song_info.drop_duplicates('song_id')

        df = df.merge(song_info, on='song_id', how='left')
        del song_info
        gc.collect()

        df['artist_name'] = df['artist_name'].fillna('unknown_artist')
        df['composer'] = df['composer'].fillna('unknown_composer')
        df['lyricist'] = df['lyricist'].fillna('unknown_lyricist')  
        df['language'] = df['language'].fillna('Na')

        print("\tAdding user gender...")
        user_gender = members_df[['msno', 'gender']].copy()
        user_gender = user_gender.drop_duplicates('msno')
        df = df.merge(user_gender, on='msno', how='left')
        df['gender'] = df['gender'].fillna('Na')

        del user_gender
        gc.collect()

        for col in history_columns:
            if col not in df.columns:
                if 'mean' in col or 'popularity' in col:
                    df[col] = 0.0
                else:
                    df[col] = 0

        print("\tmsno features...")

        msno_cols = ['msno', 'msno_total_interactions_before', 'msno_target_mean_before']
        msno_stats = train_df[msno_cols].drop_duplicates('msno', keep='last')

        df = df.merge(msno_stats, on='msno', how='left', suffixes=('', '_msno'))

        df['msno_total_interactions_before'] = df['msno_total_interactions_before'].fillna(0).astype('int32')
        df['msno_target_mean_before'] = df['msno_target_mean_before'].fillna(0.0).astype('float32')

        cols_to_drop = [col for col in df.columns if col.endswith('_msno')]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        del msno_stats, cols_to_drop
        gc.collect()

        print("\tgender features...")

        valid_gender_mask = create_valid_mask(train_df['gender'], 'gender')
        gender_stats_filtered = train_df[valid_gender_mask].copy()

        gender_cols = ['gender', 'gender_total_interactions_before', 'gender_target_mean_before']
        gender_stats = gender_stats_filtered[gender_cols].drop_duplicates('gender', keep='last')

        df = df.merge(gender_stats, on='gender', how='left', suffixes=('', '_gender'))

        df['gender_total_interactions_before'] = df['gender_total_interactions_before'].fillna(0).astype('int32')
        df['gender_target_mean_before'] = df['gender_target_mean_before'].fillna(0.0).astype('float32')

        cols_to_drop = [col for col in df.columns if col.endswith('_gender')]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        del gender_stats, gender_stats_filtered
        gc.collect()

        entities = [
            ('artist_name', 'artist_popularity_before'),
            ('composer', 'composer_popularity_before'),
            ('lyricist', 'lyricist_popularity_before'),
            ('language', 'language_popularity_before'),
            ('source_system_tab', 'source_system_tab_popularity_before'),
            ('source_screen_name', 'source_screen_name_popularity_before'),
            ('source_type', 'source_type_popularity_before')
        ]

        for entity_col, popularity_col in entities:
            print(f"\t{entity_col} features...")

            valid_mask = create_valid_mask(train_df[entity_col], entity_col)
            stats_filtered = train_df[valid_mask].copy()

            stats = stats_filtered[[entity_col, popularity_col]] \
                .drop_duplicates(entity_col, keep='last')

            df = df.merge(stats, on=entity_col, how='left', suffixes=('', f'_{entity_col}'))

            df[popularity_col] = df[popularity_col].fillna(0.0).astype('float32')

            cols_to_drop = [col for col in df.columns if col.endswith(f'_{entity_col}')]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            del stats, stats_filtered, cols_to_drop
            gc.collect()

        gender_entities = [
            ('artist_name', 'gender_artist_popularity_before'),
            ('composer', 'gender_composer_popularity_before'),
            ('lyricist', 'gender_lyricist_popularity_before')
        ]

        for entity_col, popularity_col in gender_entities:
            print(f"\tgender+{entity_col} popularity...")

            gender_mask = create_valid_mask(train_df['gender'], 'gender')
            entity_mask = create_valid_mask(train_df[entity_col], entity_col)
            combined_mask = gender_mask & entity_mask

            stats_filtered = train_df[combined_mask].copy()

            stats = stats_filtered[['gender', entity_col, popularity_col]] \
                .drop_duplicates(['gender', entity_col], keep='last')

            df = df.merge(stats, on=['gender', entity_col], how='left', suffixes=('', f'_gend_{entity_col}'))

            df[popularity_col] = df[popularity_col].fillna(0.0).astype('float32')

            cols_to_drop = [col for col in df.columns if col.endswith(f'_gend_{entity_col}')]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            del stats, stats_filtered, cols_to_drop
            gc.collect()

        pairs = [
            ('artist_name', 'msno_artist_interactions_before', 'msno_artist_target_mean_before'),
            ('composer', 'msno_composer_interactions_before', 'msno_composer_target_mean_before'),
            ('lyricist', 'msno_lyricist_interactions_before', 'msno_lyricist_target_mean_before'),
            ('language', 'msno_language_interactions_before', 'msno_language_target_mean_before'),
            ('source_system_tab', 'msno_source_system_tab_interactions_before', 'msno_source_system_tab_target_mean_before'),
            ('source_screen_name', 'msno_source_screen_name_interactions_before', 'msno_source_screen_name_target_mean_before'),
            ('source_type', 'msno_source_type_interactions_before', 'msno_source_type_target_mean_before'),

            ('artist_name', 'gender_artist_interactions_before', 'gender_artist_target_mean_before'),
            ('composer', 'gender_composer_interactions_before', 'gender_composer_target_mean_before'),
            ('lyricist', 'gender_lyricist_interactions_before', 'gender_lyricist_target_mean_before')
        ]

        for entity_col, count_col, mean_col in pairs:
            print(f"\t{entity_col} features...")

            if count_col.startswith('msno_'):

                entity_mask = create_valid_mask(train_df[entity_col], entity_col)
                stats_filtered = train_df[entity_mask].copy()

                pair_cols = ['msno', entity_col, count_col, mean_col]
                pair_stats = stats_filtered[pair_cols] \
                    .drop_duplicates(['msno', entity_col], keep='last')

                merge_on = ['msno', entity_col]
            else:

                gender_mask = create_valid_mask(train_df['gender'], 'gender')
                entity_mask = create_valid_mask(train_df[entity_col], entity_col)
                combined_mask = gender_mask & entity_mask
                stats_filtered = train_df[combined_mask].copy()

                pair_cols = ['gender', entity_col, count_col, mean_col]
                pair_stats = stats_filtered[pair_cols] \
                    .drop_duplicates(['gender', entity_col], keep='last')

                merge_on = ['gender', entity_col]

            if len(pair_stats) > 1000000:
                print(f"\t\tSplitting {len(pair_stats)} records into parts...")

                df[count_col] = 0
                df[mean_col] = 0.0

                chunk_size = 500000
                for i in range(0, len(pair_stats), chunk_size):
                    print(f"    Part {i//chunk_size + 1}/{(len(pair_stats)-1)//chunk_size + 1}...")

                    part = pair_stats.iloc[i:i+chunk_size].copy()

                    df = df.merge(
                        part, 
                        on=merge_on, 
                        how='left',
                        suffixes=('', f'_part')
                    )

                    mask = df[f'{count_col}_part'].notna()
                    df.loc[mask, count_col] = df.loc[mask, f'{count_col}_part']
                    df.loc[mask, mean_col] = df.loc[mask, f'{mean_col}_part']

                    df = df.drop(columns=[f'{count_col}_part', f'{mean_col}_part'])

                    del part
                    gc.collect()
            else:
                df = df.merge(
                    pair_stats, 
                    on=merge_on, 
                    how='left',
                    suffixes=('', f'_{entity_col}')
                )

                temp_count_col = f'{count_col}_{entity_col}'
                temp_mean_col = f'{mean_col}_{entity_col}'

                if temp_count_col in df.columns:
                    mask = df[temp_count_col].notna()
                    df.loc[mask, count_col] = df.loc[mask, temp_count_col]
                    df = df.drop(columns=[temp_count_col])

                if temp_mean_col in df.columns:
                    mask = df[temp_mean_col].notna()
                    df.loc[mask, mean_col] = df.loc[mask, temp_mean_col]
                    df = df.drop(columns=[temp_mean_col])

            df[count_col] = df[count_col].fillna(0).astype('int32')
            df[mean_col] = df[mean_col].fillna(0.0).astype('float32')

            del pair_stats
            if 'stats_filtered' in locals():
                del stats_filtered
            gc.collect()

        print(f"\tFeatures attached to {len(df)} test records")
        return df


def feature_adding_train_test(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    songs_df: pd.DataFrame, 
    members_df: pd.DataFrame, 
    feature_blocks: List[Callable[..., None]],
    train
):
    '''Creating and adding new train/test features'''
    for block in feature_blocks:
        df = block(df, train_df, songs_df, members_df, train)

    print('All new features has been added!')
    return df

def prepare_train_test(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    songs_df: pd.DataFrame, 
    members_df: pd.DataFrame, 
    feature_blocks: List[Callable[..., None]],
    train=1
):
    '''Preparing train dataset'''
    print('='*60)
    print('Start Prepare Train DF')
    print('='*60)
    cols = df.columns
    df = df.astype({col: MAIN_FEATURES[col] for col in MAIN_FEATURES if col in cols})
    df = interpolate_train_test_features(df)
    df = feature_adding_train_test(
        df, 
        train_df, 
        songs_df, 
        members_df, 
        feature_blocks, 
        train
    )
    print('='*60)
    print('Train DF has been prepared successfully!')
    print('='*60)
    return df


train_prepared = prepare_train_test(
    train.copy(deep=True), 
    train.copy(deep=True), 
    songs_prepared, 
    members_prepared,
    [history_stats_adding_song_enchanted]
)


test_prepared = prepare_train_test(
    test.copy(deep=True), 
    train_prepared,
    songs_prepared, 
    members_prepared,
    [history_stats_adding_song_enchanted],
    train=0
)


del train_prepared, test_prepared
gc.collect()


train_dataset = train_prepared\
    .merge(members_prepared, on='msno')\
    .merge(songs_prepared, on='song_id')\
    .drop(columns=[
        'msno', 
        'song_id',
        'artist_name_x',
        'composer_x',
        'lyricist_x',
        'language_x',
        'gender_x'
    ])

test_dataset = test_prepared\
    .merge(members_prepared, on='msno', how='left')\
    .merge(songs_prepared, on='song_id', how='left')\
    .drop(columns=[
        'msno', 
        'song_id',
        'artist_name_x',
        'composer_x',
        'lyricist_x',
        'language_x',
        'gender_x'
    ])

ID = test_dataset.pop('id')


train_dataset[['msno_lyricist_interactions_before',
 'msno_composer_interactions_before',
 'msno_language_interactions_before',
 'gender_total_interactions_before',
 'gender_artist_interactions_before',
 'gender_composer_interactions_before',
 'gender_lyricist_interactions_before']] = \
train_dataset[['msno_lyricist_interactions_before',
 'msno_composer_interactions_before',
 'msno_language_interactions_before',
 'gender_total_interactions_before',
 'gender_artist_interactions_before',
 'gender_composer_interactions_before',
 'gender_lyricist_interactions_before']].fillna(0)


del train_dataset, test_dataset, ID
gc.collect()


train_dataset.shape


set(test_dataset.columns) - set(train_dataset.columns), set(train_dataset.columns) - set(test_dataset.columns), 


train_dataset.isna().sum()[train_dataset.isna().sum() != 0].index.tolist()


test_dataset.isna().sum()[test_dataset.isna().sum() != 0].sum()


X, Y = train_dataset.drop(columns=['target']), train_dataset['target']
X = X[[col for col in X.columns if col not in exclude]]


categories = X.select_dtypes('category').columns
categories


del train_dataset
gc.collect()


del X, Y
gc.collect()


X_train, X_test = X.iloc[:int(len(X) * 0.8)], X.iloc[int(len(X) * 0.8):]
Y_train, Y_test = Y.iloc[:int(len(X) * 0.8)], Y.iloc[int(len(X) * 0.8):]


LGB_PARAMS = {
    'objective': 'binary', 
    'metric': ['binary_logloss', 'auc'],  
    'boosting_type': 'gbdt',
    'verbose': 1,  
    'seed': 43,
    
    'n_estimators': 300, 
    'learning_rate': 0.05,  
    
    'num_leaves': 31,  
    'max_depth': 5, 
    'min_data_in_leaf': 400,  
    'min_child_samples': 50, 
    'min_child_weight': 0.01, 
    
    'lambda_l1': 10,  
    'lambda_l2': 10, 
    'min_gain_to_split': 0.1,  
    
    'bagging_freq': 5,
    'bagging_fraction': 0.7,
    'feature_fraction': 0.5,  
    
    'cat_smooth': 50.0,  
    'cat_l2': 300,
    'max_cat_threshold': 15,  
    'min_data_per_group': 200, 
    
    'path_smooth': 1.0,
    
    'feature_pre_filter': True, 
    
    'scale_pos_weight': 1.0,
    'boost_from_average': True,
    
    'n_jobs': -1,
    'verbosity': 1,           
    'force_row_wise': True,
    'categorical_feature': categories.tolist(),
}
 
model = lgb.LGBMClassifier(**LGB_PARAMS)

model.fit(
    X,  
    Y,
    eval_set=[(X, Y)],
    eval_metric='binary_logloss',
    callbacks=[
        lgb.log_evaluation(20),
    ]
)


LGB_PARAMS = {
    'objective': 'binary', 
    'metric': ['binary_logloss', 'auc'],  
    'boosting_type': 'gbdt',
    'verbose': 1,  
    'seed': 43,
    
    'n_estimators': 260, 
    'learning_rate': 0.05,  
    
    'num_leaves': 63,  
    'max_depth': 6, 
    'min_data_in_leaf': 300,  
    'min_child_samples': 50, 
    'min_child_weight': 0.01, 
    
    'lambda_l1': 10,  
    'lambda_l2': 10, 
    'min_gain_to_split': 0.1,  
    
    'bagging_freq': 5,
    'bagging_fraction': 0.9,
    'feature_fraction': 0.7,  
    
    'max_cat_threshold': 15,  
    'min_data_per_group': 200, 
    
    'path_smooth': 1.0,
    
    'feature_pre_filter': True, 
    
    'scale_pos_weight': 1.0,
    'boost_from_average': True,
    
    'n_jobs': -1,
    'verbosity': 1,           
    'force_row_wise': True,
    'categorical_feature': categories.tolist(),
}
 
model = lgb.LGBMClassifier(**LGB_PARAMS)

model.fit(
    X,  
    Y,
    eval_set=[(X, Y)],
    eval_metric='binary_logloss',
    callbacks=[
        lgb.log_evaluation(20),
    ]
)


feature_importance = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
})

importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

print(importance_df.head(50))
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importance - LightGBM')
plt.tight_layout()
plt.show()


joblib.dump(model, 'LGBTRUES.66.joblib')


test_predictions = model.predict_proba(test_dataset[X.columns].astype({col: X[col].dtype for col in X.columns if test_dataset[col].isna().sum() == 0}))


submission_df = pd.DataFrame({
    'id': ID,
    'target': test_predictions[:, 1]
})

submission_df.to_csv('subsCFDLNAV5.csv', index=False)



def blend(
    dfs: List[pd.DataFrame], 
    weights: Optional[List[float]] = None
) -> pd.DataFrame:
    
    if weights is None:
        weights = [1/len(dfs)] * len(dfs)

    weighted_dfs = []
    for df, weight in zip(dfs, weights):
        df_weighted = df.copy()
        df_weighted['target'] = df_weighted['target'] * weight
        weighted_dfs.append(df_weighted)
    
    df_both = pd.concat(weighted_dfs, axis=0)
    df_both = df_both.groupby('id', as_index=False)['target'].sum()
    
    return df_both


subs = [
    pd.read_csv('/kaggle/working/subsCFDLNAV1.csv'),
    pd.read_csv('/kaggle/input/subs-cfdls/subsCFDLV1.csv'),
    pd.read_csv('/kaggle/input/subs-cfdls/subsCFDLV2.csv')
]

submission_df = blend(
    dfs=subs,
    weights=[0.3, 0.3, 0.4]
)

submission_df.to_csv('BLENDDLV1.csv', index=False)

