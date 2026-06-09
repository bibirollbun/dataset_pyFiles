import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
import os
import shutil
from datetime import datetime
import re

warnings.filterwarnings('ignore', category=FutureWarning)


def reduce_mem_usage(df, verbose=True):
    """Fungsi untuk mengurangi penggunaan memori DataFrame."""
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df

def preprocess_data(df, kmeans_model, neighborhood_stats, global_mean_price, global_std_price, tfidf_model):
    if 'last_review' in df.columns:
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
        df['days_since_last_review'] = (pd.to_datetime('today') - df['last_review']).dt.days
        df['days_since_last_review'].fillna(9999, inplace=True)
    if 'name' in df.columns and 'description' in df.columns:
        df['name'].fillna('', inplace=True); df['description'].fillna('', inplace=True)
        full_text = df['name'] + ' ' + df['description']
        tfidf_result = tfidf_model.transform(full_text)
        tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=tfidf_model.get_feature_names_out(), index=df.index).add_prefix('text_')
        df = df.join(tfidf_df)
    if 'neighbourhood_cleansed' in df.columns:
        df = pd.merge(df, neighborhood_stats, on='neighbourhood_cleansed', how='left')
        df['mean_price_in_hood'].fillna(global_mean_price, inplace=True)
        df['std_price_in_hood'].fillna(global_std_price, inplace=True)
        # --- PERUBAHAN: Isi nilai kosong untuk fitur rating baru ---
        df['mean_rating_in_hood'].fillna(neighborhood_stats['mean_rating_in_hood'].mean(), inplace=True)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat_lon = df[['latitude', 'longitude']].fillna(df[['latitude', 'longitude']].median())
        df['location_cluster'] = kmeans_model.predict(lat_lon)
    if 'amenities' in df.columns:
        df['amenities_count'] = df['amenities'].apply(lambda x: len(x.replace('[', '').replace(']', '').replace('"', '').split(',')) if isinstance(x, str) and x != '[]' else 0)
    if 'host_verifications' in df.columns:
        df['verifications_count'] = df['host_verifications'].apply(lambda x: len(x.replace('[', '').replace(']', '').replace('"', '').split(',')) if isinstance(x, str) and x != '[]' else 0)
    if 'host_since' in df.columns:
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
        df['host_duration_days'] = (pd.to_datetime('today') - df['host_since']).dt.days
    columns_to_drop = ['name', 'description', 'neighborhood_overview', 'host_name', 'host_location', 'host_about', 'host_neighbourhood', 'neighbourhood', 'neighbourhood_cleansed', 'property_type', 'bathrooms_text', 'has_availability', 'amenities', 'host_verifications', 'host_since', 'first_review', 'last_review', 'id']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    # Konversi dan Encoding
    for col in ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']:
        if col in df.columns: df[col] = df[col].apply(lambda x: 1 if x == 't' else 0)
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df.columns: df[col] = df[col].str.replace('%', '', regex=False).astype(float) / 100.0
    explicit_categorical_cols = ['host_response_time', 'room_type', 'city']
    df = pd.get_dummies(df, columns=[col for col in explicit_categorical_cols if col in df.columns], drop_first=True, dummy_na=True, dtype=int)
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def process_in_chunks(input_csv_path, output_dir, kmeans_model, neighborhood_stats, global_mean_price, global_std_price, tfidf_model):
    if not os.path.exists(input_csv_path): print(f"Error: File '{input_csv_path}' tidak ditemukan."); return None
    print(f"Memulai pra-pemrosesan file '{input_csv_path}' dalam chunks...")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    chunk_iterator = pd.read_csv(input_csv_path, chunksize=50000, low_memory=False)
    for i, chunk in enumerate(chunk_iterator):
        print(f"  - Memproses dan menyimpan chunk ke-{i+1}...")
        processed_chunk = preprocess_data(chunk.copy(), kmeans_model, neighborhood_stats, global_mean_price, global_std_price, tfidf_model)
        if 'price' in processed_chunk.columns:
            processed_chunk = processed_chunk[processed_chunk['price'] > 0]
        processed_chunk.to_parquet(f'{output_dir}/chunk_{i}.parquet', engine='pyarrow')
    print(f"Pra-pemrosesan selesai. Data bersih disimpan di folder '{output_dir}'.")
    df_processed = pd.read_parquet(output_dir)
    return df_processed



try:
    input_path = '/kaggle/input/sparta-2024-data-science-competition/'
    train_file = input_path + 'train.csv'
    test_file = input_path + 'test.csv'

    df_stats = pd.read_csv(train_file, usecols=['neighbourhood_cleansed', 'price', 'review_scores_rating'])
    df_stats = df_stats[df_stats['price'] > 0]
    
    price_stats = df_stats.groupby('neighbourhood_cleansed')['price'].agg(['mean', 'std']).reset_index()
    price_stats.columns = ['neighbourhood_cleansed', 'mean_price_in_hood', 'std_price_in_hood']
    
    rating_stats = df_stats.groupby('neighbourhood_cleansed')['review_scores_rating'].agg('mean').reset_index()
    rating_stats.columns = ['neighbourhood_cleansed', 'mean_rating_in_hood']
    
    neighborhood_stats = pd.merge(price_stats, rating_stats, on='neighbourhood_cleansed', how='left')

    global_mean_price, global_std_price = df_stats['price'].mean(), df_stats['price'].std()
    
    df_sample = pd.read_csv(train_file, nrows=100000, usecols=['latitude', 'longitude'])
    kmeans = KMeans(n_clusters=15, random_state=42, n_init='auto'); kmeans.fit(df_sample[['latitude', 'longitude']].dropna())
    
    df_text = pd.read_csv(train_file, usecols=['name', 'description'])
    df_text['name'].fillna('', inplace=True); df_text['description'].fillna('', inplace=True)
    df_text['full_text'] = df_text['name'] + ' ' + df_text['description']
    tfidf = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2)); tfidf.fit(df_text['full_text'])
    print("Persiapan fitur tambahan selesai.")
except FileNotFoundError:
    print(f"Error: File di path '{train_file}' tidak ditemukan."); exit()

df_train_processed = process_in_chunks(train_file, 'train_processed_chunks', kmeans, neighborhood_stats, global_mean_price, global_std_price, tfidf)

if df_train_processed is not None:
    df_train_processed = reduce_mem_usage(df_train_processed)
    y_train = np.log1p(df_train_processed.pop('price'))
    X_train = df_train_processed
    X_train.columns = [re.sub('[^A-Za-z0-9_]+', '', col) for col in X_train.columns]
    train_columns = X_train.columns.tolist()
    print(f"Data siap: {X_train.shape[0]} baris dan {X_train.shape[1]} fitur untuk training.")

    best_params = {
        'n_estimators': 2200, 
        'learning_rate': 0.092, 
        'num_leaves': 100, 
        'max_depth': 13, 
        'min_child_samples': 17, 
        'subsample': 0.689, 
        'colsample_bytree': 0.689
    }
    model = lgb.LGBMRegressor(**best_params, objective='regression_l1', metric='rmse', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)


    # --- 5. PREDIKSI PADA DATA TEST ---
    df_test_processed = process_in_chunks(test_file, 'test_processed_chunks', kmeans, neighborhood_stats, global_mean_price, global_std_price, tfidf)
    
    if df_test_processed is not None:
        df_test_processed = reduce_mem_usage(df_test_processed)
        X_test = df_test_processed.copy()
        X_test.columns = [re.sub('[^A-Za-z0-9_]+', '', col) for col in X_test.columns]
        print("Menyelaraskan kolom data test dengan data train...")
        for col in train_columns:
            if col not in X_test.columns: X_test[col] = 0
        X_test = X_test[train_columns]
        
        log_predictions = model.predict(X_test)
        final_predictions = np.expm1(log_predictions)
        
        df_test_original = pd.read_csv(test_file, usecols=['id'])
        submission = pd.DataFrame({'id': df_test_original['id'], 'price': final_predictions})
        # Di Kaggle, file disimpan di /kaggle/working/
        submission_filename = 'submission_single_best_lgbm.csv'
        submission.to_csv(submission_filename, index=False)
        
        print(f"\n File '{submission_filename}' telah berhasil dibuat!")
        print(submission.head())


