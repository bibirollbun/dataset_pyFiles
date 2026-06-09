import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pip install polars


pip install numba


import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import polars as pl
import gc
from numba import jit
from multiprocessing import Pool
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def read_jsonl_chunk(file, n=100000):
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i >= n: 
                break
            yield json.loads(line)

def count_lines_fast(filename):
    with open(filename, 'r') as f:
        return sum(1 for _ in f)

@jit(nopython=True)
def fast_feature_extraction(aid, session_aids_array, session_counts_array):
    return [int(aid in session_aids_array), session_counts_array[aid]]

def get_popular_aids_fast(train_file, top_k=169):
    aid_counts = Counter()
    
    print("Подсчет популярных aid...")
    for row in read_jsonl_chunk(train_file, n=1_000_000):
        session = row['events']
        for event in session:
            aid_counts[event['aid']] += 1
    
    popular_aids = [aid for aid, _ in aid_counts.most_common(top_k)]
    print(f"Найдено {len(popular_aids)} популярных aid")
    print(f"Размер aid_counts: {len(aid_counts)}")
    return popular_aids, aid_counts

def get_candidates_fast(session, popular_aids, top_k=100):
    session_aids = set()
    session_counts = Counter()
    
    for event in session:
        aid = event['aid']
        session_aids.add(aid)
        session_counts[aid] += 1
    
    candidates = list(session_aids) + [aid for aid in popular_aids if aid not in session_aids]
    return candidates[:top_k], session_aids, session_counts

def extract_features_fast(aid, session_aids, session_counts, global_scores=None):
    if global_scores is None:
        global_scores = {}
    features = [
        int(aid in session_aids),
        session_counts.get(aid, 0),
        global_scores.get(aid, 0),
    ]
    return features

def train_model_fast(X, y):
    model = HistGradientBoostingClassifier(
        max_iter=16,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=0
    )
    model.fit(X, y)
    print(f"Размер X: {X.shape}")
    print(f"Размер y: {y.shape}")
    return model

def prepare_training_data(train_file, popular_aids, max_samples=1_000_000):
    X_train, y_train = [], []
    sample_count = 0
    
    print("Подготовка данных для обучения...")
    for row in read_jsonl_chunk(train_file, n=max_samples):
        if sample_count >= max_samples:
            break
            
        session = row['events']
        candidates, session_aids, session_counts = get_candidates_fast(
            session, popular_aids, top_k=69
        )
        
        for event_type in ['clicks', 'carts', 'orders']:
            gt_aid = next((e['aid'] for e in session if e['type'] == event_type), None)
            if gt_aid and gt_aid in candidates:
                features = extract_features_fast(gt_aid, session_aids, session_counts)
                X_train.append(features)
                y_train.append(1)
                
                neg_candidates = [aid for aid in candidates if aid != gt_aid][:3]
                for neg_aid in neg_candidates:
                    features = extract_features_fast(neg_aid, session_aids, session_counts)
                    X_train.append(features)
                    y_train.append(0)
                    sample_count += 1
                    
        sample_count += 1
        
        if sample_count >= max_samples:
            break
    
    print(f"Подготовлено {len(X_train)} образцов")
    print(f"Размер X_train: {len(X_train)}")
    print(f"Размер y_train: {len(y_train)}")
    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32)

def train_pipeline():
    popular_aids, aid_counts = get_popular_aids_fast('/kaggle/input/otto-recommender-system/train.jsonl', top_k=1000)
    
    X_train, y_train = prepare_training_data('/kaggle/input/otto-recommender-system/train.jsonl', popular_aids, max_samples=69_000)
    
    print("Обучение модели...")
    model = train_model_fast(X_train, y_train)
    
    del X_train, y_train
    gc.collect()
    
    return model, popular_aids, aid_counts

def inference_and_save(model, popular_aids, test_file='/kaggle/input/otto-recommender-system/test.jsonl', submission_file='submission.csv'):
    test_len = count_lines_fast(test_file)
    total_rows_written = 0
    
    print(f"Инференс на {test_len} сессиях...")
    
    with open(submission_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['session_type', 'labels'])
        writer.writeheader()
        
        submission_batch = []
        batch_size = 100_000
        
        for row in tqdm(read_jsonl_chunk(test_file, n=test_len), total=test_len, desc="Processing test"):
            session = row['events']
            session_id = row['session']
            
            candidates, session_aids, session_counts = get_candidates_fast(
                session, popular_aids, top_k=100
            )
            
            for event_type in ['clicks', 'carts', 'orders']:
                features_list = [
                    extract_features_fast(aid, session_aids, session_counts)
                    for aid in candidates
                ]
                
                if features_list:
                    scores = model.predict_proba(features_list)[:, 1]
                    top_20 = [candidates[i] for i in np.argsort(scores)[-20:][::-1]]
                else:
                    top_20 = popular_aids[:20]
                
                submission_batch.append({
                    'session_type': f"{session_id}_{event_type}",
                    'labels': ' '.join(map(str, top_20))
                })
                
                if len(submission_batch) >= batch_size:
                    writer.writerows(submission_batch)
                    total_rows_written += len(submission_batch)
                    submission_batch = []
        
        if submission_batch:
            writer.writerows(submission_batch)
            total_rows_written += len(submission_batch)
    
    print(f"Submission saved: {total_rows_written} rows")
    
    expected_rows = 5015409
    if total_rows_written != expected_rows:
        print(f"WARNING: Expected {expected_rows} rows, got {total_rows_written}")
    else:
        print("✓ Correct number of rows in submission")
    
    return total_rows_written

print("Starting optimized pipeline...")

model, popular_aids, aid_counts = train_pipeline()

total_rows = inference_and_save(model, popular_aids)

print("Pipeline completed successfully!")

