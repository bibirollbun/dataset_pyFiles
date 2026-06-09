import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from datetime import timedelta
import gc

# ===== 1. LOAD DATA =====
print("Loading data...")
try:
    train = pd.read_csv('/kaggle/input/sweettv-movie-recommender/movies_dataset_10 months.csv')
    submission = pd.read_csv('/kaggle/input/sweettv-movie-recommender/submission.csv')
except:
    train = pd.read_csv('movies_dataset_10 months.csv')
    submission = pd.read_csv('submission.csv')

train['ts'] = pd.to_datetime(train['ts'])
max_date = train['ts'].max()

# ===== 2. IDENTIFY SERIES =====
print("Identifying Series...")
movie_max_ep = train.groupby('movie_id')['episode_id'].max()
series_ids = set(movie_max_ep[movie_max_ep > 0].index)

# ===== 3. POPULARITY (Base V24: 14 Days) =====
print("Calculating Item Popularity (Window: 14 days)...")
recent_activity = train[train['ts'] > (max_date - timedelta(days=14))]
item_counts = recent_activity['movie_id'].value_counts()

pop_dict = item_counts.to_dict()
max_pop = item_counts.max()

# ===== 4. WEIGHTING (Base V24: Sharp Decay) =====
print("Calculating Weights (Sharp Decay: d+0.5)...")
recent_train = train[train['ts'] > (max_date - timedelta(days=100))].copy()
recent_train['days_diff'] = (max_date - recent_train['ts']).dt.days

def calculate_weight_v31(row):
    return 1.0 / (row['days_diff'] + 0.5)

recent_train['weight'] = recent_train.apply(calculate_weight_v31, axis=1)

# ===== 5. BUILD MATRIX =====
print("Building Matrix...")
user_ids = train['user_id'].unique()
movie_ids = train['movie_id'].unique()

user_map = {u: i for i, u in enumerate(user_ids)}
movie_map = {m: i for i, m in enumerate(movie_ids)}
inv_movie_map = {i: m for m, i in movie_map.items()}

row = recent_train['user_id'].map(user_map)
col = recent_train['movie_id'].map(movie_map)
data = recent_train['weight']

M = sparse.csr_matrix((data, (row, col)), shape=(len(user_ids), len(movie_ids)))
M_normalized = normalize(M, norm='l2', axis=1)

print("Calculating Similarity...")
S = M_normalized.T.dot(M_normalized)

# ===== 6. POPULARITY INJECTION (V31: 0.25 Boost) =====
print("Injecting Popularity Boost (0.25)...")
pop_vector = np.ones(len(movie_ids))
for m_id, idx in movie_map.items():
    raw_pop = pop_dict.get(m_id, 0)
    norm_pop = raw_pop / max_pop if max_pop > 0 else 0
    # [V31 Change] Tăng nhẹ từ 0.2 lên 0.25
    # Logic: 0.5 là quá nhiều, 0.2 là an toàn. 0.25 có thể là điểm tối ưu để kéo thêm trend.
    pop_vector[idx] = 1.0 + (0.25 * norm_pop)

P_mat = sparse.diags(pop_vector)
S = S.dot(P_mat)

# Filter Noise
S.setdiag(0)
S.data[S.data < 0.01] = 0
S.eliminate_zeros()

# ===== 7. PREPARE HISTORY =====
print("Preparing History & Global Trends...")
trend_df = train[train['ts'] > (max_date - timedelta(days=14))]
global_top = trend_df['movie_id'].value_counts().head(50).index.tolist()

history_df = train[train['ts'] > (max_date - timedelta(days=90))].sort_values('ts', ascending=False)
user_history_detailed = history_df.groupby('user_id')[['movie_id', 'ts']].apply(lambda x: list(zip(x['movie_id'], x['ts']))).to_dict()

# ===== 8. PREDICTION LOGIC (V31: 24H RULE) =====
def generate_recs_v31(target_users, M, S, user_map, inv_movie_map, user_history_detailed, global_top, series_ids, max_date):
    results = []
    total = len(target_users)
    print(f"Generating for {total} users (V31: 24h Active Rule + 0.25 Pop Boost)...")
    
    SERIES_DROPOUT_DAYS = 60
    
    for idx, user_id in enumerate(target_users):
        recs = []
        seen = set()
        
        # LAYER 1: SMART RESUME
        if user_id in user_history_detailed:
            history_items = user_history_detailed[user_id]
            
            # [V31 LOGIC] Siết chặt điều kiện Active
            if history_items:
                last_seen_ts = history_items[0][1]
                days_since_active = (max_date - last_seen_ts).days
                
                # [V31 Change] Chỉ những ai xem "Hôm qua hoặc Hôm nay" (<=1) mới được Limit 4
                if days_since_active <= 1:
                    dynamic_limit = 4
                else:
                    # Còn lại (xem từ 2 ngày trước) trả về Limit 3 để an toàn cho Private Score
                    dynamic_limit = 3
            else:
                dynamic_limit = 3
            
            for mid, ts in history_items:
                if len(recs) >= dynamic_limit: break
                
                if mid in seen: continue
                days_since = (max_date - ts).days
                
                if mid in series_ids:
                    if days_since < SERIES_DROPOUT_DAYS:
                        recs.append(mid)
                        seen.add(mid)
                else:
                    if days_since < 14:
                        recs.append(mid)
                        seen.add(mid)

        # LAYER 2: CF
        if len(recs) < 5 and user_id in user_map:
            u_idx = user_map[user_id]
            scores = M[u_idx, :].dot(S)
            if scores.nnz > 0:
                top_indices = scores.indices[np.argsort(scores.data)[::-1]]
                for item_idx in top_indices:
                    mid = inv_movie_map[item_idx]
                    if mid not in seen:
                        recs.append(mid)
                        seen.add(mid)
                    if len(recs) >= 5: break
        
        # LAYER 3: GLOBAL FILLER
        for mid in global_top:
            if len(recs) >= 5: break
            if mid not in seen:
                recs.append(mid)
                seen.add(mid)
                
        results.append({'user_id': user_id, 'movie_id': ' '.join(map(str, recs[:5]))})
        if idx % 10000 == 0: print(f"Processed {idx}/{total}")
            
    return pd.DataFrame(results)

# Run V31
target_users = submission['user_id'].unique()
final_sub = generate_recs_v31(target_users, M, S, user_map, inv_movie_map, user_history_detailed, global_top, series_ids, max_date)
final_sub.to_csv('submission_v31_24h_rule.csv', index=False)
print("✅ Done! V31: Tightened Active definition (<=1 day) & Increased Pop Boost (0.25).")

