import time
import orjson
from collections import defaultdict, Counter
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

DATA_DIR = '/kaggle/input/otto-recommender-system'
TRAIN_PATH = f'{DATA_DIR}/train.jsonl'
TEST_PATH = f'{DATA_DIR}/test.jsonl'
MAX_TRAIN_SESSIONS = 2_000_000
TOP_K_POPULAR = 100
EVENT_WEIGHTS = {'clicks': 1, 'carts': 3, 'orders': 5}


def compute_item_popularity(file_path, max_sessions, weights):
    item_popularity = Counter()
    start_time = time.time()

    with open(file_path, 'r') as file:
        for session_idx, line in enumerate(file, 1):
            session_data = orjson.loads(line)
            for event in session_data['events']:
                item_popularity[event['aid']] += weights[event['type']]
            if session_idx >= max_sessions:
                break

    print(f"[Popularity] Processed {session_idx} sessions in {time.time() - start_time:.2f}s")
    return item_popularity


item_popularity = compute_item_popularity(TRAIN_PATH, MAX_TRAIN_SESSIONS, EVENT_WEIGHTS)
top_popular_items = [item for item, _ in item_popularity.most_common(TOP_K_POPULAR)]


def generate_training_data(file_path, max_sessions, top_items, item_popularity):
    training_rows = []
    start_time = time.time()

    with open(file_path, 'r') as file:
        for session_idx, line in enumerate(file, 1):
            if session_idx > max_sessions:
                break

            session_data = orjson.loads(line)
            events = session_data['events']

            recent_items = []
            for event in reversed(events):
                if event['aid'] not in recent_items:
                    recent_items.append(event['aid'])
                if len(recent_items) == 5:
                    break

            candidates = set(recent_items)
            for item in top_items:
                if len(candidates) >= 10:
                    break
                candidates.add(item)

            stats = defaultdict(lambda: {'clicks': 0, 'carts': 0, 'orders': 0, 'first_pos': -1, 'last_pos': -1})
            for idx, event in enumerate(events):
                if event['aid'] in candidates:
                    if stats[event['aid']]['first_pos'] == -1:
                        stats[event['aid']]['first_pos'] = idx
                    stats[event['aid']]['last_pos'] = idx
                    stats[event['aid']][event['type']] += 1

            for item, stat in stats.items():
                training_rows.append({
                    'clicks': stat['clicks'],
                    'carts': stat['carts'],
                    'orders': stat['orders'],
                    'first_pos': stat['first_pos'],
                    'last_pos': stat['last_pos'],
                    'popularity': item_popularity[item],
                    'label': int(stat['orders'] > 0)
                })

    print(f"[Training Data] Generated {len(training_rows)} rows in {time.time() - start_time:.2f}s")
    return pd.DataFrame(training_rows)


training_data = generate_training_data(TRAIN_PATH, MAX_TRAIN_SESSIONS, top_popular_items, item_popularity)
feature_columns = ['clicks', 'carts', 'orders', 'first_pos', 'last_pos', 'popularity']

def train_lightgbm_model(data, features):
    X = data[features]
    y = data['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    model = LGBMClassifier(
        objective='binary',
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"[Model Training] Completed in {time.time() - start_time:.2f}s")
    return model


model = train_lightgbm_model(training_data, feature_columns)

def create_submission(file_path, top_items, item_popularity, model, features, output_file='submission.csv'):
    predictions = []
    start_time = time.time()

    with open(file_path, 'r') as file:
        for line in file:
            session_data = orjson.loads(line)
            events = session_data['events']

            recent_items = []
            for event in reversed(events):
                if event['aid'] not in recent_items:
                    recent_items.append(event['aid'])
                if len(recent_items) == 5:
                    break

            candidates = set(recent_items)
            for item in top_items:
                if len(candidates) >= 10:
                    break
                candidates.add(item)

            candidate_features = []
            for item in candidates:
                stats = {'clicks': 0, 'carts': 0, 'orders': 0, 'first_pos': -1, 'last_pos': -1}
                for idx, event in enumerate(events):
                    if event['aid'] == item:
                        if stats['first_pos'] == -1:
                            stats['first_pos'] = idx
                        stats['last_pos'] = idx
                        stats[event['type']] += 1
                stats['popularity'] = item_popularity[item]
                stats['aid'] = item
                candidate_features.append(stats)

            candidate_df = pd.DataFrame(candidate_features)
            candidate_df['score'] = model.predict_proba(candidate_df[features])[:, 1]
            top_10_items = candidate_df.sort_values('score', ascending=False).head(10)['aid'].tolist()

            session_id = session_data['session']
            labels = ' '.join(map(str, top_10_items))
            predictions.extend([
                {'session_type': f'{session_id}_clicks', 'labels': labels},
                {'session_type': f'{session_id}_carts', 'labels': labels},
                {'session_type': f'{session_id}_orders', 'labels': labels}
            ])

    print(f"[Submission] Generated {len(predictions)} rows in {time.time() - start_time:.2f}s")
    pd.DataFrame(predictions).to_csv(output_file, index=False)


create_submission(TEST_PATH, top_popular_items, item_popularity, model, feature_columns)

