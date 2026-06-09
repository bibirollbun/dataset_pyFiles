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








# Trend Predictor Agent â€” MVP

This repo contains a minimal, runnable MVP for a Trend Predictor Agent for YouTubers.
It demonstrates ingestion (Google Trends + YouTube), topic clustering, momentum scoring,
LLM-based content generation (OpenAI), and a simple API + scheduler.

> WARNING: This is an MVP scaffold. Replace placeholders (API keys, quotas, scraping logic)
with production-grade implementations before any real deployment.

## Files
app py FastAPI app and endpoints
models py Pydantic / SQLModel models and DB setup
ingest py data ingestion helpers (Google Trends & YouTube stubs)
cluster py embedding clustering topic formation
generator py LLM generation (titles  scripts) using OpenAI
scheduler py periodic job runner using APScheduler
requirements txt pip requirements
README md this file

## Quick start (local)
1 Create a Python 3.10+ virtualenv
2`pip install -r requirements.txt`
3 Set environment variables:
   OPENAI_API_KEY`
   YT_API_KEY(optional for YouTube ingestion)
   TELEGRAM_BOT_TOKEN (optional, for alerts)
4. `uvicorn app:app --reload`
5. Visit `http://127.0.0.1:8000/docs` for API docs

## Notes
- This repo uses OpenAI for embeddings & generation. Costs apply.
- The ingestion code uses `pytrends` for Google Trends; YouTube uses `google-api-python-client`.
- Clustering uses DBSCAN on OpenAI embeddings.


from pytrends.request import TrendReq
from googleapiclient.discovery import build
import os
from models import RawSignal, get_session

# Basic Google Trends ingestion
pytrends = TrendReq(hl='en-US', tz=360)

def fetch_google_trends(kw_list=None):
    """Fetch rising queries for keywords (or general trending searches if kw_list None).
    Returns list of dicts: {source, text, extra}
    """
    results = []
    if kw_list:
        for kw in kw_list:
            try:
                pytrends.build_payload([kw], timeframe='now 1-d')
                related = pytrends.related_queries()
                rq = related.get(kw, {}).get('rising')
                if rq is not None:
                    for idx, row in rq.iterrows():
                        results.append({'source': 'google_trends', 'text': row['query'], 'extra': str(row.to_dict())})
            except Exception as e:
                print('pytrends error', e)
    else:
        # General trending searches (pytrends has `trending_searches`)
        try:
            ts = pytrends.trending_searches(pn='united_states')
            for i, val in enumerate(ts[0:50]):
                results.append({'source': 'google_trends', 'text': val, 'extra': None})
        except Exception as e:
            print('pytrends trending error', e)
    # Save to DB
    session = get_session()
    for r in results:
        s = RawSignal(source=r['source'], text=r['text'], extra=r.get('extra'))
        session.add(s)
    session.commit()
    session.close()
    return results

# Basic YouTube ingestion (search for recent videos matching query)
YT_API_KEY = os.environ.get('YT_API_KEY')

def fetch_youtube_videos(q, max_results=10):
    if not YT_API_KEY:
        print('YT_API_KEY not set; returning empty list')
        return []
    youtube = build('youtube', 'v3', developerKey=YT_API_KEY)
    req = youtube.search().list(q=q, part='snippet', type='video', order='date', maxResults=max_results)
    res = req.execute()
    items = res.get('items', [])
    results = []
    for it in items:
        title = it['snippet']['title']
        desc = it['snippet'].get('description', '')
        results.append({'source':'youtube', 'text': title, 'extra': desc})
    # persist
    session = get_session()
    for r in results:
        s = RawSignal(source=r['source'], text=r['text'], extra=r.get('extra'))
        session.add(s)
    session.commit()
    session.close()
    return results


from typing import Optional, List
from sqlmodel import SQLModel, Field, create_engine, Session, select
from datetime import datetime

DATABASE_URL = "sqlite:///./trend_agent.db"
engine = create_engine(DATABASE_URL, echo=False)

class RawSignal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source: str
    text: str
    extra: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Topic(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    topic_key: str
    top_keywords: str  # comma-separated
    score: float
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    example_titles: Optional[str] = None
    generated_assets: Optional[str] = None

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    return Session(engine)


from pytrends.request import TrendReq
from googleapiclient.discovery import build
import os
from models import RawSignal, get_session

# Basic Google Trends ingestion
pytrends = TrendReq(hl='en-US', tz=360)

def fetch_google_trends(kw_list=None):
    """Fetch rising queries for keywords (or general trending searches if kw_list None).
    Returns list of dicts: {source, text, extra}
    """
    results = []
    if kw_list:
        for kw in kw_list:
            try:
                pytrends.build_payload([kw], timeframe='now 1-d')
                related = pytrends.related_queries()
                rq = related.get(kw, {}).get('rising')
                if rq is not None:
                    for idx, row in rq.iterrows():
                        results.append({'source': 'google_trends', 'text': row['query'], 'extra': str(row.to_dict())})
            except Exception as e:
                print('pytrends error', e)
    else:
        # General trending searches (pytrends has `trending_searches`)
        try:
            ts = pytrends.trending_searches(pn='united_states')
            for i, val in enumerate(ts[0:50]):
                results.append({'source': 'google_trends', 'text': val, 'extra': None})
        except Exception as e:
            print('pytrends trending error', e)
    # Save to DB
    session = get_session()
    for r in results:
        s = RawSignal(source=r['source'], text=r['text'], extra=r.get('extra'))
        session.add(s)
    session.commit()
    session.close()
    return results

# Basic YouTube ingestion (search for recent videos matching query)
YT_API_KEY = os.environ.get('YT_API_KEY')

def fetch_youtube_videos(q, max_results=10):
    if not YT_API_KEY:
        print('YT_API_KEY not set; returning empty list')
        return []
    youtube = build('youtube', 'v3', developerKey=YT_API_KEY)
    req = youtube.search().list(q=q, part='snippet', type='video', order='date', maxResults=max_results)
    res = req.execute()
    items = res.get('items', [])
    results = []
    for it in items:
        title = it['snippet']['title']
        desc = it['snippet'].get('description', '')
        results.append({'source':'youtube', 'text': title, 'extra': desc})
    # persist
    session = get_session()
    for r in results:
        s = RawSignal(source=r['source'], text=r['text'], extra=r.get('extra'))
        session.add(s)
    session.commit()
    session.close()
    return results


from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import openai
import os
from models import get_session, Topic

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

EMBED_MODEL = "text-embedding-3-small"


def embed_texts(texts):
    # returns list of vectors
    resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
    return [e['embedding'] for e in resp['data']]


def cluster_recent_signals(top_n=100):
    from models import RawSignal
    session = get_session()
    signals = session.exec("SELECT * FROM rawsignal ORDER BY timestamp DESC LIMIT :n", {'n': top_n}).all()
    texts = [s.text for s in signals]
    if not texts:
        return []
    embeds = embed_texts(texts)
    X = np.array(embeds)
    # DBSCAN on cosine distances
    D = cosine_distances(X)
    clustering = DBSCAN(metric='precomputed', eps=0.35, min_samples=2).fit(D)
    labels = clustering.labels_
    clusters = {}
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        clusters.setdefault(lab, []).append(texts[i])
    topics = []
    for lab, items in clusters.items():
        # simple keywords: top frequent words
        keywords = extract_keywords(items)
        # simple momentum score: cluster size
        score = min(1.0, len(items) / 10.0)
        topic_key = f"topic_{lab}_{int(np.random.randint(1e6))}"
        t = Topic(topic_key=topic_key, top_keywords=','.join(keywords[:6]), score=score, example_titles=items[0])
        session.add(t)
        session.commit()
        topics.append(t)
    session.close()
    return topics


def extract_keywords(texts, top_k=10):
    # naive keyword extraction by frequency
    from collections import Counter
    import re
    words = []
    for t in texts:
        toks = re.findall(r"[a-zA-Z0-9]+", t.lower())
        words.extend([w for w in toks if len(w) > 2])
    c = Counter(words)
    return [w for w, _ in c.most_common(top_k)]


from apscheduler.schedulers.background import BackgroundScheduler
from ingest import fetch_google_trends, fetch_youtube_videos
from cluster import cluster_recent_signals
from generator import generate_for_topic
from telegram import Bot
import os

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')  # user chat to send alerts to (optional)

scheduler = BackgroundScheduler()


def job_ingest_and_cluster():
    print('Running ingest job...')
    # fetch general trends
    try:
        fetch_google_trends()
    except Exception as e:
        print('fetch google error', e)
    # sample youtube searches for "how to" (replace with better queries)
    try:
        fetch_youtube_videos('how to')
    except Exception as e:
        print('fetch youtube error', e)
    # cluster
    try:
        topics = cluster_recent_signals(200)
        print(f'Found {len(topics)} topics')
        # generate assets for top topics
        for t in topics[:3]:
            try:
                assets = generate_for_topic(t)
                # send telegram alert
                if TELEGRAM_TOKEN and CHAT_ID:
                    bot = Bot(token=TELEGRAM_TOKEN)
                    text = f"New Topic: {t.top_keywords}\nScore: {t.score}\nTitles:\n{assets['titles']}"
                    bot.send_message(chat_id=CHAT_ID, text=text)
            except Exception as e:
                print('generate error', e)
    except Exception as e:
        print('cluster error', e)


def start_scheduler():
    scheduler.add_job(job_ingest_and_cluster, 'interval', minutes=30, id='ingest_job')
    scheduler.start()
    print('Scheduler started')


from fastapi import FastAPI, BackgroundTasks
from models import init_db, get_session, Topic
from scheduler import start_scheduler, job_ingest_and_cluster
from generator import generate_for_topic

app = FastAPI(title='Trend Predictor Agent')

@app.on_event('startup')
def on_startup():
    init_db()
    start_scheduler()

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/run_ingest')
def run_ingest(background: BackgroundTasks):
    background.add_task(job_ingest_and_cluster)
    return {'status':'ingest scheduled'}

@app.get('/topics')
def list_topics(limit: int = 20):
    session = get_session()
    topics = session.exec("SELECT * FROM topic ORDER BY score DESC LIMIT :limit", {'limit': limit}).all()
    session.close()
    return topics

@app.post('/topics/{topic_id}/generate')
def regenerate_assets(topic_id: int):
    session = get_session()
    t = session.get(Topic, topic_id)
    session.close()
    if not t:
        return {'error': 'topic not found'}
    assets = generate_for_topic(t)
    return assets


- Replace environment keys with: OPENAI_API_KEY, YT_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
- The clustering eps hyperparameter might need tuning depending on embedding model.
- DB: for production use Postgres; use Alembic for migrations.
- This scaffold uses synchronous OpenAI calls; consider async or batching for scale.
- Add error handling, retries, and rate-limiting.



import os
import time
import math
import json
import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from flask import Flask, request, jsonify
import pandas as pd

# ML & NLP
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# YouTube API
from googleapiclient.discovery import build

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from tqdm import tqdm

# initialize NLTK resources (may download the first time)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Configuration
YT_API_KEY = os.getenv('YT_API_KEY')  # user must set
MODEL_PATH = 'trend_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'
SCALER_PATH = 'scaler.joblib'

# Thresholding constants
TREND_VIEW_MULTIPLIER = 2.0  # label video as trending if views >= TREND_VIEW_MULTIPLIER * median

app = Flask(__name__)

# ---------------------- YouTube fetching utilities ----------------------

def get_youtube_client():
    if not YT_API_KEY:
        raise EnvironmentError('Please set the YT_API_KEY environment variable with your YouTube Data API v3 key.')
    return build('youtube', 'v3', developerKey=YT_API_KEY)


def fetch_videos_from_channel(channel_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """Fetch recent videos from a channel's uploads playlist."""
    yt = get_youtube_client()
    # Get uploads playlist id from channel
    ch_req = yt.channels().list(part='contentDetails', id=channel_id)
    ch_res = ch_req.execute()
    if not ch_res['items']:
        return []
    uploads_playlist = ch_res['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    videos = []
    nextPageToken = None
    while True:
        pl_req = yt.playlistItems().list(part='snippet', playlistId=uploads_playlist, maxResults=50, pageToken=nextPageToken)
        pl_res = pl_req.execute()
        for item in pl_res.get('items', []):
            video_id = item['snippet']['resourceId'].get('videoId')
            videos.append({'videoId': video_id, 'snippet': item['snippet']})
            if len(videos) >= max_results:
                return videos
        nextPageToken = pl_res.get('nextPageToken')
        if not nextPageToken:
            break
    return videos


def fetch_videos_by_search(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    yt = get_youtube_client()
    req = yt.search().list(q=query, part='snippet', type='video', maxResults=max_results)
    res = req.execute()
    videos = [{'videoId': item['id']['videoId'], 'snippet': item['snippet']} for item in res.get('items', [])]
    return videos


def fetch_video_statistics(video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    yt = get_youtube_client()
    stats = {}
    # You can fetch up to 50 ids per call
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        req = yt.videos().list(part='snippet,statistics,contentDetails', id=','.join(chunk))
        res = req.execute()
        for item in res.get('items', []):
            vid = item['id']
            stats[vid] = item
    return stats

# ---------------------- Feature engineering ----------------------

def text_preprocess(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    # simple tokenization
    tokens = [w for w in s.split() if w.isalpha() and w not in STOPWORDS]
    tokens = [STEMMER.stem(w) for w in tokens]
    return ' '.join(tokens)


def build_feature_dataframe(video_items: List[Dict[str, Any]], stats_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for vi in video_items:
        vid = vi.get('videoId')
        snip = stats_map.get(vid, {}).get('snippet', vi.get('snippet', {}))
        stats = stats_map.get(vid, {}).get('statistics', {})

        title = snip.get('title', '')
        description = snip.get('description', '')
        tags = snip.get('tags', []) or []
        published_at = snip.get('publishedAt')
        publish_dt = None
        if published_at:
            publish_dt = datetime.datetime.fromisoformat(published_at.replace('Z', '+00:00'))

        view_count = int(stats.get('viewCount', 0))
        like_count = int(stats.get('likeCount', 0)) if stats.get('likeCount') else 0
        comment_count = int(stats.get('commentCount', 0)) if stats.get('commentCount') else 0

        rows.append({
            'videoId': vid,
            'title': title,
            'description': description,
            'tags': ' '.join(tags),
            'publishedAt': publish_dt,
            'viewCount': view_count,
            'likeCount': like_count,
            'commentCount': comment_count,
            'age_days': (datetime.datetime.utcnow() - publish_dt).days if publish_dt else None
        })
    df = pd.DataFrame(rows)
    df['text_combined'] = (df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['tags'].fillna('')).apply(text_preprocess)
    # Fill numeric
    df['age_days'] = df['age_days'].fillna(0).astype(float)
    df['viewCount'] = df['viewCount'].fillna(0).astype(float)
    df['likeCount'] = df['likeCount'].fillna(0).astype(float)
    df['commentCount'] = df['commentCount'].fillna(0).astype(float)
    return df

# ---------------------- Labeling & model helpers ----------------------

def create_labels(df: pd.DataFrame, strategy: str = 'relative') -> pd.Series:
    """Create binary labels for trending.
    Strategies:
      - relative: mark as trending if viewCount >= TREND_VIEW_MULTIPLIER * median(viewCount)
      - absolute: mark as trending if viewCount >= absolute threshold
    """
    if strategy == 'relative':
        median = df['viewCount'].median() if len(df) else 0
        threshold = median * TREND_VIEW_MULTIPLIER
    else:
        threshold = 100000  # default absolute
    labels = (df['viewCount'] >= threshold).astype(int)
    return labels


def train_model(df: pd.DataFrame) -> Tuple[Any, Any, Any]:
    # Text vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_text = vectorizer.fit_transform(df['text_combined'].fillna(''))

    # Numeric features
    X_num = df[['age_days','likeCount','commentCount']].fillna(0).values
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Combine
    from scipy.sparse import hstack
    X = hstack([X_text, X_num_scaled])

    y = create_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print('Train finished â€” accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # persist artifacts
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return clf, vectorizer, scaler


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, None
    clf = joblib.load(MODEL_PATH)
    vect = joblib.load(VECTORIZER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, vect, scaler


def predict_from_row(row: pd.Series, clf, vect, scaler) -> Dict[str, Any]:
    text = row['text_combined']
    X_text = vect.transform([text])
    X_num = scaler.transform([[row['age_days'], row['likeCount'], row['commentCount']]])
    from scipy.sparse import hstack
    X = hstack([X_text, X_num])
    prob = clf.predict_proba(X)[0][1] if hasattr(clf, 'predict_proba') else None
    pred = int(clf.predict(X)[0])
    return {'predicted_label': pred, 'probability': float(prob) if prob is not None else None}

# ---------------------- Flask endpoints ----------------------

@app.route('/train', methods=['POST'])
def api_train():
    """Train model on videos provided in the request body.
    POST JSON schema:
    {
      "videos": [ { "videoId": "..." }, ... ]
    }
    OR
    {
      "channelIds": ["UCxxx", ...],
      "maxPerChannel": 50
    }
    """
    payload = request.get_json(force=True)
    videos = []
    if payload.get('videos'):
        vids = [v.get('videoId') for v in payload['videos'] if v.get('videoId')]
        stats_map = fetch_video_statistics(vids)
        video_items = []
        for vid in vids:
            snippet = stats_map.get(vid, {}).get('snippet', {})
            video_items.append({'videoId': vid, 'snippet': snippet})
        df = build_feature_dataframe(video_items, stats_map)
        clf, vect, scaler = train_model(df)
        return jsonify({'status': 'trained', 'n_samples': len(df)}), 200

    elif payload.get('channelIds'):
        channel_ids = payload['channelIds']
        max_per = int(payload.get('maxPerChannel', 50))
        all_video_items = []
        all_vids = []
        for ch in channel_ids:
            print('Fetching channel', ch)
            items = fetch_videos_from_channel(ch, max_results=max_per)
            vids = [it['videoId'] for it in items if it.get('videoId')]
            all_vids.extend(vids)
            all_video_items.extend(items)
        stats_map = fetch_video_statistics(all_vids)
        df = build_feature_dataframe(all_video_items, stats_map)
        clf, vect, scaler = train_model(df)
        return jsonify({'status': 'trained', 'n_samples': len(df)}), 200

    else:
        return jsonify({'error': 'Provide videos list or channelIds in JSON'}), 400


@app.route('/predict', methods=['POST'])
def api_predict():
    """Predict whether a video will trend. Provide either videoId or title+description in JSON.
    POST JSON schema examples:
      { "videoId": "..." }
      { "title": "...", "description": "...", "likeCount": 10, "commentCount": 0, "publishedAt": "2025-11-10T12:00:00Z" }
    """
    payload = request.get_json(force=True)
    clf, vect, scaler = load_artifacts()
    if clf is None:
        return jsonify({'error': 'Model artifacts not found. Train first via /train.'}), 400

    if payload.get('videoId'):
        vid = payload['videoId']
        stats_map = fetch_video_statistics([vid])
        item = stats_map.get(vid)
        if not item:
            return jsonify({'error': 'Video not found or API returned no data.'}), 404
        df = build_feature_dataframe([{'videoId': vid, 'snippet': item.get('snippet', {})}], {vid: item})
        row = df.iloc[0]
        out = predict_from_row(row, clf, vect, scaler)
        return jsonify({'videoId': vid, 'result': out}), 200

    else:
        title = payload.get('title','')
        description = payload.get('description','')
        likeCount = float(payload.get('likeCount', 0))
        commentCount = float(payload.get('commentCount', 0))
        publishedAt = payload.get('publishedAt')
        publish_dt = None
        if publishedAt:
            try:
                publish_dt = datetime.datetime.fromisoformat(publishedAt.replace('Z','+00:00'))
            except Exception:
                publish_dt = None
        age_days = (datetime.datetime.utcnow() - publish_dt).days if publish_dt else 0
        text_combined = text_preprocess(title + ' ' + description)
        row = pd.Series({
            'text_combined': text_combined,
            'age_days': float(age_days),
            'likeCount': likeCount,
            'commentCount': commentCount
        })
        out = predict_from_row(row, clf, vect, scaler)
        return jsonify({'input': {'title': title}, 'result': out}), 200

# ---------------------- CLI utilities & demo ----------------------

def demo_train_with_search(queries: List[str], max_per_query: int = 30):
    all_items = []
    all_vids = []
    for q in queries:
        print(f'Fetching search: {q}')
        items = fetch_videos_by_search(q, max_per_query)
        vids = [it['videoId'] for it in items if it.get('videoId')]
        all_vids.extend(vids)
        all_items.extend(items)
    stats_map = fetch_video_statistics(all_vids)
    df = build_feature_dataframe(all_items, stats_map)
    clf, vect, scaler = train_model(df)
    print('Demo train completed. Model saved to', MODEL_PATH)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trend Predictor Agent (single-file)')
    parser.add_argument('--serve', action='store_true', help='Run Flask server')
    parser.add_argument('--demo-train', action='store_true', help='Run a demo training using sample queries')
    parser.add_argument('--queries', nargs='*', default=['python tutorial', 'music video', 'comedy skit'], help='Queries for demo train')
    args = parser.parse_args()

    if args.demo_train:
        demo_train_with_search(args.queries)
    elif args.serve:
        app.run(host='0.0.0.0', port=8080)
    else:
        print('No arguments provided. To run server: python trend_predictor_agent.py --serve')
        print('To run demo training: python trend_predictor_agent.py --demo-train')



