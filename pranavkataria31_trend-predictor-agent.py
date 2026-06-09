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


import os, sys
os.kill(os.getpid(), 9)





# Trend Predictor Agent — MVP

This repo contains a minimal, runnable MVP for a Trend Predictor Agent for YouTubers.
It demonstrates ingestion (Google Trends + YouTube), topic clustering, momentum scoring,
LLM-based content generation (OpenAI), and a simple API + scheduler.

> WARNING: This is an MVP scaffold. Replace placeholders (API keys, quotas, scraping logic)
with production-grade implementations before any real deployment.

## Files
- `app.py` — FastAPI app and endpoints
- `models.py` — Pydantic / SQLModel models and DB setup
- `ingest.py` — data ingestion helpers (Google Trends & YouTube stubs)
- `cluster.py` — embedding, clustering, topic formation
- `generator.py` — LLM generation (titles, scripts) using OpenAI
- `scheduler.py` — periodic job runner using APScheduler
- `requirements.txt` — pip requirements
- `README.md` — this file

## Quick start (local)
1. Create a Python 3.10+ virtualenv
2. `pip install -r requirements.txt`
3. Set environment variables:
   - `OPENAI_API_KEY`
   - `YT_API_KEY` (optional, for YouTube ingestion)
   - `TELEGRAM_BOT_TOKEN` (optional, for alerts)
4. `uvicorn app:app --reload`
5. Visit `http://127.0.0.1:8000/docs` for API docs.

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

