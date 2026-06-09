!pip install --upgrade pip
!pip install scikit-learn numpy pandas rake-nltk nltk matplotlib



import nltk
nltk.download('stopwords')
nltk.download('punkt')



# Single-file Kaggle-friendly Habit & Mood Agent demo
# Paste this into one Kaggle notebook cell and run.
# It will install missing packages if needed, then run the demo.

import sys, subprocess, pkgutil, os, json, uuid, pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

# ---------------------------
# Auto-install helper
# ---------------------------
def ensure_packages(pkgs):
    to_install = []
    for p in pkgs:
        if pkgutil.find_loader(p) is None:
            to_install.append(p)
    if to_install:
        print("Installing packages:", to_install)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + to_install)
    else:
        print("All required packages present.")

required = ["scikit-learn","numpy","pandas","rake-nltk","nltk","matplotlib","scipy"]
ensure_packages(required)

# Download nltk data if missing
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords", quiet=True)

# ---------------------------
# Imports (safe now)
# ---------------------------
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from scipy import sparse
import sqlite3

# silence
warnings.filterwarnings("ignore")

# ---------------------------
# Paths and data dir
# ---------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "entries.db"
TF_META = DATA_DIR / "tf_memory_meta.pkl"
TF_VECT = DATA_DIR / "tf_memory_matrix.npz"

# ---------------------------
# Data model
# ---------------------------
@dataclass
class Entry:
    id: str
    timestamp: str
    text: str
    mood_score: float = None
    sleep_hours: float = None
    steps: int = None
    tags: list = None
    recommended_action: str = None
    adhered: bool = None

    def to_dict(self):
        return asdict(self)

# ---------------------------
# Storage (SQLite)
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS entries (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        text TEXT,
        mood_score REAL,
        sleep_hours REAL,
        steps INTEGER,
        tags TEXT,
        recommended_action TEXT,
        adhered INTEGER
    )
    """)
    conn.commit()
    return conn

def save_entry(conn, entry: Entry):
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO entries
        (id,timestamp,text,mood_score,sleep_hours,steps,tags,recommended_action,adhered)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        entry.id, entry.timestamp, entry.text, entry.mood_score,
        entry.sleep_hours, entry.steps, json.dumps(entry.tags or []),
        entry.recommended_action, 1 if entry.adhered else 0 if entry.adhered is False else None
    ))
    conn.commit()

def load_entries(conn, limit=1000, asc=False):
    order = "ASC" if asc else "DESC"
    c = conn.cursor()
    c.execute(f"SELECT * FROM entries ORDER BY timestamp {order} LIMIT ?", (limit,))
    rows = c.fetchall()
    entries = []
    for r in rows:
        entries.append({
            'id': r[0], 'timestamp': r[1], 'text': r[2], 'mood_score': r[3],
            'sleep_hours': r[4], 'steps': r[5], 'tags': json.loads(r[6]) if r[6] else [],
            'recommended_action': r[7], 'adhered': bool(r[8]) if r[8] is not None else None
        })
    return entries

# ---------------------------
# TF-IDF memory (robust on Kaggle)
# ---------------------------
class TFIDFMemory:
    def __init__(self):
        self.meta = []
        self.vectorizer = None
        self.doc_matrix = None
        self._load()

    def _load(self):
        try:
            if TF_META.exists() and TF_VECT.exists():
                self.meta = pickle.load(open(TF_META, "rb"))
                self.doc_matrix = sparse.load_npz(str(TF_VECT))
                # rebuild vectorizer on load by fitting on texts
                texts = [t for (_id,t) in self.meta]
                self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
                self.doc_matrix = self.vectorizer.fit_transform(texts)
        except Exception as e:
            # if any problem, start fresh
            self.meta = []
            self.vectorizer = None
            self.doc_matrix = None

    def _save(self):
        TF_META.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.meta, open(TF_META, "wb"))
        if self.doc_matrix is not None:
            sparse.save_npz(str(TF_VECT), self.doc_matrix)

    def add(self, texts, ids):
        new_meta = list(zip(ids, texts))
        all_texts = [t for (_id,t) in self.meta] + texts
        self.meta.extend(new_meta)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self.doc_matrix = self.vectorizer.fit_transform(all_texts)
        self._save()

    def search(self, query, k=5):
        if self.doc_matrix is None or self.vectorizer is None:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.doc_matrix)[0]
        top_idx = sims.argsort()[-k:][::-1]
        results = []
        for idx in top_idx:
            if idx < len(self.meta):
                results.append(self.meta[idx])
        return results

# ---------------------------
# NLP helpers
# ---------------------------
rake = Rake()
POS = {"good","happy","great","calm","relaxed","joy","content","energetic","okay","fine"}
NEG = {"sad","anxious","angry","tired","stressed","depressed","lonely","bad","upset","worried","overwhelmed"}

def simple_mood_score(text: str) -> float:
    t = text.lower()
    p = sum(1 for w in POS if w in t)
    n = sum(1 for w in NEG if w in t)
    if p + n == 0:
        return 0.0
    return (p - n) / (p + n)

def extract_keywords(text, top_n=6):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:top_n]

# ---------------------------
# Agents
# ---------------------------
class Ingestor:
    def __init__(self, conn, memory: TFIDFMemory):
        self.conn = conn
        self.memory = memory

    def ingest(self, text, sleep_hours=None, steps=None):
        eid = str(uuid.uuid4())
        ts = datetime.utcnow().isoformat()
        mood = simple_mood_score(text)
        tags = extract_keywords(text)
        entry = Entry(id=eid, timestamp=ts, text=text, mood_score=mood,
                      sleep_hours=sleep_hours, steps=steps, tags=tags)
        save_entry(self.conn, entry)
        self.memory.add([text], [eid])
        return entry.to_dict()

class AnalyzerAgent:
    def aggregate(self, entries):
        moods = [e['mood_score'] for e in entries if e.get('mood_score') is not None]
        if not moods:
            return {'avg_mood': 0.0, 'std_mood': 0.0, 'trend': 'stable'}
        avg = float(np.mean(moods))
        std = float(np.std(moods))
        trend = 'stable'
        if len(moods) >= 3:
            if moods[-1] - moods[0] > 0.2:
                trend = 'improving'
            elif moods[-1] - moods[0] < -0.2:
                trend = 'declining'
        return {'avg_mood': avg, 'std_mood': std, 'trend': trend}

class InsightAgent:
    def find_triggers(self, entries, top_k=10):
        neg_texts = [e['text'] for e in entries if (e.get('mood_score') or 0) < 0]
        pos_texts = [e['text'] for e in entries if (e.get('mood_score') or 0) >= 0]
        corpus = neg_texts + pos_texts
        if not corpus or len(neg_texts) == 0:
            return []
        vect = TfidfVectorizer(stop_words='english', max_features=2000)
        X = vect.fit_transform(corpus)
        feature_names = vect.get_feature_names_out()
        n_neg = len(neg_texts)
        neg_mean = X[:n_neg].mean(axis=0)
        scores = np.asarray(neg_mean).ravel()
        top_idx = np.argsort(scores)[-top_k:][::-1]
        triggers = [feature_names[i] for i in top_idx]
        return triggers

class RecommendationAgent:
    def recommend(self, last_entry, insights):
        recs = []
        if last_entry.get('sleep_hours') is not None and last_entry['sleep_hours'] < 6:
            recs.append("Aim for 7–8 hours of sleep; try a wind-down routine 20–30 min before bed.")
        if last_entry.get('steps') is not None and last_entry['steps'] < 2000:
            recs.append("Do a 20-minute walk today to boost mood.")
        if last_entry.get('mood_score') is not None and last_entry['mood_score'] < -0.3:
            recs.append("Try a 5-minute breathing exercise and re-log your mood afterward.")
        if insights:
            recs.append(f"Watch for: {', '.join(insights[:3])}. These words often appear with low mood.")
        if not recs:
            recs.append("No urgent changes recommended; keep logging consistently.")
        return " ".join(recs)

# ---------------------------
# Evaluation
# ---------------------------
def compute_metrics(conn, baseline_days=3, intervention_days=7):
    df = pd.read_sql_query("SELECT * FROM entries ORDER BY timestamp ASC", conn, parse_dates=['timestamp'])
    if df.empty:
        return {}
    df['mood_score'] = df['mood_score'].fillna(0)
    baseline = df.iloc[:baseline_days]
    post = df.iloc[baseline_days:baseline_days+intervention_days]
    res = {}
    res['baseline_avg_mood'] = float(baseline['mood_score'].mean()) if not baseline.empty else None
    res['post_avg_mood'] = float(post['mood_score'].mean()) if not post.empty else None
    res['delta_avg_mood'] = None if res['baseline_avg_mood'] is None or res['post_avg_mood'] is None else res['post_avg_mood'] - res['baseline_avg_mood']
    res['baseline_std'] = float(baseline['mood_score'].std()) if not baseline.empty else None
    res['post_std'] = float(post['mood_score'].std()) if not post.empty else None
    adherence = None
    if 'adhered' in df.columns:
        s = df['adhered'].dropna()
        adherence = float(s.sum())/len(s) if len(s)>0 else None
    res['adherence_rate'] = adherence
    return res

# ---------------------------
# Demo runner (non-interactive)
# ---------------------------
def run_demo():
    conn = init_db()
    mem = TFIDFMemory()
    ing = Ingestor(conn, mem)
    analyzer = AnalyzerAgent()
    insighter = InsightAgent()
    recommender = RecommendationAgent()

    # Clean DB for demo (optional) - comment out if you want persistent data
    try:
        c = conn.cursor()
        c.execute("DELETE FROM entries")
        conn.commit()
    except Exception:
        pass

    # 3 baseline entries
    baseline_texts = [
        ("I slept terribly and felt exhausted at work", 5, 800),
        ("Work meeting stressed me out, I felt anxious and upset", 6, 1200),
        ("A neutral day, worried about deadlines but managed", 6.5, 1500)
    ]

    # 7 intervention entries
    intervention_texts = [
        ("Took a 20 minute walk and felt marginally better", 7, 3000),
        ("Tried breathing exercises; mood improved", 7.5, 2500),
        ("Met a friend, felt happy and relaxed", 8, 4000),
        ("Still worried about work but the walk helped", 7, 2800),
        ("Improved sleep hygiene last night, mood better", 7.5, 2000),
        ("Productive and calm today, felt content", 8, 3500),
        ("Good routine, stable mood", 7.5, 3000)
    ]

    print("Ingesting baseline entries...")
    for txt, slp, stp in baseline_texts:
        e = ing.ingest(txt, sleep_hours=slp, steps=stp)
    print("Ingesting intervention entries...")
    for txt, slp, stp in intervention_texts:
        e = ing.ingest(txt, sleep_hours=slp, steps=stp)

    # Load and show recent entries
    entries = load_entries(conn, limit=50, asc=True)
    print(f"\nTotal entries ingested: {len(entries)}\n")
    for i,e in enumerate(entries):
        print(f"{i+1}. {e['timestamp']} | mood: {e['mood_score']:.2f} | sleep: {e['sleep_hours']} | steps: {e['steps']}")
        print("   ", e['text'])
    print("")

    # Analysis
    stats = analyzer.aggregate(entries)
    triggers = insighter.find_triggers(entries)
    print("Aggregate stats:", stats)
    print("Top triggers:", triggers[:10])

    # Recommendation for last entry
    last = entries[-1]
    rec = recommender.recommend(last, triggers)
    print("\nRecommendation for last entry:")
    print("-", rec)

    # Semantic search example
    print("\nSemantic memory search for 'anxious':")
    hits = mem.search("anxious", k=5)
    if hits:
        for hid, txt in hits:
            print("-", hid, "|", txt[:140])
    else:
        print("No memory hits (memory empty?)")

    # Evaluation metrics
    metrics = compute_metrics(conn)
    print("\nEvaluation metrics (baseline vs intervention):")
    for k,v in metrics.items():
        print(f"  {k}: {v}")

    # Save CSV export
    df = pd.read_sql_query("SELECT * FROM entries ORDER BY timestamp ASC", conn, parse_dates=['timestamp'])
    csv_path = DATA_DIR / "entries_export.csv"
    df.to_csv(csv_path, index=False)
    print("\nExported entries CSV to:", csv_path)

    # Plot mood trend and save
    try:
        df_plot = df.copy()
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
        df_plot = df_plot.sort_values('timestamp')
        plt.figure(figsize=(8,3))
        plt.plot(df_plot['timestamp'], df_plot['mood_score'], marker='o')
        plt.title('Mood trend over time (demo)')
        plt.xlabel('Time')
        plt.ylabel('Mood score (-1..1)')
        plt.tight_layout()
        plot_path = DATA_DIR / "mood_trend.png"
        plt.savefig(plot_path)
        print("Saved mood trend plot to:", plot_path)
    except Exception as e:
        print("Plot failed:", e)

    print("\nDemo complete. Files written to data/ directory.")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    run_demo()


