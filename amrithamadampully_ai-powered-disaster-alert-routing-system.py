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
import json
import sqlite3
import requests
import logging
from datetime import datetime, timezone
from dateutil import parser as dtparser
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request, Response
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST


pip install apscheduler


NWS_USER_AGENT = os.getenv("NWS_USER_AGENT", "your-email@example.com (EmergencyAgent/1.0)")
SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", "60"))  # poll interval
DATABASE = os.getenv("DB_PATH", "alerts_memory.db")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # optional
SMTP_SENDER = os.getenv("SMTP_SENDER")  # optional: for e-mail sending (used if set)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# ---------------------------
# Observability / Metrics
# ---------------------------
ALERTS_SCRAPED = Counter("alerts_scraped_total", "Total alerts scraped from feeds")
ALERTS_ROUTED = Counter("alerts_routed_total", "Total alerts routed to users")
ALERTS_IGNORED = Counter("alerts_ignored_total", "Total alerts ignored (low severity or not targeted)")
ALERTS_ERRORS = Counter("alerts_errors_total", "Errors encountered")
LAST_SCRAPE_TIME = Gauge("last_scrape_unix_time", "Last scrape time (unix)")

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("emergency-agent")

# ---------------------------
# Minimal SQLite memory
# ---------------------------


# ---------------------------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    # users table: id, name, latitude, longitude, radius_km, subscribed_events (json), channels (json)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        telegram_chat_id TEXT,
        lat REAL,
        lon REAL,
        radius_km REAL DEFAULT 50,
        subscribed_events TEXT DEFAULT '[]',
        channels TEXT DEFAULT '["email"]'
    )
    """)
    # delivered alerts to avoid duplicates
    cur.execute("""
    CREATE TABLE IF NOT EXISTS delivered_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id TEXT UNIQUE,
        sent_at TEXT
    )
    """)
    conn.commit()
    conn.close()



def add_test_user():
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    # idempotent simple add (if not exists)
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute("""
        INSERT INTO users (name, email, telegram_chat_id, lat, lon, radius_km, subscribed_events, channels)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "Test User",
            "receiver@example.com",
            None,
            38.8977, -77.0365,  # White House (example)
            100,
            json.dumps(["Tornado Warning", "Flood Warning", "Severe Thunderstorm Warning"]),
            json.dumps(["email"])
        ))
        conn.commit()
    conn.close()


import math
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ---------------------------
# Scraper agent: NWS example
# Docs: https://api.weather.gov (use point or area queries). See NWS API docs.
# ---------------------------
NWS_BASE = "https://api.weather.gov"


def fetch_nws_alerts_for_point(lat, lon):
    """
    Fetch active alerts for a lat/lon point (NWS API provides alerts by point).
    Returns list of alert feature dicts.
    """
    try:
        url = f"{NWS_BASE}/alerts/active?point={lat},{lon}"
        headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        return features
    except Exception as e:
        logger.exception("Failed to fetch NWS alerts: %s", e)
        ALERTS_ERRORS.inc()
        return []


# Generic fetch from CAP feed (if you have a CAP feed URL)
def fetch_cap_feed(url):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        # CAP often returns XML. For brevity user can parse XML -> dict (left as exercise).
        return r.text
    except Exception as e:
        logger.exception("Failed to fetch CAP feed: %s", e)
        ALERTS_ERRORS.inc()
        return None

# ---------------------------
# Severity classifier
# Basic rule-based severity scoring using CAP fields: urgency, severity, certainty, event
# Hook: replace `rule_based_severity` with a ML model (load a model and call .predict_proba)
# ---------------------------
SEVERITY_MAP = {"Extreme": 100, "Severe": 75, "Moderate": 50, "Minor": 25, "Unknown": 10}
URGENCY_MAP = {"Immediate": 40, "Expected": 25, "Future": 10, "Past": 0, "Unknown": 5}
CERTAINTY_MAP = {"Observed": 20, "Likely": 10, "Possible": 5, "Unknown": 0}


def rule_based_severity(alert_feature):
    """
    Input: NWS alert feature (GeoJSON feature)
    Returns: score (0-100) and reasons
    """
    props = alert_feature.get("properties", {})
    event = props.get("event", "Unknown")
    severity = props.get("severity", "Unknown")
    urgency = props.get("urgency", "Unknown")
    certainty = props.get("certainty", "Unknown")
    headline = props.get("headline") or props.get("event")
    description = props.get("description") or ""
    score = 0
    reasons = []
    score += SEVERITY_MAP.get(severity, 10)
    reasons.append(f"severity={severity}")
    score += URGENCY_MAP.get(urgency, 5)
    reasons.append(f"urgency={urgency}")
    score += CERTAINTY_MAP.get(certainty, 0)
    reasons.append(f"certainty={certainty}")

    # Give extra points for keywords in headline/description
    keywords_extreme = ["evacuate", "take immediate action", "dangerous", "life-threatening"]
    for kw in keywords_extreme:
        if kw.lower() in (headline or "").lower() or kw.lower() in description.lower():
            score += 15
            reasons.append(f"keyword={kw}")

    # Clamp
    score = max(0, min(100, score))
    return score, reasons

# ---------------------------
# Routing agent: decide which users to notify and send via channels
# ---------------------------


import smtplib
from email.mime.text import MIMEText

def get_users():
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT id, name, email, telegram_chat_id, lat, lon, radius_km, subscribed_events, channels FROM users")
    rows = cur.fetchall()
    conn.close()
    users = []
    for r in rows:
        users.append({
            "id": r[0],
            "name": r[1],
            "email": r[2],
            "telegram_chat_id": r[3],
            "lat": r[4],
            "lon": r[5],
            "radius_km": r[6],
            "subscribed_events": json.loads(r[7]),
            "channels": json.loads(r[8])
        })
    return users


def has_been_delivered(alert_id):
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM delivered_alerts WHERE alert_id = ?", (alert_id,))
    r = cur.fetchone()
    conn.close()
    return r is not None


def mark_delivered(alert_id):
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO delivered_alerts (alert_id, sent_at) VALUES (?, ?)", (alert_id, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def send_email(to_email, subject, body):
    if not SMTP_SENDER or not SMTP_HOST:
        logger.warning("SMTP not configured; skipping email to %s", to_email)
        return False
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SMTP_SENDER
        msg['To'] = to_email

        s = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20)
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_SENDER, [to_email], msg.as_string())
        s.quit()
        return True
    except Exception as e:
        logger.exception("Failed to send email: %s", e)
        ALERTS_ERRORS.inc()
        return False


def send_telegram(chat_id, text):
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        logger.warning("Telegram not configured or chat_id missing")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.exception("Failed to send telegram: %s", e)
        ALERTS_ERRORS.inc()
        return False


def route_alert_to_subscribers(alert_feature, score):
    props = alert_feature.get("properties", {})
    alert_id = props.get("id") or props.get("identifier") or props.get("@id") or props.get("event") + "_" + props.get("sent", "")
    event_title = props.get("event")
    headline = props.get("headline") or ""
    description = props.get("description") or ""
    sent_time = props.get("sent") or props.get("sent", datetime.now(timezone.utc).isoformat())

    if has_been_delivered(alert_id):
        logger.info("Alert %s already delivered; skipping", alert_id)
        return

    users = get_users()
    sent_count = 0
    for u in users:
        # geographic match
        dist_km = haversine_km(u["lat"], u["lon"], props.get("geometry", {}).get("coordinates",[None,None])[1] if props.get("geometry") else u["lat"], props.get("geometry", {}).get("coordinates",[None,None])[0] if props.get("geometry") else u["lon"])
        # NWS alerts sometimes include affected zones; in production check CAP area desc or polygons.
        if dist_km <= u["radius_km"]:
            # event type match (if user subscribed)
            if u["subscribed_events"] and event_title not in u["subscribed_events"]:
                logger.debug("User %s not subscribed to %s", u["name"], event_title)
                continue

            # decide threshold — can be user-specific
            threshold = 30
            if score < threshold:
                ALERTS_IGNORED.inc()
                logger.info("Alert score %s below threshold for user %s", score, u["name"])
                continue

            # craft message
            msg = f"ALERT: {event_title}\n{headline}\n\n{description}\n\nSent: {sent_time}\nSeverity Score: {score}\nAlert ID: {alert_id}"
            # send via channels
            if "email" in u["channels"] and u["email"]:
                ok = send_email(u["email"], f"[Emergency Alert] {event_title}", msg)
                if ok:
                    sent_count += 1
            if "telegram" in u["channels"] and u["telegram_chat_id"]:
                ok = send_telegram(u["telegram_chat_id"], msg)
                if ok:
                    sent_count += 1
            # optionally add webhook, SMS, push, etc.

    if sent_count > 0:
        mark_delivered(alert_id)
        ALERTS_ROUTED.inc(sent_count)
        logger.info("Alert %s routed to %d channels", alert_id, sent_count)

# ---------------------------
# Main loop: scrape for each user location (de-duplicate alerts)
# ---------------------------


def process_cycle():
    logger.info("Starting scrape cycle")
    LAST_SCRAPE_TIME.set_to_current_time()
    init_db()  # ensure db present
    users = get_users()
    # For demo: build set of unique points to query (lat,lon)
    queried_points = {(u['lat'], u['lon']) for u in users}
    total_scraped = 0
    for (lat, lon) in queried_points:
        features = fetch_nws_alerts_for_point(lat, lon)
        ALERTS_SCRAPED.inc(len(features))
        total_scraped += len(features)
        for f in features:
            score, reasons = rule_based_severity(f)
            logger.info("Alert %s score=%s reasons=%s", f.get("id") or f.get("properties",{}).get("id", "unknown"), score, reasons)
            route_alert_to_subscribers(f, score)
    logger.info("Scrape cycle complete: scraped %d alerts", total_scraped)

# ---------------------------
# Flask server for metrics and simple admin endpoints
# ---------------------------


app = Flask(__name__)

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/health")
def health():
    return jsonify({"status":"ok", "time": datetime.now(timezone.utc).isoformat()})

# simple admin: add user (very simple; for demo)
@app.route("/admin/add_user", methods=["POST"])
def admin_add_user():
    payload = request.json
    required = ("name","lat","lon")
    if not all(k in payload for k in required):
        return jsonify({"error":"missing fields"}), 400
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO users (name, email, telegram_chat_id, lat, lon, radius_km, subscribed_events, channels)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        payload.get("name"),
        payload.get("email"),
        payload.get("telegram_chat_id"),
        float(payload.get("lat")),
        float(payload.get("lon")),
        float(payload.get("radius_km", 50)),
        json.dumps(payload.get("subscribed_events", [])),
        json.dumps(payload.get("channels", ["email"]))
    ))
    conn.commit()
    conn.close()
    return jsonify({"status":"ok"})


# ---------------------------
# Runner
# ---------------------------
if __name__ == "__main__":
    init_db()
    add_test_user()
    # Scheduler for scrape cycles
    scheduler = BackgroundScheduler()
    scheduler.add_job(process_cycle, 'interval', seconds=SCRAPE_INTERVAL_SECONDS)
    scheduler.start()
    logger.info("Scheduler started: scraping every %s seconds", SCRAPE_INTERVAL_SECONDS)

    # Run Flask for metrics and admin
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))




