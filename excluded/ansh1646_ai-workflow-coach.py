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


# Imports & basic config
import os, sys, asyncio, json, math, traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("AIWorkflowCoach_Extended")

# Output directory
os.makedirs("outputs", exist_ok=True)

# Notebook runtime safe async helper will be used later.


# Install required packages for Google Calendar + Slack (run once)
# In Kaggle you may skip install if already present. This cell is safe to re-run.
import sys
import subprocess

def pip_install(pkg):
    try:
        __import__(pkg.split('==')[0])
        print(f"✅ {pkg} already available")
    except Exception:
        print(f"⚡ Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

pip_install("google-api-python-client")
pip_install("google-auth-httplib2")
pip_install("google-auth-oauthlib")
pip_install("google-auth")
pip_install("slack_sdk")
print("\n✅ Dependency install step complete. If any install failed, check the output above.")


import os

# Ensure outputs folder exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")
    print("Created 'outputs/' directory.")
else:
    print("'outputs/' directory already exists.")


# Configuration: load credentials from Kaggle Secrets or env vars
# - Recommended: add two Kaggle secrets:
#     * GOOGLE_SERVICE_ACCOUNT_JSON  (contents of a service account JSON)
#     * SLACK_BOT_TOKEN               (xoxb-... token)
# - Also set optional: SLACK_DEFAULT_CHANNEL (e.g. "#general") or owner->channel mapping.

import os, json, logging
logger = logging.getLogger("AIWorkflowCoach_Creds")

# Try Kaggle secrets first (if running on Kaggle)
GOOGLE_SERVICE_ACCOUNT_INFO = None
SLACK_BOT_TOKEN = None
SLACK_DEFAULT_CHANNEL = os.environ.get("SLACK_DEFAULT_CHANNEL", "#general")

try:
    from kaggle_secrets import UserSecretsClient
    us = UserSecretsClient()
    # service account JSON (full JSON string)
    try:
        sa_json = us.get_secret("GOOGLE_SERVICE_ACCOUNT_JSON")
        if sa_json:
            GOOGLE_SERVICE_ACCOUNT_INFO = json.loads(sa_json)
            logger.info("✅ Loaded GOOGLE_SERVICE_ACCOUNT_JSON from Kaggle Secrets")
    except Exception as e:
        logger.info("No Google SA in Kaggle Secrets: %s", e)
    try:
        SLACK_BOT_TOKEN = us.get_secret("SLACK_BOT_TOKEN") or SLACK_BOT_TOKEN
        if SLACK_BOT_TOKEN:
            logger.info("✅ Loaded SLACK_BOT_TOKEN from Kaggle Secrets")
    except Exception as e:
        logger.info("No Slack token in Kaggle Secrets: %s", e)
except Exception:
    logger.info("Kaggle secrets not available; falling back to environment variables")

# Also support environment variables for local dev
if not GOOGLE_SERVICE_ACCOUNT_INFO:
    sa_env = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_env:
        try:
            GOOGLE_SERVICE_ACCOUNT_INFO = json.loads(sa_env)
            logger.info("✅ Loaded GOOGLE_SERVICE_ACCOUNT_JSON from environment")
        except Exception:
            logger.warning("Failed to parse GOOGLE_SERVICE_ACCOUNT_JSON from environment")

if not SLACK_BOT_TOKEN:
    SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

# Optional: impersonated user email for service account (if needed)
CALENDAR_IMPERSONATED_USER = os.environ.get("CALENDAR_IMPERSONATED_USER")  # e.g., "alice@example.com"

# Final flags
USE_GOOGLE_CALENDAR = GOOGLE_SERVICE_ACCOUNT_INFO is not None
USE_SLACK = SLACK_BOT_TOKEN is not None

logger.info(f"USE_GOOGLE_CALENDAR={USE_GOOGLE_CALENDAR}, USE_SLACK={USE_SLACK}, default_channel={SLACK_DEFAULT_CHANNEL}")


# Utilities & safe async runner
import json
import pandas as pd

def now_date():
    return pd.Timestamp(datetime.utcnow().date())

def safe_save_json(obj, path):
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"Saved JSON to {path}")
    except Exception as e:
        print(f"Error writing JSON: {e}")

# Safe async runner that works in Jupyter
async def _run_coro(coro):
    return await coro

def run_async_safe(coro):
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # schedule and wait
            task = loop.create_task(coro)
            return task
        else:
            return asyncio.run(coro)
    except RuntimeError:
        # no running loop
        return asyncio.run(coro)


# Data loader & sample tasks/notes
def load_data():
    today = now_date()
    tasks = pd.DataFrame([
        {"id":1,"title":"API /compute","owner":"alice","status":"todo","estimated_hours":8,"hours_done":0,"due_date":today - pd.Timedelta(days=6),"impact":5,"risk_count":1},
        {"id":2,"title":"Write tests","owner":"bob","status":"in_progress","estimated_hours":6,"hours_done":2,"due_date":today - pd.Timedelta(days=1),"impact":3,"risk_count":0},
        {"id":3,"title":"Design doc","owner":"charlie","status":"blocked","estimated_hours":4,"hours_done":0,"due_date":today - pd.Timedelta(days=11),"impact":4,"risk_count":2},
        {"id":4,"title":"Refactor DB","owner":"alice","status":"todo","estimated_hours":10,"hours_done":1,"due_date":today + pd.Timedelta(days=9),"impact":4,"risk_count":1},
        {"id":5,"title":"Frontend bugfix","owner":"dana","status":"in_progress","estimated_hours":3,"hours_done":1,"due_date":today + pd.Timedelta(days=1),"impact":2,"risk_count":0},
        {"id":6,"title":"Prepare slides","owner":"bob","status":"todo","estimated_hours":5,"hours_done":0,"due_date":today + pd.Timedelta(days=2),"impact":3,"risk_count":0},
    ])
    notes = pd.DataFrame([
        {"date":today - pd.Timedelta(days=1),"author":"alice","text":"blocked on 3 - waiting for schema"},
        {"date":today,"author":"bob","text":"tests need clarification: which fields to assert?"},
        {"date":today - pd.Timedelta(days=5),"author":"alice","text":"feeling overwhelmed - too many high-effort tasks"},
        {"date":today - pd.Timedelta(days=2),"author":"charlie","text":"stuck: can't reproduce bug locally"},
    ])
    return tasks, notes

tasks_df, notes_df = load_data()
print("Sample tasks:", len(tasks_df), "notes:", len(notes_df))


# Core analysis tools
from typing import Optional, Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta, timezone

def compute_priority_scores(tasks_df: pd.DataFrame,
                            today: Optional[pd.Timestamp] = None,
                            w_delay=0.4, w_impact=0.3, w_effort=0.2, w_risk=0.1
                           ) -> pd.DataFrame:
    """
    Computes the numeric priority score for each task using:
    delay_score, impact_score, effort_score, risk_score.
    Produces priority_0_100 for easy ranking.
    """

    if today is None:
        today = pd.Timestamp(datetime.now(timezone.utc).date())

    df = tasks_df.copy()
    df['due_date'] = pd.to_datetime(df['due_date'])

    # delay
    df['delay_days'] = (today - df['due_date']).dt.days.clip(lower=0)
    df['delay_score'] = (df['delay_days'] / 7.0).clip(upper=1.0)

    # remaining effort
    df['remaining_effort'] = (df['estimated_hours'] - df['hours_done']).clip(lower=0)
    max_effort = max(1.0, float(df['remaining_effort'].quantile(0.9))) if not df.empty else 1.0
    df['effort_score'] = (df['remaining_effort'] / max_effort).clip(0, 1)

    # impact
    df['impact_score'] = (df.get('impact', 0) / 5.0).clip(0, 1)

    # risk score
    if 'risk_count' in df.columns:
        max_risk = max(1.0, float(df['risk_count'].max()))
        df['risk_score'] = (df['risk_count'] / max_risk).clip(0, 1)
    else:
        df['risk_score'] = 0.0

    # weighted priority
    df['priority_score'] = (
        w_delay * df['delay_score'] +
        w_impact * df['impact_score'] +
        w_effort * df['effort_score'] +
        w_risk   * df['risk_score']
    )

    # normalize to 0–100
    max_ps = float(df['priority_score'].max()) if not df.empty else 1e-6
    df['priority_0_100'] = (df['priority_score'] / max(max_ps, 1e-6)) * 100.0

    return df.sort_values('priority_score', ascending=False).reset_index(drop=True)


def detect_bottlenecks(tasks_df: pd.DataFrame,
                       delay_threshold_days: int = 3
                      ) -> Dict[str, pd.DataFrame]:
    """
    Detect:
    - blocked tasks
    - tasks with no owner
    - overdue tasks (past due date and not done)
    - long-delay tasks (delay_days > threshold)
    """

    today = pd.Timestamp(datetime.now(timezone.utc).date())
    df = tasks_df.copy()
    df['due_date'] = pd.to_datetime(df['due_date'])

    blocked = df[df['status'].str.lower() == 'blocked']
    no_owner = df[df['owner'].isna() | (df['owner'] == '')]
    overdue = df[(df['due_date'] < today) & (df['status'] != 'done')]
    long_delay = df[(today - df['due_date']).dt.days > delay_threshold_days]

    return {
        "blocked": blocked,
        "no_owner": no_owner,
        "overdue": overdue,
        "long_delay": long_delay
    }


def extract_health_signals(notes_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect health-related phrases in meeting notes.
    """

    keywords = [
        "stuck", "blocked", "not clear", "unclear",
        "overwhelmed", "burnout", "can't", "cannot", "confused"
    ]

    regex = "|".join([kw.replace(" ", "\\s") for kw in keywords])
    hits = notes_df[
        notes_df["text"].str.contains(regex, case=False, na=False)
    ]

    return hits.to_dict(orient='records')


# DelegationAgent
from typing import Dict, List, Any
import pandas as pd

class DelegationAgent:
    """
    Flags tasks suitable for delegation and drafts helpful delegation messages.

    Heuristics:
      - Low impact (impact ≤ 2) AND high remaining effort (≥ 3h)
      - OR owner overloaded (total remaining workload > capacity_threshold)
    """

    def __init__(self, capacity_threshold: float = 8.0):
        self.capacity_threshold = capacity_threshold

    def owner_loads(self, priority_df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes total remaining effort per owner.
        """
        return (
            priority_df
            .groupby('owner')['remaining_effort']
            .sum()
            .to_dict()
        )

    def suggest_delegations(self, priority_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Produces a list of delegation suggestions with draft messages.
        """
        suggestions = []
        loads = self.owner_loads(priority_df)

        for _, row in priority_df.iterrows():
            owner = row['owner']
            rem = float(row['remaining_effort'])
            impact = float(row.get('impact', 0))
            owner_load = loads.get(owner, 0.0)

            # ---- Delegation heuristics ----
            low_impact_high_effort = (impact <= 2 and rem >= 3)
            overloaded = (owner_load > self.capacity_threshold)

            if low_impact_high_effort or overloaded:
                msg = (
                    f"Hi, I'm delegating task '{row['title']}' (id {int(row['id'])}). "
                    f"Context: estimated ~{rem:.1f}h, current owner workload={owner_load:.1f}h. "
                    f"Thanks for taking this on!"
                )

                suggestions.append({
                    "task_id": int(row['id']),
                    "from_owner": owner,
                    "reason": (
                        "low-impact, high-effort"
                        if low_impact_high_effort
                        else "owner overloaded"
                    ),
                    "owner_load": owner_load,
                    "draft_message": msg
                })

        return suggestions

# instance
delegation_agent = DelegationAgent()


# FocusAgent (time-block scheduling)
from datetime import datetime, timedelta, timezone
import pandas as pd

class FocusAgent:
    """
    Schedules daily time blocks for each owner based on priority and remaining effort.

    Rules:
      - Work window: 9:00–17:00 (8 hours)
      - Default schedulable focus hours: 6h
      - Highest priority tasks scheduled first
      - All timestamps are timezone-aware UTC (ISO8601)
    """

    def __init__(self, work_start=9, work_end=17):
        self.work_start = work_start
        self.work_end = work_end

    def schedule_for_owner(self, owner, tasks, capacity_hours=6):
        blocks = []
        remaining_capacity = capacity_hours

        # ---- timezone-aware UTC start of the workday ----
        current_start = (
            datetime.now(timezone.utc)
            .replace(hour=self.work_start, minute=0, second=0, microsecond=0)
        )

        for t in tasks:
            hours = float(t['remaining_effort'])
            if hours <= 0:
                continue

            # If adding this task exceeds capacity (and we already have blocks), stop.
            if hours > remaining_capacity and blocks:
                break

            block_hours = min(hours, remaining_capacity)
            end_time = current_start + timedelta(hours=block_hours)

            blocks.append({
                "title": t['title'],
                "start": current_start.isoformat(),
                "end": end_time.isoformat(),
                "duration_hours": round(block_hours, 2)
            })

            # Update counters
            remaining_capacity -= block_hours
            current_start = end_time

        return blocks

    def schedule(self, priority_df: pd.DataFrame, per_owner_capacity: Dict[str, float] = None):
        if per_owner_capacity is None:
            per_owner_capacity = {}

        schedule = {}

        for owner, group in priority_df.groupby('owner'):
            owner_cap = per_owner_capacity.get(owner, 6.0)

            # Convert and sort tasks by priority descending
            tasks = group.to_dict(orient='records')
            tasks_sorted = sorted(tasks, key=lambda x: -float(x['priority_0_100']))

            schedule[owner] = self.schedule_for_owner(owner, tasks_sorted, owner_cap)

        return schedule

# instance
focus_agent = FocusAgent()



# Google Calendar integration using a service account (with optional impersonation)
# Fallback: uses the previous in-memory calendar_agent if credentials not provided.

from datetime import datetime, timedelta, timezone
from dateutil import parser
from google.oauth2 import service_account
from googleapiclient.discovery import build

class GCalendarAgent:
    """
    Google Calendar agent using service account credentials.

    - Uses RFC3339 timestamps
    - Uses timezone-aware UTC datetimes
    - Provides fallback simulated calendar when no credentials exist
    """

    def __init__(self, sa_info: dict = None, impersonate_user: str = None):
        self.simulator = None

        if sa_info is None:
            # No credentials → simulator mode
            logger.warning("Google service account not provided — using in-memory simulated calendar.")
            from collections import defaultdict
            self.calendar = defaultdict(list)
            self.simulator = True
            return

        scopes = ["https://www.googleapis.com/auth/calendar"]

        try:
            credentials = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=scopes
            )
            if impersonate_user:
                credentials = credentials.with_subject(impersonate_user)

            self.service = build(
                "calendar", "v3",
                credentials=credentials,
                cache_discovery=False
            )
            self.simulator = False
            logger.info("✅ Google Calendar service initialized (API mode).")

        except Exception as e:
            logger.error("Failed to initialize Google Calendar API: %s", e)
            logger.warning("Falling back to simulated calendar.")
            from collections import defaultdict
            self.calendar = defaultdict(list)
            self.simulator = True

    # ---- Internal helper ----
    def _to_rfc3339(self, dt: datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    # ---- Add event ----
    def add_event(self, owner: str, start: datetime, end: datetime, title: str, calendar_id="primary"):
        if self.simulator:
            ev = {"start": start, "end": end, "title": title}
            self.calendar[owner].append(ev)
            return {"status": "scheduled_simulator", "event": ev}

        event_body = {
            "summary": title,
            "start": {"dateTime": self._to_rfc3339(start)},
            "end": {"dateTime": self._to_rfc3339(end)},
        }

        try:
            created = self.service.events().insert(
                calendarId=calendar_id,
                body=event_body
            ).execute()
            return {"status": "scheduled", "event": created}

        except Exception as e:
            logger.error("Google Calendar insert failed: %s", e)
            return {"status": "error", "error": str(e)}

    # ---- Find conflicts ----
    def find_conflicts(self, owner: str, start: datetime, end: datetime, calendar_id="primary"):
        if self.simulator:
            conflicts = []
            for ev in self.calendar.get(owner, []):
                ev_start = ev["start"]
                ev_end = ev["end"]
                if not (end <= ev_start or start >= ev_end):
                    conflicts.append(ev)
            return conflicts

        # Query Google Calendar
        time_min = self._to_rfc3339(start - timedelta(seconds=1))
        time_max = self._to_rfc3339(end + timedelta(seconds=1))

        try:
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime"
            ).execute()

            items = events_result.get("items", [])
            conflicts = []

            for it in items:
                s = it.get("start", {}).get("dateTime") or it.get("start", {}).get("date")
                e = it.get("end", {}).get("dateTime") or it.get("end", {}).get("date")

                s_dt = parser.isoparse(s)
                e_dt = parser.isoparse(e)

                if not (end <= s_dt or start >= e_dt):
                    conflicts.append(it)

            return conflicts

        except Exception as e:
            logger.error("Google Calendar list failed: %s", e)
            return {"status": "error", "error": str(e)}

    # ---- Schedule block ----
    def schedule_block_with_fallback(self, owner: str, start: datetime, end: datetime, title: str, calendar_id="primary"):

        conflicts = self.find_conflicts(owner, start, end, calendar_id)

        if conflicts:
            # Try sliding block forward in 30-min increments
            attempt_start = start
            attempt_end = end

            work_end = start.replace(
                hour=17, minute=0, second=0, microsecond=0
            )

            while attempt_end <= work_end + timedelta(seconds=1):
                attempt_start += timedelta(minutes=30)
                attempt_end += timedelta(minutes=30)

                if not self.find_conflicts(owner, attempt_start, attempt_end, calendar_id):
                    return self.add_event(owner, attempt_start, attempt_end, title, calendar_id)

            return {"status": "conflict", "conflicts": conflicts}

        return self.add_event(owner, start, end, title, calendar_id)


# Instantiate Calendar Agent
if USE_GOOGLE_CALENDAR:
    calendar_agent = GCalendarAgent(
        sa_info=GOOGLE_SERVICE_ACCOUNT_INFO,
        impersonate_user=CALENDAR_IMPERSONATED_USER
    )
else:
    # Simple fallback in-memory version
    from collections import defaultdict

    class InMemoryCalendarAgent:
        def __init__(self):
            self.calendar = defaultdict(list)

        def add_event(self, owner, start, end, title, calendar_id="primary"):
            ev = {"start": start, "end": end, "title": title}
            self.calendar[owner].append(ev)
            return {"status": "scheduled_simulator", "event": ev}

        def find_conflicts(self, owner, start, end, calendar_id="primary"):
            conflicts = []
            for ev in self.calendar.get(owner, []):
                if not (end <= ev["start"] or start >= ev["end"]):
                    conflicts.append(ev)
            return conflicts

        def schedule_block_with_fallback(self, owner, start, end, title, calendar_id="primary"):
            if self.find_conflicts(owner, start, end):
                return {"status": "conflict", "conflicts": self.find_conflicts(owner, start, end)}
            return self.add_event(owner, start, end, title)

    calendar_agent = InMemoryCalendarAgent()
    logger.warning("No Google Calendar credentials: using in-memory calendar agent (simulator).")


# TrendAnalyzerAgent (notes pattern detection)
class TrendAnalyzerAgent:
    """
    Analyzes notes over time to detect recurring problems per owner:
     - repeated 'blocked' occurrences
     - repeated mentions of 'overwhelmed' or 'burnout'
     - repeated clarity/requirement issues
    Produces suggestions like 'reduce WIP', 'pair programming', 'timeboxing'.
    """
    def __init__(self, lookback_days=30):
        self.lookback = lookback_days

    def analyze(self, notes_df: pd.DataFrame):
        df = notes_df.copy()

        # Normalize to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Correct use of UTC with your global datetime import
        cutoff = (datetime.utcnow().date() - timedelta(days=self.lookback))

        # Filter to the lookback window
        df = df[df["date"].dt.date >= cutoff]

        trends = {}

        # Keyword groups
        keywords = {
            "blocked": ["blocked", "stuck"],
            "overwhelmed": ["overwhelmed", "burnout"],
            "unclear": ["unclear", "not clear", "confusion", "confused"]
        }

        # Count occurrences per author
        for author in df["author"].unique():
            sub = df[df["author"] == author]
            counts = {}

            for group, kws in keywords.items():
                pattern = "|".join([kw for kw in kws])
                counts[group] = sub["text"].str.contains(
                    pattern, case=False, na=False
                ).sum()

            trends[author] = counts

        # Build heuristic suggestions
        suggestions = []
        for author, counts in trends.items():

            if counts.get("blocked", 0) >= 2:
                suggestions.append({
                    "author": author,
                    "suggestion": (
                        "Recurring blocked issues — consider assigning a mentor, "
                        "adding daily unblock time, or improving specs."
                    )
                })

            if counts.get("overwhelmed", 0) >= 1:
                suggestions.append({
                    "author": author,
                    "suggestion": (
                        "Signs of overwhelm — reduce WIP, redistribute tasks, "
                        "or use timeboxing techniques."
                    )
                })

            if counts.get("unclear", 0) >= 1:
                suggestions.append({
                    "author": author,
                    "suggestion": (
                        "Unclear requirement patterns — add early clarification, "
                        "involve a product owner, or write acceptance criteria."
                    )
                })

        return {
            "trends": trends,
            "suggestions": suggestions
        }


# instance
trend_agent = TrendAnalyzerAgent()


# Slack notifier (robust, with file fallback)
from typing import Dict, Any, List, Optional
import requests
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging

logger = logging.getLogger("SlackNotifier")

class SlackNotifier:
    """
    Simple Slack notifier with a file fallback.

    - Tries to POST to Slack chat.postMessage.
    - If Slack request fails (network, missing scopes, etc.), writes a fallback .txt
      into `outputs/` and returns the fallback path.
    - The craft_briefing(...) method produces a plain-text briefing.
    """

    def __init__(self, bot_token: Optional[str], default_channel: str = "#general"):
        self.bot_token = bot_token or ""
        self.default_channel = default_channel
        self.api_url = "https://slack.com/api/chat.postMessage"
        # Ensure outputs folder exists (Kaggle: /kaggle/working/outputs)
        Path("outputs").mkdir(parents=True, exist_ok=True)

    def _now_iso(self) -> str:
        """UTC ISO timestamp for filenames / logging."""
        return datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def send_message(self, channel: str, text: str) -> Dict[str, Any]:
        """
        Send a plain text message to Slack channel.
        Returns dict with status and metadata.
        """
        # Basic token check
        if not self.bot_token:
            logger.warning("No Slack bot token provided — using fallback file write.")
            return self._fallback_save(channel, text)

        headers = {"Authorization": f"Bearer {self.bot_token}", "Content-Type": "application/json; charset=utf-8"}
        payload = {"channel": channel, "text": text}

        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            resp_json = resp.json()
            if resp.status_code == 200 and resp_json.get("ok"):
                logger.info("Slack message sent to %s", channel)
                return {"status": "sent", "channel": channel, "resp": resp_json}
            else:
                # Slack API returned an error (e.g., missing_scope)
                logger.error("Slack API error: %s", resp_json)
                return {"status": "error", "error": resp_json, "fallback": self._fallback_save(channel, text)}
        except Exception as e:
            logger.exception("Slack send failed, saving fallback: %s", e)
            return {"status": "exception", "error": str(e), "fallback": self._fallback_save(channel, text)}

    def _fallback_save(self, channel: str, text: str) -> Dict[str, Any]:
        """Save briefing to outputs/ and return path info"""
        safe_channel = channel.strip("#@").replace("/", "_")
        filename = f"briefing_fallback_{safe_channel}_{self._now_iso()}.txt"
        path = Path("outputs") / filename
        try:
            path.write_text(text, encoding="utf-8")
            logger.info("Wrote fallback briefing to %s", str(path))
            return {"status": "fallback_saved", "path": str(path)}
        except Exception as e:
            logger.exception("Failed to write fallback briefing: %s", e)
            return {"status": "fallback_failed", "error": str(e)}

    def craft_briefing(
        self,
        owner: str,
        coach_plan: Optional[Dict[str, Any]],
        schedule: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Return a plain-text briefing for an owner (safe to send / save)."""
        lines: List[str] = []
        today_str = datetime.utcnow().date().isoformat()
        lines.append(f"Good morning {owner}, AI Workflow Coach briefing for {today_str}:")
        lines.append("")

        # Top tasks
        if not coach_plan or not coach_plan.get("top_tasks"):
            lines.append("No tasks assigned for today.")
        else:
            lines.append("Top tasks:")
            for t in coach_plan.get("top_tasks", []):
                pid = t.get("id", "N/A")
                title = t.get("title", "<untitled>")
                pr = t.get("priority", 0.0)
                rem = t.get("remaining_effort", 0.0)
                lines.append(f" - {title} (id {pid}) — priority {pr:.0f}, remaining {rem}h")

        # Defer suggestions
        if coach_plan and coach_plan.get("defer"):
            lines.append("")
            lines.append("Tasks to consider deferring or de-scoping:")
            for d in coach_plan.get("defer", []):
                lines.append(f" - {d.get('title', 'N/A')} (id {d.get('id', 'N/A')}) — priority {d.get('priority_0_100', 0):.0f}")

        # Schedule
        if schedule:
            lines.append("")
            lines.append("Scheduled focus blocks:")
            for b in schedule:
                # tolerant access to duration field
                duration = b.get("duration_hours") or b.get("duration") or "--"
                lines.append(f" - {b.get('start')} → {b.get('end')} : {b.get('title')} ({duration}h)")

        # Footer
        lines.append("")
        lines.append("If anything looks incorrect, reply with 'adjust' and I will recompute the plan.")
        return "\n".join(lines)


# Instantiate (use your environment variables or constants)
# e.g. SLACK_BOT_TOKEN and SLACK_DEFAULT_CHANNEL must be defined earlier in the notebook
slack_notifier = SlackNotifier(bot_token=globals().get("SLACK_BOT_TOKEN", ""), default_channel=globals().get("SLACK_DEFAULT_CHANNEL", "#general"))


# Enhanced orchestrator wiring all agents
from datetime import datetime, timezone, timedelta

class EnhancedFallbackOrchestrator:
    def __init__(self):
        self.memory = {}
        self.sessions = {}
        self.delegation_agent = DelegationAgent()
        self.focus_agent = focus_agent
        self.calendar_agent = calendar_agent
        self.trend_agent = trend_agent
        self.notifier = slack_notifier

    def create_session(self, session_id: str):
        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            "events": []
        }
        logger.info(f"Session created: {session_id}")

    def append_event(self, session_id: str, ev: Dict[str, Any]):
        self.sessions[session_id]["events"].append({
            "ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            "event": ev
        })

    async def daily_checkin(self, tasks_df: pd.DataFrame, notes_df: pd.DataFrame, session_id: str = "demo_daily"):
        # Start session
        self.create_session(session_id)
        self.append_event(session_id, {"type": "start"})

        # 1. Bottlenecks + health
        bottlenecks = detect_bottlenecks(tasks_df)
        health_signals = extract_health_signals(notes_df)

        # 2. Priorities
        priority_df = compute_priority_scores(tasks_df)

        # 3. Trends
        trend_report = self.trend_agent.analyze(notes_df)

        # 4. Delegations
        delegations = self.delegation_agent.suggest_delegations(priority_df)

        # 5. Focus blocks
        schedule = self.focus_agent.schedule(priority_df)

        # 6. Calendar sync
        calendar_results = {}
        for owner, blocks in schedule.items():
            calendar_results[owner] = []
            for b in blocks:

                # All timestamps created earlier are ISO8601 with UTC
                start = datetime.fromisoformat(b["start"])
                end = datetime.fromisoformat(b["end"])

                result = self.calendar_agent.schedule_block_with_fallback(
                    owner,
                    start,
                    end,
                    b["title"]
                )

                calendar_results[owner].append({
                    "block": b,
                    "calendar_result": result
                })

        # 7. Coach plan (top N tasks per owner)
        coach_plan = {}
        for owner, group in priority_df.groupby("owner"):
            assigned = []
            hours_used = 0.0

            for _, row in group.iterrows():
                remaining = float(row["remaining_effort"])

                if hours_used + remaining <= 6 or len(assigned) == 0:
                    assigned.append({
                        "id": int(row["id"]),
                        "title": row["title"],
                        "priority": float(row["priority_0_100"]),
                        "remaining_effort": remaining
                    })
                    hours_used += remaining

                if len(assigned) >= 3:
                    break

            deferred = priority_df[
                (priority_df["owner"] == owner) &
                (~priority_df["id"].isin([t["id"] for t in assigned]))
            ].head(3)[["id", "title", "priority_0_100"]].to_dict(orient="records")

            coach_plan[owner] = {
                "top_tasks": assigned,
                "defer": deferred,
                "hours_today": hours_used
            }

        # 8. Slack briefings
        briefing_results = {}
        for owner, plan in coach_plan.items():
            schedule_for_owner = schedule.get(owner, [])
            briefing_text = self.notifier.craft_briefing(owner, plan, schedule_for_owner)

            res = self.notifier.send_message(
                channel=self.notifier.default_channel,
                text=briefing_text
            )
            briefing_results[owner] = res

        # 9. Persist + return
        result = {
            "session_id": session_id,
            "coach_plan": coach_plan,
            "priority_df": priority_df,
            "bottlenecks": {k: df.to_dict(orient='records') for k, df in bottlenecks.items()},
            "health_signals": health_signals,
            "trend_report": trend_report,
            "delegations": delegations,
            "schedule": schedule,
            "calendar_results": calendar_results,
            "briefings": briefing_results
        }

        safe_save_json(
            result["coach_plan"],
            os.path.join("outputs", f"{session_id}_coach_plan.json")
        )

        self.append_event(session_id, {
            "type": "complete",
            "summary": {"top_counts": {k: len(v) for k, v in coach_plan.items()}}
        })

        self.memory[f"session:{session_id}:result"] = result

        return result


# instantiate orchestrator
enhanced_orch = EnhancedFallbackOrchestrator()


# --- Async-safe execution of daily_checkin ---
import asyncio

async def _run_test():
    tasks_df, notes_df = load_data()
    res = await enhanced_orch.daily_checkin(
        tasks_df,
        notes_df,
        session_id="demo_daily"
    )
    print("Daily check-in complete.")
    print("Coach plan saved at: outputs/demo_daily_coach_plan.json")
    return res

try:
    # Kaggle notebooks usually already have an event loop running
    loop = asyncio.get_running_loop()
    task = loop.create_task(_run_test())
    result = await task
except RuntimeError:
    # Fallback for environments without a running loop
    result = asyncio.run(_run_test())


import os

print("Files in outputs:")
if os.path.exists("outputs"):
    print(os.listdir("outputs"))
else:
    print("No outputs directory found.")


# Run orchestrator (safe in Jupyter)
async def _run_demo():
    res = await enhanced_orch.daily_checkin(tasks_df, notes_df, session_id="demo_daily_01")
    print("=== Coach Plan Overview ===")
    import pprint
    pprint.pprint({k: {"top_tasks": v["top_tasks"], "hours_today": v["hours_today"]} for k,v in res['coach_plan'].items() if not k.startswith("_")})
    print("\n=== Delegation Suggestions ===")
    pprint.pprint(res['delegations'])
    print("\n=== Trend Suggestions ===")
    pprint.pprint(res['trend_report'])
    print("\n=== Sample calendar result for 'alice' ===")
    pprint.pprint(res['calendar_results'].get('alice', []))
    print("\n=== Briefings saved to outputs/ ===")
    pprint.pprint(res['briefings'])
    return res

# run in a way that works in both script and notebook
res_task = run_async_safe(_run_demo())
# If it returned a Task because loop is running, keep the variable; else res_task is result



# Inspect outputs produced
print("Files in outputs:")
for fname in sorted(os.listdir("outputs")):
    print(" -", fname)

# Show saved coach plan file
coach_path = os.path.join("outputs", "demo_daily_01_coach_plan.json")
if os.path.exists(coach_path):
    print("\nCoach plan preview:")
    with open(coach_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # print one owner sample
    some_owner = next(iter(data.keys()))
    print(some_owner, "->", data[some_owner])
else:
    print("No coach plan saved yet.")

