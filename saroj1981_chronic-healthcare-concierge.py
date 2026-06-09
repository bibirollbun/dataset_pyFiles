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


!pip install google-adk


import asyncio
from google.adk import Agent, Runner
from google.adk.agents import SequentialAgent
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
# Import ClientError with fallback for notebook environments
try:
    from google.genai.errors import ClientError
except ImportError:
    # Fallback: try alternative import path
    try:
        from google.genai import ClientError
    except ImportError:
        # If still not available, create a base exception class
        class ClientError(Exception):
            def __init__(self, status_code, error_dict, response):
                self.status_code = status_code
                self.error = error_dict
                self.response = response
                super().__init__(f"ClientError {status_code}: {error_dict}")

# Import ADK's evaluation framework
from google.adk.evaluation import AgentEvaluator
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_metrics import BaseCriterion
from google.adk.evaluation.eval_metrics import PrebuiltMetrics

import pandas as pd
import plotly.express as px
import time
import re
from google.api_core import retry
from google.api_core.exceptions import ResourceExhausted

import os
import sqlite3
import json
from typing import Optional, Dict, List, Any, Union
from datetime import datetime


try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Setup and authentication complete (Kaggle environment).")
except ImportError:
    # Local environment: API key should be set via environment variable
    # Set GOOGLE_API_KEY in your environment or .env file
    if "GOOGLE_API_KEY" not in os.environ:
        print("Warning: GOOGLE_API_KEY not found. Please set it as an environment variable.")
    else:
        print("Setup and authentication complete (local environment).")
except Exception as e:
    print(
        f"Authentication Error: Please make sure you have set 'GOOGLE_API_KEY' environment variable. Details: {e}"
    )


WELLNESS_CSV_PATH = "/kaggle/input/oura-data/oura_data.csv"
wellness_df = pd.read_csv(WELLNESS_CSV_PATH)


print("Columns in wellness_df:")
print(wellness_df.columns.tolist())
print("\nFirst few rows:")
print(wellness_df.head())


# Database file path
DB_PATH = os.getenv("DB_PATH", "healthcare_concierge.db")


# ------------------------------------------------------------------
# Database Functions - SQLite Integration
# ------------------------------------------------------------------

def reset_database():
    """
    Nuclear option: Delete the database file and all related files.
    Use this if unlock_database() doesn't work and you're okay losing data.
    """
    import os
    import gc
    
    # Force garbage collection
    gc.collect()
    
    files_to_delete = [
        DB_PATH,
        DB_PATH + "-wal",
        DB_PATH + "-shm",
        DB_PATH + "-journal"
    ]
    
    deleted = []
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted.append(file_path)
        except Exception as e:
            print(f" Could not delete {file_path}: {e}")
    
    if deleted:
        print(f" Deleted database files: {', '.join(deleted)}")
        time.sleep(0.5)  # Wait for file system to sync
        return True
    else:
        print(" No database files found to delete")
        return False


def unlock_database():
    """
    Forcefully unlock the database by closing all connections.
    Use this if you get 'database is locked' errors.
    """
    import os
    import gc
    
    # Force garbage collection to close any lingering connections
    gc.collect()
    
    # Try multiple approaches to unlock
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Approach 1: Try to connect with a short timeout and close immediately
            conn = sqlite3.connect(DB_PATH, timeout=0.5)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except:
                pass  # Ignore errors during checkpoint
            conn.close()
            
            # Approach 2: Try to delete WAL and SHM files if they exist
            wal_file = DB_PATH + "-wal"
            shm_file = DB_PATH + "-shm"
            try:
                if os.path.exists(wal_file):
                    os.remove(wal_file)
            except:
                pass
            try:
                if os.path.exists(shm_file):
                    os.remove(shm_file)
            except:
                pass
            
            # Wait a bit longer for locks to clear
            time.sleep(0.5 * (attempt + 1))
            
            # Verify we can connect now
            test_conn = sqlite3.connect(DB_PATH, timeout=2.0)
            test_conn.close()
            
            print("Database unlocked successfully")
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(1.0)  # Wait longer between attempts
                continue
            else:
                print(f"  Could not unlock database after {max_attempts} attempts: {e}")
                print("\n Solutions:")
                print("   1. Restart your notebook kernel (Kernel → Restart Kernel)")
                print("   2. Wait 10-15 seconds and try again")
                print("   3. If in Kaggle, try 'Restart Session' from the menu")
                print("   4. As a last resort, delete the database file and recreate it:")
                print(f"      import os; os.remove('{DB_PATH}')")
                return False


def init_database():
    """Initialize SQLite database with required tables."""
    # Get absolute path for clarity (especially in Kaggle notebooks)
    abs_db_path = os.path.abspath(DB_PATH)
    conn = None
    
    # Try to unlock first, but don't fail if it doesn't work
    unlock_database()
    
    # Give it a moment after unlock attempt
    time.sleep(0.5)
    
    max_retries = 3
    for retry in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            break  # Success, exit retry loop
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower() and retry < max_retries - 1:
                print(f"Database locked, retrying ({retry + 1}/{max_retries})...")
                unlock_database()
                time.sleep(1.0 * (retry + 1))  # Exponential backoff
                continue
            else:
                raise
    
    # Now create tables (connection is already established)
    try:
        # User profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                height_cm REAL,
                weight_kg REAL,
                conditions TEXT,  -- JSON array
                medications TEXT,  -- JSON array
                allergies TEXT,  -- JSON array
                goals TEXT,  -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Supplements catalog table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS supplements (
                supplement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                category TEXT,  -- e.g., 'Sleep', 'Energy', 'Heart Health'
                info TEXT,  -- Description
                evidence_level TEXT,  -- e.g., 'Strong evidence', 'Moderate evidence'
                pubmed_link TEXT,
                webmd_link TEXT,
                amazon_search_link TEXT,
                medication_interactions TEXT,  -- JSON array of medications that may interact
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Session history table (for evaluation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_history (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                user_prompt TEXT,
                final_summary TEXT,
                risk_score INTEGER,
                care_plan TEXT,  -- JSON array
                shopping_suggestions TEXT,  -- JSON array
                execution_time_seconds REAL,
                agent_sequence TEXT,  -- JSON array of agent names in order
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
        """)
        
        # Agent evaluation metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_evaluations (
                evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_name TEXT,
                metric_name TEXT,  -- e.g., 'accuracy', 'response_time', 'tool_usage'
                metric_value REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES session_history(session_id)
            )
        """)
        
        conn.commit()
        conn.close()
        conn = None
        abs_db_path = os.path.abspath(DB_PATH)
        print(f"Database initialized: {abs_db_path}")
    except sqlite3.OperationalError as e:
        if conn:
            try:
                conn.close()
            except:
                pass
        if "database is locked" in str(e).lower():
            print("\n Database is locked and could not be unlocked automatically.")
            print("\n Please try one of these solutions (in order):")
            print("   1. Restart your notebook kernel:")
            print("      - Kaggle: Click 'Restart Session' button")
            print("      - Jupyter: Kernel → Restart Kernel")
            print("   2. Wait 10-15 seconds and try again")
            print("   3. Reset the database (deletes all data, then recreates):")
            print(f"      reset_database()")
            print(f"      init_database()")
            print(f"      seed_database()")
            raise Exception("Database is locked. Please restart kernel, wait, or use reset_database().")
        else:
            raise
    except Exception as e:
        if conn:
            try:
                conn.close()
            except:
                pass
        raise

def seed_database():
    """Seed database with sample data for demo."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM user_profiles")
        if cursor.fetchone()[0] > 0:
            print("Database already seeded. Skipping...")
            conn.close()
            conn = None
            return
        
        # Insert sample user profile
        cursor.execute("""
            INSERT INTO user_profiles (user_id, name, age, gender, height_cm, weight_kg, conditions, medications, allergies, goals)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "demo-user",
            "Alex Thompson",
            45,
            "Male",
            178,
            85,
            json.dumps(["Hypertension"]),
            json.dumps(["Lisinopril"]),
            json.dumps(["Penicillin"]),
            json.dumps(["Increase energy", "Improve sleep"])
        ))
        
        # Insert supplements catalog
        supplements = [
            {
                "name": "Melatonin Gummies (3mg)",
                "category": "Sleep hygiene",
                "info": "Melatonin is a naturally occurring hormone that regulates sleep-wake cycles",
                "evidence_level": "Strong evidence: Multiple RCTs support melatonin for sleep onset (NIH/NCCIH reviewed)",
                "pubmed_link": "https://pubmed.ncbi.nlm.nih.gov/?term=melatonin+sleep+efficacy+clinical+trial",
                "webmd_link": "https://www.webmd.com/sleep-disorders/sleep-disorders-melatonin",
                "amazon_search_link": "https://www.amazon.com/s?k=melatonin+gummies+3mg",
                "medication_interactions": json.dumps(["Blood thinners", "Immunosuppressants", "Diabetes medications"])
            },
            {
                "name": "Electrolyte Supplement Powder",
                "category": "Hydration",
                "info": "Electrolytes (sodium, potassium, magnesium) support hydration and fluid balance",
                "evidence_level": "Well-established: Electrolytes are essential nutrients (FDA recognized)",
                "pubmed_link": "https://pubmed.ncbi.nlm.nih.gov/?term=electrolyte+supplementation+hydration",
                "webmd_link": "https://www.webmd.com/diet/electrolyte-water",
                "amazon_search_link": "https://www.amazon.com/s?k=electrolyte+supplement+powder",
                "medication_interactions": json.dumps([])
            },
            {
                "name": "Omega-3 Fish Oil (1000mg EPA+DHA)",
                "category": "Heart Health",
                "info": "Omega-3 fatty acids EPA and DHA support cardiovascular health",
                "evidence_level": "Strong evidence: Multiple large studies show cardiovascular benefits (AHA recommends, FDA approved claims)",
                "pubmed_link": "https://pubmed.ncbi.nlm.nih.gov/?term=omega+3+fish+oil+cardiovascular+disease+prevention",
                "webmd_link": "https://www.webmd.com/diet/guide/the-truth-about-omega-3",
                "amazon_search_link": "https://www.amazon.com/s?k=omega+3+fish+oil+1000mg+epa+dha",
                "medication_interactions": json.dumps(["Blood thinners"])
            },
            {
                "name": "B-Complex Vitamins with Vitamin D3",
                "category": "Energy Support",
                "info": "B vitamins support energy metabolism; D3 supports immune function and bone health",
                "evidence_level": "Moderate evidence: B vitamins for energy in deficiency states; D3 for immune support (NIH Office of Dietary Supplements)",
                "pubmed_link": "https://pubmed.ncbi.nlm.nih.gov/?term=B+vitamins+energy+metabolism+vitamin+D3+immune",
                "webmd_link": "https://www.webmd.com/diet/supplement-guide-vitamin-b-complex",
                "amazon_search_link": "https://www.amazon.com/s?k=B+complex+vitamins+vitamin+D3",
                "medication_interactions": json.dumps([])
            },
            {
                "name": "Magnesium Glycinate (400mg)",
                "category": "Sleep Quality",
                "info": "Magnesium supports muscle relaxation and may improve sleep quality",
                "evidence_level": "Moderate evidence: Some RCTs show sleep benefits; strong evidence for muscle relaxation (NIH ODS)",
                "pubmed_link": "https://pubmed.ncbi.nlm.nih.gov/?term=magnesium+glycinate+sleep+efficacy",
                "webmd_link": "https://www.webmd.com/diet/supplement-guide-magnesium",
                "amazon_search_link": "https://www.amazon.com/s?k=magnesium+glycinate+400mg",
                "medication_interactions": json.dumps([])
            }
        ]
        
        for supp in supplements:
            cursor.execute("""
                INSERT INTO supplements (name, category, info, evidence_level, pubmed_link, webmd_link, amazon_search_link, medication_interactions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                supp["name"],
                supp["category"],
                supp["info"],
                supp["evidence_level"],
                supp["pubmed_link"],
                supp["webmd_link"],
                supp["amazon_search_link"],
                supp["medication_interactions"]
            ))
        
        conn.commit()
        conn.close()
        conn = None
        print("Database seeded with sample data")
    except sqlite3.OperationalError as e:
        if conn:
            try:
                conn.close()
            except:
                pass
        if "database is locked" in str(e).lower():
            print(" Database locked during seeding. Run unlock_database() and try again.")
        raise
    except Exception as e:
        if conn:
            try:
                conn.close()
            except:
                pass
        raise

def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve user profile from database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        conn = None
    except sqlite3.OperationalError as e:
        if conn:
            try:
                conn.close()
            except:
                pass
        raise
    
    if row:
        return {
            "user_id": row["user_id"],
            "name": row["name"],
            "age": row["age"],
            "gender": row["gender"],
            "height_cm": row["height_cm"],
            "weight_kg": row["weight_kg"],
            "conditions": json.loads(row["conditions"]) if row["conditions"] else [],
            "medications": json.loads(row["medications"]) if row["medications"] else [],
            "allergies": json.loads(row["allergies"]) if row["allergies"] else [],
            "goals": json.loads(row["goals"]) if row["goals"] else [],
        }
    return None

def get_supplements_by_category(category: str) -> List[Dict[str, Any]]:
    """Retrieve supplements by category."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM supplements WHERE category = ?", (category,))
        rows = cursor.fetchall()
        conn.close()
        conn = None
    except sqlite3.OperationalError as e:
        if conn:
            try:
                conn.close()
            except:
                pass
        raise
    
    return [
        {
            "name": row["name"],
            "category": row["category"],
            "info": row["info"],
            "evidence_level": row["evidence_level"],
            "pubmed_link": row["pubmed_link"],
            "webmd_link": row["webmd_link"],
            "amazon_search_link": row["amazon_search_link"],
            "medication_interactions": json.loads(row["medication_interactions"]) if row["medication_interactions"] else [],
        }
        for row in rows
    ]

def get_supplements_by_name(name_pattern: str) -> List[Dict[str, Any]]:
    """Search supplements by name pattern."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM supplements WHERE name LIKE ?", (f"%{name_pattern}%",))
        rows = cursor.fetchall()
        conn.close()
        conn = None
    except sqlite3.OperationalError as e:
        if conn:
            try:
                conn.close()
            except:
                pass
        raise
    
    return [
        {
            "name": row["name"],
            "category": row["category"],
            "info": row["info"],
            "evidence_level": row["evidence_level"],
            "pubmed_link": row["pubmed_link"],
            "webmd_link": row["webmd_link"],
            "amazon_search_link": row["amazon_search_link"],
            "medication_interactions": json.loads(row["medication_interactions"]) if row["medication_interactions"] else [],
        }
        for row in rows
    ]

def save_session_history(session_id: str, user_id: str, user_prompt: str, 
                        final_summary: str, risk_score: int, care_plan: List[str],
                        shopping_suggestions: List[str], execution_time: float,
                        agent_sequence: List[str], max_retries: int = 5):
    """Save session history for evaluation."""
    conn = None
    for attempt in range(max_retries):
        try:
            # Small delay before attempting connection (helps clear any lingering locks)
            if attempt > 0:
                time.sleep(0.5 * attempt)  # Longer delay for retries
            
            # Use longer timeout and enable WAL mode for better concurrent access
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            # Use INSERT OR REPLACE to handle duplicate session_ids
            cursor.execute("""
                INSERT OR REPLACE INTO session_history 
                (session_id, user_id, user_prompt, final_summary, risk_score, care_plan, 
                 shopping_suggestions, execution_time_seconds, agent_sequence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                user_id,
                user_prompt,
                final_summary,
                risk_score,
                json.dumps(care_plan),
                json.dumps(shopping_suggestions),
                execution_time,
                json.dumps(agent_sequence)
            ))
            
            conn.commit()
            conn.close()
            conn = None  # Mark as closed
            return  # Success, exit retry loop
        except sqlite3.OperationalError as e:
            if conn:
                try:
                    conn.close()
                except:
                    pass
                conn = None
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = 0.5 * (2 ** attempt) + (time.time() % 1) * 0.1
                time.sleep(wait_time)
                continue
            else:
                raise
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except:
                    pass
            raise

def save_evaluation_metrics_batch(metrics: List[Dict[str, Any]], max_retries: int = 5):
    """
    Save multiple evaluation metrics in a single transaction.
    Reduces database operations and lock contention.
    
    Args:
        metrics: List of dicts with keys: session_id, agent_name, metric_name, metric_value, notes
        max_retries: Maximum number of retry attempts
    """
    if not metrics:
        return
    
    conn = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(0.5 * attempt)
            
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            # Insert all metrics in a single transaction
            cursor.executemany("""
                INSERT INTO agent_evaluations (session_id, agent_name, metric_name, metric_value, notes)
                VALUES (?, ?, ?, ?, ?)
            """, [
                (m['session_id'], m['agent_name'], m['metric_name'], m['metric_value'], m.get('notes', ''))
                for m in metrics
            ])
            
            conn.commit()
            conn.close()
            conn = None
            return
        except sqlite3.OperationalError as e:
            if conn:
                try:
                    conn.close()
                except:
                    pass
                conn = None
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 0.5 * (2 ** attempt) + (time.time() % 1) * 0.1
                time.sleep(wait_time)
                continue
            else:
                raise
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except:
                    pass
            raise


def save_evaluation_metric(session_id: str, agent_name: str, metric_name: str,
                           metric_value: float, notes: str = "", max_retries: int = 5):
    """Save agent evaluation metric."""
    conn = None
    for attempt in range(max_retries):
        try:
            # Small delay before attempting connection (helps clear any lingering locks)
            if attempt > 0:
                time.sleep(0.5 * attempt)  # Longer delay for retries
            
            # Use longer timeout and enable WAL mode for better concurrent access
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO agent_evaluations (session_id, agent_name, metric_name, metric_value, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, agent_name, metric_name, metric_value, notes))
            
            conn.commit()
            conn.close()
            conn = None  # Mark as closed
            return  # Success, exit retry loop
        except sqlite3.OperationalError as e:
            if conn:
                try:
                    conn.close()
                except:
                    pass
                conn = None
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = 0.5 * (2 ** attempt) + (time.time() % 1) * 0.1
                time.sleep(wait_time)
                continue
            else:
                raise
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except:
                    pass
            raise



# Initialize database
init_database()
seed_database()


def check_dependencies():
    """
    Check if all required functions and variables are defined.
    Useful for notebook environments where code might be split across cells.
    
    Returns:
        dict: Status of each dependency
    """
    dependencies = {
        'get_user_profile': 'get_user_profile' in globals(),
        'wellness_df': 'wellness_df' in globals(),
        'init_database': 'init_database' in globals(),
        'seed_database': 'seed_database' in globals(),
    }
    
    missing = [name for name, exists in dependencies.items() if not exists]
    
    if missing:
        print("  Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\n Make sure you've run all cells that define these functions/variables.")
        return False
    
    print(" All dependencies are available.")
    return True


# Optional: just to check if the data exists
import sqlite3
import pandas as pd

conn = sqlite3.connect("healthcare_concierge.db")

# Quick check
print("Users:", pd.read_sql_query("SELECT COUNT(*) FROM user_profiles", conn).iloc[0,0])
print("Supplements:", pd.read_sql_query("SELECT COUNT(*) FROM supplements", conn).iloc[0,0])

# View all user profiles
print("\nAll Users:")
print(pd.read_sql_query("SELECT * FROM user_profiles", conn))

# View all supplements
print("\nAll Supplements:")
print(pd.read_sql_query("SELECT name, category FROM supplements", conn))

conn.close()


def log_intake_answer(question: str, answer: str, tool_context: ToolContext) -> str:
    """
    Log user intake responses (Q&A pairs) to session state.
    
    This tool is used by the intake agent to store user responses during
    the medical history collection phase. All Q&A pairs are persisted in
    session state for later analysis by the risk assessment agent.
    
    Args:
        question: The question asked to the user
        answer: The user's response to the question
        tool_context: ADK tool context containing session state
        
    Returns:
        str: Confirmation message "logged"
    """
    answers = tool_context.state.setdefault("intake_answers", [])
    answers.append({"question": question, "answer": answer})
    return "logged"

def trend_analyzer(metric: str, window: int, tool_context: ToolContext) -> dict:
    """
    Analyze trends for Oura wearable metrics using rolling averages.
    
    This tool computes rolling averages over a specified time window to identify
    trends in health metrics like heart rate, sleep quality, and readiness scores.
    The results are stored in session state for trend visualization and risk assessment.
    
    Args:
        metric: Name of the metric to analyze (supports aliases like "hr" for "resting_hr")
        window: Number of days to use for rolling average calculation
        tool_context: ADK tool context containing session state
        
    Returns:
        dict: Contains the metric name, rolling mean value, and window size
    """
    # Map user-friendly metric names to actual CSV column names
    metric_mapping = {
        "heart_rate": "resting_hr",
        "hr": "resting_hr",
        "resting_hr": "resting_hr",
        "sleep_score": "sleep_score",
        "readiness": "readiness_score",
        "readiness_score": "readiness_score",
        "hr_average": "hr_average",
        "sleep_efficiency": "sleep_efficiency",
    }
    
    actual_metric = metric_mapping.get(metric.lower(), metric)
    
    # Validate metric exists in dataset
    if actual_metric not in wellness_df.columns:
        return {"error": f"Metric '{metric}' not found. Available: {list(wellness_df.columns)}"}
    
    # Calculate rolling average (smooths out daily fluctuations)
    series = wellness_df[actual_metric].rolling(window=window, min_periods=1).mean()
    latest_value = series.iloc[-1]
    
    # Store results in session state for other agents
    tool_context.state.setdefault("trend_results", {})[actual_metric] = latest_value
    
    return {
        "metric": actual_metric,
        "rolling_mean": float(latest_value) if pd.notna(latest_value) else None,
        "window": window
    }

def wearable_ingest(tool_context: ToolContext) -> str:
    """
    Ingest the latest wearable data from Oura ring dataset.
    
    This tool extracts the most recent health metrics from the Oura CSV dataset
    and stores them in session state. The wearable agent uses this to provide
    objective health data alongside subjective user-reported symptoms.
    
    Args:
        tool_context: ADK tool context containing session state
        
    Returns:
        str: Summary string with key health metrics (heart rate, sleep scores, etc.)
    """
    # Get the most recent row from the dataset
    snapshot = wellness_df.tail(1).to_dict(orient="records")[0]
    tool_context.state["wearable_snapshot"] = snapshot
    
    # Extract Oura-specific metrics with validation
    resting_hr = snapshot.get("resting_hr")
    hr_average = snapshot.get("hr_average")
    hr_lowest = snapshot.get("hr_lowest")
    sleep_score = snapshot.get("sleep_score")
    sleep_efficiency = snapshot.get("sleep_efficiency")
    readiness_score = snapshot.get("readiness_score")
    total_sleep_duration = snapshot.get("total_sleep_duration")
    
    # Build human-readable summary with available metrics
    # Only include metrics that are not null/NaN
    metrics = []
    if resting_hr is not None and pd.notna(resting_hr):
        metrics.append(f"resting_hr={resting_hr:.1f}bpm")
    if hr_average is not None and pd.notna(hr_average):
        metrics.append(f"avg_hr={hr_average:.1f}bpm")
    if sleep_score is not None and pd.notna(sleep_score):
        metrics.append(f"sleep_score={sleep_score:.1f}")
    if sleep_efficiency is not None and pd.notna(sleep_efficiency):
        metrics.append(f"sleep_efficiency={sleep_efficiency:.1f}%")
    if readiness_score is not None and pd.notna(readiness_score):
        metrics.append(f"readiness={readiness_score}")
    if total_sleep_duration is not None and pd.notna(total_sleep_duration):
        hours = total_sleep_duration / 3600  # Convert seconds to hours
        metrics.append(f"sleep_duration={hours:.1f}hrs")
    
    summary = "wearable_ingested; latest metrics — " + ", ".join(metrics) if metrics else "wearable_ingested; no metrics available"
    return summary

def risk_calculator(tool_context: ToolContext) -> dict:
    """
    Calculate health risk score based on intake answers.
    
    This is a simplified risk calculator that counts the number of intake
    responses as a proxy for risk (more responses may indicate more complex
    health issues). In production, this would use sophisticated algorithms
    that analyze the content of answers, wearable metrics, and medical history.
    
    Args:
        tool_context: ADK tool context containing session state with intake_answers
        
    Returns:
        dict: Contains the calculated risk_score
    """
    answers = tool_context.state.get("intake_answers", [])
    # Simple risk calculation: more answers may indicate more health concerns
    # In production, this would analyze answer content and wearable metrics
    risk_score = len(answers)
    tool_context.state["risk_score"] = risk_score
    return {"risk_score": risk_score}

def care_plan_builder(tool_context: ToolContext) -> dict:
    """
    Generate informational lifestyle recommendations based on risk assessment.
    
    IMPORTANT: This tool creates general wellness lifestyle recommendations (e.g., 
    hydration, sleep hygiene) based on risk assessment. It does NOT prescribe 
    medications, provide medical treatments, or diagnose conditions. All recommendations 
    are informational and should be reviewed with licensed healthcare professionals.
    
    For higher risk cases, it suggests discussing specialist consultations with 
    healthcare providers. The care plan requires human review before finalization 
    (see care_plan_tool with require_confirmation=True).
    
    Args:
        tool_context: ADK tool context containing session state with risk_score
        
    Returns:
        dict: Contains the generated care_plan as a list of lifestyle recommendations
    """
    risk = tool_context.state.get("risk_score", 0)
    # Base care plan with general wellness lifestyle recommendations (NOT medical treatments)
    plan = ["Hydration", "Sleep hygiene"]
    # Add discussion point for specialist consultation for higher risk cases
    # Note: This is a suggestion to DISCUSS with healthcare provider, NOT a prescription
    if risk > 5:
        plan.append("Consult cardiologist")  # User should discuss this with their doctor
    tool_context.state["care_plan"] = plan
    return {"care_plan": plan}

def shopping_suggestions_with_check(tool_context: ToolContext) -> list[str]:
    """
    Provide informational supplement suggestions with evidence links (NOT prescriptions).
    
    IMPORTANT: This tool provides INFORMATION about dietary supplements for educational
    purposes only. It does NOT prescribe, diagnose, or provide medical advice. All
    supplement information should be reviewed with a licensed healthcare provider before
    use. This tool includes PubMed/WebMD links for users to research independently.
    
    This tool maps care plan recommendations to informational supplement references with
    research links. It includes medication interaction checking - if supplements that may
    interact with medications are mentioned, it flags the need for healthcare provider review.
    
    Args:
        tool_context: ADK tool context containing session state with care_plan
        
    Returns:
        list[str]: List of informational supplement references with research links
    """
    plan = tool_context.state.get("care_plan", [])
    
    # Try to use database-supplied supplements if available (from query_supplements tool)
    # Otherwise fall back to hardcoded catalog
    queried_supplements = tool_context.state.get("queried_supplements", [])
    use_database = len(queried_supplements) > 0
    
    # If we have a care plan but no database supplements yet, try querying database automatically
    if plan and not use_database:
        # Try to get supplements from database for care plan categories
        db_supplements = []
        category_map = {
            "Hydration": "Hydration",
            "Sleep hygiene": "Sleep hygiene",
            "Sleep Quality": "Sleep Quality",
            "Energy Support": "Energy Support",
            "Heart Health": "Heart Health",
        }
        for care_item in plan:
            db_category = category_map.get(care_item)
            if db_category:
                try:
                    db_supps = get_supplements_by_category(db_category)
                    db_supplements.extend(db_supps)
                except:
                    pass
        if db_supplements:
            queried_supplements = db_supplements
            use_database = True
            tool_context.state["queried_supplements"] = queried_supplements
    
    # Informational supplement catalog with research evidence links
    # NOTE: This is INFORMATION only, NOT medical advice or prescriptions
    # Evidence levels describe research status, not endorsement
    # This is used as fallback if database is not available
    product_catalog = {
        "Hydration": {
            "name": "Electrolyte Supplements",
            "info": "Electrolytes (sodium, potassium, magnesium) support hydration and fluid balance",
            "evidence": "Well-established: Electrolytes are essential nutrients (FDA recognized)",
            "pubmed": "https://pubmed.ncbi.nlm.nih.gov/?term=electrolyte+supplementation+hydration",
            "webmd": "https://www.webmd.com/diet/electrolyte-water",
            "amazon_link": "https://www.amazon.com/s?k=electrolyte+supplement+powder",
            "alternatives": ["Oral rehydration solutions", "Sports drinks with electrolytes"]
        },
        "Sleep hygiene": {
            "name": "Melatonin Supplements (3mg)",
            "info": "Melatonin is a naturally occurring hormone that regulates sleep-wake cycles",
            "evidence": "Strong evidence: Multiple RCTs support melatonin for sleep onset (NIH/NCCIH reviewed)",
            "pubmed": "https://pubmed.ncbi.nlm.nih.gov/?term=melatonin+sleep+efficacy+clinical+trial",
            "webmd": "https://www.webmd.com/sleep-disorders/sleep-disorders-melatonin",
            "amazon_link": "https://www.amazon.com/s?k=melatonin+gummies+3mg",
            "alternatives": [
                "Magnesium Glycinate (sleep support)",
                "L-Theanine (relaxation)",
                "Chamomile tea"
            ],
            "warning": "May interact with blood thinners, immunosuppressants, diabetes medications"
        },
        "Consult cardiologist": {
            "name": "Blood Pressure Monitor (Digital Upper Arm)",
            "info": "Home monitoring device for tracking cardiovascular health metrics",
            "evidence": "Clinical standard: Home BP monitoring recommended by AHA/ACC guidelines",
            "pubmed": "https://pubmed.ncbi.nlm.nih.gov/?term=home+blood+pressure+monitoring+efficacy",
            "webmd": "https://www.webmd.com/hypertension-high-blood-pressure/home-blood-pressure-monitoring",
            "amazon_link": "https://www.amazon.com/s?k=blood+pressure+monitor+digital+upper+arm",
            "alternatives": [
                "FDA-approved upper arm monitors (Omron, Withings)",
                "Validated wrist monitors"
            ]
        },
    }
    
    # General wellness supplements for informational purposes
    general_supplements = {
        "Energy Support": {
            "name": "B-Complex Vitamins with Vitamin D3",
            "info": "B vitamins support energy metabolism; D3 supports immune function and bone health",
            "evidence": "Moderate evidence: B vitamins for energy in deficiency states; D3 for immune support (NIH Office of Dietary Supplements)",
            "pubmed": "https://pubmed.ncbi.nlm.nih.gov/?term=B+vitamins+energy+metabolism+vitamin+D3+immune",
            "webmd": "https://www.webmd.com/diet/supplement-guide-vitamin-b-complex",
            "amazon_link": "https://www.amazon.com/s?k=B+complex+vitamins+vitamin+D3",
            "alternatives": ["Individual B vitamins (if specific deficiencies)", "CoQ10 for cellular energy"]
        },
        "Sleep Quality": {
            "name": "Magnesium Glycinate (400mg)",
            "info": "Magnesium supports muscle relaxation and may improve sleep quality",
            "evidence": "Moderate evidence: Some RCTs show sleep benefits; strong evidence for muscle relaxation (NIH ODS)",
            "pubmed": "https://pubmed.ncbi.nlm.nih.gov/?term=magnesium+glycinate+sleep+efficacy",
            "webmd": "https://www.webmd.com/diet/supplement-guide-magnesium",
            "amazon_link": "https://www.amazon.com/s?k=magnesium+glycinate+400mg",
            "alternatives": ["Magnesium Threonate", "Epsom salt baths (topical magnesium)"]
        },
        "Heart Health": {
            "name": "Omega-3 Fish Oil (1000mg EPA+DHA)",
            "info": "Omega-3 fatty acids EPA and DHA support cardiovascular health",
            "evidence": "Strong evidence: Multiple large studies show cardiovascular benefits (AHA recommends, FDA approved claims for cardiovascular disease)",
            "pubmed": "https://pubmed.ncbi.nlm.nih.gov/?term=omega+3+fish+oil+cardiovascular+disease+prevention",
            "webmd": "https://www.webmd.com/diet/guide/the-truth-about-omega-3",
            "amazon_link": "https://www.amazon.com/s?k=omega+3+fish+oil+1000mg+epa+dha",
            "alternatives": ["Algal omega-3 (plant-based)", "CoQ10", "Flaxseed oil"]
        }
    }
    
    # Build informational supplement list based on care plan
    # IMPORTANT: These are INFORMATION references, NOT prescriptions
    shopping_items = []
    needs_review = False
    
    # Note: Disclaimer will be added once at the end, not at the start
    
    # Use database supplements if available (prioritize database over hardcoded)
    if use_database and queried_supplements and plan:
        # Map care plan items to database supplements
        for care_item in plan:
            category_map = {
                "Hydration": "Hydration",
                "Sleep hygiene": "Sleep hygiene",
                "Sleep Quality": "Sleep Quality",
                "Energy Support": "Energy Support",
                "Heart Health": "Heart Health",
            }
            db_category = category_map.get(care_item)
            if db_category:
                # Find supplements matching this category
                matching_supps = [s for s in queried_supplements if s.get('category') == db_category]
                for supp in matching_supps:
                    item_text = f"{supp['name']}\n"
                    item_text += f"   What it is: {supp.get('info', 'N/A')}\n"
                    item_text += f"   Research Evidence Level: {supp.get('evidence_level', 'N/A')}\n"
                    # Include actual links
                    if supp.get('pubmed_link'):
                        item_text += f"   PubMed Link: {supp['pubmed_link']}\n"
                    if supp.get('webmd_link'):
                        item_text += f"   WebMD Link: {supp['webmd_link']}\n"
                    if supp.get('medication_interactions'):
                        interactions = supp['medication_interactions']
                        if isinstance(interactions, list) and interactions:
                            item_text += f"   Potential Medication Interactions: May interact with {', '.join(interactions[:3])}. Consult your doctor if you are taking these medications.\n"
                    shopping_items.append(item_text.strip())
    elif not plan:
        # If no care plan, provide general informational items
        for category, product in general_supplements.items():
            item_text = f"{product['name']}\n"
            item_text += f"   What it is: {product['info']}\n"
            item_text += f"   Research Evidence Level: {product['evidence']}\n"
            if product.get('pubmed'):
                item_text += f"   PubMed Link: {product['pubmed']}\n"
            if product.get('webmd'):
                item_text += f"   WebMD Link: {product['webmd']}\n"
            shopping_items.append(item_text.strip())
    else:
        # Map care plan items to informational supplement references
        for care_item in plan:
            if care_item in product_catalog:
                product = product_catalog[care_item]
                
                # Build detailed informational reference with links
                item_text = f"{product['name']}\n"
                item_text += f"   What it is: {product['info']}\n"
                item_text += f"   Research Evidence Level: {product['evidence']}\n"
                # Include actual links
                if product.get('pubmed'):
                    item_text += f"   PubMed Link: {product['pubmed']}\n"
                if product.get('webmd'):
                    item_text += f"   WebMD Link: {product['webmd']}\n"
                if product.get('warning'):
                    item_text += f"   Potential Medication Interactions: {product['warning']}\n"
                    needs_review = True
                medications = tool_context.state.get("user_profile", {}).get("medications", [])
                if medications and "Melatonin" in product['name']:
                    item_text += "   Potential Medication Interactions: May interact with blood thinners, immunosuppressants, diabetes medications. Consult your doctor if you are taking these medications.\n"
                    needs_review = True
                shopping_items.append(item_text.strip())
            else:
                # Generic fallback for unknown care plan items
                item_text = f"{care_item} (General wellness information)"
                shopping_items.append(item_text)
        
        # Add relevant general supplements based on risk/conditions (informational only)
        risk_score = tool_context.state.get("risk_score", 0)
        
        # Add energy support information if high risk
        if risk_score > 5:
            energy_product = general_supplements["Energy Support"]
            item_text = f"{energy_product['name']}\n"
            item_text += f"   What it is: {energy_product['info']}\n"
            item_text += f"   Research Evidence Level: {energy_product['evidence']}\n"
            if energy_product.get('pubmed'):
                item_text += f"   PubMed Link: {energy_product['pubmed']}\n"
            if energy_product.get('webmd'):
                item_text += f"   WebMD Link: {energy_product['webmd']}\n"
            shopping_items.append(item_text.strip())
    
    # Add heart health information if cardiovascular concerns (regardless of plan)
    user_profile = tool_context.state.get("user_profile", {})
    conditions = user_profile.get("conditions", [])
    if any(keyword in str(conditions).lower() for keyword in ["hypertension", "heart", "cardiac", "cardiovascular"]):
        # Check if heart health product is already in the list
        heart_already_added = any("Omega-3" in item or "Fish Oil" in item for item in shopping_items)
        if not heart_already_added:
            heart_product = general_supplements["Heart Health"]
            item_text = f"{heart_product['name']}\n"
            item_text += f"   What it is: {heart_product['info']}\n"
            item_text += f"   Research Evidence Level: {heart_product['evidence']}\n"
            if heart_product.get('pubmed'):
                item_text += f"   PubMed Link: {heart_product['pubmed']}\n"
            if heart_product.get('webmd'):
                item_text += f"   WebMD Link: {heart_product['webmd']}\n"
            shopping_items.append(item_text.strip())
    
    # Update state with shopping list for final summary
    tool_context.state["shopping_list"] = shopping_items
    tool_context.state["needs_medication_review"] = needs_review
    
    # Add concise legal disclaimer (only once at the end, check if already exists)
    disclaimer_text = "="*70 + "\nIMPORTANT: Informational Only - Consult Healthcare Provider\n" + "="*70
    if shopping_items and not any("IMPORTANT: Informational" in item for item in shopping_items):
        shopping_items.append(disclaimer_text)
    
    return shopping_items if shopping_items else [
        "General wellness supplements",
        "Amazon: https://www.amazon.com/s?k=general+wellness+supplements"
    ]

def plot_wellness(metric: str) -> str:
    """
    Generate interactive Plotly visualization for wellness metric trends.
    
    This tool creates time-series line plots for health metrics from the Oura
    dataset. It handles data cleaning (removes NaN values), formats dates,
    and produces publication-ready visualizations for trend analysis.
    
    Args:
        metric: Name of the metric to plot (supports aliases)
        
    Returns:
        str: Confirmation message with number of data points plotted
    """
    # Map user-friendly names to actual column names
    metric_mapping = {
        "heart_rate": "resting_hr",
        "hr": "resting_hr",
        "resting_hr": "resting_hr",
        "sleep_score": "sleep_score",
        "readiness": "readiness_score",
        "readiness_score": "readiness_score",
        "hr_average": "hr_average",
        "sleep_efficiency": "sleep_efficiency",
    }
    
    actual_metric = metric_mapping.get(metric.lower(), metric)
    
    # Validate metric exists
    if actual_metric not in wellness_df.columns:
        return f"Metric '{metric}' not found. Available: {list(wellness_df.columns)}"
    
    # Filter out NaN values and prepare data for visualization
    df_plot = wellness_df[['date', actual_metric]].copy()
    df_plot = df_plot.dropna(subset=[actual_metric])  # Remove rows with NaN for this metric
    
    if len(df_plot) == 0:
        return f"No valid data found for {actual_metric} (all values are NaN)"
    
    # Convert date to datetime format for proper time-series plotting
    df_plot['date'] = pd.to_datetime(df_plot['date'])
    
    # Sort by date to ensure chronological order
    df_plot = df_plot.sort_values('date')
    
    # Create interactive line plot with Plotly
    fig = px.line(
        df_plot, 
        x="date", 
        y=actual_metric, 
        title=f"{actual_metric.replace('_', ' ').title()} Trend",
        markers=True  # Add markers to show individual data points
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=actual_metric.replace('_', ' ').title(),
    )
    
    # Display the plot - works in both notebook and script environments
    try:
        # Try IPython display first (works in Jupyter/Kaggle notebooks)
        from IPython.display import display, HTML
        # Save to HTML and display
        html_file = f'plot_{actual_metric}.html'
        fig.write_html(html_file, auto_open=False)
        display(HTML(filename=html_file))
        return f"Plotted {actual_metric} trend with {len(df_plot)} data points. Plot saved to {html_file}"
    except ImportError:
        # IPython not available, try direct show
        try:
            fig.show(renderer='notebook')
            return f"Plotted {actual_metric} trend with {len(df_plot)} data points"
        except:
            try:
                fig.show()
                return f"Plotted {actual_metric} trend with {len(df_plot)} data points"
            except:
                # Save as HTML file as fallback
                try:
                    html_file = f'plot_{actual_metric}.html'
                    fig.write_html(html_file, auto_open=False)
                    return f"Plotted {actual_metric} trend with {len(df_plot)} data points. Plot saved to {html_file} (open in browser to view)"
                except:
                    return f"Plotted {actual_metric} trend with {len(df_plot)} data points (display failed, but data is plotted)"

def human_review(
    review_type: str, 
    content: str, 
    tool_context: ToolContext
) -> str:
    """
    Request human review for critical healthcare decisions.
    
    This tool implements human-in-the-loop (HITL) functionality for safety.
    Critical recommendations (care plans, medication interactions, high-risk
    assessments) are flagged for human review before being finalized. This
    ensures healthcare decisions have proper oversight.
    
    Args:
        review_type: Type of content to review (care_plan, medication, shopping, risk)
        content: The content requiring human review
        tool_context: ADK tool context containing session state
        
    Returns:
        str: Confirmation message indicating review request status
    """
    review_requests = tool_context.state.setdefault("review_requests", [])
    review_requests.append({
        "type": review_type,
        "content": content,
        "status": "pending_review"
    })
    
    return (
        f"Human review requested for {review_type}. "
        f"Content: {content}. "
        f"Status: Pending approval."
    )

def approve_recommendation(
    review_type: str,
    approved: bool,
    notes: str,
    tool_context: ToolContext
) -> str:
    """
    Approve or reject a recommendation after human review.
    
    This tool records the outcome of human review. Once a recommendation
    is approved or rejected, downstream agents can proceed accordingly.
    This completes the HITL workflow for critical healthcare decisions.
    
    Args:
        review_type: Type of content that was reviewed
        approved: True if approved, False if rejected
        notes: Notes from the human reviewer explaining the decision
        tool_context: ADK tool context containing session state
        
    Returns:
        str: Confirmation message with review outcome and notes
    """
    approvals = tool_context.state.setdefault("approvals", {})
    approvals[review_type] = {
        "approved": approved,
        "notes": notes,
        "reviewed_at": "now"  # In production, use actual timestamp (e.g., datetime.now().isoformat())
    }
    
    status = "APPROVED" if approved else "REJECTED"
    return f"Review {status} for {review_type}. Notes: {notes}"

def final_summary(tool_context: ToolContext) -> str:
    """
    Generate comprehensive final summary aggregating all workflow results.
    
    This tool collects all data from the session state (intake answers, wearable
    metrics, risk scores, care plans, shopping lists) and formats them into a
    unified summary. The final summary agent uses this to present a complete
    health assessment to the user.
    
    Args:
        tool_context: ADK tool context containing complete session state
        
    Returns:
        str: Formatted summary string
    """
    # Extract key information
    intake_answers = tool_context.state.get("intake_answers", [])
    wearable_snapshot = tool_context.state.get("wearable_snapshot", {})
    risk_score = tool_context.state.get("risk_score", 0)
    care_plan = tool_context.state.get("care_plan", [])
    shopping_list = tool_context.state.get("shopping_list", [])
    
    # Build readable summary
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("HEALTHCARE CONCIERGE - FINAL SUMMARY")
    summary_lines.append("="*80)
    
    # User Information
    if intake_answers:
        summary_lines.append("\n USER INFORMATION:")
        summary_lines.append("-" * 80)
        for qa in intake_answers[:6]:  # Limit to first 6 Q&A pairs
            summary_lines.append(f"  {qa.get('question', '')}: {qa.get('answer', '')}")
    
    # Wearable Metrics (key metrics only)
    if wearable_snapshot:
        summary_lines.append("\n KEY WEARABLE METRICS:")
        summary_lines.append("-" * 80)
        key_metrics = ['resting_hr', 'hr_average', 'sleep_score', 'sleep_efficiency', 'readiness_score']
        for metric in key_metrics:
            value = wearable_snapshot.get(metric)
            if value is not None and (isinstance(value, (int, float)) and not pd.isna(value)):
                summary_lines.append(f"  {metric.replace('_', ' ').title()}: {value}")
    
    # Risk Assessment
    summary_lines.append("\n  RISK ASSESSMENT:")
    summary_lines.append("-" * 80)
    summary_lines.append(f"  Risk Score: {risk_score}")
    
    # Care Plan
    if care_plan:
        summary_lines.append("\n CARE PLAN RECOMMENDATIONS:")
        summary_lines.append("-" * 80)
        for item in care_plan:
            summary_lines.append(f"  • {item}")
    
    # Shopping Suggestions (simplified - just count and key items)
    if shopping_list:
        summary_lines.append("\n SUPPLEMENT REFERENCES:")
        summary_lines.append("-" * 80)
        # Count actual supplement items (exclude disclaimers)
        supplement_items = [item for item in shopping_list 
                           if not item.startswith("=") and 
                           not "DISCLAIMER" in item.upper() and
                           not "INFORMATIONAL" in item.upper()]
        summary_lines.append(f"  Total supplement references: {len(supplement_items)}")
        # Show first 3 items as examples
        for item in supplement_items[:3]:
            # Extract just the product name (first line)
            lines = item.split('\n')
            if lines:
                product_name = lines[0].strip()
                if product_name:
                    summary_lines.append(f"  • {product_name}")
        if len(supplement_items) > 3:
            summary_lines.append(f"  ... and {len(supplement_items) - 3} more references")
        summary_lines.append("\n  Note: All supplement information is for educational purposes only.")
        summary_lines.append("  Consult your healthcare provider before use.")
    
    summary_lines.append("\n" + "="*80)
    
    return "\n".join(summary_lines)

# ------------------------------------------------------------------
# Database Tools - ADK Function Calling with Database
# ------------------------------------------------------------------

def query_user_profile(user_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Query user profile from database.
    
    This tool demonstrates ADK's function calling capability with database queries.
    Agents can call this to retrieve user medical history, medications, and allergies
    from the database instead of hardcoded values.
    
    Args:
        user_id: Unique identifier for the user
        tool_context: ADK tool context
        
    Returns:
        dict: User profile data including medical history, medications, allergies
    """
    profile = get_user_profile(user_id)
    
    if profile:
        # Store in session state for other agents
        tool_context.state["user_profile"] = profile
        return {
            "status": "success",
            "user_profile": profile,
            "message": f"Retrieved profile for {profile['name']}"
        }
    else:
        return {
            "status": "not_found",
            "message": f"User profile not found for user_id: {user_id}"
        }

def query_supplements(category: Union[str, None] = None, name_pattern: Union[str, None] = None, tool_context: Union[ToolContext, None] = None) -> Dict[str, Any]:
    """
    Query supplements from database by category or name.
    
    This tool allows the shopping agent to query the supplements catalog
    dynamically instead of using hardcoded product lists. Demonstrates
    ADK's ability to work with structured database queries.
    
    Args:
        category: Filter by supplement category (e.g., 'Sleep hygiene', 'Hydration')
        name_pattern: Search by name pattern (partial match)
        tool_context: ADK tool context
        
    Returns:
        list: List of supplement dictionaries with research links
    """
    if category:
        supplements = get_supplements_by_category(category)
    elif name_pattern:
        supplements = get_supplements_by_name(name_pattern)
    else:
        return {
            "status": "error",
            "message": "Must provide either 'category' or 'name_pattern' parameter"
        }
    
    if tool_context:
        tool_context.state["queried_supplements"] = supplements
    
    return {
        "status": "success",
        "count": len(supplements),
        "supplements": supplements
    }

def save_session_for_evaluation(
    session_id: str,
    user_id: str,
    user_prompt: str,
    final_summary: str,
    risk_score: int,
    care_plan: List[str],
    shopping_suggestions: List[str],
    execution_time: float,
    agent_sequence: List[str],
    tool_context: ToolContext
) -> str:
    """
    Save session data for evaluation and analytics.
    
    This tool demonstrates how to persist session data for evaluation purposes.
    Useful for measuring agent performance, response quality, and system metrics.
    
    Args:
        session_id: Unique session identifier
        user_id: User identifier
        user_prompt: Original user input
        final_summary: Final summary from summary agent
        risk_score: Calculated risk score
        care_plan: Generated care plan
        shopping_suggestions: Shopping suggestions provided
        execution_time: Total execution time in seconds
        agent_sequence: List of agents that executed in order
        tool_context: ADK tool context
        
    Returns:
        str: Confirmation message
    """
    save_session_history(
        session_id=session_id,
        user_id=user_id,
        user_prompt=user_prompt,
        final_summary=final_summary,
        risk_score=risk_score,
        care_plan=care_plan,
        shopping_suggestions=shopping_suggestions,
        execution_time=execution_time,
        agent_sequence=agent_sequence
    )
    
    return f"Session {session_id} saved for evaluation"

def log_evaluation_metric(
    session_id: str,
    agent_name: str,
    metric_name: str,
    metric_value: float,
    notes: str = "",
    tool_context: Optional[ToolContext] = None
) -> str:
    """
    Log evaluation metric for an agent.
    
    This tool allows agents to log their own performance metrics for evaluation.
    Demonstrates self-monitoring capabilities in ADK agents.
    
    Args:
        session_id: Session identifier
        agent_name: Name of the agent being evaluated
        metric_name: Name of the metric (e.g., 'response_time', 'accuracy', 'tool_usage')
        metric_value: Value of the metric
        notes: Optional notes about the metric
        tool_context: ADK tool context
        
    Returns:
        str: Confirmation message
    """
    save_evaluation_metric(
        session_id=session_id,
        agent_name=agent_name,
        metric_name=metric_name,
        metric_value=metric_value,
        notes=notes
    )
    
    return f"Logged {metric_name}={metric_value} for {agent_name} in session {session_id}"





# Configure tools with human-in-the-loop requirements
# Care plan tool: For demo purposes, don't require confirmation (pre-approvals exist in initial state)
# For production: Set require_confirmation=True and remove pre-approvals from initial state
care_plan_tool = FunctionTool(
    care_plan_builder,
    require_confirmation=False,  # Set to True for production to require human approval
)

# Shopping tool requires confirmation only if medication interactions are detected
shopping_tool = FunctionTool(
    shopping_suggestions_with_check,
    require_confirmation=lambda tool_context: tool_context.state.get("needs_medication_review", False),
)

# List of all tools available to agents
# Note: Some tools are configured separately (care_plan_tool, shopping_tool) for special handling
tools = [
    FunctionTool(log_intake_answer),
    FunctionTool(trend_analyzer),
    FunctionTool(wearable_ingest),
    FunctionTool(risk_calculator),
    FunctionTool(care_plan_builder),
    FunctionTool(shopping_suggestions_with_check),
    FunctionTool(human_review),
    FunctionTool(approve_recommendation),
    FunctionTool(plot_wellness),
    FunctionTool(final_summary),
]



# ------------------------------------------------------------------
# Using ADK's Evaluation
# ------------------------------------------------------------------

def generate_evaluation_report(session_id: str) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report for a session using database queries.
    
    This function uses ADK's evaluation framework concepts but queries the database
    directly for runtime metrics that were logged during agent execution.
    
    Args:
        session_id: Session identifier
        
    Returns:
        dict: Complete evaluation report
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get session data
    cursor.execute("SELECT * FROM session_history WHERE session_id = ?", (session_id,))
    session = cursor.fetchone()
    
    if not session:
        return {"error": "Session not found"}
    
    # Get evaluation metrics
    cursor.execute("""
        SELECT agent_name, metric_name, AVG(metric_value) as avg_value, COUNT(*) as count
        FROM agent_evaluations
        WHERE session_id = ?
        GROUP BY agent_name, metric_name
    """, (session_id,))
    metrics = cursor.fetchall()
    
    conn.close()
    
    return {
        "session_id": session_id,
        "user_id": session["user_id"],
        "execution_time": session["execution_time_seconds"],
        "risk_score": session["risk_score"],
        "agent_sequence": json.loads(session["agent_sequence"]),
        "metrics": [
            {
                "agent": row["agent_name"],
                "metric": row["metric_name"],
                "average_value": row["avg_value"],
                "count": row["count"]
            }
            for row in metrics
        ],
        "summary": {
            "total_agents": len(json.loads(session["agent_sequence"])),
            "total_execution_time": session["execution_time_seconds"],
            "risk_score": session["risk_score"]
        }
    }

async def run_evaluation_suite(session_id: str) -> Dict[str, Any]:
    """
    Run complete evaluation suite for a session using ADK's evaluation framework.
    
    This function demonstrates how to use ADK's AgentEvaluator for structured
    evaluations. For runtime metrics, it queries the database directly.
    
    Args:
        session_id: Session identifier to evaluate
        
    Returns:
        dict: Complete evaluation results
    """
    # Generate report from database (runtime metrics)
    report = generate_evaluation_report(session_id)
    
    # Add quality scores from database
    if "error" not in report:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM session_history WHERE session_id = ?", (session_id,))
        session = cursor.fetchone()
        
        if session:
            care_plan = json.loads(session["care_plan"])
            shopping_suggestions = json.loads(session["shopping_suggestions"])
            
            # Simple quality metrics based on data
            care_plan_quality = {
                "metric": "care_plan_quality",
                "care_plan_items": len(care_plan),
                "score": min(len(care_plan) * 20, 100)  # Simple scoring
            }
            
            shopping_quality = {
                "metric": "shopping_suggestions_quality",
                "suggestions_count": len(shopping_suggestions),
                "has_research_links": any("pubmed" in str(s).lower() or "webmd" in str(s).lower() for s in shopping_suggestions),
                "score": 80 if any("pubmed" in str(s).lower() or "webmd" in str(s).lower() for s in shopping_suggestions) else 40
            }
            
            report["quality_metrics"] = {
                "care_plan": care_plan_quality,
                "shopping_suggestions": shopping_quality
            }
        
        conn.close()
    
    return report


# ------------------------------------------------------------------
# Agents - Specialized AI Agents for Healthcare Workflow
# ------------------------------------------------------------------

# Agent 1: Intake Agent
# Purpose: Collect structured medical history, symptoms, and lifestyle information
intake_agent = Agent(
    name="intake_agent",
    model="gemini-2.0-flash",
    instruction="Collect user medical history, symptoms, lifestyle. You can use query_user_profile tool to retrieve user data from database.",
    tools=[FunctionTool(log_intake_answer), FunctionTool(query_user_profile)],
)

# Agent 2: Wearable Agent
# Purpose: Ingest and analyze objective health data from Oura ring wearable device
wearable_agent = Agent(
    name="wearable_agent",
    model="gemini-2.0-flash",
    instruction=(
        "Ingest the latest Oura ring wearable data from session state. "
        "After calling the `wearable_ingest` tool, summarize key vitals including: "
        "resting heart rate, average heart rate, sleep score, sleep efficiency, "
        "readiness score, and total sleep duration. "
        "Provide insights on what these metrics indicate about the user's health."
    ),
    tools=[wearable_ingest],
)

# Agent 3: Trend Agent
# Purpose: Analyze historical trends in wearable data to identify patterns
# OPTIMIZATION: Limit to 2-3 key metrics to reduce API calls on free tier
trend_agent = Agent(
    name="trend_agent",
    model="gemini-2.0-flash",
    instruction=(
        "Analyze trends in Oura ring data for KEY METRICS ONLY to save API calls. "
        "Focus on the 2-3 most important metrics: readiness_score and resting_hr. "
        "Call trend_analyzer with a metric name and window size of 7 days. "
        "Analyze only readiness_score and resting_hr trends - skip sleep_score if data is missing. "
        "Call plot_wellness for readiness_score only (most important metric). "
        "DO NOT call plot_wellness for every metric - limit to 1-2 plots maximum."
    ),
    tools=[trend_analyzer, plot_wellness],
)

# Agent 4: Risk Agent
# Purpose: Synthesize subjective (intake) and objective (wearable) data to assess health risk
risk_agent = Agent(
    name="risk_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a risk assessment agent. Your role is to synthesize the intake answers "
        "and wearable data to calculate a health risk score. Use the risk_calculator tool "
        "to compute the risk score. Focus only on risk assessment - do not provide care plans "
        "or shopping suggestions (those will be handled by specialized agents that follow). "
        "Present the risk score clearly and explain what it means based on the available data."
    ),
    tools=[risk_calculator],
)

# Agent 5: Care Plan Agent
# Purpose: Generate informational lifestyle recommendations (NOT medical treatments)
care_plan_agent = Agent(
    name="care_plan_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a wellness information assistant providing lifestyle recommendations. "
        "Your role is to suggest general wellness approaches (e.g., hydration, sleep hygiene, "
        "exercise, stress management) based on risk assessment.\n\n"
        "CRITICAL RULES:\n"
        "1. Provide LIFESTYLE recommendations only (NOT medical treatments)\n"
        "2. Do NOT prescribe medications or medical treatments\n"
        "3. Do NOT diagnose conditions\n"
        "4. For specialist referrals (e.g., 'Consult cardiologist'), emphasize that "
        "this is for the user to discuss with their healthcare provider\n"
        "5. Always frame recommendations as general wellness information\n"
        "6. Clarify that all recommendations should be reviewed with licensed healthcare professionals"
    ),
    tools=[care_plan_tool],  # Use care_plan_tool (with conditional confirmation) instead of care_plan_builder directly
)

# Agent 6: Human Review Agent
# Purpose: Ensure critical healthcare decisions are reviewed by humans before finalization
human_review_agent = Agent(
    name="human_review_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a healthcare review agent. Your job is to check if recommendations are approved.\n\n"
        "IMPORTANT: First check if care_plan already has approval in the session state:\n"
        "- Check tool_context.state.get('approvals', {}).get('care_plan', {}).get('approved')\n"
        "- If approved=True already exists, skip review and immediately proceed - do NOT call human_review\n"
        "- Simply state: 'Care plan is already pre-approved. Proceeding with workflow.'\n"
        "- Only call human_review if approval does NOT exist\n\n"
        "For this demo with pre-approvals, you should find that care_plan is already approved, "
        "so skip the review step and proceed immediately."
    ),
    tools=[FunctionTool(human_review), FunctionTool(approve_recommendation)],
)

# Agent 7: Shopping Agent
# Purpose: Provide informational supplement references (NOT prescriptions) with evidence links
shopping_agent = Agent(
    name="shopping_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are an informational assistant providing EDUCATIONAL references about dietary "
        "supplements. Your role is to present supplement INFORMATION, NOT to prescribe or "
        "recommend medical treatments. You can use query_supplements tool to retrieve supplement "
        "information from the database by category or name.\n\n"
        "CRITICAL RULES:\n"
        "1. Always state that this is INFORMATION only, NOT medical advice\n"
        "2. Present each supplement with:\n"
        "   - What it is (informational description)\n"
        "   - Research evidence level (factual research status)\n"
        "   - PubMed link (for research literature)\n"
        "   - WebMD link (for consumer-friendly information)\n"
        "3. NEVER suggest that supplements are prescriptions or medical treatments\n"
        "4. ALWAYS emphasize: 'Discuss with your healthcare provider before use'\n"
        "5. Flag medication interactions clearly and require doctor consultation\n"
        "6. Distinguish between:\n"
        "   - Providing information about supplements (OK - educational)\n"
        "   - Prescribing medications (NOT OK - requires medical license)\n\n"
        "After receiving supplement information from the shopping tool, format it clearly "
        "with all research links. Always end with clear instructions to consult healthcare "
        "professionals. Make it very clear that supplements should only be used after medical "
        "clearance, especially if the user is on medications or has medical conditions."
    ),
    tools=[shopping_tool, FunctionTool(query_supplements)],
)

# Agent 8: Final Summary Agent
# Purpose: Aggregate all workflow results into a comprehensive summary
final_summary_agent = Agent(
    name="final_summary_agent",
    model="gemini-2.0-flash",
    instruction="Produce final summary combining all results. Use save_session_for_evaluation to persist session data.",
    tools=[FunctionTool(final_summary), FunctionTool(save_session_for_evaluation), FunctionTool(log_evaluation_metric)],
)


# ------------------------------------------------------------------
# Agent Orchestration - Sequential Agent Chain
# ------------------------------------------------------------------

# SequentialAgent chains multiple specialized agents in a defined order
# Each agent receives context from previous agents through shared session state
# This enables complex multi-step reasoning that single agents cannot achieve
# 
# Workflow: Intake -> Wearable -> Trend -> Risk -> Care -> Human Review -> Shopping -> Summary
care_coordinator_agent = SequentialAgent(
    name="care_coordinator",
    description="Coordinate all sub-agents including human review for critical decisions.",
    sub_agents=[
        intake_agent,          # Step 1: Collect user health information
        wearable_agent,        # Step 2: Ingest objective wearable data
        trend_agent,           # Step 3: Analyze historical trends
        risk_agent,            # Step 4: Assess health risk
        care_plan_agent,       # Step 5: Generate care plan
        human_review_agent,    # Step 6: Human review for safety
        shopping_agent,        # Step 7: Suggest wellness products
        final_summary_agent,   # Step 8: Aggregate final summary
    ],
)


# ------------------------------------------------------------------
# SQL Queries for Evaluation Analysis
# ------------------------------------------------------------------

def get_evaluation_queries() -> Dict[str, str]:
    """
    Returns a dictionary of SQL queries for analyzing agent performance.
    
    Returns:
        dict: Dictionary mapping query names to SQL query strings
    """
    queries = {
        # Query 1: Get all sessions with execution times
        "all_sessions": """
            SELECT 
                session_id,
                user_id,
                user_prompt,
                risk_score,
                execution_time_seconds,
                created_at,
                json_array_length(agent_sequence) as num_agents
            FROM session_history
            ORDER BY created_at DESC;
        """,
        
        # Query 2: Get evaluation metrics per agent across all sessions
        "agent_metrics_summary": """
            SELECT 
                agent_name,
                metric_name,
                COUNT(*) as metric_count,
                AVG(metric_value) as avg_value,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                SUM(metric_value) as total_value
            FROM agent_evaluations
            GROUP BY agent_name, metric_name
            ORDER BY agent_name, metric_name;
        """,
        
        # Query 3: Get evaluation metrics for a specific session
        "session_metrics": """
            SELECT 
                ae.session_id,
                ae.agent_name,
                ae.metric_name,
                ae.metric_value,
                ae.notes,
                ae.created_at,
                sh.execution_time_seconds,
                sh.risk_score
            FROM agent_evaluations ae
            JOIN session_history sh ON ae.session_id = sh.session_id
            WHERE ae.session_id = ?
            ORDER BY ae.agent_name, ae.metric_name;
        """,
        
        # Query 4: Get average metrics per agent (across all sessions)
        "agent_performance_avg": """
            SELECT 
                agent_name,
                metric_name,
                AVG(metric_value) as avg_metric_value,
                COUNT(DISTINCT session_id) as sessions_counted
            FROM agent_evaluations
            GROUP BY agent_name, metric_name
            ORDER BY agent_name, avg_metric_value DESC;
        """,
        
        # Query 5: Get session performance summary
        "session_performance": """
            SELECT 
                sh.session_id,
                sh.user_id,
                sh.execution_time_seconds,
                sh.risk_score,
                COUNT(DISTINCT ae.agent_name) as agents_evaluated,
                COUNT(ae.evaluation_id) as total_metrics,
                AVG(ae.metric_value) as avg_metric_value
            FROM session_history sh
            LEFT JOIN agent_evaluations ae ON sh.session_id = ae.session_id
            GROUP BY sh.session_id
            ORDER BY sh.created_at DESC;
        """,
        
        # Query 6: Get care plan and shopping suggestions quality
        "care_plan_quality": """
            SELECT 
                session_id,
                user_id,
                risk_score,
                json_array_length(care_plan) as care_plan_items,
                json_array_length(shopping_suggestions) as shopping_items_count,
                execution_time_seconds,
                created_at
            FROM session_history
            ORDER BY created_at DESC;
        """,
        
        # Query 7: Get metrics for a specific agent across all sessions
        "agent_specific_metrics": """
            SELECT 
                session_id,
                metric_name,
                metric_value,
                notes,
                created_at
            FROM agent_evaluations
            WHERE agent_name = ?
            ORDER BY created_at DESC;
        """,
        
        # Query 8: Compare execution times across sessions
        "execution_time_comparison": """
            SELECT 
                session_id,
                user_id,
                execution_time_seconds,
                json_array_length(agent_sequence) as num_agents,
                risk_score,
                execution_time_seconds / json_array_length(agent_sequence) as time_per_agent
            FROM session_history
            ORDER BY execution_time_seconds DESC;
        """,
        
        # Query 9: Get all evaluation metrics with session details
        "detailed_evaluation_report": """
            SELECT 
                ae.evaluation_id,
                ae.session_id,
                sh.user_id,
                ae.agent_name,
                ae.metric_name,
                ae.metric_value,
                ae.notes,
                sh.execution_time_seconds,
                sh.risk_score,
                ae.created_at as metric_created_at,
                sh.created_at as session_created_at
            FROM agent_evaluations ae
            JOIN session_history sh ON ae.session_id = sh.session_id
            ORDER BY sh.created_at DESC, ae.agent_name, ae.metric_name;
        """,
        
        # Query 10: Get top performing agents by metric
        "top_agents_by_metric": """
            SELECT 
                agent_name,
                metric_name,
                AVG(metric_value) as avg_score,
                COUNT(*) as evaluation_count
            FROM agent_evaluations
            WHERE metric_name = ?
            GROUP BY agent_name, metric_name
            ORDER BY avg_score DESC
            LIMIT 10;
        """
    }
    return queries


def run_evaluation_query(query_name: str, params: tuple = None) -> List[Dict[str, Any]]:
    """
    Execute a predefined evaluation query.
    
    Args:
        query_name: Name of the query from get_evaluation_queries()
        params: Optional tuple of parameters for parameterized queries
        
    Returns:
        list: List of dictionaries containing query results
    """
    queries = get_evaluation_queries()
    
    if query_name not in queries:
        raise ValueError(
            f"Query '{query_name}' not found. Available queries: {list(queries.keys())}"
        )
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if params:
        cursor.execute(queries[query_name], params)
    else:
        cursor.execute(queries[query_name])
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert rows to list of dictionaries
    return [dict(row) for row in rows]


def print_evaluation_summary(session_id: Optional[str] = None):
    """
    Print a comprehensive evaluation summary.
    
    Args:
        session_id: Optional session ID to filter results. If None, shows all sessions.
    """
    print("\n" + "="*80)
    print("AGENT EVALUATION SUMMARY")
    print("="*80)
    
    if session_id:
        print(f"\n Session: {session_id}\n")
        metrics = run_evaluation_query("session_metrics", (session_id,))
        if metrics:
            print("Agent Metrics for this session:")
            print("-" * 80)
            for metric in metrics:
                print(
                    f"  Agent: {metric['agent_name']:<20} | "
                    f"Metric: {metric['metric_name']:<20} | "
                    f"Value: {metric['metric_value']:<10.2f} | "
                    f"Notes: {metric.get('notes', 'N/A')}"
                )
        else:
            print("  No metrics found for this session.")
            print("  Note: Metrics are logged when agents call log_evaluation_metric tool.")
    else:
        # Show overall summary
        print("\n Overall Performance Metrics:\n")
        agent_summary = run_evaluation_query("agent_metrics_summary")
        if agent_summary:
            print("Average Metrics per Agent:")
            print("-" * 80)
            for row in agent_summary:
                print(
                    f"  Agent: {row['agent_name']:<20} | "
                    f"Metric: {row['metric_name']:<20} | "
                    f"Avg: {row['avg_value']:<10.2f} | "
                    f"Count: {row['metric_count']}"
                )
        else:
            print("  No metrics found in database.")
            print("  Note: Metrics are logged when agents call log_evaluation_metric tool.")
        
        print("\n Session Performance:\n")
        session_perf = run_evaluation_query("session_performance")
        if session_perf:
            print("Session Summary:")
            print("-" * 80)
            for session in session_perf[:10]:  # Show top 10
                print(
                    f"  Session: {session['session_id'][:8]}... | "
                    f"Execution Time: {session['execution_time_seconds']:.2f}s | "
                    f"Risk Score: {session['risk_score']} | "
                    f"Agents Evaluated: {session['agents_evaluated']}"
                )
    
    print("\n" + "="*80 + "\n")



async def run_with_retry(runner, user_id, session_id, new_message, max_retries=5):
    """
    Run agent workflow with automatic retry on rate limit errors.
    
    Google's Gemini API may return 429 (Resource Exhausted) errors when
    rate limits are exceeded (free tier: 10 requests/minute). This function 
    implements retry with proper delay extraction to gracefully handle these 
    situations and retry the operation.
    
    Args:
        runner: ADK Runner instance configured with agents
        user_id: Unique identifier for the user
        session_id: Unique identifier for the session
        new_message: User message to process
        max_retries: Maximum number of retry attempts (default: 5)
        
    Yields:
        Event objects from the agent workflow
    """
    for attempt in range(max_retries):
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message,
            ):
                yield event
            break  # Success, exit retry loop
        except Exception as e:
            # Check if it's a 429 rate limit error (handle multiple exception types)
            is_rate_limit = False
            retry_delay = 60  # Default 60 seconds for rate limits
            error_message = str(e)
            error_type = type(e).__name__
            
            # Check for ResourceExhausted
            if isinstance(e, ResourceExhausted):
                is_rate_limit = True
            # Check for ClientError with 429 status (with fallback for import issues)
            elif error_type == 'ClientError' or 'ClientError' in error_type:
                error_status = getattr(e, 'status_code', None)
                if error_status == 429 or '429' in error_message or 'RESOURCE_EXHAUSTED' in error_message:
                    is_rate_limit = True
            # Check error message for 429/quota indicators (works even if ClientError import failed)
            elif '429' in error_message or 'RESOURCE_EXHAUSTED' in error_message or 'quota' in error_message.lower():
                is_rate_limit = True
            
            if is_rate_limit:
                # Try to extract retry delay from error message
                # Error format: "Please retry in 6.653287933s"
                retry_match = re.search(r'retry in ([\d.]+)\s*s', error_message, re.IGNORECASE)
                if retry_match:
                    retry_delay = float(retry_match.group(1))
                    # Add buffer to be safe
                    retry_delay = max(retry_delay + 2, 15)  # At least 15 seconds
                else:
                    # Try to extract from error details if available
                    try:
                        error_details = getattr(e, 'error', {})
                        if isinstance(error_details, dict):
                            details = error_details.get('details', [])
                            for detail in details:
                                if isinstance(detail, dict) and detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                    retry_info = detail.get('retryDelay', '')
                                    if isinstance(retry_info, str):
                                        retry_match = re.search(r'(\d+)s?', retry_info)
                                        if retry_match:
                                            retry_delay = int(retry_match.group(1)) + 2
                    except:
                        pass  # If we can't extract delay, use default
            
            # Check if it's daily quota (250/day) vs per-minute limit (10/min)
            is_daily_quota = '250' in error_message or 'quota exceeded' in error_message.lower() or 'free_tier_requests' in error_message.lower()
            
            if is_rate_limit and attempt < max_retries - 1:
                if is_daily_quota:
                    print(f"\n Daily quota exceeded (429 error).")
                    print(f"   Free tier limit: 250 requests/day per model")
                    print(f"   You've used your daily quota. Please wait until tomorrow or upgrade to paid tier.")
                    print(f"   Monitor usage: https://ai.dev/usage?tab=rate-limit")
                    print(f"   Waiting {retry_delay:.1f} seconds before retry {attempt + 1}/{max_retries}...\n")
                else:
                    print(f"\n Rate limit exceeded (429 error).")
                    print(f"   Free tier limit: 10 requests/minute per model")
                    print(f"   Waiting {retry_delay:.1f} seconds before retry {attempt + 1}/{max_retries}...")
                    print(f"   Tip: Consider upgrading to paid tier for higher limits\n")
                await asyncio.sleep(retry_delay)
            elif is_rate_limit:
                # Final attempt failed
                if is_daily_quota:
                    print(f"\n Daily quota exhausted (250 requests/day limit reached).")
                    print(f"   This workflow uses 8 agents, each making multiple API calls.")
                    print(f"   Solutions:")
                    print(f"   1. Wait until tomorrow (quota resets daily)")
                    print(f"   2. Upgrade to paid tier: https://ai.google.dev/pricing")
                    print(f"   3. Reduce number of agents or combine some agents")
                    print(f"   Monitor usage: https://ai.dev/usage?tab=rate-limit\n")
                else:
                    print(f"\n Rate limit error persists after {max_retries} attempts.")
                    print(f"   Please wait a few minutes and try again, or upgrade to paid tier.")
                    print(f"   Monitor usage: https://ai.dev/usage?tab=rate-limit\n")
                raise
            else:
                # Not a rate limit error, re-raise immediately
                raise
            # If not a rate limit error, check if it's another retryable error
            if not is_rate_limit:
                # For non-rate-limit errors, re-raise immediately
                raise


# ------------------------------------------------------------------
# Runner Setup - Configure ADK Runner with Services
# ------------------------------------------------------------------

# In-memory services for development (can be swapped for production services)
# - MemoryService: Stores agent memories across sessions
# - SessionService: Manages session state and user sessions
# - ArtifactService: Handles artifacts generated during agent execution
memory_service = InMemoryMemoryService()
session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

# Initialize the Runner with our orchestrated agent and services
runner = Runner(
    agent=care_coordinator_agent,
    app_name="Chronic Health Concierge Agent",
    memory_service=memory_service,
    session_service=session_service,
    artifact_service=artifact_service,
)

# ------------------------------------------------------------------
# Demo Data - User Profile and Initial State
# ------------------------------------------------------------------

# Sample user profile for demonstration purposes
# In production, this would be loaded from a user database
user_profile = {
    "name": "Alex Thompson",
    "age": 45,
    "gender": "Male",
    "height_cm": 178,
    "weight_kg": 85,
    "conditions": ["Hypertension"],
    "medications": ["Lisinopril"],
    "allergies": ["Penicillin"],
    "goals": ["Increase energy", "Improve sleep"],
}

def build_initial_state():
    """
    Build initial session state with user profile and wearable data.
    
    This function initializes the session state that will be shared across
    all agents in the workflow. It queries the database for user profile and
    combines it with the latest wearable metrics.
    
    Returns:
        dict: Initial session state dictionary
    """
    # Query user profile from database instead of hardcoded
    # Use try-except to handle cases where get_user_profile might not be defined (notebook environments)
    db_profile = None
    try:
        if 'get_user_profile' in globals():
            db_profile = get_user_profile("demo-user")
    except NameError:
        # Function not defined, will use fallback
        pass
    except Exception as e:
        # Database error, will use fallback
        print(f"Warning: Could not query user profile from database: {e}")
        pass
    
    if db_profile:
        user_profile = {
            "name": db_profile["name"],
            "age": db_profile["age"],
            "gender": db_profile["gender"],
            "height_cm": db_profile["height_cm"],
            "weight_kg": db_profile["weight_kg"],
            "conditions": db_profile["conditions"],
            "medications": db_profile["medications"],
            "allergies": db_profile["allergies"],
            "goals": db_profile["goals"],
        }
    else:
        # Fallback to hardcoded if database query fails
        user_profile = {
            "name": "Alex Thompson",
            "age": 45,
            "gender": "Male",
            "height_cm": 178,
            "weight_kg": 85,
            "conditions": ["Hypertension"],
            "medications": ["Lisinopril"],
            "allergies": ["Penicillin"],
            "goals": ["Increase energy", "Improve sleep"],
        }
    
    # Get the latest row as wearable snapshot
    # Handle case where wellness_df might not be defined (notebook environments)
    try:
        if 'wellness_df' not in globals() or wellness_df.empty:
            wearable_snapshot = {}
        else:
            snapshot = wellness_df.tail(1).to_dict(orient="records")[0]
        
        # Extract key Oura metrics
        wearable_snapshot = {
            "date": snapshot.get("date"),
            "resting_hr": snapshot.get("resting_hr"),
            "hr_average": snapshot.get("hr_average"),
            "hr_lowest": snapshot.get("hr_lowest"),
            "sleep_score": snapshot.get("sleep_score"),
            "sleep_efficiency": snapshot.get("sleep_efficiency"),
            "readiness_score": snapshot.get("readiness_score"),
            "total_sleep_duration": snapshot.get("total_sleep_duration"),
            "deep_sleep_duration": snapshot.get("deep_sleep_duration"),
            "rem_sleep_duration": snapshot.get("rem_sleep_duration"),
            "activity_balance": snapshot.get("activity_balance"),
            "hrv_average": snapshot.get("hrv_average"),
        }
        
        # Remove None/NaN values for cleaner state
        wearable_snapshot = {k: v for k, v in wearable_snapshot.items() if v is not None and pd.notna(v)}
    except NameError:
        # wellness_df not defined, use empty snapshot
        wearable_snapshot = {}
    except Exception as e:
        # Other error, use empty snapshot
        print(f"Warning: Could not load wearable data: {e}")
        wearable_snapshot = {}
    
    return {
        "user_profile": user_profile,
        "intake_answers": [
            {"question": "Name", "answer": user_profile["name"]},
            {"question": "Age", "answer": str(user_profile["age"])},
            {"question": "Existing conditions", "answer": ", ".join(user_profile["conditions"])},
            {"question": "Medications", "answer": ", ".join(user_profile["medications"])},
            {"question": "Allergies", "answer": ", ".join(user_profile["allergies"])},
            {"question": "Goals", "answer": ", ".join(user_profile["goals"])},
        ],
        "wearable_snapshot": wearable_snapshot,
        "trend_metrics": ["resting_hr", "sleep_score", "readiness_score", "hr_average"],
        # Pre-approve for demo (in production, this would come from human input via approve_recommendation)
        "approvals": {
            "care_plan": {"approved": True, "notes": "Plan looks good"},
        },
        "needs_medication_review": False,  # Set to True to trigger medication interaction review
    }


def close_all_db_connections():
    """
    Helper function to close any lingering database connections.
    Useful in notebook environments where connections can persist.
    This is an alias for unlock_database() for backward compatibility.
    """
    return unlock_database()


# ------------------------------------------------------------------
# Main Execution Function
# ------------------------------------------------------------------

async def run_capstone():
    # Track execution time and agent sequence for evaluation
    start_time = time.time()
    agent_sequence = []
    
    # Track agent execution
    agent_tracker = {}

    """
    Main execution function for the healthcare concierge workflow.
    
    This function:
    1. Initializes session state with user profile and wearable data
    2. Creates a new session for the user
    3. Processes the user's message through the agent chain
    4. Streams events (agent responses, tool calls) in real-time
    
    The workflow proceeds through all 8 agents sequentially, with each
    agent building on the work of previous agents through shared session state.
    
    NOTE: Free tier has a 10 requests/minute limit. With 8 agents, you may
    encounter rate limit errors. The retry function will automatically handle
    these, but you may need to wait between runs if using the free tier.
    """
    print(" Starting Healthcare Concierge Agent Workflow...")
    print("   This will run 8 agents sequentially (Intake → Wearable → Trend → Risk → Care → Review → Shopping → Summary)\n")
    
    # Close any lingering database connections (helps in notebook environments)
    try:
        close_all_db_connections()
    except:
        pass  # Ignore errors
    
    # Check dependencies (helpful for notebook environments)
    if not check_dependencies():
        raise NameError(
            "Missing required dependencies. "
            "In notebook environments, make sure you've run all cells that define "
            "get_user_profile, wellness_df, init_database, and seed_database."
        )
    
    # Build initial state from user profile and wearable data
    initial_state = build_initial_state()
    
    # Create a new session for this user interaction
    session = await session_service.create_session(
        app_name="Chronic Health Concierge Agent",
        user_id="demo-user",
        state=initial_state,
    )
    
    # User's natural language request
    user_prompt = (
        "Hi, I'm Alex. I've been feeling low energy. "
        "Please run intake, analyze my wearable data, trends, risk, care plan, "
        "shopping suggestions, and summarize everything."
    )
    
    # Convert user message to ADK Content format
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_prompt)],
    )
    
    # Execute agent workflow with retry logic for rate limit handling
    try:
        async for event in run_with_retry(
            runner, 
            user_id="demo-user",
            session_id=session.id,
            new_message=content,
        ):
            # Stream events as they occur for real-time feedback
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"{event.author}: {part.text}")
                    elif part.function_call:
                        print(f" {event.author} → ToolCall: {part.function_call}")
                    elif part.function_response:
                        print(f"{event.author} → ToolResponse: {part.function_response}")
        

        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Get final session state for evaluation
        final_session = await session_service.get_session(
            app_name="Chronic Health Concierge Agent",
            user_id="demo-user",
            session_id=session.id
        )
        final_state = final_session.state if final_session else {}
        
        # Save session for evaluation
        try:
            # Small delay to let any database operations complete
            await asyncio.sleep(0.1)
            
            risk_score = final_state.get("risk_score", initial_state.get("risk_score", 0))
            care_plan = final_state.get("care_plan", initial_state.get("care_plan", []))
            shopping_suggestions = final_state.get("shopping_list", initial_state.get("shopping_list", []))
            agent_sequence = ["intake_agent", "wearable_agent", "trend_agent", "risk_agent", "care_plan_agent", "human_review_agent", "shopping_agent", "final_summary_agent"]
            
            save_session_history(
                session_id=session.id,
                user_id="demo-user",
                user_prompt=user_prompt,
                final_summary=str(final_state.get("final_summary", "")),
                risk_score=risk_score,
                care_plan=care_plan,
                shopping_suggestions=shopping_suggestions,
                execution_time=execution_time,
                agent_sequence=agent_sequence
            )
            
            # Automatically log evaluation metrics (batch operation for efficiency)
            try:
                # Prepare all metrics for batch insert
                metrics_to_save = []
                
                # Overall workflow metric
                metrics_to_save.append({
                    'session_id': session.id,
                    'agent_name': 'workflow',
                    'metric_name': 'total_execution_time',
                    'metric_value': execution_time,
                    'notes': f'Total time for {len(agent_sequence)} agents'
                })
                
                # Risk score
                metrics_to_save.append({
                    'session_id': session.id,
                    'agent_name': 'risk_agent',
                    'metric_name': 'risk_score',
                    'metric_value': float(risk_score),
                    'notes': 'Calculated risk score based on intake answers'
                })
                
                # Care plan quality
                care_plan_quality = min(len(care_plan) * 20, 100) if care_plan else 0
                metrics_to_save.append({
                    'session_id': session.id,
                    'agent_name': 'care_plan_agent',
                    'metric_name': 'care_plan_quality',
                    'metric_value': care_plan_quality,
                    'notes': f'Quality score based on {len(care_plan)} care plan items'
                })
                
                # Shopping suggestions quality
                shopping_quality = 80 if shopping_suggestions and any("pubmed" in str(s).lower() or "webmd" in str(s).lower() for s in shopping_suggestions) else 40
                metrics_to_save.append({
                    'session_id': session.id,
                    'agent_name': 'shopping_agent',
                    'metric_name': 'shopping_suggestions_quality',
                    'metric_value': shopping_quality,
                    'notes': f'Quality score based on {len(shopping_suggestions)} suggestions with research links'
                })
                
                # Per-agent execution time (estimated average)
                time_per_agent = execution_time / len(agent_sequence) if agent_sequence else 0
                for agent_name in agent_sequence:
                    metrics_to_save.append({
                        'session_id': session.id,
                        'agent_name': agent_name,
                        'metric_name': 'estimated_execution_time',
                        'metric_value': time_per_agent,
                        'notes': 'Estimated time per agent (total time / number of agents)'
                    })
                
                # Save all metrics in a single batch operation
                save_evaluation_metrics_batch(metrics_to_save)
                
                print(f"\n Session saved for evaluation. Execution time: {execution_time:.2f}s")
                print(f" Evaluation metrics logged: {len(metrics_to_save)} metrics")
            except Exception as e:
                print(f"\n  Could not log evaluation metrics: {e}")
                print(f" Session saved for evaluation. Execution time: {execution_time:.2f}s")
            
            # Show how to evaluate agents
            print("\n" + "="*80)
            print(" TO EVALUATE AGENT PERFORMANCE:")
            print("="*80)
            print(f"\n# Print evaluation summary for this session:")
            print(f"print_evaluation_summary(session_id='{session.id}')")
            print(f"\n# Or get all evaluation metrics:")
            print(f"metrics = run_evaluation_query('session_metrics', ('{session.id}',))")
            print(f"\n# Or see overall agent performance:")
            print(f"print_evaluation_summary()  # Shows all sessions")
            print("\n" + "="*80 + "\n")
        except Exception as e:
            print(f"\n  Could not save session for evaluation: {e}")
        print("\n Workflow completed successfully!")
    except Exception as e:
        # Handle ClientError with fallback for notebook environments
        error_type = type(e).__name__
        is_rate_limit = False
        status_code = None
        
        # Check if it's a rate limit error (multiple ways it might be raised)
        if error_type == 'ClientError':
            status_code = getattr(e, 'status_code', None)
            is_rate_limit = status_code == 429
        elif '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e) or 'quota' in str(e).lower():
            is_rate_limit = True
        
        if is_rate_limit:
            error_message = str(e)
            is_daily_quota = '250' in error_message or 'quota exceeded' in error_message.lower() or 'free_tier_requests' in error_message.lower()
            
            if is_daily_quota:
                print("\n Daily quota exhausted - Could not complete after retries.")
                print("   Free tier limit: 250 requests/day per model")
                print("   This workflow uses 8 agents, each making multiple API calls.")
                print("\n Solutions:")
                print("   1. Wait until tomorrow (quota resets daily at midnight UTC)")
                print("   2. Upgrade to paid tier for higher limits: https://ai.google.dev/pricing")
                print("   3. Reduce number of agents or combine some agents")
                print("   4. Check your usage: https://ai.dev/usage?tab=rate-limit")
            else:
                print("\n Rate limit error - Could not complete after retries.")
                print("   Free tier limit: 10 requests/minute")
                print("   This workflow uses multiple agents, so it may hit the limit.")
                print("\n Solutions:")
                print("   1. Wait 2-3 minutes and try again")
                print("   2. Upgrade to paid tier for higher limits: https://ai.google.dev/pricing")
                print("   3. Use the optimized version (already applied) to reduce API calls")
            print(f"\n   Error: {str(e)[:300]}")
        else:
            # Re-raise non-rate-limit errors
            raise


check_dependencies()


# ------------------------------------------------------------------
# Execution Cell
# ------------------------------------------------------------------
import nest_asyncio, asyncio
nest_asyncio.apply()
await run_capstone()


print_evaluation_summary()

