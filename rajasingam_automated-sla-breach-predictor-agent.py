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


# ============================================================
#  Imports, Folder Setup, Logging, Utilities
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

plt.style.use("default")

# ------------------------------------------------------------
# Create output directories
# ------------------------------------------------------------
BASE_DIR = Path("./")
OUTDIR = BASE_DIR / "agent_outputs"
VIS_DIR = OUTDIR / "visuals"

OUTDIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Logging Utility
# ------------------------------------------------------------
LOGFILE = OUTDIR / "agent_log.txt"

def log(msg: str):
    """Write timestamped logs to agent_log.txt and print them."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")

log("Environment initialized. Folders ready.")

# ------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------
def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    log(f"Saved JSON → {path}")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# ------------------------------------------------------------
# Random seed (for reproducibility)
# ------------------------------------------------------------
np.random.seed(42)
log("Seed set to 42.")



# ============================================================
#  Demo Ticket Dataset Generator
# ============================================================

def generate_demo_itsm_dataset(n=2000):
    """Generate a realistic synthetic ITSM ticket dataset."""
    log(f"Generating synthetic dataset with {n} tickets...")

    priorities = ["P1", "P2", "P3", "P4"]
    priority_weights = [0.05, 0.15, 0.35, 0.45]   # more P3/P4 typical

    categories = [
        "Network", "Application", "Database", "Hardware",
        "Access", "Security", "Performance", "Others"
    ]

    assignees = [
        "Alice", "Bob", "Charlie", "David",
        "Eve", "Frank", "Grace", "Heidi"
    ]

    now = datetime.now()

    data = []

    for i in range(n):
        created = now - timedelta(hours=np.random.randint(1, 240))  # last 10 days
        pr = np.random.choice(priorities, p=priority_weights)
        
        # SLA hours by priority
        sla_hours_map = {"P1": 4, "P2": 8, "P3": 24, "P4": 72}
        sla_hours = sla_hours_map[pr]
        
        # Randomly assign resolution time
        resolve_delay = np.random.randint(1, 200)
        resolved = created + timedelta(hours=resolve_delay)

        # Label: breached if resolved time > SLA
        breached = int(resolve_delay > sla_hours)

        # Some missing/noisy text
        if np.random.rand() < 0.2:
            notes = ""
        else:
            notes = f"Resolved after investigation on {np.random.choice(categories)} system."

        row = {
            "ticket_id": f"INC-{100000+i}",
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "resolved_at": resolved.strftime("%Y-%m-%d %H:%M:%S"),
            "priority": pr,
            "assignee": np.random.choice(assignees),
            "category": np.random.choice(categories),
            "sla_hours": sla_hours,
            "resolution_notes": notes,
            "breached": breached
        }
        data.append(row)

    df = pd.DataFrame(data)
    log("Synthetic ITSM dataset generated.")
    return df


# ------------------------------------------------------------
# Load dataset if exists, else create demo one
# ------------------------------------------------------------
DEFAULT_DATA_PATH = None  # If user uploads dataset, modify here

if DEFAULT_DATA_PATH and Path(DEFAULT_DATA_PATH).exists():
    df = pd.read_csv(DEFAULT_DATA_PATH)
    log(f"Loaded dataset from {DEFAULT_DATA_PATH}")
else:
    df = generate_demo_itsm_dataset(1500)

log(f"Dataset loaded. Shape = {df.shape}")
df.head(10)



# ============================================================
#  Data Quality Validation Tool
# ============================================================

def validate_ticket_dataset(df: pd.DataFrame):
    """
    Validate schema, timestamps, priority values, missing fields,
    and prepare a data quality report.
    """
    log("Running Data Quality Validation...")

    required_cols = [
        "ticket_id", "created_at", "resolved_at", "priority",
        "assignee", "category", "sla_hours", "resolution_notes", "breached"
    ]

    quality_issues = {}
    passed = True

    # --------------------------------------------------------
    # 1. Schema Check
    # --------------------------------------------------------
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        passed = False
        quality_issues["missing_columns"] = missing_cols

    # --------------------------------------------------------
    # 2. Priority Check
    # --------------------------------------------------------
    allowed_priorities = {"P1", "P2", "P3", "P4"}
    bad_pr = df[~df["priority"].isin(allowed_priorities)]
    if len(bad_pr) > 0:
        passed = False
        quality_issues["invalid_priority_values"] = bad_pr["priority"].unique().tolist()

    # --------------------------------------------------------
    # 3. Timestamp Check
    # --------------------------------------------------------
    invalid_timestamp_rows = []
    for idx, row in df.iterrows():
        try:
            created = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
            resolved = datetime.strptime(row["resolved_at"], "%Y-%m-%d %H:%M:%S")
            if resolved < created:
                invalid_timestamp_rows.append(idx)
        except Exception:
            invalid_timestamp_rows.append(idx)

    if len(invalid_timestamp_rows) > 0:
        passed = False
        quality_issues["invalid_timestamps"] = invalid_timestamp_rows[:20]

    # --------------------------------------------------------
    # 4. SLA Hours Check
    # --------------------------------------------------------
    if (df["sla_hours"] <= 0).any():
        passed = False
        quality_issues["non_positive_sla"] = int((df["sla_hours"] <= 0).sum())

    # --------------------------------------------------------
    # 5. Missing Values
    # --------------------------------------------------------
    missing_map = df.isna().sum().to_dict()
    if any(v > 0 for v in missing_map.values()):
        quality_issues["missing_values"] = missing_map

    # --------------------------------------------------------
    # 6. Empty Resolution Notes (for audit)
    # --------------------------------------------------------
    empty_notes = df["resolution_notes"].eq("").sum()
    quality_issues["empty_resolution_notes"] = int(empty_notes)

    # --------------------------------------------------------
    # Assemble report
    # --------------------------------------------------------
    quality_report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "schema_ok": missing_cols == [],
        "passed": passed,
        "issues_found": quality_issues,
        "summary": f"Dataset passed = {passed}, with {len(quality_issues)} issue categories."
    }

    # Save report
    save_json(quality_report, OUTDIR / "quality_report.json")
    log("Data Quality Validation complete.")

    return quality_report


# Run validation tool
quality_report = validate_ticket_dataset(df)
quality_report



# ============================================================
#  Feature Engineering Tool
# ============================================================

def featurize_tickets(df: pd.DataFrame):
    """
    Create SLA-relevant numerical features for ML model training & prediction.
    """
    log("Running Feature Engineering Tool...")

    df = df.copy()

    # Convert timestamps
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["resolved_at"] = pd.to_datetime(df["resolved_at"])

    # --------------------------------------------------------
    # Time-based features
    # --------------------------------------------------------
    df["ticket_age_hours"] = (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600
    df["remaining_sla"] = df["sla_hours"] - df["ticket_age_hours"]
    df["sla_slack"] = df["remaining_sla"]  # same but explicit name

    # --------------------------------------------------------
    # Workload-based features
    # --------------------------------------------------------
    # Active ticket load per assignee
    assignee_load = df.groupby("assignee")["ticket_id"].count().to_dict()
    df["assignee_ticket_load"] = df["assignee"].map(assignee_load)

    # High-priority load (P1 + P2)
    df["is_high_priority"] = df["priority"].isin(["P1", "P2"]).astype(int)
    high_load = df.groupby("assignee")["is_high_priority"].sum().to_dict()
    df["assignee_high_load"] = df["assignee"].map(high_load)

    # --------------------------------------------------------
    # Priority one-hot encoding
    # --------------------------------------------------------
    df = pd.get_dummies(df, columns=["priority"], prefix="priority")

    # --------------------------------------------------------
    # Category one-hot encoding
    # --------------------------------------------------------
    df = pd.get_dummies(df, columns=["category"], prefix="cat")

    # --------------------------------------------------------
    # Features list for ML model
    # --------------------------------------------------------
    feature_cols = [
        "ticket_age_hours",
        "remaining_sla",
        "sla_slack",
        "assignee_ticket_load",
        "assignee_high_load",
    ] + \
    [c for c in df.columns if c.startswith("priority_")] + \
    [c for c in df.columns if c.startswith("cat_")]

    X = df[feature_cols]
    y = df["breached"]

    log(f"Feature matrix shape: {X.shape}")

    return X, y, df, feature_cols


# Run feature engineering
X, y, df_feat, feature_cols = featurize_tickets(df)
X.head()



# ============================================================
# ML Model Training Tool (UPDATED: Increased max_iter)
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix
)
import pickle

MODEL_PATH = OUTDIR / "sla_model.pkl"

def train_sla_model(X, y):
    log("Training SLA Prediction Model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Increase max_iter to avoid convergence warnings
    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_type": "LogisticRegression(max_iter=2000)"
    }

    save_json(metrics, OUTDIR / "model_metrics.json")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log(f"Model saved → {MODEL_PATH}")

    log("Training complete.")
    return model, metrics

model, metrics = train_sla_model(X, y)
metrics



# ============================================================
#  SLA Breach Prediction Tool (FIXED for JSON)
# ============================================================

def timestamp_to_str(x):
    """Convert pandas Timestamp or datetime to string."""
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.strftime("%Y-%m-%d %H:%M:%S")
    return x

def convert_record_to_json_safe(record):
    """Ensure all fields in a ticket dict are JSON serializable."""
    safe = {}
    for k, v in record.items():
        if isinstance(v, (pd.Timestamp, datetime)):
            safe[k] = v.strftime("%Y-%m-%d %H:%M:%S")
        else:
            safe[k] = v
    return safe

def predict_sla_breach(model, X, df_original, top_n=20):
    """
    Use trained model to generate breach risk scores for each ticket.
    Creates predictions.csv and returns ranked high-risk tickets.
    """
    log("Running SLA Breach Prediction Tool...")

    # -----------------------------------------
    # Risk probability from model
    # -----------------------------------------
    risk_scores = model.predict_proba(X)[:, 1]

    df_pred = df_original.copy()
    df_pred["risk"] = risk_scores

    # -----------------------------------------
    # Rank by highest risk
    # -----------------------------------------
    df_pred_sorted = df_pred.sort_values(by="risk", ascending=False)

    # -----------------------------------------
    # Save full predictions
    # -----------------------------------------
    pred_path = OUTDIR / "predictions.csv"
    df_pred_sorted.to_csv(pred_path, index=False)
    log(f"Predictions saved → {pred_path}")

    # -----------------------------------------
    # Top-N risky tickets
    # -----------------------------------------
    top_risky = df_pred_sorted.head(top_n)

    # Convert to safe JSON
    top_risky_tickets = [
        convert_record_to_json_safe(rec)
        for rec in top_risky.to_dict(orient="records")
    ]

    # Save top risky tickets
    save_json(top_risky_tickets, OUTDIR / "top_risky_tickets.json")
    log(f"Top {top_n} risky tickets saved.")

    return df_pred_sorted, top_risky_tickets


# -----------------------------------------
# Run predictions
# -----------------------------------------
df_pred, top_risky_tickets = predict_sla_breach(model, X, df_feat)

# Show top 10 risky tickets
df_pred.head(10)



# ============================================================
# Explainability Agent (Gemini-Style)
# ============================================================

def explain_ticket(ticket: dict):
    """
    Generate a human-readable explanation for a ticket's SLA breach risk.
    This is a mock-Gemini reasoning format (safe for Kaggle).
    """

    priority = ticket.get("priority", "Unknown")
    risk = round(ticket.get("risk", 0), 3)
    assignee = ticket.get("assignee", "Unknown")
    category = ticket.get("category", "Unknown")

    # Extract useful numeric features
    age = ticket.get("ticket_age_hours", None)
    sla_slack = ticket.get("sla_slack", None)
    assignee_load = ticket.get("assignee_ticket_load", None)
    assignee_high = ticket.get("assignee_high_load", None)

    text = f"""
Risk Assessment for Ticket {ticket.get('ticket_id')}:
- Predicted SLA breach probability: **{risk}**
- Priority: **{priority}**
- Category: **{category}**
- Assignee: **{assignee}**

Key contributing factors:
• Ticket age is {age:.1f} hours, which is already challenging for SLA.
• Remaining SLA (slack) is {sla_slack:.1f} hours — negative or low slack increases breach risk.
• Assignee '{assignee}' currently handling {assignee_load} open tickets and {assignee_high} high-priority tasks.
• Combined workload + priority makes this ticket more likely to breach.
Summary:
This ticket requires attention due to a combination of high workload, time spent since creation, and priority level. Reassigning or expediting resolution may reduce breach probability.
""".strip()

    return {
        "ticket_id": ticket.get("ticket_id"),
        "priority": priority,
        "risk": risk,
        "assignee": assignee,
        "category": category,
        "explanation": text
    }


def run_explainability_agent(top_risky_tickets):
    """
    Generate explanations for all high-risk tickets.
    Save to explanations.json
    """
    log("Running Explainability Agent...")

    explanations = []

    for ticket in top_risky_tickets:
        exp = explain_ticket(ticket)
        explanations.append(exp)

    save_json(explanations, OUTDIR / "explanations.json")
    log("Explainability complete.")

    return explanations


# -----------------------------------------
# Run Explainability Agent
# -----------------------------------------
explanations = run_explainability_agent(top_risky_tickets)

# Show first explanation
explanations[0]



# ============================================================
# Stakeholder Alert & Escalation Agent
# ============================================================

NOTIFY_OUT = []  # Global list to store notification payloads


def build_notification_payload(ticket):
    """
    Build email/SMS/mobile notification objects for P1/P2 tickets.
    This is MOCKED for Kaggle and saves payload JSON only.
    """

    ticket_id = ticket.get("ticket_id")
    priority = ticket.get("priority")
    risk = round(ticket.get("risk", 0), 3)
    assignee = ticket.get("assignee")
    category = ticket.get("category")

    subject = f"[SLA ALERT] {priority} ticket {ticket_id} is at HIGH RISK ({risk})"
    body = f"""
URGENT SLA ALERT

Ticket ID: {ticket_id}
Priority: {priority}
Category: {category}
Assignee: {assignee}
Predicted Breach Probability: {risk}

Immediate action is recommended to prevent SLA violation.
""".strip()

    payload = {
        "ticket_id": ticket_id,
        "priority": priority,
        "risk": risk,
        "email": {
            "to": [
                "team.manager@example.com",
                "support.lead@example.com",
                "stakeholder@example.com"
            ],
            "subject": subject,
            "body": body
        },
        "sms": {
            "to": "+911234567890",
            "message": f"High SLA Risk: {ticket_id} ({priority}) Risk={risk}"
        },
        "mobile_push": {
            "topic": "sla-alerts",
            "title": "SLA Breach Risk",
            "message": f"{ticket_id} ({priority}) may breach SLA soon."
        }
    }

    return payload


def stakeholder_alert_agent(top_tickets):
    """
    Generate alert payloads for P1/P2 high-risk tickets.
    Mock integration only (Kaggle-safe).
    """
    log("Running Stakeholder Alert & Escalation Agent...")

    alerts = []
    for ticket in top_tickets:
        if ticket.get("priority") in ["P1", "P2"]:
            payload = build_notification_payload(ticket)
            alerts.append(payload)
            NOTIFY_OUT.append(payload)

    # Save notification payloads
    save_json(alerts, OUTDIR / "notifications.json")
    log(f"Alerts generated for {len(alerts)} P1/P2 high-risk tickets.")

    return alerts


# ------------------------------------------------------------
# Run alerting agent
# ------------------------------------------------------------
alerts = stakeholder_alert_agent(top_risky_tickets)

# Display first alert (if any)
alerts[0] if len(alerts) > 0 else "No P1/P2 high-risk tickets found."

"""
# REAL SERVICENOW INTEGRATION (DO NOT RUN IN KAGGLE)
import requests

def servicenow_add_comment(ticket_id, text):
    url = f"{SN_INSTANCE}/api/now/table/incident/{ticket_id}"
    headers = {"Authorization": f"Bearer {SN_TOKEN}"}
    data = {"comments": text}
    requests.patch(url, json=data, headers=headers)
"""

"""
# REAL JIRA WEBHOOK INTEGRATION
import requests

def jira_notify(payload):
    requests.post(JIRA_WEBHOOK_URL, json=payload)
"""

"""
from twilio.rest import Client

def twilio_sms(to, message):
    client = Client(TWILIO_SID, TWILIO_AUTH)
    client.messages.create(body=message, from_=TWILIO_NUMBER, to=to)
"""

"""
import smtplib
from email.mime.text import MIMEText

def send_email_smtp(to_list, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(to_list)

    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.starttls()
        s.login(SENDER_EMAIL, PASSWORD)
        s.sendmail(SENDER_EMAIL, to_list, msg.as_string())
"""




"""
# ============================================================
# REAL SMTP EMAIL CONNECTOR (COMMENTED OUT FOR KAGGLE SAFETY)
# ============================================================

import smtplib
from email.mime.text import MIMEText

# Note:
# - SENDER_EMAIL = your email address
# - PASSWORD = your app-specific password (NOT regular Gmail password)
# - SSL/2-FA required to use SMTP in production

def send_email_smtp(to_list, subject, body):
    '''
    Send an email using Gmail SMTP.
    This function is DISABLED in Kaggle because external network
    calls are not allowed, but is useful for enterprise deployment.
    '''

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(to_list)

    # Gmail SMTP server
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    with smtplib.SMTP(smtp_server, smtp_port) as smtp:
        smtp.starttls()                     # Enable TLS
        smtp.login(SENDER_EMAIL, PASSWORD)  # Authenticate
        smtp.sendmail(SENDER_EMAIL, to_list, msg.as_string())

    print("Email sent successfully.")
"""



# ============================================================
#  Perfect Reconstruction of Original + Predictions
# ============================================================

# df contains raw fields (priority, category, notes, timestamps, etc.)
# df_pred contains risk scores from prediction tool

# Select only the risk column + ticket_id
risk_df = df_pred[["ticket_id", "risk"]]

# Merge raw df with risk using ticket_id
df_pred_original = df.merge(risk_df, on="ticket_id", how="left")

log("df_pred_original successfully reconstructed with raw fields + risk.")
df_pred_original.head()



# ============================================================
# Workload & Performance Metrics Agent (FINAL FIXED)
# ============================================================
import warnings
warnings.filterwarnings("ignore")

def workload_performance_agent(df_raw, df_pred):
    """
    df_raw  = original dataset (raw columns)
    df_pred = predictions dataframe with risk scores
    This function merges the two cleanly.
    """
    log("Running Workload & Performance Agent (Fixed)...")

    # ------------------------------------------------------------
    # STEP 1 — Merge raw dataset with risk using ticket_id
    # ------------------------------------------------------------
    df = df_raw.merge(df_pred[["ticket_id", "risk"]], on="ticket_id", how="left")

    # Priority MUST exist here — guaranteed by df_raw
    if "priority" not in df.columns:
        raise Exception("ERROR: priority column missing. Workload Agent needs raw df.")

    # ------------------------------------------------------------
    # Convert timestamps
    # ------------------------------------------------------------
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["resolved_at"] = pd.to_datetime(df["resolved_at"])

    # Resolution time
    df["resolution_time"] = (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600

    # ------------------------------------------------------------
    # Build metrics per assignee
    # ------------------------------------------------------------
    report = {}

    for engineer, grp in df.groupby("assignee"):
        total = len(grp)
        high_pr = grp["priority"].isin(["P1", "P2"]).sum()
        breaches = grp["breached"].sum()
        avg_res_time = float(grp["resolution_time"].mean())
        median_res_time = float(grp["resolution_time"].median())
        sla_compliance = float((1 - breaches/total) * 100)
        avg_slack = float(grp["sla_slack"].mean()) if "sla_slack" in grp else None

        report[engineer] = {
            "total_tickets": int(total),
            "high_priority_tickets": int(high_pr),
            "breaches": int(breaches),
            "sla_compliance_percent": round(sla_compliance, 2),
            "avg_resolution_time_hours": round(avg_res_time, 2),
            "median_resolution_time_hours": round(median_res_time, 2),
            "avg_sla_slack_hours": round(avg_slack, 2) if avg_slack else None,
            "load_index": round((total + high_pr * 2) / 10, 2)
        }

    # Save the report
    save_json(report, OUTDIR / "workload_report.json")
    log("Workload report generated.")

    return pd.DataFrame(report).T


# ------------------------------------------------------------
# RUN FIXED WORKLOAD AGENT
# ------------------------------------------------------------
# df = original raw dataframe from earlier
# df_pred = prediction dataframe from Cell 7

workload_df = workload_performance_agent(df, df_pred)
workload_df



# ============================================================
#  Ticket Closure Quality Audit Agent
# ============================================================

import re

LOW_INFO_PHRASES = [
    "fixed", "done", "ok now", "working", "issue resolved",
    "resolved", "updated", "closed", "completed"
]

RCA_KEYWORDS = [
    "root cause", "rca", "due to", "because", "fix applied",
    "resolution", "impact", "analysis"
]

def evaluate_note_quality(note):
    """Returns a quality score 0–1 and issue flags."""
    if not isinstance(note, str) or note.strip() == "":
        return 0.0, ["empty_note"]

    note_l = note.lower().strip()
    issues = []
    score = 1.0

    # Very short note
    if len(note_l) < 20:
        issues.append("too_short")
        score -= 0.4

    # Low information / template text
    if any(p in note_l for p in LOW_INFO_PHRASES):
        issues.append("low_information")
        score -= 0.3

    # Missing RCA
    if not any(k in note_l for k in RCA_KEYWORDS):
        issues.append("missing_rca")
        score -= 0.3

    # Detect excessive repetition
    words = note_l.split()
    if len(set(words)) < len(words) * 0.5:
        issues.append("repetitive_text")
        score -= 0.2

    # Clamp score
    score = max(0.0, min(1.0, score))
    return score, issues


def closure_quality_agent(df_raw, df_pred):
    """
    df_raw  = original dataset (raw, includes notes)
    df_pred = model predictions including 'risk'
    """
    log("Running Ticket Closure Quality Audit Agent...")

    # Merge risk into raw df
    df = df_raw.merge(df_pred[["ticket_id", "risk"]], on="ticket_id", how="left")

    results = []

    for _, row in df.iterrows():
        note = row.get("resolution_notes", "")
        score, issues = evaluate_note_quality(note)

        results.append({
            "ticket_id": row["ticket_id"],
            "assignee": row["assignee"],
            "priority": row["priority"],
            "risk": float(row["risk"]),
            "quality_score": score,
            "issues": issues,
            "note_preview": note[:120]
        })

    # Save report
    out_path = OUTDIR / "closure_audit_report.json"
    save_json(results, out_path)
    log(f"Closure audit saved to: {out_path}")

    return pd.DataFrame(results)


# ------------------------------------------------------------
# RUN QUALITY AUDIT AGENT
# ------------------------------------------------------------
closure_df = closure_quality_agent(df, df_pred)
closure_df.head(10)



# ============================================================
# Automated SLA Breach Root-Cause Summary Agent
# (Gemini-style LLM reasoning — offline safe)
# ============================================================

def generate_llm_style_rca(ticket):
    """
    Offline-safe LLM-like reasoning.
    Produces structured RCA text.
    """
    pr = ticket["priority"]
    risk = float(ticket["risk"])
    notes = ticket["note_preview"]
    issues = ticket["issues"]

    # --------------------------------------------------------
    # Reasoning rules (mimicking an LLM)
    # --------------------------------------------------------
    possible_causes = []

    # High priority
    if pr in ["P1", "P2"]:
        possible_causes.append("High priority incident with limited response window.")

    # Missing RCA
    if "missing_rca" in issues:
        possible_causes.append("Root cause is not clearly documented in closure notes.")

    # Empty or low information notes
    if "empty_note" in issues or "low_information" in issues:
        possible_causes.append("Closure notes lack sufficient detail for audit compliance.")

    # High predicted risk
    if risk > 0.8:
        possible_causes.append("Model indicates very high probability of SLA breach.")

    # If none detected
    if not possible_causes:
        possible_causes.append("No obvious root cause detected; requires manual validation.")

    # --------------------------------------------------------
    # Mitigation plan
    # --------------------------------------------------------
    mitigation = [
        "Provide detailed RCA with cause, impact, and resolution steps.",
        "Update monitoring or alerting thresholds for similar incidents.",
        "Rebalance workload among engineers to reduce overload.",
        "Review SLA timers and escalate early when risk is identified."
    ]

    # --------------------------------------------------------
    # Build the final summary
    # --------------------------------------------------------
    summary = {
        "ticket_id": ticket["ticket_id"],
        "priority": pr,
        "risk_score": risk,
        "main_causes": possible_causes,
        "recommended_actions": mitigation,
        "note_preview": notes,
        "llm_summary": (
            f"The ticket shows a risk score of {risk:.2f}. "
            f"Potential root causes include: {', '.join(possible_causes)}. "
            f"Recommended actions: {', '.join(mitigation)}."
        )
    }

    return summary


def rca_summary_agent(closure_df):
    """
    closure_df = output from closure_quality_agent.
    Creates RCA summaries for high-risk tickets.
    """
    log("Running RCA Summary Agent...")

    high_risk_df = closure_df[closure_df["risk"] >= 0.60]

    summaries = []
    for _, ticket in high_risk_df.iterrows():
        summaries.append(generate_llm_style_rca(ticket))

    save_json(summaries, OUTDIR / "rca_summary.json")
    log("RCA summary saved → agent_outputs/rca_summary.json")

    return pd.DataFrame(summaries)


# ------------------------------------------------------------
# RUN RCA SUMMARY AGENT
# ------------------------------------------------------------
rca_df = rca_summary_agent(closure_df)
rca_df.head(10)



# ============================================================
# Supervisor Agent (Orchestration + Session Memory)
# ============================================================

import time
import traceback

SESSION_FILE = OUTDIR / "session_memory.json"
FINAL_REPORT_PATH = OUTDIR / "final_report.json"

# ------------------------------
# Gemini mock + real stub
# ------------------------------
def gemini_supervisor_mock(prompt: str):
    """Offline mock reasoning that returns a short plan and commentary."""
    return {
        "plan": [
            "1. Validate data quality",
            "2. Featurize and train model (if not already run)",
            "3. Predict risk and identify top risks",
            "4. Run explainability, workload, closure-audit, and RCA agents",
            "5. Generate stakeholder alerts for P1/P2",
            "6. Produce final report and session manifest"
        ],
        "comment": "Mock Gemini reasoning executed successfully.",
        "prompt_used": prompt
    }

# (Optional) real Gemini stub (commented - for production only)
"""
def gemini_supervisor_real(prompt: str):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    resp = model.generate_content(prompt)
    return {"plan": resp.text, "comment": "Real Gemini response"}
"""

# ------------------------------
# Supervisor function
# ------------------------------
def supervisor_orchestrator(
    df_raw,
    df_feat,
    model,
    df_pred,
    top_risky_tickets,
    explanations=None,
    workload_df=None,
    closure_df=None,
    rca_df=None,
    use_gemini=False
):
    """
    Orchestrates the full agent pipeline, keeps session memory,
    and writes final_report.json summarizing all artifacts.
    """
    log("Supervisor Agent starting orchestration...")
    start_ts = time.time()
    manifest = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "components_run": [],
        "artifacts": {},
        "gemini_reasoning": None,
        "errors": [],
    }

    try:
        # 1) Data quality (should exist)
        try:
            quality_report = load_json(OUTDIR / "quality_report.json")
            manifest["components_run"].append("data_quality")
            manifest["artifacts"]["quality_report"] = str(OUTDIR / "quality_report.json")
            log("Loaded quality_report.json")
        except Exception:
            # Run validation if missing
            log("quality_report.json missing; running validate_ticket_dataset...")
            quality_report = validate_ticket_dataset(df_raw)
            manifest["components_run"].append("data_quality_run")
            manifest["artifacts"]["quality_report"] = str(OUTDIR / "quality_report.json")

        # 2) Ensure model & predictions exist
        if model is None:
            manifest["errors"].append("model_missing")
            log("Warning: model object is None. Predictions may be invalid.")
        else:
            manifest["components_run"].append("model_available")

        if not (OUTDIR / "predictions.csv").exists():
            log("predictions.csv missing; running prediction tool...")
            p_df, p_top = predict_sla_breach(model, X, df_feat)
            manifest["artifacts"]["predictions_csv"] = str(OUTDIR / "predictions.csv")
            manifest["components_run"].append("predictions_run")
        else:
            manifest["artifacts"]["predictions_csv"] = str(OUTDIR / "predictions.csv")
            log("Found predictions.csv")

        # 3) Explanations
        if explanations is None:
            try:
                explanations = load_json(OUTDIR / "explanations.json")
                manifest["artifacts"]["explanations"] = str(OUTDIR / "explanations.json")
                manifest["components_run"].append("explanations_loaded")
                log("Loaded explanations.json")
            except Exception:
                log("explanations.json missing; running explainability agent...")
                explanations = run_explainability_agent(top_risky_tickets)
                manifest["artifacts"]["explanations"] = str(OUTDIR / "explanations.json")
                manifest["components_run"].append("explanations_run")

        # 4) Workload report
        if workload_df is None:
            try:
                workload_report = load_json(OUTDIR / "workload_report.json")
                manifest["artifacts"]["workload_report"] = str(OUTDIR / "workload_report.json")
                manifest["components_run"].append("workload_loaded")
                log("Loaded workload_report.json")
            except Exception:
                log("workload_report.json missing; running workload_performance_agent...")
                # Reconstruct df_pred if needed
                if "df_pred" not in locals():
                    # read predictions csv
                    if (OUTDIR / "predictions.csv").exists():
                        df_pred_local = pd.read_csv(OUTDIR / "predictions.csv")
                    else:
                        df_pred_local = df_pred
                else:
                    df_pred_local = df_pred
                workload_df = workload_performance_agent(df_raw.merge(df_pred_local[["ticket_id", "risk"]], on="ticket_id", how="left"))
                manifest["artifacts"]["workload_report"] = str(OUTDIR / "workload_report.json")
                manifest["components_run"].append("workload_run")

        # 5) Closure audit
        if closure_df is None:
            if (OUTDIR / "closure_audit_report.json").exists():
                closure_df = pd.DataFrame(load_json(OUTDIR / "closure_audit_report.json"))
                manifest["artifacts"]["closure_audit_report"] = str(OUTDIR / "closure_audit_report.json")
                manifest["components_run"].append("closure_loaded")
                log("Loaded closure_audit_report.json")
            else:
                log("closure audit missing; running closure_quality_agent...")
                closure_df = closure_quality_agent(df_raw, pd.read_csv(OUTDIR / "predictions.csv"))
                manifest["artifacts"]["closure_audit_report"] = str(OUTDIR / "closure_audit_report.json")
                manifest["components_run"].append("closure_run")

        # 6) RCA summaries
        if rca_df is None:
            if (OUTDIR / "rca_summary.json").exists():
                rca_df = pd.DataFrame(load_json(OUTDIR / "rca_summary.json"))
                manifest["artifacts"]["rca_summary"] = str(OUTDIR / "rca_summary.json")
                manifest["components_run"].append("rca_loaded")
            else:
                log("rca summary missing; running rca_summary_agent...")
                # load closure_df from previous
                if closure_df is None:
                    closure_df = closure_quality_agent(df_raw, pd.read_csv(OUTDIR / "predictions.csv"))
                rca_df = rca_summary_agent(closure_df)
                manifest["artifacts"]["rca_summary"] = str(OUTDIR / "rca_summary.json")
                manifest["components_run"].append("rca_run")

        # 7) Alerts (generate if notifications not present)
        if (OUTDIR / "notifications.json").exists():
            manifest["artifacts"]["notifications"] = str(OUTDIR / "notifications.json")
            manifest["components_run"].append("alerts_loaded")
            log("Found notifications.json")
        else:
            log("Generating alerts for P1/P2 high-risk tickets...")
            alerts_local = stakeholder_alert_agent(top_risky_tickets)
            manifest["artifacts"]["notifications"] = str(OUTDIR / "notifications.json")
            manifest["components_run"].append("alerts_run")

        # 8) Gemini reasoning (mock or real)
        prompt = f"Summarize the run: risk_count={len(top_risky_tickets)}; highest_risk={round(top_risky_tickets[0]['risk'],3) if top_risky_tickets else 'NA'}"
        if use_gemini:
            # reasoning = gemini_supervisor_real(prompt)
            reasoning = {"comment": "Real Gemini not enabled in Kaggle demo."}
        else:
            reasoning = gemini_supervisor_mock(prompt)
        manifest["gemini_reasoning"] = reasoning

        # 9) Build final report
        final_report = {
            "run_time_seconds": round(time.time() - start_ts, 2),
            "summary": {
                "n_tickets": int(quality_report.get("n_rows", len(df_raw))),
                "top_risk_count": len(top_risky_tickets),
                "highest_risk_ticket": top_risky_tickets[0]["ticket_id"] if top_risky_tickets else None
            },
            "artifacts": manifest["artifacts"],
            "components": manifest["components_run"],
            "metrics": load_json(OUTDIR / "model_metrics.json") if (OUTDIR / "model_metrics.json").exists() else {},
            "workload_preview": workload_df.head(10).to_dict() if workload_df is not None else {},
            "top_risky_tickets": top_risky_tickets,
            "explanations": explanations,
            "closure_audit_summary": closure_df.head(10).to_dict() if closure_df is not None else {},
            "rca_summary_preview": rca_df.head(10).to_dict() if rca_df is not None else {},
            "notifications_preview": (load_json(OUTDIR / "notifications.json")[:10] if (OUTDIR / "notifications.json").exists() else [])
        }

        save_json(final_report, FINAL_REPORT_PATH)
        log(f"Final report saved → {FINAL_REPORT_PATH}")

        manifest["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        manifest["status"] = "success"
        save_json(manifest, SESSION_FILE)
        log(f"Session manifest saved → {SESSION_FILE}")

        return final_report

    except Exception as e:
        trace = traceback.format_exc()
        manifest["errors"].append(str(e))
        manifest["traceback"] = trace
        manifest["status"] = "failed"
        save_json(manifest, SESSION_FILE)
        log("Supervisor encountered an error; session saved for debugging.")
        log(trace)
        raise e


# ------------------------------
# Run the supervisor now (use current variables)
# ------------------------------
final_report = supervisor_orchestrator(
    df_raw=df,
    df_feat=df_feat,
    model=model,
    df_pred=df_pred,
    top_risky_tickets=top_risky_tickets,
    explanations=explanations,
    workload_df=workload_df,
    closure_df=closure_df,
    rca_df=rca_df,
    use_gemini=False  # set True in production with real Gemini key
)

# Show final report keys
list(final_report.keys())








