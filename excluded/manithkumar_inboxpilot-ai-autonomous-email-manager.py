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


!pip install --quiet --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib google-auth
!pip install --quiet --upgrade google-genai beautifulsoup4 reportlab requests python-dateutil
!pip install --quiet dateparser


from kaggle_secrets import UserSecretsClient
import shutil, os

# Update this path if your dataset path is different
SRC = "/kaggle/input/information-new2/credentials.json"  # <- change if needed
DST = "credentials.json"

if os.path.exists(SRC):
    shutil.copy(SRC, DST)
    print("Copied credentials.json to working directory.")
else:
    raise FileNotFoundError(f"{SRC} not found. Upload your credentials.json to the Kaggle dataset.")

# Load secrets
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
APPS_SCRIPT_WEBHOOK = user_secrets.get_secret("APPS_SCRIPT_WEBHOOK")  # optional
GMAIL_DISPLAY_NAME = "Manith"
EMAIL = user_secrets.get_secret("EMAIL")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing in Kaggle Secrets!")

# expose GEMINI_API_KEY to environment for SDK
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

print("Secrets loaded (GEMINI_API_KEY present). APPS_SCRIPT_WEBHOOK is", "set" if APPS_SCRIPT_WEBHOOK else "NOT set")


# Usage:
#   - On Kaggle: set KAGGLE_MODE = True (prevents OAuth/input and uses demo data)
#   - Locally:  set KAGGLE_MODE = False (runs real Gmail OAuth flow and performs real actions)

KAGGLE_MODE = False   # <<--- Set to False when running locally (outside Kaggle) for real Gmail behavior
import os, json, re, time, base64, requests, dateparser
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders

from dateutil import parser as dateutil_parser

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Gemini client
try:
    from google import genai
except:
    try:
        import google_genai as genai
    except:
        genai = None


SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events"
]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

def gmail_calendar_auth():
    if KAGGLE_MODE:
        # Kaggle cannot run OAuth, so we skip real login
        print("âš ï¸� Kaggle Mode: Gmail & Calendar authentication disabled.")
        print("âš ï¸� Real OAuth login is only supported outside Kaggle.")
        return None, None

    # ---------- REAL MODE FOR LOCAL MACHINE ----------
    creds = None
    if os.path.exists(TOKEN_FILE):
        import pickle
        with open(TOKEN_FILE, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            # Kaggle safe redirect
            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
            auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
            print("\nğŸ”— Open this URL and paste the code:")
            print(auth_url)
            code = input("\nPaste code here: ").strip()
            flow.fetch_token(code=code)
            creds = flow.credentials
        with open(TOKEN_FILE, "wb") as f:
            import pickle
            pickle.dump(creds, f)
            print("âœ” Saved token.")

    gmail = build("gmail", "v1", credentials=creds, cache_discovery=False)
    calendar = build("calendar", "v3", credentials=creds, cache_discovery=False)
    print("âœ” Gmail & Calendar Ready")
    return gmail, calendar



def list_unread_messages(service, max_results=20):
    resp = service.users().messages().list(userId="me", labelIds=["INBOX","UNREAD"], maxResults=max_results).execute()
    return resp.get("messages", [])

def _get_text_from_part(part):
    mime = part.get("mimeType","")
    data = part.get("body",{}).get("data")
    if mime == "text/plain" and data:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    if mime == "text/html" and data:
        html = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        return BeautifulSoup(html, "lxml").get_text(separator="\n")
    for p in part.get("parts",[]) or []:
        t = _get_text_from_part(p)
        if t:
            return t
    return None

def get_message(service, msg_id):
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])
    subject = next((h["value"] for h in headers if h["name"].lower()=="subject"), "")
    from_ = next((h["value"] for h in headers if h["name"].lower()=="from"), "")
    snippet = msg.get("snippet","")
    body = _get_text_from_part(payload) or snippet
    return {"id": msg_id, "subject": subject, "from": from_, "body": body, "raw": msg}

def send_reply(gmail_service, to_address, subject, body_text, thread_id=None):
    message = MIMEText(body_text)
    message["to"] = to_address
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    payload = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id
    return gmail_service.users().messages().send(userId="me", body=payload).execute()

def mark_as_read(gmail_service, msg_id):
    gmail_service.users().messages().modify(userId="me", id=msg_id, body={"removeLabelIds":["UNREAD"]}).execute()

def archive_message(gmail_service, msg_id):
    gmail_service.users().messages().modify(userId="me", id=msg_id, body={"removeLabelIds":["INBOX"]}).execute()


GEMINI_MODEL = "gemini-2.5-flash"

def init_gemini_client():
    global genai
    if genai is None:
        raise RuntimeError("Gemini SDK not available. Install google-genai.")
    # prefer explicit api_key
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in env.")
    client = genai.Client(api_key=key)
    return client

def extract_json_from_text(text):
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        cand = m.group(0)
        try:
            return json.loads(cand)
        except:
            try:
                return json.loads(cand.replace("'", '"'))
            except:
                return None
    return None

def analyze_with_gemini(subject, body):
    client = init_gemini_client()
    prompt = f"""
You are an assistant that extracts structured information from an email. Return ONLY JSON.

Email subject:
{subject}

Email body:
{body}

Return JSON keys:
- summary: short summary
- priority: "urgent" / "normal" / "spam"
- tasks: list of {{ "task": "...", "deadline": "YYYY-MM-DD or empty" }}
- suggested_reply: short reply (professional)
"""
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = getattr(resp, "text", str(resp))
    parsed = extract_json_from_text(text)
    if parsed:
        return parsed
    return {"summary": text[:400], "priority":"normal","tasks":[],"suggested_reply":""}


def log_to_sheet_via_webhook(subject, summary, priority, tasks, suggested_reply, date_str):
    webhook = APPS_SCRIPT_WEBHOOK
    if not webhook:
        print("APPS_SCRIPT_WEBHOOK not set â€” skipping sheet log.")
        return
    payload = {
        "subject": subject,
        "summary": summary,
        "priority": priority,
        "tasks": "; ".join([f"{t.get('task')}|{t.get('deadline','')}" for t in tasks]) if tasks else "",
        "suggested_reply": suggested_reply,
        "date": date_str
    }
    try:
        r = requests.post(webhook, json=payload, timeout=10)
        if r.status_code == 200:
            print("Logged to Google Sheet.")
        else:
            print("Sheet webhook returned", r.status_code, r.text)
    except Exception as e:
        print("Failed to call webhook:", e)

def create_daily_pdf(reports, filename="daily_summary.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4
    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"Daily AI Email Summary Report â€” {datetime.now().strftime('%Y-%m-%d')}")
    y -= 30
    c.setFont("Helvetica", 10)
    for r in reports:
        if y < 120:
            c.showPage()
            y = h - 60
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, (r['subject'][:90] if r['subject'] else "(no subject)"))
        y -= 14
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Summary: {r['summary'][:200]}")
        y -= 12
        c.drawString(50, y, f"Priority: {r['priority']}  Tasks: {', '.join([t.get('task') for t in r['tasks']]) if r['tasks'] else 'None'}")
        y -= 18
    c.save()
    print("Saved PDF to", filename)
    return filename

def send_pdf_email(gmail, pdf_path, to_email):
    # build message with MIMEBase attachment (binary)
    message = MIMEMultipart()
    message["to"] = to_email
    message["subject"] = "Daily AI Email Summary Report"
    body_part = MIMEText("Attached is your daily AI-generated email summary report.", "plain")
    message.attach(body_part)

    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "pdf")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", 'attachment; filename="daily_summary.pdf"')
    message.attach(part)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    gmail.users().messages().send(userId="me", body={"raw": raw_message}).execute()
    print("ğŸ“§ PDF emailed successfully!")


def create_calendar_event(calendar_service, title, description, date_iso):
    """
    Creates a 1-hour event at 09:00 IST using date_iso (YYYY-MM-DD)
    """
    if not date_iso:
        print("â�Œ No valid date provided, skipping calendar event.")
        return None

    start = f"{date_iso}T09:00:00+05:30"
    end   = f"{date_iso}T10:00:00+05:30"

    event = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start, "timeZone": "Asia/Kolkata"},
        "end":   {"dateTime": end,   "timeZone": "Asia/Kolkata"}
    }

    created = calendar_service.events().insert(
        calendarId="primary",
        body=event
    ).execute()

    print("ğŸ“… Calendar event created:", created.get("htmlLink"))
    return created.get("htmlLink")
    
def parse_deadline_to_iso(text):
    """
    Converts human dates into ISO (YYYY-MM-DD)
    Examples supported:
      - "2025-12-01"
      - "Dec 1, 2025"
      - "December 1"
      - "tomorrow"
      - "next monday"
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # Already ISO-like (YYYY-MM-DD)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text

    # Dateparser fuzzy conversion
    try:
        dt = dateparser.parse(text, dayfirst=False, fuzzy=True)
        if dt:
            return dt.strftime("%Y-%m-%d")
    except:
        return None

    return None



def process_inbox(max_msgs=20, dry_run=True, send_pdf=True, email_pdf_to=None):
    gmail, calendar = gmail_calendar_auth()
    msgs = list_unread_messages(gmail, max_results=max_msgs)
    print("Unread:", len(msgs))
    reports = []

    for m in msgs:
        msg = get_message(gmail, m["id"])
        print("\nğŸ“Œ SUBJECT:", msg["subject"])

        analysis = analyze_with_gemini(
            msg["subject"], 
            msg["body"] or msg.get("snippet","")
        )
        print("AI:", analysis)

        # Log to sheet
        log_to_sheet_via_webhook(
            msg["subject"],
            analysis.get("summary",""),
            analysis.get("priority",""),
            analysis.get("tasks",[]),
            analysis.get("suggested_reply",""),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Handle spam
        if analysis.get("priority") == "spam":
            print("Archiving spam...")
            if not dry_run:
                try:
                    archive_message(gmail, msg["id"])
                except Exception as e:
                    print("Archive failed:", e)
            else:
                print("DRY RUN: would archive.")

        # Auto reply (non-spam)
        suggested = (analysis.get("suggested_reply") or "").strip()
        if suggested and analysis.get("priority") != "spam":
            reply_text = f"Hello,\n\n{suggested}\n\nBest regards,\nManith"
            print("âœ‰ï¸� Sending reply...")
            if not dry_run:
                try:
                    to_addr = re.search(r"<([^>]+)>", msg["from"])
                    to_addr = to_addr.group(1) if to_addr else msg["from"]
                    send_reply(
                        gmail, 
                        to_addr, 
                        "Re: " + (msg["subject"] or ""), 
                        reply_text, 
                        thread_id=msg["raw"].get("threadId")
                    )
                    print("Reply sent.")
                except Exception as e:
                    print("Reply failed:", e)
            else:
                print("DRY RUN: would send reply to", msg["from"])

        for t in analysis.get("tasks", []) or []:
            dl = t.get("deadline","")
            iso = parse_deadline_to_iso(dl)

            if iso:
                print("Scheduling event for", iso, "-", t.get("task"))
                if not dry_run:
                    try:
                        link = create_calendar_event(
                            calendar, 
                            t.get("task") or "Task", 
                            msg["body"][:800],
                            iso                    # â†� FIXED (date passed correctly)
                        )
                        print("Event created:", link)
                    except Exception as e:
                        print("Calendar event creation failed:", e)
                else:
                    print("DRY RUN: would create event on", iso)

        # Mark read
        if not dry_run:
            try:
                mark_as_read(gmail, msg["id"])
            except Exception as e:
                print("Mark read failed:", e)
        else:
            print("DRY RUN: message left unread.")

        # Report entry
        reports.append({
            "subject": msg["subject"],
            "summary": analysis.get("summary",""),
            "priority": analysis.get("priority",""),
            "tasks": analysis.get("tasks",[]),
            "suggested_reply": analysis.get("suggested_reply",""),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        time.sleep(0.5)

    # PDF
    if send_pdf and reports:
        pdf_path = create_daily_pdf(reports)
        if not dry_run and email_pdf_to:
            try:
                send_pdf_email(gmail, pdf_path, email_pdf_to)
            except Exception as e:
                print("Failed to email PDF:", e)

    print("\nâœ” Processing complete.")
    return reports



import statistics, json

def evaluate_agent(reports, ground_truth):
    """
    reports: list of dicts produced by process_inbox (subject, summary, priority, tasks, suggested_reply, date)
    ground_truth: list of dicts same length as reports with keys:
        - subject (or id)
        - expected_priority ("urgent"/"normal"/"spam")
        - expected_has_reply (True/False)
        - expected_has_task_deadline (True/False)
    Returns dict with evaluation metrics.
    """
    # match by subject (best-effort)
    gt_map = {g["subject"]: g for g in ground_truth}
    matched = []
    for r in reports:
        subj = r.get("subject", "")
        gt = gt_map.get(subj)
        if gt:
            matched.append((r, gt))

    if not matched:
        print("No matched ground-truth items provided. Provide ground_truth with matching subjects to evaluate.")
        return None

    total = len(matched)
    priority_correct = 0
    reply_correct = 0
    task_deadline_correct = 0

    for r, g in matched:
        if r.get("priority","").lower() == g.get("expected_priority","").lower():
            priority_correct += 1
        has_reply = bool((r.get("suggested_reply") or "").strip())
        if has_reply == bool(g.get("expected_has_reply")):
            reply_correct += 1
        # task deadline detection
        tasks = r.get("tasks") or []
        has_deadline = any(t.get("deadline") for t in tasks)
        if bool(has_deadline) == bool(g.get("expected_has_task_deadline")):
            task_deadline_correct += 1

    metrics = {
        "items_evaluated": total,
        "priority_accuracy": priority_correct / total,
        "reply_detection_accuracy": reply_correct / total,
        "deadline_detection_accuracy": task_deadline_correct / total
    }

    # simple composite score
    metrics["composite_score"] = statistics.mean([
        metrics["priority_accuracy"],
        metrics["reply_detection_accuracy"],
        metrics["deadline_detection_accuracy"]
    ])

    # human-readable
    metrics["summary"] = (
        f"Evaluated {total} items. "
        f"Priority acc: {metrics['priority_accuracy']:.2%}, "
        f"Reply acc: {metrics['reply_detection_accuracy']:.2%}, "
        f"Deadline acc: {metrics['deadline_detection_accuracy']:.2%}, "
        f"Composite: {metrics['composite_score']:.2%}"
    )

    return metrics

def save_evaluation_results(metrics, json_path="evaluation_results.json", pdf_path="evaluation_report.pdf"):
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    # simple PDF
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4
    y = h - 60
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Agent Evaluation Report")
    y -= 30
    c.setFont("Helvetica", 11)
    lines = [
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Items evaluated: {metrics.get('items_evaluated')}",
        f"Priority accuracy: {metrics.get('priority_accuracy'):.2%}",
        f"Reply detection accuracy: {metrics.get('reply_detection_accuracy'):.2%}",
        f"Deadline detection accuracy: {metrics.get('deadline_detection_accuracy'):.2%}",
        f"Composite score: {metrics.get('composite_score'):.2%}",
        "",
        "Summary:",
        metrics.get("summary","")
    ]
    for line in lines:
        if y < 100:
            c.showPage()
            y = h - 60
        c.drawString(50, y, line)
        y -= 16
    c.save()
    print("Saved evaluation JSON ->", json_path)
    print("Saved evaluation PDF ->", pdf_path)
    return json_path, pdf_path


if KAGGLE_MODE:
    print("âš ï¸� Kaggle Mode: Using mock email data instead of Gmail API.")
    reports = [
        {
            "subject": "Demo: Welcome to InboxPilot",
            "summary": "This is mock email content used for Kaggle evaluation.",
            "priority": "normal",
            "tasks": [{"task": "Review InboxPilot", "deadline": ""}],
            "suggested_reply": "Thank you for reaching out!",
            "date": "2025-12-01 10:00:00"
        }
    ]
else:
    # FULL REAL AGENT RUN (only works outside Kaggle)
    reports = process_inbox(max_msgs=3, dry_run=False, send_pdf=True, email_pdf_to=EMAIL)

ground_truth = [
    # Example entry format:
    # {"subject": "Interview scheduled â€” Please confirm, Dec 1, 2025", "expected_priority":"urgent", "expected_has_reply":True, "expected_has_task_deadline":True},
]

if not ground_truth:
    print("No manual ground_truth provided â€” creating demo ground_truth from AI outputs (not a real evaluation).")
    ground_truth = []
    for r in reports:
        ground_truth.append({
            "subject": r["subject"],
            "expected_priority": r["priority"],       # demo: assume AI is correct
            "expected_has_reply": bool((r["suggested_reply"] or "").strip()),
            "expected_has_task_deadline": any(t.get("deadline") for t in (r.get("tasks") or []))
        })

metrics = evaluate_agent(reports, ground_truth)
if metrics:
    save_evaluation_results(metrics, json_path="evaluation_results.json", pdf_path="evaluation_report.pdf")
    print(metrics["summary"])


# ---- Kaggle Submission File ----

import json
from datetime import datetime

submission = {
    "agent_name": "InboxPilot AI â€” Autonomous Email Manager",
    "author": "Mainth Kumar",
    "description": "An autonomous Gmail agent that reads emails, classifies priorities, extracts tasks, sends replies, schedules events, logs to Google Sheets, and generates daily PDF reports.",
    "model_used": "Gemini 2.5 Flash",
    "evaluation": {
        "priority_accuracy": metrics.get("priority_accuracy"),
        "reply_accuracy": metrics.get("reply_detection_accuracy"),
        "deadline_accuracy": metrics.get("deadline_detection_accuracy"),
        "composite_score": metrics.get("composite_score")
    },
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open("submission.json", "w") as f:
    json.dump(submission, f, indent=2)

print("âœ” submission.json created successfully!")


