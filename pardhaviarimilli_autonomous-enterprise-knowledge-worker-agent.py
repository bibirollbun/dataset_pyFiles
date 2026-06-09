import pandas as pd
import numpy as np
import random
import json
import os

# --- 1. Generate sample sales CSV ---
def create_sample_sales(path='sales_sample.csv', n=300):
    dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
    products = ['A','B','C','D']
    regions = ['North', 'South', 'East', 'West']
    rows = []

    for d in dates:
        rows.append({
            'Date': d.strftime('%Y-%m-%d'),
            'Product': random.choice(products),
            'Region': random.choice(regions),
            'Revenue': int(abs(np.random.normal(2500, 900))),
            'Quantity': int(abs(np.random.poisson(5)))
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

# Create the CSV
create_sample_sales()

# --- 2. Generate meeting notes file ---
with open('meeting_notes.txt', 'w') as f:
    f.write(
        "Sprint planning meeting. Tasks: finalize data pipeline, review revenue anomalies, "
        "fix missing values. Decisions: deploy monthly sales cleaner. Owners: Ravi, Sita."
    )

# --- 3. Generate PRD file ---
with open('prd_sample.txt', 'w') as f:
    f.write(
        "Product: Smart Inventory System. Features: low-stock alerts, supplier tracking, "
        "dashboard analytics. Constraints: offline-first, mobile-friendly."
    )

# --- 4. Create memory file ---
if not os.path.exists('agent_memory.json'):
    with open('agent_memory.json', 'w') as f:
        json.dump([], f)

print("Sample data files generated successfully!")


# =============================
# STEP 3 â€” IMPORTS & HELPERS
# =============================

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Helper function to save plots
def save_fig(fig, name):
    """
    Saves a matplotlib figure as a PNG file.
    """
    fname = f"{name}.png"
    fig.savefig(fname, bbox_inches='tight')
    return fname

# 2. Initialize memory file
MEMORY_FILE = 'agent_memory.json'

# Create if missing
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, 'w') as f:
        json.dump([], f)

# 3. Function to write session memory
def append_memory(record):
    """
    Appends the results of each agent run to agent_memory.json
    """
    with open(MEMORY_FILE, 'r') as f:
        mem = json.load(f)

    mem.append(record)

    with open(MEMORY_FILE, 'w') as f:
        json.dump(mem, f, indent=2, default=str)

    print("Memory updated.")


# ---------- Quick verification / smoke test for STEP 3 ----------
import os
from datetime import datetime
import matplotlib.pyplot as plt

print("1) Check memory file exists and size:")
print("   PATH:", MEMORY_FILE)
print("   EXISTS:", os.path.exists(MEMORY_FILE))
if os.path.exists(MEMORY_FILE):
    print("   SIZE (bytes):", os.path.getsize(MEMORY_FILE))
    with open(MEMORY_FILE,'r') as f:
        mem = f.read()
    print("   CONTENT (first 300 chars):")
    print(mem[:300] if len(mem)>0 else "(empty)")

print("\n2) Test append_memory() with a small sample record:")
sample = {
    "test_ts": str(datetime.utcnow()),
    "test_note": "This is a smoke-test record from Step 3 verification."
}
append_memory(sample)

with open(MEMORY_FILE,'r') as f:
    mem_list = json.load(f)
print("   Memory now has", len(mem_list), "entries. Last entry summary:")
print(mem_list[-1])

print("\n3) Test save_fig() -> create a tiny plot and save as test_plot.png")
fig, ax = plt.subplots()
ax.plot([1,2,3],[2,3,5])
ax.set_title("Test plot")
fname = save_fig(fig, "test_plot")
plt.close(fig)
print("   Saved plot file:", fname, "Exists:", os.path.exists(fname))

print("\nâœ… STEP 3 verification complete.")


# =============================
# STEP 4 â€” INGEST TOOL + SCHEMA TOOL
# =============================

def ingest_tool(path_or_text):
    """
    Ingests CSV files or TXT files.
    Returns a dictionary: {type: 'csv' or 'text', parsed: dataframe OR text}
    """
    # If CSV file
    if isinstance(path_or_text, str) and os.path.exists(path_or_text) and path_or_text.endswith('.csv'):
        df = pd.read_csv(path_or_text)
        return {"type": "csv", "parsed": df}

    # If TXT file
    if isinstance(path_or_text, str) and os.path.exists(path_or_text) and path_or_text.endswith('.txt'):
        with open(path_or_text, "r") as f:
            text = f.read()
        return {"type": "text", "parsed": text}

    # Fallback: treat input as raw text
    return {"type": "text", "parsed": path_or_text}


def schema_tool(df):
    """
    Analyzes CSV structure:
    - column dtype
    - missing %
    - unique values
    - recommendation (keep/drop)
    """
    cols = []

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        missing_pct = float(s.isna().mean())
        unique_vals = int(s.nunique())
        recommended = "keep" if missing_pct <= 0.5 else "drop"

        cols.append({
            "name": col,
            "dtype": dtype,
            "missing_pct": round(missing_pct, 4),
            "unique": unique_vals,
            "recommended": recommended
        })

    return {
        "columns": cols,
        "rows": len(df)
    }

print("Step 4 loaded: ingest_tool + schema_tool ready.")


# =============================
# STEP 5 â€” CLEAN, EDA & VIZ TOOLS
# =============================

def clean_tool(df, auto=True):
    """
    Basic cleaning:
    - drop duplicates
    - fill numeric NaNs with median
    - fill object NaNs with 'Unknown'
    - try parsing Date columns if present
    Returns cleaned df and a list of log messages.
    """
    log = []
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        log.append(f"dropped duplicates: {dropped} rows")

    # Attempt to parse a Date-like column named 'Date'
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            log.append("parsed 'Date' column to datetime (where possible)")
        except Exception as e:
            log.append(f"date parse warning: {str(e)}")

    # Numeric imputation
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        nans = int(df[c].isna().sum())
        if nans > 0:
            med = df[c].median()
            df[c] = df[c].fillna(med)
            log.append(f"filled numeric '{c}' missing ({nans}) with median {med:.2f}")

    # Categorical imputation
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        nans = int(df[c].isna().sum())
        if nans > 0:
            df[c] = df[c].fillna('Unknown')
            log.append(f"filled text '{c}' missing ({nans}) with 'Unknown'")

    # If no columns changed, say so
    if not log:
        log.append("no cleaning actions required")

    return df, log


def eda_tool(df):
    """
    Simple EDA summary for demo:
    - shape
    - brief describe()
    - total/avg for common columns (Revenue, Quantity)
    - monthly revenue (if Date present)
    Returns a dict with results.
    """
    out = {}
    out['shape'] = df.shape
    try:
        out['describe'] = df.describe(include='all').to_dict()
    except Exception:
        out['describe'] = {}

    # Common business KPIs
    if 'Revenue' in df.columns:
        try:
            out['total_revenue'] = float(df['Revenue'].sum())
            out['avg_revenue'] = float(df['Revenue'].mean())
            out['max_revenue'] = float(df['Revenue'].max())
        except Exception:
            pass

    if 'Quantity' in df.columns:
        try:
            out['total_quantity'] = int(df['Quantity'].sum())
            out['avg_quantity'] = float(df['Quantity'].mean())
        except Exception:
            pass

    # Time series monthly revenue if Date present and parsed
    if 'Date' in df.columns:
        try:
            df2 = df.copy()
            df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
            ts = df2.dropna(subset=['Date']).groupby(df2['Date'].dt.to_period('M')) \
                    .agg({'Revenue': 'sum'}).reset_index()
            # convert Period to string key
            out['monthly_revenue'] = {str(r['Date']): float(r['Revenue']) for _, r in ts.iterrows()}
        except Exception:
            out['monthly_revenue'] = {}

    return out


def viz_tool(df, base_name='viz'):
    """
    Generates and saves common visualizations:
    - Monthly revenue line chart (if Date + Revenue)
    - Revenue by Product bar chart (if Product + Revenue)
    Returns list of saved file paths.
    """
    figs = []

    # Monthly revenue time series
    if 'Date' in df.columns and 'Revenue' in df.columns:
        try:
            df2 = df.copy()
            df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
            ts = df2.dropna(subset=['Date']).groupby(df2['Date'].dt.to_period('M'))['Revenue'].sum()
            if len(ts) > 0:
                fig, ax = plt.subplots(figsize=(8,4))
                ts.plot(ax=ax)
                ax.set_title('Monthly Revenue')
                ax.set_xlabel('Month')
                ax.set_ylabel('Revenue')
                fname = save_fig(fig, f"{base_name}_monthly_revenue")
                plt.close(fig)
                figs.append(fname)
        except Exception:
            pass

    # Revenue by Product
    if 'Product' in df.columns and 'Revenue' in df.columns:
        try:
            agg = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
            if len(agg) > 0:
                fig, ax = plt.subplots(figsize=(6,4))
                agg.plot(kind='bar', ax=ax)
                ax.set_title('Revenue by Product')
                ax.set_ylabel('Revenue')
                fname = save_fig(fig, f"{base_name}_rev_by_product")
                plt.close(fig)
                figs.append(fname)
        except Exception:
            pass

    return figs


print("Step 5 loaded: clean_tool, eda_tool, viz_tool ready.")


# =============================
# STEP 6 â€” SUMMARIZE, INSIGHT & EVAL TOOLS
# =============================

import math

def summarize_tool(text, max_sentences=4):
    """
    Very simple extractive summarizer:
    - splits on sentences and returns the first N meaningful sentences
    - returns action-items if 'Tasks' or 'Owners' appear
    (Replace with LLM calls later for better quality)
    """
    if not isinstance(text, str) or len(text.strip())==0:
        return {"summary": "", "action_items": []}

    # crude sentence split
    sents = [s.strip() for s in text.replace('\n',' ').split('.') if s.strip()]
    summary = '. '.join(sents[:max_sentences])
    # extract simple action items if present
    actions = []
    lower = text.lower()
    if 'tasks:' in lower or 'owners:' in lower or 'owners' in lower:
        # naive extraction: find phrases after 'Tasks:' or 'Owners:'
        try:
            parts = text.split('Tasks:') if 'Tasks:' in text else text.split('tasks:')
            if len(parts) > 1:
                tail = parts[1].split('.')[0].strip()
                actions.append({"task_extract": tail})
        except Exception:
            pass
    return {"summary": summary, "action_items": actions}


def insight_tool(eda_out):
    """
    Create simple business-style insights from EDA output.
    Input: dict returned by eda_tool
    Output: list of insights (dicts with insight text, impact, confidence)
    """
    insights = []
    if not isinstance(eda_out, dict):
        return insights

    # revenue insights
    tr = eda_out.get('total_revenue')
    avg_r = eda_out.get('avg_revenue')
    if tr is not None:
        if tr > 100000: 
            insights.append({"insight": f"Total revenue is high ({int(tr)}).", "impact":"High", "confidence":0.9})
        elif tr > 10000:
            insights.append({"insight": f"Revenue shows decent volume ({int(tr)}).", "impact":"Medium", "confidence":0.8})
        else:
            insights.append({"insight": f"Total revenue is low ({int(tr)}). Consider data size or business season.", "impact":"Low", "confidence":0.6})

    # trend / seasonality hint
    monthly = eda_out.get('monthly_revenue', {})
    if isinstance(monthly, dict) and len(monthly) >= 3:
        # compute simple slope over months
        try:
            values = [v for k,v in sorted(monthly.items())]
            if len(values) >= 3:
                # linear trend approx
                n = len(values)
                x = list(range(n))
                mean_x = sum(x)/n
                mean_y = sum(values)/n
                num = sum((xi-mean_x)*(yi-mean_y) for xi,yi in zip(x,values))
                den = sum((xi-mean_x)**2 for xi in x)
                slope = num / den if den != 0 else 0
                if slope > 0:
                    insights.append({"insight":"Monthly revenue trending UP.", "impact":"Medium", "confidence":0.8})
                elif slope < 0:
                    insights.append({"insight":"Monthly revenue trending DOWN.", "impact":"Medium", "confidence":0.8})
        except Exception:
            pass

    return insights


def eval_tool(result):
    """
    Produce a confidence score and warnings for the run result.
    result: dictionary produced by run_agent (partial)
    Returns: {"confidence": float(0-1), "warnings": [str,...]}
    """
    warnings = []
    confidence = 0.8

    # If EDA exists but no revenue column discovered -> lower confidence for business insights
    if 'eda' in result:
        eda = result['eda']
        if eda.get('total_revenue', None) is None:
            confidence -= 0.2
            warnings.append("No 'Revenue' column detected â€” KPI extraction limited.")
        # very small dataset
        rows = eda.get('shape', (0,0))[0] if eda.get('shape') else 0
        if rows < 10:
            confidence -= 0.15
            warnings.append("Small dataset (less than 10 rows) â€” insights may be unreliable.")
    else:
        # for text-only, rely on summary length
        summ = result.get('summary', {}).get('summary') if result.get('summary') else ""
        if not summ or len(summ.split()) < 20:
            confidence -= 0.2
            warnings.append("Short text / limited content â€” summarization might be shallow.")

    # keep confidence within [0.0, 0.99]
    confidence = max(0.0, min(0.99, confidence))

    return {"confidence": round(confidence, 2), "warnings": warnings}


print("Step 6 loaded: summarize_tool, insight_tool, eval_tool ready.")


# quick tests
print(summarize_tool(open('meeting_notes.txt').read()))
df = pd.read_csv('sales_sample.csv')
cl, log = clean_tool(df)
eda = eda_tool(cl)
print("EDA keys:", list(eda.keys()) )
print("Insights:", insight_tool(eda))
print("Eval:", eval_tool({"eda": eda}))


# =============================
# STEP 7 â€” PLANNER + AGENT ORCHESTRATOR
# =============================

def planner_tool(doc_type):
    """
    Decides which tools should run based on the file type.
    """
    if doc_type == 'csv':
        return [
            "schema",
            "clean",
            "eda",
            "viz",
            "insight",
            "eval",
            "export"
        ]
    else:  # text documents
        return [
            "summarize",
            "insight",
            "eval",
            "export"
        ]


def run_agent(input_item, recipients=None, email_mode="mock"):
    """
    The agent orchestrator:
    - read file (ingest_tool)
    - generate plan (planner_tool)
    - run tools in sequence
    - export report
    - update memory
    - send mock email (optional)
    """
    print("ğŸ”� Running agent on:", input_item)

    # Step 1 â€” Ingest file
    ing = ingest_tool(input_item)
    doc_type = ing["type"]
    parsed = ing["parsed"]

    # Step 2 â€” Planner generates workflow
    plan = planner_tool(doc_type)
    result = {"plan": plan, "type": doc_type}

    print("ğŸ§  Plan:", plan)

    # Step 3 â€” Execute according to plan
    if doc_type == "csv":
        df = parsed

        # schema
        result["schema"] = schema_tool(df)

        # clean
        df_clean, log = clean_tool(df)
        result["clean_log"] = log

        # eda
        eda = eda_tool(df_clean)
        result["eda"] = eda

        # viz
        figs = viz_tool(df_clean)
        result["figures"] = figs

        # insights
        ins = insight_tool(eda)
        result["insights"] = ins

    else:
        # summarize text
        summary = summarize_tool(parsed)
        result["summary"] = summary

        # insights (simple)
        result["insights"] = insight_tool({})

    # evaluation
    evaluation = eval_tool(result)
    result["evaluation"] = evaluation

    # Step 4 â€” Export simple Markdown report
    report_text = "# Agent Report\n\n"

    if doc_type == "csv" and "eda" in result:
        report_text += f"**Total Revenue:** {result['eda'].get('total_revenue', 'N/A')}\n\n"
        report_text += f"**Average Revenue:** {result['eda'].get('avg_revenue', 'N/A')}\n\n"

    if doc_type == "text" and "summary" in result:
        report_text += "### Summary:\n"
        report_text += result["summary"]["summary"] + "\n\n"

    report_text += "### Insights:\n"
    for i in result["insights"]:
        report_text += f"- {i['insight']} (impact: {i['impact']}, confidence: {i['confidence']})\n"

    # save report
    report_path = "agent_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    result["report"] = report_path

    # Step 5 â€” Update memory
    append_memory({
        "timestamp": str(datetime.utcnow()),
        "input_type": doc_type,
        "report": report_path,
        "insights": result["insights"]
    })

    # Step 6 â€” email (mock only in Kaggle)
    if recipients:
        if len(recipients) > 5:
            raise ValueError("â�Œ Cannot send to more than 5 recipients.")
        result["email_status"] = {
            "sent_to": recipients,
            "mode": email_mode,
            "status": "mock email logged"
        }

    print("âœ… Agent run complete.")
    return result


print("Step 7 loaded: planner_tool + run_agent ready.")


# =============================
# STEP 8 â€” EMAIL TOOLS (MOCK + SMTP WRAPPER)
# =============================

import mimetypes
import os
from email.message import EmailMessage

EMAIL_LOG = 'email_send_log.json'

# initialize log if missing
if not os.path.exists(EMAIL_LOG):
    with open(EMAIL_LOG, 'w') as f:
        json.dump([], f)

def email_tool_mock(report_path: str, recipients: list, subject: str = None, body: str = None):
    """
    Safe mock sender for Kaggle. Logs the email send to email_send_log.json
    """
    if recipients is None:
        recipients = []
    if len(recipients) > 5:
        raise ValueError("Maximum 5 recipients allowed.")

    entry = {
        "timestamp": str(datetime.utcnow()),
        "report_path": report_path,
        "recipients": recipients,
        "subject": subject or "Agent report (mock)",
        "body": (body or "")[:1000]
    }
    with open(EMAIL_LOG, 'r') as f:
        logs = json.load(f)
    logs.append(entry)
    with open(EMAIL_LOG, 'w') as f:
        json.dump(logs, f, indent=2, default=str)

    print("âœ… Mock email logged.")
    print(" To:", ", ".join(recipients) if recipients else "(no recipients)")
    print(" Report:", report_path)
    return entry

def email_tool_smtp(report_path: str, recipients: list, subject: str = None, body: str = None,
                    smtp_server: str = None, smtp_port: int = None,
                    username: str = None, password: str = None, use_tls: bool = True):
    """
    Real SMTP sender (use locally). Credentials must be provided via env vars or args.
    NOTE: Kaggle likely blocks SMTP â€” run this locally.
    """
    if recipients is None:
        recipients = []
    if len(recipients) > 5:
        raise ValueError("Maximum 5 recipients allowed.")

    smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = smtp_port or int(os.getenv('SMTP_PORT', 587))
    username = username or os.getenv('SMTP_USER')
    password = password or os.getenv('SMTP_PASS')
    if not username or not password:
        raise EnvironmentError("SMTP_USER and SMTP_PASS are required (set as env vars) for SMTP mode.")

    msg = EmailMessage()
    msg['Subject'] = subject or "Autonomous Agent Report"
    msg['From'] = username
    msg['To'] = ', '.join(recipients)
    msg.set_content(body or "Please find the attached agent report.")

    # Attach report file if exists
    if report_path and os.path.exists(report_path):
        ctype, encoding = mimetypes.guess_type(report_path)
        if ctype is None:
            ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        with open(report_path, 'rb') as fp:
            file_data = fp.read()
        msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=os.path.basename(report_path))

    # Send
    try:
        import smtplib
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
        if use_tls:
            server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        print(f"âœ… Email sent to: {', '.join(recipients)}")
        return {"status": "sent", "recipients": recipients}
    except Exception as e:
        print("â�Œ SMTP send error:", str(e))
        return {"status": "error", "error": str(e)}

def email_tool(report_path: str, recipients: list, subject: str = None, body: str = None, mode: str = None):
    """
    Wrapper: selects mode by argument or env var EMAIL_MODE ('mock' or 'smtp').
    Default: 'mock' (safe for Kaggle).
    """
    mode_use = (mode or os.getenv('EMAIL_MODE', 'mock')).lower()
    if mode_use == 'smtp':
        return email_tool_smtp(report_path, recipients, subject, body)
    else:
        return email_tool_mock(report_path, recipients, subject, body)

print("Step 8 loaded: email tools ready (default mode = mock).")


# =============================
# STEP 9 â€” DEMO RUNS + SHOW MEMORY & MOCK EMAIL
# =============================

# Demo 1: CSV pipeline
print("=== DEMO 1: CSV (sales_sample.csv) ===")
res_csv = run_agent('sales_sample.csv')   # runs full CSV pipeline
print("Plan:", res_csv['plan'])
print("EDA keys:", list(res_csv['eda'].keys()) if 'eda' in res_csv else 'N/A')
print("Figures saved:", res_csv.get('figures', []))
print("Report path:", res_csv.get('report'))

# Demo 2: Meeting notes (text)
print("\n=== DEMO 2: Meeting Notes ===")
res_notes = run_agent('meeting_notes.txt')
print("Plan:", res_notes['plan'])
print("Summary:", res_notes.get('summary',{}).get('summary','(none)'))
print("Insights:", res_notes.get('insights',[]))
print("Report path:", res_notes.get('report'))

# Demo 3: PRD
print("\n=== DEMO 3: PRD ===")
res_prd = run_agent('prd_sample.txt')
print("Plan:", res_prd['plan'])
print("Summary:", res_prd.get('summary',{}).get('summary','(none)'))
print("Insights:", res_prd.get('insights',[]))
print("Report path:", res_prd.get('report'))

# Demo 4: Show memory (last 5 entries)
print("\n=== MEMORY (last 5 entries) ===")
with open(MEMORY_FILE,'r') as f:
    mem_list = json.load(f)
for i,entry in enumerate(mem_list[-5:], start=1):
    print(f"{i}) {entry.get('timestamp')} - type: {entry.get('input_type')} - report: {entry.get('report')}")

# Demo 5: Email mock send example (max 5 recipients)
print("\n=== DEMO 5: Mock email send (5 sample addresses) ===")
sample_recipients = [
    "alice@example.com",
    "bob@example.com",
    "carol@example.com",
    "dave@example.com",
    "erin@example.com"
]
# Use the CSV report created earlier (res_csv['report'])
email_res = email_tool(res_csv.get('report'), sample_recipients, subject="Sample Agent Report", body="Please find attached.")
print("Email result:", email_res)

# Show last email log entry
print("\n=== LAST EMAIL LOG ENTRY ===")
with open(EMAIL_LOG, 'r') as f:
    logs = json.load(f)
    if logs:
        print(json.dumps(logs[-1], indent=2))
    else:
        print("(email log empty)")

print("\nâœ… DEMO RUNS COMPLETE. Check saved PNGs, agent_report.md, agent_memory.json, and email_send_log.json in your notebook files.")


# ======= Create short executive summary and a one-line pitch =======
import datetime, os, json

# Read last agent report
rep = "No agent report found."
if os.path.exists('agent_report.md'):
    with open('agent_report.md','r') as f:
        rep = f.read()

exec_summary = "# Executive Summary\n\n"
exec_summary += f"Generated: {datetime.datetime.utcnow().isoformat()} UTC\n\n"
# take first 4 non-empty paragraphs from report
parts = [p.strip() for p in rep.split("\n\n") if p.strip()]
for p in parts[:4]:
    exec_summary += p + "\n\n"

with open('executive_summary.md','w') as f:
    f.write(exec_summary)

print("Created executive_summary.md")
print("Preview (first 400 chars):\n")
print(exec_summary[:400])


# ======= Auto-generate README.md for notebook / GitHub =======

readme = """
# Autonomous Enterprise Knowledge Worker Agent â€” Pardhavi

**Short pitch:**  
An ADK-style autonomous agent that ingests CSV/TXT files, plans a multi-step workflow
(clean â†’ EDA â†’ viz â†’ summarize â†’ insights), saves memory, and exports structured 
business reports with mock email delivery (up to 5 recipients).

## How to run
1. Run the notebook top-to-bottom.
2. Use:
   run_agent('sales_sample.csv')
   run_agent('meeting_notes.txt')
   run_agent('prd_sample.txt')

## Files auto-generated
- sales_sample.csv  
- meeting_notes.txt  
- prd_sample.txt  
- agent_memory.json  
- agent_report.md  
- email_send_log.json  
- viz_*.png visualizations  

## Demonstrates
- Modular tool design (ingest, clean, eda, viz, summarize, insight, eval, export)
- Planner-driven orchestration
- Persistent memory
- Mock email sending
- Reproducible demo runs
- Clean outputs for enterprise workflows

## Notes
- Email uses *mock mode* on Kaggle.
- SMTP mode is only for local use.
- Optional LLM upgrade steps included in notebook.

## Contact
Pardhavi
"""

with open('README.md','w') as f:
    f.write(readme)

print("README.md created successfully!")


# Robust demo GIF creator: resizes/pads all images to a common size then writes GIF
import glob, os
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageOps

pngs = sorted([p for p in glob.glob("viz_*.png")])
if not pngs:
    print("No viz_*.png files found to create GIF.")
else:
    # Load images and find max width/height
    imgs = [Image.open(p).convert("RGBA") for p in pngs]
    widths, heights = zip(*(i.size for i in imgs))
    max_w, max_h = max(widths), max(heights)
    target_size = (max_w, max_h)

    frames = []
    for im in imgs:
        # Preserve aspect ratio, fit into target box
        im_thumb = ImageOps.contain(im, target_size)  # fits into target keeping aspect ratio
        # Create white background and paste centered
        bg = Image.new("RGBA", target_size, (255,255,255,255))
        paste_x = (max_w - im_thumb.width) // 2
        paste_y = (max_h - im_thumb.height) // 2
        bg.paste(im_thumb, (paste_x, paste_y), im_thumb if im_thumb.mode == 'RGBA' else None)
        # Convert to RGB (drop alpha) and to numpy array
        rgb = bg.convert("RGB")
        arr = np.array(rgb)
        frames.append(arr)

    gif_path = "demo_run.gif"
    # 1 fps is fine for static charts; change fps=1 or fps=2 as you like
    try:
        imageio.mimsave(gif_path, frames, fps=1)
        print("GIF created:", gif_path, "from files:", pngs)
    except Exception as e:
        print("Failed to write GIF:", str(e))


# ======= Final verification checklist =======
expected = [
    'sales_sample.csv',
    'meeting_notes.txt',
    'prd_sample.txt',
    'agent_memory.json',
    'agent_report.md',
    'email_send_log.json'
]

print("Checking expected files...")
for f in expected:
    print(f, ":", "FOUND" if os.path.exists(f) else "MISSING")

print("\nCheck images (viz_*.png):", sorted([p for p in os.listdir('.') if p.startswith('viz_') and p.endswith('.png')]))
print("Check demo gif:", "FOUND" if os.path.exists('demo_run.gif') else "MISSING")

print("\nIf everything is FOUND, you are ready to publish the kaggle notebook.")

