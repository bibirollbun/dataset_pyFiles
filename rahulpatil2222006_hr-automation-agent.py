import os
from kaggle_secrets import UserSecretsClient

try:
    HR_PRO = UserSecretsClient().get_secret("HR_PRO")
    os.environ["HR_PRO"] = HR_PRO
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ðŸ”‘ Authentication Error: Please make sure you have added 'HR_PRO' to your Kaggle secrets. Details: {e}")




import os
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd

# ------------------------
# Config
# ------------------------
# Name of the Kaggle secret that contains the API key (user requested)
KAGGLE_SECRET_NAME = "HR_pro"

# Export folder (Kaggle standard)
EXPORT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hr_agent")

# Pandas global display / safety settings (prevents display-format warnings)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

# ------------------------
# Try Kaggle secrets (optional)
# ------------------------
try:
    from kaggle_secrets import UserSecretsClient

    _KAGGLE_SECRETS_AVAILABLE = True
except Exception:
    _KAGGLE_SECRETS_AVAILABLE = False

# Try GenAI SDK (optional). If not present, we gracefully fallback.
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel

    _GENAI_AVAILABLE = True
except Exception:
    _GENAI_AVAILABLE = False

# ------------------------
# Column schemas
# ------------------------
EMP_COLS = ["employee_id", "name", "email", "role", "manager_id", "date_joined", "status"]
ATT_COLS = ["date", "employee_id", "in_time", "out_time", "source"]
LEAVE_COLS = ["leave_id", "employee_id", "start_date", "end_date", "type", "reason", "status", "applied_on", "approved_by"]
RESUME_COLS = ["resume_id", "employee_id", "name", "text", "score", "notes"]
INTERVIEW_COLS = ["interview_id", "candidate_name", "candidate_email", "scheduled_at", "mode", "panel", "status"]

# ------------------------
# In-memory data (init)
# ------------------------
def init_dataframes() -> Dict[str, pd.DataFrame]:
    return {
        "employees": pd.DataFrame(columns=EMP_COLS),
        "attendance": pd.DataFrame(columns=ATT_COLS),
        "leaves": pd.DataFrame(columns=LEAVE_COLS),
        "resumes": pd.DataFrame(columns=RESUME_COLS),
        "interviews": pd.DataFrame(columns=INTERVIEW_COLS),
    }


DATA = init_dataframes()

# Counters
COUNTERS = {"emp": 1000, "leave": 2000, "resume": 3000, "interview": 4000}


def _next_counter(key: str) -> int:
    COUNTERS[key] += 1
    return COUNTERS[key]


# ------------------------
# Utility: configure GenAI from Kaggle secret or env var (uses KAGGLE_SECRET_NAME)
# ------------------------
def configure_genai(secret_name: str = KAGGLE_SECRET_NAME) -> Optional[str]:
    """Load API key from Kaggle secrets or environment and configure GenAI SDK if available."""
    api_key = None
    if _KAGGLE_SECRETS_AVAILABLE:
        try:
            usc = UserSecretsClient()
            api_key = usc.get_secret(secret_name)
        except Exception:
            api_key = None

    if api_key is None:
        api_key = os.environ.get(secret_name)

    if api_key and _GENAI_AVAILABLE:
        try:
            genai.configure(api_key=api_key)
            logger.info("GenAI SDK configured")
        except Exception as e:
            logger.warning("GenAI configure failed: %s", e)
    else:
        if not api_key:
            logger.info("No API key found in Kaggle secrets or env var for '%s'", secret_name)
        if not _GENAI_AVAILABLE:
            logger.info("google.generativeai SDK not installed; GenAI calls will be mocked")

    return api_key


def genai_generate(prompt: str, model: str = "gemini-1.5-mini") -> str:
    """Generate text via GenAI SDK or return a mock response when SDK not available."""
    if not _GENAI_AVAILABLE:
        return f"[genai unavailable] simulated response to: {prompt[:200]}"
    try:
        model_obj = GenerativeModel(model)
        resp = model_obj.generate_content(prompt)
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict):
            return resp.get("output", "")
        return str(resp)
    except Exception as e:
        logger.error("GenAI generate error: %s", e)
        return f"[genai error] {e}"


# ------------------------
# Data hygiene helpers
# ------------------------
def _clean_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infs with NaN and ensure no problematic dtypes for display/ops."""
    if df is None or df.empty:
        return df
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _safe_concat(existing: pd.DataFrame, new: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Concatenate while avoiding pandas FutureWarnings by dropping fully-empty rows first."""
    if new is None or new.empty:
        return existing
    new2 = new.reindex(columns=cols).copy()
    # Drop rows that are fully NA (across the subset of columns)
    new2 = new2.dropna(how="all")
    if new2.empty:
        return existing
    # Clean numeric problems
    new2 = _clean_numeric_df(new2)
    existing = _clean_numeric_df(existing)
    return pd.concat([existing, new2], ignore_index=True)


# ------------------------
# Core HR functions
# ------------------------
def add_employee(name: str, email: str, role: str, manager_id: Optional[int] = None, date_joined: Optional[date] = None) -> Dict[str, Any]:
    if date_joined is None:
        date_joined = date.today()
    eid = _next_counter("emp")
    rec = {
        "employee_id": eid,
        "name": name,
        "email": email,
        "role": role,
        "manager_id": manager_id,
        "date_joined": pd.to_datetime(date_joined).date(),
        "status": "active",
    }
    DATA["employees"] = _safe_concat(DATA["employees"], pd.DataFrame([rec]), EMP_COLS)
    logger.info("Added employee %s (%d)", name, eid)
    return rec


def upload_attendance(df: pd.DataFrame) -> None:
    """Append attendance rows safely (avoid concat warnings)."""
    if df is None or df.empty:
        logger.info("upload_attendance: empty dataframe, nothing to append")
        return
    # Normalize timestamp columns
    df2 = df.copy()
    for c in ["date", "in_time", "out_time"]:
        if c in df2.columns:
            df2[c] = pd.to_datetime(df2[c], errors="coerce")
    DATA["attendance"] = _safe_concat(DATA["attendance"], df2, ATT_COLS)
    logger.info("upload_attendance: new total rows %d", DATA["attendance"].shape[0])


def request_leave(employee_id: int, start_date: str, end_date: str, leave_type: str = "annual", reason: str = "") -> Dict[str, Any]:
    lid = _next_counter("leave")
    s = pd.to_datetime(start_date).date()
    e = pd.to_datetime(end_date).date()
    rec = {
        "leave_id": lid,
        "employee_id": employee_id,
        "start_date": s,
        "end_date": e,
        "type": leave_type,
        "reason": reason,
        "status": "pending",
        "applied_on": datetime.utcnow(),
        "approved_by": None,
    }
    DATA["leaves"] = _safe_concat(DATA["leaves"], pd.DataFrame([rec]), LEAVE_COLS)
    logger.info("request_leave: leave %d requested by emp %s", lid, employee_id)
    return rec


def approve_leave(leave_id: int, approver_id: int, approve: bool = True) -> Dict[str, Any]:
    df = DATA["leaves"]
    idxs = df.index[df["leave_id"] == leave_id].tolist()
    if not idxs:
        raise KeyError(f"Leave id {leave_id} not found")
    i = idxs[0]
    DATA["leaves"].at[i, "status"] = "approved" if approve else "rejected"
    DATA["leaves"].at[i, "approved_by"] = approver_id
    logger.info("approve_leave: leave %s set to %s by %s", leave_id, DATA["leaves"].at[i, "status"], approver_id)
    return DATA["leaves"].loc[i].to_dict()


def generate_monthly_report(year: int, month: int, export_csv: bool = False, filename: Optional[str] = None) -> pd.DataFrame:
    emp_df = _clean_numeric_df(DATA["employees"])
    att_df = _clean_numeric_df(DATA["attendance"].copy())
    leaves_df = _clean_numeric_df(DATA["leaves"])

    active_count = int(emp_df[emp_df["status"] == "active"].shape[0])

    start = pd.to_datetime(date(year, month, 1))
    if month == 12:
        end = pd.to_datetime(date(year + 1, 1, 1)) - pd.Timedelta(days=1)
    else:
        end = pd.to_datetime(date(year, month + 1, 1)) - pd.Timedelta(days=1)

    leaves_in_month = leaves_df[(pd.to_datetime(leaves_df["start_date"]) <= end) & (pd.to_datetime(leaves_df["end_date"]) >= start)]
    leave_summary = leaves_in_month.groupby("status").size().to_dict()

    # Attendance calculations (safely)
    if not att_df.empty:
        att_df["date"] = pd.to_datetime(att_df["date"], errors="coerce").dt.date
        # compute work_hours only when both in_time and out_time parse
        with pd.option_context("mode.use_inf_as_na", True):
            att_df["work_hours"] = (pd.to_datetime(att_df["out_time"], errors="coerce") - pd.to_datetime(att_df["in_time"], errors="coerce")) / pd.Timedelta(hours=1)
        att_month = att_df[(pd.to_datetime(att_df["date"]) >= start) & (pd.to_datetime(att_df["date"]) <= end)]
        avg_hours = att_month.groupby("employee_id")["work_hours"].mean().mean()
        avg_hours = float(avg_hours) if not pd.isna(avg_hours) else None
    else:
        avg_hours = None

    summary = {
        "year": year,
        "month": month,
        "active_headcount": active_count,
        "total_leave_requests": int(leaves_in_month.shape[0]),
        "leave_summary_by_status": leave_summary,
        "avg_work_hours": avg_hours,
    }
    df_out = pd.DataFrame([summary])

    if export_csv:
        if filename is None:
            filename = EXPORT_DIR / f"hr_monthly_report_{year}_{month:02d}.csv"
        else:
            filename = EXPORT_DIR / filename
        df_out.to_csv(filename, index=False)
        logger.info("generate_monthly_report: exported to %s", filename)

    return df_out


# ------------------------
# Resumes / screening (simple)
# ------------------------
def add_resume(name: str, text: str, employee_id: Optional[int] = None) -> Dict[str, Any]:
    rid = _next_counter("resume")
    rec = {"resume_id": rid, "employee_id": employee_id, "name": name, "text": text, "score": None, "notes": None}
    DATA["resumes"] = _safe_concat(DATA["resumes"], pd.DataFrame([rec]), RESUME_COLS)
    logger.info("add_resume: %s added", rid)
    return rec


def screen_resumes(resume_ids: List[int], model: str = "gemini-1.5-mini") -> pd.DataFrame:
    df = DATA["resumes"]
    for rid in resume_ids:
        idxs = df.index[df["resume_id"] == rid].tolist()
        if not idxs:
            continue
        i = idxs[0]
        text = str(df.at[i, "text"])
        if _GENAI_AVAILABLE:
            prompt = (
                "You are an expert recruiter. Score the resume 0-100 for a full-stack developer role and give a short note.\n\n"
                f"RESUME:\n{text[:8000]}"
            )
            out = genai_generate(prompt, model=model)
            # extract first integer as score
            import re
            m = re.search(r"(\d{1,3})", out)
            score = int(m.group(1)) if m else None
            notes = out
        else:
            keywords = ["python", "django", "flask", "javascript", "react", "sql", "aws"]
            lower = text.lower()
            score = sum(lower.count(k) for k in keywords) * 10
            score = max(0, min(100, score))
            notes = "heuristic scoring"

        DATA["resumes"].at[i, "score"] = score
        DATA["resumes"].at[i, "notes"] = notes

    return DATA["resumes"][DATA["resumes"]["resume_id"].isin(resume_ids)].copy()


# ------------------------
# Interview scheduling (local)
# ------------------------
def schedule_interview(candidate_name: str, candidate_email: str, scheduled_at: datetime, mode: str = "video", panel: Optional[List[int]] = None) -> Dict[str, Any]:
    iid = _next_counter("interview")
    rec = {
        "interview_id": iid,
        "candidate_name": candidate_name,
        "candidate_email": candidate_email,
        "scheduled_at": pd.to_datetime(scheduled_at),
        "mode": mode,
        "panel": panel or [],
        "status": "scheduled",
    }
    DATA["interviews"] = _safe_concat(DATA["interviews"], pd.DataFrame([rec]), INTERVIEW_COLS)
    logger.info("schedule_interview: %s scheduled", iid)
    return rec


# ------------------------
# Simple HR Q&A wrapper
# ------------------------
def hr_agent_answer(question: str, context_docs: Optional[List[str]] = None, model: str = "gemini-1.5-mini") -> str:
    facts = [f"Active employees: {DATA['employees'][DATA['employees']['status'] == 'active'].shape[0]}", f"Total leave requests: {DATA['leaves'].shape[0]}"]
    if context_docs:
        facts.extend(context_docs)
    prompt = "You are an HR assistant. Use the facts below and answer concisely.\n\nFACTS:\n" + "\n".join(facts) + "\n\nQUESTION:\n" + question + "\n\nAnswer:"
    if _GENAI_AVAILABLE:
        return genai_generate(prompt, model=model)
    # fallback simple replies
    ql = question.lower()
    if "how many" in ql or "headcount" in ql:
        return f"Active headcount is {DATA['employees'][DATA['employees']['status'] == 'active'].shape[0]}."
    if "leave" in ql and "requests" in ql:
        return f"There are {DATA['leaves'].shape[0]} leave requests."
    return "Model not available. Install SDK or provide API key via Kaggle secrets (name: HR_pro)."


# ------------------------
# Save / load utilities (CSV)
# ------------------------
def export_all(path_prefix: Optional[Path] = None) -> Dict[str, str]:
    if path_prefix is None:
        path_prefix = EXPORT_DIR
    outputs = {}
    for k, df in DATA.items():
        out_path = Path(path_prefix) / f"hr_{k}.csv"
        # ensure we convert non-serializable dtypes safely
        df_to_save = df.copy()
        # replace inf with NaN before saving
        df_to_save = df_to_save.replace([np.inf, -np.inf], np.nan)
        df_to_save.to_csv(out_path, index=False)
        outputs[k] = str(out_path)
        logger.info("export_all: exported %s rows to %s", k, out_path)
    return outputs


def load_from_csv(path_prefix: Optional[Path] = None) -> None:
    if path_prefix is None:
        path_prefix = EXPORT_DIR
    for k in DATA.keys():
        p = Path(path_prefix) / f"hr_{k}.csv"
        if p.exists():
            try:
                DATA[k] = pd.read_csv(p)
                # Basic cleaning for numeric/infinite values
                DATA[k] = DATA[k].replace([np.inf, -np.inf], np.nan)
                logger.info("load_from_csv: loaded %s from %s", k, p)
            except Exception as e:
                logger.warning("load_from_csv: failed to load %s: %s", p, e)


# ------------------------
# Example usage (safe for Kaggle cell)
# ------------------------
if __name__ == "__main__":
    # configure genai (reads from Kaggle secrets 'HR_pro' or env var 'HR_pro')
    key = configure_genai()
    print("GenAI key present and SDK installed:", bool(key) and _GENAI_AVAILABLE)

    # Seed sample data
    e1 = add_employee("Rahul Patil", "rahul@example.com", "HR Manager")
    e2 = add_employee("Anita Rao", "anita@example.com", "Software Engineer", manager_id=e1["employee_id"])
    e3 = add_employee("Jay Singh", "jay@example.com", "Data Scientist", manager_id=e1["employee_id"])

    print("Employees:\n", DATA["employees"])

    # Upload attendance safely
    att_df = pd.DataFrame(
        [
            {"date": "2025-11-01", "employee_id": e2["employee_id"], "in_time": "2025-11-01 09:05", "out_time": "2025-11-01 18:10", "source": "biometric"},
            {"date": "2025-11-01", "employee_id": e3["employee_id"], "in_time": "2025-11-01 09:15", "out_time": "2025-11-01 17:30", "source": "biometric"},
        ]
    )
    upload_attendance(att_df)

    # Leave flow
    lv = request_leave(e2["employee_id"], "2025-11-10", "2025-11-12", "annual", "family")
    print("Applied leave:", lv)
    approved = approve_leave(lv["leave_id"], approver_id=e1["employee_id"], approve=True)
    print("Approved leave:", approved)

    # Resume and screening
    r = add_resume("Jay Singh CV", "Experienced data scientist with Python, SQL, AWS, machine learning.")
    screened = screen_resumes([r["resume_id"]])
    print("Screened resume:\n", screened)

    # Schedule interview
    iv = schedule_interview("Candidate X", "x@example.com", datetime.utcnow() + timedelta(days=2), mode="video", panel=[e1["employee_id"], e3["employee_id"]])
    print("Interview scheduled:\n", iv)

    # Monthly report
    rep = generate_monthly_report(2025, 11, export_csv=True)
    print("Monthly report:\n", rep)

    # Export CSVs
    outs = export_all()
    print("Exported files:", outs)


