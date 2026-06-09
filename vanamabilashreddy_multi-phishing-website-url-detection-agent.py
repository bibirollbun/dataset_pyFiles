# Cell 1: Environment setup and imports (deterministic, Kaggle-safe)

import os
import sys
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Scikit-learn core components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Utility
warnings.filterwarnings("ignore")

# Deterministic seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Basic environment info (useful for debugging)
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)

# Optional: check scikit-learn version for reproducibility
import sklearn
print("scikit-learn:", sklearn.__version__)



# Cell 2: Configuration and deterministic utilities

# Configuration dictionary for model and training setup
CONFIG = {
    "test_size": 0.25,        # 25% of data for testing
    "random_state": 42,       # reproducibility
    "n_estimators": 250,      # number of trees in RandomForest
    "max_depth": None,        # allow trees to expand fully
    "n_jobs": -1,             # use all CPU cores
    "class_weight": "balanced", # handle class imbalance
    "legit_threshold": 0.70   # stricter threshold for Legitimate classification
}

def set_seeds(seed: int = 42):
    """Set seeds for reproducibility across numpy and random."""
    np.random.seed(seed)
    random.seed(seed)

# Apply seed setting
set_seeds(CONFIG["random_state"])

# Print configuration for verification
print("Configuration:")
print(json.dumps(CONFIG, indent=2))



# Cell 3: Data loading with safe fallback

import re

# Primary dataset path (confirmed)
DATASET_PATH = "/kaggle/input/phishing-detection-urls/phishing_site_urls.csv"

def build_synthetic_dataset(n: int = 2000) -> pd.DataFrame:
    """Build a synthetic dataset with phishing and legitimate URLs."""
    legit_domains = [
        "www.microsoft.com", "www.google.com", "www.python.org",
        "www.kaggle.com", "www.wikipedia.org", "www.mozilla.org",
        "aiems.edu.in", "iitd.ac.in", "nic.in", "india.gov.in"
    ]
    phishing_bases = [
        "account-secure-update", "verify-info", "login-confirm",
        "billing-alert", "crypto-airdrop", "banking-support"
    ]
    tlds = ["com", "net", "org", "xyz", "top", "io"]

    labels = np.random.choice([0, 1], size=n, p=[0.55, 0.45])  # 0=Phishing, 1=Legitimate
    urls = []
    for label in labels:
        if label == 1:  # Legitimate
            d = random.choice(legit_domains)
            paths = ["", "docs", "support", "download", "about", "en-us"]
            scheme = "https"
            path = random.choice(paths)
            url = f"{scheme}://{d}/{path}".rstrip("/")
        else:  # Phishing
            base = random.choice(phishing_bases)
            scheme = random.choice(["http", "https"])
            obf = random.choice(["-", "--", "@", ""])
            sub1 = random.choice(["secure", "update", "login", "confirm", "billing"])
            sub2 = random.choice(["verify", "account", "client", "user", "portal"])
            domain_core = f"{base}{obf}{random.choice(['service','support','center','help'])}"
            tld = random.choice(tlds)
            host = f"{sub1}.{sub2}.{domain_core}.{tld}"
            tail = random.choice(["/login", "/confirm", "/secure-update", "/verify", "/account"])
            url = f"{scheme}://{host}{tail}"
        urls.append(url)
    return pd.DataFrame({"URL": urls, "Label": labels})

def load_data() -> pd.DataFrame:
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded dataset from: {DATASET_PATH} | shape={df.shape}")
        # Normalize column names
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        # Expect URL and Label columns
        if "URL" in df.columns and "Label" in df.columns:
            return df[["URL", "Label"]]
        elif "url" in df.columns and "label" in df.columns:
            df = df.rename(columns={"url": "URL", "label": "Label"})
            return df[["URL", "Label"]]
        else:
            print("Dataset missing expected columns, using synthetic fallback.")
            return build_synthetic_dataset()
    else:
        print("Dataset not found, using synthetic fallback.")
        return build_synthetic_dataset()

# Load dataset
df = load_data()
print("Dataset shape:", df.shape)
df.head()



# Cell 4: URL Feature Engineering

from urllib.parse import urlparse

# Suspicious keywords often found in phishing URLs
SUSPICIOUS_KEYWORDS = [
    "verify", "update", "secure", "login", "confirm",
    "billing", "account", "crypto", "airdrop", "bank",
    "gift", "free", "reset", "unlock", "expired"
]

# Trusted domain endings (whitelist)
TRUSTED_DOMAINS = [
    ".edu", ".edu.in", ".ac.in", ".gov", ".gov.in", ".org", ".org.in",
    ".mil", ".int", ".nic.in", ".sch.uk", ".ac.uk"
]

def extract_url_features(url: str) -> dict:
    """Extract numerical and categorical features from a URL."""
    u = str(url or "").strip()
    parsed = urlparse(u if re.match(r"^\w+://", u) else f"http://{u}")
    scheme = parsed.scheme or ""
    host = parsed.netloc.lower() if parsed.netloc else ""
    path = parsed.path or ""
    query = parsed.query or ""

    length = len(u)
    parts = host.split(".")
    tld = parts[-1] if len(parts) >= 2 else ""
    subdomains = parts[:-2] if len(parts) >= 2 else []
    num_subdomains = len([s for s in subdomains if s])

    at_count = u.count("@")
    dash_count = u.count("-")
    digit_ratio = sum(c.isdigit() for c in u) / max(1, len(u))
    special_ratio = sum(not c.isalnum() for c in u) / max(1, len(u))
    keyword_hits = sum(kw in u.lower() for kw in SUSPICIOUS_KEYWORDS)
    uses_https = 1 if scheme.lower() == "https" else 0
    tld_len = len(tld)
    path_depth = len([p for p in path.split("/") if p])
    query_params = len([q for q in query.split("&") if q])

    # Extra phishing signals
    has_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host) else 0
    contains_login = 1 if "login" in path.lower() else 0

    # Shannon entropy (captures randomness/obfuscation)
    if length > 0:
        freqs = [u.count(c)/length for c in set(u)]
        entropy = -sum(p*np.log2(p) for p in freqs if p > 0)
    else:
        entropy = 0.0

    return {
        "length": length,
        "num_subdomains": num_subdomains,
        "at_count": at_count,
        "dash_count": dash_count,
        "digit_ratio": digit_ratio,
        "special_ratio": special_ratio,
        "keyword_hits": keyword_hits,
        "uses_https": uses_https,
        "tld": tld,
        "tld_len": tld_len,
        "path_depth": path_depth,
        "query_params": query_params,
        "host": host,
        "has_ip": has_ip,
        "contains_login": contains_login,
        "entropy": entropy,
    }

def featurize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a dataframe of URLs into feature dataframe."""
    features = [extract_url_features(u) for u in df["URL"].astype(str).fillna("")]
    fdf = pd.DataFrame(features)
    fdf["Label"] = df["Label"].astype(int).values
    return fdf

# Apply feature extraction
fdf = featurize_dataframe(df)
print("Feature dataframe shape:", fdf.shape)
fdf.head()



# Cell 5: Train/Test Split and Pipeline Setup

# Separate features and labels
X = fdf.drop(columns=["Label"])
y = fdf["Label"].values

# Identify numeric and categorical columns
numeric_cols = [
    "length", "num_subdomains", "at_count", "dash_count",
    "digit_ratio", "special_ratio", "keyword_hits",
    "uses_https", "tld_len", "path_depth", "query_params",
    "has_ip", "contains_login", "entropy"
]
categorical_cols = ["tld", "host"]

# Preprocessing: numeric passthrough, categorical encoded
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
    ],
    remainder="drop",
)

# RandomForest classifier
clf = RandomForestClassifier(
    n_estimators=CONFIG["n_estimators"],
    max_depth=CONFIG["max_depth"],
    random_state=CONFIG["random_state"],
    n_jobs=CONFIG["n_jobs"],
    class_weight=CONFIG["class_weight"],
)

# Full pipeline
pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_state"],
    stratify=y
)

# Fit pipeline
pipe.fit(X_train, y_train)

print("Train/Test sizes:", X_train.shape, X_test.shape)



# Cell 6: Model Evaluation and Feature Importance

# Predictions on test set
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]  # probability of class 1 (Legitimate)

# Classification report
print("Classification Report (1 = Legitimate, 0 = Phishing):")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC AUC score
try:
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC: {auc:.4f}")
except Exception as e:
    print("ROC AUC could not be computed:", e)

# Feature importance from RandomForest
rf = pipe.named_steps["clf"]
imp = pd.DataFrame({
    "Feature": numeric_cols + categorical_cols,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 12 important features:")
print(imp.head(12))



# Cell 7: Save Pipeline for Gradio Integration

import joblib

# Save the trained pipeline
joblib.dump(pipe, "/kaggle/working/phish_url_pipeline.pkl")

# Save the feature order (important for consistent preprocessing later)
joblib.dump(list(X.columns), "/kaggle/working/feature_order.pkl")

print("âœ… Pipeline and feature order saved successfully!")
print("Files created:")
print("- /kaggle/working/phish_url_pipeline.pkl")
print("- /kaggle/working/feature_order.pkl")



# Cell 8: Multi-Agent Orchestration Components

from dataclasses import dataclass, field
from typing import Any, Dict, List
from urllib.parse import urlparse

# --- Session & Memory ---
@dataclass
class InMemorySessionService:
    session_id: str
    state: Dict[str, Any] = field(default_factory=dict)
    memory_bank: List[Dict[str, Any]] = field(default_factory=list)

    def set(self, key: str, value: Any):
        self.state[key] = value

    def get(self, key: str, default=None) -> Any:
        return self.state.get(key, default)

    def append_memory(self, record: Dict[str, Any]):
        self.memory_bank.append(record)

# --- Logger & Metrics ---
class Logger:
    def __init__(self):
        self.logs = []
        self.metrics = {"requests": 0, "invalid": 0, "phishing": 0, "legitimate": 0, "errors": 0}
    def info(self, msg: str):
        self.logs.append(("INFO", msg))
    def error(self, msg: str):
        self.logs.append(("ERROR", msg))
        self.metrics["errors"] += 1

# --- Validator ---
def is_valid_url(u: str) -> bool:
    return bool(re.match(r"^(http|https)://[^\s]+$", (u or "").strip()))

# --- Trusted domain check ---
def is_trusted_domain(host: str) -> bool:
    return any(host.endswith(t) for t in TRUSTED_DOMAINS)

# --- Reasoner ---
def reason_from_features(feats: Dict[str, Any]) -> List[str]:
    reasons = []
    if feats["keyword_hits"] > 0: reasons.append("Suspicious keyword detected")
    if feats["at_count"] > 0: reasons.append("Contains '@' symbol")
    if feats["dash_count"] > 3: reasons.append("Too many dashes")
    if feats["num_subdomains"] > 3: reasons.append("Too many subdomains")
    if feats["has_ip"] == 1: reasons.append("IP-based host")
    if feats["contains_login"] == 1: reasons.append("Login path detected")
    if feats["entropy"] > 4.0: reasons.append("High randomness/entropy")
    if not reasons: reasons.append("No suspicious patterns")
    return reasons

# --- Multi-Agent Controller ---
@dataclass
class MultiAgentController:
    pipe: Pipeline
    legit_threshold: float
    logger: Logger
    session: InMemorySessionService

    def process(self, urls: List[str]) -> pd.DataFrame:
        self.logger.metrics["requests"] += len(urls)
        results = []
        for u in urls:
            if not is_valid_url(u):
                self.logger.metrics["invalid"] += 1
                results.append({"URL": u, "Prediction": "Invalid", "Prob_legit": 0.0, "Reasons": "Invalid or incomplete URL"})
                continue

            parsed = urlparse(u)
            host = parsed.netloc.lower()

            # Whitelist trusted domains
            if is_trusted_domain(host):
                results.append({"URL": u, "Prediction": "Legitimate", "Prob_legit": 0.99, "Reasons": "Trusted domain whitelisted"})
                self.logger.metrics["legitimate"] += 1
                self.session.append_memory({"url": u, "pred": "Legitimate", "prob": 0.99})
                continue

            # Featurizer + Inference
            feats = extract_url_features(u)
            X = pd.DataFrame([feats])
            prob_legit = float(self.pipe.predict_proba(X)[0][1])
            pred = "Legitimate" if prob_legit >= self.legit_threshold else "Phishing"
            reasons = reason_from_features(feats)

            results.append({"URL": u, "Prediction": pred, "Prob_legit": prob_legit, "Reasons": ", ".join(reasons)})
            self.logger.metrics[pred.lower()] += 1
            self.session.append_memory({"url": u, "pred": pred, "prob": prob_legit, "reasons": reasons})
        return pd.DataFrame(results)

# --- Instantiate controller ---
logger = Logger()
session = InMemorySessionService(session_id="session-001")
controller = MultiAgentController(pipe=pipe, legit_threshold=CONFIG["legit_threshold"], logger=logger, session=session)

print("âœ… MultiAgentController ready. Agents: validator, featurizer, inference, reasoner, logger, memory.")



# Cell 9: Loop Agent (retry invalids) + Evaluation Stub

def loop_retry_invalid(urls: List[str]) -> List[str]:
    """
    Loop agent: If a URL is invalid (missing scheme), auto-add http:// and retry.
    """
    fixed = []
    for u in urls:
        uu = (u or "").strip()
        if not is_valid_url(uu) and uu:
            # If scheme missing, prepend http://
            if re.match(r"^\w+://", uu) is None:
                fixed.append(f"http://{uu}")
            else:
                fixed.append(uu)
        else:
            fixed.append(uu)
    return fixed

def evaluate_agent(df_results: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluation stub: Summarize agent performance on a batch of URLs.
    """
    legit = (df_results["Prediction"] == "Legitimate").sum()
    phish = (df_results["Prediction"] == "Phishing").sum()
    invalid = (df_results["Prediction"] == "Invalid").sum()
    mean_prob_legit = df_results["Prob_legit"].mean() if "Prob_legit" in df_results.columns else 0.0
    return {
        "legitimate": legit,
        "phishing": phish,
        "invalid": invalid,
        "mean_prob_legit": mean_prob_legit
    }

print("âœ… Loop agent and evaluation utilities ready.")



# Cell 10: Quick Smoke Test for Controller

# Example URLs to test the pipeline + agents
test_urls = [
    "https://google.com",                  # Legitimate
    "http://g00gle.com",                   # Phishing lookalike
    "aiems.edu.in",                        # Trusted domain (missing scheme)
    "https://aiems.edu.in/",               # Trusted domain with scheme
    "http://192.168.0.1/login",            # IP-based host
    "http://secure.verify.account-center.xyz/login",  # Phishing style
    "http://nic.in",                       # Trusted government domain
    "https://india.gov.in",                # Trusted government domain
    "http://paypal-update.xyz"             # Phishing
]

# Retry invalids by auto-fixing scheme (loop agent)
fixed_urls = loop_retry_invalid(test_urls)

# Process with controller
df_results = controller.process(fixed_urls)

# Display results
print("Agent Results:")
print(df_results)

# Metrics summary
metrics = evaluate_agent(df_results)
print("\nAgent metrics:", metrics)

# Logger metrics
print("Logger metrics:", logger.metrics)

# Session memory samples
print("\nSession memory samples (first 2):")
print(session.memory_bank[:2])



# Cell 11: Save Controller Outputs

# Save the most recent agent results (from Cell 10) to CSV
output_path = "/kaggle/working/agent_results.csv"
df_results.to_csv(output_path, index=False)

print(f"âœ… Agent results saved to: {output_path}")
print("Preview of saved file:")
print(df_results.head())



# Cell 12: Observability â€” Logging, Tracing, Metrics Snapshot

# Show the last 10 log entries
print("Logs (last 10):")
for lvl, msg in logger.logs[-10:]:
    print(f"[{lvl}] {msg}")

# Show metrics summary
print("\nMetrics snapshot:")
for k, v in logger.metrics.items():
    print(f"- {k}: {v}")

# Show session state keys
print("\nSession state keys:", list(session.state.keys()))

# Show memory bank size and a sample
print("Memory bank size:", len(session.memory_bank))
if session.memory_bank:
    print("Sample memory record:", session.memory_bank[0])



# Cell 13: Agent Deployment Note + A2A Protocol Stub

from dataclasses import dataclass

# In this notebook, deployment is local (controller variable).
# A2A (Agent-to-Agent) protocol stub: messages passed via Python dicts.

@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: Dict[str, Any]

def a2a_send(message: AgentMessage) -> Dict[str, Any]:
    """
    Simple A2A router: for demo, send to controller pipeline directly.
    """
    if message.recipient == "controller":
        urls = message.content.get("urls", [])
        urls = loop_retry_invalid(urls)  # loop agent fixes invalids
        return {"results": controller.process(urls).to_dict(orient="records")}
    return {"error": "unknown recipient"}

# Example A2A send
resp = a2a_send(
    AgentMessage(
        sender="ui",
        recipient="controller",
        content={"urls": ["g00gle.com", "https://india.gov.in"]}
    )
)

print("âœ… A2A response sample:")
print(resp)



# Cell 14: Batch Evaluation Utility

def batch_evaluate(urls: List[str], save_path: str = "/kaggle/working/batch_results.csv") -> Dict[str, Any]:
    """
    Run a batch of URLs through the controller, save results, and return summary metrics.
    """
    # Normalize URLs with loop agent
    urls_fixed = loop_retry_invalid(urls)

    # Process with controller
    df_batch = controller.process(urls_fixed)

    # Save results to CSV
    df_batch.to_csv(save_path, index=False)

    # Compute metrics
    metrics = evaluate_agent(df_batch)

    print(f"âœ… Batch evaluation complete. Results saved to {save_path}")
    print("Summary metrics:", metrics)
    return {"results": df_batch, "metrics": metrics}

# --- Example batch run ---
sample_urls = [
    "https://microsoft.com",
    "http://secure-update-login.xyz",
    "nic.in",
    "http://192.168.1.1/account",
    "https://paypal.com",
    "http://crypto-airdrop.top/verify",
    "aiems.edu.in",
    "http://banking-support-confirm.net/login"
]

batch_output = batch_evaluate(sample_urls)
batch_output["results"].head()



# Cell 15: Visual Analytics â€” Confusion Matrix + Probability Distribution

import seaborn as sns
import matplotlib.pyplot as plt

# --- Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Phishing","Legitimate"], yticklabels=["Phishing","Legitimate"])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Probability Distribution Plot ---
plt.figure(figsize=(6,4))
sns.histplot(y_prob[y_test==1], color="green", label="Legitimate", kde=True, stat="density", bins=20)
sns.histplot(y_prob[y_test==0], color="red", label="Phishing", kde=True, stat="density", bins=20)
plt.title("Probability Distribution of Legitimate vs Phishing")
plt.xlabel("Predicted Probability of Legitimate")
plt.ylabel("Density")
plt.legend()
plt.show()



# Cell 16: Threshold Tuning + ROC Curve Visualization

from sklearn.metrics import roc_curve, auc

def evaluate_thresholds(y_true, y_prob, thresholds=[0.3, 0.5, 0.7, 0.9]):
    """
    Evaluate classification metrics at different probability thresholds.
    """
    results = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        results.append({"Threshold": t, "Accuracy": acc, "Precision": prec, "Recall": rec})
    return pd.DataFrame(results)

# Evaluate thresholds
threshold_results = evaluate_thresholds(y_test, y_prob, thresholds=[0.3, 0.5, 0.7, 0.9])
print("Threshold tuning results:")
print(threshold_results)

# --- ROC Curve Visualization ---
fpr, tpr, thres = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], color="gray", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()



# Cell 17: Interactive Threshold Slider + Live Evaluation

import gradio as gr
import seaborn as sns
import matplotlib.pyplot as plt

def threshold_eval(threshold: float):
    """
    Evaluate model performance at a given threshold and return metrics + confusion matrix plot.
    """
    preds = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Phishing","Legitimate"],
                yticklabels=["Phishing","Legitimate"], ax=ax)
    ax.set_title(f"Confusion Matrix (Threshold={threshold:.2f})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()

    metrics_html = f"""
    <h4>Metrics at Threshold = {threshold:.2f}</h4>
    <ul>
      <li><b>Accuracy:</b> {acc:.3f}</li>
      <li><b>Precision:</b> {prec:.3f}</li>
      <li><b>Recall:</b> {rec:.3f}</li>
    </ul>
    """
    return metrics_html, fig

with gr.Blocks() as threshold_ui:
    gr.Markdown("### âš–ï¸� Threshold Tuning Dashboard\nAdjust the slider to change classification threshold.")
    threshold_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Threshold")
    metrics_out = gr.HTML()
    plot_out = gr.Plot()
    threshold_slider.change(fn=threshold_eval, inputs=threshold_slider, outputs=[metrics_out, plot_out])

threshold_ui.launch(inline=True)



# Cell 18: Export Metrics + Agent Summary Report

import json

def export_agent_summary(save_path: str = "/kaggle/working/agent_summary.json"):
    """
    Export a summary report of agent performance, metrics, and memory.
    """
    summary = {
        "logger_metrics": logger.metrics,
        "last_results": df_results.to_dict(orient="records") if 'df_results' in globals() else [],
        "session_state": session.state,
        "memory_bank_size": len(session.memory_bank),
        "memory_sample": session.memory_bank[:3],  # first 3 records
    }

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"âœ… Agent summary exported to {save_path}")
    return summary

# Run export
agent_summary = export_agent_summary()
print("Summary preview:")
print(json.dumps(agent_summary, indent=2)[:800], "...")  # preview first 800 chars



# Cell 19: Gradio Agent (Interactive UI)

!pip install gradio -q

import gradio as gr
import matplotlib.pyplot as plt

# Reload trained pipeline
pipe = joblib.load("/kaggle/working/phish_url_pipeline.pkl")

# --- Gradio handler ---
def gr_predict(urls_text: str):
    urls = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
    if not urls:
        return "<p style='color:red;'>âš ï¸� Please enter at least one URL.</p>", None

    # Loop agent: normalize URLs (add http:// if missing)
    urls = loop_retry_invalid(urls)
    df = controller.process(urls)

    # Styled HTML table with badges
    table_html = """
    <style>
    table {width:100%;border-collapse:collapse;font-family:'Segoe UI',sans-serif;}
    th {background:#0077b6;color:white;padding:10px;}
    td {padding:10px;border-bottom:1px solid #ddd;}
    tr:nth-child(even){background:#f7f7f7;}
    .badge{padding:4px 8px;border-radius:6px;font-weight:bold;color:white;}
    .legit{background:#2a9d8f;}
    .phish{background:#e63946;}
    .invalid{background:#6c757d;}
    .prob-green{color:#2a9d8f;font-weight:bold;}
    .prob-red{color:#e63946;font-weight:bold;}
    </style>
    <table>
    <tr><th>URL</th><th>Prediction</th><th>Probability Legitimate</th><th>Reasons</th></tr>
    """
    for _, row in df.iterrows():
        pred_class = "legit" if row["Prediction"] == "Legitimate" else "phish" if row["Prediction"] == "Phishing" else "invalid"
        prob_class = "prob-green" if row["Prediction"] == "Legitimate" else "prob-red" if row["Prediction"] == "Phishing" else ""
        badge = f"<span class='badge {pred_class}'>{row['Prediction']}</span>"
        prob_html = f"<span class='{prob_class}'>{row['Prob_legit']:.2f}</span>" if prob_class else "-"
        table_html += f"<tr><td>{row['URL']}</td><td>{badge}</td><td>{prob_html}</td><td>{row['Reasons']}</td></tr>"
    table_html += "</table>"

    # Summary graph
    valid_df = df[df["Prediction"].isin(["Legitimate", "Phishing"])]
    if not valid_df.empty:
        counts = valid_df["Prediction"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#2a9d8f" if label == "Legitimate" else "#e63946" for label in counts.index]
        counts.plot(kind="bar", color=colors, ax=ax, edgecolor="black")
        ax.set_title("Phishing vs Legitimate Predictions", fontsize=12, fontweight="bold")
        ax.set_xlabel("Prediction"); ax.set_ylabel("Count")
        plt.tight_layout()
    else:
        fig = None

    return table_html, fig

def gr_clear():
    return "", "", None

# --- Gradio UI ---
with gr.Blocks() as iface:
    gr.Markdown("""
    <div style='text-align:center; padding:10px;'>
        <h2 style='color:#0077b6;'>ğŸ”� Phishing Detection Agent</h2>
        <p>Paste one or more URLs (each on a new line) and click <b>Submit</b>.<br>
        Legitimate = <span style='color:#2a9d8f;font-weight:bold;'>Green</span>, 
        Phishing = <span style='color:#e63946;font-weight:bold;'>Red</span>, 
        Invalid = Gray.<br>
        Trusted domains (.edu, .gov, .org, .ac.in, etc.) are whitelisted.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            url_input = gr.Textbox(
                lines=6,
                label="Paste URLs here",
                placeholder="https://example.com\nhttp://g00gle.com\nhttps://aiems.edu.in"
            )
            with gr.Row():
                submit_btn = gr.Button("ğŸš€ Submit", variant="primary")
                clear_btn = gr.Button("ğŸ§¹ Clear", variant="secondary")
        with gr.Column(scale=1):
            plot_out = gr.Plot(label="ğŸ“Š Summary Graph")

    table_out = gr.HTML(label="ğŸ§¾ Results Table")

    submit_btn.click(fn=gr_predict, inputs=url_input, outputs=[table_out, plot_out])
    clear_btn.click(fn=gr_clear, inputs=None, outputs=[url_input, table_out, plot_out])

    iface.launch(inline=True)



# Cell 20: Final Notebook Conclusion + Next Steps

print("ğŸ�‰ Notebook Complete! Multi-Agent Phishing Detection System is ready.")

print("""
Summary of what we accomplished:
- âœ… Data preprocessing, train/test split, and RandomForest pipeline
- âœ… Model evaluation, feature importance, and saving pipeline
- âœ… Multi-agent orchestration (validator, featurizer, inference, reasoner, logger, memory)
- âœ… Loop agent for retrying invalid URLs
- âœ… Smoke tests, batch evaluation, and CSV export
- âœ… Gradio interactive UI for phishing detection
- âœ… Observability: logs, metrics, memory snapshots
- âœ… A2A protocol stub for agent-to-agent communication
- âœ… Visual analytics: confusion matrix, probability distributions, ROC curve
- âœ… Threshold tuning (static + interactive slider)
- âœ… Final summary export to JSON

Next Steps for Deployment:
1. ğŸ–¥ï¸� **Web UI / API**: Wrap the Gradio app into a standalone service or Flask/FastAPI endpoint.
2. ğŸ”’ **Security**: Harden trusted domain lists, add HTTPS enforcement, and integrate with threat intel feeds.
3. ğŸ“Š **Monitoring**: Connect logger + metrics to dashboards (Grafana/Prometheus).
4. â˜�ï¸� **Deployment**: Containerize with Docker and deploy to cloud (Azure, AWS, GCP).
5. ğŸ”„ **Continuous Learning**: Periodically retrain with fresh phishing datasets.
6. ğŸ‘¥ **User Testing**: Share with peers/competition judges for feedback and polish.

Congratulations â€” you now have a full agentic phishing detection notebook, complete with explainability, orchestration, and UI!
""")


