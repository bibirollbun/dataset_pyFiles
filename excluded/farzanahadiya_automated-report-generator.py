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


 # Imports & setup
import os
import json
import logging
import datetime
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Setup directories
os.makedirs("reports", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# Setup logging for observability
LOG_FILE = "agent.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ReportAgent")
logger.info("Notebook started")

# Utility for consistent timestamps
def now_ts():
    return datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")



 # Generate synthetic sales dataset (sample data)
def generate_sales_data(n_days=180, n_products=8, seed=42):
    np.random.seed(seed)
    start = datetime.date.today() - datetime.timedelta(days=n_days-1)
    dates = [start + datetime.timedelta(days=i) for i in range(n_days)]
    products = [f"Product_{i+1}" for i in range(n_products)]
    regions = ["North", "South", "East", "West"]

    rows = []
    for date in dates:
        for prod in products:
            units = max(0, int(np.random.poisson(20) + np.random.normal(0, 5)))
            price = float(round(50 + 20 * (int(prod.split("_")[1]) % 5) + np.random.normal(0, 5), 2))
            region = np.random.choice(regions, p=[0.3, 0.25, 0.2, 0.25])
            promo = np.random.choice([0, 1], p=[0.85, 0.15])
            rows.append({
                "date": date,
                "product": prod,
                "region": region,
                "units": units,
                "price": price,
                "revenue": round(units * price, 2),
                "promo": promo
            })
    df = pd.DataFrame(rows)
    return df

# Create dataset
sales_df = generate_sales_data(n_days=180, n_products=8)
sales_df["date"] = pd.to_datetime(sales_df["date"])
sales_df.head()



 # EDA and plotting utilities
def summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        total_revenue = float(df["revenue"].sum())
        total_units = int(df["units"].sum())
        top_product = df.groupby("product")["revenue"].sum().idxmax()
        top_region = df.groupby("region")["revenue"].sum().idxmax()
        avg_order_value = total_revenue / (df.shape[0] or 1)
        return {
            "total_revenue": total_revenue,
            "total_units": total_units,
            "top_product": top_product,
            "top_region": top_region,
            "avg_order_value": round(avg_order_value, 2),
        }
    except Exception as e:
        logger.exception("Error in summary_stats")
        raise

def plot_revenue_over_time(df: pd.DataFrame, outpath: str = "assets/revenue_time.png"):
    try:
        daily = df.groupby("date")["revenue"].sum().reset_index()
        plt.figure(figsize=(10,4))
        plt.plot(daily["date"], daily["revenue"], linewidth=1.5)
        plt.title("Daily Revenue")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        return outpath
    except Exception as e:
        logger.exception("plot_revenue_over_time failed")
        raise

def plot_top_products(df: pd.DataFrame, outpath: str = "assets/top_products.png", top_n=5):
    try:
        prod = df.groupby("product")["revenue"].sum().sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(8,4))
        prod.plot(kind="bar")
        plt.title(f"Top {top_n} Products by Revenue")
        plt.ylabel("Revenue")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        return outpath
    except Exception as e:
        logger.exception("plot_top_products failed")
        raise



 # Session & Memory store (simple implementations)

@dataclass
class SessionState:
    session_id: str
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    history: List[Dict[str, Any]] = field(default_factory=list)
    state_vars: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content, "ts": datetime.datetime.utcnow().isoformat()})

class MemoryStore:
    def __init__(self, path: str = "memory_store.json"):
        self.path = path
        # create file if missing
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({"memories": []}, f)
        logger.info("MemoryStore initialized at %s", path)

    def add_memory(self, key: str, value: Any, metadata: Optional[dict]=None):
        with open(self.path, "r+") as f:
            data = json.load(f)
            data["memories"].append({"id": str(uuid.uuid4()), "key": key, "value": value, "metadata": metadata or {}, "ts": datetime.datetime.utcnow().isoformat()})
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
        logger.info("Added memory: %s", key)

    def search(self, query: str) -> List[dict]:
        # very simple keyword search
        with open(self.path, "r") as f:
            data = json.load(f)
        results = []
        for mem in data.get("memories", []):
            if query.lower() in mem.get("key", "").lower() or query.lower() in str(mem.get("value", "")).lower():
                results.append(mem)
        return results

# create default stores
session_state = SessionState(session_id=f"session_{now_ts()}")
memory_store = MemoryStore()
logger.info("Session %s started", session_state.session_id)



 # Tools (these are the "capabilities" the agent uses)
def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute business KPIs used by the generator."""
    try:
        stats = summary_stats(df)
        # trend: compare last 14 days revenue vs previous 14 days
        daily = df.groupby("date")["revenue"].sum().sort_index()
        if len(daily) >= 28:
            last14 = daily[-14:].sum()
            prev14 = daily[-28:-14].sum()
            trend_pct = ((last14 - prev14) / (prev14 or 1.0)) * 100
        else:
            trend_pct = 0.0
        stats["recent_trend_pct"] = round(trend_pct, 2)
        return stats
    except Exception as e:
        logger.exception("compute_kpis failed")
        raise

def explain_insights(kpis: dict) -> str:
    """Produce a human-friendly executive summary (simple rule-based)."""
    try:
        lines = []
        lines.append(f"Total Revenue: ${kpis['total_revenue']:,}")
        lines.append(f"Total Units Sold: {kpis['total_units']:,}")
        lines.append(f"Top Product: {kpis['top_product']}")
        lines.append(f"Top Region: {kpis['top_region']}")
        if kpis.get("recent_trend_pct", 0) >= 5:
            lines.append(f"Revenue trend is positive: {kpis['recent_trend_pct']}% increase vs previous period.")
        elif kpis.get("recent_trend_pct", 0) <= -5:
            lines.append(f"Revenue trend is negative: {kpis['recent_trend_pct']}% decrease vs previous period.")
        else:
            lines.append(f"Revenue trend is stable: {kpis['recent_trend_pct']}% change.")
        # add quick recommendation heuristics
        if kpis["top_product"].lower().startswith("product_"):
            lines.append("Recommendation: Consider promotional campaigns for underperforming products in top regions.")
        return "\n".join(lines)
    except Exception as e:
        logger.exception("explain_insights failed")
        raise



 # ReportGenerator class (main "agent")
class ReportGenerator:
    def __init__(self, df: pd.DataFrame, session: SessionState, memory: MemoryStore, out_dir: str = "reports"):
        if df is None or df.empty:
            raise ValueError("Input dataframe is empty or None.")
        self.df = df.copy()
        self.session = session
        self.memory = memory
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        logger.info("ReportGenerator initialized: out_dir=%s", out_dir)

    def run(self, report_name: Optional[str] = None) -> Dict[str, Any]:
        """Run full pipeline: compute KPIs, generate plots, create HTML report."""
        try:
            # session logging
            self.session.add_message("system", "Starting report generation")
            run_id = report_name or f"report_{now_ts()}"
            logger.info("Report run started: %s", run_id)

            # 1. Compute KPIs
            kpis = compute_kpis(self.df)
            logger.info("KPIs computed")

            # 2. Create charts
            chart1 = plot_revenue_over_time(self.df, outpath=os.path.join("assets", f"revenue_time_{run_id}.png"))
            chart2 = plot_top_products(self.df, outpath=os.path.join("assets", f"top_products_{run_id}.png"))
            logger.info("Charts created: %s, %s", chart1, chart2)

            # 3. Executive summary
            summary = explain_insights(kpis)
            logger.info("Summary generated")

            # 4. Save simple data snapshot to memory for long-term recall
            self.memory.add_memory(key="last_report_kpis", value=kpis, metadata={"run_id": run_id})
            logger.info("Stored KPIs in memory")

            # 5. Build HTML report (simple)
            timestamp = datetime.datetime.utcnow().isoformat()
            html = self._build_html_report(run_id, kpis, summary, chart1, chart2, timestamp)
            fp = os.path.join(self.out_dir, f"{run_id}_sales_report.html")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info("Report saved at %s", fp)

            # 6. session & output
            self.session.add_message("agent", f"Report generated: {fp}")
            return {"report_name": run_id, "report_file": fp, "summary": summary, "kpis": kpis}
        except Exception as e:
            logger.exception("Failed to generate report in run()")
            raise

    def _build_html_report(self, run_id, kpis, summary, chart1, chart2, ts):
        # sanitize paths for HTML
        c1 = os.path.basename(chart1)
        c2 = os.path.basename(chart2)
        html = f"""
        <html>
        <head><title>Sales Report - {run_id}</title></head>
        <body style="font-family: Arial, sans-serif; margin:30px;">
        <h1>Sales Report - {run_id}</h1>
        <p><em>Generated at (UTC): {ts}</em></p>
        <h2>Executive Summary</h2>
        <pre style="background:#f6f6f6; padding:10px; border-radius:5px;">{summary}</pre>
        <h2>Key Metrics</h2>
        <ul>
        <li>Total revenue: ${kpis['total_revenue']:,}</li>
        <li>Total units sold: {kpis['total_units']:,}</li>
        <li>Top product: {kpis['top_product']}</li>
        <li>Top region: {kpis['top_region']}</li>
        <li>Recent trend (%): {kpis.get('recent_trend_pct', 0)}%</li>
        </ul>
        <h2>Charts</h2>
        <div><img src="../assets/{c1}" style="max-width:800px; width:100%;"></div>
        <div style="height:20px;"></div>
        <div><img src="../assets/{c2}" style="max-width:700px; width:80%;"></div>
        <hr>
        <p>Generated by Automated Enterprise Report Generator.</p>
        </body>
        </html>
        """
        return html



 # Run a report generation (this replaces the broken Cell 7 earlier)
try:
    rg = ReportGenerator(sales_df, session_state, memory_store, out_dir="reports")
    out = rg.run(report_name=f"auto_report_{now_ts()}")
    print("Report created:", out["report_file"])
    print("\nQuick Summary:\n", out["summary"])
except Exception as e:
    logger.exception("Failed to run ReportGenerator")
    raise



 # Preview report content (first 400 chars)
report_path = out["report_file"]
print("Report saved at:", report_path)
with open(report_path, "r", encoding="utf-8") as f:
    print(f.read()[:800])  # print first bits of HTML



 # Simple evaluation to verify essential content presence
def evaluate_report_content(report_file: str, kpis: dict) -> Dict[str, Any]:
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            html = f.read()
        checks = {
            "mentions_top_product": kpis["top_product"].lower() in html.lower(),
            "mentions_total_revenue": str(int(kpis["total_revenue"]))[:3] in html,  # crude revenue check
        }
        score = sum(checks.values()) / len(checks)
        return {"checks": checks, "score": score}
    except Exception as e:
        logger.exception("evaluate_report_content failed")
        raise

eval_res = evaluate_report_content(out["report_file"], out["kpis"])
print("Evaluation result:", eval_res)



 # Show what's in memory_store for transparency
print("Memory search for 'last_report_kpis':")
mems = memory_store.search("last_report_kpis")
print(json.dumps(mems, indent=2))



 # Final verification, list artifacts
print("Artifacts created:")
for p in os.listdir("reports"):
    print("-", os.path.join("reports", p))
print("\nMemory store:", os.path.exists("memory_store.json"))
print("Log file:", LOG_FILE, " (size bytes):", os.path.getsize(LOG_FILE))


