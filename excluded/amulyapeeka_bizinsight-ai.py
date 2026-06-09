# Install minimal deps (run once)
!pip install -q beautifulsoup4 pandas matplotlib

# Imports
import os, json, time, uuid, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
print("âœ” Libraries installed and imported successfully.")


WORKDIR = "/kaggle/working/bizinsight_ai"
os.makedirs(WORKDIR, exist_ok=True)
DATA_DIR = "/kaggle/input/sample-biz-data"  # if you upload a Kaggle dataset, else we'll create sample data
RUN_ID = lambda: f"run_{int(time.time())}"
SEO_THRESHOLD = 0.7
print(f"âœ” WORKDIR set to: {WORKDIR}\nâœ” DATA_DIR set to: {DATA_DIR}\nâœ” Environment initialized successfully.")


# Simple structured logger
logger = logging.getLogger("bizinsight")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.FileHandler(os.path.join(WORKDIR, "run.log"))
    fmt = logging.Formatter('{"time":"%(asctime)s","run_id":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}')
    h.setFormatter(fmt)
    logger.addHandler(h)

def log_info(msg, **extra):
    logger.info(msg + " | " + json.dumps(extra))

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
print(f"âœ” Logger initialized and writing to: {os.path.join(WORKDIR, 'run.log')}")


class InMemorySession:
    def __init__(self, workdir):
        self.workdir = workdir
        self.sessions = {}
    def start(self, run_id, companies):
        self.sessions[run_id] = {"companies": companies, "start": time.time(), "outputs": {}}
        return self.sessions[run_id]
    def save_output(self, run_id, key, obj):
        self.sessions[run_id]["outputs"][key] = obj
        save_json(os.path.join(self.workdir, f"{run_id}_{key}.json"), obj)
    def finish(self, run_id):
        self.sessions[run_id]["end"] = time.time()
        save_json(os.path.join(self.workdir, f"{run_id}_session.json"), self.sessions[run_id])

# simple long-term memory as JSON file
MEMORY_FILE = os.path.join(WORKDIR, "memory_bank.json")
if not os.path.exists(MEMORY_FILE):
    save_json(MEMORY_FILE, {})

def load_memory():
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
def update_memory(product_key, snapshot):
    mem = load_memory()
    mem.setdefault(product_key, []).append(snapshot)
    save_json(MEMORY_FILE, mem)
print("âœ” Session system initialized.\nâœ” Memory bank loaded at:", MEMORY_FILE)


# Create small sample dataset in-memory for demo
sample = {
    "AcmeCorp": [
        {"id":"a1","text":"Great battery life but shipping was slow. Price is high.","rating":4,"source":"shopA","date":"2025-10-01"},
        {"id":"a2","text":"Customer support was unhelpful. Love the UI.","rating":3,"source":"shopB","date":"2025-10-05"},
        {"id":"a3","text":"Price is confusing, discounts unclear.","rating":2,"source":"shopA","date":"2025-11-01"},
    ],
    "ExampleCo": [
        {"id":"e1","text":"Affordable and performs well. Shipping was fast.","rating":5,"source":"shopC","date":"2025-10-03"},
        {"id":"e2","text":"Battery drains quickly, but UI is clean.","rating":3,"source":"shopD","date":"2025-10-20"},
    ]
}
save_json(os.path.join(WORKDIR, "sample_reviews.json"), sample)
print("Sample data saved:", os.path.join(WORKDIR, "sample_reviews.json"))



@dataclass
class AgentOutput:
    ok: bool
    payload: Any
    meta: Dict[str,Any] = None

class BaseAgent:
    def __init__(self, name):
        self.name = name
    def run(self, *args, **kwargs) -> AgentOutput:
        raise NotImplementedError
print("âœ” Agent base classes loaded (AgentOutput, BaseAgent).")


class CollectorAgent(BaseAgent):
    def __init__(self, data):
        super().__init__("collector")
        self.data = data
    def run(self, company):
        # In real system: call APIs / scraper. Here we return normalized sample reviews.
        items = self.data.get(company, [])
        # normalize: lowercase text
        for it in items:
            it["text_norm"] = it["text"].strip()
        return AgentOutput(True, items, {"count": len(items)})
print("âœ” CollectorAgent loaded and ready.")


# Simple sentiment heuristic for demo: score = (rating-3)/2 for those with rating; else keyword heuristics
positive_words = {"great","fast","affordable","love","good","excellent"}
negative_words = {"slow","unhelpful","confusing","drains","high","expensive","bad"}

class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__("sentiment")
    def run(self, reviews):
        out = []
        for r in reviews:
            text = r.get("text","").lower()
            # rating-based
            if r.get("rating") is not None:
                score = (r["rating"] - 3) / 2.0  # -1..+1
            else:
                # heuristic
                pos = sum(w in text for w in positive_words)
                neg = sum(w in text for w in negative_words)
                score = (pos - neg) / max(1, pos+neg)
            out.append({"id": r["id"], "score": float(score), "text": r["text"]})
        # aggregate
        avg = np.mean([o["score"] for o in out]) if out else 0.0
        return AgentOutput(True, {"per_review": out, "avg": float(avg)}, {"n": len(out)})

class ThemeAgent(BaseAgent):
    def __init__(self):
        super().__init__("theme")
    def run(self, reviews):
        from collections import Counter
        kw_counter = Counter()
        for r in reviews:
            text = r.get("text","").lower()
            for kw in ["battery","price","shipping","support","ui","performance","discount"]:
                if kw in text:
                    kw_counter[kw] += 1
        themes = [{"theme":k,"count":v} for k,v in kw_counter.most_common()]
        return AgentOutput(True, {"themes": themes}, {"unique_themes": len(themes)})
print("âœ” SentimentAgent and ThemeAgent loaded successfully.")


class ComparisonAgent(BaseAgent):
    def __init__(self):
        super().__init__("comparison")
    def run(self, features_a, features_b):
        # features_* are lists of feature names for demo (we'll fake them)
        fa = set(features_a)
        fb = set(features_b)
        allf = sorted(fa.union(fb))
        rows = []
        for f in allf:
            rows.append({"feature": f, "A_has": f in fa, "B_has": f in fb})
        return AgentOutput(True, rows, {"n_features": len(allf)})

class InsightsAgent(BaseAgent):
    def __init__(self):
        super().__init__("insights")
    def run(self, sentiment_out, themes_out):
        # simple rule: if theme 'price' has many counts and avg sentiment < 0 => prioritize pricing fix
        themes = themes_out.get("themes",[])
        theme_map = {t["theme"]: t["count"] for t in themes}
        avg_sent = sentiment_out.get("avg",0)
        recs = []
        if theme_map.get("price",0) > 0 and avg_sent < 0:
            recs.append({"action":"Review pricing clarity","impact":"High","effort":"Low","evidence":"price mentions & negative sentiment"})
        if theme_map.get("battery",0) > 0:
            recs.append({"action":"Investigate battery issues","impact":"Medium","effort":"Medium"})
        return AgentOutput(True, {"recommendations": recs}, {"n_recs": len(recs)})
print("âœ” ComparisonAgent and InsightsAgent loaded successfully.")


def plot_sentiment_hist(per_review, out_path):
    scores = [r["score"] for r in per_review]
    plt.figure(figsize=(4,3))
    plt.hist(scores, bins=5)
    plt.title("Sentiment distribution")
    plt.savefig(out_path)
    plt.close()

def generate_markdown_report(company, sentiment_out, themes_out, insights_out, outdir):
    md = []
    md.append(f"# Report â€” {company}\n")
    md.append(f"**Avg sentiment:** {sentiment_out['avg']:.2f}\n")
    md.append("## Top themes\n")
    for t in themes_out.get("themes",[]):
        md.append(f"- {t['theme']} (mentions: {t['count']})\n")
    md.append("\n## Recommendations\n")
    for r in insights_out.get("recommendations",[]):
        md.append(f"- **{r['action']}** â€” Impact: {r['impact']}, Effort: {r['effort']}\n")
    md_text = "\n".join(md)
    out_md = os.path.join(outdir, f"report_{company}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_text)
    return out_md
print("âœ” Report generator and sentiment plotting utilities loaded.")


class Orchestrator:
    def __init__(self, data, workdir):
        self.data = data
        self.workdir = workdir
        self.session = InMemorySession(workdir)
        self.executor = ThreadPoolExecutor(max_workers=4)
    def run_companies(self, companies):
        run_id = RUN_ID()
        self.session.start(run_id, companies)
        results = {}
        # parallel collect
        collect_futures = {self.executor.submit(CollectorAgent(self.data).run, c): c for c in companies}
        collects = {}
        for fut in as_completed(collect_futures):
            company = collect_futures[fut]
            out = fut.result()
            collects[company] = out.payload
            log_info("collected", company=company, count=out.meta["count"])
            self.session.save_output(run_id, f"{company}_raw", out.payload)
        # parallel analysis (sentiment & themes)
        analysis_futures = {}
        for c, reviews in collects.items():
            analysis_futures[self.executor.submit(SentimentAgent().run, reviews)] = (c,"sentiment")
            analysis_futures[self.executor.submit(ThemeAgent().run, reviews)] = (c,"themes")
        # gather
        by_company = {c: {} for c in companies}
        for fut in as_completed(analysis_futures):
            comp,atype = analysis_futures[fut]
            out = fut.result()
            by_company[comp][atype] = out.payload
            self.session.save_output(run_id, f"{comp}_{atype}", out.payload)
            log_info("analyzed", company=comp, type=atype, meta=out.meta)
        # comparison + insights + report generation (sequential for simplicity)
        comp_agent = ComparisonAgent()
        insights_agent = InsightsAgent()
        reports = {}
        # For demo: compare first two companies if more than 1
        if len(companies) >= 2:
            a,b = companies[:2]
            # fake features lists for demo
            features_a = ["feature_x","feature_y","battery"]
            features_b = ["feature_x","feature_z","ui"]
            comp_out = comp_agent.run(features_a, features_b)
            self.session.save_output(run_id, "comparison", comp_out.payload)
        for c in companies:
            sentiment = by_company[c].get("sentiment", {"avg":0,"per_review":[]})
            themes = by_company[c].get("themes", {"themes":[]})
            insights = insights_agent.run(sentiment, themes)
            self.session.save_output(run_id, f"{c}_insights", insights.payload)
            # make chart and report
            chart_path = os.path.join(self.workdir, f"{c}_sent_hist.png")
            plot_sentiment_hist(sentiment.get("per_review",[]), chart_path)
            md_path = generate_markdown_report(c, sentiment, themes, insights.payload, self.workdir)
            reports[c] = {"chart": chart_path, "report_md": md_path}
            log_info("report_created", company=c, report=md_path)
        self.session.finish(run_id)
        return {"run_id": run_id, "reports": reports}
print("âœ” Orchestrator initialized â€” ready to run multi-agent competitive analysis.")


# Run the orchestrator on our sample data
data = sample  # from earlier cell
orch = Orchestrator(data, WORKDIR)
out = orch.run_companies(["AcmeCorp","ExampleCo"])
print("Run complete:", out["run_id"])
for comp, info in out["reports"].items():
    print(comp, "report:", info["report_md"], "chart:", info["chart"])


from IPython.display import Markdown, Image, display
for comp, info in out["reports"].items():
    display(Markdown(f"## Report for {comp}"))
    display(Markdown(open(info["report_md"], "r", encoding="utf-8").read()))
    display(Image(info["chart"]))
print("âœ” All reports and charts displayed successfully.")


# Simple validation that outputs exist
for comp, info in out["reports"].items():
    assert os.path.exists(info["report_md"]), "Missing md"
    assert os.path.exists(info["chart"]), "Missing chart"
print("Basic checks passed.")


# Cell A: Load dataset and inspect brands
import pandas as pd
DATA_PATH = "/kaggle/input/amazon-reviews-unlocked-mobile-phones/Amazon_Unlocked_Mobile.csv"

print("Loading dataset from:", DATA_PATH)
df_raw = pd.read_csv(DATA_PATH, encoding='utf-8', low_memory=False)
print("Total rows loaded:", len(df_raw))

# Inspect brand column (may be 'brand' or 'manufacturer' or similar â€” find likely candidate)
possible_brand_cols = [c for c in df_raw.columns if 'brand' in c.lower() or 'manufacturer' in c.lower()]
print("Possible brand columns:", possible_brand_cols)
# Try to infer brand column
brand_col = possible_brand_cols[0] if possible_brand_cols else None
if not brand_col:
    # fallback: try common column names
    for candidate in ['brand','Brand','manufacturer','Manufacturer','company']:
        if candidate in df_raw.columns:
            brand_col = candidate
            break

print("Using brand column:", brand_col)
# Quick look
display(df_raw.head(3))
print("âœ” Dataset loaded and brand column identified.")


# Cell B: Clean & select top N brands
NUM_COMPANIES = 3   # change this to run more/less companies
text_col_candidates = [c for c in df_raw.columns if 'review' in c.lower() or 'text' in c.lower() or 'content' in c.lower()]
rating_col_candidates = [c for c in df_raw.columns if 'rating' in c.lower() or 'score' in c.lower() or 'stars' in c.lower()]

print("Text columns candidates:", text_col_candidates)
print("Rating columns candidates:", rating_col_candidates)

# Choose text and rating columns (best guess)
text_col = text_col_candidates[0] if text_col_candidates else df_raw.columns[0]
rating_col = rating_col_candidates[0] if rating_col_candidates else None

# Keep rows with non-null brand and text
df = df_raw[[brand_col, text_col] + ([rating_col] if rating_col else [])].dropna(subset=[brand_col, text_col]).copy()
df[brand_col] = df[brand_col].astype(str).str.strip()
df[text_col] = df[text_col].astype(str).str.strip()

# Compute top brands
brand_counts = df[brand_col].value_counts()
top_brands = brand_counts.head(NUM_COMPANIES).index.tolist()
print(f"Top {NUM_COMPANIES} brands selected:", top_brands)
display(brand_counts.head(10))


# Cell C: Build sample_reviews dict used by the orchestrator
MAX_REVIEWS_PER_BRAND = 300   # tuning: limit to keep runtime reasonable in Kaggle
sample_reviews_real = {}

for brand in top_brands:
    subset = df[df[brand_col] == brand].head(MAX_REVIEWS_PER_BRAND)
    reviews_list = []
    for idx, row in subset.iterrows():
        rid = f"{brand[:6]}_{idx}"
        text = row[text_col]
        rating = float(row[rating_col]) if rating_col and not pd.isna(row[rating_col]) else None
        # Source and date unknown in this dataset; use placeholders or other columns if available
        src = str(row.get('source','')) if 'source' in row.index else 'kaggle'
        date = str(row.get('review_date', '')) if 'review_date' in row.index else ''
        reviews_list.append({"id": rid, "text": text, "rating": rating, "source": src, "date": date})
    sample_reviews_real[brand] = reviews_list

# Save a copy for reproducibility
save_json(os.path.join(WORKDIR, "sample_reviews_from_kaggle.json"), sample_reviews_real)
print("Built sample_reviews for brands:", list(sample_reviews_real.keys()))
for b in sample_reviews_real:
    print(f" - {b}: {len(sample_reviews_real[b])} reviews")



# Cell D: Preview 2 reviews per selected brand
for brand, reviews in sample_reviews_real.items():
    print(f"\n=== {brand} (showing up to 2 reviews) ===")
    for r in reviews[:2]:
        print("-", r["text"][:200].replace("\n"," "), "...", f"(rating={r['rating']})")


# Cell E: Run orchestrator on top brands (uses existing Orchestrator implementation)
data_for_orch = sample_reviews_real   # map brand -> list of review dicts
orch = Orchestrator(data_for_orch, WORKDIR)
result_real = orch.run_companies(list(data_for_orch.keys()))

print("Run complete:", result_real["run_id"])
for comp, info in result_real["reports"].items():
    print(f"{comp}: report saved to {info['report_md']}, chart saved to {info['chart']}")


# Cell F: Display reports & charts inline for each brand
for comp, info in result_real["reports"].items():
    display(Markdown(f"## Report for {comp}"))
    with open(info["report_md"], "r", encoding="utf-8") as f:
        display(Markdown(f.read()))
    display(Image(info["chart"]))

