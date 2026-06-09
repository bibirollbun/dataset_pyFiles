import os, json, uuid, time, logging
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({'figure.autolayout': True})

OUTDIR = "/kaggle/working/asac_outputs"
os.makedirs(OUTDIR, exist_ok=True)



LOGPATH = os.path.join(OUTDIR, "asac.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOGPATH), logging.StreamHandler()]
)

logger = logging.getLogger("ASAC")

CONFIG = {
    "rows": 2500,
    "start_date": "2024-01-01",
    "months": 12,
    "top_n": 10,
    "batch_size": 400
}

logger.info("Configuration loaded.")



class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, fn):
        self.tools[name] = fn

    def call(self, name, *args, **kwargs):
        return self.tools[name](*args, **kwargs)

TOOLS = ToolRegistry()

def export_csv(df, filename):
    path = os.path.join(OUTDIR, filename)
    df.to_csv(path, index=False)
    logger.info(f"Exported {filename}")
    return path

TOOLS.register("export_csv", export_csv)



def generate_sales(n, start="2024-01-01", months=12):
    np.random.seed(42)
    start = pd.to_datetime(start)

    dates = [start + timedelta(days=int(x)) for x in np.random.randint(0, months*30, n)]

    df = pd.DataFrame({
        "sales_date": dates,
        "region": np.random.choice(["North","South","East","West"], n),
        "territory": [f"Terr-{np.random.randint(1,30)}" for _ in range(n)],
        "customer_id": [f"CUST-{np.random.randint(1000,1100)}" for _ in range(n)],
        "product_id": [f"SKU-{np.random.randint(100,140)}" for _ in range(n)],
        "units_sold": np.random.randint(1,8,n)
    })

    df["unit_price"] = np.random.normal(100, 30, n).clip(10,400)
    df["discount"] = np.random.choice([0,0.05,0.1,0.15], n)
    df["sales_value"] = df["unit_price"]*df["units_sold"]*(1-df["discount"])

    df.loc[np.random.choice(df.index, 8), "sales_date"] = df["sales_date"].astype(str).str.replace("-", "/")
    df.loc[np.random.choice(df.index, 6), "unit_price"] = np.nan

    return df

raw_df = generate_sales(CONFIG["rows"], CONFIG["start_date"], CONFIG["months"])
raw_df.head()



class AgentMessage:
    def __init__(self, sender, recipient, payload):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.recipient = recipient
        self.payload = payload
        self.timestamp = time.time()

class BaseAgent:
    def __init__(self, name):
        self.name = name
        self.inbox = []

    def send(self, msg, recipient):
        recipient.inbox.append(msg)

    def process(self):
        while self.inbox:
            self.handle(self.inbox.pop(0))

    def handle(self, msg):
        raise NotImplementedError



class IngestionAgent(BaseAgent):
    def run(self, df):
        TOOLS.call("export_csv", df, "raw_uploaded.csv")
        return df

class CleaningAgent(BaseAgent):
    def run(self, df):
        df = df.copy()
        df["sales_date"] = pd.to_datetime(df["sales_date"], errors="coerce")
        df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())
        df["sales_value"] = df["unit_price"] * df["units_sold"] * (1 - df["discount"])
        df["month"] = df["sales_date"].dt.to_period("M").astype(str)
        TOOLS.call("export_csv", df, "cleaned_sales.csv")
        return df

class AnalyticsAgent(BaseAgent):
    def mrev(self, df):
        return df.groupby("month")["sales_value"].sum().reset_index()

    def topn(self, df, n=10):
        return df.groupby("product_id")["sales_value"].sum().sort_values(ascending=False).head(n)

    def outliers(self, df):
        df2 = df.copy()
        df2["z"] = stats.zscore(df2["unit_price"])
        return df2[df2["z"].abs() > 3]



ing = IngestionAgent("ing")
cleaner = CleaningAgent("clean")
ana = AnalyticsAgent("ana")

clean_df = cleaner.run(raw_df)

tasks = {
    "mrev": ana.mrev(clean_df),
    "topn": ana.topn(clean_df, CONFIG["top_n"]),
    "outliers": ana.outliers(clean_df)
}

for k, v in tasks.items():
    TOOLS.call("export_csv", v, f"{k}.csv")

tasks["mrev"].head()



class InMemorySession:
    def __init__(self):
        self.sessions = {}

    def create(self, meta=None):
        sid = str(uuid.uuid4())
        self.sessions[sid] = {"created": time.time(), "state": {}, **(meta or {})}
        return sid

    def update(self, sid, key, value):
        self.sessions[sid]["state"][key] = value

class MemoryBank:
    def __init__(self, path=os.path.join(OUTDIR, "memory.json")):
        self.path = path
        self.store = json.load(open(path)) if os.path.exists(path) else {}

    def put(self, key, value):
        self.store[key] = value
        json.dump(self.store, open(self.path, "w"), indent=2)

    def compact(self, keep=6):
        snap = self.store.get("mrev", {})
        months = sorted(snap.keys())
        compact = {m: snap[m] for m in months[-keep:]}
        self.store["compact"] = compact
        json.dump(self.store, open(self.path, "w"), indent=2)
        return compact

session = InMemorySession()
sid = session.create(meta={"user": "demo"})
session.update(sid, "rows", len(clean_df))

mem = MemoryBank()
mem.put("mrev", tasks["mrev"].set_index("month")["sales_value"].to_dict())
mem.compact()



prompt = f"""
Generate a business summary:
- Revenue months: {tasks['mrev']['month'].min()} to {tasks['mrev']['month'].max()}
- Top products: {tasks['topn'].to_dict()}
- Outliers detected: {len(tasks['outliers'])}
"""

with open(os.path.join(OUTDIR, "llm_prompt.txt"), "w") as f:
    f.write(prompt)

print(prompt)


