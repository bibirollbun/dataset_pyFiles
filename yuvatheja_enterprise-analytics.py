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


# Code cell: imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import json
import datetime

# Ensure directories
Path("data").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)



# Code cell: write synthetic sample data to data/
sample_sales = """date,region,amount,order_id,customer_id
2025-11-01,East,1000.00,ORD-1001,C001
2025-11-02,West,1500.00,ORD-1002,C002
2025-11-03,East,2000.00,ORD-1003,C003
2025-11-04,North,250.50,ORD-1004,C002
2025-11-07,South,900.00,ORD-1005,C004
2025-11-08,East,1200.00,ORD-1006,C001
2025-10-30,West,700.00,ORD-1000,C005
"""

sample_customers = """customer_id,signup_date,segment
C001,2025-01-10,SmallBiz
C002,2025-02-15,Enterprise
C003,2025-03-20,SMB
C004,2025-04-12,Startup
C005,2024-11-05,Enterprise
"""

with open("data/sample_sales.csv", "w") as f:
    f.write(sample_sales)
with open("data/sample_customers.csv", "w") as f:
    f.write(sample_customers)

print("Sample CSVs written to data/ (or replace with your own CSVs in notebook input).")



# Code cell: LLM planner stub
import re
def plan_from_query(user_query: str):
    q = user_query.lower()
    plan = {"intent": "summary", "time_range": "last_7_days", "metrics": ["sales"], "actions": ["extract","clean","analytics","report"]}
    if "top" in q and "region" in q:
        plan['intent'] = 'top_regions'
    # parse "last N days/weeks/months"
    m = re.search(r'last (\d+) (day|days|week|weeks|month|months)', q)
    if m:
        n = int(m.group(1)); unit = m.group(2)
        plan['time_range'] = f"last_{n}_{unit}"
    elif 'weekly' in q or 'week' in q:
        plan['time_range'] = 'last_7_days'
    elif 'monthly' in q or 'month' in q:
        plan['time_range'] = 'last_30_days'
    return plan

# Quick demo:
print(plan_from_query("Generate weekly sales report and top 5 regions for last 7 days"))



# Code cell: extract_data
def extract_data(paths=None):
    paths = paths or {"sales": "data/sample_sales.csv", "customers": "data/sample_customers.csv"}
    sales = pd.read_csv(paths['sales'])
    customers = pd.read_csv(paths['customers'])
    return {"sales": sales, "customers": customers}

data = extract_data()
data['sales'].head()



# Code cell: cleaning functions
def clean_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    df = sales_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'amount', 'region'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    df['region'] = df['region'].astype(str).str.strip().str.title()
    return df

def clean_customers(customers_df: pd.DataFrame) -> pd.DataFrame:
    df = customers_df.copy()
    if 'signup_date' in df.columns:
        df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df = df.dropna(subset=['customer_id'])
    return df

sales = clean_sales(data['sales'])
customers = clean_customers(data['customers'])
sales.head()



# Analytics with robust time-range handling (uses dataset max date)
def filter_time_range(df: pd.DataFrame, time_range: str):
    # If df has no 'date' column, return as-is
    if 'date' not in df.columns:
        return df

    # Use dataset's max date (so synthetic data works even if notebook clock differs)
    now = pd.to_datetime(df['date']).max()

    if time_range == "all":
        return df

    if time_range.startswith('last_'):
        parts = time_range.split('_')
        try:
            n = int(parts[1])
            unit = parts[2]
        except:
            n, unit = 7, 'days'

        if 'day' in unit:
            start = now - pd.Timedelta(days=n)
        elif 'week' in unit:
            start = now - pd.Timedelta(weeks=n)
        elif 'month' in unit:
            start = now - pd.Timedelta(days=30*n)
        else:
            start = now - pd.Timedelta(days=7)

        return df[df['date'] >= start]

    return df

def compute_kpis(sales_df: pd.DataFrame, customers_df: pd.DataFrame, time_range='last_7_days'):
    # Filter by time range (robust)
    filtered = filter_time_range(sales_df, time_range) if 'date' in sales_df.columns else sales_df.copy()

    # Total sales in range
    total_sales = float(filtered['amount'].sum()) if 'amount' in filtered.columns else float(filtered['SalesAmount'].sum())

    # Sales by region - make sure region column name matches
    region_col = 'region' if 'region' in filtered.columns else 'Region'
    amount_col = 'amount' if 'amount' in filtered.columns else 'SalesAmount'

    by_region = filtered.groupby(region_col)[amount_col].sum().sort_values(ascending=False)

    # Sales by product (optional)
    if 'product' in filtered.columns or 'Product' in filtered.columns:
        pcol = 'product' if 'product' in filtered.columns else 'Product'
        by_product = filtered.groupby(pcol)[amount_col].sum().sort_values(ascending=False)
    else:
        by_product = pd.Series(dtype=float)

    # Customer count
    cust_col = 'customer_id' if 'customer_id' in customers_df.columns else ('CustomerId' if 'CustomerId' in customers_df.columns else None)
    total_customers = int(customers_df[cust_col].nunique()) if cust_col else None

    summary = {
        "total_sales": total_sales,
        "total_customers": total_customers,
        "recent_period": time_range
    }

    return {"summary": summary, "by_region": by_region, "by_product": by_product, "filtered": filtered}

# Example run using the planner's time_range (if you used the LLM planner above)
user_query = "Generate weekly sales report and top 5 sales regions for the last 7 days"
plan = plan_from_query(user_query)
kpis = compute_kpis(sales, customers, time_range=plan.get('time_range', 'last_7_days'))

# Print summary to verify
print("KPI Summary:", kpis['summary'])
print("\nTop regions (head):")
print(kpis['by_region'].head())



from PIL import Image as PILImage, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def generate_dashboard(by_region, out_path="outputs/dashboard_regions.png"):
    # If empty, make a placeholder image explaining no data
    if by_region is None or by_region.empty:
        w, h = 800, 400
        img = PILImage.new("RGB", (w, h), color=(255,255,255))
        d = ImageDraw.Draw(img)
        msg = "No sales in selected time range to display."
        try:
            # Try a default font (may vary on environment)
            f = ImageFont.load_default()
            d.text((20, h//2 - 10), msg, fill=(0,0,0), font=f)
        except:
            d.text((20, h//2 - 10), msg, fill=(0,0,0))
        img.save(out_path)
        return out_path

    # Plot normally when data exists
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    # If it's a Series, plot directly
    if isinstance(by_region, pd.Series):
        by_region.head(10).plot(kind='bar', ax=ax)
    else:
        # Convert to series if needed
        s = pd.Series(by_region).head(10)
        s.plot(kind='bar', ax=ax)

    ax.set_title("Top Regions by Sales")
    ax.set_xlabel("Region")
    ax.set_ylabel("Sales")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

dashboard_path = generate_dashboard(kpis['by_region'], out_path="outputs/dashboard_regions.png")
print("Dashboard saved to:", dashboard_path)



from IPython.display import Image, display, Markdown

# Show KPI summary
summary_df = pd.DataFrame([kpis['summary']])
display(Markdown("### KPI Summary"))
display(summary_df)

# Show regional sales table
display(Markdown("### Sales by Region"))
if not kpis['by_region'].empty:
    region_table = kpis['by_region'].reset_index().rename(columns={0:'sales'} if kpis['by_region'].name is None else {kpis['by_region'].name:'sales'})
    # ensure column name
    region_table.columns = ['region','sales'] if region_table.shape[1]==2 else region_table.columns
    display(region_table.head(10))
else:
    display(Markdown("_No regional sales for selected period._"))

# Display dashboard image
display(Markdown("### Dashboard"))
display(Image(dashboard_path))



# Save insight text
top_region = kpis['by_region'].idxmax() if (kpis['by_region'] is not None and not kpis['by_region'].empty) else None
top_amount = float(kpis['by_region'].max()) if (kpis['by_region'] is not None and not kpis['by_region'].empty) else 0.0

insight_text = (
    f"Automated Sales Summary ({kpis['summary']['recent_period']}): "
    f"Total sales = ${kpis['summary']['total_sales']:.2f}. "
    f"Unique customers = {kpis['summary']['total_customers']}. "
)

if top_region:
    insight_text += f"Top region: {top_region} (${top_amount:.2f})."
else:
    insight_text += "No top region (no sales in period)."

with open("outputs/summary_insight.txt", "w") as f:
    f.write(insight_text)

print("Insight saved to outputs/summary_insight.txt")
print(insight_text)



import sqlite3, json, datetime

def save_report_to_memory(db_path="outputs/reports.sqlite", query_text=""):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY,
        created_at TEXT,
        query TEXT,
        kpis_json TEXT,
        dashboard_path TEXT
    )""")
    now = datetime.datetime.utcnow().isoformat()
    cur.execute("INSERT INTO reports (created_at, query, kpis_json, dashboard_path) VALUES (?, ?, ?, ?)",
               (now, query_text, json.dumps(kpis['summary']), dashboard_path))
    conn.commit()
    conn.close()
    return db_path

db_path = save_report_to_memory(query_text=user_query)
print("Saved report to:", db_path)

# Show last 5 reports
conn = sqlite3.connect(db_path)
rows = conn.execute("SELECT id, created_at, query, kpis_json, dashboard_path FROM reports ORDER BY id DESC LIMIT 5").fetchall()
conn.close()

import pandas as pd
history_df = pd.DataFrame(rows, columns=['id','created_at','query','kpis_json','dashboard_path'])
display(history_df)



from PIL import Image as PILImage

thumb_in = dashboard_path
thumb_out = "outputs/thumbnail_560x280.png"

if os.path.exists(thumb_in):
    img = PILImage.open(thumb_in).convert("RGB")
    img_thumb = img.resize((560,280), PILImage.LANCZOS)
    img_thumb.save(thumb_out, format="PNG")
    print("Thumbnail created:", thumb_out)
else:
    print("Dashboard image not found; run the dashboard cell first.")


