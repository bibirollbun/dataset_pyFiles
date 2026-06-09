import os
import sys
import math
import time
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

# Try to import commonly used libraries; install if missing (Kaggle may have most preinstalled)
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception as e:
    print('Some libraries missing, attempting to install...')
    # Install quietly
    !pip install --quiet pandas numpy matplotlib
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

# Optional LLM client libraries will be imported later if keys are present (keeps cold start minimal)

print('Environment ready. Pandas version:', pd.__version__)



@dataclass
class BizConfig:
    date_col: str = 'date'
    product_col: str = 'product_id'
    revenue_col: str = 'revenue'
    units_col: str = 'units_sold'
    category_col: str = 'category'
    product_name_col: str = 'product_name'
    rolling_days: int = 7
    anomaly_z_thresh: float = 2.0

cfg = BizConfig()

# Small utility functions

def reduce_mem_usage(df: pd.DataFrame, verbose: bool=True) -> pd.DataFrame:
    """Attempt to reduce dataframe memory usage by downcasting numeric types."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype('uint8')
                    elif c_max < 65535:
                        df[col] = df[col].astype('uint16')
                    elif c_max < 4294967295:
                        df[col] = df[col].astype('uint32')
                    else:
                        df[col] = df[col].astype('uint64')
                else:
                    if c_min > -32768 and c_max < 32767:
                        df[col] = df[col].astype('int16')
                    elif c_min > -2147483648 and c_max < 2147483647:
                        df[col] = df[col].astype('int32')
                    else:
                        df[col] = df[col].astype('int64')
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Mem usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df

print('Config and utilities loaded')



import glob

# Look for CSV/XLSX files in /kaggle/input
input_dir = '/kaggle/input'
found = []
if os.path.exists(input_dir):
    found = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True) + glob.glob(os.path.join(input_dir, '**', '*.xlsx'), recursive=True)

print('Files found under /kaggle/input:', len(found))
for f in found[:10]:
    print(' -', f)

# Utility: load a preferred dataset automatically

def load_best_dataset():
    """Strategy:
    1) If user attached CSV/XLSX in /kaggle/input, pick the first one
    2) Else attempt to download UCI Online Retail Excel (if internet enabled)
    3) Else fall back to a small synthetic dataset (ensures notebook runs headlessly)
    """
    if found:
        path = found[0]
        print('Using dataset from:', path)
        if path.lower().endswith('.csv'):
            return pd.read_csv(path)
        else:
            return pd.read_excel(path)
    # Try to download UCI Online Retail dataset
    uci_xlsx = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
    try:
        print('No attached dataset found. Attempting to download UCI Online Retail dataset...')
        df = pd.read_excel(uci_xlsx)
        print('Downloaded UCI Online Retail (shape=', df.shape, ')')
        return df
    except Exception as e:
        print('Download failed or internet disabled:', e)
        # Fallback: synthetic dataset
        print('Using synthetic demo dataset (small)')
        df_demo = pd.DataFrame({
            'InvoiceNo':[1001,1002,1003,1004],
            'StockCode':['S001','S002','S003','S004'],
            'Description':['Alpha','Beta','Gamma','Delta'],
            'Quantity':[10,2,5,7],
            'InvoiceDate':[pd.Timestamp('2025-11-01'),pd.Timestamp('2025-11-02'),pd.Timestamp('2025-11-02'),pd.Timestamp('2025-11-03')],
            'UnitPrice':[10.0,20.0,5.0,15.0],
            'CustomerID':[12345,12346,12347,12348],
            'Country':['United Kingdom','United Kingdom','France','Germany']
        })
        df_demo['Revenue'] = df_demo['Quantity'] * df_demo['UnitPrice']
        return df_demo

# Load dataset
raw_df = load_best_dataset()
print('\nPreview:')
print(raw_df.head().to_string(index=False))

# Reduce memory usage but only for compatible columns
try:
    raw_df = reduce_mem_usage(raw_df)
except Exception as e:
    print('Memory reduction skipped due to:', e)

raw_df.shape



# Standardize columns and create unified schema for BI

def prepare_sales_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # lower-case columns for flexible matching
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower():c for c in df.columns}

    # Candidate column names
    date_cols = [k for k in col_map if 'date' in k]
    qty_cols  = [k for k in col_map if 'quantity' in k or 'qty' in k or 'units' in k]
    price_cols = [k for k in col_map if 'price' in k]
    revenue_cols = [k for k in col_map if 'revenue' in k or 'amount' in k or 'total' in k]
    prod_cols = [k for k in col_map if 'stock' in k or 'product' in k or 'sku' in k]
    cat_cols = [k for k in col_map if 'category' in k or 'cat' in k]

    # Map chosen names to standard schema
    # date
    if date_cols:
        df[cfg.date_col] = pd.to_datetime(df[col_map[date_cols[0]]], errors='coerce')
    else:
        df[cfg.date_col] = pd.NaT

    # quantity
    if qty_cols:
        df[cfg.units_col] = pd.to_numeric(df[col_map[qty_cols[0]]], errors='coerce').fillna(0).astype(int)
    else:
        df[cfg.units_col] = 0

    # price
    if price_cols:
        df['unit_price'] = pd.to_numeric(df[col_map[price_cols[0]]], errors='coerce').fillna(0.0)
    else:
        df['unit_price'] = 0.0

    # revenue
    if revenue_cols:
        df[cfg.revenue_col] = pd.to_numeric(df[col_map[revenue_cols[0]]], errors='coerce').fillna(0.0)
    else:
        df[cfg.revenue_col] = df[cfg.units_col] * df['unit_price']

    # product id / name
    if prod_cols:
        df[cfg.product_col] = df[col_map[prod_cols[0]]].astype(str)
    else:
        df[cfg.product_col] = df.index.astype(str)

    # category
    if cat_cols:
        df[cfg.category_col] = df[col_map[cat_cols[0]]].astype(str)
    else:
        df[cfg.category_col] = 'Unknown'

    # product name if available
    name_cols = [k for k in col_map if 'description' in k or 'productname' in k or 'name' in k]
    if name_cols:
        df[cfg.product_name_col] = df[col_map[name_cols[0]]].astype(str)
    else:
        df[cfg.product_name_col] = df[cfg.product_col]

    # Drop obviously invalid rows
    df = df.dropna(subset=[cfg.date_col])

    # Ensure revenue numeric
    df[cfg.revenue_col] = pd.to_numeric(df[cfg.revenue_col], errors='coerce').fillna(0.0)
    df[cfg.units_col] = pd.to_numeric(df[cfg.units_col], errors='coerce').fillna(0).astype(int)

    # Add helper columns
    df['day'] = df[cfg.date_col].dt.date
    df['week'] = df[cfg.date_col].dt.isocalendar().week

    return df

clean_df = prepare_sales_df(raw_df)
print('Cleaned shape:', clean_df.shape)
clean_df.head().T



def compute_kpis(df: pd.DataFrame, cfg: BizConfig) -> dict:
    out = {}
    # overall
    out['total_revenue'] = float(df[cfg.revenue_col].sum())
    out['total_units'] = int(df[cfg.units_col].sum())
    out['order_count'] = int(df.index.nunique())
    out['aov'] = float(out['total_revenue'] / max(1, out['order_count']))

    # by product
    prod = df.groupby(cfg.product_col).agg(
        revenue=(cfg.revenue_col,'sum'),
        units=(cfg.units_col,'sum')
    ).reset_index().sort_values('revenue', ascending=False)
    out['by_product'] = prod

    # daily revenue
    daily = df.groupby(cfg.date_col)[cfg.revenue_col].sum().rename('daily_revenue').reset_index()
    daily['rolling_revenue'] = daily['daily_revenue'].rolling(cfg.rolling_days, min_periods=1).mean()
    # anomaly z-score
    daily['z'] = (daily['daily_revenue'] - daily['daily_revenue'].mean()) / (daily['daily_revenue'].std() + 1e-9)
    out['daily'] = daily
    out['anomalies'] = daily.loc[daily['z'].abs() > cfg.anomaly_z_thresh]

    # by category
    try:
        cat = df.groupby(cfg.category_col).agg(revenue=(cfg.revenue_col,'sum'), units=(cfg.units_col,'sum')).reset_index()
        out['by_category'] = cat.sort_values('revenue', ascending=False)
    except Exception:
        out['by_category'] = pd.DataFrame()

    return out

kpis = compute_kpis(clean_df, cfg)
print('Total revenue:', kpis['total_revenue'])
print('Anomalies found:', len(kpis['anomalies']))
kpis['by_product'].head()



def plot_daily_revenue(daily_df, out_pdf_path=None):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(daily_df[cfg.date_col], daily_df['daily_revenue'], marker='o', linewidth=1)
    ax.plot(daily_df[cfg.date_col], daily_df['rolling_revenue'], linestyle='--', label=f'{cfg.rolling_days}-day avg')
    ax.set_title('Daily Revenue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if out_pdf_path:
        plt.savefig(out_pdf_path.replace('.pdf', '_daily.png'))
    plt.show()

plot_daily_revenue(kpis['daily'])



import json
import random

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')  # change to 'gemini' or '' to disable

# Attempt to get keys
try:
    OPENAI_KEY = user_secrets.get_secret('OPENAI_API_KEY') if 'user_secrets' in globals() else os.getenv('OPENAI_API_KEY')
except Exception:
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
try:
    GEMINI_KEY = user_secrets.get_secret('GEMINI_API_KEY') if 'user_secrets' in globals() else os.getenv('GEMINI_API_KEY')
except Exception:
    GEMINI_KEY = os.getenv('GEMINI_API_KEY')

print('LLM_PROVIDER=', LLM_PROVIDER)

# Lightweight adapter
def llm_call(prompt: str, provider: str=LLM_PROVIDER, model: str=None, max_tokens: int=300, temperature: float=0.0):
    """Generic LLM call wrapper. Returns text or raises an exception.
    Implementations:
      - OpenAI via openai.ChatCompletion
      - Gemini via google.generativeai (if configured)
    """
    if not provider:
        raise RuntimeError('LLM provider not configured')

    # Small retry/backoff
    attempts = 0
    while attempts < 3:
        try:
            if provider == 'openai':
                if not OPENAI_KEY:
                    raise RuntimeError('OPENAI API key not found')
                import openai
                openai.api_key = OPENAI_KEY
                model = model or 'gpt-4'
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{'role':'system','content':'You are a business analyst.'}, {'role':'user','content':prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return resp.choices[0].message.content.strip()

            if provider == 'gemini':
                if not GEMINI_KEY:
                    raise RuntimeError('GEMINI API key not found')
                # Using Google generative AI library if available
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_KEY)
                model = model or 'gemini-1.5'
                resp = genai.chat.completions.create(model=model, messages=[{'author':'user','content':prompt}])
                return resp['candidates'][0]['content']

            raise RuntimeError('Unsupported LLM provider')
        except Exception as e:
            attempts += 1
            wait = 2 ** attempts
            logging.warning(f'LLM call attempt {attempts} failed: {e}. Retrying in {wait}s...')
            time.sleep(wait)
    # If all retries fail, raise
    raise RuntimeError('LLM calls failed after retries')

# Example quick test (commented out by default)
# try:
#     print(llm_call('Say hello in one sentence.', provider='openai'))
# except Exception as e:
#     print('LLM test skipped:', e)

print('LLM adapter ready (will be used only when you set LLM_PROVIDER and keys)')



import json

def template_insights(kpis: dict) -> list:
    res = []
    res.append({'severity':'info', 'insight':f"Total revenue: ${kpis['total_revenue']:.2f}", 'evidence':''})
    res.append({'severity':'info', 'insight':f"Total units sold: {kpis['total_units']}", 'evidence':''})
    res.append({'severity':'info', 'insight':f"Average order value: ${kpis['aov']:.2f}", 'evidence':''})
    if not kpis['anomalies'].empty:
        for _, r in kpis['anomalies'].iterrows():
            res.append({'severity':'high', 'insight':f"Anomaly on {r[cfg.date_col].date()}: ${r['daily_revenue']:.2f}", 'evidence':f"z={r['z']:.2f}"})
    # top product
    if not kpis['by_product'].empty:
        top = kpis['by_product'].iloc[0]
        res.append({'severity':'high', 'insight':f"Top product: {top[cfg.product_col]} with revenue ${top['revenue']:.2f}", 'evidence':''})
    return res


def insights_agent(kpis: dict, use_llm: bool=True) -> list:
    if use_llm:
        try:
            # Build a compact prompt
            top_products = kpis['by_product'].head(5).to_dict(orient='records')
            recent = kpis['daily'].tail(14)[[cfg.date_col, 'daily_revenue']].to_dict(orient='records')
            prompt = (
                'You are a senior business analyst. Given the KPIs and recent daily revenue, produce up to 5 concise, prioritized insights in JSON array. '\
                'Each item should be {"severity":"critical|high|medium|low","insight":"...","evidence":"..."}. '\
                f'KPIs: total_revenue={kpis["total_revenue"]:.2f}, total_units={kpis["total_units"]}, aov={kpis["aov"]:.2f}. '\
                'Top products: ' + json.dumps(top_products) + '\n' + 'Recent daily: ' + json.dumps(recent)
            )
            text = llm_call(prompt)
            # Try to parse JSON from model
            try:
                parsed = json.loads(text)
                return parsed
            except Exception:
                # If parsing fails, wrap raw text
                return [{'severity':'low','insight':text,'evidence':'llm_raw'}]
        except Exception as e:
            logging.warning('LLM insights failed: %s', e)
            return template_insights(kpis)
    else:
        return template_insights(kpis)


def recommendation_agent(insights: list, use_llm: bool=True) -> list:
    if use_llm:
        try:
            prompt = 'Given these insights, suggest 3 actionable recommendations with impact and effort estimates in JSON array: ' + json.dumps(insights)
            text = llm_call(prompt)
            try:
                return json.loads(text)
            except Exception:
                return [{'action':i.get('insight')[:120], 'impact':'medium', 'effort':'low', 'rationale':i.get('evidence','')} for i in insights[:3]]
        except Exception as e:
            logging.warning('LLM recommendations failed: %s', e)
            return [{'action':'Focus on top product','impact':'high','effort':'low','rationale':'Top revenue contributor'}]
    else:
        return [{'action':'Focus on top product','impact':'high','effort':'low','rationale':'Top revenue contributor'}]

print('Insight & Recommendation agents ready')



def generate_markdown_report(kpis, insights, recs, out_path='bizintel_report.md'):
    parts = []
    parts.append('# BizIntel Report')
    parts.append(f'**Period:** {kpis["daily"][cfg.date_col].min().date() if not kpis["daily"].empty else "N/A"} - {kpis["daily"][cfg.date_col].max().date() if not kpis["daily"].empty else "N/A"}')
    parts.append('\n## Key Metrics')
    parts.append(f'- Total revenue: ${kpis["total_revenue"]:.2f}')
    parts.append(f'- Total units: {kpis["total_units"]}')
    parts.append(f'- AOV: ${kpis["aov"]:.2f}')
    parts.append('\n## Insights')
    for it in insights:
        parts.append(f'- [{it.get("severity")}] {it.get("insight")} ({it.get("evidence","")})')
    parts.append('\n## Recommendations')
    for r in recs:
        if isinstance(r, dict):
            parts.append(f'- {r.get("action")} (impact: {r.get("impact")}, effort: {r.get("effort")})')
        else:
            parts.append(f'- {str(r)}')
    text = '\n\n'.join(parts)
    with open(out_path, 'w') as f:
        f.write(text)
    print('Wrote', out_path)
    return out_path


def save_pdf_plots(kpis, pdf_path='bizintel_report.pdf'):
    with PdfPages(pdf_path) as pdf:
        # daily revenue
        fig, ax = plt.subplots(figsize=(10,5))
        d = kpis['daily']
        ax.plot(d[cfg.date_col], d['daily_revenue'], marker='o')
        ax.plot(d[cfg.date_col], d['rolling_revenue'], linestyle='--')
        ax.set_title('Daily Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # top products
        fig, ax = plt.subplots(figsize=(8,5))
        top = kpis['by_product'].head(10)
        ax.barh(top[cfg.product_col], top['revenue'])
        ax.invert_yaxis()
        ax.set_title('Top Products by Revenue')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print('Saved PDF:', pdf_path)


# Pipeline runner

def run_pipeline(df, use_llm=False):
    dfc = prepare_sales_df(df)
    kpis = compute_kpis(dfc, cfg)
    insights = insights_agent(kpis, use_llm=use_llm)
    recs = recommendation_agent(insights, use_llm=use_llm)
    md = generate_markdown_report(kpis, insights, recs, out_path='bizintel_report.md')
    save_pdf_plots(kpis, pdf_path='bizintel_report.pdf')
    return {'kpis':kpis, 'insights':insights, 'recommendations':recs, 'md':md}

# Run the pipeline (default: LLM disabled for safe versioning)
results = run_pipeline(clean_df, use_llm=False)
print('\nInsights:')
for it in results['insights'][:10]:
    print('-', it)

print('\nRecommendations:')
for r in results['recommendations']:
    print('-', r)


