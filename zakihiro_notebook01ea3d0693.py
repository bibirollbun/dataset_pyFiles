# å¿…è¦�ãƒ©ã‚¤ãƒ–ãƒ©ãƒªï¼ˆåˆ�å›�ã�®ã�¿ï¼‰
!pip install -q feedparser beautifulsoup4 lxml requests

#!/usr/bin/env python3

import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from urllib.parse import quote_plus

import feedparser
import pandas as pd
from bs4 import BeautifulSoup

socket.setdefaulttimeout(10)

# -----------------------------
# ã‚»ã‚¯ã‚¿ãƒ¼ â†’ ãƒ†ã‚£ãƒƒã‚«ãƒ¼ï¼ˆæŠœç²‹ï¼‰
# -----------------------------
sp500_by_sector = {
    "Information Technology": ["AAPL", "MSFT", "NVDA"],
    "Health Care": ["JNJ", "PFE", "ABBV"],
    "Financials": ["JPM", "BAC", "WFC"],
    # ä»–ã‚»ã‚¯ã‚¿ãƒ¼ã‚‚å¿…è¦�ã�ªã‚‰è¿½åŠ 
}

# ä½•ä»¶ã�«ã�™ã‚‹ã�‹ï¼ˆNone ã�ªã‚‰å…¨ãƒ†ã‚£ãƒƒã‚«ãƒ¼ï¼‰
PER_SECTOR_LIMIT = 5   # ä¾‹: 3 ã�«ã�™ã‚‹ã�¨å�„ã‚»ã‚¯ã‚¿ãƒ¼3éŠ˜æŸ„ã� ã�‘

ALL_RSS_FEEDS = []
for sector, tickers in sp500_by_sector.items():
    use_tickers = tickers if PER_SECTOR_LIMIT is None else tickers[:PER_SECTOR_LIMIT]
    for ticker in use_tickers:
        q = quote_plus(f"{ticker} stock")  # ã‚¯ã‚¨ãƒªã‚’URLã‚¨ãƒ³ã‚³ãƒ¼ãƒ‰
        ALL_RSS_FEEDS.append({
            "sector": sector,
            "ticker": ticker,
            "url": f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en",
        })

print(f"Feeds: {len(ALL_RSS_FEEDS)}")

# -----------------------------
# ãƒ¦ãƒ¼ãƒ†ã‚£ãƒªãƒ†ã‚£
# -----------------------------
def clean_html(html_text: str) -> str:
    """HTMLã‚¿ã‚°é™¤å�» + ç©ºç™½æ­£è¦�åŒ–"""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text()
    return re.sub(r"\s+", " ", text).strip()

def fetch_feed(feed_info: dict, max_items: int = 50):
    url = feed_info["url"]
    sector = feed_info["sector"]
    ticker = feed_info["ticker"]

    rows = []
    try:
        feed = feedparser.parse(url)
        for e in feed.entries[:max_items]:
            rows.append(
                {
                    "sector": sector,
                    "ticker": ticker,
                    "title": e.get("title", "") or "",
                    "summary": clean_html(e.get("summary", "") or ""),
                    "link": e.get("link", "") or "",
                    "published": e.get("published", "") or "",
                }
            )
    except Exception as ex:
        print(f"â�Œ Error fetching {ticker} ({sector}): {ex}")
    return rows

def download_all_feeds(feeds: list, max_workers: int = 10, max_items: int = 20):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_feed, f, max_items): f for f in feeds}
        for fut in as_completed(futures):
            results.extend(fut.result())
    return results

# -----------------------------
# FinBERT æ�¨è«–ï¼ˆProsusAI/finbertï¼‰
# -----------------------------
from transformers import pipeline
import torch

def build_finbert_pipeline(model_name: str = "ProsusAI/finbert"):
    # GPUã�Œã�‚ã‚Œã�°è‡ªå‹•ã€�ã�ªã�‘ã‚Œã�°CPU
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
        return_all_scores=True,   # â†� pos/neu/neg ã�®å…¨ç¢ºç�‡ã‚’å�–å¾—
        truncation=True
    )
    return clf

def run_finbert(df: pd.DataFrame, text_col: str, batch_size: int = 32, max_length: int = 256):
    """
    df[text_col] ã‚’ FinBERT ã�§ãƒ�ãƒƒãƒ�æ�¨è«–ã�—ã€�ç¢ºç�‡ã�¨ã‚¹ã‚³ã‚¢åˆ—ã‚’è¿½åŠ ã�—ã�¦è¿”ã�™ã€‚
    è¿½åŠ åˆ—:
      - finbert_p_pos, finbert_p_neu, finbert_p_neg
      - finbert_label  (æœ€å°¤ãƒ©ãƒ™ãƒ«)
      - finbert_sentiment_score  (-1..+1)
    """
    # ç©ºæ–‡å­—ã�¯é�¿ã�‘ã‚‹
    texts = df[text_col].fillna("").astype(str).tolist()

    clf = build_finbert_pipeline()
    results = []
    # pipeline ã�® batched æ�¨è«–
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = clf(batch, truncation=True, max_length=max_length)
        results.extend(out)

    # çµ�æ�œã‚’ãƒ‘ãƒ¼ã‚¹
    p_pos, p_neu, p_neg, label, score = [], [], [], [], []
    val_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    for item in results:
        # item ã�¯ [{'label': 'positive', 'score': 0.98}, {'label': 'negative',...}, {'label': 'neutral',...}] ã�®ãƒªã‚¹ãƒˆ
        probs = {d["label"].lower(): float(d["score"]) for d in item}
        # ãƒ©ãƒ™ãƒ«æ¬ æ��ã�«å‚™ã�ˆã�¦æ—¢å®šå€¤0
        pos = probs.get("positive", 0.0)
        neu = probs.get("neutral", 0.0)
        neg = probs.get("negative", 0.0)
        p_pos.append(pos); p_neu.append(neu); p_neg.append(neg)

        # æœ€å°¤ãƒ©ãƒ™ãƒ«
        top_label = max(probs.items(), key=lambda kv: kv[1])[0] if probs else "neutral"
        label.append(top_label)

        # ã‚¹ã‚«ãƒ©ãƒ¼åŒ–ï¼ˆ-1..+1ï¼‰
        score.append(pos * 1.0 + neu * 0.0 + neg * (-1.0))

    df = df.copy()
    df["finbert_p_pos"] = p_pos
    df["finbert_p_neu"] = p_neu
    df["finbert_p_neg"] = p_neg
    df["finbert_label"] = label
    df["finbert_sentiment_score"] = score
    return df

# -----------------------------
# å®Ÿè¡Œéƒ¨åˆ†
# -----------------------------
if __name__ == "__main__":
    items = download_all_feeds(ALL_RSS_FEEDS, max_workers=10, max_items=20)
    df = pd.DataFrame(items, columns=["sector", "ticker", "title", "summary", "link", "published"])
    print("Before sentiment:")
    print(df.head())

    # ã‚¿ã‚¤ãƒˆãƒ«ï¼‹è¦�ç´„ã�§ãƒ†ã‚­ã‚¹ãƒˆã‚’æ§‹ç¯‰ï¼ˆé‡�è¤‡ãƒ»ç©ºç™½ã‚’æ•´ç�†ï¼‰
    df["text_for_sent"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.replace(r"\s+", " ", regex=True).str.strip()

    # FinBERT æ�¨è«–ï¼ˆãƒ�ãƒƒãƒ�å‡¦ç�†ï¼‰
    if len(df):
        df = run_finbert(df, text_col="text_for_sent", batch_size=32, max_length=256)
    else:
        df["finbert_p_pos"] = df["finbert_p_neu"] = df["finbert_p_neg"] = df["finbert_sentiment_score"] = 0.0
        df["finbert_label"] = ""

    # ä»•ä¸Šã�’ï¼šä¸»è¦�åˆ—ã‚’ä¸¦ã�¹æ›¿ã�ˆ
    cols = ["sector","ticker","title","summary","link","published",
            "finbert_label","finbert_p_pos","finbert_p_neu","finbert_p_neg","finbert_sentiment_score"]
    df = df[cols]

    print("\nAfter FinBERT inference:")
    print(df.head())
    print(f"\nâœ… Collected {len(df)} news items with FinBERT sentiment")

from google.cloud import bigquery
import pandas as pd

# ==== è¨­å®š ====
PROJECT_ID = "artful-reef-472804-k8"
DATASET_ID = "news_data"
TABLE_ID   = "quick_summary"

client = bigquery.Client(project=PROJECT_ID)

# -----------------------------
# ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆä½œæˆ�ï¼ˆå­˜åœ¨ã�—ã�ªã�‘ã‚Œã�°ï¼‰
# -----------------------------
dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
dataset = bigquery.Dataset(dataset_ref)
dataset.location = "us-central1"

try:
    client.get_dataset(dataset_ref)  # å­˜åœ¨ç¢ºèª�
    print(f"âœ… Dataset already exists: {DATASET_ID}")
except Exception:
    client.create_dataset(dataset)   # ç„¡ã�‘ã‚Œã�°ä½œæˆ�
    print(f"ğŸ†• Created dataset: {DATASET_ID}")

# -----------------------------
# DataFrame â†’ BigQueryã�«ãƒ­ãƒ¼ãƒ‰
# -----------------------------
table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND  # è¿½è¨˜ã€‚ãƒ†ãƒ¼ãƒ–ãƒ«ã�Œç„¡ã�‘ã‚Œã�°è‡ªå‹•ä½œæˆ�ã�•ã‚Œã‚‹
)

load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
load_job.result()

print(f"âœ… Loaded {len(df)} rows into {table_ref}")

import yfinance as yf
import pandas as pd

# ===== Date range and tickers per sector =====
start_date = "2020-01-01"
end_date   = "2020-12-31"

sp500_by_sector = {
    "Information Technology": ["AAPL", "MSFT", "NVDA"],
    "Health Care": ["JNJ", "PFE", "ABBV"],
    "Financials": ["JPM", "BAC", "WFC"],
    # Add more sectors/tickers if needed
}

# Map ticker -> sector and build the complete ticker list
ticker_to_sector = {t: sector for sector, lst in sp500_by_sector.items() for t in lst}
tickers = sorted(ticker_to_sector.keys())

# ===== Fetch prices (columns are a MultiIndex: (Ticker, Attribute)) =====
raw = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False,
    progress=False,
    group_by="ticker",
)

# ===== Convert to a long (row-concatenated) format and add Date/Sector/Ticker =====
# raw.columns: MultiIndex(levels=[Ticker, Attribute])
# â†’ stack(level=0) to move Ticker to rows
long_df = (
    raw
    .stack(level=0)                                       # index: Date, Ticker / columns: Attribute
    .rename_axis(index=["Date", "Ticker"], columns="Attribute")
    .reset_index()                                        # Date, Ticker, Open, High, Low, Close, Adj Close, Volume
)

# Add Sector column
long_df.insert(1, "Sector", long_df["Ticker"].map(ticker_to_sector))

# Reorder columns (keep only those that exist)
desired = ["Date", "Sector", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
long_df = long_df.reindex(columns=[c for c in desired if c in long_df.columns])

# Optional sorting
long_df = long_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

import gradio as gr
import plotly.graph_objects as go
from datetime import timedelta


class SentimentDashboard:
    def __init__(self, df_prices: pd.DataFrame, df_news: pd.DataFrame):
        """
        df_prices: price data
            required columns: ['Date','Sector','Ticker','Open','High','Low','Close','Adj Close','Volume']
        df_news: news data
            required columns: ['sector','ticker','title','summary','link','published','finbert_sentiment_score']
        """
        # Validate price data
        required = ['Date','Sector','Ticker','Open','High','Low','Close','Adj Close','Volume']
        miss = [c for c in required if c not in df_prices.columns]
        if miss:
            raise ValueError(f"Missing required columns in price data: {miss}")

        self.df_prices = df_prices.copy()
        self.df_prices['Date'] = pd.to_datetime(self.df_prices['Date'], errors='coerce')
        self.df_prices = self.df_prices.dropna(subset=['Date']).sort_values(['Ticker','Date']).reset_index(drop=True)

        # Validate news data
        news_required = ['sector','ticker','title','summary','link','published','finbert_sentiment_score']
        miss_news = [c for c in news_required if c not in df_news.columns]
        if miss_news:
            raise ValueError(f"Missing required columns in news data: {miss_news}")

        self.df_news = df_news.copy()
        self.df_news['published'] = pd.to_datetime(self.df_news['published'], errors='coerce')

        self.available_tickers = sorted(self.df_prices['Ticker'].dropna().unique().tolist())
        self.stocks_data = self._build_sentiment_from_news()

        # Aggregation tables
        self.sector_sentiment = self.df_news.groupby("sector")["finbert_sentiment_score"].mean().sort_values(ascending=False)
        self.ticker_sentiment = self.df_news.groupby("ticker")["finbert_sentiment_score"].mean().sort_values(ascending=False)

    def _build_sentiment_from_news(self):
        """Build per-ticker sentiment info from news."""
        stocks_data = {}
        for ticker in self.available_tickers:
            sec = self.df_prices.loc[self.df_prices['Ticker']==ticker, 'Sector'].iloc[0]
            news_df = self.df_news[self.df_news['ticker']==ticker].sort_values('published', ascending=False)

            if news_df.empty:
                sentiment_score = 0.0
                trend = [0.0]*7
                news_items = []
            else:
                sentiment_score = news_df['finbert_sentiment_score'].mean()
                trend = news_df['finbert_sentiment_score'].head(7).tolist()
                news_items = []
                for _, row in news_df.head(5).iterrows():
                    score = float(row['finbert_sentiment_score'])
                    typ = "positive" if score > 0.2 else ("negative" if score < -0.2 else "neutral")
                    news_items.append(
                        (row['title'], score, typ,
                         row['published'].strftime("%Y-%m-%d %H:%M") if not pd.isna(row['published']) else "",
                         row['link'], row['summary'])
                    )

            stocks_data[ticker] = {
                'name': f"{ticker} Corp.",
                'sector': sec,
                'sentiment': sentiment_score,
                'trend': trend,
                'news': news_items
            }
        return stocks_data

    def filter_data_by_timerange(self, ticker, time_range):
        tdf = self.df_prices[self.df_prices['Ticker']==ticker].copy().sort_values('Date')
        if tdf.empty:
            return tdf
        end_date = tdf['Date'].max()
        if time_range == '1D':   start_date = end_date - timedelta(days=1)
        elif time_range == '5D': start_date = end_date - timedelta(days=5)
        elif time_range == '1M': start_date = end_date - timedelta(days=30)
        elif time_range == '3M': start_date = end_date - timedelta(days=90)
        elif time_range == '1Y': start_date = end_date - timedelta(days=365)
        else:                    start_date = tdf['Date'].min()
        out = tdf[tdf['Date'] >= start_date]
        if out.empty:
            out = tdf.tail(60)
        return out

    def create_stock_chart(self, ticker, time_range):
        """Create a candlestick chart with Plotly."""
        data = self.filter_data_by_timerange(ticker, time_range)
        if data.empty: return None
        fig = go.Figure(data=go.Candlestick(
            x=data['Date'], open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name=f'{ticker} Price',
            increasing_line_color='#10b981', decreasing_line_color='#ef4444'
        ))
        if len(data) > 20:
            fig.add_trace(go.Scatter(
                x=data['Date'], y=data['Close'].rolling(20).mean(),
                mode='lines', name='20-day MA', line=dict(width=2)
            ))
        if len(data) > 50:
            fig.add_trace(go.Scatter(
                x=data['Date'], y=data['Close'].rolling(50).mean(),
                mode='lines', name='50-day MA', line=dict(width=2)
            ))
        stock_name = self.stocks_data.get(ticker, {}).get('name', f'{ticker} Corp.')
        fig.update_layout(
            title=f'{stock_name} ({ticker}) - {time_range}',
            xaxis_title='Date', yaxis_title='Price ($)',
            template='plotly_white', height=460,
            showlegend=True, xaxis_rangeslider_visible=False
        )
        return fig

    def get_current_price_info(self, ticker):
        tdf = self.df_prices[self.df_prices['Ticker']==ticker].sort_values('Date')
        if len(tdf) < 2: return None
        latest, prev = tdf.iloc[-1], tdf.iloc[-2]
        price = float(latest['Close']); prev_price = float(prev['Close'])
        change = price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price else 0.0
        return {'price': price,'change': change,'change_pct': change_pct,
                'volume': int(latest['Volume']),'date': latest['Date']}

    def get_dashboard_html(self, selected_stock, chart_range, news_range):
        if selected_stock not in self.stocks_data:
            return "<div>Selected ticker not found.</div>"
        stock = self.stocks_data[selected_stock]
        pi = self.get_current_price_info(selected_stock)
        if not pi:
            return "<div>Could not compute price info.</div>"

        # === News time-window filter ===
        end_date = self.df_news['published'].max()
        if news_range == '1D':
            start_date = end_date - pd.Timedelta(days=1)
        elif news_range == '5D':
            start_date = end_date - pd.Timedelta(days=5)
        elif news_range == '1M':
            start_date = end_date - pd.Timedelta(days=30)
        elif news_range == '3M':
            start_date = end_date - pd.Timedelta(days=90)
        elif news_range == '1Y':
            start_date = end_date - pd.Timedelta(days=365)
        else:
            start_date = self.df_news['published'].min()

        df_news_range = self.df_news[(self.df_news['published'] >= start_date) & (self.df_news['published'] <= end_date)]

        # === Recompute aggregates (within the selected window) ===
        sector_sentiment = df_news_range.groupby("sector")["finbert_sentiment_score"].mean().sort_values(ascending=False)
        ticker_sentiment = df_news_range.groupby("ticker")["finbert_sentiment_score"].mean().sort_values(ascending=False)

        # === News (latest 5 + Top 10 in window) ===
        news_df = df_news_range[df_news_range['ticker'] == selected_stock].copy()
        latest_news = news_df.sort_values("published", ascending=False).head(5)
        top_news = news_df.reindex(news_df['finbert_sentiment_score'].abs().sort_values(ascending=False).index).head(10)

        def render_news(df, title):
            items = "".join(
                f"""<div class=\"news-item {'positive' if s>0.2 else 'negative' if s<-0.2 else 'neutral'}\">\n
                       <div class=\"news-title\"><a href=\"{row['link']}\" target=\"_blank\">{row['title']}</a></div>\n
                       <div class=\"news-meta\"><span>{row['published'].strftime('%Y-%m-%d %H:%M')}</span><span style=\"font-weight:600\">{s:+.2f}</span></div>\n
                       <div class=\"news-summary\">{row['summary']}</div>\n
                     </div>"""
                for _, row in df.iterrows() for s in [float(row['finbert_sentiment_score'])]
            )
            return f"<h5 class='subhead'>{title}</h5>{items}"

        news_html = render_news(latest_news, "Latest News (Last 5)") + render_news(top_news, f"Top 10 in Window ({news_range})")

        # === Tiles ===
        def score_class(v):
            if v >= 0.6: return "strong-positive"
            if v >= 0.2: return "positive"
            if v > -0.2: return "neutral"
            if v > -0.6: return "negative"
            return "strong-negative"

        sector_tiles = "".join(
            f'<div class="tile {score_class(val)}"><div class="name">{sec}</div><div class="score">{float(val):+.2f}</div></div>'
            for sec, val in sector_sentiment.items()
        )
        ticker_tiles = "".join(
            f'<div class="tile {score_class(val)}"><div class="name">{tic}</div><div class="score">{float(val):+.2f}</div></div>'
            for tic, val in ticker_sentiment.items()
        )
        competitor_list = "".join(
            f'<div class="row"><div class="ticker">{tic}</div>'
            f'<div class="meter-wrap"><div class="meter"><div class="bar" style="width:{max(0.0, float(val))*100:.0f}%"></div></div>'
            f'<div class="val">{float(val):+.2f}</div></div></div>'
            for tic, val in ticker_sentiment.head(4).items()
        )

        # === HTML + CSS (note: curly braces escaped as {{ }}) ===
        return f"""
        <div class="wrap" style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans',sans-serif;color:#0f172a">
          <style>
            .wrap{{background:linear-gradient(180deg,#eef2ff,#f8fafc 30%,#f1f5f9);padding:8px 0}}
            .subhead{{margin:14px 0 8px;color:#334155}}
            /* Tiles */
            .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
            .tile{{border-radius:14px;color:#fff;padding:16px;display:flex;flex-direction:column;justify-content:center;align-items:center;min-height:82px;box-shadow:inset 0 -24px 40px rgba(0,0,0,.08)}}
            .tile .name{{font-size:13px;margin-bottom:6px;opacity:.95}}
            .tile .score{{font-size:18px;font-weight:700}}
            .tile.strong-positive{{background:#059669}}
            .tile.positive{{background:#10b981}}
            .tile.neutral{{background:#f59e0b}}
            .tile.negative{{background:#dc2626}}
            .tile.strong-negative{{background:#991b1b}}
            /* Competitor list */
            .list{{display:flex;flex-direction:column;gap:10px;margin-top:10px}}
            .row{{display:grid;grid-template-columns:1fr 180px;align-items:center;background:#fff;border-radius:14px;padding:12px 14px;border:1px solid rgba(2,6,23,.06)}}
            .meter-wrap{{display:grid;grid-template-columns:1fr 64px;gap:10px;align-items:center}}
            .meter{{position:relative;height:8px;background:#e2e8f0;border-radius:999px;overflow:hidden}}
            .bar{{position:absolute;height:100%;background:linear-gradient(90deg,#22c55e,#10b981)}}
            .val{{text-align:right;font-weight:700;color:#16a34a}}
            /* News (light backgrounds) */
            .news-item{{padding:12px;border-radius:10px;margin-bottom:12px;border-left:4px solid}}
            .news-item.positive{{background:rgba(16,185,129,.1);border-left-color:#10b981}}
            .news-item.negative{{background:rgba(239,68,68,.1);border-left-color:#ef4444}}
            .news-item.neutral{{background:rgba(245,158,11,.1);border-left-color:#f59e0b}}
            .news-title a{{color:#0f172a;text-decoration:underline}}
            .news-meta{{display:flex;justify-content:space-between;font-size:12px;color:#64748b}}
            .news-summary{{font-size:12px;color:#475569;margin-top:4px}}
            @media (max-width:760px){{ .grid{{grid-template-columns:repeat(2,1fr)}} .row{{grid-template-columns:1fr}} }}
          </style>

          <h3 style="margin:0 0 6px">{stock['name']} ({selected_stock})</h3>
          <div style="color:#64748b">Sector: {stock['sector']} | Last Update: {pi['date'].strftime('%Y-%m-%d')}</div>
          <div style="margin-top:10px;font-size:28px;font-weight:700;color:#10b981">${pi['price']:.2f}</div>
          <div>{pi['change']:+.2f} ({pi['change_pct']:+.2f}%)</div>

          <h4 style="margin-top:18px">Sector Sentiment</h4>
          <div class="grid">{sector_tiles}</div>

          <h4 style="margin-top:18px">Ticker Sentiment</h4>
          <div class="grid">{ticker_tiles}</div>

          <h4 style="margin-top:18px">Competitor Sentiment (Top 4)</h4>
          <div class="list">{competitor_list}</div>

          <h4 style="margin-top:18px">News</h4>
          {news_html}
        </div>
        """

# Gradio interface

def create_gradio_interface(df_prices: pd.DataFrame, df_news: pd.DataFrame):
    """Build and return a Gradio demo app.

    Notes:
      - df_prices must contain ['Date','Sector','Ticker','Open','High','Low','Close','Adj Close','Volume']
      - df_news must contain ['sector','ticker','title','summary','link','published','finbert_sentiment_score']
    """
    dashboard = SentimentDashboard(df_prices, df_news)

    def update_dashboard_and_charts(stock_symbol, chart_range, news_range):
        if not stock_symbol or stock_symbol not in dashboard.available_tickers:
            return go.Figure(), "<div>No ticker selected.</div>"
        fig = dashboard.create_stock_chart(stock_symbol, chart_range)
        html = dashboard.get_dashboard_html(stock_symbol, chart_range, news_range)
        return fig, html

    tickers = dashboard.available_tickers
    initial_ticker = tickers[0] if tickers else None

    with gr.Blocks(title="Investor Sentiment Dashboard") as demo:
        gr.Markdown("# ğŸ“ˆ Investor Sentiment Dashboard (with time-windowed news)")

        with gr.Row():
            stock_dropdown = gr.Dropdown(choices=tickers, label="Ticker", value=initial_ticker)
            chart_range_dropdown = gr.Dropdown(
                choices=[("1D","1D"),("5D","5D"),("1M","1M"),("3M","3M"),("1Y","1Y")],
                label="Price Range", value="1M"
            )
            news_range_dropdown = gr.Dropdown(
                choices=[("1D","1D"),("5D","5D"),("1M","1M"),("3M","3M"),("1Y","1Y"),("ALL","ALL")],
                label="News Range", value="1M"
            )

        stock_chart = gr.Plot(
            value=dashboard.create_stock_chart(initial_ticker, "1M") if initial_ticker else go.Figure(),
            label="Price Chart"
        )
        dashboard_html = gr.HTML(
            value=dashboard.get_dashboard_html(initial_ticker, "1M", "1M") if initial_ticker else "<div>No data.</div>",
            label="Overview"
        )

        for c in (stock_dropdown, chart_range_dropdown, news_range_dropdown):
            c.change(
                fn=update_dashboard_and_charts,
                inputs=[stock_dropdown, chart_range_dropdown, news_range_dropdown],
                outputs=[stock_chart, dashboard_html]
            )
    return demo

# Example runner
if __name__ == "__main__":
    demo = create_gradio_interface(long_df, df)
    demo.launch(share=True)



from IPython.display import HTML

# HTMLã‚³ãƒ¼ãƒ‰ã‚’å¤‰æ•°ã�«ä¿�å­˜
html_content = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>æŠ•è³‡ã‚»ãƒ³ãƒ�ãƒ¡ãƒ³ãƒˆåˆ†æ��ãƒ€ãƒƒã‚·ãƒ¥ãƒœãƒ¼ãƒ‰</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.min.js"></script>
    <!-- æ™‚ç³»åˆ—ã‚¹ã‚±ãƒ¼ãƒ«ç”¨ã‚¢ãƒ€ãƒ—ã‚¿ -->
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f7fa; padding: 20px; line-height: 1.6;
        }
        .container {
            max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); overflow: hidden;
        }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
        .header h1 { font-size: 24px; margin-bottom: 10px; }
        .content { padding: 30px; }
        .stock-info {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;
            padding: 20px; background: #f8f9fc; border-radius: 8px;
        }
        .stock-details h2 { font-size: 28px; color: #2d3748; }
        .stock-details p { color: #718096; margin: 5px 0; }
        .stock-price { text-align: right; }
        .price { font-size: 36px; font-weight: bold; color: #2d3748; }
        .price-change { color: #e53e3e; font-size: 18px; margin-top: 5px; }
        .chart-container {
            margin: 30px 0; padding: 20px; background: white; border: 1px solid #e2e8f0; border-radius: 8px;
        }
        .chart { width: 100%; height: 400px; position: relative; }
        .sectors { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 30px 0; }
        .sector-box { padding: 20px; border-radius: 8px; text-align: center; color: white; font-weight: bold; }
        .sector-it, .sector-health, .sector-finance { background: linear-gradient(135deg, #ff9500 0%, #ff7b00 100%); }
        .recommendations { margin: 30px 0; }
        .recommendations h3 { margin-bottom: 20px; color: #2d3748; font-size: 20px; }
        .rec-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
        .rec-item { padding: 15px; border-radius: 8px; text-align: center; color: white; font-weight: bold; font-size: 14px; }
        .rec-strong-buy { background: #38a169; } .rec-buy { background: #68d391; } .rec-hold { background: #ff9500; }
        .pe-table { margin: 30px 0; }
        .pe-table h3 { margin-bottom: 20px; color: #2d3748; font-size: 20px; }
        .pe-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .pe-item {
            display: flex; justify-content: space-between; align-items: center; padding: 15px;
            background: #f7fafc; border-radius: 6px; border-left: 4px solid #38a169;
        }
        .news-section { margin-top: 30px; }
        .news-section h3 { margin-bottom: 20px; color: #2d3748; font-size: 20px; }
        .news-item { padding: 15px; margin-bottom: 10px; background: #f7fafc; border-radius: 6px; border-left: 4px solid #4299e1; }
        .news-date { color: #718096; font-size: 12px; margin-bottom: 5px; }
        .news-title { color: #2d3748; font-size: 14px; margin-bottom: 5px; }
        .news-impact { float: right; padding: 2px 8px; border-radius: 12px; font-size: 12px; color: white; }
        .impact-negative { background: #e53e3e; } .impact-positive { background: #38a169; } .impact-neutral { background: #718096; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ğŸ“Š æŠ•è³‡ã‚»ãƒ³ãƒ�ãƒ¡ãƒ³ãƒˆåˆ†æ��ãƒ€ãƒƒã‚·ãƒ¥ãƒœãƒ¼ãƒ‰ï¼ˆé–¢é€£éŠ˜æŸ„ãƒ‹ãƒ¥ãƒ¼ã‚¹é�‹ç”¨ç‰ˆï¼‰</h1>
        </div>
        
        <div class="content">
            <div class="stock-info">
                <div class="stock-details">
                    <h2>AAPL Corp. (AAPL)</h2>
                    <p>ã‚»ã‚¯ã‚¿ãƒ¼: Information Technology | æ¥­ç¨®ã‚³ãƒ¼ãƒ‰: 2020-12-30</p>
                </div>
                <div class="stock-price">
                    <div class="price">$133.72</div>
                    <div class="price-change">-1.15 (-0.85%)</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3 style="margin-bottom: 15px; color: #2d3748;">AAPL æ ªä¾¡ãƒ�ãƒ£ãƒ¼ãƒˆï¼ˆ6ãƒ¶æœˆï¼‰</h3>
                <div class="chart">
                    <canvas id="stockChart"></canvas>
                </div>
            </div>
            
            <div class="sectors">
                <div class="sector-box sector-it">Information Technology<br>+4.78</div>
                <div class="sector-box sector-health">Health Care<br>+0.15</div>
                <div class="sector-box sector-finance">Financials<br>+0.10</div>
            </div>
            
            <div class="recommendations">
                <h3>æ�¨å®šé–¢é€£ã‚»ãƒ³ãƒ�ãƒ¡ãƒ³ãƒˆåŠ¹æ�œ</h3>
                <div class="rec-grid">
                    <div class="rec-item rec-strong-buy">IT<br>+0.25</div>
                    <div class="rec-item rec-buy">AAPL<br>+0.08</div>
                    <div class="rec-item rec-hold">WFC<br>+0.10</div>
                    <div class="rec-item rec-hold">ABC<br>+0.17</div>
                    <div class="rec-item rec-hold">DEF<br>+0.17</div>
                    <div class="rec-item rec-hold">GHI<br>+0.10</div>
                    <div class="rec-item rec-hold">JKL<br>+0.08</div>
                    <div class="rec-item rec-hold">MNO<br>-0.01</div>
                    <div class="rec-item rec-hold">PQR<br>-0.04</div>
                </div>
            </div>
            
            <div class="pe-table">
                <h3>æ¥­ç¨®åˆ¥ã‚»ãƒ³ãƒ�ãƒ¡ãƒ³ãƒˆçµ�æ�œ</h3>
                <div class="pe-grid">
                    <div class="pe-item"><span>PFE</span><span style="color: #38a169;">+0.32</span></div>
                    <div class="pe-item"><span>AAPL</span><span style="color: #38a169;">+0.24</span></div>
                    <div class="pe-item"><span>WFC</span><span style="color: #38a169;">+0.19</span></div>
                    <div class="pe-item"><span>BAC</span><span style="color: #38a169;">+0.17</span></div>
                </div>
            </div>
            
            <div class="news-section">
                <h3>é–¢é€£ãƒ‹ãƒ¥ãƒ¼ã‚¹</h3>
                <div class="news-item">
                    <div class="news-date">2025-09-22 21:15</div>
                    <div class="news-title">UBS Reiterates Neutral on Apple (AAPL), Sees Mixed iPhone 17 Preorder Demand - Yahoo Finance
                        <span class="news-impact impact-negative">-0.46</span>
                    </div>
                </div>
                <div class="news-item">
                    <div class="news-date">2025-09-22 20:46</div>
                    <div class="news-title">Apple (NASDAQ:AAPL) Stock Rises by 4.3% Following Analyst Upgrade - MarketBeat
                        <span class="news-impact impact-positive">+0.92</span>
                    </div>
                </div>
                <div class="news-item">
                    <div class="news-date">2025-09-22 20:30</div>
                    <div class="news-title">Apple stock times past early 2026 high as investors rally back - Appleinsider
                        <span class="news-impact impact-positive">+0.78</span>
                    </div>
                </div>
                <div class="news-item">
                    <div class="news-date">2025-09-22 20:48</div>
                    <div class="news-title">Apple (AAPL) Stock Is Up, What You Need To Know - Yahoo Finance
                        <span class="news-impact impact-positive">+0.86</span>
                    </div>
                </div>
                <div class="news-item">
                    <div class="news-date">2025-09-22 19:05</div>
                    <div class="news-title">BAAPL stock via All other early reports of stronger than expected demand for the iPhone 17 lineup - Appleinsider
                        <span class="news-impact impact-positive">+0.92</span>
                    </div>
                </div>
                <div class="news-item">
                    <div class="news-date">2025-09-22 03:09</div>
                    <div class="news-title">Apple (NASDAQ:AAPL) Stock Rises by 4.3% Following Analyst Upgrade - MarketBeat
                        <span class="news-impact impact-positive">+0.92</span>
                    </div>
                </div>
                <div class="news-item">
                    <div class="news-date">2025-09-22 18:44</div>
                    <div class="news-title">Apple Stock (AAPL) - Could Rocket on 'Pent-Up Consumer Upgrade Cycle' Says Analyst Dan Ives - TipRanks
                        <span class="news-impact impact-positive">+0.89</span>
                    </div>
                </div>
                <div class="news-item">
                    <div class="news-date">2025-09-22 20:48</div>
                    <div class="news-title">Apple (AAPL) Stock Is Up, What You Need To Know - Yahoo Finance
                        <span class="news-impact impact-positive">+0.86</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 6ãƒ¶æœˆåˆ†ã�®ãƒ€ãƒŸãƒ¼æ ªä¾¡ï¼†å‡ºæ�¥é«˜ãƒ‡ãƒ¼ã‚¿ã‚’ç”Ÿæˆ�ï¼ˆå¹³æ—¥ã�®ã�¿ï¼‰
        function generateStockData() {
            const prices = [];
            const volumes = [];
            const startDate = new Date();
            startDate.setMonth(startDate.getMonth() - 6);
            let currentPrice = 130;

            for (let i = 0; i < 180; i++) {
                const date = new Date(startDate);
                date.setDate(startDate.getDate() + i);
                // åœŸæ—¥ã‚’ã‚¹ã‚­ãƒƒãƒ—
                if (date.getDay() === 0 || date.getDay() === 6) continue;

                const volatility = 0.02;
                const change = (Math.random() - 0.5) * volatility * currentPrice;
                const close = currentPrice + change;

                prices.push({ x: date.toISOString().split('T')[0], y: parseFloat(close.toFixed(2)) });
                volumes.push({ x: date.toISOString().split('T')[0], y: Math.floor(5_000_000 + Math.random() * 20_000_000) });

                currentPrice = close;
            }
            return { prices, volumes };
        }

        const { prices, volumes } = generateStockData();

        // æ ªä¾¡ãƒ�ãƒ£ãƒ¼ãƒˆï¼ˆãƒ©ã‚¤ãƒ³ï¼‰
        const priceCtx = document.getElementById('stockChart').getContext('2d');
        new Chart(priceCtx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'AAPL çµ‚å€¤',
                    data: prices, // [{x: date, y: close}]
                    tension: 0.2,
                    pointRadius: 0,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (items) => new Date(items[0].parsed.x).toLocaleDateString('ja-JP'),
                            label: (ctx) => `çµ‚å€¤: ${ctx.parsed.y.toFixed(2)}`
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'week', displayFormats: { week: 'MM/dd' } },
                        title: { display: true, text: 'æ—¥ä»˜' }
                    },
                    y: {
                        beginAtZero: false,
                        title: { display: true, text: 'æ ªä¾¡ ($)' }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""

# ãƒ€ãƒƒã‚·ãƒ¥ãƒœãƒ¼ãƒ‰ã‚’è¡¨ç¤º
HTML(html_content)




