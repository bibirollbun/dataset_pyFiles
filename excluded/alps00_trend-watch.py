import os
from kaggle_secrets import UserSecretsClient

SECRETS_CLIENT = UserSecretsClient()

try:
    GEMINI_API_KEY = SECRETS_CLIENT.get_secret("GOOGLE_API_KEY")
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    print("âœ… LLM setup complete (GOOGLE_API_KEY loaded).")
except Exception as exc:
    raise RuntimeError(
        "Add 'GOOGLE_API_KEY' to Kaggle Secrets before running the analysis agent."
    ) from exc


!pip -q install feedparser httpx[socks] beautifulsoup4 pandas



import asyncio
from datetime import datetime
from typing import List, Dict, Any

import feedparser
import httpx
import pandas as pd
from bs4 import BeautifulSoup




RSS_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss.xml",
        "region": "global",
        "category": "vendor"
    },
    {
        "name": "Hugging Face News",
        "url": "https://huggingface.co/blog/feed.xml",
        "region": "global",
        "category": "vendor"
    },
    {
        "name": "Google AI",
        "url": "https://blog.google/technology/ai/rss/",
        "region": "global",
        "category": "bigtech"
    },
    {
        "name": "Tietoevry Tech",
        "url": "https://www.tietoevry.com/en/blogs/topic/artificial-intelligence/rss/",
        "region": "finland",
        "category": "enterprise"
    },
    {
        "name": "Gofore Insights",
        "url": "https://gofore.com/en/insights/feed/",
        "region": "finland",
        "category": "consulting"
    },
    {
        "name": "Siili Solutions",
        "url": "https://www.siili.com/blog/rss.xml",
        "region": "finland",
        "category": "consulting"
    }
]

HTML_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "Microsoft Industry Blog",
        "url": "https://cloudblogs.microsoft.com/industry-blog/microsoft-in-business/",
        "region": "global",
        "category": "enterprise",
        "keywords": ["AI", "Copilot", "cloud"]
    },
    {
        "name": "Yle Tech",
        "url": "https://yle.fi/aihe/teknologia",
        "region": "finland",
        "category": "media",
        "keywords": ["tekoÃ¤ly", "digitalisaatio"]
    }
]

SOURCE_REGISTRY = {
    "rss": RSS_SOURCES,
    "html": HTML_SOURCES,
}

SOURCE_REGISTRY



def normalize_record(source: Dict[str, Any], *, title: str, url: str, summary: str = "", published: datetime | None = None) -> Dict[str, Any]:
    return {
        "source": source["name"],
        "region": source["region"],
        "category": source.get("category", "n/a"),
        "title": title.strip() if title else "(untitled)",
        "url": url,
        "summary": summary.strip(),
        "published": published.isoformat() if published else None,
    }


def _parse_feed_entries(feed, source):
    records = []
    for entry in feed.entries[:8]:  # cap to keep context compact
        title = entry.get("title", "")
        link = entry.get("link", "")
        if not link:
            continue
        summary = entry.get("summary", "")
        published = None
        if entry.get("published_parsed"):
            published = datetime(*entry.published_parsed[:6])
        records.append(normalize_record(source, title=title, url=link, summary=summary, published=published))
    return records


async def collect_rss(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20) as client:
        tasks = []
        for source in sources:
            tasks.append(client.get(source["url"], headers={"User-Agent": "TrendWatchBot/0.1"}))
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    records: List[Dict[str, Any]] = []
    for source, response in zip(sources, responses):
        if isinstance(response, Exception):
            print(f"âš ï¸� RSS fetch failed for {source['name']}: {response}")
            continue
        feed = feedparser.parse(response.text)
        records.extend(_parse_feed_entries(feed, source))
    return records


def _extract_links(html: str, base_url: str, keywords: List[str]) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        lower = f"{href.lower()} {text.lower()}"
        if any(keyword.lower() in lower for keyword in keywords):
            if href.startswith("http"):
                links.append((text or "(link)", href))
        if len(links) >= 8:
            break
    return links


async def collect_html(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        tasks = [client.get(src["url"], headers={"User-Agent": "TrendWatchBot/0.1"}) for src in sources]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    records: List[Dict[str, Any]] = []
    for source, response in zip(sources, responses):
        if isinstance(response, Exception):
            print(f"âš ï¸� HTML fetch failed for {source['name']}: {response}")
            continue
        links = _extract_links(response.text, source["url"], source.get("keywords", []))
        for title, url in links:
            records.append(normalize_record(source, title=title, url=url, summary=f"Link scraped from {source['url']}"))
    return records


async def run_collectors() -> pd.DataFrame:
    rss_records, html_records = await asyncio.gather(
        collect_rss(RSS_SOURCES),
        collect_html(HTML_SOURCES),
    )
    df = pd.DataFrame(rss_records + html_records)
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
    return df




import nest_asyncio, asyncio
nest_asyncio.apply()

loop = asyncio.get_event_loop()
trend_df = loop.run_until_complete(run_collectors())
print(f"Collected {len(trend_df)} candidate stories from {trend_df['source'].nunique() if not trend_df.empty else 0} sources.")
trend_df.head()




import json
from collections import deque

import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"  # falls back to the latest flash model available


class SessionMemory:
    def __init__(self, max_items: int = 20):
        self.buffer = deque(maxlen=max_items)

    def add(self, item: str):
        self.buffer.append(item)

    def snapshot(self) -> str:
        if not self.buffer:
            return "(empty)"
        return " | ".join(list(self.buffer))


MEMORY = SessionMemory(max_items=30)




def compact_articles(df: pd.DataFrame, limit: int = 12) -> List[str]:
    if df.empty:
        return []
    df_sorted = df.sort_values(by="published", ascending=False).head(limit)
    snippets = []
    for _, row in df_sorted.iterrows():
        snippet = (
            f"Source: {row['source']} | Region: {row['region']} | "
            f"Title: {row['title']} | Summary: {row['summary'][:400]} | URL: {row['url']}"
        )
        snippets.append(snippet)
    return snippets


def build_prompt(snippets: List[str]) -> str:
    context = "\n".join(snippets) or "No articles collected"
    memory_context = MEMORY.snapshot()
    instructions = f"""
You are Trend Watch, a market intelligence analyst. Using the articles below, produce structured JSON with the following fields:
- "themes": list of strings describing recurring topics (max 5)
- "signals": list of objects {{"source", "highlight", "implication"}} emphasising how organisations implement GenAI/digital solutions
- "sentiment": overall tone (positive/neutral/negative)
- "recommendations": 2 bullet-style suggestions on how to respond in the next quarter
- "memory_tags": 3 short tags we can store to memory

Rules:
- focus on implementation details, customer segments, technologies, and market moves
- mention Finnish players if present, but cover global ones too
- keep outputs concise; 2-3 sentences per signal
- reference memory context when trends repeat: {memory_context}

Articles:
{context}
"""
    return instructions


def analyze_trends(df: pd.DataFrame) -> Dict[str, Any]:
    snippets = compact_articles(df)
    if not snippets:
        return {"themes": [], "signals": [], "sentiment": "neutral", "recommendations": [], "memory_tags": []}

    prompt = build_prompt(snippets)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    text = response.text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # attempt to fix via braces
        fixed = text[text.find('{'): text.rfind('}') + 1]
        data = json.loads(fixed)

    for tag in data.get("memory_tags", []):
        MEMORY.add(tag)
    return data




analysis_payload = analyze_trends(trend_df)
analysis_payload



import plotly.express as px
from textwrap import fill


def coverage_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"sources": {}, "regions": {}}
    by_source = df.groupby("source").size().sort_values(ascending=False)
    by_region = df.groupby("region").size().sort_values(ascending=False)
    return {"sources": by_source.to_dict(), "regions": by_region.to_dict()}


def build_report(df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    metrics = coverage_metrics(df)
    lines = ["# Trend Watch Weekly Snapshot", ""]
    lines.append(f"Sentiment: **{analysis.get('sentiment', 'neutral').capitalize()}**")
    lines.append("## Themes")
    for theme in analysis.get("themes", []):
        lines.append(f"- {theme}")
    lines.append("\n## Signals")
    for signal in analysis.get("signals", []):
        source = signal.get("source", "Unknown")
        highlight = fill(signal.get("highlight", ""), width=110)
        implication = fill(signal.get("implication", ""), width=110)
        lines.append(f"**{source}:** {highlight}\nâ†’ _{implication}_\n")
    lines.append("## Recommendations")
    for rec in analysis.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.append("\n## Coverage metrics")
    lines.append(f"Sources covered: {len(metrics['sources'])} | Regions: {len(metrics['regions'])}")
    for src, count in metrics["sources"].items():
        lines.append(f"- {src}: {count} articles")
    return "\n".join(lines)


report_md = build_report(trend_df, analysis_payload)
print(report_md[:1000])




fig_sources = px.bar(
    x=list(coverage_metrics(trend_df)["sources"].keys()),
    y=list(coverage_metrics(trend_df)["sources"].values()),
    title="Articles per source"
)
fig_sources



fig_regions = px.pie(
    names=list(coverage_metrics(trend_df)["regions"].keys()),
    values=list(coverage_metrics(trend_df)["regions"].values()),
    title="Coverage by region"
)
fig_regions



# Evaluation: ensure each RSS source produced at least one record
rss_names = {src["name"] for src in RSS_SOURCES}
covered_sources = set(trend_df["source"].unique()) if not trend_df.empty else set()
missing = rss_names - covered_sources

if missing:
    print(f"âš ï¸� Missing insights from: {', '.join(missing)}")
else:
    print("âœ… All RSS sources yielded at least one story")




from pathlib import Path

report_path = Path("/kaggle/working/trend_watch_report.md")
report_path.write_text(report_md, encoding="utf-8")
print(f"Report saved to {report_path}")
report_md[:500]



