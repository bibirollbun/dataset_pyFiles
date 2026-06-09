"""News Agent - fetches free news from Reddit and Hacker News and produces a short summary.

Usage:
    from news_agent import run_news_agent
    report = run_news_agent(skill_topic="technology", reddit_subs=("news","worldnews"), limit=8, use_genai=True)
    print(report["notification_text"])  # prints compact summary

The module will try to use Google ADK (Gemini) for summarization when available
(use_genai=True). If ADK is not available it will fall back to a simple extractive summarizer.
"""
from typing import List, Dict, Any, Tuple
import requests
from collections import Counter
from urllib.parse import quote_plus
import time
import os

USER_AGENT = "KaggleNewsAgent/1.0 (+https://kaggle.com)"
HEADERS = {"User-Agent": USER_AGENT}

# Basic stopwords to filter from keywords
STOPWORDS = set(
    "the and for with that from this will have are about which what when where who why how like new news says say report reports".split()
)

# Topic -> recommended subreddits mapping (small starter set)
TOPIC_SUBREDDITS = {
    "technology": ("technology", "technews", "gadgets"),
    "science": ("science", "space", "futurology"),
    "bollywood": ("bollywood", "IndianCinema", "movies"),
    "movies": ("movies", "film", "TrueFilm"),
    "ai": ("artificial", "MachineLearning", "mlnews"),
    "finance": ("finance", "economics", "investing"),
    "crypto": ("CryptoCurrency", "bitcoin", "ethtrader"),
}

# Try to detect Google ADK (Gemini) for optional summarization
ADK_AVAILABLE = False
try:
    from google.adk.agents import Agent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import InMemoryRunner
    ADK_AVAILABLE = True
except Exception:
    ADK_AVAILABLE = False


def fetch_reddit(subreddit: str = "news", limit: int = 8) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/r/{quote_plus(subreddit)}/hot.json?limit={limit}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        j = r.json()
        items = []
        for c in j.get("data", {}).get("children", []):
            d = c.get("data", {})
            items.append({
                "title": d.get("title"),
                "score": d.get("score", 0),
                "url": d.get("url"),
                "source": f"reddit/{subreddit}",
            })
        return items
    except Exception:
        return []


def fetch_hackernews(limit: int = 10) -> List[Dict[str, Any]]:
    # Use Algolia HN API which is free and does not require auth
    url = f"https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage={limit}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        j = r.json()
        items = []
        for h in j.get("hits", []):
            items.append({
                "title": h.get("title") or h.get("story_title") or h.get("comment_text"),
                "score": h.get("points", 0) or 0,
                "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "source": "hackernews",
            })
        return items
    except Exception:
        return []


def fetch_reddit_search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search Reddit for a query (no auth) returning matching post titles.
    Uses the public search.json endpoint and returns normalized items.
    """
    url = f"https://www.reddit.com/search.json?q={quote_plus(query)}&sort=new&limit={limit}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        j = r.json()
        items = []
        for c in j.get("data", {}).get("children", []):
            d = c.get("data", {})
            items.append({
                "title": d.get("title"),
                "score": d.get("score", 0),
                "url": d.get("url"),
                "source": "reddit/search",
            })
        return items
    except Exception:
        return []


def fetch_hackernews_search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search Hacker News via Algolia for a query and return matching stories."""
    url = f"https://hn.algolia.com/api/v1/search?query={quote_plus(query)}&tags=story&hitsPerPage={limit}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        j = r.json()
        items = []
        for h in j.get("hits", []):
            items.append({
                "title": h.get("title") or h.get("story_title") or h.get("comment_text"),
                "score": h.get("points", 0) or 0,
                "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "source": "hackernews/search",
            })
        return items
    except Exception:
        return []


def aggregate_news(sources: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    all_items = []
    for s in sources:
        for it in s:
            if it and it.get("title"):
                all_items.append(it)
    # sort by score descending when available
    all_items.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_items


def simple_summarizer(items: List[Dict[str, Any]], top_n: int = 5) -> Dict[str, Any]:
    """Improved summarizer:
    - Deduplicates identical titles across sources
    - Shuffles headlines slightly to avoid identical ordering each run
    - Uses regex tokenization and a larger stopword set
    """
    import random
    import re

    ADDITIONAL_STOPWORDS = {
        "about", "would", "could", "should", "like", "just", "also",
        "one", "new", "news", "today", "says", "say", "report", "reports",
    }

    # Deduplicate by normalized title
    seen = set()
    uniq_titles = []
    for it in items:
        t = (it.get("title") or "").strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq_titles.append(t)

    # If no unique titles, fall back to provided list
    if not uniq_titles:
        uniq_titles = [(it.get("title") or "") for it in items]

    # Slight randomization so repeated runs can vary a bit
    random.shuffle(uniq_titles)

    # Choose top_n headlines to display
    top_titles = uniq_titles[:max(1, min(top_n, len(uniq_titles)))]

    # Tokenize words: alphabetic words of length >=4
    words = []
    for t in top_titles:
        for w in re.findall(r"\b[a-z]{4,}\b", (t or "").lower()):
            if w in STOPWORDS or w in ADDITIONAL_STOPWORDS:
                continue
            words.append(w)

    kw = [w for w, _ in Counter(words).most_common(10)]

    notification_text = f"News agent — {len(items)} unique items found\nTop headlines:\n"
    for i, tt in enumerate(top_titles, 1):
        notification_text += f"{i}. {tt}\n"
    if kw:
        notification_text += "\nTop keywords: " + ", ".join(kw[:8])
    else:
        notification_text += "\nTop keywords: (none)"

    return {"notification_text": notification_text, "top_titles": top_titles, "keywords": kw}


def genai_summarize(text: str, prompt: str = None, model_name: str = "gemini-2.5-flash-lite") -> str:
    """If ADK (Gemini) available, run a short summarization Agent and return text. Otherwise raise.

    This function handles both synchronous and asynchronous runner implementations. If
    runner.run_debug returns a coroutine (common in notebook kernels), we attempt to
    await it safely using asyncio.run when possible. If the event loop is already
    running (e.g. in Jupyter), we try to apply nest_asyncio to allow run_until_complete;
    if that is not possible we fall back to polling the scheduled task for a short timeout.
    """
    if not ADK_AVAILABLE:
        raise RuntimeError("ADK/Gemini not available")

    # Build a tiny agent that calls the model to summarize the provided text
    instruction = (
        "You are a summarizer. Read the following headlines/text and produce:\n"
        "1) a 3-bullet summary of main themes,\n"
        "2) a 1-line notification-ready headline.\n\n"
        "Input:\n" + text
    )
    try:
        model = Gemini(model=model_name)
        agent = Agent(name="GenSummarizer", model=model, instruction=instruction)
        runner = InMemoryRunner(agent=agent)

        # Call run_debug which may be sync or return a coroutine depending on ADK version
        result = runner.run_debug(instruction)

        # If result is a coroutine, try to execute it appropriately
        try:
            import asyncio
            if asyncio.iscoroutine(result):
                # If no running loop, use asyncio.run
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    resp = asyncio.run(result)
                else:
                    # We're in an active event loop (notebook). Try nest_asyncio to allow run_until_complete
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        resp = loop.run_until_complete(result)
                    except Exception:
                        # As a last resort, schedule the coroutine and poll it briefly
                        fut = asyncio.ensure_future(result)
                        import time as _time
                        t0 = _time.time()
                        timeout = 10.0
                        while not fut.done() and _time.time() - t0 < timeout:
                            _time.sleep(0.1)
                        if fut.done():
                            resp = fut.result()
                        else:
                            raise RuntimeError("Async GenAI runner did not complete; install nest_asyncio in notebooks to enable GenAI summarization.")
            else:
                resp = result
        except Exception as e:
            return f"[genai_error] {e}"

        # extract textual content from response events (compatible with previous logic)
        out = ""
        if isinstance(resp, (list, tuple)):
            for ev in resp[::-1]:
                try:
                    parts = getattr(ev, 'content', None)
                    if parts and getattr(parts, 'parts', None):
                        part0 = parts.parts[0]
                        out = getattr(part0, 'text', '') or str(part0)
                        if out:
                            break
                except Exception:
                    continue
        else:
            try:
                parts = getattr(resp, 'content', None)
                if parts and getattr(parts, 'parts', None):
                    part0 = parts.parts[0]
                    out = getattr(part0, 'text', '') or str(part0)
            except Exception:
                out = str(resp)

        # Try to clean up/close any async clients to avoid background aclose errors
        def _attempt_close(obj):
            """Try to close or aclose an object safely. Handles coroutines and running event loops."""
            try:
                import asyncio
                if obj is None:
                    return
                # prefer aclose if present
                aclose = getattr(obj, 'aclose', None)
                close = getattr(obj, 'close', None)
                if callable(aclose):
                    try:
                        coro = aclose()
                        if asyncio.iscoroutine(coro):
                            loop = asyncio.get_event_loop()
                            if not loop.is_running():
                                asyncio.run(coro)
                            else:
                                try:
                                    import nest_asyncio
                                    nest_asyncio.apply()
                                    loop.run_until_complete(coro)
                                except Exception:
                                    # schedule and attempt to wait briefly
                                    fut = asyncio.ensure_future(coro)
                                    import time as _time
                                    t0 = _time.time()
                                    timeout = 5.0
                                    while not fut.done() and _time.time() - t0 < timeout:
                                        _time.sleep(0.05)
                                    # ignore if not done
                    except Exception:
                        pass
                elif callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            except Exception:
                pass

        # best-effort cleanup
        try:
            _attempt_close(runner)
            _attempt_close(agent)
            _attempt_close(model)
            # some SDKs expose an api client object
            _attempt_close(getattr(model, 'api_client', None))
            _attempt_close(getattr(model, '_api_client', None))
        except Exception:
            pass

        return out or ""
    except Exception as e:
        return f"[genai_error] {e}"


def run_news_agent(skill_topic: str = "technology", reddit_subs: Tuple[str, ...] = None, limit: int = 8, use_genai: bool = True, filter_by_topic: bool = True, use_search_when_filtering: bool = True, debug: bool = False) -> Dict[str, Any]:
    """Run the news agent for a given topic.

    - If `reddit_subs` is None, a small mapping is used to choose relevant subreddits.
    - When `filter_by_topic` is True and `use_search_when_filtering` is True, perform site-wide searches for the topic to get more relevant items.
    - Set debug=True to print fetched titles for inspection.
    """
    # choose subreddits based on topic when not provided
    if reddit_subs is None:
        reddit_subs = TOPIC_SUBREDDITS.get(skill_topic.lower(), (skill_topic, "news"))

    # If filtering by topic, prefer search endpoints to get topic-specific items
    reddit_items = []
    hn_items = []
    if filter_by_topic and skill_topic and use_search_when_filtering:
        # use search to find topic-specific posts
        reddit_items = fetch_reddit_search(skill_topic, limit=limit * 3)
        hn_items = fetch_hackernews_search(skill_topic, limit=limit * 3)
    else:
        # fetch reddit subs
        for sub in reddit_subs:
            reddit_items.extend(fetch_reddit(sub, limit))
            time.sleep(0.2)
        hn_items = fetch_hackernews(limit)

    # aggregate
    all_items = aggregate_news([reddit_items, hn_items])

    if debug:
        print("DEBUG: raw fetched items (first 30):")
        for it in all_items[:30]:
            print(it.get("source"), "|", it.get("title"))

    # If requested, filter headlines to those that reference the topic keywords
    if filter_by_topic and skill_topic and not use_search_when_filtering:
        tokens = [t for t in skill_topic.lower().split() if len(t) > 2]
        if tokens:
            filtered = []
            for it in all_items:
                title = (it.get("title") or "").lower()
                url = (it.get("url") or "").lower()
                if any(tok in title or tok in url for tok in tokens):
                    filtered.append(it)
            # if filtering removed everything, keep original list
            if filtered:
                all_items = filtered

    # Deduplicate and shuffle headlines to avoid repeated identical outputs
    seen_titles = set()
    uniq_items = []
    for it in all_items:
        title = (it.get("title") or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        uniq_items.append(it)

    # Slight randomization
    try:
        import random
        random.shuffle(uniq_items)
    except Exception:
        pass

    # Use the deduped/shuffled list for summarization
    all_items = uniq_items

    # Prepare text for summarization
    text_blob = "\n".join([it.get("title", "") for it in all_items[:40]])

    summary = None
    if use_genai and ADK_AVAILABLE:
        try:
            summary_text = genai_summarize(text_blob)
            summary = {"method": "genai", "text": summary_text}
        except Exception:
            summary = {"method": "fallback", "text": simple_summarizer(all_items)["notification_text"]}
    else:
        summary = {"method": "fallback", "text": simple_summarizer(all_items)["notification_text"]}

    report = {
        "skill_topic": skill_topic,
        "items_found": len(all_items),
        "items": all_items,
        "summary": summary,
        "notification_text": summary["text"],
    }
    return report


if __name__ == "__main__":
    # quick demo: change the topic here and it will affect subreddit selection and filtering
    topic = "Technology"  # <-- change this to your desired topic
    rpt = run_news_agent(skill_topic=topic, reddit_subs=None, limit=12, use_genai=False)
    print(rpt["notification_text"])





