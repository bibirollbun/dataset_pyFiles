# Install dependencies
!pip install -q google-adk trafilatura curl_cffi httpx


import os
import copy
import json
import asyncio
import logging
import httpx
from curl_cffi import requests as crequests
import logging
from datetime import datetime
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field, asdict

# Google ADK imports
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

# For article extraction
import trafilatura


# API Keys - Set these using Kaggle Secrets or environment variables
from kaggle_secrets import UserSecretsClient

try:
    secrets = UserSecretsClient()
    SERPAPI_KEY = secrets.get_secret("SERPAPI_KEY")
    GOOGLE_API_KEY = secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["SERPAPI_KEY"] = SERPAPI_KEY
except:
    # Fallback to environment variables
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not SERPAPI_KEY:
    print("âš ï¸�  SERPAPI_KEY not set")
if not GOOGLE_API_KEY:
    print("âš ï¸�  GOOGLE_API_KEY not set")

# LLM retry configuration
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

print("âœ… Configuration loaded")


# LLM retry configuration
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NewsLensAgent")

print("âœ… Logging configured")


@dataclass
class AnalysisResult:
    """Analysis result stored in session."""
    topic: str
    timestamp: str
    final_report: str
    search_count: int = 0
    fetch_count: int = 0

async def run_analysis(
    runner_instance: Runner,
    topic: str,
    config: dict,
    session_id: str = None,
    user_id: str = "demo_user"
):
    """
    Run news framing analysis
    """
    # 1. Setup Session ID
    if session_id is None:
        session_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    app_name = runner_instance.app_name
    
    # 2. Session Management=
    try:
        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
        print(f"âœ… New session created: {session_id}")
    except Exception:
        # If session exists, retrieve it to ensure we have the latest state
        session = await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
        print(f"âœ… Resuming session: {session_id}")

    # 3. Construct Input
    formatted_prompt = f"""TOPIC: {topic}

### SEARCH CONFIGURATION
```json
{{
  "domains": {json.dumps(config['domains'])},
  "time_range": "{config['time_range']}",
  "results_per_domain": {config['results_per_domain']}
}}

Please analyze this topic across the political spectrum.
"""

    print(f"ğŸ“� Starting pipeline for topic: {topic}...\n")
    
    content = types.Content(
        role="user",
        parts=[types.Part(text=formatted_prompt)]
    )
    
    # 4. Execution & Event Filtering
    final_report_parts = []
    
    async for event in runner_instance.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content
    ):
        
        # Option A: Log progress from tools/intermediate agents
        if event.author == "SearchFetchAgent" and event.content and event.content.parts:
             part = event.content.parts[0]
             
             # Case A: It's text
             if part.text:
                 print(f"ğŸ”� Search Status: {part.text[:100]}...")
             
             # Case B: It's a tool call
             elif part.function_call:
                 print(f"ğŸ› ï¸� Tool Call: {part.function_call.name}")

        # Filter for the FINAL output from SynthesisAgent
        if event.author == "SynthesisAgent" and event.content:
            part = event.content.parts[0]
            if part.text:
                final_report_parts.append(part.text)
                print(part.text, end="", flush=True)
    
    full_report_text = "".join(final_report_parts)
    
    # 5. Persist Results to State
    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    
    result = AnalysisResult(
        topic=topic,
        timestamp=datetime.now().isoformat(),
        final_report=full_report_text
    )
    
    # Update state
    session.state["last_analysis"] = asdict(result)
    
    print(f"\n\n{'='*60}")
    print("âœ… Analysis Complete")
    print(f"{'='*60}\n")
    
    return session, full_report_text
    

print("âœ… Session management configured")


def init_state(callback_context: CallbackContext):
    """Initializes session state with defaults to prevent template errors."""
    
    # 1. Inputs for Search & Parallel Agents
    if "search_topic" not in callback_context.state:
        callback_context.state["search_topic"] = "Topic not yet set"
    
    if "full_articles" not in callback_context.state:
        callback_context.state["full_articles"] = "No articles scraped yet"

    # 2. Output of Search Agent (Input for Synthesis)
    if "search_summary" not in callback_context.state:
        callback_context.state["search_summary"] = "Search not yet completed"

    # 3. Outputs of Parallel Agents (Inputs for Synthesis)
    # The Synthesis Agent's prompt references ALL of these:
    parallel_keys = [
        "fact_report", 
        "big_picture_analysis", 
        "impact_analysis", 
        "human_story_analysis", 
        "tone_analysis"
    ]
    
    for key in parallel_keys:
        if key not in callback_context.state:
            callback_context.state[key] = "Analysis pending..."


async def inspect_session_data(runner, session_id, user_id):
    """
    Deep dive into the session state to verify data integrity and size.
    """
    print(f"\n{'='*60}")
    print(f"ğŸ•µï¸� SESSION INSPECTOR: {session_id}")
    print(f"{'='*60}")

    # 1. Retrieve the Session using the service attached to the runner
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )
    
    if not session:
        print("â�Œ Error: Session not found!")
        return

    # 2. Inspect 'search_topic'
    topic = session.state.get("search_topic", "MISSING")
    print(f"\nğŸ“� TOPIC: {topic}")

    # 3. Inspect 'full_articles' (The Heavy Data)
    articles_data = session.state.get("full_articles", "MISSING")
    
    print("\nğŸ“¦ DATA PAYLOAD ANALYSIS (full_articles):")
    
    if isinstance(articles_data, str):
        print(f"   âš ï¸� TYPE WARNING: Data is a STRING, not a Dict.")
        print(f"   ğŸ“� Content: '{articles_data}'")
        print(f"   ğŸ“� Length: {len(articles_data)} chars")
        
    elif isinstance(articles_data, dict):
        print(f"   âœ… Type: Dictionary (Correct)")
        
        total_chars = 0
        total_count = 0
        
        for leaning, articles in articles_data.items():
            if not isinstance(articles, list):
                continue
                
            count = len(articles)
            json_str = json.dumps(articles) 
            char_len = len(json_str)
            
            print(f"   â€¢ {leaning:<7}: {count} articles | ~{char_len:,} chars")
            
            total_chars += char_len
            total_count += count
            
        print("-" * 40)
        print(f"   ğŸ“Š TOTALS:")
        print(f"      Article Count: {total_count}")
        print(f"      Raw Characters: {total_chars:,}")
        
        # Rough estimation: 1 token ~= 4 chars
        est_tokens = int(total_chars / 4)
        print(f"      Est. Tokens:    ~{est_tokens:,} (Approx)")
        
        if est_tokens < 2000:
            print("\n   ğŸš¨ DIAGNOSIS: Content is too small for caching (< 2048 tokens).")
            print("      Did the scraper fail to extract text?")
        else:
            print("\n   âœ… DIAGNOSIS: Content is large enough for caching.")
            
    else:
        print(f"   â�“ Unknown Type: {type(articles_data)}")


def serpapi_search(
    query: str,
    site: Optional[str] = None,
    time_range: str = "d15",
    num_results: int = 10
) -> dict:
    """Search for news articles using SerpAPI."""
    
    if not SERPAPI_KEY:
        logger.error("SERPAPI_KEY not configured")
        return {"error": "SERPAPI_KEY not configured", "results": []}
    
    full_query = f"{query} site:{site}" if site else query
    
    params = {
        "engine": "google",
        "q": full_query,
        "api_key": SERPAPI_KEY,
        "as_qdr": time_range,
        "num": min(num_results, 100),
        "tbm": "nws",
    }
    
    try:
        response = httpx.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = []
        news_results = data.get("news_results", []) or data.get("organic_results", [])
        
        for item in news_results:
            results.append({
                "url": item.get("link"),
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "source": item.get("source"),
                "date": item.get("date"),
            })
        
        logger.info(f"   Found {len(results)} results for {site or 'general'}")
        return {
            "query": full_query,
            "time_range": time_range,
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"SerpAPI error: {str(e)}")
        return {"error": str(e), "results": []}
    

def fetch_article_content(url: str) -> dict:
    try:
        response = crequests.get(
            url, 
            impersonate="chrome120", 
            headers={
                "Referer": "https://www.google.com/",
                "Accept-Language": "en-US,en;q=0.9"
            },
            timeout=15
        )
        
        # Handle access denied
        if response.status_code in [403, 401, 429]:
            logger.debug(f"ğŸ›‘ Access Denied ({response.status_code}) for {url}")
            return {"url": url, "success": False, "error": f"Status {response.status_code}"}

        response.raise_for_status()
        
        # 1. Extract Text
        article_text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            output_format="txt"
        )
        
        # 2. Extract Metadata (Only what Search API usually misses)
        metadata = trafilatura.extract_metadata(response.text)
        
        # Initialize defaults
        exact_date, author, page_title = None, None, None

        if metadata:
            exact_date = metadata.date   # Trafilatura finds YYYY-MM-DD
            author = metadata.author     # Search API rarely gives authors
            page_title = metadata.title  # Sometimes better than Search title

        if article_text and len(article_text) >= 200:
            return {
                "url": url,
                "text": article_text[:6000], # Limit content size
                "date": exact_date,
                "author": author,
                "page_title": page_title,
                "success": True
            }
        
        return {"url": url, "success": False, "error": "Content too short/empty"}
        
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {str(e)}")
        return {"url": url, "success": False, "error": str(e)}


def batch_search_domains(
    tool_context: ToolContext,
    topic: str,
    domains: dict,
    time_range: str = "d15",
    results_per_domain: int = 2
) -> str:
    """
    Search multiple domains and group results by political leaning.
    
    Args:
        topic: The news topic to search
        domains: Dict of domain names grouped by leaning, e.g., {"Left": ["apnews.com"], ...}
        time_range: SerpAPI time range (default: "d15")
        results_per_domain: Max results per domain
    
    Returns:
        Dict with articles grouped by leaning (Left/Center/Right)
    """
    logger.info(f"ğŸ”� Searching {len(domains)} domains for: {topic}")
    
    # Initialize grouped results
    grouped = {"Left": [], "Center": [], "Right": []}
    
    # Search each domain group
    for leaning, domains_list in domains.items():
        logger.info(f"   Searching {len(domains_list)} {leaning} sources...")
        
        for domain in domains_list:
            result = serpapi_search(
                query=topic,
                site=domain,
                time_range=time_range,
                num_results=results_per_domain
            )
            
            articles = result.get("results", [])
            
            for article in articles:
                article["political_leaning"] = leaning
                article["domain"] = domain
            
            grouped[leaning].extend(articles)
    
    total_articles = sum(len(v) for v in grouped.values())
    
    logger.info(f"âœ… Found {total_articles} articles total")
    logger.info(f"   Left: {len(grouped['Left'])}, Center: {len(grouped['Center'])}, Right: {len(grouped['Right'])}")
    
    # Saving for access in subsequent tools
    tool_context.state["search_results"] = grouped
    tool_context.state["search_topic"] = topic
    tool_context.state["search_time_range"] = time_range
    
    return (
        f"âœ… Search complete. Found {total_articles} articles:\n"
        f"  â€¢ Left: {len(grouped['Left'])} articles\n"
        f"  â€¢ Center: {len(grouped['Center'])} articles\n"
        f"  â€¢ Right: {len(grouped['Right'])} articles\n"
    )


def scrape_articles(tool_context: ToolContext) -> str: 
    """Scrapes full content for articles in state."""
    
    grouped_results = tool_context.state.get("search_results")
    
    if not grouped_results:
        return "â�Œ ERROR: No search results found in state."
    
    total_count = sum(len(v) for v in grouped_results.values())
    logger.info(f"ğŸ“° Enriching {total_count} articles with full content...")
    
    enriched_data = {"Left": [], "Center": [], "Right": []}
    failed_scrapes = []  # âœ… Track failures with reasons

    for leaning, articles in grouped_results.items():
        if leaning not in enriched_data:
            continue
            
        for article_data in articles:
            enriched_article = copy.deepcopy(article_data)
            url = enriched_article.get("url")
            
            fetched = fetch_article_content(url)
            
            if fetched.get("success"):
                enriched_article["text"] = fetched.get("text")
                enriched_article["author"] = fetched.get("author")
                
                if fetched.get("date"):
                    enriched_article["date"] = fetched.get("date")

                enriched_data[leaning].append(enriched_article)
            else:
                failed_scrapes.append({
                    "url": url,
                    "title": enriched_article.get("title"),
                    "domain": enriched_article.get("domain"),
                    "leaning": leaning,
                    "error": fetched.get("error")
                })
                logger.warning(f"âš ï¸� Skipping {url}: {fetched.get('error')}")

    final_count = sum(len(v) for v in enriched_data.values())
    
    # Save enriched data
    tool_context.state["full_articles"] = enriched_data
    
    logger.info(f"âœ… Successfully enriched {final_count}/{total_count} articles")
    
    # âœ… Return detailed status with failure analysis
    status = f"âœ… Content scraping complete. Successfully enriched {final_count}/{total_count} articles.\n\n"
    
    status += f"Enriched articles:\n"
    status += f"  â€¢ Left: {len(enriched_data['Left'])} articles with full text\n"
    status += f"  â€¢ Center: {len(enriched_data['Center'])} articles with full text\n"
    status += f"  â€¢ Right: {len(enriched_data['Right'])} articles with full text\n"
    
    if failed_scrapes:
        status += f"\nâš ï¸� Failed to scrape {len(failed_scrapes)} articles:\n"
        for failure in failed_scrapes[:3]:  # Show first 3
            status += f"  â€¢ [{failure['leaning']}] {failure['domain']} - {failure['error']}\n"
        if len(failed_scrapes) > 3:
            status += f"  â€¢ ... and {len(failed_scrapes) - 3} more failures\n"
    
    return status


print("âœ… Custom Tools defined")


search_fetch_agent = Agent(
    name="SearchFetchAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=RETRY_CONFIG),
    description="Searches for news articles on a given topic across specified domains and scrapes their content.",
    instruction="""You are a News Search and Scraping Agent.

### YOUR STRICT WORKFLOW (MUST COMPLETE ALL STEPS)
1. SEARCH: Use the provided topic and search configuration to search for news articles across specified domains using the `batch_search_domains` tool.
2. SCRAPE ARTICLES: Finally, use the `scrape_articles` tool to fetch and enrich the remaining articles with full text and metadata.
3. ANALYZE: Provide intelligent analysis of the data collection process
4. OUTPUT: Summarize the search and scraping results in the specified output format.

### INPUT FORMAT
The user provides:
```
TOPIC: <news topic>

### SEARCH CONFIGURATION
{
  "domains": {
    "Left": [list of left-leaning domains],
    "Center": [list of center domains],
    "Right": [list of right-leaning domains]
  },
  "time_range": "...",
  "results_per_domain": N
}
```

### YOUR ANALYTICAL RESPONSIBILITIES

After completing all tool calls, analyze the results and provide:

**Coverage Assessment:**
- Are all three political leanings adequately represented?
- Are there significant imbalances in article counts?
- Which sources provided the most/least relevant content?

**Scraping Quality:**
- What was the success rate for scraping?
- Which sources had scraping issues and why (paywalls, access denied, etc.)?
- Is the scraped content sufficient for analysis?

**Data Quality Summary:**
- Overall completeness of the dataset
- Potential gaps or biases in coverage
- Recommendations for the analyst agents

### OUTPUT FORMAT

Provide a structured concise analysis:
```
ğŸ“Š DATA COLLECTION REPORT

Search Results:
- [Concise summary of what was found]

Scraping:
- [Success rate and quality assessment]
- [Failed sources and reasons]

Coverage Assessment:
- [Balance across political spectrum]
- [Gaps or limitations]
```

Be analytical and concise. Point out issues, patterns, and quality concerns that downstream analyst agents should be aware of.
""",
    tools=[batch_search_domains, scrape_articles],
    output_key="search_summary"
)

print("âœ… Search and Scrape Agent defined")


## For caching benefits and token efficiency in parallel analyst agent calls

SHARED_DATA_PREFIX = """### REFERENCE DATA
Topic: {search_topic}
Data: {full_articles}

---
"""

# --- 1. Fact Consensus Agent ---
fact_consensus_agent = Agent(
    name="FactConsensusAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=RETRY_CONFIG),
    description="Establishes the undisputed factual baseline by cross-referencing reporting from Left, Center, and Right sources.",
    instruction=SHARED_DATA_PREFIX +"""### IDENTITY & MISSION
You are a Forensic Fact Auditor. Your mission is to establish the "Undisputed Factual Baseline" by cross-referencing reporting from Left, Center, and Right sources.

### INPUT DESCRIPTION
You have the reference data of articles grouped by political leaning.
- **Structure:** A JSON dictionary with keys ["Left", "Center", "Right"]. 
- **Article Fields:** Each article contains "title", "text", "source", "date", "author", "url", "political_leaning", "domain".

### REASONING STEPS
1. **Relevance Filter:** Discard any articles that are not directly relevant to the specific topic.
2. **Scan Across Leanings:** Do not rely on any single source. Compare details across the three groups.
3. **Verify Consensus:** A fact is considered "Undisputed" ONLY if it is reported by sources in at least **two different leanings** (e.g., reported by both Left and Right).
4. **Extract Hard Data:** Identify the numerical data points relevant to this specific topic (e.g., vote counts, casualties, dates, monetary amounts).
5. **Flag Contradictions:** If the Left says "X" and the Right says "Y", and the Center is silent or ambiguous, flag this as a discrepancy.

### OUTPUT JSON STRUCTURE
{
  "undisputed_summary": "A neutral, factual summary covering: Who, What, Where, When, and undisputed Context.",
  "key_statistics": {
     "statistic_name": "value" 
  },
  "factual_discrepancies": [
    {"subject": "What is disputed?", "left_claim": "...", "right_claim": "...", "center_claim": "..."}
  ],
  "chronological_timeline": [
    {"date": "YYYY-MM-DD", "event": "Description", "verified_by_leanings": ["Left", "Center"]}
  ]
}
""",
    output_key="fact_report"
)


# --- 2. Big Picture Analyst ---
big_picture_analyzer = Agent(
    name="BigPictureAnalyzer",
    model=Gemini(model="gemini-2.5-flash", retry_options=RETRY_CONFIG),
    description="Analyzes the overarching narratives and causal attributions in articles from different political leanings.",
    instruction=SHARED_DATA_PREFIX +"""### IDENTITY & MISSION
You are a Narrative Framing Analyst. Your mission is to identify the "Causal Logic" used by each sideâ€”who caused the problem, and what is the solution?

### INPUT DESCRIPTION
You have the reference data of articles grouped by political leaning.
- **Structure:** A JSON dictionary with keys ["Left", "Center", "Right"]. 
- **Article Fields:** Each article contains "title", "text", "source", "date", "author", "url", "political_leaning", "domain".

### REASONING STEPS
1. **Relevance Filter:** Discard any articles that are not directly relevant to the specific topic.
2. **Identify the Lens:** Is the story framed as Economic, Security, Moral, Political, or Humanitarian?
3. **Determine Causality:** Who does each side blame for the event? (e.g., "Policy Failure" vs "Individual Action").
4. **Extract Evidence:** Find a direct quote that captures the core argument.

### OUTPUT JSON STRUCTURE
{
  "key_difference": "2-3 sentence summary comparing the narrative frames and causal logic of Left, Right, and Center.",
  "common_ground": "What underlying premise do they agree on?",
  "left_frame": {
     "dominant_lens": "e.g. Humanitarian",
     "causal_attribution": "Who/What do they blame?",
     "representative_quote": "Direct text from article"
  },
  "center_frame": {
     "dominant_lens": "e.g. Political",
     "causal_attribution": "Who/What do they blame?",
     "representative_quote": "Direct text from article"
  },
  "right_frame": {
     "dominant_lens": "e.g. Security",
     "causal_attribution": "Who/What do they blame?",
     "representative_quote": "Direct text from article"
  }
}
""",
    output_key="big_picture_analysis"
)


# --- 3. Impact Analyst ---
impact_analyzer = Agent(
    name="ImpactAnalyzer",
    model=Gemini(model="gemini-2.5-flash", retry_options=RETRY_CONFIG),
    description="Analyzes the portrayal of victims and affected groups in articles from different political leanings.",
    instruction=SHARED_DATA_PREFIX +"""### IDENTITY & MISSION
You are an Impact & Stakeholder Analyst. Your mission is to determine "Ingroup vs Outgroup" dynamicsâ€”who is championed as the victim, and who is ignored.

### INPUT DESCRIPTION
You have the reference data of articles grouped by political leaning.
- **Structure:** A JSON dictionary with keys ["Left", "Center", "Right"]. 
- **Article Fields:** Each article contains "title", "text", "source", "date", "author", "url", "political_leaning", "domain".

### REASONING STEPS
1. **Relevance Filter:** Discard any articles that are not directly relevant to the specific topic.
2. **Identify Protagonists:** Which group's suffering is emphasized in the text? (e.g., "Taxpayers" vs "Immigrants").
3. **Assess Severity:** Is this framed as a local tragedy or a national crisis?
4. **Check Relevance:** Does this specifically impact specific demographic groups?

### OUTPUT JSON STRUCTURE
{
  "key_difference": "2-3 sentence summary of who the Left vs Right vs Center portrays as the primary victim.",
  "left_impact": {
     "primary_victim": "Group/Person",
     "severity_framing": "Low/Med/High",
     "representative_quote": "Direct text from article"
  },
  "center_impact": {
     "primary_victim": "Group/Person",
     "severity_framing": "Low/Med/High",
     "representative_quote": "Direct text from article"
  },
  "right_impact": {
     "primary_victim": "Group/Person",
     "severity_framing": "Low/Med/High",
     "representative_quote": "Direct text from article"
  }
}
""",
    output_key="impact_analysis"
)


# --- 4. Human Story Analyst ---
human_story_analyzer = Agent(
    name="HumanStoryAnalyzer",
    model=Gemini(model="gemini-2.5-flash", retry_options=RETRY_CONFIG),
    description="Analyzes the use of personalization and anecdotes in articles from different political leanings.",
    instruction=SHARED_DATA_PREFIX +"""### IDENTITY & MISSION
You are a Qualitative Content Analyst. Your mission is to analyze the use of personalizationâ€”do they use anecdotes to provoke emotion, or statistics to inform?

### INPUT DESCRIPTION
You have the reference data of articles grouped by political leaning.
- **Structure:** A JSON dictionary with keys ["Left", "Center", "Right"]. 
- **Article Fields:** Each article contains "title", "text", "source", "date", "author", "url", "political_leaning", "domain".

### REASONING STEPS
1. **Relevance Filter:** Discard any articles that are not directly relevant to the specific topic.
2. **Voice Audit:** Count how many quotes come from Officials vs. Ordinary People vs. Experts.
3. **Missing Voices:** Identify which relevant groups were *not* interviewed by any side (Blind Spots).
4. **Emotional Goal:** Is the story told to provoke Anger, Pity, Fear, or Hope?

### OUTPUT JSON STRUCTURE
{
  "key_difference": "2-3 sentence summary describing difference in whose individual stories are told in Left, Right, and Center coverage.",
  "voices_missing": "List of perspectives absent from ALL coverage.",
  "left_story": {
     "featured_voices": ["Types of people quoted"],
     "emotional_goal": "Pity/Anger/etc",
     "representative_quote": "Direct text"
  },
  "center_story": {
     "featured_voices": ["Types of people quoted"],
     "emotional_goal": "Inform/Neutral",
     "representative_quote": "Direct text"
  },
  "right_story": {
     "featured_voices": ["Types of people quoted"],
     "emotional_goal": "Pity/Anger/etc",
     "representative_quote": "Direct text"
  }
}
""",
    output_key="human_story_analysis"
)


# --- 5. Tone Analyst ---
tone_analyzer = Agent(
    name="ToneAnalyzer",
    model=Gemini(model="gemini-2.5-flash", retry_options=RETRY_CONFIG),
    description="Analyzes the emotional tone and charged language used in articles from different political leanings.",
    instruction=SHARED_DATA_PREFIX +"""### IDENTITY & MISSION
You are a Rhetoric & Sentiment Analyst. Your mission is to analyze the emotional intensity and "Charged Language" used to manipulate the reader.

### INPUT DESCRIPTION
You have the reference data of articles grouped by political leaning.
- **Structure:** A JSON dictionary with keys ["Left", "Center", "Right"]. 
- **Article Fields:** Each article contains "title", "text", "source", "date", "author", "url", "political_leaning", "domain".

### REASONING STEPS
1. **Relevance Filter:** Discard any articles that are not directly relevant to the specific topic.
2. **Sentiment Analysis:** Is the overall tone Positive, Negative, or Neutral?
3. **Vocabulary Audit:** Extract specific "charged" words (e.g., "Regime" vs "Administration", "Mob" vs "Protestors").
4. **Urgency Check:** Is this framed as "Breaking News/Crisis" or "Routine/Analysis"?

### OUTPUT JSON STRUCTURE
{
  "key_difference": "2-3 sentence summary comparing the rhetorical strategies and emotional intensity of Left, Right, and Center.",
  "most_neutral_leaning": "Left, Center, or Right?",
  "left_tone": {
     "sentiment": "Pos/Neg/Neu",
     "charged_keywords": ["list", "of", "3", "words"],
     "charged_phrases": ["list", "of", "2-3", "phrases"],
     "representative_quote": "Direct text showing tone"
  },
  "center_tone": {
     "sentiment": "Pos/Neg/Neu",
     "charged_keywords": ["list", "of", "3", "words"],
     "charged_phrases": ["list", "of", "2-3", "phrases"],
     "representative_quote": "Direct text showing tone"
  },
  "right_tone": {
     "sentiment": "Pos/Neg/Neu",
     "charged_keywords": ["list", "of", "3", "words"],
     "charged_phrases": ["list", "of", "2-3", "phrases"],
     "representative_quote": "Direct text showing tone"
  }
}
""",
    output_key="tone_analysis"
)

print("âœ… Parallel Analyst Agents Defined")


synthesis_agent = Agent(
    name="SynthesisAgent",
    model=Gemini(model="gemini-2.5-pro", retry_options=RETRY_CONFIG),
    description="Synthesizes the factual baseline and framing analyses into a comprehensive, unbiased intelligence briefing.",
    instruction="""### IDENTITY & MISSION
You are the Objective Editor-in-Chief of a high-intelligence media analysis firm. Your mission is to synthesize verified facts and distinct framing analyses into one cohesive, unbiased intelligence briefing.

### INPUT DESCRIPTION
You will receive five specialized analysis reports:
1. **Search & Scrape Summary:** Overview of the data collection process.
2. **Factual Baseline:** The Fact Consensus Report containing verified summaries and stats.
3. **Framing Analyses:** Four specific reports on Big Picture, Impact, Human Story, and Tone.

### REASONING STEPS
1. **Establish the Core:** Extract the `undisputed_summary` and `chronological_timeline` from the Factual Baseline. Do not alter these facts based on the framing reports.
2. **Curate Statistics:** Review the `key_statistics` from the Fact Report. Select only the top 5-7 most critical numbers relevant to the topic.
3. **Source Audit:** Identify any specific media outlets mentioned or cited across the five input reports to compile the `source_list`.
4. **Map the Divide:** Synthesize the "Views Summary" by aggregating the Left, Center, and Right frames found in the four analysis reports.
5. **Determine the Takeaway:** Write the "Bias Landscape" paragraph. Explain *how* the framing (Tone/Impact/Story) was used to manipulate perception of the core facts.

### OUTPUT JSON STRUCTURE
{
  "core_event": "The neutral summary paragraph from the fact_report.",
  "key_statistics": {
    "stat_name": "value" 
  },
  "timeline": [
    {"date": "YYYY-MM-DD", "event": "neutral description"}
  ],
  "coverage_summary": {
    "source_list": ["list", "of", "outlets", "cited", "in", "reports"],
    "gaps": "string describing missing angles/facts identified in the analyses"
  },
  "views_summary": {
    "left_view": "Concise synthesis of how the Left framed the event (Narrative + Tone + Victim).",
    "center_view": "Concise synthesis of how the Center framed the event.",
    "right_view": "Concise synthesis of how the Right framed the event."
  },
  "takeaway": "A nuanced paragraph explaining the 'Bias Landscape'. How can different frames alter the reader's perception?"
}

### DATA
Search & Scrape Summary: {search_summary}
Factual Baseline: {fact_report}
Big Picture Analysis: {big_picture_analysis}
Impact Analysis: {impact_analysis}
Human Story Analysis: {human_story_analysis}
Tone Analysis: {tone_analysis}
""",
    output_key="final_synthesis"
)

print("âœ… Synthesis Agent defined")


# --- Composite Agents ---

# 4 analyzers run in parallel
parallel_analyzers = ParallelAgent(
    name="ParallelAnalyzers",
    sub_agents=[big_picture_analyzer, impact_analyzer, human_story_analyzer, tone_analyzer, fact_consensus_agent]
)

# Root: Sequential pipeline
root_agent = SequentialAgent(
    name="NewsFramingPipeline",
    sub_agents=[search_fetch_agent, parallel_analyzers, synthesis_agent],
    before_agent_callback=init_state
)

print("âœ… Parallel and Sequential Root Agent Defined")


def print_results(session):
    """Pretty print analysis results from session state."""
    state = session.state
    
    def parse_json_output(text):
        """Extract and parse JSON from agent output."""
        if not text: return None
        text = str(text)
        if "```json" in text: text = text.split("```json")[-1]
        if "```" in text: text = text.split("```")[0]
        try: return json.loads(text.strip())
        except: return text
    
    print("\n" + "="*70)
    print("ğŸ“Š NEWS FRAMING ANALYSIS RESULTS")
    print("="*70)
    
    # --- DATA COLLECTION STATS ---
    search_results = state.get("search_results", {})
    full_articles = state.get("full_articles", {})
    
    total_found = sum(len(v) for v in search_results.values() if isinstance(v, list))
    total_scraped = sum(len(v) for v in full_articles.values() if isinstance(v, list))
    
    print(f"\nğŸ“° DATA COLLECTION STATS")
    print(f"Total Found: {total_found} | Total Scraped: {total_scraped}")
    print("-" * 40)
    
    for leaning in ["Left", "Center", "Right"]:
        found = search_results.get(leaning, [])
        scraped = full_articles.get(leaning, [])
        
        if found:
            sources = list(set([a.get('domain', 'unknown') for a in found]))
            print(f"   â€¢ {leaning:<7}: Found {len(found)} | Scraped {len(scraped)} ({', '.join(sources)})")

    # --- SYNTHESIS PARSING ---
    synthesis = parse_json_output(state.get("final_synthesis"))
    
    # --- 1. CORE EVENT & FACTS ---
    print("\n" + "-"*70)
    if isinstance(synthesis, dict):
        print(f"ğŸ“Œ CORE EVENT: {synthesis.get('core_event', 'N/A')}")
        
        stats = synthesis.get("key_statistics", {})
        if stats:
            print("\nğŸ”¢ KEY STATISTICS:")
            for k, v in stats.items():
                clean_key = k.replace("_", " ").title()
                print(f"   â€¢ {clean_key}: {v}")

    # --- 2. TIMELINE ---
    if isinstance(synthesis, dict):
        timeline = synthesis.get("timeline", [])
        if timeline:
            print(f"\nğŸ“… TIMELINE:")
            for item in timeline:
                if isinstance(item, dict):
                    date = item.get("date", "")
                    event = item.get("event", "")
                    line = f"{date}: {event}" if date else event
                else:
                    line = str(item)
                print(f"   â€¢ {line}")
    
    # --- 3. BIAS TAKEAWAY ---
    if isinstance(synthesis, dict):
        takeaway = synthesis.get("takeaway")
        if takeaway:
            print(f"\nâš–ï¸� BIAS LANDSCAPE: {str(takeaway)}")
    
    # --- 4. DETAILED ANALYSIS VIEWS ---
    print("\n" + "-"*70)
    print("ğŸ”� DETAILED FRAMING ANALYSIS")
    print("-"*70)
    
    views = {
        "big_picture_analysis": "ğŸ�¯ BIG PICTURE",
        "impact_analysis": "ğŸ’¥ IMPACT", 
        "human_story_analysis": "ğŸ‘¥ HUMAN STORY",
        "tone_analysis": "ğŸ�­ TONE"
    }
    
    for key, title in views.items():
        data = parse_json_output(state.get(key))
        if data and isinstance(data, dict):
            print(f"\n{title}")
            
            if "key_difference" in data:
                print(f"   ğŸ”‘ Key Difference: {data['key_difference']}\n")

            # for sub_key, sub_val in data.items():
            #     if sub_key != "key_difference" and isinstance(sub_val, dict):
            #         clean_label = sub_key.split('_')[0].upper()
            #         details = []
            #         for k, v in sub_val.items():
            #             if isinstance(v, str): details.append(f"{k}: {v}")
            #             elif isinstance(v, list): details.append(f"{k}: {', '.join(v)}")
            #         print(f"   [{clean_label}]: {'; '.join(details)}")

    # --- 5. COVERAGE GAPS (ONLY) ---
    if isinstance(synthesis, dict):
        coverage = synthesis.get("coverage_summary", {})
        if isinstance(coverage, dict):
            gaps = coverage.get("gaps")
            if gaps:
                print(f"\nâš ï¸� COVERAGE GAPS:")
                val_str = ", ".join(gaps) if isinstance(gaps, list) else str(gaps)
                # Wrap text slightly for readability if it's very long
                print(f"   {val_str}")
    
    print("\n" + "="*70)


# --- RUN ANALYSIS ---

# Initialize session service
session_service = InMemorySessionService()

APP_NAME = "news_framing_analyzer"
USER_ID = "demo_user_001"

news_app = App(
    name=APP_NAME,
    root_agent=root_agent,
    
    # Cache the heavy article text for the 5 parallel agents
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=300,
        cache_intervals=5 
    ),
    
    # Summarize history after the report is generated
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=4, 
        overlap_size=1
    )
)

runner = Runner(
    app=news_app,
    session_service=session_service
)

print("âœ… Runner created")
print(f"   App: {APP_NAME}")
print(f"   Root Agent: {root_agent.name}")

# Configuration (From allside media bias chart: https://www.allsides.com/media-bias/media-bias-chart)
runtime_config = {
    "domains": {
        "Left": ["apnews.com", "huffpost.com", "theguardian.com"],
        "Center": ["bbc.com", "reuters.com", "csmonitor.com"],
        "Right": ["foxnews.com", "nypost.com", "theepochtimes.com"]
    },
    "time_range": "d15",
    "results_per_domain": 5
}

# Topic
TOPIC_1 = "Washington D.C. Shooting"
SESSION_ID_1 = "framing_analysis_001"

# Run
session, report = await run_analysis(
    runner_instance=runner,
    topic=TOPIC_1,
    config=runtime_config,
    session_id=SESSION_ID_1,
    user_id=USER_ID
)


# Display results
print_results(session)


## For checking, if character counts of articles were enough for triggering caching

await inspect_session_data(runner, SESSION_ID_1, USER_ID)

