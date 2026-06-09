import os
import httpx
import xml.etree.ElementTree as ET

from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# ==========================
# CONFIGURE GEMINI API KEY
# ==========================

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    raise RuntimeError(
        "Missing GOOGLE_API_KEY in Kaggle secrets. Please add it."
    ) from e

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-2.5-flash-lite"

# ==========================
# CONSTANTS (TheSportsDB + Sky)
# ==========================

SPORTSDB_LEAGUE_ID = 4328              # Premier League ID on TheSportsDB
SPORTSDB_SEASON = "2025-2026"          # Adjust if needed (e.g., "2023-2024")
SPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"

SKY_RSS_URL = "https://www.skysports.com/rss/11095"  # PL RSS feed


# ==========================
# FETCH SCORES FOR GAMEWEEK (ROUND) 13
# ==========================

def fetch_scores_gameweek(round_number: int = 13):
    """
    Fetch all matches from a specific Premier League gameweek (round)
    using TheSportsDB eventsround.php endpoint.
    """
    url = (
        f"{SPORTSDB_BASE}/eventsround.php"
        f"?id={SPORTSDB_LEAGUE_ID}&r={round_number}&s={SPORTSDB_SEASON}"
    )
    print(f"â�¡ï¸� Fetching PL gameweek {round_number} from TheSportsDB...")
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()

    # TheSportsDB uses "events" for this endpoint
    events = data.get("events") or []
    if not events:
        raise RuntimeError(
            f"No events returned for round {round_number}, "
            f"season {SPORTSDB_SEASON}. Raw JSON: {data}"
        )

    matches = []
    for e in events:
        # Skip incomplete / not yet played games
        if not (e.get("strHomeTeam") and e.get("strAwayTeam")):
            continue
        if e.get("intHomeScore") is None or e.get("intAwayScore") is None:
            continue

        try:
            home_goals = int(e["intHomeScore"])
            away_goals = int(e["intAwayScore"])
        except (ValueError, TypeError):
            continue

        matches.append(
            {
                "home": e["strHomeTeam"],
                "away": e["strAwayTeam"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "date": e.get("dateEvent"),
                "round": e.get("intRound"),
                "source": "sportsdb-eventsround",
            }
        )

    print(f"âœ… Retrieved {len(matches)} matches for round {round_number}.")
    return matches


# ==========================
# SKY SPORTS RSS CONTEXT
# ==========================

def fetch_quotes_from_sky():
    """
    Fetch Sky Sports PL RSS and extract headline + description.
    Uses built-in XML parser.
    """
    try:
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            resp = client.get(SKY_RSS_URL)
            resp.raise_for_status()
            xml_text = resp.text
    except httpx.HTTPError as e:
        print(f"âš ï¸� Error fetching Sky Sports RSS: {e}")
        print("   Continuing with empty quotes/context.\n")
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"âš ï¸� Error parsing Sky Sports RSS XML: {e}")
        print("   Continuing with empty quotes/context.\n")
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    quotes = []
    for item in channel.findall("item")[:10]:  # first 10 items
        title_el = item.find("title")
        desc_el = item.find("description")
        link_el = item.find("link")
        quotes.append(
            {
                "headline": title_el.text if title_el is not None else "",
                "summary": desc_el.text if desc_el is not None else "",
                "link": link_el.text if link_el is not None else "",
            }
        )
    return quotes


# ==========================
# ARTICLE WRITER (3â€“5 sentences per match)
# ==========================

def generate_article(scores, quotes):
    """
    Use Gemini to write ~3â€“5 sentences per match for the specified gameweek.
    """
    prompt = f"""
You are a Premier League football journalist.

Write short match summaries for the specified gameweek based on the structured data.

MATCH DATA (JSON-like list of dicts):
{scores}

Sky Sports context (headlines + summaries) (JSON-like list of dicts):
{quotes}

Requirements:
- For each match in the list, write a separate block.
- Each block should have 3â€“5 sentences describing the result, key talking points,
  and a bit of context (form, stakes, or notable performers).
- You may loosely use the Sky Sports headlines/summaries for flavour, but do NOT invent
  obviously fake details or minute-by-minute descriptions.
- Plain text output, structured like:

Match: <Home> <home_goals>-<away_goals> <Away>
Sentence 1.
Sentence 2.
Sentence 3.
(optional Sentence 4â€“5)

(blank line)
Next match...
"""

    model = genai.GenerativeModel(MODEL_NAME)
    result = model.generate_content(prompt)
    return result.text


# ==========================
# FACT CHECK (LIGHT)
# ==========================

def fact_check_article(article_text, scores):
    """
    Light fact-checking of article text vs the score data.
    """
    prompt = f"""
You are a fact-checker.

ARTICLE:
{article_text}

SCORES DATA (JSON-like):
{scores}

Task:
- Check that scorelines and team names in the article match the JSON-like data.
- Correct any mismatches directly in the text.
- At the end, append:

Fact-check summary:
- <notes of any corrections, or 'No corrections needed.'>

Return ONLY the corrected article with the summary.
"""

    model = genai.GenerativeModel(MODEL_NAME)
    result = model.generate_content(prompt)
    return result.text


# ==========================
# PUBLISHER (PRINT)
# ==========================

def publish_article(article_text):
    """Publish by printing to stdout."""
    print("\n" + "=" * 70)
    print("FINAL PUBLISHED ARTICLE")
    print("=" * 70)
    print(article_text)
    print("=" * 70)
    print("End of article.\n")


# ==========================
# MAIN PIPELINE
# ==========================

def run_pipeline():
    print("ğŸ�† Running PL newsroom pipeline (TheSportsDB GW13 + Sky RSS)...\n")

    try:
        scores = fetch_scores_gameweek(round_number=13)
    except Exception as e:
        print(f"â�Œ Failed to fetch scores for GW13: {e}")
        return

    quotes = fetch_quotes_from_sky()

    print("=== GAMEWEEK 13 SCORES (from TheSportsDB) ===")
    for s in scores:
        print(s)

    print("\n=== SKY SPORTS CONTEXT (headlines/summaries) ===")
    for q in quotes[:5]:
        print(f"- {q['headline']}")

    # Generate short match summaries
    article = generate_article(scores, quotes)
    print("\n=== DRAFT ARTICLE (SHORT SUMMARIES) ===")
    print(article)

    # Fact-check against score data
    checked_article = fact_check_article(article, scores)
    print("\n=== FACT-CHECKED ARTICLE ===")
    print(checked_article)

    # "Publish"
    publish_article(checked_article)


if __name__ == "__main__":
    run_pipeline()





