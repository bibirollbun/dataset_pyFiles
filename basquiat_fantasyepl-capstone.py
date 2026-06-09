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


from pathlib import Path

WRITING_SAMPLE_PATHS = [
    "/kaggle/input/writing-samples/Gameweek5Review.txt",
    "/kaggle/input/writing-samples/Gameweek13Preview.txt",
]

writing_samples = {}

for path in WRITING_SAMPLE_PATHS:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Missing writing sample: {file_path}")

    writing_samples[file_path.name] = file_path.read_text(encoding="utf-8")

print(f"Loaded {len(writing_samples)} writing samples.")

for name, text in writing_samples.items():
    print(f"{name}: {len(text):,} characters")


COMMISSIONER_WRITING_SAMPLES = "\n\n".join(
    [
        f"--- {name} ---\n{text}"
        for name, text in writing_samples.items()
    ]
)

print(f"Loaded {len(writing_samples)} writing samples.")
print(f"Total commissioner sample context: {len(COMMISSIONER_WRITING_SAMPLES):,} characters")


COMMISSIONER_STYLE_GUIDE = """
Write like Emmett, commissioner of a private Fantrax Fantasy EPL league.

Identity:
- You are inside the league, not an outside analyst.
- You speak to friends and league rivals, not a public audience.
- You are funny, direct, competitive, and league-specific.
- You can be smug, self-aware, and lightly sarcastic, but not mean.

Use:
- Manager names and team names when available.
- Matchup stakes, table position, recent form, rivalry context, and league history.
- Short, punchy transitions.
- Specific football/player/fixture context only when it is present in the supplied data.
- The writing samples as the main model for rhythm, humor, and structure.

Avoid:
- Generic fantasy football analysis.
- Corporate-sounding prose.
- Overexplaining the obvious.
- Inventing scores, standings, injuries, player news, fixtures, or league history.
- Pretending to know something that is not in the data.
"""


COMMISSIONER_STYLE_CONTEXT = f"""
Commissioner style guide:
{COMMISSIONER_STYLE_GUIDE}

Commissioner writing samples:
{COMMISSIONER_WRITING_SAMPLES}
"""

print("Commissioner style context created âœ…")
print(f"Style context length: {len(COMMISSIONER_STYLE_CONTEXT):,} characters")


!pip install -q "google-generativeai>=0.7.2" "google-adk>=0.5.0"

print("âœ… Installed google-generativeai + google-adk")


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

    print("âœ… Gemini API key setup complete.")
    print("âœ… GOOGLE_GENAI_USE_VERTEXAI set to FALSE.")
except Exception as e:
    print(
        "ğŸ”‘ Authentication Error: Please make sure you have added "
        "'GOOGLE_API_KEY' to Kaggle Secrets and enabled notebook access."
    )
    raise e


# === Fantasy EPL Commissioner Agent â€” ADK Imports ===

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

print("âœ… ADK components imported successfully.")


def get_gw_winner(gameweek: int) -> str:
    """
    Return the winner/result for a specific Fantasy EPL gameweek.
    Current prototype data includes Gameweek 12.
    """
    gw_results = {
        12: "Cullen beat Liese."
    }

    return gw_results.get(
        gameweek,
        f"Gameweek {gameweek} result is not loaded in the current dataset."
    )


gw_tool = FunctionTool(get_gw_winner)

print("gw_tool created âœ…")
print("Sanity:", get_gw_winner(12))


# === League metadata, Fantrax snapshot loader, and lookup tools ==========
# Static metadata: managers, club support, rivalry notes.
# Weekly snapshot data: loaded from the latest Fantrax standings CSV.
# In production, replace the CSV weekly or automate the Fantrax export pull.

from pathlib import Path
import csv
import re
import pandas as pd

# --- Static manager/team metadata ---------------------------------------

TEAMS = {
    "We Go Again": {
        "manager": "Emmett",
        "club_support": "Liverpool",
    },
    "Turnip the Volume": {
        "manager": "Will",
        "club_support": "Liverpool",
    },
    "Gunner Gang": {
        "manager": "Brandyn Six",
        "club_support": "Arsenal",
    },
    "YouBeckerBelieveIt": {
        "manager": "Liese",
        "club_support": "Liverpool",
        "aliases": ["YouBeckerBelievelt"],
    },
    "FUMNUFC": {
        "manager": "F. Scott Fein",
        "club_support": "Liverpool",
    },
    "CFahey": {
        "manager": "Cullen",
        "club_support": None,
    },
    "Keep Calm": {
        "manager": "Ryan T",
        "club_support": "Manchester United",
    },
    "Bluedozers": {
        "manager": "E",
        "club_support": "Chelsea",
    },
    "Mighty Lumpers": {
        "manager": "Jon Fein",
        "club_support": None,
        "notes": "Twin brother of F. Scott â€“ the Fein Derby.",
    },
    "tmeach": {
        "manager": "Thomas",
        "club_support": "Liverpool",
    },
    "Undefeatable Blue Marble Dude Gunners": {
        "manager": "Ben",
        "club_support": "Arsenal",
    },
    "The Absolute Wirtz": {
        "manager": "Chad",
        "club_support": "Spurs",
    },
    "Larchuma FC": {
        "manager": "Lucas",
        "club_support": None,
    },
    "conorbritton": {
        "manager": "Conor B",
        "club_support": None,
    },
}

TEAM_ALIASES = {
    alias: team_name
    for team_name, info in TEAMS.items()
    for alias in info.get("aliases", [])
}


def canonical_team_name(team_name: str) -> str:
    """Return the canonical Fantrax team name when an alias is used."""
    return TEAM_ALIASES.get(team_name, team_name)


# --- Fantrax CSV parsing -------------------------------------------------

def find_latest_fantrax_csv() -> Path:
    """
    Find the newest Fantrax standings CSV available to the notebook.

    Recommended workflow:
    - Upload/export the weekly Fantrax CSV.
    - Place it in Kaggle input or /kaggle/working/data/raw/.
    - This function finds the latest matching file.
    """
    search_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working/data/raw"),
        Path("."),
    ]

    patterns = [
        "*Fantrax*Standing*.csv",
        "*Fantrax*Standings*.csv",
        "*fantrax*standing*.csv",
        "*fantrax*standings*.csv",
        "*Fantrax_Standing*.csv",
        "*Fantrax_Standings*.csv",
        "*fantrax_standing*.csv",
        "*fantrax_standings*.csv",
    ]

    candidates = []

    for root in search_roots:
        if root.exists():
            for pattern in patterns:
                candidates.extend(root.rglob(pattern))

    # Remove duplicates while preserving paths
    candidates = list(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            "No Fantrax standings CSV found. Upload a file with a name like "
            "'Fantrax-Standings-Marley Superior.csv' or 'Fantrax_Standings_CSV.csv'."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_number(value):
    """Convert Fantrax numeric strings like '3,406.5' into numbers."""
    if value is None or value == "":
        return None

    cleaned = str(value).replace(",", "").strip()

    try:
        number = float(cleaned)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


def parse_fantrax_sections(csv_path: Path) -> dict:
    """
    Parse a Fantrax CSV that contains multiple sections:
    - Standings
    - Gameweek XX
    - Gameweek YY
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    sections = {}
    i = 0

    while i < len(rows):
        row = rows[i]

        # Skip blank rows
        if not row or not any(cell.strip() for cell in row):
            i += 1
            continue

        section_name = row[0].strip()
        i += 1

        if i >= len(rows):
            break

        headers = [h.strip() for h in rows[i]]
        i += 1

        data_rows = []
        while i < len(rows) and any(cell.strip() for cell in rows[i]):
            data_rows.append(rows[i])
            i += 1

        if headers and data_rows:
            sections[section_name] = pd.DataFrame(data_rows, columns=headers)

    return sections


FANTRAX_CSV_PATH = find_latest_fantrax_csv()
sections = parse_fantrax_sections(FANTRAX_CSV_PATH)

print(f"Loaded Fantrax CSV: {FANTRAX_CSV_PATH}")


# --- Build current league table from Fantrax snapshot --------------------

standings_df = sections["Standings"].copy()

standings_df = standings_df.rename(
    columns={
        "Rk": "rank",
        "Team": "team",
        "W": "wins",
        "D": "draws",
        "L": "losses",
        "Points": "points",
        "Win%": "win_pct",
        "FPtsF": "fpts_for",
        "FPtsA": "fpts_against",
        "Streak": "streak",
    }
)

for col in ["rank", "wins", "draws", "losses", "points", "win_pct", "fpts_for", "fpts_against"]:
    standings_df[col] = standings_df[col].apply(parse_number)

standings_df["team"] = standings_df["team"].apply(canonical_team_name)
standings_df["record"] = (
    standings_df["wins"].astype(str)
    + "-"
    + standings_df["draws"].astype(str)
    + "-"
    + standings_df["losses"].astype(str)
)

LEAGUE_TABLE = standings_df.to_dict(orient="records")


# --- Build current and previous gameweek matchup snapshots ---------------

gameweek_sections = {}

for section_name, df in sections.items():
    match = re.match(r"Gameweek\s+(\d+)", section_name)
    if match:
        gameweek_sections[int(match.group(1))] = df.copy()

if not gameweek_sections:
    raise ValueError("No Gameweek sections found in Fantrax CSV.")

CURRENT_GAMEWEEK = max(gameweek_sections.keys())
PREVIOUS_GAMEWEEK = max([gw for gw in gameweek_sections.keys() if gw < CURRENT_GAMEWEEK], default=None)


def matchup_df_to_rows(df: pd.DataFrame, gameweek: int) -> list:
    """Convert Fantrax matchup table into normalized matchup rows."""
    rows = []

    for _, row in df.iterrows():
        away_team = canonical_team_name(row.iloc[0])
        away_fpts = parse_number(row.iloc[1])
        home_team = canonical_team_name(row.iloc[2])
        home_fpts = parse_number(row.iloc[3])

        rows.append(
            {
                "gameweek": gameweek,
                "away": away_team,
                "away_fpts": away_fpts,
                "home": home_team,
                "home_fpts": home_fpts,
            }
        )

    return rows


CURRENT_GAMEWEEK_MATCHUPS = matchup_df_to_rows(
    gameweek_sections[CURRENT_GAMEWEEK],
    CURRENT_GAMEWEEK,
)

PREVIOUS_GAMEWEEK_RESULTS = (
    matchup_df_to_rows(gameweek_sections[PREVIOUS_GAMEWEEK], PREVIOUS_GAMEWEEK)
    if PREVIOUS_GAMEWEEK is not None
    else []
)

# Backward-compatible name for your existing agent/tool wording.
UPCOMING_FIXTURES = CURRENT_GAMEWEEK_MATCHUPS


# --- Tool functions to expose this data to the agent ---------------------

def get_league_table() -> list:
    """Return the current Fantrax league table snapshot."""
    return LEAGUE_TABLE


def get_upcoming_fixtures() -> list:
    """Return the current gameweek matchup snapshot from Fantrax."""
    return UPCOMING_FIXTURES


def get_previous_gameweek_results() -> list:
    """Return the previous gameweek results from the Fantrax snapshot."""
    return PREVIOUS_GAMEWEEK_RESULTS


def get_team_info(team_name: str) -> dict:
    """Return mostly static manager/team metadata for a given Fantasy EPL team."""
    canonical_name = canonical_team_name(team_name)
    info = TEAMS.get(canonical_name)

    if info is None:
        return {"error": f"Unknown team: {team_name}"}

    return {
        "team": canonical_name,
        **info,
    }


league_table_tool = FunctionTool(get_league_table)
fixtures_tool = FunctionTool(get_upcoming_fixtures)
previous_results_tool = FunctionTool(get_previous_gameweek_results)
team_info_tool = FunctionTool(get_team_info)

print(f"Loaded {len(TEAMS)} teams.")
print(f"Loaded {len(LEAGUE_TABLE)} league table rows.")
print(f"Current gameweek: {CURRENT_GAMEWEEK}")
print(f"Loaded {len(CURRENT_GAMEWEEK_MATCHUPS)} current gameweek matchups.")
print(f"Previous gameweek: {PREVIOUS_GAMEWEEK}")
print(f"Loaded {len(PREVIOUS_GAMEWEEK_RESULTS)} previous gameweek results.")
print("League lookup tools created âœ…")


# === League context + style embedded into commissioner_agent ============

# --- Scoring rubric (summary only; flavour, not hard math) --------------

SCORING_RUBRIC_SUMMARY = """
General:
- Fantrax H2H points league with custom scoring by position.

Goalkeepers:
- Clean sheet on field: +4 pts
- Goals against: -1 pt for every 2 GA (cumulative)
- Goal scored: +10 pts
- Assists (official or fantasy): +9 pts
- Key passes: +6 pts
- Minutes played: +1 pt per minute, with 1â€“59 and 60â€“90 tracked separately
- Saves: +2 pts for every 3 saves (cumulative)
- Penalty save: +8 pts
- Own goal: -4 pts
- Yellow card: -3 pts
- Red card: -7 pts

Outfield baseline (non-positional overrides):
- Goal: +9 pts
- Assist (official or fantasy): +6 pts
- Key pass: +2 pts
- Shot on target: +2 pts
- Tackles won + interceptions: +1 pt per 3 combined (cumulative)
- Minutes played: +1 pt per minute (1â€“59, 60â€“90 tracked)
- Penalty missed: -2 pts
- Own goal: -4 pts
- Yellow card: -3 pts
- Red card: -7 pts

Defender overrides:
- Clean sheet: +6 pts
- Goal: +10 pts
- Aerials won: +1 pt each (instead of 0.5)
- Goals against: -1 pt for every 2 GA
- Tackles won + interceptions: +1 pt per 2 combined

Midfielder overrides:
- Clean sheet: +1 pt
- Goal: +9 pts (same as forwards but from deeper positions)

Forward overrides:
- Goal: +9 pts (pure finishing rewarded heavily)
"""

print("Scoring context loaded âœ…")



# === Agent instruction ====================================================

AGENT_INSTRUCTION = f"""
You are the Fantasy EPL Commissioner AI for Emmett's private Fantrax league.

Your job is to write accurate, funny, league-specific Fantasy EPL previews and recaps using:
- Fantrax snapshot data
- league standings
- current and previous gameweek matchups
- manager/team metadata
- scoring context
- commissioner writing samples

Core behavior:
- Write like an insider speaking to the league, not like a neutral fantasy analyst.
- Use the commissioner voice and writing samples as the style model.
- Be league-specific, funny, competitive, and direct.
- Prefer concrete matchup stakes over generic fantasy analysis.
- Keep trash talk playful and good-natured.

Use the available tools proactively:
- Use get_league_table for current standings, records, points, fantasy points, and streaks.
- Use get_upcoming_fixtures for the current gameweek matchups.
- Use get_previous_gameweek_results for the previous gameweek results.
- Use get_team_info for manager names, club allegiances, rivalry notes, and league metadata.
- Use get_gw_winner only when asked about a specific loaded gameweek result.

Accuracy rules:
- Treat Fantrax snapshot data as the source of truth for standings, matchups, and scores.
- Do not invent scores, standings, fixtures, injuries, player news, rosters, transactions, or league history.
- Do not explain why a manager won or lost unless player-level scoring data is available.
- If player-level data is missing, you may discuss matchup stakes and scorelines, but do not invent player performances.
- If needed data is missing, say what is missing.

Preview rules:
- For previews, use the league table and current gameweek matchups.
- Mention table position, records, points, streaks, and rivalry/personality context where relevant.
- Do not pretend to know lineups, injuries, or player news unless those data are provided.

Review rules:
- For reviews, use previous gameweek results and the updated league table.
- Discuss winners, losers, margins, table implications, and recurring league narratives.
- Do not attribute wins to specific players unless player-score data are provided.

Commissioner style context:
{COMMISSIONER_STYLE_CONTEXT}

Scoring context:
{SCORING_RUBRIC_SUMMARY}
"""

print("Agent instruction created âœ…")
print(f"Instruction length: {len(AGENT_INSTRUCTION):,} characters")


# === Agent, session, runner, and chat helper =============================

APP_NAME = "fantasy_epl_commish"
USER_ID = "emmett"
SESSION_ID = "dev-session-1"

# --- Create the commissioner agent ---------------------------------------

commissioner_agent = LlmAgent(
    name="fantasy_commissioner",
    model="gemini-2.5-flash-lite",
    description="Agent that writes Fantasy EPL previews and recaps for Emmett's Fantrax league.",
    instruction=AGENT_INSTRUCTION,
    tools=[
        gw_tool,
        league_table_tool,
        fixtures_tool,
        previous_results_tool,
        team_info_tool,
    ],
)

# --- Create session service ----------------------------------------------

session_service = InMemorySessionService()

await session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
)

print("Session created âœ…")

# --- Create runner --------------------------------------------------------

runner = Runner(
    agent=commissioner_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

print("Runner ready âœ…")


def extract_text(parts) -> str:
    """Extract text from ADK response parts, including tool/function responses."""
    texts = []

    for part in parts or []:
        # Normal model text
        text = getattr(part, "text", None)
        if text:
            texts.append(str(text))

        # Tool/function response content
        function_response = getattr(part, "function_response", None)
        if function_response:
            response = getattr(function_response, "response", None)
            if response:
                texts.append(str(response))

    combined = "\n".join([text.strip() for text in texts if text and text.strip()])
    return combined if combined else "No text in final response."


def talk_to_commissioner(message: str) -> str:
    """Send a prompt to the Fantasy EPL commissioner agent and return final text."""
    user_msg = types.Content(
        role="user",
        parts=[types.Part(text=message)],
    )

    final_text = "No response received."

    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_msg,
    ):
        if event.is_final_response():
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None) if content else None
            final_text = extract_text(parts)

    return final_text


print("Commissioner agent ready âœ…")


# === Smoke tests ==========================================================

print(talk_to_commissioner("In one short sentence, introduce yourself."))
print(talk_to_commissioner("Who won gameweek 12?"))


print(
    talk_to_commissioner(
        "Write a short preview of the current Fantrax gameweek. "
        "Use the league table and current matchups. "
        "Do not mention player news."
    )
)


print(
    talk_to_commissioner(
        "Using our Fantrax scoring and league table, write a short preview of this upcoming gameweek's fixtures."
    )
)



print(
    talk_to_commissioner(
        "Write a detailed preview for this weekend's Fantrax gameweek (Gameweek 14) in my usual style."
    )
)



from difflib import SequenceMatcher

def run_minimal_evaluation():
    """
    Minimal evaluation for the capstone:
    - Sends a fixed test question to the agent.
    - Compares the agent's output to the expected answer.
    - Prints a simple 'Response Match Score' between 0 and 1.
    """
    user_input = "Who won gameweek 12?"
    expected_output = "Cullen beat Liese."

    actual_output = talk_to_commissioner(user_input)

    # crude similarity score (0â€“1)
    score = SequenceMatcher(
        None, expected_output.lower().strip(), actual_output.lower().strip()
    ).ratio()

    print("User input:          ", user_input)
    print("Expected response:   ", expected_output)
    print("Actual agent output: ", actual_output)
    print("Response Match Score:", round(score, 3))

run_minimal_evaluation()


