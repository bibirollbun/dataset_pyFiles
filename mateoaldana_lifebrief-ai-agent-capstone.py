import os
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from kaggle_secrets import UserSecretsClient

# ADK imports
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner

# Get API key from Kaggle Secrets and expose it as env variable for ADK

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



!adk create life_brief_agent --model gemini-2.0-flash --api_key $GOOGLE_API_KEY


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


%%writefile life_brief_agent/agent.py

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini


"""
LifeBrief Agent â€“ Morning Newspaper for a Professional in Madrid (Synthetic Data)

This cell defines:
- Synthetic datasets (news, soccer, music, events, flights, scholarships, groceries)
- A simple, anonymized UserProfile for a professional based in Madrid
- A small in-memory â€œmemoryâ€� store for display preferences and exercise state
- Tool functions used by the ADK Agent, including a composed get_morning_life_brief

All data is synthetic and self-contained for the Agents Intensive Capstone.
"""

# -----------------------------------------------------------------------------
# 1. User profile and in-memory "data sources"
# -----------------------------------------------------------------------------


@dataclass
class UserProfile:
    user_id: str
    name: str
    home_city: str
    country: str
    tz: str
    home_airport: str
    industry_tags: List[str]
    soccer_teams: List[str]
    music_artists: List[str]
    health_goal: str
    preferred_grocery_store: str  # "Costco" or "Walmart" (or stand-ins)
    

DEFAULT_PROFILE = UserProfile(
    user_id="default",
    name="Professional",
    home_city="Madrid",
    country="Spain",
    tz="Europe/Madrid",
    home_airport="MAD",
    industry_tags=[
        "engineering",
        "energy",
        "renewable energy",
        "sustainability",
        "innovation",
        "energy transition",
        "mechanical engineering",
        "cleantech",
        "decarbonization",
    ],
    # Favorite teams â€“ synthetic usage
    soccer_teams=[
        "Bayern Munich",
        "Crystal Palace",
        "Sporting CP",
        "Colombia Men's National Team",
    ],
    # EDM / electronic artists (synthetic preferences)
    music_artists=["Disclosure", "Fred Again..", "Kaytranada", "Salute DJ"],
    health_goal=(
        "Visible muscle plus strong cardio and injury resilience, "
        "with 6 Ã— 10K runs, 4 strength days per week, and a daily 100s challenge "
        "(100 pushups, 100 situps, 100 squats)."
    ),
    preferred_grocery_store="Costco",
)

# --- Simple memory store -------------------------------------------------------

DEFAULT_DISPLAY_PREFS = {
    "detail_level": "normal",  # "minimal", "normal", "verbose"
    "show_emojis": True,
    "layout": "sections",      # "sections" or "plain"
}

DEFAULT_EXERCISE_STATE = {
    "injured": False,
    "streak_days": 0,
    "last_workout_date": None,
}

DEFAULT_ECONOMIST_PREFS = {
    "sections": [
        "international",
        "science-and-technology",
        "latin-america",
        "energy",
        "canada",
        "colombia",
        "decarbonization",
    ],
    "last_article_id": None,
    "last_section_index": 0,
}

MEMORY: Dict[str, Dict[str, Any]] = {
    "default": {
        "display_prefs": DEFAULT_DISPLAY_PREFS.copy(),
        "exercise_state": DEFAULT_EXERCISE_STATE.copy(),
        "economist": DEFAULT_ECONOMIST_PREFS.copy(),
    }
}


def _get_user_mem(user_id: str = "default") -> Dict[str, Any]:
    """Return (and lazily initialize) the memory bucket for a user_id."""
    if user_id not in MEMORY:
        MEMORY[user_id] = {
            "display_prefs": DEFAULT_DISPLAY_PREFS.copy(),
            "exercise_state": DEFAULT_EXERCISE_STATE.copy(),
            "economist": DEFAULT_ECONOMIST_PREFS.copy(),
        }
    return MEMORY[user_id]


# --- Synthetic â€œnewsâ€� items related to energy / agents -------------------------

NEWS_ITEMS = [
    {
        "title": "Student teams race to design ultra-efficient hydrogen storage systems",
        "url": "https://example.com/h2-storage-students",
        "tags": ["engineering", "energy", "innovation", "energy transition"],
    },
    {
        "title": "Renewable microgrids boost energy resilience in remote communities",
        "url": "https://example.com/microgrid-resilience",
        "tags": ["renewable energy", "sustainability", "equity"],
    },
    {
        "title": "AI agents move from prototypes to production in energy management",
        "url": "https://example.com/ai-agents-energy",
        "tags": ["innovation", "agents", "energy transition"],
    },
]

# --- Synthetic Economist-style article data -----------------------------------

ECONOMIST_ARTICLES = [
    {
        "id": "eco-001",
        "title": "Why non-alcoholic drinks are reshaping nightlife",
        "section": "business",
        "url": "https://example.com/economist-na-nightlife",
        "summary": "How grown-up non-alcoholic options are changing bar menus and margins.",
    },
    {
        "id": "eco-002",
        "title": "The economics of green hydrogen: hype vs reality",
        "section": "energy",
        "url": "https://example.com/economist-green-h2",
        "summary": "Hydrogen's promise in the energy transition and the hurdles to scaling.",
    },
    {
        "id": "eco-003",
        "title": "AI agents escape the lab",
        "section": "science-and-technology",
        "url": "https://example.com/economist-ai-agents",
        "summary": "Why agentic AI could change how people work with software.",
    },
    {
        "id": "eco-004",
        "title": "Power, politics and pipelines in Latin America",
        "section": "latin-america",
        "url": "https://example.com/economist-latam-energy",
        "summary": "How Latin America is juggling fossil exports and renewables.",
    },
    {
        "id": "eco-005",
        "title": "A patchwork of climate policy",
        "section": "international",
        "url": "https://example.com/economist-international-climate",
        "summary": "Why global climate efforts are fragmented, and where progress is real.",
    },
]

# --- Synthetic soccer fixtures -------------------------------------------------


SOCCER_FIXTURES = [
    {
        "team": "Bayern Munich",
        "last_result": "Bayern 3â€“1 Dortmund",
        "next_opponent": "RB Leipzig",
        "next_date": "2025-12-06",
        "competition": "Bundesliga",
    },
    {
        "team": "Crystal Palace",
        "last_result": "Palace 1â€“1 West Ham",
        "next_opponent": "Arsenal",
        "next_date": "2025-12-07",
        "competition": "Premier League",
    },
    {
        "team": "Sporting CP",
        "last_result": "Sporting 2â€“1 Porto",
        "next_opponent": "Benfica",
        "next_date": "2025-12-05",
        "competition": "Primeira Liga",
    },
]

# --- Synthetic music releases / concerts (EDM-ish) ----------------------------


MUSIC_RELEASES = [
    {
        "artist": "Calvin Harris",
        "title": "Sunset Skyline (Single)",
        "release_date": "2025-11-20",
        "url": "https://example.com/calvin-sunset",
    },
    {
        "artist": "Martin Garrix",
        "title": "Voltage Dreams (EP)",
        "release_date": "2025-11-18",
        "url": "https://example.com/garrix-voltage",
    },
]

CONCERTS = [
    {
        "artist": "Calvin Harris",
        "city": "Madrid",
        "venue": "Estadio Santiago BernabÃ©u",
        "date": "2026-06-15",
        "url": "https://example.com/calvin-madrid",
    },
    {
        "artist": "Above & Beyond",
        "city": "Barcelona",
        "venue": "Palau Sant Jordi",
        "date": "2026-03-05",
        "url": "https://example.com/above-beyond-barcelona",
    },
]

# --- Synthetic calendar / tasks ------------------------------------------------


CALENDAR_EVENTS = [
    {
        "title": "Thermodynamics midterm",
        "date": "2025-12-02",
        "time": "10:00",
        "location": "Engineering building",
        "type": "school_deadline",
    },
    {
        "title": "Startup investor check-in",
        "date": "2025-12-03",
        "time": "14:00",
        "location": "Online meeting",
        "type": "meeting",
    },
]

TASKS = [
    {"task": "Finish problem set 4", "due_date": "2025-12-01", "status": "open"},
    {"task": "Update agents capstone notebook", "due_date": "2025-12-02", "status": "open"},
    {"task": "Reply to partner email", "due_date": "2025-12-04", "status": "open"},
]

# --- Synthetic local events / date ideas for Madrid ---------------------------


LOCAL_EVENTS = [
    {
        "city": "Madrid",
        "name": "Winter Lights Walk",
        "date": "2025-12-06",
        "category": "outdoors/date-idea",
        "url": "https://example.com/madrid-winter-lights",
    },
    {
        "city": "Madrid",
        "name": "Indie Live Music Night",
        "date": "2025-12-08",
        "category": "music/date-idea",
        "url": "https://example.com/madrid-indie-night",
    },
]

# --- Synthetic flights --------------------------------------------------------


FLIGHT_DEALS = [
    {
        "origin": "MAD",
        "destination": "LIS",
        "price": 150,
        "currency": "EUR",
        "start_window": "2026-05-01",
        "end_window": "2026-08-31",
        "airline": "Iberia / TAP",
    },
    {
        "origin": "MAD",
        "destination": "MEX",
        "price": 650,
        "currency": "EUR",
        "start_window": "2026-03-01",
        "end_window": "2026-06-30",
        "airline": "AeromÃ©xico",
    },
]

# --- Synthetic scholarships ---------------------------------------------------


SCHOLARSHIPS = [
    {
        "name": "Mechanical Engineering Excellence Award",
        "deadline": "2026-01-15",
        "level": "undergrad",
        "region": "Europe",
        "url": "https://example.com/mech-excellence",
    },
    {
        "name": "Sustainable Energy Innovation Bursary",
        "deadline": "2025-12-20",
        "level": "undergrad",
        "region": "Europe",
        "url": "https://example.com/energy-bursary",
    },
]

# --- Grocery catalog ----------------------------------------------------------


GROCERY_CATALOG = [
    # Costco-like
    {
        "store": "Costco",
        "name": "Boneless Skinless Chicken Breast",
        "category": "protein",
        "health_tags": ["lean_protein"],
        "unit": "kg",
    },
    {
        "store": "Costco",
        "name": "Organic Mixed Greens",
        "category": "produce",
        "health_tags": ["fiber", "micronutrients"],
        "unit": "box",
    },
    {
        "store": "Costco",
        "name": "Brown Basmati Rice",
        "category": "carb",
        "health_tags": ["complex_carb"],
        "unit": "kg",
    },
    # Walmart-like
    {
        "store": "Walmart",
        "name": "Frozen Mixed Vegetables",
        "category": "produce",
        "health_tags": ["fiber"],
        "unit": "bag",
    },
    {
        "store": "Walmart",
        "name": "Greek Yogurt (Plain, 0%)",
        "category": "protein",
        "health_tags": ["lean_protein"],
        "unit": "tub",
    },
]


# -----------------------------------------------------------------------------
# 2. Helper utilities
# -----------------------------------------------------------------------------


def _today_str(tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    today = datetime.datetime.now(tz).date()
    return today.isoformat()


# -----------------------------------------------------------------------------
# 3. Tools: profile & overview
# -----------------------------------------------------------------------------


def get_user_profile(user_id: str = "default") -> Dict[str, Any]:
    """Return a simple, anonymized user profile."""
    profile = DEFAULT_PROFILE
    return {
        "status": "success",
        "profile": {
            "user_id": profile.user_id,
            "name": profile.name,
            "home_city": profile.home_city,
            "country": profile.country,
            "tz": profile.tz,
            "home_airport": profile.home_airport,
            "health_goal": profile.health_goal,
            "preferred_grocery_store": profile.preferred_grocery_store,
            "industry_tags": profile.industry_tags,
        },
    }


def get_today_overview(user_id: str = "default") -> Dict[str, Any]:
    """High-level 'today at a glance' summary: date, city, time zone, and a short headline."""
    profile = DEFAULT_PROFILE
    today = _today_str(profile.tz)
    return {
        "status": "success",
        "overview": {
            "date": today,
            "home_city": profile.home_city,
            "country": profile.country,
            "tz": profile.tz,
            "headline": f"Good morning, {profile.name}. Today is {today} in {profile.home_city}, {profile.country}.",
        },
    }


# -----------------------------------------------------------------------------
# 4. Tools: memory for display preferences & exercise state
# -----------------------------------------------------------------------------


def get_display_preferences(user_id: str = "default") -> Dict[str, Any]:
    mem = _get_user_mem(user_id)
    return {
        "status": "success",
        "display_prefs": mem.get("display_prefs", DEFAULT_DISPLAY_PREFS.copy()),
    }


def update_display_preferences(
    user_id: str = "default",
    detail_level: Optional[str] = None,
    show_emojis: Optional[bool] = None,
    layout: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update how the user wants information displayed.

    Example triggers (from natural language):
      - "From now on keep it really short."
      - "Stop using emojis."
      - "Give me more detail every morning."
      - "Make it plain text instead of sections."
    """
    mem = _get_user_mem(user_id)
    prefs = mem.get("display_prefs", DEFAULT_DISPLAY_PREFS.copy())

    if detail_level in {"minimal", "normal", "verbose"}:
        prefs["detail_level"] = detail_level
    if show_emojis is not None:
        prefs["show_emojis"] = bool(show_emojis)
    if layout in {"sections", "plain"}:
        prefs["layout"] = layout

    mem["display_prefs"] = prefs
    return {"status": "success", "display_prefs": prefs}


def get_exercise_state(user_id: str = "default") -> Dict[str, Any]:
    mem = _get_user_mem(user_id)
    return {"status": "success", "exercise_state": mem.get("exercise_state", {})}


def update_exercise_state(
    user_id: str = "default",
    injured: Optional[bool] = None,
    did_workout_today: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Update the exercise state based on user input.

    Example natural language mappings:
      - "My knee is better now." -> injured=False
      - "My knee is acting up again." -> injured=True
      - "I completed today's workout." -> did_workout_today=True
    """
    mem = _get_user_mem(user_id)
    state = mem.get("exercise_state", DEFAULT_EXERCISE_STATE.copy())

    tz = DEFAULT_PROFILE.tz
    today = _today_str(tz)

    if injured is not None:
        state["injured"] = bool(injured)

    if did_workout_today:
        if state.get("last_workout_date") != today:
            state["streak_days"] = state.get("streak_days", 0) + 1
            state["last_workout_date"] = today

    mem["exercise_state"] = state
    return {"status": "success", "exercise_state": state}


# -----------------------------------------------------------------------------
# 5. Tools: work / study (calendar + tasks)
# -----------------------------------------------------------------------------


def get_calendar_summary(days_ahead: int = 3, user_id: str = "default") -> Dict[str, Any]:
    profile = DEFAULT_PROFILE
    tz = profile.tz
    today = datetime.date.fromisoformat(_today_str(tz))

    upcoming = []
    for ev in CALENDAR_EVENTS:
        ev_date = datetime.date.fromisoformat(ev["date"])
        delta = (ev_date - today).days
        if 0 <= delta <= days_ahead:
            item = ev.copy()
            item["days_until"] = delta
            upcoming.append(item)

    return {"status": "success", "events": upcoming}


def get_task_summary(max_items: int = 3, user_id: str = "default") -> Dict[str, Any]:
    open_tasks = [t for t in TASKS if t["status"] == "open"]
    open_tasks.sort(key=lambda t: t["due_date"])
    return {"status": "success", "tasks": open_tasks[:max_items]}


# -----------------------------------------------------------------------------
# 6. Tools: health & exercise â€“ weekly training plan
# -----------------------------------------------------------------------------

WEEKLY_WORKOUT_PLAN = {
    # 0 = Monday, 6 = Sunday
    0: (
        "MONDAY â€” Easy Run + Light Full-Body / Core\n"
        "â€¢ Run: 10K at easy pace.\n"
        "â€¢ Gym (3 Ã— 12â€“15, lightâ€“moderate): goblet squats, machine chest press, "
        "seated row, cable woodchop, reverse fly, bird dog.\n"
        "â€¢ Daily 100s: 100 pushups, 100 situps, 100 squats (spread through the day)."
    ),
    1: (
        "TUESDAY â€” Lower Body A (Quads + Glutes)\n"
        "â€¢ Run: 10K easy.\n"
        "â€¢ Gym (3â€“4 sets): leg press, step-ups, leg curl, walking lunges, "
        "standing calf raises, hip thrust/glute bridge.\n"
        "â€¢ Daily 100s: pushups, situps, squats."
    ),
    2: (
        "WEDNESDAY â€” Tempo Run + Mobility\n"
        "â€¢ Run: 10K total â€“ 2K warm-up, 3 Ã— 2K at tempo pace (80â€“85%) with "
        "1K jog between, 1K cool-down.\n"
        "â€¢ Mobility (20â€“30 min): hip 90/90, pigeon, calf wall stretch, "
        "cat-cow, thread-the-needle, foam rolling.\n"
        "â€¢ Daily 100s: pushups, situps, squats."
    ),
    3: (
        "THURSDAY â€” Upper Body (Push + Pull)\n"
        "â€¢ Run: 10K easy.\n"
        "â€¢ Gym (3â€“4 sets): machine chest press, seated shoulder press, "
        "lat pulldown, seated row, triceps rope pushdown, DB curls.\n"
        "â€¢ Daily 100s: pushups, situps, squats."
    ),
    4: (
        "FRIDAY â€” Rest / Recovery Day\n"
        "â€¢ Run: No run (or optional 5K recovery jog / long walk).\n"
        "â€¢ Mobility: 25â€“40 min yoga-style flow and foam rolling focused on "
        "hips, calves, hamstrings, lower back.\n"
        "â€¢ Daily 100s: done at a relaxed pace, broken into small sets."
    ),
    5: (
        "SATURDAY â€” Long Easy Run + Core Stability\n"
        "â€¢ Run: 10â€“12K at relaxed pace.\n"
        "â€¢ Core & stability (3 rounds): plank, side plank, hanging knee raises, "
        "Pallof press, glute bridge holds.\n"
        "â€¢ Daily 100s: pushups, situps, squats."
    ),
    6: (
        "SUNDAY â€” Lower Body B (Posterior Chain)\n"
        "â€¢ Run: 10K easy.\n"
        "â€¢ Gym (3â€“4 sets): Romanian deadlifts with DBs, Bulgarian split squats, "
        "lateral band walks, single-leg calf raises, glute med clamshells, "
        "reverse hypers.\n"
        "â€¢ Daily 100s: pushups, situps, squats."
    ),
}

DAILY_100S_TEXT = (
    "Daily 100s Challenge: aim for 100 pushups, 100 situps, and 100 squats "
    "spread throughout the day, not all in one block."
)


def get_today_workout(user_id: str = "default", injured: Optional[bool] = None) -> Dict[str, Any]:
    profile = DEFAULT_PROFILE
    tz = profile.tz
    today_dt = datetime.datetime.now(ZoneInfo(tz))
    weekday = today_dt.weekday()  # 0=Mon ... 6=Sun

    mem = _get_user_mem(user_id)
    state = mem.get("exercise_state", DEFAULT_EXERCISE_STATE.copy())

    if injured is None:
        injured = state.get("injured", False)

    if injured:
        suggestions = {
            0: "Upper body strength and core, avoid deep squats.",
            1: "30 min brisk walk and stretching.",
            2: "Upper body strength plus light band work.",
            3: "Mobility-focused day with gentle yoga.",
            4: "Low-impact cardio such as cycling or elliptical for 30 min.",
            5: "Bodyweight strength movements without impact.",
            6: "Gentle yoga and foam rolling; listen to your body.",
        }
        workout_text = suggestions.get(
            weekday, "Easy movement and gentle mobility; adapt to how your body feels."
        )
    else:
        workout_text = WEEKLY_WORKOUT_PLAN.get(
            weekday, "Easy movement and stretching."
        )

    return {
        "status": "success",
        "workout": workout_text,
        "daily_100s": DAILY_100S_TEXT,
        "health_goal": profile.health_goal,
        "weekday_index": weekday,
        "exercise_state": state,
    }


# -----------------------------------------------------------------------------
# 7. Tools: interests â€“ soccer & music
# -----------------------------------------------------------------------------


def get_soccer_update(user_id: str = "default", max_teams: int = 3) -> Dict[str, Any]:
    profile = DEFAULT_PROFILE
    result_items = []
    for team in profile.soccer_teams[:max_teams]:
        for fixture in SOCCER_FIXTURES:
            if fixture["team"].lower() == team.lower():
                result_items.append(fixture)
                break
    return {"status": "success", "teams": result_items}


def get_music_update(
    user_id: str = "default",
    city_filter: Optional[str] = None,
) -> Dict[str, Any]:
    profile = DEFAULT_PROFILE
    if city_filter is None:
        city_filter = profile.home_city

    releases = [r for r in MUSIC_RELEASES if r["artist"] in profile.music_artists]
    concerts = [
        c
        for c in CONCERTS
        if c["artist"] in profile.music_artists
        and c["city"].lower() == city_filter.lower()
    ]

    return {
        "status": "success",
        "releases": releases,
        "concerts": concerts,
        "city": city_filter,
    }


# -----------------------------------------------------------------------------
# 8. Tools: events / flights / scholarships
# -----------------------------------------------------------------------------


def get_local_events(
    user_id: str = "default",
    days_ahead: int = 14,
    category_filter: Optional[str] = None,
) -> Dict[str, Any]:
    profile = DEFAULT_PROFILE
    tz = profile.tz
    today = datetime.date.fromisoformat(_today_str(tz))

    events: List[Dict[str, Any]] = []
    for ev in LOCAL_EVENTS:
        if ev["city"].lower() != profile.home_city.lower():
            continue
        ev_date = datetime.date.fromisoformat(ev["date"])
        delta = (ev_date - today).days
        if 0 <= delta <= days_ahead:
            if category_filter and category_filter not in ev["category"]:
                continue
            item = ev.copy()
            item["days_until"] = delta
            events.append(item)

    return {"status": "success", "events": events}


def get_flight_deals(
    origin: Optional[str] = None,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    max_results: int = 3,
    user_id: str = "default",
) -> Dict[str, Any]:
    profile = DEFAULT_PROFILE
    if origin is None:
        origin = profile.home_airport

    today = datetime.date.today()
    if window_start is None:
        window_start = (today + datetime.timedelta(days=180)).isoformat()
    if window_end is None:
        window_end = (today + datetime.timedelta(days=270)).isoformat()

    ws = datetime.date.fromisoformat(window_start)
    we = datetime.date.fromisoformat(window_end)

    deals: List[Dict[str, Any]] = []
    for deal in FLIGHT_DEALS:
        if deal["origin"].upper() != origin.upper():
            continue
        d_start = datetime.date.fromisoformat(deal["start_window"])
        d_end = datetime.date.fromisoformat(deal["end_window"])
        if d_start <= we and d_end >= ws:
            deals.append(deal)

    deals.sort(key=lambda d: d["price"])
    return {
        "status": "success",
        "origin": origin,
        "window_start": window_start,
        "window_end": window_end,
        "deals": deals[:max_results],
    }


def get_scholarship_reminders(
    user_id: str = "default", days_ahead: int = 45
) -> Dict[str, Any]:
    profile = DEFAULT_PROFILE
    tz = profile.tz
    today = datetime.date.fromisoformat(_today_str(tz))

    upcoming: List[Dict[str, Any]] = []
    for s in SCHOLARSHIPS:
        d = datetime.date.fromisoformat(s["deadline"])
        delta = (d - today).days
        if 0 <= delta <= days_ahead:
            item = s.copy()
            item["days_until"] = delta
            upcoming.append(item)

    upcoming.sort(key=lambda s: s["days_until"])
    return {"status": "success", "scholarships": upcoming}


# -----------------------------------------------------------------------------
# 9. Tools: grocery list generation
# -----------------------------------------------------------------------------


def generate_grocery_list(
    store: Optional[str] = None,
    focus: str = "support 6 Ã— 10K runs, 4 strength days, and the daily 100s challenge",
    user_id: str = "default",
) -> Dict[str, Any]:
    profile = DEFAULT_PROFILE
    if store is None:
        store = profile.preferred_grocery_store

    store = store.capitalize()
    candidate_items = [i for i in GROCERY_CATALOG if i["store"] == store]

    proteins = [i for i in candidate_items if i["category"] == "protein"]
    carbs = [i for i in candidate_items if i["category"] == "carb"]
    produce = [i for i in candidate_items if i["category"] == "produce"]

    items: List[Dict[str, Any]] = []

    if proteins:
        items.append(
            {
                "name": proteins[0]["name"],
                "category": "protein",
                "unit": proteins[0]["unit"],
                "rationale": "Lean protein to support muscle repair after runs and strength training.",
            }
        )

    if carbs:
        items.append(
            {
                "name": carbs[0]["name"],
                "category": "carb",
                "unit": carbs[0]["unit"],
                "rationale": "Complex carbs to fuel repeated 10K runs and a full training week.",
            }
        )

    if produce:
        items.append(
            {
                "name": produce[0]["name"],
                "category": "produce",
                "unit": produce[0]["unit"],
                "rationale": "Greens and vegetables for fiber and micronutrients.",
            }
        )

    return {
        "status": "success",
        "store": store,
        "focus": focus,
        "items": items,
        "note": (
            "In a real deployment, this could be connected to a live store API "
            "to check stock and prices. Here it is synthetic and fixed."
        ),
    }


# -----------------------------------------------------------------------------
# 10. Tools: Economist-style article-of-the-day (synthetic)
# -----------------------------------------------------------------------------


def get_economist_article_of_the_day(user_id: str = "default") -> Dict[str, Any]:
    """
    Returns one Economist-style article for the day using synthetic data.

    For a real deployment, this could be replaced with:
      - An Economist RSS feed (subject to terms of use), or
      - An official API for article metadata.

    For this competition, we stay fully synthetic and self-contained.
    """
    mem = _get_user_mem(user_id)
    econ_prefs = mem.get("economist", DEFAULT_ECONOMIST_PREFS.copy())

    sections = econ_prefs.get("sections", DEFAULT_ECONOMIST_PREFS["sections"])
    idx = econ_prefs.get("last_section_index", 0)
    section = sections[idx % len(sections)]

    last_id = econ_prefs.get("last_article_id")

    candidates = [a for a in ECONOMIST_ARTICLES if a["section"] == section]
    if not candidates:
        candidates = ECONOMIST_ARTICLES

    chosen = None
    for art in candidates:
        if art["id"] != last_id:
            chosen = art
            break
    if chosen is None:
        chosen = candidates[0]

    econ_prefs["last_article_id"] = chosen["id"]
    econ_prefs["last_section_index"] = (idx + 1) % len(sections)
    econ_prefs["sections"] = sections
    mem["economist"] = econ_prefs

    return {
        "status": "success",
        "article": chosen,
        "preferences": econ_prefs,
        "note": (
            "Synthetic Economist-style data for the capstone. "
            "In real use, replace with RSS/API integration."
        ),
    }


# -----------------------------------------------------------------------------
# 11. High-level tool: Morning LifeBrief
# -----------------------------------------------------------------------------


def get_morning_life_brief(
    user_id: str = "default",
    include_flights: bool = True,
    include_scholarships: bool = True,
    include_events: bool = True,
    include_economist: bool = True,
) -> Dict[str, Any]:
    """
    High-level tool that composes a structured 'morning newspaper' summary
    by calling the lower-level helpers.
    """
    profile = DEFAULT_PROFILE
    tz = profile.tz

    today_overview = get_today_overview(user_id=user_id)
    calendar = get_calendar_summary(user_id=user_id)
    tasks = get_task_summary(user_id=user_id)
    workout = get_today_workout(user_id=user_id)
    soccer = get_soccer_update(user_id=user_id)
    music = get_music_update(user_id=user_id, city_filter=profile.home_city)
    display_prefs = get_display_preferences(user_id=user_id)

    events_block: Optional[Dict[str, Any]] = None
    flights_block: Optional[Dict[str, Any]] = None
    scholarships_block: Optional[Dict[str, Any]] = None
    economist_block: Optional[Dict[str, Any]] = None

    if include_events:
        events_block = get_local_events(user_id=user_id)
    if include_flights:
        flights_block = get_flight_deals(user_id=user_id, max_results=1)
    if include_scholarships:
        scholarships_block = get_scholarship_reminders(user_id=user_id)
    if include_economist:
        economist_block = get_economist_article_of_the_day(user_id=user_id)

    return {
        "status": "success",
        "tz": tz,
        "display_prefs": display_prefs.get("display_prefs", DEFAULT_DISPLAY_PREFS),
        "sections": {
            "overview": today_overview.get("overview", {}),
            "work_and_study": {
                "calendar": calendar.get("events", []),
                "tasks": tasks.get("tasks", []),
                "economist_article": economist_block.get("article") if economist_block else None,
                "news": NEWS_ITEMS,
            },
            "health_and_exercise": {
                "workout": workout.get("workout"),
                "daily_100s": workout.get("daily_100s"),
                "health_goal": workout.get("health_goal"),
                "exercise_state": workout.get("exercise_state"),
            },
            "fun_and_interests": {
                "soccer": soccer.get("teams", []),
                "music": {
                    "releases": music.get("releases", []),
                    "concerts": music.get("concerts", []),
                },
                "events": events_block.get("events", []) if events_block else [],
            },
            "planning_ahead": {
                "flights": flights_block.get("deals", []) if flights_block else [],
                "scholarships": scholarships_block.get("scholarships", []) if scholarships_block else [],
            },
        },
        "usage_hint": (
            "This is a structured view. As the agent, consult display_prefs to decide "
            "how detailed to be when summarizing this into a morning briefing. "
            "Offer follow-up prompts like 'tell me more about soccer', 'expand music', "
            "or 'show only flights'."
        ),
    }

print("âœ… LifeBrief tools and synthetic data initialized.")


MODEL_NAME = "gemini-2.0-flash"

root_agent = Agent(
    name="life_brief_agent",
    model=Gemini(model=MODEL_NAME),
    description=(
        "An AI concierge that generates a personalized 'morning newspaper' "
        "for a busy professional based in Madrid and can help with planning, "
        "training, trips, scholarships, and groceries."
    ),
    instruction=(
        "You are LifeBrief, a calm, concise morning newspaper agent.\n\n"
        "GENERAL BEHAVIOR\n"
        "- When the user says things like 'good morning' or 'give me my brief', "
        "call the get_morning_life_brief tool first.\n"
        "- Then translate its structured data into a clear, easy-to-read summary.\n"
        "- Organize the answer into sections: Today at a glance, Work & study, "
        "Health & exercise, Fun & interests, and Planning ahead.\n"
        "- Call get_display_preferences before summarizing and respect:\n"
        "  â€¢ detail_level: 'minimal' -> ultra short bullets; "
        "    'normal' -> moderate detail; 'verbose' -> more detail per section.\n"
        "  â€¢ show_emojis: if False, avoid emojis; if True, you may use them sparingly.\n"
        "  â€¢ layout: if 'sections', use headings; if 'plain', use paragraph-style.\n"
        "- If the user asks to change how information is displayed, call "
        "update_display_preferences and confirm the change.\n\n"
        "EXERCISE & MOTIVATION\n"
        "- For workouts, use get_today_workout (which reads the injury state). "
        "If the user says their injury is better or worse, use update_exercise_state.\n"
        "- Always mention the Daily 100s challenge in a motivating but non-judgmental way.\n\n"
        "ECONOMIST & INDUSTRY CONTENT\n"
        "- Use get_economist_article_of_the_day to surface one synthetic Economist-style "
        "article related to international affairs, science/technology, Latin America, "
        "or energy, cycling across sections over days.\n"
        "- In Work & study, briefly show the article's title, short summary, and URL.\n"
        "- Do NOT claim you are logged into any personal Economist account or bypassing "
        "paywalls; you are making content recommendations only.\n\n"
        "TOOL USAGE\n"
        "- When the user asks for specifics (e.g., 'make a grocery list', 'show me "
        "flights', 'show scholarship deadlines'), call the most relevant tool and "
        "format the answer clearly.\n"
        "- Avoid overwhelming detail by default. Offer follow-up prompts like "
        "'Ask me to expand flights' or 'Explain the Economist article in simple terms'.\n\n"
        "TONE\n"
        "- Speak to the user as 'you'. Keep the tone supportive, practical, and "
        "slightly motivational, especially around training and study deadlines.\n"
    ),
    tools=[
        # Profile / overview
        get_user_profile,
        get_today_overview,
        # Memory
        get_display_preferences,
        update_display_preferences,
        get_exercise_state,
        update_exercise_state,
        # Work / study
        get_calendar_summary,
        get_task_summary,
        # Health / exercise
        get_today_workout,
        # Interests
        get_soccer_update,
        get_music_update,
        # Events / trips / scholarships
        get_local_events,
        get_flight_deals,
        get_scholarship_reminders,
        # Groceries
        generate_grocery_list,
        # Economist
        get_economist_article_of_the_day,
        # High-level brief
        get_morning_life_brief,
    ],
)

print("âœ… LifeBrief root_agent created.")



print("âœ… LifeBrief root_agent loaded by ADK.")


from life_brief_agent.agent import root_agent


runner = InMemoryRunner(agent=root_agent)
print("âœ… InMemoryRunner initialized.")


#You can play with the agent in the ADK Web UI if you start running the next cell, run this cell, and follow the link. 
#url_prefix = get_adk_proxy_url()



#!adk web --log_level DEBUG --url_prefix {url_prefix}


# Demo 1: Morning LifeBrief summary
async def morning_demo():
    await runner.run_debug(
        "Good morning! Please give me my Lifebrief morning summary. "
        "Follow my display preferences and include one Economist-style "
        "article suggestion and one scholarship reminder if available.",
        verbose=False,
    )

await morning_demo()



# Demo 2: Update display preferences
async def compact_mode_demo():
    await runner.run_debug(
        "From now on, keep my morning brief very compact, with minimal detail and no emojis.",
        verbose=False,
    )
    await runner.run_debug(
        "Now give me my morning brief again.",
        verbose=False,
    )

await compact_mode_demo()


# Demo 3: Generate my grocery list for Costco
async def grocery_demo():
    await runner.run_debug(
        "Make me a Costco grocery list for this week.",
        verbose=False,
    )

await grocery_demo()


