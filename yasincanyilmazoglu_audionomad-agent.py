# STEP 1: Environment Setup - run this once per session

# Remove some conflicting libraries quietly
!pip uninstall -y google-cloud-bigquery-storage google-cloud-translate bigframes > /dev/null 2>&1

# Install required packages
!pip install -U -q google-adk google-generativeai folium

print("âœ… Environment setup complete. Ready to proceed.")


# STEP 2: Configure API Key & Initialize Gemini model

import os
from kaggle_secrets import UserSecretsClient

from google.adk.models.google_llm import Gemini

# 1. Get API key from Kaggle secrets
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"  # Use direct Gemini API
    print("âœ… Google API Key successfully configured.")
except Exception as e:
    raise RuntimeError(
        "â�Œ GOOGLE_API_KEY not found. Go to 'Add-ons -> Secrets' and create one."
    ) from e

# 2. Initialize Gemini model (flash = fast + cheap, perfect for agents)
try:
    llm = Gemini(model="gemini-2.0-flash")
    print(f"âœ… Model initialized successfully: {llm.model}")
except Exception as e:
    raise RuntimeError(f"â�Œ Error initializing Gemini model: {e}")



# STEP 3: Define Custom Visualization Tools (folium map + YouTube player)

import folium
from IPython.display import display, HTML

def generate_interactive_map(location_name: str, coords, zoom_start: int = 12):
    """
    Renders an interactive map centered on the given coordinates using folium.
    """
    lat, lon = coords
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom_start,
        max_bounds=True,   # prevents infinite world tiling
    )
    folium.Marker(location=[lat, lon], popup=location_name).add_to(m)
    display(m)

def embed_smart_player(video_url: str, start_time: int = 0):
    """
    Embeds a YouTube player iframe for the given video URL.
    Handles NONE or non-YouTube URLs gracefully.
    """
    if not video_url or video_url.upper() == "NONE":
        print("âš ï¸� VIDEO_URL is NONE or empty. Skipping video player.")
        return

    if ("youtube.com" not in video_url) and ("youtu.be" not in video_url):
        print(f"âš ï¸� VIDEO_URL is not a YouTube link, skipping player: {video_url}")
        return

    vid_id = None

    if "youtu.be/" in video_url:
        vid_id = video_url.split("youtu.be/")[-1].split("?")[0]
    elif "watch?v=" in video_url:
        vid_id = video_url.split("watch?v=")[-1].split("&")[0]
    elif "embed/" in video_url:
        vid_id = video_url.split("embed/")[-1].split("?")[0]
    else:
        print(f"âš ï¸� Could not parse YouTube video ID from URL: {video_url}")
        return

    src = f"https://www.youtube.com/embed/{vid_id}?start={start_time}"

    html = f"""
    <iframe
        width="640"
        height="360"
        src="{src}"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen
    ></iframe>
    """
    display(HTML(html))

print("âœ… Visualization tools ready.")



# STEP 4: Define AudioNomad Agent & Runner

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.runners import InMemoryRunner
from google.genai import types

# System prompt: makes the agent behave like a Sonic Ethnomusicologist
audionomad_prompt = """
You are 'AudioNomad', an AI-powered Sonic Travel Guide.
YOUR GOAL: To reveal the "Sound Signature" of any location the user asks about.

USER INPUT: A city name or place.

==================== HOW TO ANSWER ====================

STRUCTURE YOUR ANSWER IN 3 SECTIONS:

SECTION 1 â€“ TITLE
- One short, catchy line like: "Izmir: Where the Aegean Sings with BaÄŸlama and Waves"

SECTION 2 â€“ SOUND SIGNATURE (FOCUS ON INSTRUMENTS)
- 2â€“4 sentences.
- Briefly mention ambient/atmospheric sounds (bazaar, sea, traffic, nature) BUT
- You MUST explicitly mention AT LEAST TWO local traditional instruments by name (e.g. "baÄŸlama", "zurna", "ney", "kanun", "oud").
- Describe how these instruments sound and how they blend with the city.

SECTION 3 â€“ MUSIC RECOMMENDATION
- 1â€“2 sentences.
- Describe the kind of performance the user is about to hear (e.g. "a live baÄŸlama taksimi from a local meyhane in Izmir" or "a ney solo echoing through a historic mosque courtyard").

==================== TOOL & VIDEO RULES ====================

You have access to the `google_search` tool.

To find the VIDEO_URL in the data block you MUST:

1. Call `google_search` with a query like:
   "YouTube [CITY NAME] traditional music [instrument name]"
   Example: "YouTube Izmir traditional music baÄŸlama"

2. From the results, choose a link that is a REAL YouTube video:
   - The URL MUST contain either "youtube.com/watch?v=" OR "youtu.be/".

3. Copy that URL EXACTLY into VIDEO_URL in the data block.

If you CANNOT find any YouTube video after using the tool, then:
- Set VIDEO_URL: NONE   (in ALL CAPS)

NEVER invent fake domains or non-YouTube links.
NEVER use shortened or ambiguous links that are not clearly YouTube.

==================== FINAL REQUIRED DATA BLOCK ====================

After your 3 sections above, you MUST end your response with this exact format:

DATA_BLOCK_START
LOCATION_NAME: [Name of the city/place in plain text]
LATITUDE: [Latitude number, e.g., 38.4237]
LONGITUDE: [Longitude number, e.g., 27.1428]
VIDEO_URL: [The YouTube link you found, or NONE]
DATA_BLOCK_END

RULES:
- ALWAYS output the DATA_BLOCK, even if coordinates are approximate.
- Do NOT add extra text after DATA_BLOCK_END.
- Do NOT change the field names (LOCATION_NAME, LATITUDE, LONGITUDE, VIDEO_URL).
"""

# 1) Create the agent
sonic_agent = Agent(
    name="AudioNomad",
    model=llm,
    instruction=audionomad_prompt,
    tools=[google_search],
)

# 2) Attach the agent to an in-memory runner
runner = InMemoryRunner(
    agent=sonic_agent,
    app_name="audionomad_capstone",
)

print("âœ… AudioNomad agent and runner initialized.")


# STEP 5: Orchestration â€“ end-to-end demo function

import re

async def run_audionomad_demo(user_query: str):
    """
    Runs a full AudioNomad interaction:
    1. Creates an ADK session
    2. Sends the user query
    3. Streams events and collects the final text
    4. Parses the DATA_BLOCK
    5. Renders a map + YouTube player
    """
    print(f"ğŸ‘¤ User: {user_query}\n")
    print("â�³ AudioNomad is researching coordinates & sounds...\n")

    # 1) Create a new session for this user
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="demo_user",
    )

    # 2) Wrap the user message into ADK Content
    new_message = types.Content(
        role="user",
        parts=[types.Part(text=user_query)],
    )

    # 3) Run the agent asynchronously and collect events
    events = []
    async for event in runner.run_async(
        user_id="demo_user",
        session_id=session.id,
        new_message=new_message,
    ):
        events.append(event)

    # 4) Build the final text from the response events
    full_text = ""
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    full_text += part.text

    # Optional: uncomment to debug raw model output
    # print("=== RAW MODEL OUTPUT ===")
    # print(full_text)

    # ==== 4A: Show only the descriptive part to the user ====
    print("\nğŸ¤– AudioNomad Description:\n")
    if "DATA_BLOCK_START" in full_text:
        print(full_text.split("DATA_BLOCK_START")[0].strip())
    else:
        print(full_text.strip())

    print("\n" + "=" * 40)
    print("ğŸ—ºï¸� DYNAMIC VISUALIZATION ENGINE")
    print("=" * 40)

    # ==== 4B: Parse the DATA_BLOCK ====
    try:
        loc_name_match = re.search(r"LOCATION_NAME:\s*(.*)", full_text)
        lat_match      = re.search(r"LATITUDE:\s*([+-]?\d+(\.\d+)?)", full_text)
        lon_match      = re.search(r"LONGITUDE:\s*([+-]?\d+(\.\d+)?)", full_text)
        video_match    = re.search(r"VIDEO_URL:\s*(.*)", full_text)

        if not (loc_name_match and lat_match and lon_match and video_match):
            raise AttributeError("Data block missing one or more fields.")

        location_name = loc_name_match.group(1).strip()
        lat = float(lat_match.group(1))
        lon = float(lon_match.group(1))
        video_url = video_match.group(1).strip()

        print(f"\nğŸ“� Extracted Data: {location_name} ({lat}, {lon})")

        # 5) Render map & video player
        generate_interactive_map(location_name, [lat, lon], zoom_start=13)

        print(f"\nâ�¯ï¸� Found Video: {video_url}")
        embed_smart_player(video_url, start_time=10)

        print("\nâœ… Fully Dynamic Generation Successful!")

    except AttributeError:
        print("âš ï¸� Data block not found or incomplete in agent response. Showing fallback map.")
        generate_interactive_map("Location Not Parsed", [20, 0], zoom_start=2)


# Try different cities: "Izmir", "Athens", "Kyoto", "New York", ...
await run_audionomad_demo("Istanbul, Turkey")


