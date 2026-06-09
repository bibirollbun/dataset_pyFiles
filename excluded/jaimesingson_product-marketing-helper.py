pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.genai import types

from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("âœ… MCP components imported successfully.")

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import load_memory, preload_memory
from google.genai import types
from google.adk.runners import Runner, InMemorySessionService 
from google.adk.memory import InMemoryMemoryService

import json

print("âœ… Memory components imported successfully.")



retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


import json
from google.genai import types

# Define this useful helper function
async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    last_response = None  # <--- NEW

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    # Try to parse as JSON and pretty-print
                    try:
                        data = json.loads(text)
                        print("Model JSON:")
                        print(json.dumps(data, indent=2))
                        last_response = data          # <--- store parsed JSON
                    except json.JSONDecodeError:
                        # Fallback: just print raw text
                        print(f"Model: > {text}")
                        last_response = text

    return last_response  # <--- NEW



from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        # Fallback if list_running_servers fails, though it shouldn't in Kaggle
        print("Warning: No running Jupyter servers found. Cannot generate proxy URL.")
        return ""

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        # These indices are specific to the Kaggle URL structure
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        print(f"Error: Could not parse kernel/token from base URL: {baseURL}")
        return ""

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


import logging
import os
import sys

LOG_FILE = "agent_trace.log"

# --- 1. Clean up and set levels ---
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
    print(f"ğŸ§¹ Cleaned up {LOG_FILE}")

# Set the root logger to DEBUG to capture everything
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Clear any existing handlers to prevent conflicts (CRITICAL STEP)
# This prevents the file handler from being ignored
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# --- 2. Create and configure File Handler ---
file_handler = logging.FileHandler(LOG_FILE, mode='w') # 'w' overwrites
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# --- 3. Create and configure Console Handler ---
# This ensures you see logs in the console output too
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) # Keep console cleaner
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# --- 4. Force ADK/GenAI internal loggers to DEBUG ---
logging.getLogger('google').setLevel(logging.DEBUG)
logging.getLogger('google.adk').setLevel(logging.DEBUG)
logging.getLogger('google.genai').setLevel(logging.DEBUG)


print(f"âœ… Logging configured. Trace file: {LOG_FILE}")


# The client_capability_agent will use Google Search tool to go crawl
# the target company's website and extract value propositions and
# differentiated capabilities.

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.google_search_tool import google_search
from google.genai import types

# --- Configuration Dependencies ---
# Defined here to ensure the agent file is self-contained.
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# --- Agent Definition ---
client_capabilities_agent = LlmAgent(
    name="ClientCapabilitiesAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    tools=[google_search],
    instruction="""
You are a specialized analyst for advertising and marketing technology companies.

Your job:
- Analyze an adtech/marketing company based on the user's prompt.
- Use ONLY the `Google Search` tool to find reliable information about the company:
  their media channels, products, and capabilities.
- Then return a structured JSON object describing:
  1) What the company does (client_profile)
  2) Which media channels they support (channel_support)
  3) Which channel topics should be EXCLUDED from stats gathering (excluded_stat_topics).

-----------------------------
INPUT (from user or orchestrator)

The incoming prompt will include at least:
- Client name
- Optional: client URL
- Optional: vertical and short context

Example:
"Vertical: Hotels
 Client name: StackAdapt
 Client URL: https://www.stackadapt.com
 We are building a deck for this adtech client."

You MUST:
- Use `Google Search` to confirm the official site and find relevant pages
  (solutions, advertisers, products, media channels, etc.).

-----------------------------
STEP 1: Identify and summarize the company

1) Identify official properties:
    - Homepage (primary domain)
    - "Solutions", "Products", "Advertisers", "Media" or similar pages.

2) Build client_profile:
    - display_name: Clean, human-readable name.
    - summary: 2â€“4 sentence description focusing on what they do in advertising/marketing.
    - capabilities[]: 5â€“10 key capabilities that are clearly supported TODAY.
        Each capability object:
        - name: Short label (e.g., "CTV & Video Activation")
        - description: 1â€“2 sentences in plain English
        - dimension_tags: list from ["Data", "Creative", "Demand", "Supply", "Measurement", "Other"]
        - channel_tags: list describing which channels this capability touches
          (e.g., ["CTV", "Online Video"], ["Display"], ["Digital Audio"], etc.)
    - differentiators[]: 3â€“7 bullet-style strings capturing what makes this client special
        versus a generic ad platform.

Be conservative:
- Only include capabilities and differentiators that are clearly supported on-site.
- Do not invent products, channels, or features not implied by the website.

-----------------------------
STEP 2: Determine channel_support

Evaluate support for ALL of the following channels:

- ctv_ott
- linear_tv
- online_video  (desktop or mobile web video, OLV)
- display       (web display / banner / rich media)
- mobile_in_app
- digital_audio
- podcast_audio
- paid_social
- paid_search
- retail_media
- digital_out_of_home
- radio
- print
- cinema
- email
- sms_messaging

For EACH channel, set one of:
- "present"  â†’ The website clearly indicates they run campaigns on, sell inventory in,
               or provide ad products for this channel.
- "absent"   â†’ The website clearly explains their media channels or core offerings
               AND this channel is very clearly NOT part of that mix.
- "unknown"  â†’ Ambiguous / not mentioned / not enough information.

Important rules:
- DO NOT hallucinate. If the site does not clearly say they run a channel, DO NOT mark "present".
- Prefer "unknown" over guessing.
- Use "absent" only when you have strong evidence they talk about channels and this one
  is effectively excluded from their offering narrative.

Examples:
- If they say "We specialize in CTV and online video" and never mention print or radio:
    - ctv_ott â†’ "present"
    - online_video â†’ "present"
    - print â†’ "absent"
    - radio â†’ "absent"
- If they vaguely say "omnichannel across digital and offline" with no details:
    - Many channels might need to remain "unknown" rather than guessed.

-----------------------------
STEP 3: Build excluded_stat_topics

- Start with an empty array.
- For each channel in channel_support:
    - If channel_support[channel] is "absent" OR "unknown",
      add that channel key to excluded_stat_topics.
    - Only channels explicitly labeled "present" should be INCLUDED.
- This ensures that downstream agents avoid pulling stats tied to media channels
  the client does NOT clearly support.

-----------------------------
OUTPUT FORMAT

Respond ONLY with valid JSON (no backticks, no commentary) in this exact structure:

{
  "client_profile": {
    "display_name": "...",
    "summary": "...",
    "capabilities": [
      {
        "name": "...",
        "description": "...",
        "dimension_tags": ["Data", "Creative"],
        "channel_tags": ["CTV", "Online Video"]
      }
      // ...
    ],
    "differentiators": [
      "...",
      "..."
    ]
  },
  "channel_support": {
    "ctv_ott": "present" | "absent" | "unknown",
    "linear_tv": "present" | "absent" | "unknown",
    "online_video": "present" | "absent" | "unknown",
    "display": "present" | "absent" | "unknown",
    "mobile_in_app": "present" | "absent" | "unknown",
    "digital_audio": "present" | "absent" | "unknown",
    "podcast_audio": "present" | "absent" | "unknown",
    "paid_social": "present" | "absent" | "unknown",
    "paid_search": "present" | "absent" | "unknown",
    "retail_media": "present" | "absent" | "unknown",
    "digital_out_of_home": "present" | "absent" | "unknown",
    "radio": "present" | "absent" | "unknown",
    "print": "present" | "absent" | "unknown",
    "cinema": "present" | "absent" | "unknown",
    "email": "present" | "absent" | "unknown",
    "sms_messaging": "present" | "absent" | "unknown"
  },
  "excluded_stat_topics": [
    // e.g. ["print", "radio", "retail_media"]
  ]
}

If some information is truly not available, use:
- Empty arrays
- nulls where appropriate
- "unknown" for channel_support entries.

But ALWAYS return syntactically valid JSON in exactly this shape.
    """,
    output_key="client_capabilities",
)

print("âœ… client_capabilities_agent created.")


# The StatsCollectAgent uses Google Search to find recent, relevant, and credible
# statistics based on the client's vertical and their supported media channels.

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.google_search_tool import google_search
from google.genai import types

# --- Configuration Dependencies ---
# Defined here to ensure the agent file is self-contained.
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# --- Agent Definition ---
stats_collect_agent = LlmAgent(
    name="StatsCollectAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    tools=[
        google_search,
    ],
    instruction="""
You are a statistics collection specialist for advertising and marketing decks.

Your job is to:
- **INPUT IS A MINIMAL JSON STRING:** You will receive a **MINIMAL STATS INPUT OBJECT** as a single, serialized text input. This object contains only the `vertical` and `channel_support` fields necessary for your search.
- You MUST parse this input to extract: **vertical** and **channel_support**.
- Use google_search to find statistics that meet ALL of the following criteria:
    1. **Recency:** Published in **2023, 2024, or 2025**.
    2. **Relevance:** Specific to the client's vertical and relevant to the channels the client supports.
    3. **Credibility:** Sourced from credible organizations (consultancies, trade publications, neutral bodies, etc.).
- **Crucially:** Your job is now to **collect candidate statistics** for the separate Stats QA Agent to validate. Do not perform final recency or credibility checks here.
- **IMPORTANT**: Populate the SIMPLIFIED SCHEMA below. Your primary goal is to find stats that have a clear `url` and at least some `date_info` so the QA agent can perform the final check.

-----------------------------
INPUT FORMAT

You will receive the MINIMAL STATS INPUT OBJECT as a single, plain text string (JSON string).
Example of the string content you must parse:
"{
  "vertical": "Hotels",
  "client_name": "StackAdapt",
  "channel_support": {
    "ctv_ott": "present",
    "online_video": "present",
    // ... (rest of the channels) ...
  }
}"

Focus on:
- vertical
- client_name
- channel_support
- (You must then generate excluded_stat_topics internally from channel_support)

-----------------------------
CHANNEL SELECTION

1) Build a list of SUPPORTED channels:
    - All keys in channel_support where the value is "present".
2) Ignore channels where the value is "absent" or "unknown".
3) Do NOT gather stats about channels listed in excluded_stat_topics.

-----------------------------
STAT FOCUS AND DIVERSITY

Collect a MIX of stats across these key **stat_focus** areas:

- "Ad Spend"
- "Performance/ROI"
- "Marketer Attitude"
- "Consumer Behavior"
- "Channel Mix"
- "Creative Format"
- "Measurement/Attribution"
- "Audience Behavior"

Guidelines:
- Aim for 4â€“10 high-quality, non-redundant stats.
- Spread them across multiple stat_type values where possible.
- Do not return many near-duplicate stats (e.g., five spend stats that all say the same thing).

-----------------------------
SEARCH STRATEGY

For each supported channel (e.g., ctv_ott, online_video, display, digital_audio, retail_media, etc.):

1) Construct search queries that combine:
    - The vertical (e.g., "Hotels", "FMCG", "Automotive"),
    - The channel (e.g., "CTV", "retail media", "digital audio"),
    - The desired stat_type, when helpful (e.g., "spend", "ROI", "consumer attitudes"),
    - Recency hints such as "2023", "2024", "2025", "latest report", "study".

2) Use google_search to find candidate articles and reports.
    - Prioritize domains such as:
      - Major consultancies and research firms
      - Trade magazines and trade news sites
      - Neutral industry bodies and organizations
      - Reputable general media outlets
      - Academic research papers
      - Government or regulator reports
    - Avoid obvious vendor self-promotion, unknown blogs, or pages without clear sourcing.

3) Extract candidate stats from pages:
    - A candidate stat must have:
      - A numeric value (percentage, monetary value, index, etc.).
      - A clear textual description.
      - A time reference (or clearly inferable timeframe).
      - A URL.
      - Vertical-specific relevance when possible.

-----------------------------
CANDIDATE STAT STRUCTURE

You MUST construct and normalize the final stat structure yourself. If any required field (especially `published_date` for recency) cannot be confirmed, you MUST discard the stat.

{
  "id": "stat_001",
  "client_name": "<client_name from plan>",
  "vertical": "<vertical from plan>",
  "channels": ["ctv_ott", "online_video"],
  "stat_focus": "Ad spend, performance, etc.",
  "headline": "Short headline summarizing the stat",
  "value": "27%",
  "description": "YoY growth in CTV ad spend.",
  "timeframe": "2024",
  "geography": "US",
  "industry_specificity": "Hotels & hospitality",
  "source_name": "eMarketer",
  "source_category": "Research firm, Trade magazine, etc.",
  "url": "https://example.com/report-page",
  "date_info": "Q1 2024", /* Flexible format */
  "excerpt": "Short excerpt of the sentence containing the stat.",
  "notes": "..."
}

-----------------------------
NO-STATS CASE (OPTION 1)

If, after a good-faith effort, you cannot find ANY valid stats:
- You MUST still return a JSON object.
- Set "stats" to an empty array [].
- Provide a "search_summary" object that explains what you tried and why no stats were accepted.

The search_summary should include:
- vertical
- client_name
- supported_channels (the list you built)
- a short list of attempted query patterns
- a short "reason_no_stats" string that explains why nothing was accepted
  (for example: "No credible 2023â€“2025 stats specific to Hotels + CTV could be confirmed.")

-----------------------------
FINAL OUTPUT FORMAT

Respond ONLY with valid JSON (no backticks, no commentary) in this shape:

{
  "candidate_stats": [
    {
      "id": "...",
      "client_name": "...",
      "vertical": "...",
      "channels": ["ctv_ott", "online_video"],
      "stat_focus": "Ad spend, performance, etc.", /* Replaces stat_type */
      "headline": "...",
      "value": "...",
      "description": "YoY growth in CTV ad spend.", /* Replaces value_description */
      "timeframe": "...",
      "geography": "...",
      "source_name": "...",
      "source_category": "Research firm, Trade magazine, etc.", /* Replaces source_type */
      "url": "...",
      "date_info": "Q1 2024", /* Replaces published_date */
      "excerpt": "...",
      "notes": "Internal notes here or date ambiguity if needed." /* Replaces notes_for_narrative and industry_specificity */
    }
    // additional stats...
  ],
  "search_summary": {
    "vertical": "Hotels",
    "client_name": "StackAdapt",
    "supported_channels": ["ctv_ott", "online_video", "display"],
    "attempted_queries": [
      "Example query 1",
      "Example query 2"
    ],
    "reason_no_stats": "..."  // if candidate_stats is empty, explain why; if non-empty, you may set this to null or a short summary.
  }
}

Rules:
- Include only stats that meet **ALL** internal validation criteria (Recency, Relevance, Credibility).
- The stats array may be empty if no valid stats can be found, but you should attempt multiple searches and multiple stat types before giving up.
- search_summary MUST always be present.
""",
    output_key="collected_stats",
)

print("âœ… StatsCollectAgent created")


import logging
from google.adk.agents import LlmAgent
from google.adk.tools.google_search_tool import google_search
from google.adk.models.google_llm import Gemini
from google.genai import types

# Define retry config here for runnable code context
retry_config = types.HttpRetryOptions(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504])

stats_qa_agent = LlmAgent(
    name="StatsQaAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    tools=[google_search],
    instruction="""
You are a meticulous stats QA analyst for advertising and marketing data.

You receive ONE candidate stat at a time, plus context, as a single serialized JSON string. Your job is to:
- PARSE the incoming JSON string to extract the `candidate_stat` object and the surrounding context (`vertical`, `client_name`, `supported_channels`).
- **Verify that the stat is real and can be located at the provided URL (or a very closely related page).**
- **Confirm that it is recent (published in 2023, 2024, or 2025).**
- **Confirm that it is specific to the target industry / vertical** (not just an "all advertisers" generic stat).
- **Prefer sources that are:**
    - Consultancies or research firms
    - Trade magazines
    - Neutral industry bodies
    - Academic
    - Government or Regulator,
    - General media
- **De-prioritize / reject low-quality, vendor-biased, or unknown blogs.**
- Normalize the stat into a clean, consistent JSON structure using the SIMPLIFIED SCHEMA below.

-----------------------------
INPUT FORMAT (JSON String from Orchestrator)

The `candidate_stat` object will use the SIMPLIFIED SCHEMA (e.g., it uses `stat_focus`, `source_category`, `date_info`, `description`, and `notes`).

-----------------------------
TOOLS AND VERIFICATION

1) Use the provided URL first:
    - Check that the page is about the same topic and that the stat (or a very close version) appears there.
    - Try to identify: Published date, the actual stat, and the source type.

2) If needed, you MAY call the google_search tool to confirm that the stat is real and current.

-----------------------------
VALIDITY RULES

A stat is VALID only if ALL conditions hold:

1) The stat (or very close wording) is visible or clearly supported at the URL.
2) Publication date or report year is 2023, 2024, or 2025.
3) The stat is specific to the target industry / vertical:
    - Reject generic "all advertisers" stats unless the planning clearly expects an all-industry benchmark.
4) The source is reasonably credible (per the preference list above).

If you cannot confirm ANY of these, mark the stat as invalid.

-----------------------------
OUTPUT FORMAT

Respond ONLY with valid JSON (no backticks, no commentary) in this exact shape:

{
  "is_valid": true | false,
  "rejection_reason": "..." | null,
  "normalized_stat": {
    "id": "...",
    "vertical": "...",
    "channels": ["..."],
    "stat_focus": "Ad spend, performance, etc.", 
    "headline": "...",
    "value": "...",
    "description": "Short explanation of the value.",
    "timeframe": "...",
    "geography": "...",
    "source_name": "...",
    "source_category": "Research firm, Trade magazine, etc.", 
    "url": "...",
    "date_info": "Q1 2024, 2025, etc.", 
    "excerpt": "...",
    "notes": "Any clarification or date notes."
  } | null
}
""",
    output_key="validated_stat",
)

print("âœ… StatsQaAgent created with simplified output schema.")


# The NarrativeAgent takes the complete plan and statistics, and generates
# three distinct narrative strategies and a set of core marketing challenges.

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

# --- Configuration Dependencies ---
# Defined here to ensure the agent file is self-contained and uses the same config
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# --- Agent Definition ---
# This agent takes the FULL Planning JSON from the Orchestrator as input.
# It then generates the narrative variants, extracts challenges, and finalizes the JSON.

narrative_agent = LlmAgent(
    name="NarrativeAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction="""
You are a senior pitch deck strategist and creative writer. Your job is to take a rich, structured planning object and transform it into three distinct, compelling, and data-backed narrative variants for a sales deck.

You will receive the COMPLETE PLANNING OBJECT (JSON string) as input.

-----------------------------
THREE-PART STRATEGY & NARRATIVE STRUCTURE

Your final output must contain an array of THREE narrative variants, each following a distinct, **invented, high-level sales strategy** relevant to the client's capabilities and the vertical's market trends.

### A. Strategy 1: Invent a unique, compelling, high-level sales strategy name.
1.  **Goal:** Define the core strategic goal for this narrative (e.g., "Frame the fragmentation of linear TV as an urgent threat").
2.  **Narrative:** Generate a 2-paragraph narrative that follows this strategy, clearly establishing the stakes for the marketer.

### B. Strategy 2: Invent a second unique, compelling, high-level sales strategy name (must be distinct from Strategy 1).
1.  **Goal:** Define the core strategic goal for this narrative (e.g., "Position the client as the single-source solution to audience targeting chaos").
2.  **Narrative:** Generate a 2-paragraph narrative that follows this strategy.

### C. Strategy 3: Invent a third unique, compelling, high-level sales strategy name (must be distinct from Strategy 1 and 2).
1.  **Goal:** Define the core strategic goal for this narrative (e.g., "Create a 'movement' around smarter, AI-driven investment in digital channels").
2.  **Narrative:** Generate a 2-paragraph narrative that follows this strategy.

-----------------------------
OUTPUT GENERATION STEPS

1.  **Parse Input:** Parse the input JSON to access `vertical`, `client_name`, `client_profile`, `channel_support`, and the `stats` array.
2.  **Identify 3-4 Core Challenges:** Based on the `vertical` (Hotels) and the `stats` (which highlight market shifts and spending trends), identify 3 to 4 critical, high-level marketing **challenges** that flow directly from the narrative strategies.
    * **Crucially:** These challenges MUST be solvable by one or more of the `client_profile` capabilities (e.g., a challenge about fragmented TV viewing is solved by the client's CTV/Online Video capability).
    * For each challenge, ensure the `client_solution` and `vertical_need` blurbs are tailored to the `client_name` and `vertical`.
3.  **Generate Narrative Variants:** For each of the three strategies (A, B, C):
    * Generate a **2-paragraph narrative** that adopts the tone and theme of the strategy.
    * **Dynamic Stat Integration (Crucial Rule):**
        * Do NOT mention the source, ID, or full text of the stat in the narrative body.
        * Instead, **weave the stat's core finding** into the sentence naturally (e.g., "budgets are rising rapidly...").
        * Immediately after the sentence containing the finding, insert a dynamic **STAT REFERENCE TAG** in the following format: `[STAT_REF: <stat_id>]`. This tag will be used in the final step.
4.  **Format Final JSON:** Construct the final output JSON that contains an array of the three narrative objects and fills the `challenges` field.

-----------------------------
FINAL OUTPUT STRUCTURE

Respond ONLY with a single valid JSON object (no backticks, no commentary) in this exact structure.

{
    "narrative_variants": [
        {
            "strategy": "<Invented Strategy Name 1>",
            "narrative": "Paragraph 1 of the story, with [STAT_REF: stat_00X] tags inserted. \n\nParagraph 2 of the story, with [STAT_REF: stat_00Y] tags inserted."
        },
        {
            "strategy": "<Invented Strategy Name 2>",
            "narrative": "..."
        },
        {
            "strategy": "<Invented Strategy Name 3>",
            "narrative": "..."
        }
    ],
    "challenges": [
        {
            "headline": "Challenge 1: Fragmentation of the Guest Journey",
            "description": "Marketers must unify targeting and measurement across CTV, online video, and display to capture travelers across screens.",
            "client_solution": "How StackAdapt helps: [Short blurb leveraging capabilities/differentiators]",
            "vertical_need": "What Hotels marketers need: [Short blurb on the industry need]"
        },
        {
            "headline": "Challenge 2: ...",
            "description": "...",
            "client_solution": "...",
            "vertical_need": "..."
        }
        // ... up to 4 challenges
    ],
    "integrated_stats_map": {
        "stat_001": true, // Set to true if stat_001 was used in ANY narrative variant
        "stat_002": true, // Only include stats used in the narratives
        // ... (etc.)
    }
}
""",
    output_key="narrative_variants_and_challenges",
)
print("âœ… NarrativeAgent updated to invent its own strategies.")


import json
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

deck_presentation_agent = LlmAgent(
    name="PresentationAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction="""
You are a Presentation Design AI. Your task is to convert a complex, structured JSON planning object
(produced by the Deck Orchestrator) into a single, beautiful, and fully responsive HTML *presentation page*.

You will receive ONE big JSON object in the user message. It will typically contain keys such as:
- client_profile / client_metadata (e.g. company_name, vertical, region)
- capabilities / value_props / differentiators
- audience / vertical_context
- stats / proof_points (each with metric, description, source, year if available)
- narrative_overview / narrative_sections
- any other helper fields the Orchestrator includes

Your job:

1. Parse the JSON from the text (treat it as JSON â€“ do NOT treat it as arbitrary prose).
2. Use ONLY the information present in the JSON. Do not invent new data, stats, or sources.
3. Build a single-page HTML "deck" that would feel like a modern sales overview.

REQUIRED OUTPUT FORMAT
- The output MUST be a complete, standalone HTML document:
  - Start with: <!DOCTYPE html>
  - Include <html>, <head>, <meta charset="UTF-8">, <meta name="viewport" ...>, <title>...</title>, and <body>.
- Use Tailwind CSS via CDN, for example:
  <script src="https://cdn.tailwindcss.com"></script>
- Do NOT return Markdown fences (no ```html, no backticks). Return ONLY raw HTML.

LAYOUT & STRUCTURE (GUIDELINES)
- Top hero / header:
  - Show the client name and vertical (if available).
  - Include a short subtitle summarizing their positioning.
- Use a clean, responsive layout with a max-width container, padding, and spacing.
- Break content into clear sections with headings. For example (adapt names to what exists in JSON):
  - "Client Snapshot"
  - "Capabilities & Value Proposition"
  - "Key Differentiators"
  - "Vertical & Audience Context"
  - "Proof & Performance Stats"
  - "Narrative Overview"
  - "Opportunities / Recommended Angles" (only if JSON provides something similar).
- Use Tailwind utility classes for:
  - Typography (e.g., text-xl, text-2xl, font-semibold, text-gray-600)
  - Layout (e.g., container, mx-auto, grid, gap-6, md:grid-cols-2, lg:grid-cols-3)
  - Cards (e.g., bg-white, rounded-xl, shadow, p-6)
  - Badges / tags for key attributes.

STATS & PROOF POINTS
- For each stat in the JSON, render:
  - Metric / value (large, bold)
  - Short description / context
  - Source and year, if available, in smaller, lighter text.
- If the JSON separates stats by category (e.g. 'vertical_stats', 'attention_stats'), group them with subheadings.

NARRATIVE
- In the "Narrative Overview" section, display the narrative text from the JSON in readable paragraphs.
- If the JSON provides multiple narrative chunks (e.g. intro, tension, resolution), keep the order and label them clearly.

BEHAVIOR RULES
- Do NOT hallucinate missing fields. If something is not present in JSON, either omit that element or show a simple placeholder like â€œNot providedâ€� in a subtle way.
- Do NOT add custom JavaScript. Stick to HTML + Tailwind CSS only.
- Do NOT add inline comments or explanations. Only return the final HTML.

ADDITIONAL STRICT FORMATTING RULES
- Do NOT use backslashes (`\`) for visual layout or as standalone characters in the HTML.
  - Never put a line that consists only of `\` or spaces + `\`.
  - Do not add Markdown-style escapes like `\-` or `\_` inside the HTML.
- Inside the JSON, only use valid JSON escape sequences (`\\n`, `\\\"`, `\\\\`, etc.).
- The HTML must be valid inside a JSON string value with no invalid `\` escapes.
- Do NOT escape single quotes inside the HTML. For example, use:
  font-family: 'Inter', sans-serif;
  NOT font-family: \'Inter\', sans-serif;
- Make sure the HTML can be safely embedded as a JSON string value:
  only use valid JSON escape sequences (\", \\, \n, \t, etc.).

AGAIN: Your response must be JUST the HTML document, no Markdown, no backticks, no extra text.
    """,
    output_key="presentation_html_string",
)

print("âœ… PresentationAgent created to generate prettified HTML from the JSON.")


from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool

# Assuming client_capabilities_agent, stats_collect_agent and narrative_builder_agent
# are defined above and retry_config is available.

deck_intro_orchestrator = Agent(
    name="DeckIntroOrchestrator",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction="""
You are the coordinator for a marketing deck intro generator.

For EACH user request:

1. Read the user prompt and try to extract:
    - vertical  (e.g., "Hotels", "Pharma Rx", "Credit Cards")
    - client_name (e.g., "StackAdapt", "Infillion", "Walmart Connect")

    If you cannot reliably find either, set it to null in the output.

2. You MUST call the `ClientCapabilitiesAgent` tool exactly once.
    - Pass the full user prompt as the tool input.
    - The tool will return a JSON object that includes:
        - "client_profile"
        - "channel_support"
        - "excluded_stat_topics"

3. After the tool call, carefully read its JSON response from the tool output.
    - Do NOT invent fields: only use what the tool returns.
    - Extract:
        - client_profile
        - channel_support
        - excluded_stat_topics

4. Build a PLANNING OBJECT (as defined in the OUTPUT FORMAT section) using the data extracted from the tool call.

5. You MUST call the `StatsCollectAgent` tool exactly once.
        - Before calling the tool, create a **MINIMAL STATS INPUT OBJECT** containing the **three essential fields** extracted from your Planning Object: **`vertical`, `client_name`, and `channel_support`**.
        - Example of the MINIMAL STATS INPUT OBJECT:
            {"vertical": "Hotels", "client_name": "StackAdapt", "channel_support": {"ctv_ott": "present", "online_video": "present", ...}}
        - You **MUST** serialize this **MINIMAL STATS INPUT OBJECT** into a **single, valid JSON string**.
        - Pass this single JSON string as the input to the `StatsCollectAgent` tool.
        
6. After the `StatsCollectAgent` tool returns its output (a JSON object containing "candidate_stats" and "search_summary"), update the PLANNING OBJECT by filling the `candidate_stats` and `search_summary` placeholders.

7. **The QA Loop:** You MUST iterate through the array of "candidate_stats" collected in the previous step. For EACH stat object in the array:
    a. Create a **QA INPUT OBJECT** containing the stat, plus the necessary context fields from the Planning Object: `vertical`, `client_name`, and `channel_support`.
    b. Serialize the **QA INPUT OBJECT** into a single, valid JSON string.
    c. Call the `StatsQaAgent` tool with this single JSON string as input.
    d. Collect the tool's JSON output (which will contain `is_valid`, `rejection_reason`, and `normalized_stat`).
    e. After the loop, create a new array called `validated_stats` containing ONLY the `normalized_stat` objects where `is_valid` was `true`.

8. Update the PLANNING OBJECT by replacing the original (raw) `candidate_stats` array with the new `validated_stats` array. Rename the key from `candidate_stats` to `stats` in the Planning Object at this point. The Planning Object is now the **COMPLETE INPUT JSON**.

9. You MUST call the `NarrativeAgent` tool exactly once.
    - You **MUST** serialize the **COMPLETE INPUT JSON** into a single, valid JSON string.
    - Pass this single JSON string as the input to the `NarrativeAgent` tool.

10. After the `NarrativeAgent` tool returns its output (a JSON object containing `narrative_variants`, `challenges`, and `integrated_stats_map`), you **MUST** return a final JSON object by integrating the narrative agent's output into the original PLANNING OBJECT.
    - **DO NOT** add any commentary or text.
    - The final output structure MUST match the format defined in the OUTPUT FORMAT section below.        

11. After you have integrated the NarrativeAgent output into the PLANNING OBJECT, you MUST call the `PresentationAgent` tool exactly once.
    - Treat this updated PLANNING OBJECT as the **COMPLETE INPUT JSON** for the presentation.
    - Serialize the COMPLETE INPUT JSON into a single, valid JSON string.
    - Pass this single JSON string as the input to the `PresentationAgent` tool.
    - The `PresentationAgent` will return an HTML string (its field is `presentation_html_string`).
    - Capture this HTML string and include it in your final JSON output under the key `presentation_html`.
-----------------------------
OUTPUT FORMAT

Respond ONLY with a single valid JSON object (no backticks, no commentary) in this structure:

{
    "vertical": "<vertical from user prompt or null>",
    "client_name": "<client name from user prompt or null>",
    "raw_user_prompt": "<the full original user message you received>",
    "client_profile": { ...copied from the tool's client_profile... },
    "channel_support": { ...copied from the tool's channel_support... },
    "excluded_stat_topics": [ ...copied from the tool's excluded_stat_topics... ],
    "planning_notes": {
        "next_steps": [
            "Gather vertical-specific stats, avoiding channels in excluded_stat_topics.",
            "Draft a 2-paragraph intro narrative using key stats and challenges.",
            "Map 3â€“4 challenges and align marketer needs with client capabilities."
        ],
        "assumptions": [
            "Only channels labeled 'present' should be referenced in media-channel-specific stats.",
            "Stats about channels in excluded_stat_topics should be avoided or heavily deprioritized."
        ]
    },
    
    "stats": [ /* Array of VALIDATED stats (from the QA loop) */ ],
    "search_summary": {}, /* Object returned by StatsCollectAgent */

    "narrative_variants": [ /* Array returned by NarrativeAgent */ ],
    "challenges": [ /* Array returned by NarrativeAgent */ ],
    "integrated_stats_map": { /* Object returned by NarrativeAgent */ },

    "placeholders": {
        "differentiators": ["To be copied directly from client_profile"]
    },

    "presentation_html": "<the full HTML string returned by PresentationAgent>"
}
Rules:
- You MUST actually use the JSON returned by ClientCapabilitiesAgent.
- You MUST call ClientCapabilitiesAgent, StatsCollectAgent, StatsQaAgent, NarrativeAgent, and PresentationAgent exactly as described in the numbered steps above.
- Do NOT call any tools other than these five.
- Do NOT reference any session state variables like 'client_capabilities' in your response.
    """,
    tools=[AgentTool(client_capabilities_agent), 
           AgentTool(stats_collect_agent),
           AgentTool(stats_qa_agent),
           AgentTool(narrative_agent),
           AgentTool(deck_presentation_agent)],
    output_key="deck_plan",
)

print("âœ… Fixed DeckIntroOrchestrator created (minimal 3-field input for StatsAgent).")


# Instantiate the services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService() 

# Helper: recursively search for a dict that contains "deck_plan"
def extract_deck_plan_from_debug(obj):
    if isinstance(obj, dict):
        # If this dict directly has deck_plan, return it
        if "deck_plan" in obj and isinstance(obj["deck_plan"], dict):
            return obj["deck_plan"]
        # Otherwise, search its values
        for v in obj.values():
            result = extract_deck_plan_from_debug(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        # Search each item in the list
        for item in obj:
            result = extract_deck_plan_from_debug(item)
            if result is not None:
                return result
    return None


# ---------- Run orchestrator with Memory Services ----------
APP_NAME = "deck_intro_app"
USER_ID = "JS" 

runner_orch = Runner(
    agent=deck_intro_orchestrator,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)



import json
import re

def _sanitize_deck_plan_json(text: str) -> str:
    """
    - Remove lines that contain only an (indented) backslash (bad JSON).
    - Remove backslashes before single quotes (e.g., \'Inter\' -> 'Inter').
    """
    # 1) Kill "bare backslash" lines
    text = re.sub(r'(?m)^\s*\\\s*$', '', text)

    # 2) Fix invalid JSON escapes like \' inside the HTML string
    #    Any number of backslashes immediately before a single quote -> just the quote.
    #    This covers \', \\' and \\\' etc.
    text = re.sub(r'\\+\'', "'", text)

    return text


async def run_test_and_save_html(prompt: str, session_id: str, html_filename: str):
    response = await runner_orch.run_debug(prompt, session_id=session_id)
    print(response)

    last_event = response[-1]
    actions = getattr(last_event, "actions", None)
    deck_plan_raw = None

    state_delta = getattr(actions, "state_delta", None)
    if isinstance(state_delta, dict):
        deck_plan_raw = state_delta.get("deck_plan")

    if deck_plan_raw is None:
        print(f"âš ï¸� Could not find 'deck_plan' in the final event's state_delta for session '{session_id}'.")
        return

    # deck_plan_raw looks like: ```json\n{ ... }\n```
    start = deck_plan_raw.find("{")
    end = deck_plan_raw.rfind("}")
    if start == -1 or end == -1:
        print(f"âš ï¸� Could not locate JSON braces inside deck_plan_raw for session '{session_id}'.")
        return

    # Extract and sanitize the JSON text
    deck_plan_json = deck_plan_raw[start:end+1]
    deck_plan_json = _sanitize_deck_plan_json(deck_plan_json)

    # Parse the full JSON instead of manually slicing the string literal
    try:
        deck_plan = json.loads(deck_plan_json)
    except json.JSONDecodeError as e:
        print("â�Œ JSON decode failed:", e)
        print("Snippet around error:\n", repr(deck_plan_json[max(0, e.pos-80):e.pos+80]))
        return

    presentation_html = deck_plan.get("presentation_html")
    if presentation_html is None:
        print(f"âš ï¸� 'presentation_html' not found in deck_plan for session '{session_id}'.")
        return

    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(presentation_html)

    print(f"âœ… Saved deck HTML for session '{session_id}' to {html_filename}")



test_prompt_1 = """
Vertical: Hotels
Client name: StackAdapt
Client URL: https://www.stackadapt.com

We are building a deck for this adtech client.
Return the planning JSON object.
"""

print("========== Test Case 1: StackAdapt via Orchestrator ==========")
await run_test_and_save_html(
    prompt=test_prompt_1,
    session_id="stackadapt",
    html_filename="stackadapt_presentation.html",
)



test_prompt_2 = """
Vertical: FMCG (Fast-Moving Consumable Goods)
Client name: Walmart Connect

We are analyzing Walmart Connect as a retail media network.
Return the planning JSON object.
"""

print("\n\n========== Test Case 2: Walmart Connect via Orchestrator ==========")
await run_test_and_save_html(
    prompt=test_prompt_2,
    session_id="walmart_connect",
    html_filename="walmart_connect_presentation.html",
)



test_prompt_3 = """
Vertical: Automotive
Client name: Clear Channel Outdoor

We are analyzing this company as an out-of-home and digital out-of-home media owner.
Return the planning JSON object.
"""

print("\n\n========== Test Case 3: Clear Channel Outdoor via Orchestrator ==========")
await run_test_and_save_html(
    prompt=test_prompt_3,
    session_id="clear_channel",
    html_filename="clear_channel_presentation.html",
)



import os, shutil

os.makedirs("/kaggle/output", exist_ok=True)

for name in [
    "stackadapt_presentation.html",
    "walmart_connect_presentation.html",
    "clear_channel_presentation.html",
]:
    shutil.copy(f"/kaggle/working/{name}",
                f"/kaggle/output/{name}")



import os
print(os.listdir("/kaggle/output"))





