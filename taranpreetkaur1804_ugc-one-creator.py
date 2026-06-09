import os
import json
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from google.adk.runners import Runner


from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)  

print("âœ… ADK components imported successfully.")

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

print("Retry options added successfully.")


# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


# Input Section
input_text = {
  "video_contexts":"Me packing a tiny backpack for a weekend in Coorg",
  "angle":"Travel Aâ€“Z",
  "length_seconds":30,
  "tone":"chill",
  "language":"English",
  "voiceover_lang":"English",
  "hashtags_count":3,
  "cta_type":"follow and save",
  "format":"single"
}
# video_context (string, required) â€” one-sentence description of on-screen action.
#angle (enum, required) â€” UGC|Travel Aâ€“Z|Negative hook|Tutorial|POV|Storytime|Product demo
#length_seconds (int, optional, default=30) â€” 5â€“120 (cap)
#tone (string, optional, default="authentic energetic")
#language (enum, default=English) â€” English|Hindi|Bilingual
#voiceover_lang (enum, default=same as language) â€” English|Hindi
#hashtags_count (int, optional, default=8) â€” 4â€“15
#cta_type (enum, default=follow) â€” follow|save|comment|visit_link|watch_next
#format (enum, default=single) â€” single|multi-cut|carousel


# Context Reader Agent: Run first. Parse the raw user input (video_context + form fields).
# Normalize, validate, infer missing defaults, and produce an internal creative plan used by later agents.
context_agent = Agent(
    name="ContextReader",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You will receive the user's input as a JSON string (user message).
    Parse it safely.

    Expected keys:
    - video_context
    - angle
    - length_seconds
    - tone
    - language
    - voiceover_lang
    - hashtags_count
    - cta_type
    - format
    You are the Context Reader. 
    Read inputs, validate, normalize, and produce a concise creative brief (3â€“5 bullet points) with tone and constraints. 
    Never ask the user clarifying questions â€” instead, pick sensible defaults and mark what you assumed.
    If video_contexts is missing â†’ set video_context = "short-form video about angle" and mark assumed.
    If length_seconds <5 â†’ set to 5; >120 â†’ set to 120 and mark truncation.
    Ensure hashtags_count in 4â€“15; clamp otherwise.
    If voiceover_lang is not supplied, default to English.
    """,
    output_key="context_outline",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… context_agent created.")

# Hook Generator Agent: Produce 2 strong hook options (A/B). Each hook must be 3â€“8 words, match angle and tone, and be optimized for stopping scroll in first 1â€“3 seconds.
hook_agent = Agent(
    name="HookGeneratorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are the Hook Generator. Use the context_outline to craft exactly two hooks. 
    Keep them short, curiosity-driven or emotionally charged. 
    For travel: use sensory words. For UGC: prioritize honest/relatable phrasing. 
    Avoid clickbait or unsafe content. 
    Enforce 3 â‰¤ word_count â‰¤ 8. If the hook is longer, shorten it.
    No personal data allowed.
    Do not repeat the same hook twice.
    """,
    output_key="hook_draft",  # The result of this agent will be stored with this key.
)

print("âœ… hook_agent created.")



# Script Composer Agent: Produce two script variations (SCRIPT_A time-coded; SCRIPT_B short alternative). 
# Each script uses the chosen hook (default to Hook A) and aligns visuals to voice lines. 
# Ensure overall timing matches length_seconds.
script_agent = Agent(
    name="ScriptComposerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are the Script Composer. Use context_outline + hook_draft to output:
    SCRIPT_A: time-coded segments that sum to length_seconds (approximate).
    SCRIPT_B: a concise alternative (3â€“5 lines) with a different angle or pacing.
    Follow cadence: 5â€“8 words per ~2â€“3 seconds. Insert clear Visual: cues and Voice: lines for each segment. Keep sentences speakable.
    Sum of segment durations â‰ˆ length_seconds (Â±2s).
    Each Voice: line max 12 words.
    Total script must not exceed max token/word limits.
    If format=multi-cut, provide cut markers (CUT 1 / CUT 2) instead of continuous timing.
    """,
    output_key="scripts",  # This is the final output of the entire pipeline.
)

print("script_agent created.")

# Caption Agent: Create two caption options: SHORT (1â€“2 lines) and LONG (3â€“5 lines). 
# LONG caption must end with a CTA line matching cta_type.
caption_agent = Agent(
    name="CaptionCuratorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are the Caption Writer. 
    Convert the context_outline + scripts.A into IG/TikTok-style captions. 
    SHORT should be punchy; LONG should tell a micro-story and finish with the CTA. 
    Add optional 1â€“2 suggested emoji placements.
    SHORT: â‰¤ 2 lines, total â‰¤ 150 characters (recommended).
    LONG: 3â€“5 lines, â‰¤ 450 characters.
    Include CTA verbatim in the LONG caption end.
    """,
    output_key="captions",  # The result of this agent will be stored in the session state with this key.
)

print("caption_agent created")

# CTA Agent: Given cta_type, create primary CTA text and micro on-screen micro-copy (sticker/button text) 
# and a one-line reason why it works (for user confidence).
cta_agent = Agent(
    name="CTAAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are the CTA Specialist. 
    Output concise CTA messaging aligned to platform behavior and user goal. 
    Provide 2 micro-text options (shorter) for on-screen overlays.
    On-screen micro text â‰¤ 5 words.
    Avoid ambiguous CTAs; tie to content (e.g., â€œSave this packing listâ€� not just â€œSaveâ€�).
    """,
    output_key="CTA",  # The result of this agent will be stored in the session state with this key.
)
print("cta_agent created")

# Hashtag Curator Agent: Produce exactly hashtags_count hashtags:
# a balanced mix of niche + broad + 1 trending tag candidate + 1 branded tag placeholder (if provided).
hashtag_agent = Agent(
    name="HashtagAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are the Hashtag Curator. 
    Use context_outline to suggest relevant hashtag tokens. 
    Keep tags under 30 chars and no spaces. 
    Avoid spammy #viral unless it naturally fits.
    Output length must equal hashtags_count.
    All tags lowercase, no spaces, no special characters except underscore.
    """,
    output_key="hashtags",  # The result of this agent will be stored in the session state with this key.
)

print("hashtag_agent created")

# Thumbnail Text Agent: Produce 3 thumbnail text options (3â€“5 words each) 
# that maximize curiosity and fit overlaid image constraints. 
# Also provide suggested text emphasis (which word to bold/uppercase).
thumbnail_agent = Agent(
    name="ThumbnailAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are the Thumbnail Text Designer. 
    Create short, high-contrast-friendly lines based on context_outline. 
    Avoid punctuation-heavy lines. 
    For travel, emphasize sensory or destination. 
    Provide a color-contrast suggestion tag (dark_bg or light_bg) based on the likely thumbnail.
    Each option has 3â€“5 words.
    No emoji.
    Provide emphasis_index (which word to emphasize).
    """,
    output_key="thumbnail_text",  # The result of this agent will be stored in the session state with this key.
)

print("thumbnail_agent created")

# Voiceover Author Agent: Produce conversational voiceover scripts in requested language (English/Hindi/Bilingual). 
# Include [pause] markers every 8â€“12 words. Match cadence to length_seconds. 
# Output both a spoken VO and a separate on-screen subtitle-ready line-by-line transcript if requested.
voiceover_agent = Agent(
    name="VoiceOverAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are the Voiceover Author. 
    Use the scripts + hook_draft to craft a natural spoken script. 
    Insert [pause] markers. 
    Keep syllable flow friendly to read aloud. 
    If bilingual, indicate which lines are Hindi vs English and provide transliteration where necessary.
    [pause] every 8â€“12 words. If a long sentence exceeds 12 words, split or add a pause.
    Total spoken words should match timing: approx 5â€“8 words per 2â€“3 seconds.
    No rhyming or tongue twisters unless the user requests them.
    Provide both voiceover_script and subtitle_transcript so the editor can paste subtitles.
    """,
    output_key="voiceover_script",  # The result of this agent will be stored in the session state with this key.
)
print("voiceover_agent created")



aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
You are the Aggregator. You receive outputs from multiple upstream agents (ContextReader, HookGenerator,
ScriptComposer, CaptionWriter, CTASpecialist, HashtagCurator, ThumbnailDesigner, VoiceoverAuthor).
Your job is to merge these into a single well-structured JSON final payload for the user.

EXPECTED INPUT (as JSON or parsed object):
{
  "context": <creative_brief or normalized_inputs from ContextReader>,
  "hooks": <hook_agent output: list of hooks>,
  "script": <script_agent output: {script_a, script_b}>,
  "captions": <caption_agent output>,
  "cta": <cta_agent output>,
  "hashtags": <hashtag_agent output>,
  "thumbnails": <thumbnail_agent output>,
  "voiceover": <voiceover_agent output>,
  "assumptions": <list>  # optional
}

AGGREGATION RULES:
- Produce a top-level object with these keys exactly:
  - hook (choose primary hook text; prefer hooks[0] if present)
  - hooks (full hooks array)
  - script_a (structured list of timed segments)
  - script_b (alt text)
  - captions { short, long, emoji_suggestions? }
  - cta { cta_text, on_screen_options, rationale? }
  - hashtags (array)
  - thumbnail_options (array)
  - voiceover { language, voiceover_script, subtitle_transcript }
  - assumptions (accumulated assumptions from earlier agents)
  - partial_failures (map of agent_name -> {error:..., fallback_used:...}) â€” include if any child reported errors
  - metadata { generated_at: ISO8601, pipeline_version: "v1" }

VALIDATION HOOK:
- Do not perform deep validation here; only ensure fields exist and are well-typed (e.g., hashtags is list).
- If a downstream child returned {"ok": False, "error": "..."} treat it as a partial failure:
  - include the error in partial_failures
  - attempt a safe fallback (see FALLBACKS) and mark fallback_used=true

FALLBACKS (simple rule-based):
- Missing hook -> derive from `context.video_context` by returning first 6 words + "..." (mark fallback).
- Missing hashtags -> generate 6 simple tags by lowercasing nouns from `context.video_context` + ["savethis","shorts"].
- Missing captions -> make SHORT = first script voice line; LONG = short story using script_a summary + CTA from cta input.
- Missing voiceover -> convert script_a voice lines into voiceover with `[pause]` inserted every 10 words.

OUTPUT:
Return ONLY valid JSON (no extra commentary) under the output key `aggregated_output`.
Include all fields per AGGREGATION RULES. If any fallback used, set partial_failures accordingly.

If inputs are not parsable JSON, respond with:
{ "error": "invalid_input", "message": "Could not parse upstream inputs", "raw_input": "<first 200 chars>" }
""",
    output_key="aggregated_output",
)

print("âœ… aggregator_agent created.")

validator_agent = Agent(
    name="ValidatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
You are the Validator for the UGC content pipeline. You receive the aggregated_output produced by the AggregatorAgent
and the original normalized_inputs from ContextReader. Run a ruleset, attempt auto-fixes where safe, and return a
validation report and (optionally) an auto-fixed aggregated_output.

INPUT:
{
  "aggregated_output": { ... },   # full aggregated JSON
  "normalized_inputs": { ... }    # includes length_seconds, hashtags_count, cta_type, etc.
}

VALIDATION RULES (must check each and return boolean + details):
1. hook_word_count_ok: hook must be between 3 and 8 words.
   - If false, auto-fix by truncating to first 8 words or rewriting to preserve meaning (prefer truncation).
2. hashtag_count_ok: number of hashtags must equal normalized_inputs.hashtags_count.
   - If fewer, append simple rule-based hashtags until count matches (use nouns from video_context and "savethis").
   - If more, keep top N (first in list) and mark removed.
3. script_timing_ok: sum of script_a segment durations must be within Â±2s of normalized_inputs.length_seconds.
   - If off, rebalance the last segment duration to meet length_seconds.
4. caption_cta_ok: long caption must end with CTA text matching cta_type (e.g., "Save this for later" for save).
   - If missing, append CTA sentence to long caption.
5. voiceover_pause_ok: voiceover_script must contain [pause] markers approx every 8â€“12 words.
   - If missing or sparse, insert [pause] every 10 words (do not change semantic content).
6. thumbnail_word_count_ok: each thumbnail option 3â€“5 words.
   - If violation, auto-truncate to first 5 words.
7. overall_schema_ok: aggregated_output must contain all required top-level keys (hook, hooks, script_a, captions, cta, hashtags, thumbnail_options, voiceover).
   - If keys missing, mark as errors and attempt to reconstruct from available data.

AUTOFIX POLICY:
- Perform autorepairs for the specific rules above. Each auto-fix must be recorded in `auto_fixes` with:
  { "rule": "<rule_name>", "fix_description": "...", "previous_value": <short snapshot>, "new_value": <short snapshot> }.
- Do not invent brand claims or factual assertions. If a field cannot be safely auto-fixed, do NOT fabricate; add to `errors` list and leave original data unchanged.

OUTPUT SCHEMA (return JSON ONLY under output_key `validation`):
{
  "valid": <bool>,
  "errors": [ { "field":"", "message":"" }, ... ],
  "auto_fixes": [ { "rule":"", "fix_description":"", "previous_value":"", "new_value":"" }, ... ],
  "final_output": { ... }  # the possibly auto-fixed aggregated_output
}

ADDITIONAL:
- If any fix was applied, set valid=false if there remain unfixable errors; otherwise set valid=true.
- Prefer minimal, conservative fixes. Document everything in auto_fixes.
- If input parsing fails, return:
  { "valid": false, "errors":[{"field":"input","message":"could not parse input"}], "auto_fixes":[], "final_output":null }

Return ONLY the JSON validation object (no commentary).
""",
    output_key="validation",
)

print("âœ… validator_agent created.")




# Create a ParallelAgent for fan-out
parallel_downstream = ParallelAgent(
    name="ParallelPostScript",
    sub_agents=[
        caption_agent,
        cta_agent,
        hashtag_agent,
        thumbnail_agent,
        voiceover_agent
    ],
)

# Finally, the sequential root pipeline
root_agent = SequentialAgent(
    name="UGC_Pipeline_Root",
    sub_agents=[
        context_agent,
        hook_agent,
        script_agent,
        parallel_downstream,   # runs caption/cta/hashtags/thumbnail/voiceover in parallel
        aggregator_agent,
        validator_agent
    ],
)


# Run root_agent (SequentialAgent)
input_text = json.dumps(input_text)

APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session
MODEL_NAME = "gemini-2.5-flash-lite"

# Step 2: Set up Session Management
# InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()

# Step 3: Create the Runner
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service,plugins=[LoggingPlugin()])

print("âœ… Stateful agent initialized!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Using: {session_service.__class__.__name__}")

await run_session(
    runner,input_text,
    "stateful-agentic-session",
)

# runner = InMemoryRunner(agent=root_agent)
# response =  await runner.run_debug(input_text, verbose=True)

