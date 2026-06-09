import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_path = "/kaggle/input/dataflow/soundspark1.png"

img = mpimg.imread(img_path)
h, w = img.shape[:2]

plt.figure(figsize=(w/250, h/250), dpi=100)
plt.imshow(img)
plt.axis("off")
plt.show()


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


from google.genai import types
from google.adk.runners import Runner
from google.adk.tools import google_search, load_memory
from google.adk.models.google_llm import Gemini
from google.adk.memory import InMemoryMemoryService      # using InMemory Memory Service
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.sessions import DatabaseSessionService    # Database based persistent session
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.agents import SequentialAgent, LlmAgent, Agent
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

import re
import os
import json
import uuid
import time
import httpx
import asyncio
import librosa
import sqlite3
import logging
import warnings
import numpy as np
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from scipy.signal import butter, lfilter
from typing import Dict, Any, Optional, Callable, Literal, Annotated, List

warnings.filterwarnings("ignore")

print("[IMPORT]: âœ”ï¸� All components imported successfully!")


from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()

    os.environ["FREESOUND_API_KEY"] = user_secrets.get_secret("FREESOUND_API_KEY")  # to use MCP 
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")        
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")                    # To use huggingface space
    os.environ["GOOGLE_CLOUD_PROJECT"] = user_secrets.get_secret("PROJECT_ID")
    print("[ENV LOADING]: âœ”ï¸� All secrets loaded!")
except Exception as e:
    print("[ENV LOADING]: â�Œ Error loading the secrets!")


# Generate synthetic audio samples for demo

import numpy as np, soundfile as sf, os
OUTDIR = "tests/sample_audio"
os.makedirs(OUTDIR, exist_ok=True)

sr = 22050
duration = 2.0  # seconds

def write_wav(arr, path):
    sf.write(path, arr, sr)
    print("Wrote", path)

# 1. clean sub sine (sub_bass.wav)
t = np.linspace(0, duration, int(sr*duration), endpoint=False)
sub = 0.5 * np.sin(2*np.pi*55*t)  # 55Hz sub
write_wav(sub, os.path.join(OUTDIR, "sub_bass.wav"))

# 2. gritty "reese-ish" bass (detune two saws + bit crush-ish)
f1 = 100
saw1 = 0.25 * (2*(t*f1 - np.floor(0.5 + t*f1)))
saw2 = 0.25 * (2*(t*(f1*1.01) - np.floor(0.5 + t*(f1*1.01))))
gritty = saw1 + saw2
# simple soft clip
gritty = np.tanh(gritty * 3.0)
write_wav(gritty, os.path.join(OUTDIR, "gritty_bass.wav"))

# 3. warm pad (filtered noise + slow envelope)
noise = np.random.normal(0, 0.2, size=t.shape)
env = np.linspace(0,1,t.size)**0.6
pad = np.convolve(noise*env, np.ones(500)/500, mode='same')
write_wav(pad, os.path.join(OUTDIR, "warm_pad.wav"))

# 4. pluck (short percussive pluck)
pluck = np.sin(2*np.pi*440*t) * np.exp(-6*t)
write_wav(pluck, os.path.join(OUTDIR, "pluck.wav"))

# 5. click/hit (percussive transient)
hit = np.zeros_like(t)
hit[0:200] = np.linspace(1,0,200)
write_wav(hit, os.path.join(OUTDIR, "hit.wav"))

# 6. vocal-chop-like (granular short noisy bursts)
vc = np.zeros_like(t)
for i in range(6):
    start = int(i * sr * 0.3)
    end = start + 300
    if end < len(vc):
        vc[start:end] += np.random.normal(0, 0.6, size=(end-start))*np.hanning(end-start)
write_wav(vc, os.path.join(OUTDIR, "vocal_chop.wav"))

print("Sample files:", os.listdir(OUTDIR))



print("[HELPER]: â„¹ï¸� Setting Up Helpers!!")

# 1. To deal with errors like rate limited, service hindrances or unavailability 
retry_config = types.HttpRetryOptions(
    attempts=5,  
    exp_base=7,  
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

print("[HELPER]: âœ”ï¸� Retry Config!")

# ---------------------------------------------------------------


# 2. Simple utility to JSONize the JSON String like response from LLMs
def give_json(res: str):
    """
    A utility to structure the text false JSON string from LLMs to JSON String
    This eliminates any texts or words out side of the JSON string parantheses 

    args:
        res : string, LLM's faulty JSON response
    return
        JSON Strnig Dict in Python 
    """
    # Use a regex to extract the content between the first { and the last }
    json_match = re.search(r'\{.*\}', res, re.DOTALL)
    
    if json_match:
        # If a JSON-like object is found, use that for parsing
        json_string = json_match.group(0)
    else:
        # Fallback to the raw string if no JSON is found
        json_string = res
        
    try:
        response = json.loads(json_string)
        return response
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from LLM: {e}")
        print("Raw LLM response:")
        print(repr(res)) # Using repr() shows hidden characters like newlines
        return None

print("[HELPER]: âœ”ï¸� Give JSON!")


# --------------------------------------------------------------------


async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
    USER_ID: str = "",
):
    print(f"\n ### Session: {session_name}")
    app_name = runner_instance.app_name
    session_service = runner_instance.session_service

    # create or get session
    try:
        session = await session_service.create_session(app_name=app_name, user_id=USER_ID, session_id=session_name)
    except Exception:
        session = await session_service.get_session(app_name=app_name, user_id=USER_ID, session_id=session_name)

    if not user_queries:
        print("No queries!")
        return ""

    if isinstance(user_queries, str):
        user_queries = [user_queries]

    all_results = []

    for query_text in user_queries:
        print(f"\nUser > {query_text}")
        query_content = types.Content(role="user", parts=[types.Part(text=query_text)])

        # We'll accumulate the full response text here
        accumulated = ""
        # Track what we've already printed so we can print only the new suffix
        last_printed_len = 0

        # Stream events
        async for event in runner_instance.run_async(user_id=USER_ID, session_id=session.id, new_message=query_content):
            # Build text from text parts only (ignore function_call / non-text parts)
            event_text = ""
            # print(event)
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                for p in event.content.parts:
                    # Accept only actual text parts
                    txt = getattr(p, "text", None)
                    if txt and txt != "None":
                        event_text += txt

            # If we got new text, append and print only the new suffix
            if event_text:
                # Append the event_text to accumulated. Sometimes SDK returns
                # overlapping or concatenated text; accumulate then print the
                # suffix that hasn't been printed yet.
                accumulated += event_text

                # Compute the unprinted suffix
                if len(accumulated) > last_printed_len:
                    to_print = accumulated[last_printed_len:]
                    # print the incremental new text once
                    print("\n")
                    print("assistant >", to_print)
                    # update last_printed_len so we don't reprint same text
                    last_printed_len = len(accumulated)

            # Many SDKs provide event.type or event.status to indicate completion.
            # If present and indicates completion, break. Otherwise keep streaming until generator ends.
            # Safe checks â€” won't error if attributes aren't present.
            event_type = getattr(event, "type", None)
            event_status = getattr(event, "status", None)
            # Common names that might indicate completion: "response.completed", "completed", "finished"
            if event_type in ("response.completed", "completed", "finished") or event_status in ("completed", "finished", "done"):
                break

        # append full accumulated response for this query
        all_results.append(accumulated)

    return "\n\n".join(all_results)



# --------------------------------------------------------------------


# 4. Routing Utility, to choose the correct agent to handle user's query
import re

def route_prompt(prompt: str) -> dict:
    """
    Strict-ish router that accepts path quoted or unquoted anywhere after /design.
    Returns:
      {"output": 1, "path": "<raw_path>"}  # when /design + path found
      {"output": 2, "path": "/vocal"}       # when /vocal
      {"output": 3, "path": "no"}           # otherwise
    """

    if not isinstance(prompt, str):
        prompt = str(prompt or "")

    p = prompt.strip()
    p_lower = p.lower()

    # /vocal
    if p_lower.startswith("/vocal"):
        return {"output": 2, "path": "/vocal"}

    # Only consider /design mode for path extraction
    if p_lower.startswith("/design"):
        # pattern: path: "..." OR '...' OR unquoted-token (no spaces)
        # allow optional whitespace around ':'
        pattern = r'path\s*:\s*(?:"([^"]+)"|\'([^\']+)\'|([^\s"\'`]+))'
        m = re.search(pattern, p, flags=re.IGNORECASE)
        if m:
            # group1 => double-quoted, group2 => single-quoted, group3 => unquoted
            raw = m.group(1) or m.group(2) or m.group(3)
            # normalize Windows backslashes (optional) â€” keep as-is for downstream
            return {"output": 1, "path": raw}
        else:
            # /design present but no path token detected
            return {"output": 1, "path": "no"}

    # default chat
    return {"output": 3, "path": "no"}



print("[HELPER]: âœ”ï¸� Router!")


print("[SCHEMA]: â„¹ï¸� Loading Agent Output Schema Design!")

# This schema is used for Audio Classifier Agent, the only agent that uses no tool
class ClassificationOutput(BaseModel):
    style_tags: Annotated[
        List[str],
        Field(min_length=1, max_length=4)
    ] = Field(description="Up to 4 descriptive style tags.")

    genre_suggestions: Annotated[
        List[str],
        Field(min_length=1, max_length=3)
    ] = Field(description="Up to 3 genre suggestions.")

    texture: Literal[
        "gritty", "warm", "bright", "dark",
        "percussive", "smooth", "wide"
    ] = Field(description="Texture word.")

    confidence: Annotated[
        float,
        Field(ge=0.0, le=1.0)
    ] = Field(description="Confidence 0â€“1 score.")


print("[SCHEMA]: âœ”ï¸� Classification Schema Loaded!")


# 1. this tool is for feature_agent to extract basic audio descriptors
def _to_scalar(x):
    """Convert numpy arrays / numpy scalars / iterables to Python native floats/ints when possible."""
    if x is None:
        return None
    # if numpy array or list/tuple -> try to pick a representative scalar
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(x)
            if arr.size == 0:
                return None
            # prefer single-element value if present, else mean
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
            return float(arr.mean())
        except Exception:
            try:
                return float(x[0])
            except Exception:
                return None
    if isinstance(x, np.generic):
        return x.item()
    try:
        return float(x)
    except Exception:
        return x

def compute_basic_descriptors(path: str, sr: int = 22050) -> Dict[str, Any]:
    """
    Compute lightweight descriptors for an audio file and return JSON-safe python types.
    Tempo is guaranteed to be either a float or None.

    ARGS:
        path: file path given in prompt by the user
        sr: sample rate, can be chosen for high quality sounds
    RETURN:
        Dictionary containing all the descriptors of audio
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    duration = float(len(y) / sr)

    # tempo (may be scalar or array-like)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # print(f"[feature extractor] : The tempo = {tempo}")
        changed_tempo = _to_scalar(tempo)
        # print(f"[feature extractor] : Changed The tempo = {changed_tempo}")
        # ensure tempo is float (or None)
        if tempo is not None:
            try:
                tempo = float(tempo)
            except Exception:
                pass
    except Exception:
        tempo = None

    # spectral features
    spec_cent = _to_scalar(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bw = _to_scalar(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zcr = _to_scalar(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = _to_scalar(np.mean(librosa.feature.rms(y=y)))

    # Harmonic and Percussive Energy
    # First, separate the audio into harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Calculate the mean RMS energy for each component
    harmonic_energy = _to_scalar(np.mean(librosa.feature.rms(y=y_harmonic)))
    percussive_energy = _to_scalar(np.mean(librosa.feature.rms(y=y_percussive)))

    # print(f"harmonic_energy: {harmonic_energy, type(harmonic_energy)}\n percussive_energy: {percussive_energy, type(percussive_energy)}")
    
    # --- Pitch Feature ---
    
    # 8. Estimated Pitch
    # We use pyin (probabilistic YIN) to estimate the fundamental frequency (F0)
    # This returns f0 (pitch), voiced_flag, and voiced_probs
    f0, _, _ = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7')
    )
    
    # f0 contains NaN for unvoiced frames. We use np.nanmean
    # to calculate the average pitch, *ignoring* the unvoiced frames.
    estimated_pitch = _to_scalar(np.nanmean(f0))

    # print(f"estimated pitch : {estimated_pitch}")

    return {
        "duration": duration,
        "tempo": changed_tempo,
        "spectral_centroid": spec_cent,
        "spectral_bandwidth": spec_bw,
        "zero_crossing_rate": zcr,
        "rms": rms,
        "harmonic_energy": harmonic_energy,
        "percussive_energy": percussive_energy,
        "estimated_pitch_hz": estimated_pitch if not np.isnan(estimated_pitch) else 0.0,
    }

print("[APP_1][TOOL]: âœ”ï¸� Feature Extractor Tool loaded for _feature_agent")

# -------------------------------------------------



# a simple helper to wake up sleeping freesound-mcp-server, by pinging it 

logger = logging.getLogger("mcp_wakeup")
logger.addHandler(logging.NullHandler())

# ---- Defaults you can override when calling functions ----
DEFAULT_PROBE_TIMEOUT = 5.0
DEFAULT_MAX_WAKE_SECONDS = 120
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 10.0

# ---- Core utilities ----

async def probe_server(url: str, timeout: float = DEFAULT_PROBE_TIMEOUT, headers: dict | None = None) -> Optional[int]:
    """
    Probe a given URL once. Returns HTTP status code if reachable, or None on network error.
    - Accepts headers for authenticated probes (e.g., Authorization).
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers, follow_redirects=True)
            return resp.status_code
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPError) as exc:
        logger.debug("probe_server exception: %s", exc)
        return None

async def wait_for_wakeup(
    probe_url: str,
    headers: dict | None = None,
    max_wait: int = DEFAULT_MAX_WAKE_SECONDS,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    accept_status_predicate=None,
    on_wakeup_message: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Polls probe_url until it responds with a status considered 'up' (default: status < 500),
    or until max_wait seconds elapse.

    Returns True if the server became reachable (status accepted), False on timeout.

    Parameters:
    - probe_url: URL to probe (e.g., root or /health).
    - headers: optional HTTP headers for probe.
    - max_wait: overall maximum wait time in seconds.
    - initial_backoff / max_backoff: backoff parameters (seconds).
    - probe_timeout: timeout for each probe attempt.
    - accept_status_predicate: optional function(int)->bool to decide which HTTP codes mean "up".
        Default: lambda status: status is not None and status < 500
    - on_wakeup_message: optional callback(msg) called once when waking begins (for UI).
    """
    accept_status_predicate = accept_status_predicate or (lambda status: status is not None and status < 500)

    start = time.time()
    backoff = initial_backoff
    first_notify = True

    while True:
        elapsed = time.time() - start
        if elapsed > max_wait:
            logger.warning("wait_for_wakeup: timed out after %.1f seconds", elapsed)
            return False

        status = await probe_server(probe_url, timeout=probe_timeout, headers=headers)

        if accept_status_predicate(status):
            # If we printed a wakeup message previously, optionally indicate success.
            if not first_notify and on_wakeup_message:
                try:
                    on_wakeup_message("Server is up (status {}).".format(status))
                except Exception:
                    pass
            return True

        # Not up yet -> notify first time and continue polling
        if first_notify:
            msg = "Server appears to be asleep / unreachable. Attempting to wake (polling {})...".format(probe_url)
            if on_wakeup_message:
                try:
                    on_wakeup_message(msg)
                except Exception:
                    pass
            else:
                logger.info(msg)
            first_notify = False
        else:
            logger.debug("Still waiting for server: status=%s, elapsed=%.1f", status, elapsed)

        await asyncio.sleep(min(backoff, max_backoff))
        backoff *= 1.8



freesound_api_key = os.getenv("FREESOUND_API_KEY", "")
MCP_SERVER_URI = "https://freesound-mcp-server.onrender.com/mcp"  # the deloyment is free to use for all


print("[APP_1][MCP_TOOL] : â„¹ï¸� Checking the MCP server status")

ok = await wait_for_wakeup(MCP_SERVER_URI, on_wakeup_message=print)  # gives server status True or False

if ok:
    print("[APP_1][MCP_TOOL] : âœ”ï¸� Server woke up!!")
else:
    print("[APP_1][MCP_TOOL] : â�Œ Server did NOT wake up in time")



# MCP integration with Freesound org
mcp_sound_server = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url= MCP_SERVER_URI,
        headers= {"Authorization": freesound_api_key},
    ),
)

print("[APP_1][MCP_TOOL]: âœ”ï¸� MCP Tool created")


# ================================================
# 1 Audio Feature Extracter tool 
_feature_agent_instance = Agent(
    name="feature_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config), 
    description="A simple agent that can describe the given audio sample.",
    instruction="""
    You are a audio feature extrator agent, you are not suppose to chat with user.
    1. The user will provide a prompt containing a file path.
    2. Use the 'compute_basic_descriptors' tool with that 'audio_path' to extract audio features.
    3. Your final output MUST be a valid JSON object with a SINGLE key 
       named 'descriptors'.
    """,
    tools=[compute_basic_descriptors],
    output_key="descriptors"
)

print("[APP_1][Agents] : âœ”ï¸� Audio feature agent created!")

# ==================================================





# ==================================================
# 2 Classifier agent, to classify the genre, mood of the given audio sample 
_classifier_agent_instance = Agent(
    name="classifier_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are an expert audio classifier.
    
    1. You will receive JSON string {descriptors} containing audio features.
    2. Analyze these features to determine the nature of the sound.
    3. STRICTLY Return JSON with keys: 
        "style_tags" (list of up to 4 descriptive tags), 
        "genre_suggestions" (list up to 3), 
        "texture" (one of: 'gritty','warm','bright','dark','percussive','smooth','wide'), 
        "confidence" (0-1 float). 
    """,
    output_schema=ClassificationOutput,
    output_key="classification",
)

print("[APP_1][Agents] : âœ”ï¸� Audio Classifier agent created!")

# ==================================================






# ===================================================
# 3. Recommender agent, recommends complimentory sounds, layers, fx chains 
_recommender_agent = Agent(
    name="recommender_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a sound recommender, who have professional and creative knowledge about sound designing and musical genres.
    1. use the {classification} information of the audio given by the user and user prompt
    2. if user prompt has an intent or goal to do with given sound use that to give recommendations of the sounds or you can use your own creative approach 
    3. Style rules snippet (JSON) which lists typical layers, fx_chains, sample_keywords and preset_tweaks for the detected style.
    
    Produce a JSON object: {{
        "recommendations": [
            {{ "id": "<id>", "type":"layer|fx_chain|preset_tweak|sample_keyword|variation",
            "title":"", "short_description":"", "actionable_parameters":{{}}, "confidence":0.0 }}
        ]
        }}

    Constraints:
    - Produce 4 recommendations, ranked by confidence (highest first).
    - For each "actionable_parameters" include concrete parameters (e.g. cutoff_hz, gain_db, synth: 'sine', filter: {{...}}).
    - Output MUST BE a JSON
    """,
    output_key="recommendations",
) 

print("[APP_1][Agents] : âœ”ï¸� Recommender Agent is created!")
# ===================================================







# ===================================================
# 4. sample search agent
_sample_search_agent = Agent(
    name="sample_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="This agent will rely on recommendations and search for such sounds and show it to users to preview it",
    instruction="""You are a sample sound searcher
    - using the type layer in {recommendations}, use the tool 'mcp_sound_server' to look for 5 distict sounds to recommend to user
    - lit the 5 found sounds in below manner 
        - found sound sample name : it's preview URL IMPORTATN! THE URL COMES AFTER SOUND NAME AND ALL URLs MUST BE WORKING ONES  
    """,
    tools=[mcp_sound_server],
    output_key='preview_sounds'
)

print("[APP_1][Agents] : âœ”ï¸� Sample Searching Agent is created!")
# ===================================================








# ===================================================
# aggregator agent to do the aggregation of the information
_aggregator_agent = Agent(
    name="aggregator_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="This is the main face of the sound designer agent, it will manage other subagents too and pass them the key info needed",
    instruction="""
    Combine these three results into one response
    - {classification} turn this JSON into bullet point, consider Top level to be parent and do indentation for childs bullet points, REDACT CONFIDENCE VALUE
    - {recommendations} turn this JSON into bullet point, consider Top level to be parent and do indentation for childs bullet points REDACT CONFIDENCE VALUE & DON'T REWRITE 'short_description'
    - {preview_sounds}, DON'T REWRITE AND PRESERVE THE STRUCTURE
    - All points MUST be one liner
    """,
)
   

print("[APP_1][Agents] : âœ”ï¸� Aggregator Agent is created!")
# ===================================================




# Orchestrator Sequential Workflow Pipeline
sound_design= SequentialAgent(
    name= "sound_design",
    description="This is the start of the pipeline that orchestrates and runs the sub-agents in the sequential manner.",
    sub_agents=[_feature_agent_instance, _classifier_agent_instance, _recommender_agent, _sample_search_agent, _aggregator_agent], 
)

print("[APP_1][PIPELINE] : âœ”ï¸� Orchestrator Pipeline created!")


# Wrapping eveything into one APP
# Orchestrator App Wrapper for advanced feature access
sound_design_app = App(
    name="agents",
    root_agent=sound_design,   # TODO : we need to replace orchestrator with an agent that can take these values and work on them, root agent is messing up.
    resumability_config=ResumabilityConfig(is_resumable=True),
)

print("[APP_1][ORCHESTRATOR APP] : âœ”ï¸� Orchestrator App created!")


"""
This part just uses independent LLM Agent, that can think on it's own about the sound
and tries to come up with some tweaks and fx chain instruction that is later on passed 
on to a python function that generates the sound on by following the LLM Instructions
""" 

# the agent: synth_agent
synth_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite"),
    name="synth_agent",
    instruction="""You are a sound creative synthesizer agent
    - You take user prompt and the audio sample via audio_path
    - if user has any intent for the sound try to choose fx process for that, otherwise you are free to create your own fx chain
    - example: {
                "tool": "synthesis_tool",
                "function": "apply_patch",
                "args": {
                    "input_audio_path": "relative path",
                    "out_path": "relative path",
                    "sr": 22050,
                    "mix_ratio": 0.75,
                    "params": { ... }        // structured parameters (MUST)
                    }
                }
    - Rules & constraints:
        1. Types: Use proper JSON types â€” numbers should be numbers (not strings), booleans true/false, arrays for lists, objects for maps.
        2. If you include an instruction string, it will be parsed; **prefer** returning the structured `params` object (more reliable).
        3. Allowed `params` keys (optional; include only those required): 
            - "sub_sine": {"enabled": bool, "freq_hz": number, "amp": number (0-1), "lowpass_cutoff": number (hz) }
            - "noise": {"enabled": bool, "amp": number}
            - "distortion": {"enabled": bool, "drive": number}
            - "delay": {"enabled": bool, "ms": integer, "feedback": number (0-1)}
            - "global_lowpass": number (hz)
            - "global_highpass": number (hz)
        4. `mix_ratio` (0-1): proportion of original audio in final mix. Use values like 0.6, 0.75.
        5. Keep numeric values realistic (Hz frequencies typically 20-20000, delay ms 10-600, amp 0-1, drive 0.5-3).
        6. If uncertain about exact numbers, pick conservative defaults that produce musical results (e.g., sub_sine amp 0.4-0.6, lowpass 120Hz).
        7. If you cannot find a clear param mapping, include a conservative default `params` object with `sub_sine.enabled = true` and sensible defaults.
        8. **Do not** request arbitrary code execution or unvalidated paths.
        9. Output must parse as JSON with no extra characters.
    """,
)


# App wrapper for advance feature enabling
synth_app = App(
    name="synth_app",
    root_agent=synth_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

print("[APP_1][SYNTH_APP] : âœ”ï¸� Synthesizer app is created!")


def load_mono(path: str, sr: int = 22050):
    y, sr2 = librosa.load(path, sr=sr, mono=True)
    return y, sr

def _sine_wave(freq_hz: float, duration_s: float, sr: int = 22050, amp: float = 0.5):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return amp * np.sin(2 * np.pi * freq_hz * t)

def _lowpass(signal, cutoff, sr, order=4):
    nyquist = 0.5 * sr
    norm_cutoff = max(1e-6, min(cutoff / nyquist, 0.999))
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

def _highpass(signal, cutoff, sr, order=4):
    nyquist = 0.5 * sr
    norm_cutoff = max(1e-6, min(cutoff / nyquist, 0.999))
    b, a = butter(order, norm_cutoff, btype='high', analog=False)
    return lfilter(b, a, signal)

def _soft_distort(signal, drive=1.0):
    # simple tanh distortion
    return np.tanh(signal * drive)

def _add_delay(signal, sr, delay_ms=60, feedback=0.2):
    delay_s = delay_ms / 1000.0
    delay_samples = int(sr * delay_s)
    out = np.copy(signal)
    for i in range(delay_samples, len(signal)):
        out[i] += feedback * out[i - delay_samples]
    # normalize
    m = np.max(np.abs(out)) + 1e-9
    if m > 1.0:
        out = out / m * 0.95
    return out

def _add_noise(signal, noise_amp=0.02):
    noise = np.random.randn(len(signal)) * noise_amp
    return signal + noise

def apply_patch(
    input_audio_path: str,
    out_path: str,
    instructions: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    sr: int = 22050,
    mix_ratio: float = 0.75
) -> Dict[str, Any]:
    """
    High-level tool: loads input audio, interprets instructions or params,
    applies synthesis and effects and writes out_path.
    Returns metadata with applied params and path.

    arg:
        input_audio_path: original audio file path uploaded by the user 
        out_path: path of a new synthesized file being written to
        instructions: LLM given JSON based instructions
        params: LLM suggested tweaks
        sr: sample rate of the audio file
        mix_ration: wet and dry ratio of the fx into original file

    return:
        JSON String that has output path to synthesized file, and applied params on it 
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # load
    y, sr = load_mono(input_audio_path, sr=sr)
    duration = len(y) / sr

    # If params are explicitly provided, trust them. Otherwise parse instructions.
    # if params is None:
    #     params = interpret_instructions(instructions or "", y, sr)

    # Start with original (or silence) and build layers
    base = y.copy()
    out = base.copy()

    # SUB-SINE layer
    if params.get("sub_sine", {}).get("enabled", False):
        f = params["sub_sine"].get("freq_hz", params["sub_sine"].get("ratio_freq_hz", 55.0))
        amp = params["sub_sine"].get("amp", 0.5)
        sub = _sine_wave(f, duration, sr=sr, amp=amp)
        # optional lowpass on sub
        if params["sub_sine"].get("lowpass_cutoff"):
            sub = _lowpass(sub, params["sub_sine"]["lowpass_cutoff"], sr)
        out = out * mix_ratio + sub * (1.0 - mix_ratio)

    # Noise
    if params.get("noise", {}).get("enabled", False):
        out = _add_noise(out, params["noise"].get("amp", 0.01))

    # Distortion
    if params.get("distortion", {}).get("enabled", False):
        drive = params["distortion"].get("drive", 1.0)
        out = _soft_distort(out, drive=drive)

    # Lowpass/Highpass global
    if params.get("global_lowpass"):
        out = _lowpass(out, params["global_lowpass"], sr)
    if params.get("global_highpass"):
        out = _highpass(out, params["global_highpass"], sr)

    # Delay
    if params.get("delay", {}).get("enabled", False):
        out = _add_delay(out, sr, delay_ms=params["delay"].get("ms", 60), feedback=params["delay"].get("feedback", 0.15))

    # Normalize and clip-safe
    maxv = np.max(np.abs(out)) + 1e-9
    if maxv > 1.0:
        out = out / maxv * 0.95

    sf.write(out_path, out.astype(np.float32), sr)

    return {"ok": True, "path": out_path, "params": params}




# this helper method executes the above PATCH APPLIER utility

ALLOWED_FUNCTIONS = {
    "apply_patch": apply_patch
}

def simple_freq_extract(text: str):
    # find frequencies like 120Hz, 120 Hz, or numbers followed by Hz
    m = re.search(r"(\d{2,4})\s*hz", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # find ms for delay
    return None

def interpret_instructions(instructions: str, sr: int = 22050):
    """
    Very small rule-based parser: returns params dict.
    LLM should ideally output structured params; this parser helps when LLM gives plain text.

    args:
        instructions: str, expected JSON string but if in case its plain text
        sr: sample rate
  
    """
    t = instructions.lower()
    params = {}

    # Sub-sine: look for "sub", "one octave below", "octave"
    if "sub" in t or "one octave" in t or "octave below" in t:
        # estimate pitch from audio if needed: for simplicity default 55Hz
        base_freq = 55.0
        if "hz" in t:
            f = simple_freq_extract(t)
            if f:
                base_freq = f
        params["sub_sine"] = {"enabled": True, "freq_hz": base_freq / 1.0, "amp": 0.45}
        # optional lowpass
        if "lowpass" in t:
            f_lp = simple_freq_extract(t) or 120.0
            params["sub_sine"]["lowpass_cutoff"] = f_lp

    # Distortion
    if "distort" in t or "distortion" in t or "drive" in t:
        drive = 1.0
        dm = re.search(r"drive\s*(?:=|:)?\s*(\d(?:\.\d)?)", t)
        if dm:
            try:
                drive = float(dm.group(1))
            except:
                pass
        params["distortion"] = {"enabled": True, "drive": drive}

    # Noise
    if "noise" in t:
        na = re.search(r"noise\s*(?:amp)?\s*(?:=|:)?\s*(0?\.\d+|\d)", t)
        amp = 0.01
        if na:
            try:
                amp = float(na.group(1))
            except:
                pass
        params["noise"] = {"enabled": True, "amp": amp}

    # Delay
    if "delay" in t or "echo" in t:
        ms = 60
        mm = re.search(r"(\d{1,3})\s*ms", t)
        if mm:
            ms = int(mm.group(1))
        fb = 0.15
        fbm = re.search(r"feedback\s*(?:=|:)?\s*(0?\.\d+|\d)", t)
        if fbm:
            try:
                fb = float(fbm.group(1))
            except:
                pass
        params["delay"] = {"enabled": True, "ms": ms, "feedback": fb}

    # Global filters
    if "lowpass" in t and "sub" not in t:
        f_lp = simple_freq_extract(t) or 8000
        params["global_lowpass"] = f_lp
    if "highpass" in t:
        f_hp = simple_freq_extract(t) or 20
        params["global_highpass"] = f_hp

    # Default fallback: enable sub-sine if nothing recognized
    if not params:
        params = {"sub_sine": {"enabled": True, "freq_hz": 55.0, "amp": 0.4}}

    return params

def execute_tool(function: str, args: Dict[str, Any], file_path: str, out_path: str) -> Dict[str, Any]:
    """
    Validate and execute a limited set of functions. Returns JSON-serializable result.

    args:
        function: takes the enum function name 
        args: LLM JSON string's keys 
    return:
        JSON String a python DICT 
    """
    if function not in ALLOWED_FUNCTIONS:
        return {"ok": False, "error": f"Function {function} not allowed."}
    # required args
    if "input_audio_path" not in args or "out_path" not in args:
        return {"ok": False, "error": "Missing required args: input_audio_path and out_path"}

    instructions = args.get("instructions", "")
    params = args.get("params")  # structured override
    sr = int(args.get("sr", 22050))
    mix_ratio = float(args.get("mix_ratio", 0.75))

    # if instructions present and no params provided, parse them:
    if params is None:
        params = interpret_instructions(instructions or "", None, sr)

    # finally call the function
    try:
        res = ALLOWED_FUNCTIONS[function](
            input_audio_path=file_path,
            out_path=out_path,
            instructions=instructions,
            params=params,
            sr=sr,
            mix_ratio=mix_ratio
        )
        return {"ok": True, "result": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}


print("[APP_1][SYNTH_APP] : âœ”ï¸� execute_tool utility created")


# this helper has two sub-utilities defined above, code_exec_tool, and apply_patch
def handle_llm_instruct(
    llm_json: Any, 
    file_path: str, 
    out_path: str
) -> Dict[str, Any]:
    """
    Accepts either:
    - a dict representing the LLM tool call
    - a JSON string containing the tool call

    file_path  -> absolute/relative path of uploaded input audio
    out_path   -> desired output synthesized file path

    Injects these into args, overrides the LLM output path, 
    and calls execute_tool to run apply_patch.
    """

    # 1. Parse safely if llm_json is a string
    if isinstance(llm_json, str):
        llm_json = llm_json.strip()
        if not llm_json:
            return {"ok": False, "error": "Empty LLM response string"}
        try:
            call = json.loads(llm_json)
        except Exception as e:
            return {"ok": False, "error": f"Failed to parse LLM JSON string: {e}"}
    # 2. Accept dict directly
    elif isinstance(llm_json, dict):
        call = llm_json
    else:
        return {"ok": False, "error": f"Unexpected LLM response type: {type(llm_json)}"}

    # 3. Validate tool
    if call.get("tool") != "synthesis_tool":
        return {"ok": False, "error": f"Unsupported tool: {call.get('tool')}"}

    func = call.get("function")
    if not func:
        return {"ok": False, "error": "Missing 'function' field in LLM output"}

    # 4. Extract args
    args = call.get("args", {})
    if not isinstance(args, dict):
        return {"ok": False, "error": "'args' must be a dict"}

    # 5. Override paths (so agent cannot write anywhere else)
    args["input_audio_path"] = file_path
    args["out_path"] = out_path  # forced override for safety

    # 6. Call your tool executor
    try:
        resp = execute_tool(func, args, file_path, out_path)
        return resp
    except Exception as e:
        return {"ok": False, "error": f"execute_tool raised: {e}"}

print("[APP_1][SYNTH_APP]: âœ”ï¸� SYNTH LLM Response Handler utility created!")


print("[APP_2][A2A_HF_SPACE] : â„¹ï¸� Checking the A2A server status")

okay = await wait_for_wakeup("https://shivenduu-songgeneratora2a.hf.space", on_wakeup_message=print)  # gives server status True or False

if okay:
    print("[APP_2][A2A_HF_SPACE] : âœ”ï¸� A2A Server woke up!!")
else:
    print("[APP_1][A2A_HF_SPACE] : â�Œ Server did NOT wake up in time")


from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

# this is the publicly deployed A2A Agent Server 
A2A_HF_SPACE_URI = "https://shivenduu-songgeneratora2a.hf.space"


_remote_generator_agent = RemoteA2aAgent(
    name="remote_generate_song_agent",
    description="This is a remote agent that can generate audios from given text.",
    agent_card=f"{A2A_HF_SPACE_URI}{AGENT_CARD_WELL_KNOWN_PATH}",
    timeout=1000.0
)

print("[APP_2][A2A]: âœ”ï¸� Remote Agent onbaord!")
print(f"[APP_2][A2A]: âœ”ï¸� Agent Card: {A2A_HF_SPACE_URI}{AGENT_CARD_WELL_KNOWN_PATH}")


_generate_song_local = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="customer_support_agent",
    description="A customer support assistant that helps customers with product inquiries and information.",
    instruction="""
    You are an agent that can create audio from text, YOU ARE NOT A CHATBOT
    - user will give prompt in JSON String format         
    - use the _remote_generator_agent sub-agent to perform this task by giving JSON as input to it
    - MUST extract the 'url' from sub-agent and show it to user like JSON below
        {
        "url" : give the url here
        }
    """,
    sub_agents=[_remote_generator_agent],
    output_key='audio_url'
)


print("[APP_2][A2A]: âœ”ï¸� Local Agent is created!")


song_app = App(
    name="song_app",
    root_agent=_generate_song_local,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

print("[APP_2][A2A]: âœ”ï¸� Song App is created!")


async def test_a2a_communication(user_query: str, user_id: Optional[str] = None, session_id: Optional[str] = None):
    """
    Test the A2A communication between Customer Support Agent and Product Catalog Agent.

    This function:
    1. Creates a new session for this conversation
    2. Sends the query to the Customer Support Agent
    3. Support Agent communicates with Product Catalog Agent via A2A
    4. Displays the response

    Args:
        user_query: The question to ask the Customer Support Agent

    Return:
        Dict: agent_response
    """
    agent_response = ""

    # Setup session management (required by ADK)
    db_url = "sqlite:///song_bank.db"
    session_service = DatabaseSessionService(db_url=db_url)

    # Session identifiers
    user_id = user_id or "user_01"
    # Use unique session ID for each test to avoid conflicts
    session_id = session_id or "test_session_01"

    # CRITICAL: Create session BEFORE running agent (synchronous, not async!)
    # This pattern matches the deployment notebook exactly
    try:
        session = await session_service.create_session(app_name=song_app.name, user_id=user_id, session_id=session_id)
    except Exception:
        session = await session_service.get_session(app_name=song_app.name, user_id=user_id, session_id=session_id)

    # Create runner for the Agent
    # The runner manages the agent execution and session state
    runner = Runner(
        app=song_app, session_service=session_service
    )

    # Adding two important default params with query 
    user_query_wdefault_param = user_query +", cfg_coef: 1.5, temperature: 0.8"
    
    # Create the user message
    # This follows the same pattern as the deployment notebook
    test_content = types.Content(parts=[types.Part(text=user_query_wdefault_param)])

    # Display query
    print(f"\nğŸ‘¤ User: {user_query}")
    print(f"\nğŸ�§ A2A agent:")
    print("-" * 60)

    # Run the agent asynchronously (handles streaming responses and A2A communication)
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=test_content
    ):
        # events.append(event)
        if event.content and event.content.parts:
            # Filter out empty or "None" responses before printing
            if (event.content.parts[0].text != "None" and event.content.parts[0].text):
                print(f"assistant > ", event.content.parts[0].text)
                agent_response += event.content.parts[0].text 


    print(f"{'-'*20} Adding session to Long Term Memory {'-'*20}")

    song_sess = await runner.session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)

    print(f"[APP_2]: âœ”ï¸� Song Generation Agent Session saved in Long term memory!")
    print("-" * 60)
    return agent_response


# Run the test
print("[APP_2][A2A]: âœ”ï¸� A2A communication Helper is created!")


# lighter chat_agent without any heavy tools and MCP calls and with low latency to reply to user   
chat_agent = LlmAgent(
    name="chat_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="This agent deals with a scenario where user don't need any file analysis or have sound designing sample suggestions user might ask like: what was my previously sent file, It will simply do the database Session look up for this instead of redoing the orchestrator pipeline",
    instruction="""You are SoundSpark's conversational assistant. 
    - Answer the user's question. Use the 'load_memory' tool if the answer might be in past conversations.
    - Do not assume or invent user's uploaded files. If the user refers to 'my previous file' and no session file exists, ask them to re-upload or provide the file path.
    - If the user asks a general chatty question unrelated to audio (e.g., 'how are you', 'what plugins are good for reverb'), simply reply if unsure you may use google_search tool.
    - If unclear whether we need audio analysis, choose "clarify" and ask a 1-line clarifying question.
    """,
    tools=[load_memory]#[PreloadMemoryTool()] 
)



# Chat App wrapper
chat_app = App(
    name="agents",
    root_agent=chat_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
) 

print("[APP_3][CHAT]: âœ”ï¸� Chat app is created!")


# This cell is needed to be run once 
# from google.colab import auth

# auth.authenticate_user(project_id=os.environ["GOOGLE_CLOUD_PROJECT"])


import vertexai
from google.adk.memory import VertexAiMemoryBankService

# GOOGLE_CLOUD_LOCATION 
GOOGLE_CLOUD_LOCATION="us-central1"

# Set to 1 to use Vertex AI, or 0 to use Google AI Studio
GOOGLE_GENAI_USE_VERTEXAI=1



vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=GOOGLE_CLOUD_LOCATION,
)


client = vertexai.Client(
  project=os.environ["GOOGLE_CLOUD_PROJECT"],
  location=GOOGLE_CLOUD_LOCATION,
)

# Memory Bank instance using the default configuration.
agent_engine = client.agent_engines.create()

# resource name to interact with Agent Engine instance later on.
print(agent_engine.api_resource.name)

agent_engine_id = agent_engine.api_resource.name.split("/")[-1]

memory_service = VertexAiMemoryBankService(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=GOOGLE_CLOUD_LOCATION,
    agent_engine_id=agent_engine_id
)

print(f"[VertexAi]: âœ”ï¸� Vertex Ai Memory Service ready!")


# for persistent memory
db_url = "sqlite:///memory_bank.db"
session_service = DatabaseSessionService(db_url=db_url)


async def run_workflow(prompt: str, user_id: Optional[str] = None, session_id: Optional[str] = None):
    """
    Run the full agent workflow with user prompt

    Args: 
        prompt: user prompt in string
    """
    user_id = user_id or "user_01"
    session_id = session_id or "test_session_01"
    
    # print(f"user > {prompt}")

    # routing logic
    route_data = route_prompt(prompt)
    
    if route_data['output'] == 1:

        # extract the audio file path to be used by Synth App
        sample = route_data['path']
        
        print(f"{'-'*20} Sound Design App Invoked {'-'*20}")
        runner_1 = Runner(app=sound_design_app, session_service=session_service, memory_service=memory_service)
        sda_response = await run_session(runner_1, prompt, session_id, user_id)

        print("\n\n")
        print(f"{'-'*20} Synth App Invoked {'-'*20}")
        runner_synth = Runner(app=synth_app, session_service=session_service, memory_service=memory_service)

        # we need to capture the Agent response as instruction to act on them while synthesizing the sound
        synth_response = await run_session(runner_synth, prompt, session_id, user_id)

        # convert the LLM response to JSON
        synth_response_json = give_json(synth_response)

        # the handle_llm_instruct gives back synthesized audio's path and status OK, if nothing fails
        audio_synth_status = handle_llm_instruct(synth_response_json, sample, f"/kaggle/working/tests/synthesis_demo/{Path(sample).stem}_new.mp3")

        print(f"Synth App Status: {audio_synth_status}")

        print(f"{'-'*20} Adding session to Long Term Memory {'-'*20}")

        desgn_sess = await runner_1.session_service.get_session(app_name=runner_1.app_name, user_id=user_id, session_id=session_id)
        synth_sess = await runner_synth.session_service.get_session(app_name=runner_synth.app_name, user_id=user_id, session_id=session_id)
        await memory_service.add_session_to_memory(desgn_sess) 
        await memory_service.add_session_to_memory(synth_sess)

        print("[App 1]: Memory Saved!\n\n")

        # return the responses so that URLs, can be shown in playable manner after extraction
        return {"agent_response": [sda_response, audio_synth_status]}


    elif route_data['output'] == 2:

        print(f"{'-'*20} Song Generation Agent Invoked {'-'*20}")

        # this mode has all the DB session memory code inside the test_a2a_communication method 
        
        # test_a2a_communication does synchronus session creation, for a2a
        song_gen_response = await test_a2a_communication(prompt)

        # parse the LLM response into JSON
        song_gen_response_json = give_json(song_gen_response)

        # This agent gives back simple url to the generated audio
        return {"agent_response":[song_gen_response_json]}

    elif route_data['output'] == 3:

        print(f"{'-'*20} Chat agent Invoked {'-'*20}")

        runner_chat = Runner(app=chat_app, session_service=session_service, memory_service=memory_service)
        await run_session(runner_chat, prompt, session_id, user_id)

        print(f"{'-'*20} Adding session to Long Term Memory {'-'*20}")

        chat_sess = await runner_chat.session_service.get_session(app_name=runner_chat.app_name, user_id=user_id, session_id=session_id)
        await memory_service.add_session_to_memory(chat_sess) 

        print(f"\n{'-'*20} [APP_3][Chat App]: âœ”ï¸� Memory Added! {'-'*20}")

        return {"agent_response":[None]}
        

print("[Workflow]: âœ”ï¸� Run workflow utility created")


await run_workflow("What did we discuss so far")


await run_workflow("Hi! My name is Shivendu, I am a bass music producer. But I like the genre R&B more!")


await run_workflow("Hi! what did we discussed earlier, and what's my name, what genre I producer and what genre I like more?")


await run_workflow("What is my name, my favourite genre, and the genre I produce?", "user_01", "test_session_02")


await run_workflow("What did we talk so far?", "user_01", "test_session_03")


from google.adk.sessions import InMemorySessionService, Session

# temporarily we'll use InMemory Session Service instead of DB Session for this demo
new_runner = Runner(app=chat_app, session_service=session_service, memory_service=memory_service) 
await run_session(new_runner, "My name and about my music preferences?", "test_session_04", "user_01")


from IPython.display import display, HTML, Audio
import re
from typing import List

# Utility: extract URLs robustly
def extract_urls(text: str) -> List[str]:
    """
    Extracts URLs from a long string and returns a list of cleaned URLs.
    Handles both http and https. Removes trailing punctuation like ),. or ] if accidentally captured.
    """
    if not text:
        return []
    # Basic URL regex (http/https)
    raw_urls = re.findall(r"https?://[^\s'\"<>()\[\]]+", text)
    cleaned = []
    for u in raw_urls:
        # strip trailing punctuation commonly attached
        u = u.rstrip('.,;:)"\'')
        cleaned.append(u)
    # deduplicate while preserving order
    seen = set()
    result = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result

# Utility: show up to `max_players` audio players for given URLs
def show_audio_players(urls: List[str], max_players: int = 5):
    """
    Display up to max_players HTML <audio> controls for the URLs.
    If no URLs provided, prints a friendly message.
    """
    if not urls:
        display(HTML("<b>No audio Suggestion Found.</b>"))
        return
    # Limit to max_players
    urls_to_show = urls[:max_players]
    # Build simple HTML with one audio player per URL
    html_parts = []
    for i, url in enumerate(urls_to_show, start=1):
        # Each player in its own block with the URL shown
        player_html = f"""
        <div style="margin:8px 0; padding:6px; border:1px solid #eee; border-radius:6px;">
            <div style="font-size:0.9em; margin-bottom:6px;"><b>URL {i}:</b> <a href="{url}" target="_blank" rel="noopener">{url}</a></div>
            <audio controls preload="none" style="width:100%;">
                <source src="{url}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
        </div>
        """
        html_parts.append(player_html)
    combined = "<div>" + "\n".join(html_parts) + "</div>"
    display(HTML(combined))

# for local files to be played as audio in notebook
def to_notebook_url(path):
    """
    Convert a local Kaggle file path to a notebook-accessible URL.
    If already a URL, return unchanged.
    """
    if path.startswith("http://") or path.startswith("https://"):
        return path

    # Local Kaggle file case
    if path.startswith("/kaggle/working/"):
        rel = path.replace("/kaggle/working/", "")
        return f"/files/{rel}"

    return path  # fallback

print("âœ”ï¸� URL extractor created!")


sample_sound_path = "/kaggle/working/tests/sample_audio/sub_bass.wav"

prompt = f"/design How can I make an 808 out of this sound, path:{sample_sound_path}"

response = await run_workflow(prompt) 

# extract the urls
sd_agent_resp = response.get("agent_response",[])[0]
synth_url = response.get("agent_response",[])[1].get("result",{}).get("path", "")

# to show them

# Run extraction
mcp_urls = extract_urls(sd_agent_resp)

# Print the list (safe even if empty)
# print("Extracted URLs:", mcp_urls)

# Display up to 5 audio players
print("\n")
print(" ----------------- Audio Player for MCP Searched Sound (If not playable, open the url in new tab) -----------------\n")
show_audio_players(mcp_urls, max_players=5)

print("\n")
print("----------------- Audio Player for Synthesized Sound -----------------\n")
Audio(filename=synth_url)


prompt = f"""/vocal lyrics: [chorus]
forsake the lord, says the devil
partake the apple, key to heaven
a whisper of sin in a midnight reverie
as shadows dance slow to a blue-note melody,
description: male, jazz, piano,
genre: jazz,
duration: 10"""

a2a_response = await run_workflow(prompt)


audio_url = a2a_response.get("agent_response",{})[0].get("url","")

print("------------------ You audio is ready ------------------")
show_audio_players([audio_url], max_players=1)


await run_workflow("What sounds we worked with previously?")

