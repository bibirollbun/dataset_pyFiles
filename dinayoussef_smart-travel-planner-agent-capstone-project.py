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


# ==== Imports & Setup ====
import os, time, json, uuid
import pandas as pd
import requests
from typing import List, Dict, Any, Optional

from IPython.display import display, Markdown, HTML
from jupyter_server.serverapp import list_running_servers
from kaggle_secrets import UserSecretsClient

# ADK / Gemini
!pip install -q google-generativeai google-adk

from google.genai import types
from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search


# ==== Secrets ====
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception as e:
    print(f"ðŸ”‘ Secrets Error: Add GOOGLE_API_KEY (required) to Kaggle secrets. Details: {e}")


HOTELS_XLSX_PATH = "/kaggle/input/hotels-netherlands/HotelFinalDataset.xlsx"
hotels_df = pd.read_excel(HOTELS_XLSX_PATH, sheet_name=0)

if "Unnamed: 0" in hotels_df.columns:
    hotels_df = hotels_df.drop(columns=["Unnamed: 0"])

hotels_df["PriceValue"] = (
    hotels_df["Price"].astype(str)
    .str.replace(r"[^\d.]", "", regex=True)
    .astype(float)
)

display(hotels_df.head())
display(Markdown(f"**Rows:** {len(hotels_df)}"))
display(Markdown(f"**Columns:** {list(hotels_df.columns)}"))


# ==== Observability ====
metrics = {"calls": {"hotel_agent":0,"landmark_agent":0,"preference_agent":0,"aggregator":0},
           "latency_ms": {"hotel_agent":[],"landmark_agent":[],"preference_agent":[],"aggregator":[]}}

def log_event(label: str, payload: Any):
    print(f"[LOG] {label}: {payload}")

def track_latency(agent_name: str, start_time: float):
    elapsed = (time.time() - start_time) * 1000
    metrics["latency_ms"][agent_name].append(elapsed)
    print(f"[TRACE] {agent_name} latency: {elapsed:.2f} ms")

def summary_metrics():
    print("\n[METRICS] Summary")
    for agent, calls in metrics["calls"].items():
        lat = metrics["latency_ms"][agent]
        avg = sum(lat)/len(lat) if lat else 0.0
        print(f"- {agent}: calls={calls}, avg_latency_ms={avg:.2f}")


# ==== Agents ====
def preference_agent(preferences: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    metrics["calls"]["preference_agent"] += 1
    log_event("PreferenceAgent::preferences", preferences)
    track_latency("preference_agent", start)
    return preferences

def hotel_agent(city: str, top_n: int = 3, sort_by: str = "PriceValue", ascending: bool = True,
                min_price: Optional[float] = None, max_price: Optional[float] = None) -> pd.DataFrame:
    start = time.time()
    metrics["calls"]["hotel_agent"] += 1

    subset = hotels_df[hotels_df["City"].str.lower().str.contains(city.strip().lower())].copy()
    if min_price is not None:
        subset = subset[subset["PriceValue"] >= float(min_price)]
    if max_price is not None:
        subset = subset[subset["PriceValue"] <= float(max_price)]
    subset = subset.sort_values(by=sort_by, ascending=ascending)

    top_hotels = subset.head(top_n)
    log_event("HotelAgent::top_hotels", top_hotels[["Name","City","Price","Rating"]].to_dict("records"))
    track_latency("hotel_agent", start)
    return top_hotels[["Name","City","Price","Rating"]]

# ==== Landmark Agent (Gemini via ADK) ====
FALLBACK_LANDMARKS = {
    "Amsterdam": ["Rijksmuseum","Van Gogh Museum","Anne Frank House","Dam Square","Vondelpark"],
    "Rotterdam": ["Erasmus Bridge","Cube Houses","Markthal","Euromast"],
    "Utrecht": ["Dom Tower","Railway Museum","Centraal Museum","Oude Hortus"],
    "The Hague": ["Mauritshuis","Binnenhof","Scheveningen Pier","Peace Palace"]
}

retry_config = types.HttpRetryOptions(attempts=3, exp_base=2, initial_delay=1, http_status_codes=[429,500,503])
gemini_model = Gemini(model="gemini-2.5-flash-lite", api_key=GOOGLE_API_KEY, retry_options=retry_config)

landmark_llm = LlmAgent(
    name="landmark_llm",
    model=gemini_model,
    instruction="Summarize briefly why the given landmarks are iconic for travelers."
)
runner = InMemoryRunner(agent=landmark_llm)

async def landmark_agent(city: str, limit: int = 5) -> Dict[str, Any]:
    start = time.time()
    metrics["calls"]["landmark_agent"] += 1

    results = FALLBACK_LANDMARKS.get(city, ["Landmarks not available"])
    clean = results[:limit]

    prompt = f"Explain briefly why these landmarks in {city} are iconic: {', '.join(clean)}"
    try:
        response = await runner.run_debug(prompt)
        summary = response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        print(f"[ERROR] Gemini summarization failed: {e}")
        summary = f"In {city}, notable landmarks include {', '.join(clean)}."

    payload = {"landmarks": clean, "summary": summary}
    log_event("LandmarkAgent::result", payload)
    track_latency("landmark_agent", start)
    return payload


# ==== Aggregator ====
async def aggregator(city: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    metrics["calls"]["aggregator"] += 1

    min_price = preferences.get("min_price")
    max_price = preferences.get("max_price")
    budget = preferences.get("budget")

    if budget == "budget" and max_price is None:
        max_price = 120.0
    elif budget == "luxury" and min_price is None:
        min_price = 200.0

    hotels_top3 = hotel_agent(city, top_n=3, sort_by="PriceValue", ascending=True,
                              min_price=min_price, max_price=max_price)
    landmarks_best = await landmark_agent(city, limit=5)

    recommendation = {
        "city": city,
        "hotels_top3": hotels_top3.to_dict("records"),
        "landmarks_best": landmarks_best["landmarks"],
        "landmarks_summary": landmarks_best["summary"],
        "reasoning": "Hotels ranked by price; landmarks summarized via Gemini ADK."
    }

    log_event("Aggregator::recommendation", recommendation)
    track_latency("aggregator", start)
    return recommendation


# ==== ADK Web UI Helper ====
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"
    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")
    baseURL = servers[0]["base_url"]
    path_parts = baseURL.split("/")
    kernel, token = path_parts[2], path_parts[3]
    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"
    display(HTML(f"<a href='{url}' target='_blank'>Open ADK Web UI â†—</a>"))
    return url_prefix

# ==== Define ADK Agent for GUI ====
root_agent = Agent(
    name="travel_concierge",
    model=Gemini(model="gemini-2.5-flash-lite", api_key=GOOGLE_API_KEY),
    description="Recommends hotels and landmarks in any city based on user queries.",
    instruction="Use hotel_agent and landmark_agent to answer travel queries. Provide concise, bullet-pointed responses.",
    tools=[google_search],  # you can register more tools if needed
)

runner = InMemoryRunner(agent=root_agent)

# ==== Launch ADK Web UI ====
!adk create travel-concierge --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY
url_prefix = get_adk_proxy_url()
!adk web --url_prefix {url_prefix}

