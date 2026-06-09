# ================================================================
# NOTEBOOK: France Route Planner + Long TSP + Sessions + Memory (Google ADK)
# + Optional Observability (Logs, Plugins)
# ================================================================

# ------------------------------------------------
# 1. Imports & API Key (Kaggle secrets)
# ------------------------------------------------
import os
import math
import uuid
import logging
from typing import Any, Dict, List

import requests
from kaggle_secrets import UserSecretsClient

from google.genai import types

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.apps.app import App
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.function_tool import FunctionTool

# >>> NEW: Memory imports
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory

# >>> NEW: Logging plugin imports
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest


print("âœ… Done.")


# ------------------------------------------------
# 1.1 Configure Gemini API key
# ------------------------------------------------
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key loaded from Kaggle secrets.")
except Exception as e:
    print(
        "ðŸ”‘ Authentication Error: please add the secret 'GOOGLE_API_KEY' "
        f"in Kaggle secrets. Details: {e}"
    )

# ------------------------------------------------
# 1.2 Logging configuration & cleanup
# ------------------------------------------------
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ðŸ§¹ Cleaned up {log_file}")

logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG if OBSERVABILITY_ENABLED else logging.INFO,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print(f"âœ… Logging configured (OBSERVABILITY_ENABLED={OBSERVABILITY_ENABLED})")

# ------------------------------------------------
# 1.3 Retry options & Model
# ------------------------------------------------
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

MODEL_NAME = "gemini-2.5-flash-lite"

# ------------------------------------------------
# 1.4 Sessions & persistence
# ------------------------------------------------
APP_NAME = "france_route_app"
USER_ID = "default_user"

db_url = "sqlite:///france_route_sessions.db"
session_service = DatabaseSessionService(db_url=db_url)
print("âœ… Session service with SQLite DB initialized.")

# >>> NEW: Memory service (in-memory for dev / course)
memory_service = InMemoryMemoryService()
print("âœ… InMemoryMemoryService initialized.")


# ------------------------------------------------
# 1.5 Example Plugin for observability
# ------------------------------------------------
class CountInvocationPlugin(BasePlugin):
    """A custom plugin that counts agent and LLM invocations."""

    def __init__(self) -> None:
        super().__init__(name="count_invocation")
        self.agent_count: int = 0
        self.llm_request_count: int = 0

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        self.agent_count += 1
        logging.info(f"[CountInvocationPlugin] Agent runs so far: {self.agent_count}")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        self.llm_request_count += 1
        logging.info(
            f"[CountInvocationPlugin] LLM requests so far: {self.llm_request_count}"
        )


# ------------------------------------------------
# 1.6 Helper: ask_agent_no_trace -> single final answer
# ------------------------------------------------
async def ask_agent_no_trace(
    runner: Runner,
    query: str,
    session_name: str = "default_session",
) -> str:
    """
    Sends a request to an ADK agent using sessions,
    but does NOT display intermediate responses (tool calls, etc.).

    Returns and prints the LAST non-empty text answer from the model.
    """
    # Create or fetch the session
    try:
        session = await session_service.create_session(
            app_name=runner.app_name, user_id=USER_ID, session_id=session_name
        )
    except Exception:
        session = await session_service.get_session(
            app_name=runner.app_name, user_id=USER_ID, session_id=session_name
        )

    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_answer: str | None = None

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            txt = event.content.parts[0].text
            if txt and txt != "None":
                final_answer = txt

    if final_answer:
        print(final_answer)
        return final_answer
    else:
        print("(No final text answer)")
        return ""


# >>> NEW: automatic memory saving callback
async def auto_save_to_memory(callback_context):
    """
    Automatically save the current session to memory after each agent turn.
    This mirrors the Day 3 course pattern using after_agent_callback.
    """
    try:
        inv_ctx = callback_context._invocation_context
        mem = getattr(inv_ctx, "memory_service", None)
        sess = getattr(inv_ctx, "session", None)
        if mem is not None and sess is not None:
            await mem.add_session_to_memory(sess)
    except Exception as e:
        # Fail silently to avoid breaking the agent if memory fails.
        print(f"[auto_save_to_memory] Warning: could not save session to memory: {e}")


# ================================================================
# 2. GEO: Multi-source geocoding (France + global)
# ================================================================

def _geocode_fr_gouv(address: str) -> Dict[str, Any]:
    """
    Geocode with api-adresse.data.gouv.fr (main, France).
    """
    try:
        url = "https://api-adresse.data.gouv.fr/search/"
        params = {"q": address, "limit": 1}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        if not features:
            return {"status": "error", "message": "No result (data.gouv.fr)"}
        f = features[0]
        coords = f["geometry"]["coordinates"]  # [lon, lat]
        lon = float(coords[0])
        lat = float(coords[1])
        label = f["properties"].get("label", address)
        return {
            "status": "success",
            "lat": lat,
            "lon": lon,
            "label": label,
            "source": "data_gouv_fr",
        }
    except Exception as e:
        return {"status": "error", "message": f"data.gouv.fr error: {e}"}


def _geocode_fr_nominatim(address: str) -> Dict[str, Any]:
    """
    Geocode with Nominatim (OpenStreetMap) restricted to France.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": address,
            "format": "json",
            "limit": 1,
            "countrycodes": "fr",
        }
        headers = {"User-Agent": "kaggle-adk-france-route-planner"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return {"status": "error", "message": "No result (Nominatim FR)"}
        item = data[0]
        lat = float(item["lat"])
        lon = float(item["lon"])
        label = item.get("display_name", address)
        return {
            "status": "success",
            "lat": lat,
            "lon": lon,
            "label": label,
            "source": "nominatim_fr",
        }
    except Exception as e:
        return {"status": "error", "message": f"Nominatim FR error: {e}"}


def _geocode_global_nominatim(address: str) -> Dict[str, Any]:
    """
    Geocode with global Nominatim (OpenStreetMap), NOT restricted to France.
    Used as a general fallback.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "json", "limit": 1}
        headers = {"User-Agent": "kaggle-adk-global-geocoder"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return {"status": "error", "message": "No result (Nominatim global)"}
        item = data[0]
        lat = float(item["lat"])
        lon = float(item["lon"])
        label = item.get("display_name", address)
        return {
            "status": "success",
            "lat": lat,
            "lon": lon,
            "label": label,
            "source": "nominatim_global",
        }
    except Exception as e:
        return {"status": "error", "message": f"Nominatim global error: {e}"}


def geocode_one_address(address: str) -> Dict[str, Any]:
    """
    Geocode an address using a FALLBACK CHAIN:
      1) api-adresse.data.gouv.fr (France)
      2) Nominatim France
      3) Global Nominatim
    """
    res1 = _geocode_fr_gouv(address)
    if res1["status"] == "success":
        return res1

    res2 = _geocode_fr_nominatim(address)
    if res2["status"] == "success":
        return res2

    res3 = _geocode_global_nominatim(address)
    if res3["status"] == "success":
        return res3

    return {
        "status": "error",
        "message": (
            f"Geocoding failed for '{address}'. "
            f"Details: [data.gouv.fr] {res1.get('message')} ; "
            f"[Nominatim FR] {res2.get('message')} ; "
            f"[Nominatim global] {res3.get('message')}"
        ),
    }


# ================================================================
# 3. DISTANCE + TSP (Travelling Salesman)
# ================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def build_distance_matrix(coords: List[Dict[str, float]]) -> List[List[float]]:
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0.0
            else:
                dist[i][j] = haversine_km(
                    coords[i]["lat"], coords[i]["lon"],
                    coords[j]["lat"], coords[j]["lon"],
                )
    return dist


def nearest_neighbor_tour(distance_matrix: List[List[float]], start: int = 0) -> List[int]:
    n = len(distance_matrix)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)

    current = start
    while unvisited:
        next_city = min(
            unvisited,
            key=lambda j: distance_matrix[current][j],
        )
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return tour


def two_opt(tour: List[int], distance_matrix: List[List[float]]) -> List[int]:
    def tour_length(t: List[int]) -> float:
        s = 0.0
        for i in range(len(t) - 1):
            s += distance_matrix[t[i]][t[i + 1]]
        return s

    improved = True
    best = tour[:]
    best_cost = tour_length(best)

    while improved:
        improved = False
        for i in range(1, len(best)):
            for j in range(i + 2, len(best) - 1):
                new_tour = best[:]
                new_tour[i:j] = reversed(new_tour[i:j])
                new_cost = tour_length(new_tour)
                if new_cost < best_cost:
                    best = new_tour
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break

    return best


def compute_tsp_route(coords: List[Dict[str, float]]) -> Dict[str, Any]:
    if not coords or len(coords) < 2:
        return {
            "status": "error",
            "message": "At least two points are required to optimize a route.",
        }

    dist_matrix = build_distance_matrix(coords)
    init_tour = nearest_neighbor_tour(dist_matrix, start=0)
    best_tour = two_opt(init_tour, dist_matrix)

    total_distance = 0.0
    legs = []
    for idx in range(len(best_tour) - 1):
        i = best_tour[idx]
        j = best_tour[idx + 1]
        d = dist_matrix[i][j]
        total_distance += d
        legs.append({"from_index": i, "to_index": j, "distance_km": d})

    return {
        "status": "success",
        "tour_indices": best_tour,
        "distance_matrix": dist_matrix,
        "legs": legs,
        "total_distance_km": total_distance,
    }


# ================================================================
# 4. STATE TOOLS FOR THE "SHORT" ROUTE SYSTEM
# ================================================================

def reset_route_state(tool_context: ToolContext) -> Dict[str, Any]:
    tool_context.state["route:addresses"] = []
    tool_context.state["route:client_name"] = ""
    tool_context.state["route:stops"] = []
    tool_context.state["route:solution"] = {}
    return {"status": "success"}


def save_route_addresses(
    tool_context: ToolContext,
    addresses_text: str,
    client_name: str = "",
) -> Dict[str, Any]:
    lines = [l.strip(" -â€¢\t") for l in addresses_text.split("\n")]
    addresses = [l for l in lines if l]

    tool_context.state["route:addresses"] = addresses
    if client_name:
        tool_context.state["route:client_name"] = client_name

    return {
        "status": "success",
        "count": len(addresses),
        "addresses": addresses,
    }


def geocode_route_addresses(tool_context: ToolContext) -> Dict[str, Any]:
    addresses = tool_context.state.get("route:addresses", [])
    if not addresses:
        return {
            "status": "error",
            "message": "No addresses to geocode (route:addresses is empty).",
        }

    stops = []
    errors = []
    for idx, addr in enumerate(addresses):
        geo = geocode_one_address(addr)
        if geo["status"] == "success":
            stops.append(
                {
                    "id": idx,
                    "address": addr,
                    "label": geo["label"],
                    "lat": geo["lat"],
                    "lon": geo["lon"],
                }
            )
        else:
            errors.append({"address": addr, "error": geo["message"]})

    if not stops:
        return {
            "status": "error",
            "message": "Geocoding failed for all addresses.",
            "errors": errors,
        }

    tool_context.state["route:stops"] = stops
    return {
        "status": "partial_success" if errors else "success",
        "stops_count": len(stops),
        "errors": errors,
    }


def run_tsp_for_current_route(tool_context: ToolContext) -> Dict[str, Any]:
    stops = tool_context.state.get("route:stops", [])
    if len(stops) < 2:
        return {
            "status": "error",
            "message": "Less than 2 geocoded points, optimization not possible.",
        }

    coords = [{"lat": s["lat"], "lon": s["lon"]} for s in stops]
    tsp_res = compute_tsp_route(coords)
    if tsp_res["status"] != "success":
        return tsp_res

    order = tsp_res["tour_indices"]
    ordered_stops = [stops[i] for i in order]

    solution = {
        "order_indices": order,
        "ordered_stops": ordered_stops,
        "legs": tsp_res["legs"],
        "total_distance_km": tsp_res["total_distance_km"],
    }

    tool_context.state["route:solution"] = solution
    return {"status": "success", "solution": solution}


def get_route_context(tool_context: ToolContext) -> Dict[str, Any]:
    return {
        "status": "success",
        "addresses": tool_context.state.get("route:addresses", []),
        "client_name": tool_context.state.get("route:client_name", ""),
        "stops": tool_context.state.get("route:stops", []),
        "solution": tool_context.state.get("route:solution", {}),
    }


# ================================================================
# 5. TOOLS FOR LONG TSP (jobs, history, events)
# ================================================================

def _get_jobs_state(tool_context: ToolContext) -> Dict[str, Any]:
    return tool_context.state.get("long_tsp:jobs", {})


def _save_jobs_state(tool_context: ToolContext, jobs: Dict[str, Any]) -> None:
    tool_context.state["long_tsp:jobs"] = jobs


def _get_current_job_id(tool_context: ToolContext) -> str:
    return tool_context.state.get("long_tsp:current_job_id", "")


def _set_current_job_id(tool_context: ToolContext, job_id: str) -> None:
    tool_context.state["long_tsp:current_job_id"] = job_id


def start_long_tsp_job_tool(
    tool_context: ToolContext,
    addresses_text: str,
    job_label: str = "",
) -> Dict[str, Any]:
    """
    Create a "long job": geocoding + TSP + initialization of tour state.
    """
    lines = [l.strip(" -â€¢\t") for l in addresses_text.split("\n")]
    addresses = [l for l in lines if l]

    if len(addresses) < 2:
        return {
            "status": "error",
            "message": "At least two addresses are required for a long job.",
        }

    # Geocode addresses
    stops = []
    errors = []
    for idx, addr in enumerate(addresses):
        geo = geocode_one_address(addr)
        if geo["status"] == "success":
            stops.append(
                {
                    "index": idx,
                    "address": addr,
                    "label": geo["label"],
                    "lat": geo["lat"],
                    "lon": geo["lon"],
                }
            )
        else:
            errors.append({"address": addr, "error": geo["message"]})

    if not stops:
        return {
            "status": "error",
            "message": "Geocoding failed for all addresses.",
            "errors": errors,
        }

    coords = [{"lat": s["lat"], "lon": s["lon"]} for s in stops]
    tsp_res = compute_tsp_route(coords)
    if tsp_res["status"] != "success":
        return tsp_res

    order = tsp_res["tour_indices"]
    legs = tsp_res["legs"]
    total_distance_km = tsp_res["total_distance_km"]

    job_id = f"JOB-{uuid.uuid4().hex[:8]}"
    if not job_label:
        job_label = f"long_tour_{job_id}"

    jobs = _get_jobs_state(tool_context)

    visited_indices = [order[0]]          # consider the first point as "already at departure"
    completed_legs: List[Dict[str, Any]] = []
    remaining_order = order[1:]

    job = {
        "job_id": job_id,
        "label": job_label,
        "addresses": addresses,
        "stops": stops,
        "order": order,
        "legs": legs,
        "total_distance_km": total_distance_km,
        "visited_indices": visited_indices,
        "completed_legs": completed_legs,
        "remaining_order": remaining_order,
        "events": [],
        "km_completed": 0.0,
    }

    jobs[job_id] = job
    _save_jobs_state(tool_context, jobs)
    _set_current_job_id(tool_context, job_id)

    return {
        "status": "success",
        "job_id": job_id,
        "job_label": job_label,
        "total_points": len(stops),
        "total_distance_km": total_distance_km,
        "geocode_errors": errors,
    }


def advance_on_current_job_tool(
    tool_context: ToolContext,
    current_position_text: str,
    mark_stop_completed: bool = True,
    include_previous_unvisited: bool = False,
) -> Dict[str, Any]:
    """
    From the current position, recompute a local tour toward remaining stops.

    IMPORTANT BEHAVIOR:
    - By default (include_previous_unvisited=False):
        We assume that all earlier stops in the global optimal order are
        definitively "done" (either visited or skipped and no longer needed).
        So we mark all stops BEFORE the current stop, plus the current stop,
        as visited/excluded.
    - If include_previous_unvisited=True:
        We assume the user skipped earlier stops but STILL wants to visit them.
        In that case, we only mark the current stop as visited, and we keep all
        earlier unvisited stops in the remaining list so they can be included
        in the new optimization.
    """
    job_id = _get_current_job_id(tool_context)
    jobs = _get_jobs_state(tool_context)
    if not job_id or job_id not in jobs:
        return {
            "status": "error",
            "message": "No active long job. Start a long optimization first.",
        }

    job = jobs[job_id]

    # Automatically mark the stop corresponding to current position
    if mark_stop_completed:
        geo_now = geocode_one_address(current_position_text)
        if geo_now["status"] == "success":
            lat_now = geo_now["lat"]
            lon_now = geo_now["lon"]
            best_i = None
            best_d = float("inf")
            for s in job["stops"]:
                d = haversine_km(lat_now, lon_now, s["lat"], s["lon"])
                if d < best_d:
                    best_d = d
                    best_i = s["index"]

            if best_i is not None and best_d < 0.5:
                order = job.get("order", [])
                if best_i in order:
                    pos = order.index(best_i)

                    if include_previous_unvisited:
                        # Case: user skipped earlier stops but still wants to visit them
                        # Mark ONLY the current stop as visited, keep earlier ones as remaining.
                        if best_i not in job["visited_indices"]:
                            job["visited_indices"].append(best_i)
                    else:
                        # Default behavior: everything before + current are considered "done"
                        for idx in order[:pos + 1]:
                            if idx not in job["visited_indices"]:
                                job["visited_indices"].append(idx)

    # Compute remaining stops (excluding visited ones)
    remaining = [idx for idx in job["order"] if idx not in job["visited_indices"]]
    if not remaining:
        jobs[job_id] = job
        _save_jobs_state(tool_context, jobs)
        return {
            "status": "done",
            "message": "All planned addresses have been visited.",
            "job_id": job_id,
            "km_completed": job["km_completed"],
        }

    # Re-geocode current position for local TSP
    geo_now = geocode_one_address(current_position_text)
    if geo_now["status"] != "success":
        jobs[job_id] = job
        _save_jobs_state(tool_context, jobs)
        return {
            "status": "error",
            "message": "Unable to geocode current position.",
        }

    current_point = {
        "index": -1,
        "address": current_position_text,
        "label": geo_now["label"],
        "lat": geo_now["lat"],
        "lon": geo_now["lon"],
    }
    remaining_stops = [job["stops"][i] for i in remaining]
    coords = [{"lat": current_point["lat"], "lon": current_point["lon"]}] + [
        {"lat": s["lat"], "lon": s["lon"]} for s in remaining_stops
    ]

    tsp_res = compute_tsp_route(coords)
    if tsp_res["status"] != "success":
        jobs[job_id] = job
        _save_jobs_state(tool_context, jobs)
        return tsp_res

    tour_indices = tsp_res["tour_indices"]
    legs = tsp_res["legs"]

    ordered_next_stops = []
    for idx in tour_indices:
        if idx == 0:
            continue
        s = remaining_stops[idx - 1]
        ordered_next_stops.append(s)

    job["remaining_order"] = [s["index"] for s in ordered_next_stops]

    jobs[job_id] = job
    _save_jobs_state(tool_context, jobs)

    return {
        "status": "success",
        "job_id": job_id,
        "current_position_label": current_point["label"],
        "next_stops": ordered_next_stops,
        "local_legs": legs,
    }


def get_visited_history_tool(tool_context: ToolContext) -> Dict[str, Any]:
    job_id = _get_current_job_id(tool_context)
    jobs = _get_jobs_state(tool_context)
    if not job_id or job_id not in jobs:
        return {
            "status": "error",
            "message": "No active long job.",
        }

    job = jobs[job_id]
    visited_indices = job.get("visited_indices", [])
    visited_stops = [job["stops"][i] for i in visited_indices]

    return {
        "status": "success",
        "job_id": job_id,
        "visited_stops": visited_stops,
    }


# def update_remaining_stops_tool(
#     tool_context: ToolContext,
#     action: str = "",
#     addresses_text: str = "",
# ) -> Dict[str, Any]:
#     """
#     Update remaining stops.
#     Policy:
#       - By default (or any action â‰  'replace_remaining_explicit'): APPEND
#       - Only for action='replace_remaining_explicit': full REPLACE
#     This guarantees that even if the LLM hallucinates action='replace_remaining',
#     we still APPEND by default and never wipe the existing list unintentionally.
#     """
#     job_id = _get_current_job_id(tool_context)
#     jobs = _get_jobs_state(tool_context)
#     if not job_id or job_id not in jobs:
#         return {
#             "status": "error",
#             "message": "No active long job.",
#         }

#     job = jobs[job_id]
#     visited_indices = job.get("visited_indices", [])

#     # Normalize action
#     action_norm = (action or "").strip().lower()

#     # REPLACE mode only for explicit action
#     is_replace = action_norm == "replace_remaining_explicit"
#     # All other cases (including 'replace_remaining', 'append_remaining', empty, etc.) => append
#     is_append = not is_replace

#     # Parse new address list
#     lines = [l.strip(" -â€¢\t") for l in (addresses_text or "").split("\n")]
#     new_addrs = [l for l in lines if l]

#     if not new_addrs:
#         if is_replace:
#             # Explicit replace with empty list => no remaining stops
#             job["order"] = visited_indices[:]
#     else:
#         new_stops = []
#         for addr in new_addrs:
#             geo = geocode_one_address(addr)
#             if geo["status"] == "success":
#                 new_stops.append(
#                     {
#                         "index": len(job["stops"]) + len(new_stops),
#                         "address": addr,
#                         "label": geo["label"],
#                         "lat": geo["lat"],
#                         "lon": geo["lon"],
#                     }
#                 )

#         job["stops"].extend(new_stops)
#         new_indices = [s["index"] for s in new_stops]

#         if is_replace:
#             # Explicit replacement: [already visited] + [new remaining]
#             job["order"] = visited_indices[:] + new_indices
#         elif is_append:
#             # Append (default)
#             job["order"].extend(new_indices)

#     # Recompute remaining indices
#     visited_indices = job.get("visited_indices", [])
#     remaining = [idx for idx in job["order"] if idx not in visited_indices]
#     job["remaining_order"] = remaining

#     jobs[job_id] = job
#     _save_jobs_state(tool_context, jobs)

#     return {
#         "status": "success",
#         "job_id": job_id,
#         "remaining_indices": remaining,
#     }




def append_new_stops_tool(tool_context: ToolContext, addresses_text: str) -> Dict[str, Any]:
    """
    Add new addresses to remaining stops (APPEND).
    Used for: "add", "update list", "from now on", etc.
    """
    return update_remaining_stops_tool(
        tool_context=tool_context,
        action="append_remaining",  # will be treated as APPEND
        addresses_text=addresses_text,
    )


def replace_remaining_stops_tool(tool_context: ToolContext, addresses_text: str) -> Dict[str, Any]:
    """
    Replace remaining stops with a new list (explicit REPLACE).
    Used only if the user explicitly requests a full replacement.
    """
    return update_remaining_stops_tool(
        tool_context=tool_context,
        action="replace_remaining_explicit",
        addresses_text=addresses_text,
    )

# NEW: delete stops tool (by index OR exact address OR exact label)
def remove_stops_tool(tool_context: ToolContext, addresses_or_indices_text: str) -> Dict[str, Any]:
    """
    Delete stops from the job.
    Input can be:
      - indices (one per line): 3
      - exact address lines
      - exact geocoder label lines
    Only unvisited stops are removed (visited stops are skipped).
    """
    return update_remaining_stops_tool(tool_context=tool_context, action="remove_stops", addresses_text=addresses_or_indices_text)


def add_route_event_tool(tool_context: ToolContext, description: str) -> Dict[str, Any]:
    job_id = _get_current_job_id(tool_context)
    jobs = _get_jobs_state(tool_context)
    if not job_id or job_id not in jobs:
        return {"status": "error", "message": "No active long job."}

    job = jobs[job_id]
    job.setdefault("events", [])
    job["events"].append(
        {
            "id": f"EVT-{uuid.uuid4().hex[:6]}",
            "description": description,
        }
    )
    jobs[job_id] = job
    _save_jobs_state(tool_context, jobs)
    return {"status": "success", "job_id": job_id, "event_count": len(job["events"])}


def get_km_and_events_summary_tool(tool_context: ToolContext) -> Dict[str, Any]:
    job_id = _get_current_job_id(tool_context)
    jobs = _get_jobs_state(tool_context)
    if not job_id or job_id not in jobs:
        return {"status": "error", "message": "No active long job."}

    job = jobs[job_id]
    visited_indices = job.get("visited_indices", [])
    legs = job.get("legs", [])
    total_dist = job.get("total_distance_km", 0.0)

    km_done = 0.0
    for leg in legs:
        i = leg["from_index"]
        j = leg["to_index"]
        if i in visited_indices and j in visited_indices:
            km_done += leg["distance_km"]
    job["km_completed"] = km_done
    jobs[job_id] = job
    _save_jobs_state(tool_context, jobs)

    return {
        "status": "success",
        "job_id": job_id,
        "km_completed": km_done,
        "km_total": total_dist,
        "events": job.get("events", []),
    }


def get_remaining_stops_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool: returns the list of REMAINING stops in planned order.
    Used for:
    - "Give me the list of addresses to visit"
    - "What addresses remain?"
    """
    job_id = _get_current_job_id(tool_context)
    jobs = _get_jobs_state(tool_context)
    if not job_id or job_id not in jobs:
        return {"status": "error", "message": "No active long job."}

    job = jobs[job_id]
    remaining_indices = job.get("remaining_order")

    if remaining_indices is None:
        visited = set(job.get("visited_indices", []))
        remaining_indices = [idx for idx in job.get("order", []) if idx not in visited]
        job["remaining_order"] = remaining_indices
        jobs[job_id] = job
        _save_jobs_state(tool_context, jobs)

    remaining_stops = [job["stops"][i] for i in remaining_indices]

    return {
        "status": "success",
        "job_id": job_id,
        "remaining_indices": remaining_indices,
        "remaining_stops": remaining_stops,
    }


# ================================================================
# 6. AGENTS: Planner, Geo, Route, Report for FranceRouteSystem
# ================================================================

def RouteOptimizerAgent(tool_context: ToolContext, payload: str = "") -> Dict[str, Any]:
    """
    Stub called if the LLM hallucinates a 'RouteOptimizerAgent' tool.
    It does nothing and returns an explicit error message.
    """
    return {
        "status": "error",
        "message": (
            "RouteOptimizerAgent is not a tool to call directly. "
            "Use only reset_route_state and save_route_addresses in this context."
        ),
    }


planner_agent = LlmAgent(
    name="PlannerAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
You are the PLANNER agent for route optimization in France.

Your role:
1. Understand the user's request (trucks, client, region, language, etc.).
2. Extract the addresses to visit from the user's text.
3. Call the `reset_route_state` tool at the beginning of a new tour.
4. Call the `save_route_addresses` tool with:
   - addresses_text: a single string containing all addresses,
     one per line (including the depot).
   - client_name if the client name is mentioned.

IMPORTANT:
- You MUST NOT call other tools (like RouteOptimizerAgent,
  RouteAgent, GeoAgent, etc.).
- Detailed optimization (TSP, distances) will be done by
  other agents in the chain.

If you need optimization, just prepare the data.

Language rule:
- Always respond in the same main language as the user's message.
  If the user writes in English, you respond in English.
  If the user writes in French, you respond in French.
""",
    tools=[
        FunctionTool(func=reset_route_state),
        FunctionTool(func=save_route_addresses),
        FunctionTool(func=RouteOptimizerAgent),  # stub
        load_memory,
    ],
    after_agent_callback=auto_save_to_memory,  # auto-save sessions to memory
)

geo_agent = LlmAgent(
    name="GeoAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
You are the GEO & DATA CLEANING agent.

Your role:
1. Check that tour addresses are ready in state
   (key `route:addresses`).
2. Call the `geocode_route_addresses` tool to geocode all addresses.
3. If some addresses fail, explain it but continue if possible.

Do not perform TSP yourself.

Language rule:
- Always respond in the same main language as the user's message.
""",
    tools=[
        FunctionTool(func=geocode_route_addresses),
    ],
    after_agent_callback=auto_save_to_memory,
)

route_agent = LlmAgent(
    name="RouteAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
You are the ROUTE OPTIMIZER agent.

Your role:
1. Use the `run_tsp_for_current_route` tool to compute a
   TSP approximate route from geocoded points (`route:stops`).
2. Do NOT try to recompute the order yourself, always delegate to the tool.

You can provide a short technical summary (total distance, number of points)
but the detailed report is the responsibility of the REPORT agent.

Language rule:
- Always respond in the same main language as the user's message.
""",
    tools=[
        FunctionTool(func=run_tsp_for_current_route),
    ],
    after_agent_callback=auto_save_to_memory,
)

report_agent = LlmAgent(
    name="ReportAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
You are the REPORT & EXPLANATION agent for tours in France.

1. Call the `get_route_context` tool to retrieve:
   - the list of addresses (`route:addresses`),
   - geocoded points (`route:stops`),
   - the TSP solution (`route:solution`).
2. Generate a clear report for the user:
   - optimized order of points (listing depot and clients),
   - estimated total distance,
   - optionally a rough time estimate (e.g. 50 km/h),
   - simple justification ("we start with X because it's closest to the depot", etc.).

Language rule:
- The language of the report MUST be:
  - French if the user's request is in French,
  - otherwise in the same main language as the request (for example, English
    if the user writes in English).
- Do NOT switch languages unless the user explicitly asks you to translate
  or to answer in another language.
""",
    tools=[
        FunctionTool(func=get_route_context),
        load_memory,
    ],
    after_agent_callback=auto_save_to_memory,
)

root_agent = SequentialAgent(
    name="FranceRouteSystem",
    sub_agents=[
        planner_agent,
        geo_agent,
        route_agent,
        report_agent,
    ],
)

# Here: empty list when observability is disabled
root_plugins = [LoggingPlugin(), CountInvocationPlugin()] if OBSERVABILITY_ENABLED else []

root_app = App(
    name="FranceRouteApp",
    root_agent=root_agent,
    plugins=root_plugins,
)

root_runner = Runner(
    app=root_app,
    session_service=session_service,
    memory_service=memory_service,  # <<< MEMORY ENABLED
)

print("âœ… Root multi-agent FranceRouteSystem initialized.")


# ================================================================
# 7. AGENT LongTspManager: long jobs + history + report
#    (automatic addition + protected update_remaining + language control + memory)
# ================================================================

long_tsp_manager_agent = LlmAgent(
    name="LongTspManagerAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
You are the LONG TOUR MANAGEMENT agent (LongTspManager) for France.

LANGUAGE RULE (VERY IMPORTANT):
- You must ALWAYS respond in the same main language as the user's message.
- If the user's request is in English, you respond in English.
- If the user's request is in French, you respond in French.
- Do NOT switch languages unless the user explicitly asks you to.
- Example:
  - User: "I am starting a long delivery route..." â†’ YOU ANSWER IN ENGLISH.
  - User: "Je lance une tournÃ©e longue..."       â†’ YOU ANSWER IN FRENCH.

ROUTING RULES (YOU MUST STRICTLY FOLLOW THEM):

1. When the user asks:
   - "What addresses are remaining?"
   - "Give me the list of addresses to visit"
   - "What do I still have to do?"

   â†’ DO NOT ask for their current position.
   â†’ DO NOT recompute a local TSP.
   â†’ CALL `get_remaining_stops_tool()`.

2. When the user provides a list of addresses to start a big tour:
   â†’ CALL `start_long_tsp_job_tool(addresses_text, job_label)`.

3. When the user provides their current position and asks for the best continuation
   (e.g. "compute the best next route", "what is my next stop?",
   "calculate the best remaining route from here"):

   DEFAULT CASE:
   â†’ CALL `advance_on_current_job_tool(current_position_text, mark_stop_completed=True)`
   â†’ This DEFAULT behavior assumes that all previous stops in the global route
     are now considered "done" (either visited or skipped and no longer needed).
     In other words, all stops BEFORE the current one, plus the current one,
     are excluded from future optimizations.

   SPECIAL CASE: the user clearly says that they SKIPPED previous stops BUT STILL
   WANTS TO VISIT THEM LATER. For example:
     - "I skipped the previous stops but I still want to visit them."
     - "J'ai sautÃ© les adresses prÃ©cÃ©dentes mais je veux quand mÃªme les faire."
   â†’ In that case, you MUST CALL:
       `advance_on_current_job_tool(current_position_text, mark_stop_completed=True, include_previous_unvisited=True)`
   â†’ This will mark ONLY the current stop as completed and KEEP earlier unvisited
     stops in the remaining list so they are included in the new optimization.

   SPECIAL CASE: the user clearly says that they SKIPPED previous stops and DOES NOT
   WANT TO VISIT THEM ANYMORE. For example:
     - "I skipped the previous stops and I don't want to visit them anymore."
     - "J'ai sautÃ© les adresses prÃ©cÃ©dentes et je ne veux plus les faire."
   â†’ In that case, use the DEFAULT call:
       `advance_on_current_job_tool(current_position_text, mark_stop_completed=True)`
   â†’ Earlier stops will be considered definitively done/excluded.

4. When they ask for the list of already visited addresses:
   â†’ CALL `get_visited_history_tool()`.

5. When they want to MODIFY REMAINING STOPS:

   âš  BY DEFAULT, you must assume they want to ADD addresses
   to the list of remaining stops, not replace it.

   - For all phrases such as:
     "Here is the new list of clients to deliver from now on",
     "Update the remaining stops according to this list",
     "Add these addresses",
     "Add these clients",
     "Complete the list with...",
     "I want to add new addresses",
     YOU MUST:
       â†’ CALL `append_new_stops_tool(addresses_text=...)`.

   - You must only use `replace_remaining_stops_tool(addresses_text=...)`
     if the user EXPLICITLY requests a full replacement, for example:
       "replace the remaining stops with...",
       "keep only these addresses",
       "remove all other stops and keep only these ones".

   IMPORTANT:
   - Even if you hallucinate a direct call to `update_remaining_stops_tool`,
     note that this tool is safe: it will APPEND by default
     and only fully replace remaining stops for
     action='replace_remaining_explicit'.

6. When they report an incident, delay, or event on the route:
   â†’ CALL `add_route_event_tool(description)`.

7. When they request a tour summary (km covered, total distance, events, etc.):
   â†’ CALL `get_km_and_events_summary_tool()`.

MEMORY RULES:
- You have access to a long-term MemoryService through:
  - `preload_memory` (automatic retrieval before each turn),
  - and the framework's automatic callback `auto_save_to_memory`.
- This means you can recall preferences or constraints the user mentioned
  in previous days or sessions (for example: "I drive an electric van",
  "avoid toll roads", "I prefer to visit hospitals at the end of the tour").
- You should use this information to better explain and justify your routing
  decisions when it is relevant.

GENERAL:
- Do not invent new tools.
- Always use the tools provided: start_long_tsp_job_tool, advance_on_current_job_tool,
  get_visited_history_tool, append_new_stops_tool, replace_remaining_stops_tool,
  add_route_event_tool, get_km_and_events_summary_tool, get_remaining_stops_tool,
  update_remaining_stops_tool (as a last resort).
- Only require the user's current position if they request optimization
  from their position.
- Always explain the answer in a simple and pedagogical way,
  in the same language as the user's request.
""",
    tools=[
        FunctionTool(func=start_long_tsp_job_tool),
        FunctionTool(func=advance_on_current_job_tool),
        FunctionTool(func=get_visited_history_tool),
        FunctionTool(func=append_new_stops_tool),
        FunctionTool(func=replace_remaining_stops_tool),
        FunctionTool(func=remove_stops_tool),  # âœ… NEW TOOL REGISTERED
        FunctionTool(func=add_route_event_tool),
        FunctionTool(func=get_km_and_events_summary_tool),
        FunctionTool(func=get_remaining_stops_tool),
        FunctionTool(func=update_remaining_stops_tool),
        preload_memory,
    ],
    after_agent_callback=auto_save_to_memory,  # auto-save every turn to memory
)

# Same here: empty plugin list if observability is disabled
long_plugins = [LoggingPlugin(), CountInvocationPlugin()] if OBSERVABILITY_ENABLED else []

long_app = App(
    name="FranceLongTspApp",
    root_agent=long_tsp_manager_agent,
    plugins=long_plugins,
)

long_runner = Runner(
    app=long_app,
    session_service=session_service,
    memory_service=memory_service,  # <<< MEMORY ENABLED HERE TOO
)

print("âœ… LongTspManagerAgent initialized (auto add + safe update_remaining + memory support).")


