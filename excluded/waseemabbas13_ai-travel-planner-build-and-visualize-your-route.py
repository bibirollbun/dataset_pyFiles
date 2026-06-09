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


!pip install -q -U google-generativeai duckduckgo-search


!pip install -q -U google-generativeai duckduckgo-search folium geopy


!pip install -q -U google-generativeai duckduckgo-search folium geopy polyline


import os
import google.generativeai as genai
from duckduckgo_search import DDGS
from kaggle_secrets import UserSecretsClient

# Access the secret key
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)


# Cell 3: Logistics, Mapping & Routing Tools
from duckduckgo_search import DDGS
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
import requests
import polyline

# --- Global Storage ---
map_pins = []
map_routes = []

# --- Tool 1: Pin a Location ---
def add_to_map(location_name: str, category: str, day: str, details: str = ""):
    """
    Pins a specific spot or hotel to the map.
    For Alternative Hotels, set category="Hotel_Alt" and day="Base".
    """
    map_pins.append({
        "name": location_name,
        "category": category,
        "day": day,
        "details": details
    })
    return f"Pinned {location_name}."

# --- Tool 2: Add a Transport Route ---
def add_transport_leg(start_location: str, end_location: str, mode: str, cost: str, duration: str):
    """
    Draws a route. IMPORTANT: 'cost' and 'duration' should contain COMPARISONS (e.g. Taxi vs Metro).
    """
    map_routes.append({
        "start": start_location,
        "end": end_location,
        "mode": mode,
        "cost": cost,
        "duration": duration
    })
    return f"Route added: {start_location} to {end_location}."

# --- Tool 3: Search Logistics ---
def search_itinerary_details(query_type: str, location: str, detail_param: str = "", origin: str = ""):
    """
    Master search tool for prices and logistics.
    """
    search_query = ""
    if query_type == "best_time": search_query = f"best time to visit {location} weather"
    elif query_type == "flights": search_query = f"flight price from {origin} to {location} {detail_param}"
    elif query_type == "attractions": search_query = f"top tourist attractions {location} blog"
    elif query_type == "nearby_hotel": search_query = f"hotel near {location} {detail_param} price"
    elif query_type == "transport": search_query = f"transport cost taxi vs public transport from {location} to {detail_param}"

    print(f"ğŸ•µï¸� Agent searching ({query_type}): {search_query}...") 
    try:
        # max_results=5 to ensure we find alternative hotels
        results = DDGS().text(keywords=search_query, max_results=5)
        return str(results)
    except:
        return "Search failed."

# Register ALL tools
tools_list = [search_itinerary_details, add_to_map, add_transport_leg]


# Cell 4: Initialize the Master Architect Agent

model_name = 'gemini-2.0-flash' 

model = genai.GenerativeModel(
    model_name=model_name,
    tools=tools_list,
    system_instruction="""
    You are an Elite Travel Architect. Your goal is to build a **Visual Itinerary**.
    
    **MAPPING INSTRUCTIONS:**
    1. **Pin Spots:** Use `add_to_map`.
       * The `day` parameter MUST be "Day 1", "Day 2", etc.
       * For the Main Hotel: `category`="Hotel", `day`="Base".
       * **NEW:** Find 3 Alternative Hotels nearby and pin them with `category`="Hotel_Alt", `day`="Base".
       
    2. **Draw Routes:** Use `add_transport_leg` to connect ONLY the Main Hotel and daily spots.
       * **CRITICAL:** In the `cost` and `duration` fields, you MUST provide a **Comparison**.
       * Example Cost: "Taxi: $15 | Metro: $2"
       * Example Duration: "Taxi: 10m | Metro: 25m"
    
    **EXECUTION:**
    1. Plan the trip using the Main Hotel.
    2. Pin the Main Hotel, 3 Alternatives, and all Spots.
    3. Draw the route connections with comparison data.
    4. Output a detailed Markdown report.
    """
)

chat = model.start_chat(enable_automatic_function_calling=True)
print(f"âœ… Master Agent Initialized!")


# Cell 5: Run Agent & Draw Custom "Day - X" Map (Stable Version)

import time
from google.api_core import exceptions
from IPython.display import Markdown
from folium.features import DivIcon
import requests
import polyline
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- CONFIGURATION ---
destination = "Rome, Italy"
origin = "New York"
mode = "Economy"
days = 3
# ---------------------

# 1. Clear previous data
map_pins = []
map_routes = []

print(f"ğŸš€ Agent is designing the Master Plan for {destination}...")

user_prompt = f"""
Plan a {days}-day trip to {destination} from {origin} in {mode} mode.
1. **Report:** Full financial breakdown and itinerary.
2. **Map:** Connect Primary Hotel -> Spot 1 -> Spot 2 for each day using `add_transport_leg`.
3. **Comparison:** For every transport leg, include costs for BOTH Taxi and Public Transport.
4. **Alternatives:** Find and pin 3 alternative hotels in the same area.
"""

# 2. AUTOMATIC RETRY LOGIC
response = None
max_retries = 3

for attempt in range(max_retries):
    try:
        response = chat.send_message(user_prompt)
        print("âœ… Agent finished successfully!")
        break 
    except exceptions.ResourceExhausted:
        wait_time = 40 
        print(f"âš ï¸� Speed Limit Hit! Cooling down for {wait_time}s...")
        time.sleep(wait_time)
    except Exception as e:
        print(f"â�Œ Unexpected Error: {e}")
        break

# 3. Render Output
if response:
    display(Markdown(response.text))

    print(f"\nğŸ—ºï¸� Generating Interactive Map...")

    try:
        # --- FIX: Added 'timeout=10' to prevent ReadTimeoutError ---
        geolocator = Nominatim(user_agent="travel_agent_final_stable", timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        
        location_coords = geocode(destination)
        if location_coords:
            my_map = folium.Map(location=[location_coords.latitude, location_coords.longitude], zoom_start=13)

            day_colors = {"Day 1": "#007bff", "Day 2": "#28a745", "Day 3": "#6f42c1", "Day 4": "#fd7e14", "Base": "#dc3545"}

            # --- DRAW PINS ---
            for spot in map_pins:
                try:
                    loc = geocode(spot["name"])
                    if loc:
                        coord = [loc.latitude, loc.longitude]
                        day = spot["day"]
                        category = spot.get("category", "")
                        color = day_colors.get(day, "black")
                        
                        day_label = day.upper().replace("DAY", "DAY -")

                        # A. TINY DOTS for Alternative Hotels
                        if category == "Hotel_Alt":
                             folium.CircleMarker(
                                location=coord,
                                radius=5,          
                                color="gray",      
                                fill=True,
                                fill_color="white",
                                fill_opacity=1,
                                popup=f"<b>Alternative Option</b><br>{spot['name']}",
                                tooltip=f"ğŸ�¨ Alt: {spot['name']}"
                            ).add_to(my_map)

                        # B. MAIN PINS
                        else:
                            if day == "Base":
                                html = f"""
                                <div style="
                                    background-color: white; border: 2px solid {color}; color: {color};
                                    border-radius: 8px; padding: 2px 6px; font-size: 10px; font-weight: bold;
                                    box-shadow: 2px 2px 4px rgba(0,0,0,0.3); white-space: nowrap;">
                                    ğŸ�¨ {spot['name']}
                                </div>"""
                                icon_size = (100, 30)
                            else:
                                html = f"""
                                <div style="
                                    background-color: {color}; color: white;
                                    border-radius: 15px; 
                                    padding: 5px 10px;
                                    text-align: center; font-weight: bold; font-size: 11px;
                                    border: 2px solid white;
                                    box-shadow: 3px 3px 6px rgba(0,0,0,0.4);
                                    white-space: nowrap; font-family: sans-serif;">
                                    {day_label}
                                </div>
                                <div style="
                                    position: absolute; top: -25px; left: 50%; transform: translateX(-50%);
                                    background: rgba(255,255,255,0.9); padding: 2px 5px; border-radius: 4px;
                                    font-size: 9px; font-weight: bold; border: 1px solid #ccc;
                                    white-space: nowrap;">
                                    {spot['name']}
                                </div>
                                """
                                icon_size = (80, 40)

                            folium.Marker(
                                location=coord,
                                icon=DivIcon(icon_size=icon_size, icon_anchor=(40, 20), html=html),
                                tooltip=f"<b>{spot['name']}</b>"
                            ).add_to(my_map)
                except: pass

            # --- DRAW ROUTES ---
            for route in map_routes:
                try:
                    loc_start = geocode(route["start"])
                    loc_end = geocode(route["end"])
                    
                    if loc_start and loc_end:
                        # Use OSRM for routing
                        url = f"http://router.project-osrm.org/route/v1/driving/{loc_start.longitude},{loc_start.latitude};{loc_end.longitude},{loc_end.latitude}?overview=full"
                        # Added timeout here too just in case
                        res = requests.get(url, timeout=10).json()
                        
                        if "routes" in res:
                            geometry = res["routes"][0]["geometry"]
                            decoded_points = polyline.decode(geometry)
                            
                            # Blue Route Line
                            folium.PolyLine(decoded_points, color="#007bff", weight=4, opacity=0.8, dash_array='10, 10').add_to(my_map)
                            
                            # Minimal Bus Icon
                            midpoint = decoded_points[len(decoded_points) // 2]
                            tooltip_html = f"""
                            <div style="font-family: sans-serif; font-size: 12px; min-width: 180px;">
                                <b>ğŸš† Transport Options</b><hr style="margin:5px 0;">
                                <b>Leg:</b> {route['start']} â�” {route['end']}<br>
                                <b>â�±ï¸� Time:</b> {route.get('duration', '?')}<br>
                                <b>ğŸ’µ Cost:</b> {route.get('cost', '?')}<br>
                            </div>
                            """
                            mini_box_html = f"""
                            <div style="background-color: white; border: 1px solid #007bff; border-radius: 50%; width: 24px; height: 24px; text-align: center; line-height: 24px; font-size: 14px; box-shadow: 2px 2px 4px rgba(0,0,0,0.3); cursor: help; color: #007bff;">ğŸšŒ</div>
                            """
                            
                            folium.Marker(
                                location=midpoint,
                                icon=DivIcon(icon_size=(24, 24), icon_anchor=(12, 12), html=mini_box_html),
                                tooltip=tooltip_html 
                            ).add_to(my_map)

                except Exception as e:
                    print(f"Route error: {e}")

            display(my_map)
            my_map.save("Travel_Plan_Map.html")
            print("âœ… Map saved to output.")
        else:
            print("â�Œ Destination coordinates not found.")
    except Exception as e:
        print(f"â�Œ Map Error: {e}")
else:
    print("â�Œ Failed to get response after retries.")

