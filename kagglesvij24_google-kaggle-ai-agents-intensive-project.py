pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")


def show_python_code_and_result(response):
    for i in range(len(response)):
        # Check if the response contains a valid function call result from the code executor
        if (
            (response[i].content.parts)
            and (response[i].content.parts[0])
            and (response[i].content.parts[0].function_response)
            and (response[i].content.parts[0].function_response.response)
        ):
            response_code = response[i].content.parts[0].function_response.response
            if "result" in response_code and response_code["result"] != "```":
                if "tool_code" in response_code["result"]:
                    print(
                        "Generated Python Code >> ",
                        response_code["result"].replace("tool_code", ""),
                    )
                else:
                    print("Generated Python Response >> ", response_code["result"])


print("âœ… Helper functions defined.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


!pip install googlemaps python-dotenv



import os
from dotenv import load_dotenv

 
os.environ["GOOGLE_MAPS_API_KEY"] = ""




from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters, StdioConnectionParams



google_maps_api_key=""


from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

maps_toolset = McpToolset(

    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="path/to/google-maps-cli",
            args=["", google_maps_api_key]
        )
    )
)



!pip install googlemaps



import googlemaps

gmaps = googlemaps.Client(key="")



gmaps = googlemaps.Client(key="")



def find_places_nearby(query, location="London, UK", radius=5000, type=None):
    geocode_result = gmaps.geocode(location)
    if not geocode_result:
        return f"Could not geocode location: {location}"
    
    latlng = geocode_result[0]['geometry']['location']
    places = gmaps.places_nearby(location=(latlng['lat'], latlng['lng']),
                                 radius=radius,
                                 keyword=query,
                                 type=type)
    
    results = places.get('results', [])
    if not results:
        return f"No places found for '{query}' near {location}."
    
    return [
        {
            "name": place["name"],
            "address": place.get("vicinity"),
            "rating": place.get("rating")
        }
        for place in results[:5]
    ]



import requests
import os

# Set my API key
api_key = os.environ.get("") or ""

def find_places_nearby(query, location="London", radius=5000):
    # Step 1: Geocode the location
    geo_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={api_key}"
    geo_resp = requests.get(geo_url).json()
    if not geo_resp.get("results"):
        return f"Could not geocode location: {location}"
    
    latlng = geo_resp["results"][0]["geometry"]["location"]

    # Step 2: Use Places API (New)
    places_url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating"
    }
    payload = {
        "location": {
            "latitude": latlng["lat"],
            "longitude": latlng["lng"]
        },
        "radius": radius,
        "includedTypes": ["restaurant"],
        "keyword": query
    }

    resp = requests.post(places_url, headers=headers, json=payload).json()
    return [
        {
            "name": p["displayName"]["text"],
            "address": p["formattedAddress"],
            "rating": p.get("rating", "N/A")
        }
        for p in resp.get("places", [])
    ]



results = find_places_nearby("Italian restaurant", location="London")
for i, place in enumerate(results, 1):
    print(f"{i}. {place['name']} - {place['address']} (Rating: {place.get('rating', 'N/A')})")



os.environ["GOOGLE_MAPS_API_KEY"] = ""
api_key = os.environ.get("GOOGLE_MAPS_API_KEY")



def maps_assistant_agent(user_query, location="London", radius=5000):
    # Step 1: Geocode
    geo_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={api_key}"
    geo_resp = requests.get(geo_url).json()
    latlng = geo_resp["results"][0]["geometry"]["location"]

    # Step 2: Nearby search
    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{latlng['lat']},{latlng['lng']}",
        "radius": radius,
        "keyword": user_query,
        "type": "restaurant",
        "key": api_key
    }
    resp = requests.get(places_url, params=params).json()
    ...



user_query = "Italian restaurant"



radius = 5000  # or whatever value you want
params = {
    "location": f"{latlng['lat']},{latlng['lng']}",
    "radius": radius,
    "keyword": user_query,
    "type": "restaurant",
    "key": api_key
}



places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
params = {
    "location": f"{latlng['lat']},{latlng['lng']}",
    "radius": radius,
    "keyword": user_query,
    "type": "restaurant",
    "key": api_key
}
resp = requests.get(places_url, params=params).json()



def maps_assistant_agent(user_query, location="London", radius=5000):
    geo_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={api_key}"
    geo_resp = requests.get(geo_url).json()
    if not geo_resp.get("results"):
        return f"Could not geocode location: {location}"
    
    latlng = geo_resp["results"][0]["geometry"]["location"]

    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{latlng['lat']},{latlng['lng']}",
        "radius": radius,
        "keyword": user_query,
        "type": "restaurant",
        "key": api_key
    }
    resp = requests.get(places_url, params=params).json()
    places = resp.get("results", [])
    if not places:
        return f"No results found for '{user_query}' near {location}."

    response_lines = []
    for i, place in enumerate(places[:5], 1):
        name = place["name"]
        address = place.get("vicinity", "Address not available")
        rating = place.get("rating", "N/A")
        response_lines.append(f"{i}. {name} â€” {address} (Rating: {rating})")

    return "\n".join(response_lines)



import requests

api_key = ""
location = "London"
radius = 5000
user_query = "Italian restaurant"

# Step 1: Geocode location
geo_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={api_key}"
geo_resp = requests.get(geo_url).json()
latlng = geo_resp["results"][0]["geometry"]["location"]

# Step 2: Nearby search
places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
params = {
    "location": f"{latlng['lat']},{latlng['lng']}",
    "radius": radius,
    "keyword": user_query,
    "type": "restaurant",
    "key": api_key
}
resp = requests.get(places_url, params=params).json()

# Step 3: Display results
places = resp.get("results", [])
for i, place in enumerate(places[:5], 1):
    name = place["name"]
    address = place.get("vicinity", "Address not available")
    rating = place.get("rating", "N/A")
    print(f"{i}. {name} â€” {address} (Rating: {rating})")


