
# Install required packages (Colab only; Kaggle often has these preinstalled)
!pip install --quiet google-generativeai folium pandas matplotlib geopy pillow streamlit streamlit-folium
print("Install step complete (or already installed).")




import os
# --- GEMINI API KEY (embedded as requested) ---
os.environ['GEMINI_API_KEY'] = "insert your api key"   # <<-- YOUR API KEY (keep private)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Safety flag: set to True only after you intentionally want to run live API calls
ENABLE_GENIE = False

print("Gemini key present:", ("YES" if GEMINI_API_KEY and "YOUR" not in GEMINI_API_KEY else "NO"))
print("ENABLE_GENIE:", ENABLE_GENIE)




# Example parameters (edit if you want)
destination = "Paris"
days = 3
month = "April"
budget_level = "medium"
interests = ["museums", "cafes", "architecture", "art"]
traveler_type = "solo student traveler"




import json, pandas as pd, matplotlib.pyplot as plt, folium, webbrowser, datetime, os
from IPython.display import display, HTML
from geopy.geocoders import Nominatim
try:
    import google.generativeai as genai
except Exception:
    genai = None

geolocator = Nominatim(user_agent="trip_smart_notebook")

# Demo Paris itinerary JSON (used when no API key provided)
sample_itinerary = {
  "trip_summary": "A relaxed 3-day Paris itinerary focused on art, architecture, and cafe culture.",
  "estimated_daily_budget_usd": {"min": 60, "max": 110, "comment": "Moderate: museums, cafes, and occasional transport."},
  "day_plans": [
    {"day": 1, "theme": "Classic Paris",
     "morning": {"title":"Louvre Highlights","description":"Short highlights: Mona Lisa, Egyptian rooms.","address_or_area":"Louvre Museum","approx_cost_usd":20,"travel_tips":"Buy timed tickets online."},
     "afternoon": {"title":"Seine Walk & Île de la Cité","description":"Stroll by the river, visit Île de la Cité.","address_or_area":"Île de la Cité","approx_cost_usd":20,"travel_tips":"Pack a small picnic."},
     "evening": {"title":"Eiffel Tower View","description":"Sunset photo spots from Trocadéro.","address_or_area":"Eiffel Tower","approx_cost_usd":10,"travel_tips":"Avoid peak hours for photos."}},
    {"day": 2, "theme": "Montmartre & Museums",
     "morning": {"title":"Montmartre Walk","description":"Artists’ square and Sacré-Cœur.","address_or_area":"Montmartre","approx_cost_usd":0,"travel_tips":"Wear comfortable shoes."},
     "afternoon": {"title":"Musée d'Orsay","description":"Impressionist masterpieces.","address_or_area":"Musée d'Orsay","approx_cost_usd":16,"travel_tips":"Check exhibit timings."},
     "evening": {"title":"Seine Dinner","description":"Optional dinner cruise or bistro meal.","address_or_area":"Seine River","approx_cost_usd":35,"travel_tips":"Reserve in advance."}},
    {"day": 3, "theme": "Hidden Gems",
     "morning": {"title":"Le Marais & Boutiques","description":"Explore narrow streets and galleries.","address_or_area":"Le Marais","approx_cost_usd":10,"travel_tips":"Look for small artisan shops."},
     "afternoon": {"title":"Luxembourg Gardens Picnic","description":"Relax at the gardens.","address_or_area":"Luxembourg Gardens","approx_cost_usd":15,"travel_tips":"Bring pastries from a local boulangerie."},
     "evening": {"title":"Latin Quarter Night","description":"Jazz bars and student cafes.","address_or_area":"Latin Quarter","approx_cost_usd":20,"travel_tips":"Check music schedules."}}
  ],
  "hotel_suggestions": [
    {"area":"Saint-Germain","hotel_type":"mid-range","why_here":"Central and charming","approx_price_range_usd":[80,150]},
    {"area":"Le Marais","hotel_type":"mid-range","why_here":"Historic and walkable","approx_price_range_usd":[70,130]}
  ],
  "local_tips": ["Buy a Museum Pass for discounts","Use metro for efficient travel","Carry a small umbrella"]
}

def itinerary_to_dataframe(itinerary_json):
    rows = []
    for day in itinerary_json.get("day_plans", []):
        for tod in ["morning","afternoon","evening"]:
            b = day.get(tod, {})
            rows.append({
                "Day": day.get("day"),
                "Theme": day.get("theme", ""),
                "TimeOfDay": tod.title(),
                "Title": b.get("title", ""),
                "Area/Address": b.get("address_or_area", ""),
                "ApproxCostUSD": b.get("approx_cost_usd", None),
                "Tips": b.get("travel_tips", ""),
            })
    return pd.DataFrame(rows)

def hotels_to_dataframe(itinerary_json):
    rows = []
    for h in itinerary_json.get("hotel_suggestions", []):
        price = h.get("approx_price_range_usd", [None, None])
        rows.append({
            "Area": h.get("area", ""),
            "Type": h.get("hotel_type", ""),
            "WhyHere": h.get("why_here", ""),
            "PriceMin": price[0],
            "PriceMax": price[1],
        })
    return pd.DataFrame(rows)

def create_budget_chart(itinerary_json):
    days = [f"Day {d['day']}" for d in itinerary_json.get("day_plans", [])]
    costs = [sum([d[t].get("approx_cost_usd", 0) for t in ["morning","afternoon","evening"]]) for d in itinerary_json.get("day_plans", [])]
    plt.figure(figsize=(6,3.5))
    plt.bar(days, costs)  # matplotlib used, no explicit colors set
    plt.title("Estimated Daily Costs (USD)")
    plt.xlabel("Day"); plt.ylabel("USD")
    plt.tight_layout()
    plt.show()

def create_map(itinerary_json, destination="Paris"):
    coords = {
        "Louvre Museum": (48.8606,2.3376),
        "Île de la Cité": (48.8530,2.3499),
        "Eiffel Tower": (48.8584,2.2945),
        "Montmartre": (48.8867,2.3431),
        "Musée d'Orsay": (48.8600,2.3266),
        "Le Marais": (48.8570,2.3572),
        "Luxembourg Gardens": (48.8462,2.3372),
        "Latin Quarter": (48.8493,2.3470),
        "Seine River": (48.8566,2.3522)
    }
    m = folium.Map(location=[48.8566,2.3522], zoom_start=12)
    for d in itinerary_json.get("day_plans", []):
        for tod in ["morning","afternoon","evening"]:
            b = d.get(tod,{})
            a = b.get("address_or_area","")
            coord = coords.get(a)
            if coord:
                folium.Marker(location=coord, popup=f"Day {d['day']} {tod.title()}: {b.get('title','')}").add_to(m)
    fname = f"trip_smart_{destination.replace(' ','_').lower()}_map.html"
    m.save(fname)
    print(f"Map saved as {fname} (open from Files panel or download).")

# Gemini live wrapper (gated)
def generate_with_gemini(destination, days, month, budget_level, interests, traveler_type):
    key = os.environ.get("GEMINI_API_KEY","YOUR_GEMINI_API_KEY")
    if not (key and "YOUR" not in key):
        raise RuntimeError("No valid Gemini API key found. Set os.environ['GEMINI_API_KEY'] and ENABLE_GENIE = True to run live.")
    if genai is None:
        raise RuntimeError("google-generativeai library not available.")
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    SYSTEM_INSTRUCTIONS = """You are an expert travel planner. Return valid JSON following the notebook schema."""
    user_prompt = f"Destination: {destination}\nMonth: {month}\nDays: {days}\nBudget: {budget_level}\nInterests: {', '.join(interests)}\nTraveler: {traveler_type}\nRequirements: Provide a JSON itinerary with day plans, hotels, and local tips."
    resp = model.generate_content([{"role":"system","parts":[SYSTEM_INSTRUCTIONS]},{"role":"user","parts":[user_prompt]}], generation_config={"temperature":0.7, "max_output_tokens":1200})
    txt = resp.text
    try:
        return json.loads(txt)
    except Exception:
        s = txt.find('{'); e = txt.rfind('}'); cleaned = txt[s:e+1]
        return json.loads(cleaned)




# Demo mode: use sample itinerary so judges can run without an API key
itinerary = sample_itinerary
print("\n✨ TRIP SUMMARY ✨\n", itinerary.get("trip_summary"))
print("\nEstimated daily budget:", itinerary.get("estimated_daily_budget_usd"))

# Chart
create_budget_chart(itinerary)

# Tables
df_plan = itinerary_to_dataframe(itinerary)
print("\nDAY-BY-DAY PLAN:")
display(df_plan)

df_hotels = hotels_to_dataframe(itinerary)
print("\nHOTEL SUGGESTIONS:")
display(df_hotels)

print("\nLOCAL TIPS:")
for t in itinerary.get("local_tips", []):
    print("-", t)

# Map
create_map(itinerary, destination)




# Gemini live generation (ONLY run if you set ENABLE_GENIE = True and provided a valid GEMINI_API_KEY)
if 'ENABLE_GENIE' in globals() and ENABLE_GENIE:
    try:
        print("Running Gemini live generation...")
        live_itinerary = generate_with_gemini(destination, days, month, budget_level, interests, traveler_type)
        print("Generated itinerary summary:", live_itinerary.get("trip_summary"))
        create_budget_chart(live_itinerary)
        display(itinerary_to_dataframe(live_itinerary))
        display(hotels_to_dataframe(live_itinerary))
        create_map(live_itinerary, destination)
    except Exception as e:
        print("Gemini generation failed or was not enabled:", e)
else:
    print("Gemini live mode not enabled. Set ENABLE_GENIE = True and provide a valid GEMINI_API_KEY to run.")




# Ask TripSmart AI (simple conversational cell). Works with Gemini if ENABLE_GENIE True.
def ask_trip_smart(query):
    if 'ENABLE_GENIE' in globals() and ENABLE_GENIE:
        try:
            bot = genai.GenerativeModel("gemini-1.5-flash")
            system = "You are TripSmart, an expert travel assistant. Keep answers concise and actionable."
            resp = bot.generate_content([{"role":"system","parts":[system]},{"role":"user","parts":[query]}], generation_config={"temperature":0.7, "max_output_tokens":500})
            return resp.text
        except Exception as e:
            return f"Error calling Gemini: {e}"
    else:
        # Demo safe fallback answer
        return "Demo mode: TripSmart suggests visiting local museums early in the morning to avoid crowds."
# Example usage:
print(ask_trip_smart('Recommend a 2-hour morning plan near the Louvre for a solo traveler.'))




# Export the current itinerary (itinerary variable) to a printable HTML
def itinerary_to_html(itinerary, title="TRIP SMART Itinerary"):
    html = f"""<html><head><meta charset='utf-8'><title>{title}</title>
    <style>body{{font-family:Arial,Helvetica,sans-serif; padding:20px; max-width:900px; margin:auto;}}
    h1{{background: linear-gradient(90deg,#0b0f12,#001f1c); color:#00ffd5; padding:10px; border-radius:6px;}}
    .day{{border-bottom:1px solid #eee; padding:8px 0;}}
    table{{width:100%; border-collapse: collapse;}}
    th, td{{border:1px solid #ddd; padding:8px; text-align:left;}}
    </style></head><body>
    <h1>{title}</h1>
    <h3>Trip summary</h3><p>{itinerary.get('trip_summary')}</p>
    <h3>Estimated daily budget</h3><p>{itinerary.get('estimated_daily_budget_usd')}</p>
    <h3>Day by day</h3>
    """
    for d in itinerary.get('day_plans', []):
        html += f"<div class='day'><h4>Day {d['day']} — {d.get('theme','')}</h4>"
        for tod in ['morning','afternoon','evening']:
            b = d.get(tod,{})
            html += f"<strong>{tod.title()}:</strong> {b.get('title','')} — {b.get('description','')}<br><em>Area:</em> {b.get('address_or_area','')} — <em>Cost:</em> {b.get('approx_cost_usd','')}<br><small>Tip: {b.get('travel_tips','')}</small><br><br>"
        html += "</div>"
    html += "<h3>Hotel suggestions</h3><table><tr><th>Area</th><th>Type</th><th>Price Range</th></tr>"
    for h in itinerary.get('hotel_suggestions', []):
        pr = h.get('approx_price_range_usd',[None,None])
        html += f"<tr><td>{h.get('area')}</td><td>{h.get('hotel_type')}</td><td>{pr[0]} - {pr[1]}</td></tr>"
    html += "</table><h3>Local tips</h3><ul>"
    for t in itinerary.get('local_tips',[]): html += f"<li>{t}</li>"
    html += "</ul></body></html>"
    return html

# Write file
html = itinerary_to_html(sample_itinerary, title=f"TRIP SMART — {destination} Itinerary")
with open("TRIP_SMART_itinerary.html","w", encoding="utf-8") as f:
    f.write(html)
print("Saved TRIP_SMART_itinerary.html — download from Files panel and open in browser to print/save as PDF.")


