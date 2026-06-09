%pip install --upgrade geopandas shapely fiona pyproj rtree pyarrow scipy


%pip install --upgrade bigframes google-cloud-automl google-cloud-translate google-ai-generativelanguage tensorflow 


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)


project_id = 'big-query-project-472510' #@param{type:"string"}
import bigframes.pandas as bpd
bpd.options.bigquery.ordering_mode = "partial" # Optional: partial ordering mode can accelerate executions and save costs

import bigframes.exceptions
import warnings
warnings.filterwarnings("ignore", category=bigframes.exceptions.AmbiguousWindowWarning)


shape_dir = "/kaggle/input/10000-crop-field-boundaries-across-india"


# 
from pathlib import Path
import geopandas as gpd
from pyproj import Geod

# === 1) CONFIG ===
# Point this to your .shp file (the restâ€”.dbf/.shx/.prj/.cpgâ€”must be in the same folder)
SHAPEFILE = Path(shape_dir) / "india_10k_fields.shp"
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

# === 2) READ ===
# GeoPandas will automatically pick up .dbf/.shx/.prj/.cpg
gdf = gpd.read_file(SHAPEFILE)

print("\nâœ… Loaded", len(gdf), "polygons")
print("Columns:", list(gdf.columns))
print("CRS:", gdf.crs)

# === 3) ENSURE GEOGRAPHIC CRS (EPSG:4326) ===
# Many web things expect lon/lat (WGS84)
if gdf.crs is None:
    # If the .prj was missing (rare), you can set it manually here:
    # gdf.set_crs("EPSG:4326", inplace=True)
    raise ValueError("No CRS found. Check that the .prj file sits with the .shp.")
if gdf.crs.to_string().upper() != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

# === 4) LIGHT CLEANUP + AREA (geodesic) ===
geod = Geod(ellps="WGS84")
# Compute geodesic area for each polygon (accounts for Earth's curvature)
def area_ha(row):
    # row.geometry can be MultiPolygon/Polygon
    geom = row.geometry
    if geom is None or geom.is_empty:
        return None
    # geodesic area in m^2 â†’ hectares
    area = 0.0
    for poly in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
        lon, lat = poly.exterior.coords.xy
        poly_area, _ = geod.polygon_area_perimeter(lon, lat)
        area += abs(poly_area)
    return area / 10_000.0

gdf["area_ha"] = gdf.apply(area_ha, axis=1)

# Create a simple id if none exists
if "fid" not in gdf.columns:
    gdf["fid"] = range(1, len(gdf) + 1)

# Reorder a tidy view (keep all original attrs too)
tidy_cols = ["fid", "area_ha"] + [c for c in gdf.columns if c not in ("fid", "area_ha", "geometry")]
gdf = gdf[tidy_cols + ["geometry"]]

print("\nSample rows:")
print(gdf.head(3))

# === 5) WRITE GEOJSONS ===
full_geojson = OUT_DIR / "india_10k_fields_full.geojson"
sample_geojson = OUT_DIR / "india_10k_fields_sample5.geojson"

gdf.to_file(full_geojson, driver="GeoJSON")
gdf.head(5).to_file(sample_geojson, driver="GeoJSON")

# Optional: write GeoParquet (fast, compact, great for analytics)
try:
    gdf.to_parquet(OUT_DIR / "india_10k_fields.parquet")
except Exception as e:
    print("Parquet export skipped (install pyarrow):", e)

print(f"\nðŸ“¦ Wrote:\n  â€¢ {full_geojson}\n  â€¢ {sample_geojson}")
print("Tip: load the *sample5* in your web map first, then swap to the full file.")



import folium
from IPython.display import IFrame, HTML, display

# Center on India (or Karnataka if filtered)
m = folium.Map(location=[15, 76], zoom_start=6, tiles="cartodbpositron")

# Add fields layer
folium.Choropleth(
    geo_data=gdf.to_json(),
    data=gdf,
    columns=["fid", "area"],   # fid must be a unique ID
    key_on="feature.properties.fid",
    fill_color="YlOrRd",
    fill_opacity=0.6,
    line_opacity=0.2,
    legend_name="Field size (ha)"
).add_to(m)

# Add tooltips
folium.GeoJson(
    gdf,
    name="Fields",
    tooltip=folium.GeoJsonTooltip(fields=["fid", "area_ha"],
                                  aliases=["Field ID", "Size (ha)"],
                                  localize=True)
).add_to(m)

m.save("field_sizes_map.html")

display(m)

## Other ways of displaying the map 
# # 2) Or inline the HTML (useful if iframes are blocked)
# HTML(open('field_sizes_map.html', encoding='utf-8').read())
# IFrame('/kaggle/working/field_sizes_map.html', width='100%', height=600)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

BINS = 60   # histogram bins
PCTS = [10, 25, 50, 75, 90]  # percentiles to annotate

areas = gdf["area_ha"].dropna().to_numpy()
areas = areas[areas > 0]  # guard against zeros/negatives

mean_area = float(np.mean(areas)) if areas.size else np.nan
pct_vals = {p: float(np.percentile(areas, p)) for p in PCTS}

# ---------- KDE ----------
xgrid = None
kde_y = None

if areas.size > 1:
    kde = gaussian_kde(areas)
    xgrid = np.linspace(np.min(areas), np.percentile(areas, 99.5), 500)
    kde_y = kde(xgrid)

# ---------- PLOT ----------
plt.figure(figsize=(10, 6))

# Histogram (density so KDE overlays nicely)
n, bins, patches = plt.hist(
    areas,
    bins=BINS,
    density=True,
    alpha=0.65,
    edgecolor="white",
)

# KDE overlay (if available)
if xgrid is not None and kde_y is not None:
    plt.plot(xgrid, kde_y, linewidth=2)

# Vertical markers: mean + percentiles
plt.axvline(mean_area, linestyle="--", linewidth=2)
for p, val in pct_vals.items():
    plt.axvline(val, linestyle=":", linewidth=1.8)

# Labels & cosmetics
plt.title("Distribution of Field Sizes (hectares)", fontsize=15)
plt.xlabel("Field size (ha)", fontsize=12)
plt.ylabel("Density", fontsize=12)

# Legend text
legend_lines = [f"Mean = {mean_area:.2f} ha"] + [f"P{p} = {pct_vals[p]:.2f} ha" for p in PCTS]
legend_text = " | ".join(legend_lines)
plt.text(
    0.99, 0.98, legend_text,
    ha="right", va="top", transform=plt.gca().transAxes, fontsize=10
)

plt.tight_layout()
plt.show()


# ---------- PRINT SUMMARY ----------
print("\nSummary (ha):")
print(f"Count: {len(areas)}")
print(f"Mean:  {mean_area:.3f}")
print("Percentiles:", {p: round(v, 3) for p, v in pct_vals.items()})


import json
from shapely.geometry.base import BaseGeometry

# 1) Filter the polygons you want to show
subset = gdf[gdf["fid"].isin([6449, 6450, 6451, 6452])].copy()

# Keep only polygons (if geometry was ever replaced, this protects Folium)
subset = subset[subset.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

# 2) Drop any attribute columns that contain Shapely geometries (e.g., 'centroid')
geom_cols = []
for c in subset.columns:
    if c == subset.geometry.name:
        continue
    series = subset[c].dropna()
    if len(series) and isinstance(series.iloc[0], BaseGeometry):
        geom_cols.append(c)
if geom_cols:
    subset = subset.drop(columns=geom_cols)

# (Optional) keep only a few safe props
keep = [subset.geometry.name] + [c for c in ["fid", "area_ha"] if c in subset.columns]
subset = subset[keep]

# 3) Map center: use union_all (Shapely 2) or fallback to unary_union
try:
    combined = subset.geometry.union_all()          # Shapely 2.x
except AttributeError:
    combined = subset.unary_union                  # Deprecated but OK as fallback
center = [combined.centroid.y, combined.centroid.x]

# 4) Build the map â€” use fit_bounds to auto-zoom correctly
m = folium.Map(location=center, zoom_start=12)      # zoom_start will be overridden by fit_bounds
gj = folium.GeoJson(data=json.loads(subset.to_json()), name="filtered_fields")
gj.add_to(m)
m.fit_bounds(gj.get_bounds())    #auto zoom

display(m)


from shapely.geometry import box
from shapely import union_all, make_valid, minimum_rotated_rectangle  # Shapely 2.x

# Merge all fields into one polygon (may be multi-part)
minx, miny, maxx, maxy = subset.total_bounds
bbox_rect = box(minx, miny, maxx, maxy)
gdf_bbox = gpd.GeoDataFrame({"name": ["bbox_rect"]}, geometry=[bbox_rect], crs="EPSG:4326")

# 4) Build the map â€” use fit_bounds to auto-zoom correctly
m = folium.Map(location=center, zoom_start=12)      # zoom_start will be overridden by fit_bounds
gj = folium.GeoJson(data=json.loads(gdf_bbox.to_json()), name="filtered_fields")
gj.add_to(m)
m.fit_bounds(gj.get_bounds())    #auto zoom

display(m)


print(gdf_bbox["geometry"][0])


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

import warnings
# Suppress the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Set project for bigframes (fixes the error you saw)
# close session before starting another bigframes.pandas.close_session()
bigframes.options.bigquery.project = f"{project_id}" 


import yaml

# Define the same structure as Python dictionaries / lists
wheat_rules = {
    "region": "Banda, Bundelkhand, Uttar Pradesh",
    "crop": "Wheat (Rabi, Novâ€“Apr)",
    "units": {
        "temp": "Â°C",
        "wind": "km/h",
        "rain": "mm",
        "et": "mm",
        "soil_moisture": "% of field capacity",
    },
    "soil": {
        "defaults": {
            "whc_mm_per_m": 120,
            "root_depth_m": 0.8,
            "depletion_fraction": 0.5,
        },
        "formulas": {
            "TAW_mm": "whc_mm_per_m * root_depth_m",
            "RAW_mm": "TAW_mm * depletion_fraction",
        },
    },
    "wheat": {
        "phenology": {
            "gdd_base_c": 5,
            "gdd_cap_c": 30,
            "stage_thresholds": {
                "emergence": 150,
                "tillering": 450,
                "booting": 800,
                "flowering": 1050,
                "grain_fill": 1350,
                "maturity": 1500,
            },
        },
        "kc_values": {
            "initial": 0.35,
            "mid": 1.15,
            "late": 0.45,
        },
    },
    "advice": {
        "sowing": {
            "window": {"start": "11-05", "end": "12-15"},
            "conditions": {
                "min_temp_c": 8,
                "max_temp_c": 30,
                "soil_moisture_min_pct": 55,
                "rain_prev10d_max_mm": 60,
            },
            "logic": """in_window = date_between(today, window.start, window.end)
ok_temp   = (tmin_c >= conditions.min_temp_c) and (tmax_c <= conditions.max_temp_c)
ok_soil   = soil_moist_pct >= conditions.soil_moisture_min_pct
ok_rain   = rain_prev10d <= conditions.rain_prev10d_max_mm
decision  = in_window and ok_temp and ok_soil and ok_rain
score     = 0.4*norm01(soil_moist_pct, 40, 80) + 0.3*bool_score(ok_temp) + 0.3*bool_score(ok_rain)""",
        },
        "irrigation": {
            "method": {
                "et0": "Hargreaves if only Tmin/Tmax available, else Penmanâ€“Monteith if RH/wind/solar present",
                "etc": "et0 * kc(stage)",
            },
            "trigger": "depletion_mm >= RAW_mm",
            "suggested_depth_mm": "min(depletion_mm, 0.7 * TAW_mm)",
            "window_pref": "wind10m < 10 and rh in 40..80 and t2m < 32",
        },
        "spray": {
            "ok_if": "wind10m <= 12 and rh in 40..80 and t2m <= 32 and rain_next6h == 0",
            "note": "Prefer 06â€“10 or 17â€“19 hours",
        },
        "stresses": {
            "frost_risk": {
                "trigger": "tmin_next3d <= 2",
                "message": "Frost risk: irrigate lightly pre-dawn or use smoke/shelterbelts; avoid sprays.",
            },
            "heat_risk": {
                "trigger": "stage in ['booting','flowering','grain_fill'] and tmax_next3d >= 32",
                "message": "Heat risk: advance irrigation by 24h; shift sprays to early morning.",
            },
        },
        "harvest": {
            "window": {
                "logic": """ok = (max_next2d_rain == 0) and (mean_next2d_rh <= 65) and (mean_next2d_wind in 3..12)
decision = ok""",
                "message_true": "Good drying window in next 48 h; plan harvest/threshing.",
                "message_false": "Unsettled weather; avoid harvesting.",
            },
        },
    },
}

# Write to file
with open("wheat_rules.yaml", "w") as f:
    yaml.dump(wheat_rules, f, sort_keys=False)

print("YAML file saved as wheat_rules.yaml")



# Make an intersect for outskirts of Banda city , in Bundelkhand UP, India
# Harvesting season for Wheat (Kharif) Marh to April 

sql = f"""
SELECT
  t1.init_time,
  f.time,
  e.2m_temperature - 273.15 AS temp_2m_celcius,
  SQRT(POW(e.10m_u_component_of_wind, 2) + POW(e.10m_v_component_of_wind, 2)) AS wind_speed,
  e.total_precipitation_12hr AS total_precipitation_12hr,
  ST_X(ST_Centroid(t1.geography_polygon)) AS longitude,
  ST_Y(ST_Centroid(t1.geography_polygon)) AS latitude
FROM `big-query-project-472510.weathernext_gen_forecasts.126478713_1_0` AS t1
CROSS JOIN UNNEST(t1.forecast) AS f
CROSS JOIN UNNEST(f.ensemble) AS e
WHERE t1.init_time >= TIMESTAMP('2024-03-12 00:00:00 UTC') 
  AND t1.init_time <  TIMESTAMP('2024-03-27 00:00:00 UTC')
  AND ST_INTERSECTS(
        t1.geography_polygon,
        ST_GEOGFROMTEXT('{gdf_bbox["geometry"][0]}')  
      )
ORDER BY f.time
"""

df_harvesting = bpd.read_gbq(sql).sort_values("init_time")
df_harvesting["crop_stage"] = "harvesting"
df_harvesting.iloc[0]


# Make an intersect for outskirts of Banda city , in Bundelkhand UP, India
# Sowing season for wheat Nov  

sql = f"""
SELECT
  t1.init_time,
  f.time,
  e.2m_temperature - 273.15 AS temp_2m_celcius,
  SQRT(POW(e.10m_u_component_of_wind, 2) + POW(e.10m_v_component_of_wind, 2)) AS wind_speed,
  e.total_precipitation_12hr AS total_precipitation_12hr,
  ST_X(ST_Centroid(t1.geography_polygon)) AS longitude,
  ST_Y(ST_Centroid(t1.geography_polygon)) AS latitude
FROM `big-query-project-472510.weathernext_gen_forecasts.126478713_1_0` AS t1
CROSS JOIN UNNEST(t1.forecast) AS f
CROSS JOIN UNNEST(f.ensemble) AS e
WHERE t1.init_time >= TIMESTAMP('2024-11-12 00:00:00 UTC') 
  AND t1.init_time <  TIMESTAMP('2024-11-27 00:00:00 UTC')
  AND ST_INTERSECTS(
        t1.geography_polygon,
        ST_GEOGFROMTEXT('{gdf_bbox["geometry"][0]}')  
      )
ORDER BY f.time
"""

df_sowing = bpd.read_gbq(sql).sort_values("init_time")
df_sowing["crop_stage"] = "sowing"
df_sowing.iloc[0]


from bigframes.ml import llm

# Define the model you want to use
model_name = "gemini-1.5-pro-002"

gemini = llm.GeminiTextGenerator()
gemini.model_name


import yaml

with open("wheat_rules.yaml", "r") as f:
    rules = yaml.safe_load(f)


# take sample from the sowing and harvesting BigFrames 


crop_stage = "SOWING"
prompt = f"""
Your responses must be based on two main sources of information:

1.  **Local Field Data:** The farmer will provide you with a YAML file containing up-to-date, hyper-specific information about their farm. This data will include details like:
    * `location`: e.g., latitude and longitude, or specific address.
    * `crop_stage`: {crop_stage}.
    * `weather_data`: 'temp_2m_celcius', 'wind_speed', 'total_precipitation_12hr','longitude', 'latitude'
   

2.  **Industry Best Practices:** 

**Your task is to analyze the provided industry standards and generate a clear, actionable report for the farmer. Your report should include:**
**Agriculture Standard Rules**
{rules}

* **Timely Alerts:** Immediately highlight any critical issues or urgent actions required. For example, "URGENT: Forecast shows a high chance of frost tonight. Protect your young plants."
* **Actionable Recommendations:** Provide specific advice tailored to the current crop stage and environmental conditions. This could include suggestions on irrigation schedules, nutrient application, pest management, or harvesting.
* **Predictive Insights:** Based on the data, anticipate potential future issues (e.g., "The current low humidity could increase the risk of powdery mildew if not monitored.")
* **Justification:** Briefly explain *why* you are making a specific recommendation. Use simple, direct language.



**Tone and Style:**

* Be direct, clear, and concise.
* Use a tone that is helpful and knowledgeable, like a trusted co-pilot.
* Avoid jargon where possible. If technical terms are necessary, explain them simply.
* Format your response using clear headings and bullet points for readability.

**Initial Interaction:**

Generate your analysis basis the provided data points 

"""


df_sowing_sample = df_sowing.sample(n=3, random_state=233)
answer_alt = gemini.predict(df_sowing_sample, prompt=[prompt,df_sowing_sample['crop_stage']] )
df_selected = answer_alt.to_pandas().reset_index(drop=False)[['time', 'ml_generate_text_llm_result']].sort_values(['time'])
import pandas as pd
pd.set_option("display.max_colwidth", None)
# Display as a nice table in the notebook
from IPython.display import display
display(df_selected[["time", "ml_generate_text_llm_result"]])


crop_stage = "HARVESTING"
prompt = f"""
Your responses must be based on two main sources of information:

1.  **Local Field Data:** The farmer will provide you with a YAML file containing up-to-date, hyper-specific information about their farm. This data will include details like:
    * `location`: e.g., latitude and longitude, or specific address.
    * `crop_stage`: {crop_stage}.
    * `soil_metrics`: e.g., `moisture`: percentage, `pH`: value, `nutrients`: key nutrients and their levels (e.g., nitrogen, phosphorus, potassium).
    * `weather_data`: e.g., `temperature`: current, forecast, `rainfall`: recent, forecast, `humidity`: percentage.
   

2.  **Industry Best Practices:** You have been trained on a vast dataset of global agricultural knowledge. You should leverage this expertise to provide recommendations that align with sustainable and effective farming practices.

**Your task is to analyze the provided industry standards and generate a clear, actionable report for the farmer. Your report should include:**
**Agriculture Standard Rules**
{rules}

* **Timely Alerts:** Immediately highlight any critical issues or urgent actions required. For example, "URGENT: Forecast shows a high chance of frost tonight. Protect your young plants."
* **Actionable Recommendations:** Provide specific advice tailored to the current crop stage and environmental conditions. This could include suggestions on irrigation schedules, nutrient application, pest management, or harvesting.
* **Predictive Insights:** Based on the data, anticipate potential future issues (e.g., "The current low humidity could increase the risk of powdery mildew if not monitored.")
* **Justification:** Briefly explain *why* you are making a specific recommendation. Use simple, direct language.



**Tone and Style:**

* Be direct, clear, and concise.
* Use a tone that is helpful and knowledgeable, like a trusted co-pilot.
* Avoid jargon where possible. If technical terms are necessary, explain them simply.
* Format your response using clear headings and bullet points for readability.

**Initial Interaction:**

Generate your analysis basis the provided data points 

"""


df_harvesting_sample = df_harvesting.sample(n=3, random_state=233)
answer_alt = gemini.predict(df_harvesting_sample, prompt=[prompt, df_harvesting_sample['crop_stage']])
df_selected = answer_alt.to_pandas().reset_index(drop=False)[['time', 'ml_generate_text_llm_result']].sort_values(['time'])

from IPython.display import display
display(df_selected[["time", "ml_generate_text_llm_result"]])

