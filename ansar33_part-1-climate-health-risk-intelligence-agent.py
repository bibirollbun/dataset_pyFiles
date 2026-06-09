# Cell1 - Notebook 1 - imports & setup
from pathlib import Path
import os, json, time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

# where we'll store files
WORKDIR = Path("/kaggle/working/env_resilience_agent")
WORKDIR.mkdir(parents=True, exist_ok=True)

print("WORKDIR:", WORKDIR)
print("Listing /kaggle/working (first 100 lines):")
!ls -la /kaggle/working | sed -n '1,120p'


# Cell 2: API keys and live/demo toggles
import os
from kaggle_secrets import UserSecretsClient
us = UserSecretsClient()

# Pull secrets from Kaggle vault
os.environ["OPENAI_KEY"] = us.get_secret("OPENAI_KEY") or ""

AQICN_TOKEN = os.environ.get("AQICN_TOKEN")
OPENWEATHER_KEY = os.environ.get("OPENWEATHER_KEY")
OPENAI_KEY = os.environ.get("OPENAI_KEY")

USE_LIVE_AQI = bool(AQICN_TOKEN)
USE_LIVE_WEATHER = bool(OPENWEATHER_KEY)
USE_LIVE_LLM = bool(OPENAI_KEY)

print("AQICN_TOKEN:", bool(AQICN_TOKEN))
print("OPENWEATHER_KEY:", bool(OPENWEATHER_KEY))
print("OPENAI_KEY:", bool(OPENAI_KEY))
print("USE_LIVE_AQI:", USE_LIVE_AQI, "USE_LIVE_WEATHER:", USE_LIVE_WEATHER, "USE_LIVE_LLM:", USE_LIVE_LLM)


# Cell 3: AQICN & OpenWeather helpers (safe — return None if keys not present)
import requests, math

def fetch_aqi_city(city):
    """Try AQICN live, else return None (we will fallback later)."""
    if not USE_LIVE_AQI:
        return None
    try:
        url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        if j.get("status") != "ok":
            return None
        data = j.get("data", {})
        return {
            "aqi": data.get("aqi"),
            "dominantpol": data.get("dominantpol"),
            "time": data.get("time", {}).get("s"),
            "city": data.get("city",{}).get("name")
        }
    except Exception as e:
        print("AQI fetch error:", e)
        return None

def fetch_weather_city(city):
    """Try OpenWeather live, else return None"""
    if not USE_LIVE_WEATHER:
        return None
    try:
        params = {"q": city, "appid": OPENWEATHER_KEY, "units": "metric"}
        url = "https://api.openweathermap.org/data/2.5/weather"
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        # minimal safe extraction
        main = j.get("main", {})
        wind = j.get("wind", {})
        weather = (j.get("weather") or [{}])[0]
        return {
            "temp": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "wind_speed": wind.get("speed"),
            "descr": weather.get("description")
        }
    except Exception as e:
        print("OpenWeather fetch error:", e)
        return None


# Test if helper functions work (demo mode)
print("Testing AQI fetch (synthetic fallback expected):")
print(fetch_aqi_city("Lucknow"))

print("\nTesting Weather fetch (synthetic fallback expected):")
print(fetch_weather_city("Lucknow"))


# Cell 4: generate and save synthetic CSVs (aqi.csv, ndvi_grid.csv, complaints.csv)
rng = np.random.RandomState(42)

# 1) aqi.csv (30 days history)
days = 30
dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
aqi_vals = rng.normal(loc=80, scale=25, size=days).clip(20,300).astype(int)
aqi_df = pd.DataFrame({"date": dates, "aqi": aqi_vals})
aqi_df["pm25"] = (aqi_df["aqi"] * 0.6 + rng.normal(0,5,size=days)).round().astype(int)
aqi_df.to_csv(WORKDIR/"aqi.csv", index=False)

# 2) ndvi_grid.csv (grid points)
nx, ny = 30, 20
grid = rng.uniform(0.05, 0.6, size=(ny, nx))
lats = np.linspace(26.8, 27.0, ny)
lons = np.linspace(80.9, 81.1, nx)
rows = []
for i,lat in enumerate(lats):
    for j,lon in enumerate(lons):
        rows.append({"lat": float(lat), "lon": float(lon), "ndvi": float(grid[i,j])})
ndvi_df = pd.DataFrame(rows)
ndvi_df.to_csv(WORKDIR/"ndvi_grid.csv", index=False)

# 3) complaints.csv (sample text complaints)
texts = [
  "Overflowing garbage near market","Open burning in sector 5",
  "Dust and smog during morning hours","Uncollected trash near bus stop",
  "Construction dust near river","Smell of burning plastic near lane","No greenery near school"
]
complaints = []
for i in range(40):
    complaints.append({
        "id": i,
        "text": rng.choice(texts),
        "lat": 26.85 + rng.rand()*0.1,
        "lon": 80.95 + rng.rand()*0.1
    })
complaints_df = pd.DataFrame(complaints)
complaints_df.to_csv(WORKDIR/"complaints.csv", index=False)

print("Synthetic data saved:")
!ls -la /kaggle/working/env_resilience_agent


# Cell 5: quick read & show heads so you see outputs immediately
import pandas as pd, pathlib
p = pathlib.Path("/kaggle/working/env_resilience_agent")

for fname in ["aqi.csv", "ndvi_grid.csv", "complaints.csv"]:
    fp = p/fname
    print("->", fname, "exists:", fp.exists())
    if fp.exists():
        if fname.endswith(".csv"):
            print(pd.read_csv(fp).head(), "\n")


# Cell 6: safe recreate fallback if any file missing (runs only if needed)
from pathlib import Path
p = Path("/kaggle/working/env_resilience_agent")
missing = [f for f in ["aqi.csv","ndvi_grid.csv","complaints.csv"] if not (p/f).exists()]
if missing:
    print("Missing files:", missing, " --> recreating synthetic files now.")
    # paste same generator code (or call function). For brevity, re-run Cell4 code here:
    # (We will call the generator function block again)
    # For simplicity we just call the previous code by re-executing it: run Cell 4 again manually if needed.
else:
    print("All files present. Ready.")


# Cell 7: process AQI history and create a simple short-term forecast
import pandas as pd
from pathlib import Path
WORKDIR = Path("/kaggle/working/env_resilience_agent")

# read raw aqi
aqi_raw = pd.read_csv(WORKDIR/"aqi.csv", parse_dates=["date"])
aqi_raw = aqi_raw.sort_values("date").reset_index(drop=True)

# basic cleaning
aqi_raw["aqi"] = pd.to_numeric(aqi_raw["aqi"], errors="coerce").fillna(method="ffill").astype(int)
aqi_raw["pm25"] = pd.to_numeric(aqi_raw.get("pm25", pd.Series()), errors="coerce").fillna(method="ffill").astype(int)

# aggregated daily stats (already daily but keep pattern)
aqi_daily = aqi_raw.groupby(aqi_raw["date"].dt.date).agg(
    aqi_mean=("aqi","mean"),
    aqi_max=("aqi","max"),
    pm25_mean=("pm25","mean")
).reset_index().rename(columns={"date":"day"})
aqi_daily["day"] = pd.to_datetime(aqi_daily["day"])

# simple forecast: 7-day moving average shifted forward 1 day as naive forecast
aqi_daily["ma7"] = aqi_daily["aqi_mean"].rolling(7, min_periods=1).mean()
# create forecast rows for next 3 days
last_date = aqi_daily["day"].max()
forecast_horizon = 3
fcast_rows = []
last_ma = aqi_daily["ma7"].iloc[-1]
for i in range(1, forecast_horizon+1):
    fdate = last_date + pd.Timedelta(days=i)
    # simple model: trend = difference of last two means
    trend = (aqi_daily["aqi_mean"].iloc[-1] - aqi_daily["aqi_mean"].iloc[-2]) if len(aqi_daily) > 1 else 0
    pred = max(0, int(round(last_ma + trend * 0.5)))
    fcast_rows.append({"date": fdate, "pred_aqi": pred})
fcast_df = pd.DataFrame(fcast_rows)

# save processed outputs
aqi_daily.to_csv(WORKDIR/"aqi_history_processed.csv", index=False)
fcast_df.to_csv(WORKDIR/"aqi_forecast.csv", index=False)

print("AQI processed rows:", len(aqi_daily))
print("Forecast rows saved:", len(fcast_df))


# Cell 8: create NDVI summary (mean per coarse tile) and complaints summary
import pandas as pd
import numpy as np
from pathlib import Path
WORKDIR = Path("/kaggle/working/env_resilience_agent")

# NDVI grid read & simple heatmap-ready aggregation
ndvi = pd.read_csv(WORKDIR/"ndvi_grid.csv")
# compute mean NDVI across whole grid and per-row summary
ndvi_overall = ndvi["ndvi"].mean()
ndvi_row = ndvi.groupby(np.round(ndvi["lat"], 3)).agg(ndvi_mean=("ndvi","mean")).reset_index().rename(columns={"lat":"lat_approx"})

# complaints read & nearest-count per coarse lat/lon cell
complaints = pd.read_csv(WORKDIR/"complaints.csv")
# rough spatial bin by rounding coordinates
complaints["lat_bin"] = complaints["lat"].round(2)
complaints["lon_bin"] = complaints["lon"].round(2)
complaints_agg = complaints.groupby(["lat_bin","lon_bin"]).size().reset_index(name="count")
# top complaint texts sample
top_texts = complaints["text"].value_counts().head(5).to_dict()

# Save results for Notebook-2 use
pd.DataFrame([{"ndvi_overall": float(ndvi_overall)}]).to_csv(WORKDIR/"ndvi_overall.csv", index=False)
ndvi_row.to_csv(WORKDIR/"ndvi_row_summary.csv", index=False)
complaints_agg.to_csv(WORKDIR/"complaints_agg.csv", index=False)
pd.DataFrame([{"top_texts": str(top_texts)}]).to_csv(WORKDIR/"complaints_summary.csv", index=False)

print("NDVI overall:", round(ndvi_overall,4))
print("NDVI row summary rows:", len(ndvi_row))
print("Complaints agg rows:", len(complaints_agg))
print("Top complaint texts:", top_texts)


# Cell 9: final verify and optional save a small metadata file (used in writeup)
from pathlib import Path
import json
WORKDIR = Path("/kaggle/working/env_resilience_agent")

files = sorted([p.name for p in WORKDIR.iterdir()])
meta = {
    "created_at": pd.Timestamp.now().isoformat(),
    "files": files,
    "author": "Ansar Ahmad",
    "notebook": "01_data_setup.ipynb"
}
with open(WORKDIR/"metadata.json","w",encoding="utf8") as f:
    json.dump(meta,f,indent=2)

print("Files in WORKDIR now:")
for f in files:
    print("-", f)
print("\nMetadata written to:", WORKDIR/"metadata.json")

