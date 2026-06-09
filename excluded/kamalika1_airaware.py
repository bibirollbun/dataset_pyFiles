# ============================================================
# Upgraded AQI Guardian Agent (Kaggle Notebook) - Corrected
# - Timezone fixes applied (no tz-naive / tz-aware errors)
# - Planner Agent (rule-based)
# - Health Risk Score (custom formula)
# - Pollution Source Estimator (wind-backed inference)
# - Multi-city comparison
# - Forecasting (RandomForest lag model)
# - Memory (memory.json)
# - Plots (Plotly)
# - Kaggle-friendly (no Streamlit)
# ============================================================

import requests
import pandas as pd
import numpy as np
import json
import os
import datetime as dt
import math
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -------------------------
# Config & Memory
# -------------------------
MEMORY_FILE = "memory.json"
DEFAULT_LOC = "Delhi, India"
DEFAULT_ALERT = 100

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"prefs": {"last_locations": [DEFAULT_LOC], "alert_threshold": DEFAULT_ALERT}, "history": {}}
    return {"prefs": {"last_locations": [DEFAULT_LOC], "alert_threshold": DEFAULT_ALERT}, "history": {}}

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2, default=str)

memory = load_memory()

# -------------------------
# Helpers: geocode
# -------------------------
def geocode(place):
    """Return (lat, lon) or None"""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        r = requests.get(url, params={"q": place, "format": "json", "limit": 1}, headers={"User-Agent": "aqi-agent"})
        if r.status_code == 200 and r.json():
            d = r.json()[0]
            return float(d["lat"]), float(d["lon"])
    except Exception as e:
        print("Geocode error:", e)
    return None

# -------------------------
# Fetch PM2.5 from OpenAQ
# -------------------------
def fetch_pm25(lat, lon, hours=72, radius_m=25000):
    try:
        end = dt.datetime.utcnow()
        start = end - dt.timedelta(hours=hours)
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": radius_m,
            "parameter": "pm25",
            "date_from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 1000,
            "sort": "desc",
            "order_by": "datetime"
        }
        r = requests.get("https://api.openaq.org/v2/measurements", params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
        rows = []
        for m in results:
            ts = m["date"]["utc"]
            val = m["value"]
            rows.append({"utc": pd.to_datetime(ts), "pm25": float(val)})
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["utc", "pm25"])
        # normalize timezone: make tz-aware UTC
        if df["utc"].dt.tz is None:
            df["utc"] = df["utc"].dt.tz_localize("UTC")
        else:
            df["utc"] = df["utc"].dt.tz_convert("UTC")
        df = df.drop_duplicates(subset=["utc"]).sort_values("utc").reset_index(drop=True)
        return df
    except Exception as e:
        print("OpenAQ fetch failed:", e)
        return pd.DataFrame(columns=["utc", "pm25"])

# -------------------------
# Fetch weather (including wind)
# -------------------------
def fetch_weather(lat, lon, hours_past=72, hours_future=48):
    try:
        now = dt.datetime.utcnow()
        start = (now - dt.timedelta(hours=hours_past)).strftime("%Y-%m-%dT%H:00:00Z")
        end = (now + dt.timedelta(hours=hours_future)).strftime("%Y-%m-%dT%H:00:00Z")
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m",
            "start": start, "end": end, "timezone": "UTC"
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        hours = j.get("hourly", {}).get("time", [])
        temps = j.get("hourly", {}).get("temperature_2m", [])
        hums = j.get("hourly", {}).get("relativehumidity_2m", [])
        ws = j.get("hourly", {}).get("windspeed_10m", [])
        wd = j.get("hourly", {}).get("winddirection_10m", [])
        df = pd.DataFrame({
            "utc": pd.to_datetime(hours),
            "temp": temps,
            "rh": hums,
            "wind_spd": ws,
            "wind_dir": wd
        })
        # ensure tz-aware UTC
        if df["utc"].dt.tz is None:
            df["utc"] = df["utc"].dt.tz_localize("UTC")
        else:
            df["utc"] = df["utc"].dt.tz_convert("UTC")
        return df
    except Exception as e:
        print("Weather fetch failed:", e)
        return pd.DataFrame(columns=["utc", "temp", "rh", "wind_spd", "wind_dir"])

# -------------------------
# AQI functions
# -------------------------
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

def pm25_to_aqi(c):
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return None
    for clow, chigh, ilow, ihigh in PM25_BREAKPOINTS:
        if clow <= c <= chigh:
            return round(((ihigh - ilow) / (chigh - clow)) * (c - clow) + ilow)
    return None

def aqi_category(aqi):
    if aqi is None:
        return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

# -------------------------
# Health Risk Score (custom)
# -------------------------
def health_risk_score(aqi, temp, rh):
    if aqi is None:
        return None, "Unknown"
    aqi_norm = min(max(aqi / 500.0, 0.0), 1.0)
    if temp is None or (isinstance(temp, float) and math.isnan(temp)):
        temp_pen = 0.0
    else:
        if 15 <= temp <= 28:
            temp_pen = 0.0
        else:
            temp_pen = min(1.0, abs(temp - 21) / 25.0)
    if rh is None or (isinstance(rh, float) and math.isnan(rh)):
        rh_pen = 0.0
    else:
        rh_pen = min(1.0, max(0.0, (rh - 60) / 40.0))
    score = 100 * (0.6 * aqi_norm + 0.25 * rh_pen + 0.15 * temp_pen)
    score = max(0.0, min(100.0, score))
    if score < 30: label = "Low"
    elif score < 55: label = "Moderate"
    elif score < 75: label = "High"
    else: label = "Extreme"
    return round(score, 1), label

# -------------------------
# Planner Agent (rule-based)
# -------------------------
def planner_agent(forecast_df, user_profile=None):
    if user_profile is None:
        user_profile = {"sensitive": False, "has_children": False, "athlete": False}
    if forecast_df is None or forecast_df.empty:
        return {"note": "No forecast available to plan."}
    df = forecast_df.copy()
    # ensure utc tz-aware
    if df["utc"].dt.tz is None:
        df["utc"] = df["utc"].dt.tz_localize("UTC")
    else:
        df["utc"] = df["utc"].dt.tz_convert("UTC")
    df["hour"] = df["utc"].dt.hour
    bins = {
        "Morning (06-12)": df[(df["hour"] >= 6) & (df["hour"] < 12)],
        "Afternoon (12-18)": df[(df["hour"] >= 12) & (df["hour"] < 18)],
        "Evening/Night (18-06)": df[(df["hour"] >= 18) | (df["hour"] < 6)]
    }
    plan = {}
    for period, sub in bins.items():
        if sub.empty:
            avg_pm = None
            avg_aqi = None
        else:
            avg_pm = float(sub["pm25"].mean())
            avg_aqi = pm25_to_aqi(avg_pm)
        actions = []
        if avg_aqi is None:
            actions.append("No data for this period.")
        else:
            cat = aqi_category(avg_aqi)
            if cat in ["Good", "Moderate"]:
                actions.append("Normal outdoor activities OK; sensitive people should monitor.")
            elif cat == "Unhealthy for Sensitive Groups":
                actions.append("Sensitive people: avoid long, intense outdoor exertion.")
                actions.append("Consider scheduling outdoor activities for better periods.")
            elif cat == "Unhealthy":
                actions.append("Limit outdoor activities for everyone; prefer indoor exercise.")
                actions.append("Use air purifiers indoors if available.")
            elif cat in ["Very Unhealthy", "Hazardous"]:
                actions.append("Avoid outdoor activities. Keep windows closed; use filtered mask if going out.")
                actions.append("Check on vulnerable household members.")
        if user_profile.get("athlete", False):
            actions.append("Athlete: shift heavy training to better air hours or indoors.")
        if user_profile.get("has_children", False):
            actions.append("Children: avoid prolonged outdoor play during worst hours.")
        if user_profile.get("sensitive", False):
            actions.append("Sensitive: carry medication/inhaler; minimize trips outside during high AQI.")
        plan[period] = {"avg_pm25": avg_pm, "avg_aqi": avg_aqi, "actions": actions}
    return plan

# -------------------------
# Forecast model (lags + hour)
# -------------------------
def train_predictor_hourly(df_obs, hours_ahead=24):
    if df_obs is None or df_obs.empty:
        return None
    df = df_obs.copy().dropna(subset=["pm25"]).sort_values("utc")
    # ensure tz-aware before resample
    if df["utc"].dt.tz is None:
        df["utc"] = df["utc"].dt.tz_localize("UTC")
    else:
        df["utc"] = df["utc"].dt.tz_convert("UTC")
    df = df.set_index("utc").resample("1H").mean().interpolate(limit_direction="both").reset_index()
    if len(df) < 12:
        return None
    for lag in [1,2,3,6]:
        df[f"lag_{lag}"] = df["pm25"].shift(lag)
    df["hour"] = df["utc"].dt.hour
    df = df.dropna().reset_index(drop=True)
    if len(df) < 12:
        return None
    X = df[[f"lag_{l}" for l in [1,2,3,6]] + ["hour"]].values
    y = df["pm25"].values
    model = RandomForestRegressor(n_estimators=50, random_state=0)
    model.fit(X, y)
    last = df.iloc[-1:].copy()
    results = []
    cur = last.copy()
    for i in range(hours_ahead):
        feat = np.array([cur[f"lag_{l}"].values[0] for l in [1,2,3,6]] + [int((cur["hour"].values[0] + 1) % 24)])
        pred = float(model.predict(feat.reshape(1, -1))[0])
        next_dt = cur["utc"].values[0] + np.timedelta64(1, "h")
        next_row = {"utc": pd.to_datetime(next_dt), "pm25": pred}
        # update lags
        for l in [1,2,3,6]:
            if l == 1:
                next_row[f"lag_{l}"] = pred
            else:
                prev = f"lag_{l-1}"
                if prev in cur:
                    next_row[f"lag_{l}"] = float(cur[prev].values[0])
                else:
                    next_row[f"lag_{l}"] = float(cur["pm25"].values[0])
        next_row["hour"] = int((cur["hour"].values[0] + 1) % 24)
        results.append(next_row)
        cur = pd.DataFrame([next_row])
    pred_df = pd.DataFrame(results)
    # ensure tz-aware UTC
    if pred_df["utc"].dt.tz is None:
        pred_df["utc"] = pred_df["utc"].dt.tz_localize("UTC")
    else:
        pred_df["utc"] = pred_df["utc"].dt.tz_convert("UTC")
    return pred_df[["utc", "pm25"]]

# -------------------------
# Pollution Source Estimator
# -------------------------
def compass_sector(angle):
    sectors = [
        (11.25, "N"), (33.75, "NNE"), (56.25, "NE"), (78.75, "ENE"),
        (101.25, "E"), (123.75, "ESE"), (146.25, "SE"), (168.75, "SSE"),
        (191.25, "S"), (213.75, "SSW"), (236.25, "SW"), (258.75, "WSW"),
        (281.25, "W"), (303.75, "WNW"), (326.25, "NW"), (348.75, "NNW"), (360.0, "N")
    ]
    angle = angle % 360
    for deg, name in sectors:
        if angle <= deg:
            return name
    return "N"

def estimate_source(pm_df, weather_df, spike_threshold=1.5):
    if pm_df is None or pm_df.empty or weather_df is None or weather_df.empty:
        return {"note": "Insufficient data for source estimation."}
    # make tz-aware
    if pm_df["utc"].dt.tz is None:
        pm_df["utc"] = pm_df["utc"].dt.tz_localize("UTC")
    else:
        pm_df["utc"] = pm_df["utc"].dt.tz_convert("UTC")
    if weather_df["utc"].dt.tz is None:
        weather_df["utc"] = weather_df["utc"].dt.tz_localize("UTC")
    else:
        weather_df["utc"] = weather_df["utc"].dt.tz_convert("UTC")
    pm_hr = pm_df.set_index("utc").resample("1H").mean().interpolate(limit_direction="both").reset_index()
    merged = pd.merge_asof(pm_hr.sort_values("utc"), weather_df.sort_values("utc"), on="utc", direction="nearest", tolerance=pd.Timedelta("1H"))
    if merged.empty:
        return {"note": "No merged data."}
    merged["pm_roll_med3"] = merged["pm25"].rolling(3, min_periods=1).median()
    merged["spike_ratio"] = merged["pm25"] / (merged["pm_roll_med3"] + 1e-6)
    spikes = merged[merged["spike_ratio"] >= spike_threshold]
    if spikes.empty:
        return {"note": "No significant PM2.5 spikes detected in the window."}
    dir_weights = {}
    for _, r in spikes.iterrows():
        wd = r.get("wind_dir", None)
        ws = r.get("wind_spd", 1.0) if not pd.isna(r.get("wind_spd", np.nan)) else 1.0
        ratio = float(r["spike_ratio"])
        if pd.isna(wd):
            continue
        sector = compass_sector(float(wd))
        w = ws * max(0.0, (ratio - 1.0))
        dir_weights[sector] = dir_weights.get(sector, 0.0) + w
    if not dir_weights:
        return {"note": "Spike detected but no wind direction data available at those hours."}
    top_sector = max(dir_weights.items(), key=lambda x: x[1])[0]
    spikes["hour"] = spikes["utc"].dt.hour
    peak_hours = spikes["hour"].value_counts().head(3).index.tolist()
    suggestion = f"Dominant wind sector during spikes: {top_sector}. Spikes often occur around hours {peak_hours} (UTC)."
    suggestion += f" This suggests pollutant sources are likely {top_sector} of the location (e.g., industry/traffic there)."
    if any(h in range(6,10) or h in range(17,20) for h in peak_hours):
        suggestion += " Peak hours align with traffic windows — traffic emissions may be contributing."
    return {"top_sector": top_sector, "peak_hours": peak_hours, "suggestion": suggestion, "weights": dir_weights}

# -------------------------
# Plot helpers
# -------------------------
def plot_observed_and_forecast(pm_df, pred_df=None, title="PM2.5 Observed & Forecast"):
    fig = go.Figure()
    if pm_df is not None and not pm_df.empty:
        hr = pm_df.set_index("utc").resample("1H").mean().interpolate(limit_direction="both").reset_index()
        fig.add_trace(go.Scatter(x=hr["utc"], y=hr["pm25"], name="Observed PM2.5", mode="lines+markers"))
    if pred_df is not None and not pred_df.empty:
        fig.add_trace(go.Scatter(x=pred_df["utc"], y=pred_df["pm25"], name="Forecast PM2.5", mode="lines"))
    fig.update_layout(title=title, xaxis_title="UTC", yaxis_title="PM2.5 (µg/m³)", height=450)
    return fig

def plot_multi_city(city_data_list):
    fig = go.Figure()
    for cd in city_data_list:
        name = cd.get("name", "city")
        df = cd.get("pm_df")
        if df is None or df.empty:
            continue
        hr = df.set_index("utc").resample("1H").mean().interpolate(limit_direction="both").reset_index()
        fig.add_trace(go.Scatter(x=hr["utc"], y=hr["pm25"], name=name))
    fig.update_layout(title="Multi-city PM2.5 Comparison", xaxis_title="UTC", yaxis_title="PM2.5 (µg/m³)", height=500)
    return fig

# -------------------------
# Main interactive flow (Notebook)
# -------------------------
print("=== AQI Guardian Agent — Upgraded & Fixed (Timezone-safe) ===")
prev = memory["prefs"].get("last_locations", [DEFAULT_LOC])
default_input = "; ".join(prev)
raw = input(f"Enter locations separated by ';'  (default: {default_input}): ").strip()
if raw == "":
    locations = prev
else:
    locations = [s.strip() for s in raw.split(";") if s.strip()]

print("\nOptional profile flags for planning (press Enter to skip):")
sensitive_input = input("Sensitive person? (y/N): ").strip().lower()
children_input = input("Has children? (y/N): ").strip().lower()
athlete_input = input("Athlete/heavy training? (y/N): ").strip().lower()
user_profile = {"sensitive": sensitive_input == "y", "has_children": children_input == "y", "athlete": athlete_input == "y"}

city_results = []

for loc in locations:
    print(f"\n--- Processing: {loc} ---")
    coords = geocode(loc)
    if not coords:
        print("Could not geocode:", loc)
        continue
    lat, lon = coords
    print(f"Coordinates: {lat:.4f}, {lon:.4f}")
    pm_df = fetch_pm25(lat, lon, hours=96)  # last 4 days
    weather_df = fetch_weather(lat, lon, hours_past=72, hours_future=48)

    if pm_df is None or pm_df.empty:
        print("No PM2.5 data found for this location.")
    else:
        print(f"Fetched {len(pm_df)} PM2.5 records (latest at {pm_df['utc'].max()}).")

    pred_df = train_predictor_hourly(pm_df, hours_ahead=24)
    if pred_df is not None:
        print("Forecast ready for next 24 hours.")
    else:
        print("Not enough data to train forecast — skipping forecast.")

    latest_pm = None
    latest_aqi = None
    latest_time = None
    if pm_df is not None and not pm_df.empty:
        latest_row = pm_df.iloc[-1]
        latest_pm = float(latest_row["pm25"])
        latest_time = latest_row["utc"]
        latest_aqi = pm25_to_aqi(latest_pm)

    # Find nearest weather snapshot safely (tz-aware)
    weather_snapshot = None
    if weather_df is not None and not weather_df.empty:
        # ensure weather_df UTC tz-aware (fetch_weather already does this)
        ref = latest_time if latest_time is not None else pd.Timestamp.utcnow()
        # make ref tz-aware UTC
        if getattr(ref, "tzinfo", None) is None:
            ref = pd.Timestamp(ref).tz_localize("UTC")
        else:
            ref = pd.Timestamp(ref).tz_convert("UTC")
        # ensure weather_df utc is tz-aware (already done in fetch_weather)
        # compute index of nearest
        diffs = (weather_df["utc"] - ref).abs()
        if not diffs.empty:
            idx = diffs.idxmin()
            weather_snapshot = weather_df.iloc[idx].to_dict()

    temp = weather_snapshot.get("temp") if weather_snapshot is not None else None
    rh = weather_snapshot.get("rh") if weather_snapshot is not None else None

    risk_score, risk_label = (None, "Unknown")
    if latest_aqi is not None:
        risk_score, risk_label = health_risk_score(latest_aqi, temp, rh)

    planner = planner_agent(pred_df if pred_df is not None else pm_df, user_profile=user_profile)
    source_est = estimate_source(pm_df, weather_df, spike_threshold=1.6)

    result = {
        "name": loc,
        "lat": lat, "lon": lon,
        "pm_df": pm_df,
        "weather_df": weather_df,
        "pred_df": pred_df,
        "latest_pm": latest_pm,
        "latest_aqi": latest_aqi,
        "latest_time": str(latest_time) if latest_time is not None else None,
        "temp": temp, "rh": rh,
        "risk_score": risk_score, "risk_label": risk_label,
        "planner": planner,
        "source_est": source_est
    }
    city_results.append(result)

    print(f"Latest PM2.5: {latest_pm if latest_pm is not None else 'N/A'}")
    print(f"Latest AQI (PM2.5): {latest_aqi if latest_aqi is not None else 'N/A'} ({aqi_category(latest_aqi) if latest_aqi is not None else 'N/A'})")
    print(f"Health risk score: {risk_score} ({risk_label})")
    if isinstance(planner, dict) and "note" in planner:
        print("Planner:", planner["note"])
    else:
        for p, d in planner.items():
            print(f"  {p}: avg AQI {d['avg_aqi']} -> actions: {(d['actions'][:2])}")
    print("Source estimator:", source_est.get("suggestion", source_est.get("note")))
    memory["history"][loc] = {"fetched_at": str(dt.datetime.utcnow()), "sample": (pm_df.tail(24).to_dict(orient="records") if pm_df is not None else [])}
    save_memory(memory)

# -------------------------
# Multi-city plotting
# -------------------------
if city_results:
    print("\nRendering multi-city comparison plot...")
    fig = plot_multi_city(city_results)
    fig.show()

# -------------------------
# Detailed report & visual per city
# -------------------------
for res in city_results:
    print("\n==============================")
    print("City:", res["name"])
    print("==============================")
    print("Coordinates:", res["lat"], res["lon"])
    print("Latest reading:", res["latest_pm"], "at", res["latest_time"])
    print("AQI:", res["latest_aqi"], aqi_category(res["latest_aqi"]) if res["latest_aqi"] is not None else "")
    print("Health Risk Score:", res["risk_score"], res["risk_label"])
    print("\nPlanner recommendations:")
    if isinstance(res["planner"], dict) and "note" in res["planner"]:
        print(" ", res["planner"]["note"])
    else:
        for period, info in res["planner"].items():
            print(f" - {period}:")
            print(f"    avg PM2.5 {info['avg_pm25']}, avg AQI {info['avg_aqi']}")
            for act in info["actions"]:
                print(f"      • {act}")
    print("\nPollution source inference:")
    print(" ", res["source_est"].get("suggestion", res["source_est"].get("note")))

    fig = plot_observed_and_forecast(res["pm_df"], res["pred_df"], title=f"PM2.5 — {res['name']}")
    fig.show()

    report_lines = []
    report_lines.append(f"AQI Guardian Report — {res['name']}")
    report_lines.append(f"Coords: {res['lat']}, {res['lon']}")
    report_lines.append(f"Latest PM2.5: {res['latest_pm']}")
    report_lines.append(f"Latest AQI: {res['latest_aqi']} ({aqi_category(res['latest_aqi']) if res['latest_aqi'] is not None else 'N/A'})")
    report_lines.append(f"Health Risk Score: {res['risk_score']} ({res['risk_label']})")
    report_lines.append("\nPlanner Summary:")
    if isinstance(res["planner"], dict) and "note" in res["planner"]:
        report_lines.append(" " + res["planner"]["note"])
    else:
        for period, info in res["planner"].items():
            report_lines.append(f" {period}: avg PM2.5 {info['avg_pm25']}, avg AQI {info['avg_aqi']}")
            for act in info["actions"]:
                report_lines.append("   - " + act)
    report_lines.append("\nSource Estimation:")
    report_lines.append(" " + res["source_est"].get("suggestion", res["source_est"].get("note")))
    fname = f"aqi_report_{res['name'].replace(' ', '_')}.txt"
    with open(fname, "w") as f:
        f.write("\n".join([str(x) for x in report_lines]))
    print(f"Report saved as {fname}")

print("\nAll done. Memory saved to", MEMORY_FILE)



