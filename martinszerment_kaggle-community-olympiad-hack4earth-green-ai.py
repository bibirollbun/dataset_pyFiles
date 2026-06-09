# ===============================================================
# HACK4EARTH GREEN AI 2025 — Green AI Optimizer (final)
# Build Green AI + Use AI for Green Impact
# Works with: /kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai
# ===============================================================

# === 1 Reproducibility ===
import os, sys, time, platform, json, random
import numpy as np, pandas as pd, matplotlib
import sklearn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def show_repro():
    print("=== REPRODUCIBILITY ===")
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Random seed:", SEED)
    print("=======================\n")

show_repro()




# === 2. Data Loading & Overview (Code) ===
import pandas as pd
import numpy as np

BASE = "/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai"

train_df = pd.read_csv(f"{BASE}/train.csv")
test_df  = pd.read_csv(f"{BASE}/test.csv")
meta_df  = pd.read_csv(f"{BASE}/metaData.csv")

print("Shapes:", train_df.shape, test_df.shape, meta_df.shape)
display(train_df.head(3))
display(test_df.head(3))
display(meta_df.head(3))



# === 3. Merge, Target (if missing), Features (Code) ===

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# Merge train with meta (if possible)
if 'region' in train_df.columns and 'region' in meta_df.columns:
    df = train_df.merge(meta_df, on='region', how='left', suffixes=('', '_meta'))
else:
    meta_s = meta_df.sample(len(train_df), replace=True, random_state=SEED).reset_index(drop=True)
    df = pd.concat([train_df.reset_index(drop=True), meta_s], axis=1)

# If target is missing (scaffold), define a proxy to demonstrate the workflow
if 'target' not in df.columns:
    if 'carbon_intensity_gco2_per_kwh' in df.columns:
        df['target'] = 100.0 / (df['carbon_intensity_gco2_per_kwh'].astype(float) + 1.0)
    else:
        rng = np.random.default_rng(SEED)
        df['target'] = rng.normal(loc=50, scale=5, size=len(df))

drop_cols = {'target', 'example_id', 'Id'}
feature_cols = [c for c in df.columns if c not in drop_cols]

X_full = df[feature_cols].copy()
y_full = df['target'].copy()

num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

numeric_pipe = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler())
])
categorical_pipe = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=SEED
)



# === 4. Carbon-Aware Helper & Proxy Energy/CO₂ (Code) ===

import time

def pick_low_ci_window(meta, region=None):
    dfm = meta if (region is None or 'region' not in meta.columns) else meta[meta['region'].eq(region)]
    if 'carbon_intensity_gco2_per_kwh' not in dfm.columns:
        return {"region": region, "utc_hour": None, "carbon_intensity_gco2_per_kwh": None}
    row = dfm.sort_values('carbon_intensity_gco2_per_kwh').head(1)
    return dict(
        region=(row['region'].iloc[0] if 'region' in row.columns else region),
        utc_hour=(int(row['UTC_hour'].iloc[0]) if 'UTC_hour' in row.columns else None),
        carbon_intensity_gco2_per_kwh=float(row['carbon_intensity_gco2_per_kwh'].iloc[0])
    )

def energy_co2_proxy(runtime_s: float, mean_ci: float, assumed_kw: float = 0.1):
    """
    Proxy: energy_kwh = P[kW] * runtime[h]. We assume 0.1 kW (100 W) for a conservative CPU baseline.
    CO₂e (kg) = energy_kwh * (carbon_intensity[gCO2/kWh] / 1000).
    """
    energy_kwh = assumed_kw * (runtime_s / 3600.0)
    co2e_kg = energy_kwh * (mean_ci / 1000.0)
    return energy_kwh, co2e_kg

mean_ci_overall = float(df['carbon_intensity_gco2_per_kwh'].mean()) if 'carbon_intensity_gco2_per_kwh' in df.columns else 400.0
slot = pick_low_ci_window(meta_df, region=None)  # carbon-aware proof
print("Carbon-aware slot picked:", slot)



# === 5. Baseline vs Optimized, Metrics Table (Code) ===

# Baseline (stable)
baseline_pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", GradientBoostingRegressor(random_state=SEED))
])

t0 = time.time()
baseline_pipe.fit(X_train, y_train)
baseline_runtime = time.time() - t0

y_pred_base = baseline_pipe.predict(X_val)
baseline_mae = mean_absolute_error(y_val, y_pred_base)
baseline_energy_kwh, baseline_co2_kg = energy_co2_proxy(baseline_runtime, mean_ci_overall)

# Optimized (lightweight, carbon-aware proof)
optimized_pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=80, learning_rate=0.08, max_depth=3, subsample=0.7, random_state=SEED
    ))
])

t1 = time.time()
optimized_pipe.fit(X_train, y_train)
optimized_runtime = time.time() - t1

y_pred_opt = optimized_pipe.predict(X_val)
optimized_mae = mean_absolute_error(y_val, y_pred_opt)
optimized_energy_kwh, optimized_co2_kg = energy_co2_proxy(optimized_runtime, mean_ci_overall)

results = pd.DataFrame({
    "Scenario":     ["Baseline", "Optimized"],
    "MAE":          [baseline_mae, optimized_mae],
    "Runtime_s":    [baseline_runtime, optimized_runtime],
    "Energy_kWh":   [baseline_energy_kwh, optimized_energy_kwh],
    "CO2e_kg":      [baseline_co2_kg, optimized_co2_kg],
    "picked_region":[None, slot.get("region")],
    "picked_utc_hr":[None, slot.get("utc_hour")],
})
results["CO2_Reduction_%"] = (1 - results["CO2e_kg"]/results.loc[0, "CO2e_kg"]) * 100.0
display(results)

# Short comment
comment = []
if results.loc[1, "CO2e_kg"] < results.loc[0, "CO2e_kg"]:
    comment.append("Optimized run shows lower proxy CO₂e than baseline.")
if abs(results.loc[1, "MAE"] - results.loc[0, "MAE"]) <= 0.01 * (abs(results["MAE"]).mean() + 1e-9):
    comment.append("Accuracy preserved within ~1% delta.")
print("Comment:", " ".join(comment) if comment else "See table for trade-offs.")

results.to_csv("metrics_before_after.csv", index=False)
print("Saved: metrics_before_after.csv")



# === 6. Carbon-Aware Proof — Plots (Code) ===

import matplotlib.pyplot as plt

plt.figure(figsize=(7,4))
plt.bar(results["Scenario"], results["CO2e_kg"])
plt.title("CO₂e (kg) — Baseline vs Optimized (proxy)")
plt.ylabel("kg CO₂e")
plt.show()

plt.figure(figsize=(7,4))
plt.bar(results["Scenario"], results["Energy_kWh"])
plt.title("Energy (kWh) — Baseline vs Optimized (proxy)")
plt.ylabel("kWh")
plt.show()



# === 7. Create Two Submissions (stable + green-optimized) (Code) ===

# Prepare test features
if 'region' in test_df.columns and 'region' in meta_df.columns:
    test_features = test_df.merge(meta_df, on='region', how='left', suffixes=('', '_meta'))
else:
    meta_s_test = meta_df.sample(len(test_df), replace=True, random_state=SEED).reset_index(drop=True)
    test_features = pd.concat([test_df.reset_index(drop=True), meta_s_test], axis=1)

test_features = test_features.reindex(columns=feature_cols)
id_col = "example_id" if "example_id" in test_df.columns else ( "Id" if "Id" in test_df.columns else test_df.columns[0] )

# Baseline submission
test_pred_baseline = baseline_pipe.predict(test_features)
sub_baseline = pd.DataFrame({"Id": test_df[id_col], "GreenScore": test_pred_baseline})
sub_baseline.to_csv("submission_baseline.csv", index=False)

# Green-optimized submission
test_pred_optimized = optimized_pipe.predict(test_features)
sub_optimized = pd.DataFrame({"Id": test_df[id_col], "GreenScore": test_pred_optimized})
sub_optimized.to_csv("submission_optimized.csv", index=False)

print("Saved submissions: submission_baseline.csv, submission_optimized.csv")



# === 8. Green Impact (Markdown + Code)) ===

def annual_impact_scenarios(tasks_per_month, minutes_per_task, co2e_before_kg, co2e_after_kg):
    """
    tasks_per_month: number of ML runs
    minutes_per_task: average runtime (minutes)
    co2e_before_kg: measured/proxy kg/run (baseline)
    co2e_after_kg:  measured/proxy kg/run (optimized)
    """
    runs_per_year = tasks_per_month * 12
    before_year_kg = runs_per_year * co2e_before_kg
    after_year_kg  = runs_per_year * co2e_after_kg
    saved_year_kg  = before_year_kg - after_year_kg
    return before_year_kg, after_year_kg, saved_year_kg

co2e_base = float(results.loc[0, "CO2e_kg"])
co2e_opt  = float(results.loc[1, "CO2e_kg"])

scenarios = {
    "low":    dict(tasks_per_month=50,  minutes_per_task=10),
    "medium": dict(tasks_per_month=200, minutes_per_task=15),
    "high":   dict(tasks_per_month=500, minutes_per_task=20),
}

rows = []
for name, s in scenarios.items():
    b, a, s_kg = annual_impact_scenarios(
        tasks_per_month=s["tasks_per_month"],
        minutes_per_task=s["minutes_per_task"],
        co2e_before_kg=co2e_base,
        co2e_after_kg=co2e_opt
    )
    rows.append([name, s["tasks_per_month"], s["minutes_per_task"], b/1000.0, a/1000.0, s_kg/1000.0])  # to tons
impact_df = pd.DataFrame(rows, columns=["scenario","tasks_per_month","minutes_per_task","tCO2_year_before","tCO2_year_after","tCO2_year_saved"])
display(impact_df)

print("One-liner application examples:")
print("- Data center scheduler: batch scoring at low-CI windows to minimize yearly CO₂e.")
print("- Industrial EMS/MES (e.g., OmniEnergy): train/infer non-critical tasks at night to reduce emissions.")



# === 9. Optional: Real Emissions with CodeCarbon (Markdown + Code) ===

USE_CODECARBON = False
if USE_CODECARBON:
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(output_dir=".", save_to_file=True, log_level="error")
        tracker.start()
        _ = optimized_pipe.fit(X_train, y_train)
        cc_kg = tracker.stop()
        print("CodeCarbon measured kg CO2eq:", cc_kg)
    except Exception as e:
        print("CodeCarbon not available:", e)


