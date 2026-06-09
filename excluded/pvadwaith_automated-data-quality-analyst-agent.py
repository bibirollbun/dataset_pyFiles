import pandas as pd
import numpy as np

current_data = {
    "id": list(range(1,16)),
    "age": [25,31,29,42,35,np.nan,27,27,51,23,38,44,30,np.nan,47],
    "income": [42000,52000,np.nan,88000,54000,61000,47000,47000,99000,40000,72000,85000,50000,70000,np.nan],
    "gender": ["Male","Female","Male","Female",np.nan,"Male","Female","Female","Other","Male","Female","Male","Female","Male","Female"],
    "city": ["Bangalore","Chennai","Mumbai","Delhi","Hyderabad","Pune","Bangalore","Bangalore","Mumbai","Chennai","Delhi","Hyderabad","Bangalore","Mumbai","Delhi"],
    "joined_date": ["2022-05-12","2021-11-03","2023-02-18","2019-08-21","2020-01-10","2022-10-05","2023-06-15","2023-06-15","2018-12-30","2023-01-01","2020-05-05","2021-04-22","2022-07-07","2023-03-28","2019-10-16"],
    "purchase_amount": [230,np.nan,180,760,300,np.nan,210,210,1450,95,480,np.nan,250,310,990],
    "is_subscribed": ["Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Maybe","Yes","No","Yes","Yes"],
    "last_login_days": [2,5,7,1,12,np.nan,3,3,20,1,4,6,2,np.nan,16]
}

baseline_data = {
    "id": list(range(1,16)),
    "age": [24,32,29,41,36,28,27,50,22,39,44,31,35,46,40],
    "income": [40000,52000,48000,88000,52000,54000,46000,96000,38000,70000,82000,48000,57000,90000,75000],
    "gender": ["Male","Female","Male","Female","Female","Male","Female","Female","Male","Female","Male","Female","Female","Male","Female"],
    "city": ["Bangalore","Chennai","Mumbai","Delhi","Hyderabad","Pune","Bangalore","Mumbai","Chennai","Delhi","Hyderabad","Bangalore","Mumbai","Delhi","Chennai"],
    "joined_date": ["2021-04-09","2021-09-12","2022-11-11","2019-05-22","2020-02-01","2022-09-18","2023-03-10","2019-12-25","2023-01-10","2020-04-21","2021-02-14","2022-05-28","2023-06-03","2019-09-10","2020-11-16"],
    "purchase_amount": [180,220,150,690,250,200,190,880,85,430,600,200,240,830,500],
    "is_subscribed": ["Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No"],
    "last_login_days": [5,4,8,2,11,5,4,17,2,5,7,3,4,15,6]
}

df_current = pd.DataFrame(current_data)
df_baseline = pd.DataFrame(baseline_data)

df_current.to_csv("sample_current.csv", index=False)
df_baseline.to_csv("sample_baseline.csv", index=False)

print("Files created:")
df_current.head(), df_baseline.head()



# ============================
# PARAMETERS (edit only these)
# ============================

INPUT_PATH = "sample_current.csv"           # if working in /kaggle/working
BASELINE_PATH = "sample_baseline.csv"       # set to None if not using baseline
OUT_DIR = "examples"                        # folder where report + visuals will be saved
GEMINI_MODEL = "gemini-2.5-flash"           # model used for structured recommendations



import random, numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# print package versions for judges
import sys, pandas as pd
print("python", sys.version.split()[0], "pandas", pd.__version__)


# run in notebook cell
!pip install -q -U google-genai pandas matplotlib scikit-learn




from kaggle_secrets import UserSecretsClient
import os

secret_client = UserSecretsClient()
try:
    gemini_key = secret_client.get_secret("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = gemini_key
except Exception:
    print("GEMINI key not found in Kaggle Secrets — Gemini calls will be skipped.")



# Cell: imports and tiny helpers
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from IPython.display import display, Image, JSON

# GEMINI client (reads GEMINI_API_KEY from env)
client = genai.Client()

OUT_DIR = "examples"
IMAGES_DIR = f"{OUT_DIR}/images"
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

def safe_json_load(s):
    try:
        return json.loads(s)
    except Exception:
        # try to extract first JSON-like substring
        import re
        m = re.search(r'(\{.*\}|\[.*\])', s, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return {"raw_text": s}
        return {"raw_text": s}



# Cell: load data (use a sample CSV in the repo or upload to Kaggle data)
#INPUT_PATH = "/kaggle/working/sample_current.csv"
# change if you uploaded a different file
if not Path(INPUT_PATH).exists():
    print("Upload your CSV to `data/` or change INPUT_PATH.")
df = pd.read_csv(INPUT_PATH)
print("Loaded:", INPUT_PATH, "shape:", df.shape)
df.head()



# Cell: profile dataframe (keeps it compact but useful)
def profile_dataframe(df: pd.DataFrame) -> dict:
    profile = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": {},
        "n_duplicates": int(df.duplicated().sum())
    }
    for c in df.columns:
        s = df[c]
        meta = {
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            meta.update({
                "mean": None if s.dropna().empty else float(s.mean()),
                "std": None if s.dropna().empty else float(s.std()),
                "min": None if s.dropna().empty else float(s.min()),
                "max": None if s.dropna().empty else float(s.max())
            })
        profile["columns"][c] = meta
    return profile

profile = profile_dataframe(df)
print("Profile summary: rows", profile["n_rows"], "cols", profile["n_columns"])
# quick view of top issues
col_issues = {c:v for c,v in profile['columns'].items() if v['pct_missing']>0 or v['n_unique']<5}
len(col_issues), list(col_issues.items())[:3]



# Cell: generate and display visuals for quick demo
def save_hist(series, out_dir, name):
    path = Path(out_dir)/f"{name}_hist.png"
    plt.figure()
    try:
        series = series.dropna()
        plt.hist(series, bins=30)
        plt.title(name)
        plt.savefig(path, bbox_inches="tight")
    finally:
        plt.close()
    return str(path)

def save_missing_map(df, out_dir):
    path = Path(out_dir)/"missingness_map.png"
    plt.figure(figsize=(10, max(2, len(df.columns)*0.25)))
    plt.imshow(df.isna().T, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.xlabel("rows")
    plt.title("Missingness map")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)

# generate visuals for first up to 4 numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
visual_paths = []
for c in num_cols:
    p = save_hist(df[c], IMAGES_DIR, c.replace(" ", "_"))
    visual_paths.append(p)

# missingness map
visual_paths.append(save_missing_map(df, IMAGES_DIR))

# display examples inline
for p in visual_paths:
    display(Image(p, width=600))



# Cell: ask Gemini to take the profile and return structured recommendations
prompt_intro = """
You are a concise data-quality assistant. Input: `profile_json` that lists
columns with dtype, pct_missing, n_unique, and simple stats.
Return a JSON object with key "recommendations" containing a list where each entry:
- column: <column name>
- issue: one-line issue summary
- recommendation: one-line suggested remediation (e.g., median impute, drop column)
Do not return any extra text outside the JSON.
"""

input_payload = {
    "prompt_intro": prompt_intro,
    "profile": profile
}

# Compose the content; using string because some SDK versions expect text
contents = prompt_intro + "\n\n" + json.dumps(profile)

resp = client.models.generate_content(
    model="gemini-2.5-flash",  # adjust model as desired
    contents=contents
)

# Extract text and parse robustly
raw_text = getattr(resp, "text", None) or getattr(resp, "output", None) or str(resp)
if isinstance(raw_text, list): raw_text = raw_text[0]
if hasattr(resp, "text") and not raw_text:
    raw_text = resp.text

reco_json = safe_json_load(raw_text)
print("Parsed Gemini response (keys):", list(reco_json.keys()) if isinstance(reco_json, dict) else type(reco_json))
reco_json



# Cell: assemble and save final report
report = {
    "profile": profile,
    "recommendations_from_gemini": reco_json,
    "visuals": visual_paths
}
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
with open(f"{OUT_DIR}/sample_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Saved report:", f"{OUT_DIR}/sample_report.json")
display(JSON(report))



# Cell: helper to ask Gemini again with stricter prompt if parsing failed
def ask_gemini_strict(profile, max_attempts=2):
    attempts=0
    while attempts<max_attempts:
        attempts+=1
        contents = "Return ONLY valid JSON (no commentary). Schema: {recommendations:[{column,issue,recommendation}]}\n\n" + json.dumps(profile)
        r = client.models.generate_content(model="gemini-2.5-flash", contents=contents)
        txt = getattr(r,"text", str(r))
        parsed = safe_json_load(txt)
        if isinstance(parsed, dict) and "recommendations" in parsed:
            return parsed
    return {"error": "could-not-parse", "raw": txt}

# try re-ask if needed
if not (isinstance(reco_json, dict) and "recommendations" in reco_json):
    reco_json = ask_gemini_strict(profile, max_attempts=2)
    print("Re-ask result keys:", reco_json.keys() if isinstance(reco_json, dict) else type(reco_json))



# Cell: very simple checks to include in your notebook for judges
errors = []
if profile["n_rows"] <= 0:
    errors.append("no-rows")
if profile["n_columns"] <= 0:
    errors.append("no-columns")
if not Path(OUT_DIR + "/sample_report.json").exists():
    errors.append("report-not-saved")

print("Basic checks OK" if not errors else "Checks failed: " + ", ".join(errors))


