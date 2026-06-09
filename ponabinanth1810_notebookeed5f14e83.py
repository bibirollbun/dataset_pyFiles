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


!pip install -q fastapi uvicorn[standard] scikit-learn pandas joblib prometheus_client kaggle google-cloud-storage requests



# Create kaggle.json (only in private notebooks)
import os, json
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
kaggle_token = "<YOUR_KAGGLE_TOKEN_OR_USE_ENV>"
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"token": kaggle_token}, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)



from prometheus_client import Summary, Counter, REGISTRY

def get_metric(metric_class, name, desc):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return metric_class(name, desc)

REQUEST_TIME  = get_metric(Summary, 'inference_latency_seconds', "Latency of inference")
PREDICTIONS   = get_metric(Counter, 'predictions_total', "Total predictions served")
PIPELINE_RUNS = get_metric(Counter, 'pipeline_runs_total', "Total ETL pipeline runs executed")



# minimal pipeline (paste in a cell)
import os, time, joblib, json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

WORKDIR = "/kaggle/working/enterprise_agent"
os.makedirs(WORKDIR, exist_ok=True)
RAW_CSV = os.path.join(WORKDIR, "raw.csv")
PROCESSED_CSV = os.path.join(WORKDIR, "processed.csv")
MODEL_FILE = os.path.join(WORKDIR, "model.joblib")
REGISTRY_FILE = os.path.join(WORKDIR, "registry.json")

# Example: if dataset already attached via Kaggle "Add data", load it:
# df = pd.read_csv("/kaggle/input/<dataset-folder>/<file>.csv")
# For demo: create tiny synthetic dataframe if not present
if not os.path.exists(RAW_CSV):
    df = pd.DataFrame({
        "country": ["US","FR","US","IT","FR"],
        "price": [20, 35, 15, 50, 40],
        "points": [87, 92, 85, 95, 90]
    })
    df.to_csv(RAW_CSV, index=False)

def preprocess(in_path=RAW_CSV, out_path=PROCESSED_CSV):
    df = pd.read_csv(in_path)
    if 'points' not in df.columns:
        raise RuntimeError("No 'points' target found. Adapt this cell to your dataset.")
    df = df[['country','price','points']].dropna()
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    df.to_csv(out_path, index=False)
    print("Preprocessed:", out_path)
    return out_path

def train_and_save(processed_csv=PROCESSED_CSV, model_path=MODEL_FILE):
    df = pd.read_csv(processed_csv)
    X = pd.get_dummies(df[['country']])
    X['price'] = df['price']
    y = df['points']
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds) if len(y_val)>0 else None
    rmse = mean_squared_error(y_val, preds, squared=False) if len(y_val)>0 else None
    artifact = {"features": list(X.columns), "metrics": {"r2": r2, "rmse": rmse}, "trained_at": time.time()}
    joblib.dump({"model": model, "artifact": artifact}, model_path)
    # save registry
    reg = {}
    if os.path.exists(REGISTRY_FILE):
        reg = json.load(open(REGISTRY_FILE))
    ver = str(int(time.time()))
    reg[ver] = artifact
    json.dump(reg, open(REGISTRY_FILE, "w"), indent=2)
    print("Model saved:", model_path, "metrics:", artifact['metrics'])
    return model_path, artifact

# run cells stepwise:
processed = preprocess()
model_path, artifact = train_and_save(processed)



import joblib
data = joblib.load(MODEL_FILE)
print("Features:", data['artifact']['features'])
print("Metrics:", data['artifact']['metrics'])



import pandas as pd
import joblib
data = joblib.load(MODEL_FILE)
model = data['model']
features = data['artifact']['features']

def predict(country, price):
    row = {f:0 for f in features}
    col = f"country_{country}"
    if col in row: row[col]=1
    if 'price' in row: row['price'] = price
    X = pd.DataFrame([row], columns=features)
    return float(model.predict(X)[0])

print("Sample prediction:", predict("US", 30))


