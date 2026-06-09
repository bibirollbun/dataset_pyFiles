"""
A minimal "agent" that inspects the dataset and decides an execution plan.
It is intentionally rule-based to be deterministic and explainable.
"""
from tools import load_data, quick_stats, train_and_select, pairplot_image

def detect_task(df, target):
    # simple heuristic: if target is int with 2 unique values -> classification
    ser = df[target]
    if ser.dtype.kind in "biu" and ser.nunique() == 2:
        return "classification"
    if ser.dtype.kind in "biu" and ser.nunique() > 20:
        return "regression"
    # fallback: if target numeric -> regression else classification
    return "regression" if ser.dtype.kind in "fiu" else "classification"

def run_agent(data_path, target_column):
    res = load_data(data_path)
    if res["status"]!="ok":
        return res
    df = res["dataframe"]
    plan = []
    # 1. quick summary
    plan.append("quick_stats")
    stats = quick_stats(df)
    # 2. detect task
    task = detect_task(df, target_column)
    plan.append(f"detected_task:{task}")
    # 3. create pairplot for numeric subset (for report/UI)
    plan.append("pairplot")
    img = pairplot_image(df)
    # 4. train models and produce report
    plan.append("train_and_select")
    train_res = train_and_select(df, target_column, task=task)
    return {
        "status":"ok",
        "plan_executed": plan,
        "task": task,
        "summary": stats,
        "pairplot_base64": img["image_base64"],
        "training": train_res
    }


"""
A minimal "agent" that inspects the dataset and decides an execution plan.
It is intentionally rule-based to be deterministic and explainable.
"""
%%writefile tools.py
# paste your full tools.py content here

from tools import load_data, quick_stats, train_and_select, pairplot_image

def detect_task(df, target):
    # simple heuristic: if target is int with 2 unique values -> classification
    ser = df[target]
    if ser.dtype.kind in "biu" and ser.nunique() == 2:
        return "classification"
    if ser.dtype.kind in "biu" and ser.nunique() > 20:
        return "regression"
    # fallback: if target numeric -> regression else classification
    return "regression" if ser.dtype.kind in "fiu" else "classification"

def run_agent(data_path, target_column):
    res = load_data(data_path)
    if res["status"]!="ok":
        return res
    df = res["dataframe"]
    plan = []
    # 1. quick summary
    plan.append("quick_stats")
    stats = quick_stats(df)
    # 2. detect task
    task = detect_task(df, target_column)
    plan.append(f"detected_task:{task}")
    # 3. create pairplot for numeric subset (for report/UI)
    plan.append("pairplot")
    img = pairplot_image(df)
    # 4. train models and produce report
    plan.append("train_and_select")
    train_res = train_and_select(df, target_column, task=task)
    return {
        "status":"ok",
        "plan_executed": plan,
        "task": task,
        "summary": stats,
        "pairplot_base64": img["image_base64"],
        "training": train_res
    }


import streamlit as st
import pandas as pd
from src.agent import run_agent
from src.config import DEFAULT_DATA_PATH, REPORT_PATH

st.set_page_config(page_title="AutoML Agent Demo", layout="wide")
st.title("AutoML Agent — Demo")

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
if uploaded is not None:
    data_path = uploaded
    df = pd.read_csv(uploaded)
else:
    st.info("No file uploaded — using sample dataset.")
    data_path = DEFAULT_DATA_PATH
    df = pd.read_csv(DEFAULT_DATA_PATH)

st.write("Dataset preview:")
st.dataframe(df.head(10))

target = st.selectbox("Select target column", options=df.columns.tolist())

if st.button("Run AutoML Agent"):
    with st.spinner("Agent is planning and running..."):
        result = run_agent(data_path, target)
    if result["status"]=="ok":
        st.success("Agent completed!")
        st.write("Plan:", result["plan_executed"])
        st.write("Detected task:", result["task"])
        st.subheader("Summary")
        st.write(pd.DataFrame(result["summary"]["summary_table"]))
        st.subheader("Pairplot (numeric sample)")
        st.image("data:image/png;base64," + result["pairplot_base64"])
        st.subheader("Report")
        st.text(open(result["training"]["report_path"]).read())
        st.download_button("Download report", data=open(result["training"]["report_path"]).read(), file_name="report.md")
    else:
        st.error("Agent failed: " + str(result))


"""
Model training helpers: basic pipelines for classification & regression.
This is intentionally simple so it runs on Kaggle CPU kernels.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from lightgbm import LGBMClassifier, LGBMRegressor

def _column_types(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols

def _build_pipeline(model, X):
    num_cols, cat_cols = _column_types(X)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ]) if len(cat_cols)>0 else "passthrough"
    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")
    pipe = Pipeline([("preproc", preproc), ("model", model)])
    return pipe

def auto_train_models(X, y, task="classification", random_state=42):
    models = []
    if task == "classification":
        candidates = [
            ("logreg", LogisticRegression(max_iter=1000, random_state=random_state)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=random_state)),
            ("lgbm", LGBMClassifier(n_estimators=500, random_state=random_state))
        ]
    else:
        candidates = [
            ("ridge", Ridge(random_state=random_state)),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=random_state)),
            ("lgbm", LGBMRegressor(n_estimators=500, random_state=random_state))
        ]

    for name, mdl in candidates:
        pipe = _build_pipeline(mdl, X)
        pipe.fit(X, y)
        models.append({"name": name, "model": pipe})
    return models

def evaluate_model(pipe, X_test, y_test, task="classification"):
    preds = pipe.predict(X_test)
    if task == "classification":
        if hasattr(pipe, "predict_proba"):
            try:
                probs = pipe.predict_proba(X_test)[:,1]
                return roc_auc_score(y_test, probs)
            except Exception:
                return accuracy_score(y_test, preds)
        else:
            return accuracy_score(y_test, preds)
    else:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return -rmse  # higher better for selection


def build_markdown_report(df, target, models_info, best):
    lines = []
    lines.append("# AutoML Agent Report\n")
    lines.append("## Dataset\n")
    lines.append(f"- Rows: {df.shape[0]}\n- Columns: {df.shape[1]}\n")
    lines.append(f"Target: **{target}**\n")
    lines.append("## Models trained\n")
    for m in models_info:
        lines.append(f"- **{m['name']}** - test_score: {m.get('test_score', 'n/a')}\n")
    lines.append("\n## Best Model\n")
    lines.append(f"- Name: **{best['name']}**\n")
    lines.append(f"- Test score: **{best.get('test_score')}**\n")
    lines.append("\n## Notes\n")
    lines.append("This report was autogenerated by the AutoML Agent. Use the model at `outputs/best_model.joblib` for inference.\n")
    return "\n".join(lines)


"""
CLI to run the agent end-to-end. Example:
python run_agent.py --data data/sample_heart.csv --target target
"""
import argparse
from src.agent import run_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sample_heart.csv")
    parser.add_argument("--target", type=str, required=True)
    args = parser.parse_args()
    out = run_agent(args.data, args.target)
    if out["status"]=="ok":
        print("Plan executed:", out["plan_executed"])
        print("Task detected:", out["task"])
        print("Report created at:", out["training"]["report_path"])
        print("Best model saved at:", out["training"]["model_path"])
    else:
        print("Agent failed:", out)

if __name__=="__main__":
    main()


"""
Tools available to the agent. Each function returns a dict with 'status' and
relevant outputs so the agent can plan next steps.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from .pipeline import auto_train_models, evaluate_model
from .report import build_markdown_report
from .config import REPORT_PATH, MODEL_PATH

def load_data(path):
    df = pd.read_csv(path)
    return {"status":"ok", "dataframe": df, "n_rows": df.shape[0], "n_cols": df.shape[1]}

def quick_stats(df, max_rows=10):
    desc = df.describe(include='all', datetime_is_numeric=True).T
    missing = df.isnull().sum().to_frame("missing")
    types = df.dtypes.to_frame("dtype")
    summary = pd.concat([types, missing, desc], axis=1)
    # convert to serializable dict
    return {"status":"ok", "summary_table": summary.reset_index().rename(columns={"index":"feature"}).to_dict(orient="records")}

def pairplot_image(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:6]
    sns.set(style="whitegrid")
    plt.clf()
    g = sns.pairplot(df[numeric_cols].dropna().sample(min(500, len(df))), diag_kind="kde")
    buf = io.BytesIO()
    g.savefig(buf, format="png")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf8")
    return {"status":"ok", "image_base64": data}

def train_and_select(df, target, task="classification", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if task=="classification" else None)
    models_info = auto_train_models(X_train, y_train, task=task, random_state=random_state)
    # evaluate candidates
    best = None
    for m in models_info:
        score = evaluate_model(m["model"], X_test, y_test, task=task)
        m["test_score"] = score
        if best is None or score > best["test_score"]:
            best = m
    # save best model
    import joblib
    joblib.dump(best["model"], MODEL_PATH)
    report_md = build_markdown_report(df, target, models_info, best)
    with open(REPORT_PATH, "w") as f:
        f.write(report_md)
    return {"status":"ok", "models": models_info, "best": best, "report_path": REPORT_PATH, "model_path": MODEL_PATH}


import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE, "data")
OUTPUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "sample_heart.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.md")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")

