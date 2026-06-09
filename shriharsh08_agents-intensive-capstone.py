#  Imports & helpers
import os
import pickle
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# do NOT use XGBoost for stability
_HAS_XGB = False

def log(msg: str):
    print(f"[LOG] {msg}")

log(f"Imports done. xgboost available: {_HAS_XGB}")



#  load_data
def load_data(path: str = None, fallback: str = "titanic") -> pd.DataFrame:
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        log(f"Loaded dataset from {path} with shape {df.shape}")
        return df
    if fallback == "titanic":
        df = sns.load_dataset("titanic")
        log(f"Loaded built-in Titanic dataset with shape {df.shape}")
        return df
    raise ValueError("Unsupported fallback dataset")



# eda_summary
def eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    summary['shape'] = df.shape
    summary['dtypes'] = df.dtypes.astype(str).to_dict()
    missing = df.isna().sum()
    summary['missing'] = dict(missing[missing > 0].sort_values(ascending=False))
    num_df = df.select_dtypes(include=[np.number])
    summary['numeric_description'] = num_df.describe().to_dict() if not num_df.empty else {}
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    cat_summary = {}
    for col in cat_cols:
        cat_summary[col] = {
            'unique_values': int(df[col].nunique(dropna=True)),
            'top_values': df[col].value_counts(dropna=True).head(5).to_dict()
        }
    summary['categorical_summary'] = cat_summary
    summary['head'] = df.head(5).to_dict(orient='list')
    return summary



# missing_value_tool
def missing_value_tool(df: pd.DataFrame, numeric_strategy: str = "median", cat_strategy: str = "mode") -> Dict[str, Any]:
    res = {}
    miss = df.isna().sum()
    res['missing_summary'] = dict(miss[miss > 0])
    n = len(df)
    res['heavily_missing'] = [c for c,count in res['missing_summary'].items() if count / n > 0.7]
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df_clean[c].isna().sum() > 0:
            df_clean[c] = df_clean[c].fillna(df_clean[c].median() if numeric_strategy=='median' else df_clean[c].mean())
    cat_cols = df_clean.select_dtypes(include=['object','category']).columns
    for c in cat_cols:
        if df_clean[c].isna().sum() > 0:
            df_clean[c] = df_clean[c].fillna(df_clean[c].mode().iloc[0])
    res['strategy'] = f"numeric={numeric_strategy}, categorical={cat_strategy}"
    res['imputed_df'] = df_clean
    return res



# outlier_tool
def outlier_tool(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> Dict[str, Any]:
    result = {"summary": {}, "details": {}}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            result["summary"][col] = 0
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        low, high = q1 - iqr_multiplier*iqr, q3 + iqr_multiplier*iqr
        mask = (df[col] < low) | (df[col] > high)
        idx = df[mask].index.tolist()
        vals = df.loc[mask, col].dropna().unique().tolist()[:6]
        result["summary"][col] = len(idx)
        result["details"][col] = {"lower": float(low), "upper": float(high), "examples": vals}
    return result



# feature_engineering_tool
def feature_engineering_tool(df: pd.DataFrame, target_col: str = "survived") -> Dict[str, Any]:
    df2 = df.copy()
    drop_cols = [c for c in ['deck','alive','who','class'] if c in df2.columns]
    df2 = df2.drop(columns=drop_cols, errors='ignore')
    if target_col not in df2.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")
    y = df2[target_col].reset_index(drop=True)
    X = df2.drop(columns=[target_col]).reset_index(drop=True)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    scaler = StandardScaler()
    existing_num = [c for c in num_cols if c in X_enc.columns]
    if existing_num:
        X_enc[existing_num] = scaler.fit_transform(X_enc[existing_num])
    return {"X": X_enc, "y": y, "dropped_columns": drop_cols, "numeric_columns": num_cols, "categorical_columns": cat_cols, "scaler": scaler}



# task_detection
def task_detection(y) -> str:
    if not hasattr(y, "dtype"):
        y = pd.Series(y)
    unique_vals = pd.Series(y).dropna().unique()
    n_unique = len(unique_vals)
    if pd.api.types.is_integer_dtype(y) and n_unique <= 20:
        return "classification"
    if pd.api.types.is_numeric_dtype(y) and n_unique > 20:
        return "regression"
    return "classification"



# train_and_evaluate (supports classification OR regression automatically) - (no XGBoost)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_and_evaluate(X_df, y_series, task="classification", random_state=42):
    results = {}
    X = X_df.copy()
    y = pd.Series(y_series).reset_index(drop=True)

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=stratify)

    if task == "classification":
        # Logistic Regression
        try:
            lr = LogisticRegression(max_iter=2000)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            results['logistic'] = {
                'model': lr,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
        except Exception as e:
            results['logistic'] = {'error': str(e)}

        # Random Forest Classifier
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            results['rf'] = {
                'model': rf,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
        except Exception as e:
            results['rf'] = {'error': str(e)}

    else:  # regression
        try:
            lr_reg = LinearRegression()
            lr_reg.fit(X_train, y_train)
            y_pred = lr_reg.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            results['linear'] = {
                'model': lr_reg,
                'rmse': rmse,
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
        except Exception as e:
            results['linear'] = {'error': str(e)}

        try:
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=random_state)
            rf_reg.fit(X_train, y_train)
            y_pred = rf_reg.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            results['rf'] = {
                'model': rf_reg,
                'rmse': rmse,
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
        except Exception as e:
            results['rf'] = {'error': str(e)}

    results['X_test'] = X_test
    results['y_test'] = y_test.reset_index(drop=True)
    return results



# insight_tool
def insight_tool(eda_res, mv_res, out_res, fe_res, train_res):
    lines = []
    lines.append(f"Dataset shape: {eda_res['shape'][0]} rows, {eda_res['shape'][1]} cols")
    if mv_res.get('heavily_missing'):
        lines.append(f"Heavily missing columns: {mv_res['heavily_missing']}")
    if mv_res.get('missing_summary'):
        top = list(mv_res['missing_summary'].items())[:3]
        top_missing = ", ".join([f"{k}:{v}" for k,v in top])
        lines.append(f"Top missing counts: {top_missing}")
    out_summary = {k:v for k,v in out_res['summary'].items() if v>0}
    if out_summary:
        lines.append("Columns with notable outliers: " + ", ".join([f"{k}({v})" for k,v in out_summary.items()]))
    lines.append(f"Dropped columns during FE: {fe_res.get('dropped_columns', [])}")
    lines.append(f"Final feature matrix shape: {fe_res['X'].shape}")
    # Build model perf strings intelligently
    perf = []
    for name, info in train_res.items():
        if name in ['X_test','y_test']: continue
        if isinstance(info, dict) and 'error' in info:
            perf.append(f"{name}: ERROR")
        elif isinstance(info, dict):
            # classification metrics
            if 'accuracy' in info or 'f1' in info:
                perf.append(f"{name}: acc={info.get('accuracy',0):.3f}, f1={info.get('f1',0):.3f}")
            # regression metrics
            elif 'rmse' in info or 'r2' in info:
                perf.append(f"{name}: rmse={info.get('rmse',0):.3f}, r2={info.get('r2',0):.3f}")
    if perf:
        lines.append("Model performance: " + " | ".join(perf))
    lines.append("Recommendations:")
    lines.append("- Add domain-specific feature engineering (e.g., name/title, family size).")
    lines.append("- Evaluate class imbalance; consider class weights/resampling.")
    lines.append("- Persist pipeline and add monitoring for production.")
    return "\n".join(lines)



#  Final Auto-Target Agent 
def infer_target(df, prefer_names=None):
    if prefer_names is None:
        prefer_names = ['target','label','y','class','survived','outcome','price','amount','sales','response']
    cols = df.columns.tolist()
    n = len(df)
    nunique = df.nunique(dropna=True)
    # exact name
    for c in cols:
        if c.lower() in [p.lower() for p in prefer_names]:
            return c, f"Matched common name '{c}'"
    # avoid ID-like
    non_id = [c for c in cols if nunique.get(c,0) < 0.95 * n]
    if not non_id:
        non_id = cols.copy()
    # small cardinality
    small_card = [c for c in non_id if 1 < nunique.get(c,0) <= 20]
    if small_card:
        chosen = sorted(small_card, key=lambda x: (nunique[x], x))[0]
        return chosen, f"Low-cardinality column '{chosen}'"
    # numeric regression
    numeric_non_id = [c for c in non_id if pd.api.types.is_numeric_dtype(df[c]) and nunique.get(c,0) > 1]
    if numeric_non_id:
        chosen = sorted(numeric_non_id, key=lambda x: abs(nunique[x] - (n/10)))[0]
        return chosen, f"Numeric column '{chosen}'"
    # fallback
    candidates = [c for c in non_id if nunique.get(c,0) > 1]
    if candidates:
        chosen = sorted(candidates, key=lambda x: (nunique[x], x))[0]
        return chosen, f"Fallback column '{chosen}'"
    return cols[0], "Fallback to first column"

class SimpleAgentAutoTarget:
    def __init__(self):
        self.memory = {"last_dataset": None, "last_df_shape": None, "last_models": None}

    def call_tool(self, name, *args, **kwargs):
        tools = {
            "load": load_data,
            "eda": eda_summary,
            "missing": missing_value_tool,
            "outliers": outlier_tool,
            "fe": feature_engineering_tool,
            "task": task_detection,
            "train": train_and_evaluate,
            "insight": insight_tool
        }
        if name not in tools:
            raise ValueError(f"Unknown tool: {name}")
        return tools[name](*args, **kwargs)

    def run_pipeline(self, path=None, target_col=None, prefer_target_names=None):
        df0 = self.call_tool("load", path)
        self.memory["last_dataset"] = path if path else "titanic"
        self.memory["last_df_shape"] = df0.shape
        eda_res = self.call_tool("eda", df0)
        mv_res = self.call_tool("missing", df0)
        df_clean = mv_res["imputed_df"]
        out_res = self.call_tool("outliers", df_clean)
        if target_col:
            chosen, reason = target_col, "User-specified target"
        else:
            chosen, reason = infer_target(df_clean, prefer_names=prefer_target_names)
        log(f"Target selected: '{chosen}'  â€” Reason: {reason}")
        fe_res = self.call_tool("fe", df_clean, target_col=chosen)
        task_type = self.call_tool("task", fe_res["y"])
        log(f"Detected task: {task_type}")
        train_res = self.call_tool("train", fe_res["X"], fe_res["y"], task=task_type)
        insight_text = self.call_tool("insight", eda_res, mv_res, out_res, fe_res, train_res)
        print("[LOG] INSIGHTS:\n" + insight_text)
        model_scores = {}
        for k,v in train_res.items():
            if isinstance(v, dict) and 'accuracy' in v:
                model_scores[k] = v['accuracy']
        self.memory["last_models"] = model_scores
        return {"target": chosen, "reason": reason, "eda": eda_res, "missing": mv_res, "outliers": out_res, "fe": fe_res, "task": task_type, "train": train_res, "insight": insight_text, "memory": self.memory}

log("Final Auto-Target Agent defined.")




# â€” SAMPLE PIPELINE RUN (COMMENTED FOR SUBMISSION)
# HOW TO USE:
# -----------
# 1. Upload your dataset in the left sidebar.
# 2. Uncomment the lines below.
# 3. Replace 'your_dataset.csv' with your file name.
#
# Example:
# agent = SimpleAgentAutoTarget()
# result = agent.run_pipeline(path='your_dataset.csv')
#
# 'result' will include:
#   - target
#   - reason
#   - eda
#   - missing
#   - outliers
#   - fe
#   - train
#   - insight
#   - memory
# ============================================================

# agent = SimpleAgentAutoTarget()
# result = agent.run_pipeline(path='sales.csv')   # Replace with your dataset



# disabled for Kaggle submission
print("Artifact saving disabled for Kaggle submission (artifacts already in GitHub).")



# write README.md and print checklist
readme_lines = [
"Agents Intensive Capstone â€” Enterprise Agent for Automated Data Analysis",
"",
"Track: Enterprise Agents",
"Author: Shriharsh Salokhe",
"",
"Summary:",
"- ADK-style Agent (Tools, Agent, Workflow, Memory)",
"- Auto target detection, EDA, imputation, outliers, FE, model training, insights",
"",
"How to run:",
"1) Upload CSV to Colab or mount Drive",
"2) Run cells top-to-bottom",
"3) Call: result = agent.run_pipeline(path='yourfile.csv')",
"",
"Artifacts saved in /artifacts: best model, feature_columns.json, preprocessing_meta.json",
]
with open("README.md","w") as f:
    f.write("\n".join(readme_lines))
log("README.md created.")
print("\n[SUBMISSION CHECKLIST]")
for item in [
    "Notebook runs top-to-bottom without errors",
    "Track: Enterprise Agents",
    "Demos: Tools, Agent, Workflow, Memory",
    "Artifacts saved",
    "README.md exists",
    "requirements.txt present",
    "Submit before deadline",
]:
    print(" - " + item)





