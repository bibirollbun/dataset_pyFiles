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


# ============================================================
# WMS AutoClean AI - Multi-Agent SKU/MSKU Cleaning & Mapping
# Kaggle Notebook (with Gemini Integration)
# ============================================================

# 1. INSTALL & IMPORTS
# ------------------------------------------------------------
!pip install -q google-generativeai

import os
import json
from typing import Dict, Any, List

import pandas as pd
import google.generativeai as genai


# 2. CONFIG & FLAGS
# ------------------------------------------------------------

# If True -> create & use sample data.
# If False -> use your real dataset at INPUT_FILE_PATH.
USE_SAMPLE_DATA = True

class Config:
    # Change this if using real dataset
    INPUT_FILE_PATH = "/kaggle/working/raw_skus.csv"
    CLEANED_OUTPUT_PATH = "/kaggle/working/cleaned_skus.csv"
    ISSUES_OUTPUT_PATH = "/kaggle/working/issues_report.csv"
    AUDIT_LOG_PATH = "/kaggle/working/audit_log.json"

    GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"  # or any available model name


def log(message: str):
    print(f"[LOG] {message}")


def load_audit_log(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_audit_log(path: str, log_data: List[Dict[str, Any]]):
    with open(path, "w") as f:
        json.dump(log_data, f, indent=2)


# 3. SAMPLE DATA (only for demo)
# ------------------------------------------------------------
if USE_SAMPLE_DATA:
    sample_data = [
        {"sku": "SKU001", "product_name": "Tata Salt 1kg",        "brand": "Tata",    "unit": "1 KG",     "quantity": 100},
        {"sku": "SKU002", "product_name": "Tata salt 1Kg ",       "brand": "Tata",    "unit": "1kg",      "quantity": 50},
        {"sku": "SKU003", "product_name": "Fortune Oil 1L",       "brand": "Fortune", "unit": "1 L",      "quantity": 80},
        {"sku": "SKU004", "product_name": "fortune oil 1 litre",  "brand": "Fortune", "unit": "1000 ml",  "quantity": 80},
        {"sku": "SKU005", "product_name": "No Name",              "brand": None,      "unit": "NA",       "quantity": -5},
    ]
    raw_df = pd.DataFrame(sample_data)
    raw_df.to_csv(Config.INPUT_FILE_PATH, index=False)
    print("Sample raw_skus.csv created at:", Config.INPUT_FILE_PATH)
    display(raw_df)


# 4. GEMINI SETUP
# ------------------------------------------------------------

def setup_gemini():
    """
    Configure Gemini using Kaggle secret GOOGLE_API_KEY.
    If key is missing, LLM-based mapping will be skipped.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log("WARNING: GOOGLE_API_KEY not found in environment. MappingAgent will use fallback logic.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=Config.GEMINI_MODEL_NAME,
            generation_config={"response_mime_type": "application/json"}  # ğŸ‘ˆ force JSON output
        )
        log(f"Gemini model '{Config.GEMINI_MODEL_NAME}' configured.")
        return model
    except Exception as e:
        log(f"ERROR: Failed to configure Gemini model: {e}")
        return None

# 5. BASE AGENT CLASS
# ------------------------------------------------------------

class Agent:
    def __init__(self, name: str):
        self.name = name

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


# 6. INGESTION AGENT
# ------------------------------------------------------------

class IngestionAgent(Agent):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log(f"{self.name}: Loading input file from {Config.INPUT_FILE_PATH} ...")
        df = pd.read_csv(Config.INPUT_FILE_PATH)
        log(f"{self.name}: Loaded {len(df)} rows.")
        context["raw_df"] = df
        return context


# 7. CLEANING AGENT
# ------------------------------------------------------------

class CleaningAgent(Agent):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log(f"{self.name}: Cleaning data...")
        df = context["raw_df"].copy()

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Strip & standardize common text fields
        for col in ["product_name", "brand", "unit"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Simple unit normalization example
        if "unit" in df.columns:
            df["unit_norm"] = (
                df["unit"]
                .str.lower()
                .replace({
                    "1 kg": "1kg",
                    "1 kg ": "1kg",
                    "1kg ": "1kg",
                    "1 l": "1l",
                    "1 litre": "1l",
                    "1000 ml": "1l",
                })
            )

        # Remove duplicates based on sku + product_name
        before = len(df)
        df = df.drop_duplicates(subset=["sku", "product_name"], keep="first")
        after = len(df)
        log(f"{self.name}: Removed {before - after} duplicate rows.")

        context["clean_df"] = df
        return context


# 8. MAPPING AGENT (Gemini + fallback)
# ------------------------------------------------------------

class MappingAgent(Agent):
    def __init__(self, name: str, use_llm: bool = True):
        super().__init__(name)
        self.use_llm = use_llm
        self.model = None

    def _init_model_if_needed(self):
        if self.use_llm and self.model is None:
            self.model = setup_gemini()

    def _safe_parse_json(self, text: str):
        """
        Try to parse JSON from Gemini response text.
        Handles cases with markdown code fences or extra text.
        """
        text = (text or "").strip()

        # Remove markdown ```json ... ``` if present
        if text.startswith("```"):
            lines = text.splitlines()
            # drop lines that are just ``` or ```json
            lines = [
                line for line in lines
                if not line.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()

        return json.loads(text)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log(f"{self.name}: Inferring MSKU relationships...")
        df = context["clean_df"].copy()
        self._init_model_if_needed()

        msku_list = []
        explanations = []

        # If no model (missing key or config), fallback to rule-based grouping
        if not self.model:
            log(f"{self.name}: No Gemini model available. Using fallback rule-based MSKU grouping.")
            if "brand" in df.columns and "unit_norm" in df.columns:
                df["msku"] = df["brand"].fillna("UNKNOWN") + "_" + df["unit_norm"].fillna("NA")
            else:
                if "sku" in df.columns:
                    df["msku"] = df["sku"].astype(str).str[:5]
                else:
                    df["msku"] = "MSKU_UNKNOWN"

            context["mapped_df"] = df
            context["mapping_explanations"] = []
            return context

        # LLM-based mapping: row-by-row (for demo; in real world, youâ€™d batch)
        for idx, row in df.iterrows():
            sku = row.get("sku", "")
            pname = row.get("product_name", "")
            brand = row.get("brand", "")
            unit = row.get("unit", "")

            prompt = f"""
You are a product data normalization assistant for a warehouse.

Given this product row:

- SKU: {sku}
- Product Name: {pname}
- Brand: {brand}
- Unit: {unit}

1. Suggest a standardized (canonical) product name.
2. Suggest an MSKU key such that all identical products share the same MSKU.
3. Explain your reasoning briefly.

Return ONLY valid JSON in this format (no extra text):

{{
  "msku": "<grouping_key>",
  "canonical_name": "<standard_product_name>",
  "reason": "<short_reason>"
}}
"""

            try:
                resp = self.model.generate_content(prompt)
                raw_text = (resp.text or "").strip()

                data = self._safe_parse_json(raw_text)
                msku = data.get("msku", "MSKU_UNKNOWN")
                explanation = data

            except Exception as e:
                log(f"{self.name}: Error using Gemini for row {idx}: {e}")
                msku = "MSKU_FALLBACK"
                explanation = {
                    "error": str(e),
                    "raw_response": raw_text if 'raw_text' in locals() else ""
                }

            msku_list.append(msku)
            explanations.append(explanation)

        # Assign model-generated MSKU list
        df["msku"] = msku_list

        # Normalize MSKU format for consistency
        df["msku"] = (
            df["msku"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "-", regex=False)
            .str.replace("_", "-", regex=False)
        )

        # Update context
        context["mapped_df"] = df
        context["mapping_explanations"] = explanations

        return context


# 9. VALIDATION AGENT
# ------------------------------------------------------------

class ValidationAgent(Agent):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log(f"{self.name}: Validating data against business rules...")
        df = context["mapped_df"].copy()
        issues = []

        # Rule 1: SKU should not be empty
        if "sku" in df.columns:
            empty_sku_rows = df[df["sku"].isna()]
            for idx, row in empty_sku_rows.iterrows():
                issues.append({
                    "row_index": int(idx),
                    "issue": "Missing SKU",
                    "row_data": row.to_dict(),
                })

        # Rule 2: Quantity should not be negative
        if "quantity" in df.columns:
            bad_qty_rows = df[df["quantity"] < 0]
            for idx, row in bad_qty_rows.iterrows():
                issues.append({
                    "row_index": int(idx),
                    "issue": "Negative quantity",
                    "row_data": row.to_dict(),
                })

        log(f"{self.name}: Found {len(issues)} issues.")
        context["validated_df"] = df
        context["issues"] = issues
        return context


# 10. REPORTING AGENT
# ------------------------------------------------------------

class ReportingAgent(Agent):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log(f"{self.name}: Saving cleaned data and issues report...")

        df_cleaned = context["validated_df"]
        issues = context.get("issues", [])

        # Save cleaned data
        df_cleaned.to_csv(Config.CLEANED_OUTPUT_PATH, index=False)
        log(f"{self.name}: Saved cleaned data to {Config.CLEANED_OUTPUT_PATH}")

        # Save issues as CSV (if any)
        if issues:
            issues_df = pd.DataFrame(issues)
            issues_df.to_csv(Config.ISSUES_OUTPUT_PATH, index=False)
            log(f"{self.name}: Saved issues report to {Config.ISSUES_OUTPUT_PATH}")
        else:
            log(f"{self.name}: No issues to save.")

        summary = {
            "input_rows": len(context["raw_df"]),
            "cleaned_rows": len(df_cleaned),
            "issues_count": len(issues),
        }
        context["summary"] = summary
        log(f"{self.name}: Summary -> {summary}")

        return context


# 11. AUDIT AGENT
# ------------------------------------------------------------

class AuditAgent(Agent):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log(f"{self.name}: Updating audit log...")
        audit_log = load_audit_log(Config.AUDIT_LOG_PATH)

        run_summary = {
            "input_rows": len(context["raw_df"]),
            "cleaned_rows": len(context["validated_df"]),
            "issues_count": len(context["issues"]),
        }
        audit_log.append(run_summary)
        save_audit_log(Config.AUDIT_LOG_PATH, audit_log)

        context["audit_log"] = audit_log
        log(f"{self.name}: Audit log now has {len(audit_log)} entries.")
        return context


# 12. ORCHESTRATOR + PIPELINE RUN
# ------------------------------------------------------------

def run_pipeline():
    context: Dict[str, Any] = {}

    agents = [
        IngestionAgent("IngestionAgent"),
        CleaningAgent("CleaningAgent"),
        MappingAgent("MappingAgent", use_llm=True),
        ValidationAgent("ValidationAgent"),
        ReportingAgent("ReportingAgent"),
        AuditAgent("AuditAgent"),
    ]

    for agent in agents:
        context = agent.run(context)

    return context


# Run pipeline once (for demo)
final_context = run_pipeline()

print("\n=== FINAL SUMMARY ===")
print(final_context.get("summary", {}))

print("\n=== CLEANED DATA (HEAD) ===")
display(final_context["validated_df"].head())

print("\n=== ISSUES (if any) ===")
if final_context["issues"]:
    display(pd.DataFrame(final_context["issues"]).head())
else:
    print("No issues found.")

print("\n=== AUDIT LOG ===")
print(final_context["audit_log"])











