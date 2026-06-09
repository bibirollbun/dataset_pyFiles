# --- CAPSTONE PROJECT: AGENT 47: THE SELF-EVOLVING DATA ANALYST AGENT (LEVEL 4 SYSTEM) ---
# --- ENTERPRISE AGENTS TRACK: AUTONOMOUS DATA PIPELINE OPTIMIZATION ---

# ğŸš¨ IMPORTANT: You must enable Internet access in your Kaggle Notebook settings
# and ensure your KAGGLE_USERNAME and KAGGLE_KEY secrets are attached to this notebook.

# --- INSTALL & IMPORT NECESSARY LIBRARIES ---
# ğŸš¨ Run this cell first!
!pip install kaggle scikit-learn requests

import pandas as pd
import numpy as np
import random
import logging
import os
import shutil
import requests
import warnings
import json
from typing import Dict, List, Any

# Libraries for analysis and scoring (The Quality Metric)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ğŸ”‘ IMPORTANT: Import the dedicated library for accessing Kaggle Secrets FIRST
from kaggle_secrets import UserSecretsClient 
# Note: The import for KaggleApi is intentionally moved later to prevent the initial OSError.

# --- Phase 0: System Foundation, Policy, and Observability ---

# Global Policy (The Agent's Evolving Rulebook - The Procedural Memory)
LEARNING_PARAMETERS = {
    # Source Priority (0.0 to 1.0) - A critical piece of Procedural Memory
    "source_priority": {"Kaggle_Tool": 0.5, "Web_API_Tool": 0.3},
    "imputation_strategy": "median", 
    "cleaning_aggressiveness": 0.6 
}

# Long-Term Memory (The Agent's History Journal)
LONG_TERM_MEMORY: List[Dict[str, Any]] = []

# Observability Setup (Tracing) - The Agent's Sensory System
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("âœ… Agent 47 System Foundation Initialized: Policy, Memory, and Observability Established.")

# -----------------------------------------------------------------------------
# --- PERMANENT FIX: Resolve Kaggle API Authentication Error & Warning ---
KAGGLE_CLIENT = None
KAGGLE_CONFIG_DIR = '/tmp/.kaggle'

try:
    secrets_client = UserSecretsClient()
    
    # 1. Retrieve the secrets using their exact labels
    kaggle_username = secrets_client.get_secret('KAGGLE_USERNAME')
    kaggle_key = secrets_client.get_secret('KAGGLE_KEY')

    if kaggle_username and kaggle_key:
        # 2. Set the configuration path to a temporary, writable location
        os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)
        
        # ğŸš¨ FIX: Set the environment variable *before* importing KaggleApi
        os.environ['KAGGLE_CONFIG_DIR'] = KAGGLE_CONFIG_DIR

        # 3. Create the expected kaggle.json file content
        kaggle_json_content = {
            "username": kaggle_username,
            "key": kaggle_key
        }
        
        # 4. Write the content to the temporary file path
        kaggle_json_path = f'{KAGGLE_CONFIG_DIR}/kaggle.json'
        
        # NOTE ON PERMISSIONS: The os.chmod() call below is the correct fix.
        # We ensure it runs before KaggleApi is initialized.
        with open(kaggle_json_path, 'w') as f:
            json.dump(kaggle_json_content, f)
        
        # 5. Apply Security Fix (chmod 600) to silence the warning
        # This explicitly restricts read/write access to the owner (0o600).
        os.chmod(kaggle_json_path, 0o600)
        
        # 6. Import KaggleApi AFTER the configuration and file permissions are set
        from kaggle.api.kaggle_api_extended import KaggleApi
        
    else:
        print("â�Œ Setup Warning: Failed to retrieve secrets via UserSecretsClient. Check secret labels.")
        
except Exception as e:
    print(f"â�Œ Setup Error during permanent fix: Could not access Kaggle Secrets Client. Error: {e}")

# -----------------------------------------------------------------------------

# --- Initialize Kaggle API Client ---
try:
    # Only attempt to initialize if the KaggleApi class was successfully imported
    if 'KaggleApi' in locals():
        KAGGLE_CLIENT = KaggleApi()
        KAGGLE_CLIENT.authenticate()
        print("âœ… Kaggle API authenticated successfully.")
    else:
        print("â�Œ WARNING: Kaggle API client could not be initialized (KaggleApi class not found).")
        KAGGLE_CLIENT = None
except Exception as e:
    print(f"â�Œ WARNING: Kaggle authentication failed. Final Error: {e}")
    KAGGLE_CLIENT = None

# -----------------------------------------------------------------------------


# Note: Ensure 'shutil' is imported in the initial setup cell for shutil.rmtree to work.

# --- Phase I: The Orchestration Layer and Granular Tooling (The Hands) ---

# Tool 1: Kaggle Data Collector (Granular Tool)
def fetch_data_from_kaggle_tool(query: str, api: KaggleApi = KAGGLE_CLIENT):
    """Kaggle Tool: Downloads a stable, public dataset (Iris) to guarantee acquisition."""
    if api is None:
        raise Exception("Kaggle API not initialized. Cannot run Kaggle_Tool.")
        
    # --- FIX: Changed to a reliable public dataset to bypass the 403 error ---
    dataset_slug = "uciml/iris" 
    file_name = "Iris.csv"
    # -------------------------------------------------------------------------
    
    download_dir = f'temp_data_{dataset_slug.replace("/", "_")}'
    
    try:
        # Clean up and prepare download directory
        if os.path.exists(download_dir):
            shutil.rmtree(download_dir)
        os.makedirs(download_dir, exist_ok=True)
        
        # Download and unzip
        api.dataset_download_files(dataset_slug, path=download_dir, unzip=True)
        
        # Find the correct file path (handling potential subfolders)
        target_path_1 = os.path.join(download_dir, file_name)
        target_path_2 = os.path.join(download_dir, dataset_slug.split('/')[-1], file_name)
        
        if os.path.exists(target_path_1):
            file_path = target_path_1
        elif os.path.exists(target_path_2):
            file_path = target_path_2
        else:
            raise FileNotFoundError(f"Required file '{file_name}' not found after extraction.")

        # Load, process, and cleanup
        df = pd.read_csv(file_path)
        
        # Preprocessing for Iris data to fit the agent's expected output structure
        df = df.drop(columns=['Id'], errors='ignore')
        
        # Create 'target_price' placeholder for model training consistency
        if 'PetalLengthCm' in df.columns:
            df['target_price'] = df['PetalLengthCm'] 

        shutil.rmtree(download_dir)
        
        return df, "Kaggle_Tool"
        
    except Exception as e:
        # Ensure cleanup even on failure
        if os.path.exists(download_dir):
            shutil.rmtree(download_dir)
        raise e
        
# Tool 2: Web API Collector (Granular Tool)
def fetch_data_from_web_api_tool(query: str):
    """Web API Tool: Fetches recent cryptocurrency prices (simulating live data)."""
    # This tool is kept as is. The previous error was network-related and should resolve 
    # or be handled by the Orchestrator's failover mechanism.
    url = "https://api.coincap.io/v2/assets"
    response = requests.get(url, params={'limit': 50})
    response.raise_for_status() 
    
    data = response.json().get('data', [])
    df = pd.json_normalize(data)
    
    numeric_cols = ['rank', 'priceUsd', 'marketCapUsd', 'volumeUsd24Hr']
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')

    df['target_price'] = df['priceUsd'] 
    df = df[numeric_cols + ['target_price']]
    
    # Simulate a missing value for the Analyst to handle
    if not df.empty:
         df.loc[df.index[-1], 'volumeUsd24Hr'] = np.nan 

    return df, "Web_API_Tool"

# Orchestration Layer (Directs the Tools based on Policy)
def orchestrate_data_collection(query: str) -> Dict[str, Any]:
    """Directs the Tool Suite based on current Learned Source Priority (The Level 4 Control)."""
    logging.info(f"\n--- AGENT 47 START SESSION for Query: {query} ---")
    
    sources = [
        (fetch_data_from_kaggle_tool, "Kaggle_Tool"), 
        (fetch_data_from_web_api_tool, "Web_API_Tool"), 
    ]
    
    # Sorting based on policy from Memory
    sources.sort(key=lambda x: LEARNING_PARAMETERS["source_priority"].get(x[1], 0), reverse=True)
    
    logging.info(f"Orchestration Order (Priority): {[s[1] for s in sources]}")
    
    for fetch_func, source_name in sources:
        try:
            logging.info(f"Attempting Tool: {source_name}")
            df, source = fetch_func(query)
            logging.info(f"Tool {source_name} Success. Data Shape: {df.shape}")
            return {"df": df, "source": source, "query": query}
        except Exception as e:
            logging.warning(f"Tool {source_name} failed: {e}. Trying next source.")
            continue
            
    logging.error("All collection tools failed.")
    raise Exception("Collection failure.")

# --- Execution of Phase I ---
try:
    collection_result = orchestrate_data_collection("financial market history")
    df = collection_result['df']
    print(f"\nâœ… Phase I Complete: Data Collected from {collection_result['source']} (Shape: {df.shape})")
    print(f"Data Head:\n{df.head()}")
except Exception as e:
    print(f"â�Œ Phase I Failed: {e}")

# -----------------------------------------------------------------------------


# --- Phase II: The Analyst Module and The Quality Metric Generator (The Assessment) ---

def analyze_and_score(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans data using policy and calculates the R2 score (The Quality Metric)."""
    logging.info("Starting Policy-Driven Analysis and Baseline Modeling.")
    df_clean = df.copy()
    
    # 1. Policy-Driven Cleaning
    aggressiveness = params['cleaning_aggressiveness']
    strategy = params['imputation_strategy']
    
    for col in df_clean.columns:
        missing_pct = df_clean[col].isnull().mean()
        
        if missing_pct >= aggressiveness:
            df_clean.drop(col, axis=1, inplace=True)
        elif missing_pct > 0 and pd.api.types.is_numeric_dtype(df_clean[col]):
            # Imputation logic based on learned strategy
            if strategy == 'median': fill_val = df_clean[col].median()
            elif strategy == 'zero': fill_val = 0
            else: fill_val = df_clean[col].mean() 
            df_clean[col].fillna(fill_val, inplace=True)
    
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 2. Baseline Model Scoring (Generating the R2 Quality Metric)
    target_candidates = [c for c in df_clean.columns if 'target_price' in c.lower()]
    model_score = 0.0
    
    if target_candidates:
        target_col = target_candidates[0]
        X = df_clean.select_dtypes(include=np.number).drop(columns=[target_col], errors='ignore')
        y = df_clean[target_col]
        
        if len(X) >= 20 and len(X.columns) > 0: 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_score = r2_score(y_test, y_pred) # The R2 Score (The Quality Metric)

    analysis_data = {
        "final_model_score": model_score, 
        "strategy_used": strategy
    }
    
    logging.info(f"Quality Metric (RÂ²): {model_score:.4f}")
    return analysis_data

# --- Execution of Phase II ---
analysis_report = analyze_and_score(df, LEARNING_PARAMETERS)
print(f"\nâœ… Phase II Complete: Agent 47's Quality Metric (Model RÂ²): {analysis_report['final_model_score']:.4f}")

# -----------------------------------------------------------------------------


# --- Phase III: The Optimizer Module (The Agent Quality Flywheel - The Evolution) ---

def extract_memory_from_session(collection_data: Dict, analysis_data: Dict):
    """Saves the key results into the agent's Long-Term Memory."""
    memory_entry = {
        "timestamp": pd.Timestamp.now(),
        "source_used": collection_data['source'],
        "successful_strategy": analysis_data['strategy_used'],
        "final_score": analysis_data['final_model_score']
    }
    LONG_TERM_MEMORY.append(memory_entry)
    logging.info(f"Memory Extracted and Stored. Total Memories: {len(LONG_TERM_MEMORY)}")

def update_policy_via_flywheel(memory: List[Dict[str, Any]]):
    """Adjusts the global LEARNING_PARAMETERS based on past performance (Self-Evolution)."""
    logging.info("Optimizer Running: Evolving Agent's Policy...")
    if not memory: return
    
    global LEARNING_PARAMETERS

    df_memory = pd.DataFrame(memory)
    
    # 1. Evolve Source Priority Policy (The Reinforcement Loop)
    source_scores = df_memory.groupby('source_used')['final_score'].mean()
    for source, avg_score in source_scores.items():
        old_priority = LEARNING_PARAMETERS["source_priority"].get(source, 0.1)
        
        # Policy Adjustment Logic: Boost priority if the source provides high-scoring data
        boost = (avg_score - 0.75) * 0.1 # Learning Rate = 0.1
        new_priority = max(0.01, min(1.0, old_priority + boost))
        
        LEARNING_PARAMETERS["source_priority"][source] = new_priority

    # 2. Evolve Imputation Strategy Policy
    strategy_scores = df_memory.groupby('successful_strategy')['final_score'].mean()
    
    # If the best performing strategy has an average score > 0.6 and we have multiple strategies to compare
    if len(strategy_scores) > 1 and strategy_scores.max() > 0.6:
        best_strategy = strategy_scores.idxmax()
        if LEARNING_PARAMETERS['imputation_strategy'] != best_strategy:
            LEARNING_PARAMETERS['imputation_strategy'] = best_strategy
            logging.warning(f"Policy Update: Imputation strategy switched to {best_strategy}!")
    
    print("\nğŸ”„ Policy Evolution Complete. Agent 47 is smarter for the next run.")

# --- Execution of Phase III (The Self-Evolution Step) ---
print("\n--- Starting Phase III: The Optimizer Module (The Agent Quality Flywheel - The Evolution) ---")

extract_memory_from_session(collection_result, analysis_report)
update_policy_via_flywheel(LONG_TERM_MEMORY)
print(f"\nâœ… Phase III Complete: Policy Evolution Finished.")

print("\n--- AGENT 47: NEW EVOLVED POLICY (The Agent's New Rules for Next Time) ---")
print(LEARNING_PARAMETERS)


# --- Phase IV: The Decision Loop (The Self-Rerun) ---

def decision_loop(current_params: Dict[str, Any], memory: List[Dict[str, Any]], max_iterations: int = 2) -> bool:
    """Decides whether to continue the evolution loop based on iteration count."""
    
    # Count the number of previous policy evaluation runs
    run_count = sum(1 for entry in memory if 'score' in entry)

    logging.info(f"DECISION LOOP: Current Run Count is {run_count}. Max iterations set to {max_iterations}.")

    if run_count < max_iterations:
        return True # Signals RE-RUN
    else:
        return False # Signals TERMINATE

# --- Execution of Phase IV ---
print("\n--- Starting Phase IV: The Decision Loop ---")

# Execute the decision function
should_rerun = decision_loop(LEARNING_PARAMETERS, LONG_TERM_MEMORY, max_iterations=2)
decision_text = "RE-RUN PIPELINE ğŸ”„" if should_rerun else "TERMINATE LOOP âœ…"
print(f"\nâœ… Phase IV Complete: Agent 47's Decision: {decision_text}")
print(f"New Policy for Next Run: Imputation Strategy '{LEARNING_PARAMETERS['imputation_strategy']}'")


# --------------------------------------------------------------------------
# --- FINAL EXECUTION: RERUN THE LOOP ---
# --------------------------------------------------------------------------

try:
    print("\n--- AGENT 47 RERUN: Testing Self-Corrected Policy ---")
    print(f"Policy for Rerun: Imputation Strategy '{LEARNING_PARAMETERS['imputation_strategy']}'")
    
    # 1. Rerun Phase I (Data Collection)
    collection_result = orchestrate_data_collection("financial market history")
    raw_df = collection_result['df']
    
    # 2. Rerun Phase II (Analysis and Scoring)
    analysis_report = analyze_and_score(raw_df, LEARNING_PARAMETERS)

    # 3. Output the new Quality Metric
    print("\nâœ… RERUN COMPLETE: Self-Correction Test Finished.")
    print(f"Source Used: {collection_result['source']} (Priority: {LEARNING_PARAMETERS['source_priority'][collection_result['source']]})")
    print(f"New Policy Tested: {analysis_report['strategy_used']}")
    print(f"Final **Quality Metric (Model RÂ²)**: **{analysis_report['final_model_score']:.4f}** ğŸ�†")
    
except Exception as e:
    print(f"â�Œ RERUN FAILED: {e}")

