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


# ==============================================================================
# 1. SETUP, INSTALLATION & VERIFICATION (v2)
# ==============================================================================
# This cell sets up the environment, forcefully installs all required offline 
# packages, verifies their versions, and then runs the final import check.

import os
import sys
import time
import subprocess
from pathlib import Path

print("--- Step 1: Setting up environment ---")
start_time = time.time()

# --- 1. Define Core Paths ---
PROJECT_CODE_DATASET = 'jarc-reactor-sp-baseline-code-jul18'
WHEELS_DATASET = 'jarc-minimal-packages-submission-train-only'

PROJECT_INPUT_ROOT = Path(f"/kaggle/input/{PROJECT_CODE_DATASET}")
WHEEL_ROOT = Path(f'/kaggle/input/{WHEELS_DATASET}/offline_wheels_kaggle')
PROJECT_CODE_ROOT = PROJECT_INPUT_ROOT / 'arc_reactor' / 'arc_reactor'
PYTHON_PATH = str(PROJECT_CODE_ROOT)

# --- 2. Set PYTHONPATH ---
if PROJECT_CODE_ROOT.exists():
    print(f"Adding {PYTHON_PATH} to sys.path and PYTHONPATH.")
    if PYTHON_PATH not in sys.path:
        sys.path.insert(0, PYTHON_PATH)
    os.environ['PYTHONPATH'] = (os.environ.get('PYTHONPATH', '') + f':{PYTHON_PATH}')
else:
    raise FileNotFoundError(f"Source code root not found at {PROJECT_CODE_ROOT}.")

# --- 3. Force Install Offline Wheels ---
print("Starting offline package installation...")
if not WHEEL_ROOT.exists():
    raise FileNotFoundError(f"Wheel directory not found at {WHEEL_ROOT}.")

packages_to_install = [
    "hydra-core", "omegaconf", "orjson==3.11.3", "optuna", "tbparse",
    "pytorch-lightning==2.5.4", "torchmetrics==1.8.1", "lightning-utilities==0.15.2",
    "antlr4-python3-runtime==4.9.3", "pygame==2.6.1", "propcache==0.3.2",
    "numpy==1.26.4", "scipy==1.15.3", "sympy==1.13.1", "networkx==3.4.2", "joblib==1.5.2"
]

pip_executable = [sys.executable, '-m', 'pip']
install_command = pip_executable + [
    'install', '--no-cache-dir', '--no-index', f'--find-links={WHEEL_ROOT}',
    '--no-deps', '--ignore-installed' 
] + packages_to_install

result = subprocess.run(install_command, capture_output=True, text=True)
if result.returncode != 0:
    print("❌ ERROR: pip installation failed.")
    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)
    raise RuntimeError("Failed to install required packages.")
else:
    print("Pip install command executed. See output for details:")
    print(result.stdout)

# --- 4. Post-Install Verification ---
print("\n--- Verifying installed package versions ---")
try:
    import omegaconf
    import antlr4
    print(f"✅ omegaconf version: {omegaconf.__version__} | Path: {omegaconf.__file__}")
    # MODIFIED: The antlr4 package does not have a __version__ attribute.
    print(f"✅ antlr4 module imported successfully | Path: {antlr4.__file__}")
except ImportError as e:
    print(f"❌ Failed to import a critical package after installation: {e}")

# --- 5. Create Lightning Shim ---
try:
    import pytorch_lightning as pl
    if "lightning" not in sys.modules:
        sys.modules["lightning"] = pl
    print("Shim created: 'import lightning' now works.")
except ImportError:
    print("[WARN] pytorch_lightning not found, shim not created.")

# --- 6. Final Verification ---
print("\n--- Final import verification ---")
try:
    from jarc_reactor.evaluate import EvaluationManager
    print("✅ Source code and dependencies imported successfully.")
except ImportError as e:
    print(f"❌ ERROR: Failed to import source code. Check installation. Details: {e}")
    raise e

print(f"\nEnvironment setup complete in {time.time() - start_time:.2f} seconds.")


# ==============================================================================
# 3. RUN THE EVALUATION & SUBMISSION SCRIPT (CELL 3) - CORRECTED
# ==============================================================================
# This single command executes the entire, battle-tested evaluation pipeline.
# All logic is contained within your source code, making this notebook robust.

print("--- Step 3: Running the evaluation and submission script ---")
inference_start_time = time.time()

# --- DEFINE KEY PATHS ---
CHECKPOINT_PATH = "/kaggle/input/jarc-reactor-sp-baseline-code-jul18/arc_reactor/arc_reactor/model-epoch=01-step=1078-train_loss=train_loss=0.740191996.ckpt"
SUBMISSION_LOG_PATH = "/kaggle/working/evaluation_and_submission.log"
OUTPUT_DIR = "/kaggle/working/evaluation_results"

# **FIX**: Point to the correct evaluation data directory included with your project code.
# This directory contains JSON files with the required 'train' sections for context.
EVAL_DATA_DIR = "/kaggle/input/jarc-reactor-sp-baseline-code-jul18/arc_reactor/arc_reactor/jarc_reactor/data/evaluation_data"
CONFIG_DIR_PATH = "/kaggle/input/jarc-reactor-sp-baseline-code-jul18/arc_reactor/arc_reactor/jarc_reactor/conf"
ARC_TEST_DATA = ''
# --- CONSTRUCT AND RUN THE COMMAND ---
# We now point both evaluation.data_dir and data.testing_data_dir to the correct location.
#!python -m jarc_reactor.evaluate \
#    --config-dir {CONFIG_DIR_PATH} \
#    --config-name config \
#    model.checkpoint_path={CHECKPOINT_PATH} \
#    evaluation.output_dir={OUTPUT_DIR} \
#    evaluation.data_dir={EVAL_DATA_DIR} \
#    data.testing_data_dir={EVAL_DATA_DIR} \
#    model=tiny_test \
#    | tee {SUBMISSION_LOG_PATH}

print(f"\nInference and submission generation complete in {time.time() - inference_start_time:.2f} seconds.")
print(f"Check the output files in the '{OUTPUT_DIR}' directory.")


# ==============================================================================
# 4. RUN THE ADVANCED SUBMISSION SCRIPT (CELL 4) - CORRECTED
# ==============================================================================
# This cell uses `run_submission.py` to test the high-quality inference pipeline.
# This logic is SEPARATE from `evaluate.py` and is what you would use to
# generate a final, competitive submission.

print("--- Step 4: Running the advanced submission script ---")
submission_start_time = time.time()

# --- DEFINE KEY PATHS ---
# **FIX**: Define the absolute path to the script itself.
SCRIPT_PATH = "/kaggle/input/jarc-reactor-sp-baseline-code-jul18/arc_reactor/arc_reactor/run_submission.py"
CHECKPOINT_PATH = "/kaggle/input/jarc-reactor-sp-baseline-code-jul18/arc_reactor/arc_reactor/model-epoch=01-step=1078-train_loss=train_loss=0.740191996.ckpt"
CONFIG_PATH = "/kaggle/input/jarc-reactor-sp-baseline-code-jul18/arc_reactor/arc_reactor/checkpoint_config.yaml"
TEST_JSON_PATH = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
OUTPUT_PATH = "/kaggle/working/submission.json"
LOG_PATH = "/kaggle/working/run_submission.log"

# --- CONSTRUCT AND RUN THE COMMAND ---
# **FIX**: We now call the script directly via its path, instead of using `python -m`.
# This automatically handles the necessary Python path for the script's internal imports.
!python {SCRIPT_PATH} \
    --config {CONFIG_PATH} \
    --checkpoint {CHECKPOINT_PATH} \
    --test-json {TEST_JSON_PATH} \
    --output {OUTPUT_PATH} \
    | tee {LOG_PATH}

# To enable Product-of-Experts instead (very slow!), replace the override above with:
# poe.enabled=true \
#     speculative_decoding.enabled=true \
print(f"\nAdvanced submission generation complete in {time.time() - submission_start_time:.2f} seconds.")
print(f"Final submission file is located at: {OUTPUT_PATH}")
print(f"A detailed log was saved to: {LOG_PATH}")

# --- Verify the output ---
print("\n--- Verifying submission file ---")
if Path(OUTPUT_PATH).exists():
    print(f"✅ Success! Submission file created at {OUTPUT_PATH}")
    # Optional: Print the first 200 characters of the submission file for a quick check.
    with open(OUTPUT_PATH, 'r') as f:
        print("\n--- Submission file preview ---")
        print(f.read(200) + "...")
else:
    print(f"❌ ERROR: Submission file was not created. Check the log at {LOG_PATH} for details.")


# ==============================================================================
# 4. FINAL VERIFICATION (CELL 4)
# ==============================================================================
import json
print("--- Step 4: Final Verification ---")
OUTPUT_PATH = Path('/kaggle/working/submission.json') # Path is defined inside SubmissionManager

if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH, 'r') as f:
        data = json.load(f)
        num_predictions = len(data.keys())
        print(f"✅ File '{OUTPUT_PATH.name}' is present and contains {num_predictions} task predictions.")
        print(f"   Example task IDs: {list(data.keys())[:3]}")
else:
    print(f"❌ ERROR: {OUTPUT_PATH.name} was not created. Check the log file for errors: /kaggle/working/submission.log")

