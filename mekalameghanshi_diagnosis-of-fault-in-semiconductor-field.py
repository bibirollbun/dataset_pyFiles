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


##This notebook presents a multi-agent system that analyzes SPICE simulation logs to identify faults in semiconductor circuits. It includes:

#  Loop agent for hypothesis testing
#  Custom SPICE parser tool
#  LLM-style fault explanation
#  Memory bank for known fault codes
#  Logging and CLI deployment

#This agent simulates how engineers diagnose layout or simulation issues using intelligent automation


"""
Kaggle-compatible Fault Diagnosis Notebook
Features:
- File uploader widget (works in Kaggle notebooks)
- Safe logging to /kaggle/working/
- Robust SPICE log parser
- Improved hypothesis generation & testing
- Memory bank lookup
- Simple offline LLM-style explainer
- Error-count visualization (matplotlib)
- Exception handling & progress messages

How to use:
1. Run the cells in order.
2. Upload one or more SPICE log files using the uploader widget.
3. Click the run button next to `process_uploaded_files()` cell or call `process_uploaded_files()`.

Notes about Kaggle environment:
- Uploaded files via the widget are available only during the session.
- If you mounted a dataset in /kaggle/input/, use `run_fault_diagnosis('/kaggle/input/yourdataset/yourfile.log')`.

"""

# %%
# Imports & setup
import os
import io
import logging
import traceback
from collections import Counter
from datetime import datetime

# Visualization
import matplotlib.pyplot as plt

# File upload widget (ipywidgets should be available on Kaggle)
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except Exception:
    WIDGETS_AVAILABLE = False

# Configure logging to a writable location in Kaggle
LOG_PATH = '/kaggle/working/fault_agent_log.txt'
logging.basicConfig(filename=LOG_PATH,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_event(msg):
    logging.info(msg)
    # Also print for notebook interactivity
    print(msg)

# %%
# Parser & core agent logic

def parse_spice_log_file_object(fileobj, filename='uploaded.log'):
    """Parse a file-like object for SPICE errors/faults.
    Returns list of stripped lines that look like faults.
    """
    try:
        raw = fileobj.read()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='ignore')
    except Exception:
        fileobj.seek(0)
        raw = fileobj.read().decode('utf-8', errors='ignore')

    lines = raw.splitlines()
    # Heuristic collection: lines containing ERROR|FAULT|WARN|FATAL|short|leakage
    keywords = ["ERROR", "FAULT", "WARN", "FATAL", "short", "leakage", "overlap", "latch"]
    parsed = [line.strip() for line in lines if any(k.lower() in line.lower() for k in keywords)]
    log_event(f"Parsed {len(parsed)} candidate fault lines from {filename}")
    return parsed


def generate_hypotheses(log_lines):
    """Turn raw fault-like lines into normalized hypothesis strings.
    We perform light normalization and deduplication.
    """
    hyps = []
    for line in log_lines:
        normalized = ' '.join(line.split())  # collapse whitespace
        hyps.append(normalized)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for h in hyps:
        key = h.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq


def test_hypothesis(hypothesis):
    """Richer heuristic tests for confirmation.
    Returns: ('confirmed'|'possible'|'unconfirmed', score)
    """
    text = hypothesis.lower()
    score = 0
    reasons = []

    if 'short' in text:
        score += 50
        reasons.append('short detected')
    if 'leak' in text or 'leakage' in text:
        score += 45
        reasons.append('leakage detected')
    if 'latch' in text or 'latch-up' in text:
        score += 40
        reasons.append('latch-up detected')
    if 'error' in text or 'fatal' in text or 'warn' in text:
        score += 10
        reasons.append('error/warning tag')
    # small heuristics
    if 'node' in text and ('0' in text or 'gnd' in text):
        score += 5
        reasons.append('ground node mention')

    # final classification
    if score >= 50:
        return 'confirmed', score, ';'.join(reasons)
    elif score >= 30:
        return 'possible', score, ';'.join(reasons)
    else:
        return 'unconfirmed', score, ';'.join(reasons)


def fault_loop(log_lines, stop_on_first=True):
    """Loop through hypotheses, test, and return results.
    Returns list of dicts with hypothesis, status, score.
    If stop_on_first is True, returns the first confirmed hypothesis (as single-item list).
    """
    results = []
    for hyp in generate_hypotheses(log_lines):
        try:
            status, score, reasons = test_hypothesis(hyp)
            results.append({
                'hypothesis': hyp,
                'status': status,
                'score': score,
                'reasons': reasons
            })
            if stop_on_first and status == 'confirmed':
                log_event(f"Confirmed hypothesis: {hyp} (score={score})")
                return results
        except Exception as e:
            log_event(f"Error testing hypothesis: {e}\n{traceback.format_exc()}")
    return results

# %%
# Explainer + memory bank

memory_bank = {
    "F001": "Gate leakage due to thin oxide",
    "F002": "Drain-source short in NMOS",
    "F003": "Latch-up condition in CMOS",
}


def recall_fault(code):
    return memory_bank.get(code, "Fault code not found in memory bank.")


def explain_fault(hypothesis_text):
    """Provide a simple LLM-style explanation without external APIs.
    If hypothesis_text is a results list/dict, handle gracefully.
    """
    try:
        if isinstance(hypothesis_text, dict):
            text = hypothesis_text.get('hypothesis', '')
        elif isinstance(hypothesis_text, (list, tuple)) and hypothesis_text:
            # take top item
            item = hypothesis_text[0]
            if isinstance(item, dict):
                text = item.get('hypothesis', '')
            else:
                text = str(item)
        else:
            text = str(hypothesis_text)

        text_low = text.lower()
        if 'short' in text_low:
            return "This indicates a short circuit between nodes, possibly due to layout overlap or metal-to-metal contact. Investigate node names and physical proximity."
        if 'leak' in text_low or 'leakage' in text_low:
            return "This indicates leakage current. Check isolation regions, bias conditions, and temperature during simulation."
        if 'latch' in text_low:
            return "This may suggest a latch-up condition; review well ties and guard rings."
        if 'error' in text_low or 'fatal' in text_low:
            return "Generic simulation error â€” examine surrounding log lines for stack traces or prior warnings."
        return "Unable to determine fault type from the given line. Review surrounding log context or run a targeted simulation."
    except Exception:
        return "Error generating explanation."

# %%
# Visualization helpers

def plot_fault_summary(results_list, title='Fault Summary'):
    """Plot counts of confirmed/possible/unconfirmed entries."""
    statuses = [r['status'] for r in results_list]
    counter = Counter(statuses)
    labels = list(counter.keys())
    values = [counter[k] for k in labels]

    plt.figure(figsize=(6,3))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

# %%
# Runner + file uploader

UPLOADED_FILES = {}


def on_upload_change(change):
    # callback for widget
    if not change.get('new'):
        return
    for fname, fobj in uploader.value.items():
        UPLOADED_FILES[fname] = fobj
    log_event(f"{len(UPLOADED_FILES)} file(s) uploaded: {', '.join(UPLOADED_FILES.keys())}")


if WIDGETS_AVAILABLE:
    uploader = widgets.FileUpload(accept='.log,.txt,.spice', multiple=True)
    uploader.observe(on_upload_change, names='value')
    display(widgets.HTML('<b>Upload SPICE log files (multiple allowed):</b>'))
    display(uploader)
else:
    print('ipywidgets not available in this environment; use run_fault_diagnosis with a filepath.')


# Helper: convert uploader fileobj to in-memory file-like
def _get_uploaded_fileobj(fname):
    item = UPLOADED_FILES.get(fname)
    if not item:
        raise FileNotFoundError(f"Uploaded file {fname} not found")
    # item is dict with keys: name, type, size, content (bytes)
    content = item.get('content')
    return io.BytesIO(content)


def run_fault_diagnosis_on_fileobj(fileobj, filename='uploaded.log', stop_on_first=True):
    log_event(f"Starting diagnosis for {filename}")
    try:
        parsed = parse_spice_log_file_object(fileobj, filename=filename)
        if not parsed:
            log_event('No fault-like lines found in the file.')
            return []
        results = fault_loop(parsed, stop_on_first=stop_on_first)
        # If results is empty dict list, still derive explanations
        if not results:
            log_event('No hypotheses produced any results')
            return []
        # Ensure result format: if stop_on_first returned early, it may be single-item list
        plot_fault_summary(results, title=f'Fault Summary: {filename}')
        # Provide explanations for top results
        for r in results:
            print('\n---')
            print('Hypothesis:', r['hypothesis'])
            print('Status:', r['status'])
            print('Score:', r['score'])
            print('Reasons:', r['reasons'])
            print('Explanation:', explain_fault(r))
        return results
    except Exception as e:
        log_event(f"Exception during diagnosis: {e}\n{traceback.format_exc()}")
        return []


def process_uploaded_files(stop_on_first=True):
    """Process all files currently uploaded via the widget."""
    if not UPLOADED_FILES:
        print('No uploaded files found. Use the uploader widget to upload files.')
        return {}
    summary = {}
    for fname in list(UPLOADED_FILES.keys()):
        try:
            fobj = _get_uploaded_fileobj(fname)
            res = run_fault_diagnosis_on_fileobj(fobj, filename=fname, stop_on_first=stop_on_first)
            summary[fname] = res
        except Exception as e:
            log_event(f"Failed processing {fname}: {e}")
    log_event('Processing of uploaded files complete.')
    return summary

# %%
# Utility: direct filepath runner (useful if using /kaggle/input)

def run_fault_diagnosis(filepath, stop_on_first=True):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'rb') as f:
        return run_fault_diagnosis_on_fileobj(f, filename=os.path.basename(filepath), stop_on_first=stop_on_first)

# %%
# Example usage cell (commented but ready)

# To run on uploaded files via the widget, run:
# process_uploaded_files()

# To run on a dataset mounted at /kaggle/input:
# run_fault_diagnosis('/kaggle/input/spice-logs/log1.txt')

# You can also list uploaded filenames:
# list(UPLOADED_FILES.keys())

# %%
# End of notebook
print('\nNotebook setup complete.\n- Upload files using the widget above and then call process_uploaded_files().\n- Logs are written to: ' + LOG_PATH)
""



# Create a dummy SPICE log file
dummy_log = """
INFO: Simulation started
ERROR: Drain-Source short detected in NMOS M1
WARNING: Temperature high
FAULT: Leakage current detected in PMOS M3
INFO: Simulation completed
"""

with open("dummy_spice.log", "w") as f:
    f.write(dummy_log)

print("Dummy file created!")



run_fault_diagnosis("dummy_spice.log")


# ============================================================
#  ğŸ”§  FAULT DIAGNOSIS AGENT â€” FULL WORKING NOTEBOOK
# ============================================================

# This notebook contains:
# 1. Loop agent for hypothesis testing
# 2. SPICE log parser
# 3. LLM explainer logic
# 4. Memory bank for known faults
# 5. Observability (logging)
# 6. Dummy SPICE file generator (so it runs without files)
# 7. Final end-to-end test
# ============================================================


# ---------------------------
# 1ï¸�âƒ£ Loop Agent for Fault Diagnosis
# ---------------------------

def generate_hypotheses(log_lines):
    return [line for line in log_lines if "ERROR" in line or "FAULT" in line]

def test_hypothesis(hypothesis):
    if "short" in hypothesis.lower():
        return "confirmed"
    elif "leakage" in hypothesis.lower():
        return "confirmed"
    else:
        return "unconfirmed"

def fault_loop(log_lines):
    for hypothesis in generate_hypotheses(log_lines):
        result = test_hypothesis(hypothesis)
        if result == "confirmed":
            return hypothesis
    return "No fault confirmed"


# ---------------------------
# 2ï¸�âƒ£ SPICE Log Parser Tool
# ---------------------------

def parse_spice_log(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if "ERROR" in line or "FAULT" in line]


# ---------------------------
# 3ï¸�âƒ£ LLM Explainer Agent
# ---------------------------

def explain_fault(fault_line):
    if "short" in fault_line.lower():
        return "This indicates a short circuit between nodes, possibly due to layout overlap."
    elif "leakage" in fault_line.lower():
        return "Leakage current is flowing due to poor isolation or high temperature."
    else:
        return "Fault type unknown. Please verify with layout or simulation."


# ---------------------------
# 4ï¸�âƒ£ Memory Bank for Known Faults
# ---------------------------

memory_bank = {
    "F001": "Gate leakage due to thin oxide",
    "F002": "Drain-source short in NMOS",
    "F003": "Latch-up condition in CMOS"
}

def recall_fault(code):
    return memory_bank.get(code, "Fault code not found in memory bank.")


# ---------------------------
# 5ï¸�âƒ£ Logging / Observability
# ---------------------------

import logging
logging.basicConfig(filename="fault_agent_log.txt", level=logging.INFO)

def log_event(event):
    logging.info(f"{event}")


# ---------------------------
# 6ï¸�âƒ£ Create Dummy SPICE Log File (so notebook runs with no files)
# ---------------------------

dummy_log = """
INFO: Simulation started
ERROR: Drain-Source short detected in NMOS M1
WARNING: Temperature high
FAULT: Leakage current detected in PMOS M3
INFO: Simulation completed
"""

with open("dummy_spice.log", "w") as f:
    f.write(dummy_log)

print("âœ” Dummy SPICE log created: dummy_spice.log")


# ---------------------------
# 7ï¸�âƒ£ CLI-like runner (works inside notebook)
# ---------------------------

def run_fault_diagnosis(file_path):
    log_event(f"Started fault diagnosis on {file_path}")
    log_lines = parse_spice_log(file_path)
    
    fault = fault_loop(log_lines)
    explanation = explain_fault(fault)
    
    print("ğŸ§ª Fault Identified:", fault)
    print("ğŸ“– Explanation:", explanation)
    
    log_event(f"Diagnosis complete: {fault}")


# ---------------------------
# 8ï¸�âƒ£ Run Full System
# ---------------------------

run_fault_diagnosis("dummy_spice.log")


