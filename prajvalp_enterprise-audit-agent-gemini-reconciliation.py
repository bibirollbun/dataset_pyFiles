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


!pip install langgraph langchain pandas pillow gradio fastmcp pydantic google-generativeai python-dotenv nest_asyncio


import os
import json
import pandas as pd
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
from fastmcp import FastMCP
import gradio as gr
import glob
from dotenv import load_dotenv
import google.generativeai as genai
import time
import nest_asyncio

# Apply nest_asyncio to fix event loop issues in Colab/Jupyter
nest_asyncio.apply()

# --- CONFIGURATION ---

# Load API Key (Support for .env and Kaggle Secrets)
load_dotenv()

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GEMINI_API_KEY")
except:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("âš ï¸� WARNING: GEMINI_API_KEY not found. AI tools will fail.")
else:
    genai.configure(api_key=api_key)
    print("âœ… Gemini API Configured")

# --- MOCK DATABASE ---
PO_DATABASE = {
    "PO-100": {"po_number": "PO-100", "vendor": "Acme Corp", "amount": 1000.00, "currency": "USD"},
    "PO-101": {"po_number": "PO-101", "vendor": "Beta Inc", "amount": 500.00, "currency": "USD"},
    "PO-102": {"po_number": "PO-102", "vendor": "Gamma LLC", "amount": 750.00, "currency": "USD"},
    "PO-103": {"po_number": "PO-103", "vendor": "Delta Co", "amount": 1200.00, "currency": "USD"},
    "PO-104": {"po_number": "PO-104", "vendor": "Epsilon Ltd", "amount": 300.00, "currency": "USD"},
    "PO-105": {"po_number": "PO-105", "vendor": "Zeta Industries", "amount": 150.00, "currency": "USD"},
    "PO-106": {"po_number": "PO-106", "vendor": "Eta Services", "amount": 2000.00, "currency": "USD"},
    "PO-200": {"po_number": "PO-200", "vendor": "Alpha Corp", "amount": 5000.00, "currency": "USD"},
    "PO-201": {"po_number": "PO-201", "vendor": "Gamma LLC", "amount": 1000.00, "currency": "USD"}
}

# --- FASTMCP TOOLS ---

mcp = FastMCP("FinancialRecon")

# 1. Define Logic Functions

def read_csv_invoices(file_path: str) -> List[Dict]:
    """Reads invoices from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        records = df.to_dict(orient='records')
        cleaned = []
        for r in records:
            r = {k: (v if pd.notna(v) else None) for k, v in r.items()}
            cleaned.append(r)
        return cleaned
    except Exception as e:
        return [{"error": str(e)}]

def read_json_invoices(file_path: str) -> List[Dict]:
    """Reads invoices from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        invoices = []
        if isinstance(data, list):
            for item in data:
                if "invoices" in item:
                    invoices.extend(item["invoices"])
                else:
                    invoices.append(item)
        elif isinstance(data, dict):
             if "invoices" in data:
                 invoices.extend(data["invoices"])
             else:
                 invoices.append(data)
        
        normalized = []
        for inv in invoices:
            norm = {}
            norm["invoice_id"] = inv.get("invoice_id") or inv.get("id")
            norm["vendor_name"] = inv.get("vendor_name") or inv.get("vendor")
            norm["amount"] = inv.get("amount") or inv.get("total")
            norm["po_number"] = inv.get("po_number") or inv.get("po")
            normalized.append(norm)
            
        return normalized
    except Exception as e:
        return [{"error": str(e)}]

def _gemini_extract(file_path: str, mime_type: str) -> List[Dict]:
    """Helper to extract invoice data using Gemini."""
    try:
        model = genai.GenerativeModel("gemini-3-pro")
        myfile = genai.upload_file(file_path, mime_type=mime_type)
        
        # Wait for processing if needed (mostly for video, but good practice)
        while myfile.state.name == "PROCESSING":
            time.sleep(1)
            myfile = genai.get_file(myfile.name)
            
        prompt = """
        Extract the following invoice details from this document and return them as a JSON object:
        - invoice_id (string)
        - vendor_name (string)
        - amount (float, numeric value only)
        - po_number (string)
        - notes (string, any warnings, illegible text, or extra info)
        
        If a field is missing, use null.
        If there are multiple invoices, return a list of objects.
        Return ONLY the JSON. Do not include markdown formatting.
        """
        
        result = model.generate_content([myfile, prompt])
        text = result.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        return [{"error": f"Gemini Extraction Failed: {str(e)}"}]

def read_image_invoice(file_path: str) -> List[Dict]:
    """Extracts invoice data from an image using Gemini."""
    return _gemini_extract(file_path, mime_type="image/png") # Defaulting to png, Gemini detects actual type usually

def read_pdf_invoice(file_path: str) -> List[Dict]:
    """Extracts invoice data from a PDF using Gemini."""
    return _gemini_extract(file_path, mime_type="application/pdf")

# 2. Register Tools

@mcp.tool()
def tool_read_csv(file_path: str) -> List[Dict]:
    return read_csv_invoices(file_path)

@mcp.tool()
def tool_read_json(file_path: str) -> List[Dict]:
    return read_json_invoices(file_path)

@mcp.tool()
def tool_read_image(file_path: str) -> List[Dict]:
    return read_image_invoice(file_path)

@mcp.tool()
def tool_read_pdf(file_path: str) -> List[Dict]:
    return read_pdf_invoice(file_path)

def get_po_details(po_number: str) -> Optional[Dict]:
    return PO_DATABASE.get(po_number)


# --- STATE DEFINITION ---

class ReconciliationState(TypedDict):
    files_to_process: List[str]
    extracted_invoices: List[Dict]
    processed_results: List[Dict]
    current_file: Optional[str]
    logs: List[str]

# --- AGENTS / NODES ---

def file_router(state: ReconciliationState) -> ReconciliationState:
    """Routes files to FastMCP tools."""
    files = state["files_to_process"]
    if not files:
        return {"current_file": None}
    
    current_file = files[0]
    remaining_files = files[1:]
    
    extracted = []
    ext = os.path.splitext(current_file)[1].lower()
    
    # Call the logic functions directly
    if ext == ".csv":
        extracted = read_csv_invoices(current_file)
    elif ext == ".json":
        extracted = read_json_invoices(current_file)
    elif ext in [".jpg", ".png", ".jpeg"]:
        extracted = read_image_invoice(current_file)
    elif ext == ".pdf":
        extracted = read_pdf_invoice(current_file)
    else:
        extracted = []
        
    # Add source info
    for record in extracted:
        record["source_file"] = os.path.basename(current_file)
        
    return {
        "files_to_process": remaining_files,
        "current_file": current_file,
        "extracted_invoices": state["extracted_invoices"] + extracted,
        "logs": state["logs"] + [f"Ingested {len(extracted)} records from {os.path.basename(current_file)}"]
    }

def reconciliation_agent(state: ReconciliationState) -> ReconciliationState:
    """Reconciles invoices against POs."""
    queue = state["extracted_invoices"]
    if not queue:
        return {}
    
    invoice = queue[0]
    remaining_queue = queue[1:]
    
    po_number = invoice.get("po_number")
    po_data = get_po_details(po_number)
    
    result = {
        "invoice_id": invoice.get("invoice_id"),
        "vendor": invoice.get("vendor_name"),
        "invoice_amount": invoice.get("amount"),
        "po_number": po_number,
        "source": invoice.get("source_file"),
        "status": "PENDING",
        "notes": invoice.get("notes") or ""
    }
    
    if not po_number:
        result["status"] = "FLAGGED"
        result["notes"] += " Missing PO Number."
    elif not po_data:
        result["status"] = "FLAGGED"
        result["notes"] += " PO not found in DB."
    else:
        try:
            inv_amt = float(invoice.get("amount", 0))
            po_amt = float(po_data.get("amount", 0))
            variance = inv_amt - po_amt
            
            if abs(variance) < 0.01:
                result["status"] = "MATCHED"
                result["notes"] += " Perfect match."
            elif abs(variance) < 50.0:
                result["status"] = "AUTO-RESOLVED"
                result["notes"] += f" Variance {variance:.2f} within threshold."
            else:
                result["status"] = "VARIANCE"
                result["notes"] += f" Significant variance: {variance:.2f}."
        except:
             result["status"] = "ERROR"
             result["notes"] += " Invalid amount format."
            
    return {
        "extracted_invoices": remaining_queue,
        "processed_results": state["processed_results"] + [result],
        "logs": state["logs"] + [f"Processed {invoice.get('invoice_id')}: {result['status']}"]
    }

# --- GRAPH CONSTRUCTION ---

workflow = StateGraph(ReconciliationState)
workflow.add_node("router", file_router)
workflow.add_node("reconciler", reconciliation_agent)
workflow.set_entry_point("router")

def router_condition(state: ReconciliationState):
    if state["extracted_invoices"]:
        return "reconciler"
    elif state["files_to_process"]:
        return "router"
    else:
        return END

def reconciler_condition(state: ReconciliationState):
    if state["extracted_invoices"]:
        return "reconciler"
    elif state["files_to_process"]:
        return "router"
    else:
        return END

workflow.add_conditional_edges("router", router_condition)
workflow.add_conditional_edges("reconciler", reconciler_condition)

app = workflow.compile()


# --- GRADIO UI ---

import warnings
warnings.filterwarnings("ignore")

def process_files(files):
    if not files:
        return "No files uploaded.", pd.DataFrame()
    
    file_paths = [f.name for f in files]
    
    initial_state = {
        "files_to_process": file_paths,
        "extracted_invoices": [],
        "processed_results": [],
        "current_file": None,
        "logs": []
    }
    
    try:
        final_state = app.invoke(initial_state, config={"recursion_limit": 1000})
        logs_text = "\n".join(final_state["logs"])
        results_df = pd.DataFrame(final_state["processed_results"])
    except Exception as e:
        logs_text = f"Error during processing: {str(e)}"
        results_df = pd.DataFrame()
    
    return logs_text, results_df

with gr.Blocks(title="Financial Reconciliation System") as demo:
    gr.Markdown("# ğŸ§¾ Multi-Agent Financial Reconciliation System")
    gr.Markdown("Upload invoices (CSV, JSON, PDF, Images) to reconcile them against the mock PO database.")
    
    with gr.Row():
        file_input = gr.File(file_count="multiple", label="Upload Invoices")
        process_btn = gr.Button("Process Invoices", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ğŸ“œ Audit Logs")
            logs_output = gr.Textbox(label="System Logs", lines=10)
        with gr.Column():
            gr.Markdown("### ğŸ“Š Reconciliation Results")
            results_output = gr.Dataframe(label="Results")
            
    process_btn.click(
        fn=process_files,
        inputs=file_input,
        outputs=[logs_output, results_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)




