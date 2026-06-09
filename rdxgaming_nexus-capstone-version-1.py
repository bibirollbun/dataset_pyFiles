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


# [CELL 1] NEXUS SETUP
# Installing PDF readers and AI tools

!pip install -q -U google-generativeai
!pip install -q PyPDF2

import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import os
import io
import PyPDF2
import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output

# 1. AUTHENTICATE
try:
    user_secrets = UserSecretsClient()
    my_api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=my_api_key)
    print("âœ… NEXUS SYSTEMS: ONLINE. Connected to Gemini.")
except:
    print("â�Œ ERROR: API Key missing. Please check Kaggle Secrets.")

# 2. STYLE
style_css = """
<style>
    .nexus-header {
        background: linear-gradient(90deg, #1c1c1c, #3a3a3a);
        padding: 20px;
        border-radius: 8px;
        color: #e0e0e0;
        text-align: center;
        border-bottom: 3px solid #00ff9d;
        font-family: monospace;
    }
    .nexus-btn { width: 100%; font-weight: bold; }
</style>
"""
display(widgets.HTML(style_css))


# [CELL 2] THE RESEARCH BRAIN (Fixed for Kaggle ipywidgets)

# 1. SMART MODEL SELECTOR (Self-Healing)
def get_best_model():
    """Finds the best available Gemini model."""
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Priority: Pro (better for reading) -> Flash (faster)
        for m in available_models:
            if 'gemini-1.5-pro' in m: return m 
        for m in available_models:
            if 'flash' in m: return m
        return available_models[0]
    except:
        return 'models/gemini-1.5-flash'

CURRENT_MODEL_NAME = get_best_model()
print(f"âœ… NEXUS BRAIN: Using {CURRENT_MODEL_NAME}")

# --- 2. TOOL: PDF EXTRACTOR (Fixed for ipywidgets 8.0+) ---
def extract_text_from_uploads(uploaded_files):
    """Reads raw bytes from Kaggle Upload Widget and converts to text."""
    combined_text = ""
    
    # CASE A: Newer ipywidgets (Tuple of Dictionaries)
    if isinstance(uploaded_files, tuple):
        for file_info in uploaded_files:
            try:
                filename = file_info.get('name', 'Unknown PDF')
                content = file_info.get('content')
                
                # Check if content is bytes (needed for PdfReader)
                if isinstance(content, memoryview):
                    content = bytes(content)
                
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                
                text = f"\n--- START OF DOCUMENT: {filename} ---\n"
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                text += f"\n--- END OF DOCUMENT: {filename} ---\n"
                combined_text += text
                
            except Exception as e:
                return f"Error reading {filename}: {str(e)}"

    # CASE B: Older ipywidgets (Dictionary)
    else:
        for filename, file_info in uploaded_files.items():
            try:
                content = file_info['content'] 
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                
                text = f"\n--- START OF DOCUMENT: {filename} ---\n"
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                text += f"\n--- END OF DOCUMENT: {filename} ---\n"
                combined_text += text
            except Exception as e:
                return f"Error reading {filename}: {str(e)}"
            
    return combined_text

# --- 3. AGENT: THE SYNTHESIZER ---
def run_nexus_agent(raw_text):
    model = genai.GenerativeModel(CURRENT_MODEL_NAME)
    
    nexus_prompt = """
    You are NEXUS, a Senior Research Synthesis Agent.
    
    YOUR TASK:
    You have been provided with the text of one or multiple research papers/documents.
    Perform a high-level "Comparative Analysis".
    
    OUTPUT FORMAT (Markdown):
    ## ğŸ§¬ NEXUS Synthesis Report
    
    ### 1. Executive Summary
    (2-3 sentences explaining what these papers are collectively about).
    
    ### 2. Key Consensus
    (What do all documents AGREE on?)
    
    ### 3. Critical Conflicts & Differences
    (Where do they DISAGREE? Or what unique angle does each paper take?)
    * **Conflict 1:** ...
    * **Conflict 2:** ...
    
    ### 4. Methodological Strengths/Weaknesses
    (Critique the data/methods if mentioned).
    
    ### 5. Final Conclusion
    (The scientific verdict).
    """
    
    try:
        # Generate analysis
        response = model.generate_content([nexus_prompt, raw_text])
        return response.text
    except Exception as e:
        return f"â�Œ Analysis Failed: {str(e)}"


# [CELL 3] THE INTERFACE

# Header
display(widgets.HTML('<div class="nexus-header"><h1>ğŸ§¬ N E X U S</h1><h3>Research Synthesis Agent</h3></div>'))

# 1. Upload Widget
lbl_upload = widgets.HTML("<b>Step 1: Upload Research Papers (PDF)</b>")
uploader = widgets.FileUpload(
    accept='.pdf',  # Accept only .pdf files
    multiple=True   # Allow multiple files at once
)

# 2. Button & Output
btn_analyze = widgets.Button(
    description='RUN SYNTHESIS',
    button_style='success', # Green
    icon='microchip',
    layout=widgets.Layout(width='100%', margin='20px 0px 0px 0px')
)
out_nexus = widgets.Output()

# 3. Logic
def on_analyze_click(b):
    out_nexus.clear_output()
    
    # Check if files exist
    if not uploader.value:
        with out_nexus:
            print("âš ï¸� Please upload at least one PDF file first.")
            return
            
    with out_nexus:
        print(f"ğŸ“„ Reading {len(uploader.value)} document(s)...")
        
        # Extract Text
        raw_text = extract_text_from_uploads(uploader.value)
        
        if len(raw_text) < 50:
            print("â�Œ Error: Could not extract text. Are the PDFs scanned images? (NEXUS needs selectable text).")
            return
            
        print("ğŸ§  NEXUS is synthesizing connections... (This may take 30s)")
        
        # Run Agent
        report = run_nexus_agent(raw_text)
        
        clear_output()
        display(Markdown(report))

btn_analyze.on_click(on_analyze_click)

# Display UI
display(widgets.VBox([lbl_upload, uploader, btn_analyze, out_nexus]))

