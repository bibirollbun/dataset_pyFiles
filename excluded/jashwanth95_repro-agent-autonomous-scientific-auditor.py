# Shiva


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


# Cell 1: Install Stable Dependencies
!pip install -U -q google-generativeai pypdf wikipedia

import os
import json
import uuid
import logging
import requests
import wikipedia
from datetime import datetime
from pypdf import PdfReader
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# OBSERVABILITY
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("EnterpriseSystem")

# AUTH
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GEMINI_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    print("âœ… Authenticated with Gemini 2.5-flash")
except Exception as e:
    print(f"â�Œ Auth Error: {e}")


import warnings
warnings.filterwarnings('ignore')


# Cell 2: Official Patterns Implementation

# --- FEATURE: SESSIONS & MEMORY ---
# Implementation of the 'InMemorySessionService' pattern from Google ADK
class InMemorySessionService:
    def __init__(self):
        self._store = {} # The "Database"

    def create_session(self, user_id="default"):
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        self._store[session_id] = {
            "user_id": user_id,
            "history": [],        # Chat History
            "artifacts": {}       # Context Engineering Artifacts
        }
        logger.info(f"ğŸ†• Session Created: {session_id}")
        return session_id

    def get_history(self, session_id):
        return self._store[session_id]["history"]

    def add_turn(self, session_id, role, text):
        self._store[session_id]["history"].append({"role": role, "parts": [text]})

    def save_artifact(self, session_id, key, data):
        """Save structured context (Context Engineering)"""
        self._store[session_id]["artifacts"][key] = data
        logger.info(f"ğŸ’¾ Artifact Saved: {key} -> Session {session_id}")

    def get_artifact(self, session_id, key):
        return self._store[session_id]["artifacts"].get(key)

# --- FEATURE: A2A PROTOCOL ---
# Implementation of the 'Agent Card' standard for discovery
def generate_agent_card(agent_name, capabilities):
    """Generates an A2A-compliant Agent Card (agent.json)."""
    card = {
        "schema_version": "v1",
        "metadata": {
            "name": agent_name,
            "id": f"agent-{uuid.uuid4().hex[:6]}",
            "created_at": datetime.now().isoformat()
        },
        "capabilities": capabilities,
        "interfaces": {"a2a": {"protocol": "http", "version": "1.0"}}
    }
    # We save this to disk to prove A2A compliance
    filename = f"{agent_name.lower()}_card.json"
    with open(filename, "w") as f:
        json.dump(card, f, indent=2)
    logger.info(f"ğŸ“‡ A2A Card Generated: {filename}")
    return card

# --- FEATURE: MCP CONFIGURATION ---
# We generate the config file required to connect to an MCP server
def generate_mcp_config():
    config = {
        "mcpServers": {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"]
            }
        }
    }
    with open("mcp_config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("ZC MCP Config Generated: mcp_config.json")

print("âœ… Enterprise Protocols (Session, A2A, MCP) Initialized")


# Cell 3: Enterprise Agent Class 
import base64
import requests
import json
import wikipedia
from pypdf import PdfReader
from google.generativeai import protos
import google.generativeai as genai

# --- 1. TOOLS ---
def wiki_search(query: str) -> str:
    """Searches Wikipedia to verify the paper's concept."""
    try:
        results = wikipedia.search(query)
        if not results: return "No Wikipedia results found."
        page = wikipedia.page(results[0], auto_suggest=False)
        return f"Wikipedia Title: {page.title}\nSummary: {page.summary[:500]}"
    except Exception as e: return f"Wikipedia Error: {str(e)}"

def inspect_github_content(repo_url: str, specific_file_path: str = None) -> dict:
    """Two-Mode Tool: List files OR Read a specific file."""
    try:
        clean = repo_url.rstrip("/")
        parts = clean.split("/")
        owner, repo = parts[-2], parts[-1]
        if specific_file_path:
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{specific_file_path}"
            resp = requests.get(api_url)
            if resp.status_code == 200:
                content = resp.json()['content']
                decoded = base64.b64decode(content).decode('utf-8', errors='ignore')
                return {"file_path": specific_file_path, "code_snippet": decoded[:8000]}
            return {"error": f"Could not read file: {specific_file_path}"}
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        resp = requests.get(api_url)
        if resp.status_code != 200: return {"error": "Repo not found"}
        files = [i['name'] for i in resp.json()]
        return {"root_files": files}
    except Exception as e: return {"error": str(e)}

def read_pdf_content(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            if i >= 12: break
            text += page.extract_text() + "\n"
        return text
    except Exception as e: return f"Error: {str(e)}"

# --- 2. AGENT CLASS ---
class EnterpriseAgent:
    def __init__(self, name, instructions, tools=None, enable_code_execution=False):
        self.name = name
        self.instructions = instructions
        self.session_service = InMemorySessionService() 
        self.active_tools = tools if tools else []
        if enable_code_execution:
            self.active_tools.append("code_execution")
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash", 
            tools=self.active_tools, 
            system_instruction=instructions
        )
        generate_agent_card(name, ["reasoning", "forensic-analysis"])

    def run(self, session_id, prompt, inject_artifact=None):
        final_prompt = prompt
        if inject_artifact:
            context_str = json.dumps(inject_artifact, indent=2)
            final_prompt = f"""[SCIENTIFIC TRUTH FROM PAPER]\n{context_str}\n\n[YOUR TASK]\n{prompt}"""

        history = self.session_service.get_history(session_id)
        chat = self.model.start_chat(history=history, enable_automatic_function_calling=True)
        
        logger.info(f"ğŸ¤– {self.name} investigating...")
        try:
            # ğŸ› ï¸� FIX: Explicitly handle code execution response
            response = chat.send_message(final_prompt)
            self.session_service.add_turn(session_id, "user", final_prompt)
            
            # Check for text FIRST, even if code was executed
            if response.text:
                self.session_service.add_turn(session_id, "model", response.text)
                return response.text
            
            # Fallback for code execution without text output
            if response.parts and response.parts[0].executable_code:
                 # Try to get the next turn which might contain the text report
                try:
                    next_response = chat.send_message("Please provide the final report text based on the code execution.")
                    if next_response.text:
                         self.session_service.add_turn(session_id, "model", next_response.text)
                         return next_response.text
                except:
                    pass # Fallback if that fails
                return "Code Executed Successfully, but no final text report was generated."
                
            return "Task Completed."
        except Exception as e:
            return f"Execution Error: {e}"

print("âœ… Enterprise Forensic Agents Ready ")


# Cell 4: Run the Forensic Math Audit
from IPython.display import HTML
import markdown
import os

# 1. Setup Services
global_session_service = InMemorySessionService()
session_id = global_session_service.create_session(user_id="kaggle_judge")
generate_mcp_config()

# 2. Define Agents
bg_checker = EnterpriseAgent(
    "Background_Checker", 
    "Verify paper credibility via Wikipedia.", 
    tools=[wiki_search]
)
bg_checker.session_service = global_session_service 

paper_analyst = EnterpriseAgent(
    "Paper_Analyst", 
    "Extract the 'Key Algorithms' and 'Mathematical Formulas' from PDF.", 
    tools=[read_pdf_content]
)
paper_analyst.session_service = global_session_service

code_forensics = EnterpriseAgent(
    "Code_Forensics", 
    "You are a Mathematical Code Auditor. Your job is to find the ACTUAL PYTHON FUNCTION DEFINITION (def ...) and extract the math logic.", 
    tools=[inspect_github_content] 
)
code_forensics.session_service = global_session_service

# Academic Report Writer
writer = EnterpriseAgent(
    "Report_Writer", 
    "You are a Scientific Reviewer. Write a formal, academic-style report on the reproducibility of the paper's claims in the codebase. Use formal language, clear headings (Abstract, Methodology, Results, Discussion), and cite the provided evidence.", 
    enable_code_execution=True
)
writer.session_service = global_session_service

# --- EXECUTION ---
pdf_path = "/kaggle/input/bert2019/bert.pdf" 
repo_url = "https://github.com/google-research/bert"

print(f"\nğŸš€ Starting Forensic Math Audit: {session_id}\n")

# PHASE 1: Verify Topic
print("ğŸŒ� Phase 1: Topic Verification")
bg_info = bg_checker.run(session_id, f"Search for 'BERT language model'")

# PHASE 2: Extract Algorithms
print("ğŸ“� Phase 2: Extracting Math Specs")
paper_math = paper_analyst.run(session_id, f"Read {pdf_path}. explicitly extract the Activation Function (Gelu/Relu) and Optimizer details.")

research_artifact = {"source": "BERT", "algorithms": paper_math[:1000]}
global_session_service.save_artifact(session_id, "research_math", research_artifact)
print(f"   > Specs Captured: {paper_math[:100]}...\n")

# PHASE 3: Forensic Code Audit
print("ğŸ’» Phase 3: Forensic Code Audit (Hunting for Math)")
audit_findings = code_forensics.run(
    session_id, 
    f"""
    Target Repo: {repo_url}
    
    TASK 1: Find the file 'modeling.py'.
    TASK 2: Read it and find the function definition `def gelu` or `def get_activation`.
    TASK 3: Extract the EXACT 3-4 lines of code that perform the mathematical calculation.
    
    Compare this math against the standard definition.
    """,
    inject_artifact=research_artifact 
)
print(f"   > Forensics Complete.\n")

# PHASE 4: Final Academic Report
print("âš–ï¸� Phase 4: Final Academic Forensic Report")
final_prompt = f"""
[EVIDENCE 1: PAPER SPECS] {paper_math}
[EVIDENCE 2: CODE IMPLEMENTATION] {audit_findings}

Generate a formal **Reproducibility Audit Report** in an academic style.

**Structure:**
1.  **Abstract:** A brief summary.
2.  **Methodology:** Describe the audit process.
3.  **Results:**
    * **Reproducibility Score:** Assign a score (1-4) with a justification.
    * **Mathematical Verification:** Present the code snippet and analyze its equivalence.
    * **Discrepancy Analysis:** Provide a table comparing the "Paper Claim" vs. "Code Reality".
4.  **Discussion:** Discuss implications and ambiguity.
5.  **Conclusion:** A final statement.
6.  **References:** List sources.

**Tone:** Formal, objective, and scientific.
"""
final_report = writer.run(session_id, final_prompt)

# ğŸ› ï¸� UPDATE: Use HTML display to prevent truncation
html_report = markdown.markdown(final_report, extensions=['tables'])
display(HTML(f"<h1>ğŸ“œ Academic Forensic Audit Report</h1>{html_report}"))

# Kaggle-Friendly File Saving
output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)
report_filename = os.path.join(output_dir, f"submission_report.md")
with open(report_filename, "w") as f:
    f.write(final_report)

print(f"\nâœ… Report Saved to Disk: {report_filename}")
print("   (This file will be available in the 'Output' tab after the notebook finishes running.)")


import google.generativeai as genai
from google.generativeai import generative_models

print("ğŸ”� Gemini SDK Version:", genai.__version__)
print("\n=== Testing Google Search Tool Support ===\n")

tests = {}

# Test 1 â€” tools=[{"google_search":{}}]
try:
    genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=[{"google_search": {}}]
    )
    tests["tools=[{'google_search':{}}]"] = "âœ” WORKS"
except Exception as e:
    tests["tools=[{'google_search':{}}]"] = f"â�Œ {type(e).__name__}: {e}"

# Test 2 â€” tools=[{"google_search_retrieval":{}}]
try:
    genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=[{"google_search_retrieval": {}}]
    )
    tests["tools=[{'google_search_retrieval':{}}]"] = "âœ” WORKS"
except Exception as e:
    tests["tools=[{'google_search_retrieval':{}}]"] = f"â�Œ {type(e).__name__}: {e}"

# Test 3 â€” tool_config={"google_search_retrieval":{}}
try:
    genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tool_config={"google_search_retrieval": {}}
    )
    tests["tool_config={'google_search_retrieval':{}}"] = "âœ” WORKS"
except Exception as e:
    tests["tool_config={'google_search_retrieval':{}}"] = f"â�Œ {type(e).__name__}: {e}"

# Test 4 â€” ToolType: GoogleSearchRetrieval()
try:
    from google.generativeai.types import tool_types
    search_tool = tool_types.GoogleSearchRetrieval()
    genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=[search_tool]
    )
    tests["tool_types.GoogleSearchRetrieval()"] = "âœ” WORKS"
except Exception as e:
    tests["tool_types.GoogleSearchRetrieval()"] = f"â�Œ {type(e).__name__}: {e}"

# Show results
print("=== RESULTS ===")
for name, result in tests.items():
    print(f"{name:45}  â†’  {result}")



import google.generativeai as genai

print("SDK Version:", genai.__version__)

print("\n=== AVAILABLE MODELS (v1beta) ===")
models = genai.list_models()

for m in models:
    print("-", m.name)



# # Cell 3: Enterprise Agent Class (Forensic Code Tools)
# import base64
# import requests
# import json
# import wikipedia
# from pypdf import PdfReader
# from google.generativeai import protos
# import google.generativeai as genai

# # --- 1. TOOLS ---

# def wiki_search(query: str) -> str:
#     """Searches Wikipedia to verify the paper's concept."""
#     try:
#         # Broader search to ensure hits
#         results = wikipedia.search(query)
#         if not results: return "No Wikipedia results found."
#         # Get the most relevant page
#         page = wikipedia.page(results[0], auto_suggest=False)
#         return f"Wikipedia Title: {page.title}\nSummary: {page.summary[:500]}"
#     except Exception as e: return f"Wikipedia Error: {str(e)}"

# def inspect_github_content(repo_url: str, specific_file_path: str = None) -> dict:
#     """
#     Two-Mode Tool:
#     1. If 'specific_file_path' is None -> Lists all files in the repo root.
#     2. If 'specific_file_path' is provided (e.g. 'modeling.py') -> Reads the ACTUAL CODE.
#     """
#     try:
#         clean = repo_url.rstrip("/")
#         parts = clean.split("/")
#         owner, repo = parts[-2], parts[-1]
        
#         # Mode 2: Read Specific File (Forensic Mode)
#         if specific_file_path:
#             api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{specific_file_path}"
#             resp = requests.get(api_url)
#             if resp.status_code == 200:
#                 content = resp.json()['content']
#                 decoded = base64.b64decode(content).decode('utf-8', errors='ignore')
#                 # Return first 8000 chars of code (enough to check algorithms)
#                 return {"file_path": specific_file_path, "code_snippet": decoded[:8000]}
#             return {"error": f"Could not read file: {specific_file_path}"}
            
#         # Mode 1: List Files (Discovery Mode)
#         api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
#         resp = requests.get(api_url)
#         if resp.status_code != 200: return {"error": "Repo not found"}
#         files = [i['name'] for i in resp.json()]
#         return {"root_files": files}

#     except Exception as e: return {"error": str(e)}

# def read_pdf_content(pdf_path: str) -> str:
#     try:
#         reader = PdfReader(pdf_path)
#         text = ""
#         for i, page in enumerate(reader.pages):
#             if i >= 12: break # Read more pages for algorithm details
#             text += page.extract_text() + "\n"
#         return text
#     except Exception as e: return f"Error: {str(e)}"

# # --- 2. AGENT CLASS ---
# class EnterpriseAgent:
#     def __init__(self, name, instructions, tools=None, enable_code_execution=False):
#         self.name = name
#         self.instructions = instructions
#         self.session_service = InMemorySessionService() 
#         self.active_tools = tools if tools else []
        
#         if enable_code_execution:
#             self.active_tools.append("code_execution")
        
#         # Using Gemini 2.5 Flash for Deep Reasoning
#         self.model = genai.GenerativeModel(
#             model_name="gemini-2.5-flash", 
#             tools=self.active_tools, 
#             system_instruction=instructions
#         )
#         generate_agent_card(name, ["reasoning", "forensic-analysis"])

#     def run(self, session_id, prompt, inject_artifact=None):
#         final_prompt = prompt
        
#         if inject_artifact:
#             # We inject the "Scientific Truth" to compare against
#             context_str = json.dumps(inject_artifact, indent=2)
#             final_prompt = f"""
#             [SCIENTIFIC TRUTH FROM PAPER]
#             {context_str}
            
#             [YOUR TASK]
#             {prompt}
#             """

#         history = self.session_service.get_history(session_id)
#         chat = self.model.start_chat(history=history, enable_automatic_function_calling=True)
        
#         logger.info(f"ğŸ¤– {self.name} investigating...")
#         try:
#             response = chat.send_message(final_prompt)
#             self.session_service.add_turn(session_id, "user", final_prompt)
            
#             if response.parts and response.parts[0].text:
#                 self.session_service.add_turn(session_id, "model", response.text)
#                 return response.text
            
#             if response.parts and response.parts[0].executable_code:
#                 return "Code Executed Successfully."
                
#             return "Task Completed."
#         except Exception as e:
#             return f"Execution Error: {e}"

# print("âœ… Enterprise Forensic Agents Ready")


# # # Cell 4: Run the Forensic Math Audit
# # from IPython.display import Markdown

# # # 1. Setup Services
# # global_session_service = InMemorySessionService()
# # session_id = global_session_service.create_session(user_id="kaggle_judge")
# # generate_mcp_config()

# # # 2. Define Agents
# # bg_checker = EnterpriseAgent(
# #     "Background_Checker", 
# #     "Verify paper credibility via Wikipedia.", 
# #     tools=[wiki_search]
# # )
# # bg_checker.session_service = global_session_service 

# # paper_analyst = EnterpriseAgent(
# #     "Paper_Analyst", 
# #     "Extract the 'Key Algorithms' and 'Mathematical Formulas' from PDF.", 
# #     tools=[read_pdf_content]
# # )
# # paper_analyst.session_service = global_session_service

# # code_forensics = EnterpriseAgent(
# #     "Code_Forensics", 
# #     "You are a Mathematical Code Auditor. Your job is to find the ACTUAL PYTHON FUNCTION DEFINITION (def ...) and extract the math logic.", 
# #     tools=[inspect_github_content] 
# # )
# # code_forensics.session_service = global_session_service

# # writer = EnterpriseAgent("Report_Writer", "Write a Forensic Audit Report.", enable_code_execution=True)
# # writer.session_service = global_session_service

# # # --- EXECUTION ---
# # pdf_path = "/kaggle/input/bert2019/bert.pdf" 
# # repo_url = "https://github.com/google-research/bert"

# # print(f"\nğŸš€ Starting Forensic Math Audit: {session_id}\n")

# # # PHASE 1: Verify Topic
# # print("ğŸŒ� Phase 1: Topic Verification")
# # bg_info = bg_checker.run(session_id, f"Search for 'BERT language model'")

# # # PHASE 2: Extract Algorithms
# # print("ğŸ“� Phase 2: Extracting Math Specs")
# # paper_math = paper_analyst.run(session_id, f"Read {pdf_path}. explicitly extract the Activation Function (Gelu/Relu) and Optimizer details.")

# # research_artifact = {"source": "BERT", "algorithms": paper_math[:1000]}
# # global_session_service.save_artifact(session_id, "research_math", research_artifact)
# # print(f"   > Specs Captured: {paper_math[:100]}...\n")

# # # PHASE 3: Forensic Code Audit (Aggressive Math Hunt)
# # print("ğŸ’» Phase 3: Forensic Code Audit (Hunting for Math)")
# # audit_findings = code_forensics.run(
# #     session_id, 
# #     f"""
# #     Target Repo: {repo_url}
    
# #     TASK 1: Find the file 'modeling.py'.
# #     TASK 2: Read it and find the function definition `def gelu` or `def get_activation`.
# #     TASK 3: Extract the EXACT 3-4 lines of code that perform the mathematical calculation.
    
# #     Compare this math against the standard definition.
# #     """,
# #     inject_artifact=research_artifact 
# # )
# # print(f"   > Forensics Complete.\n")

# # # PHASE 4: Technical Report (Score & Diff)
# # print("âš–ï¸� Phase 4: Final Forensic Report")
# # final_prompt = f"""
# # [EVIDENCE 1: PAPER SPECS] {paper_math}
# # [EVIDENCE 2: CODE IMPLEMENTATION] {audit_findings}

# # Generate a "Reproducibility Discrepancy Report".

# # 1. **Reproducibility Score (1-4):**
# #    - 4: Perfect match.
# #    - 3: Match, but paper was vague (Ambiguity Penalty).
# #    - 2: Minor code deviations.
# #    - 1: Major mismatch.

# # 2. **The "Math Check":**
# #    - Show the Code Snippet found.
# #    - Explain if the formula is mathematically equivalent to the paper's claim.

# # 3. **Discrepancy Table:**
# #    - Create a Markdown Table comparing "Paper Claim" vs "Code Reality".
   
# # 4. **Implementation Proof:**
# #    - Quote the exact file name and line number range found.
# # """
# # final_report = writer.run(session_id, final_prompt)

# # display(Markdown(f"# ğŸ“Š Forensic Audit Report\n\n{final_report}"))
# # # FEATURE: Save Report to Disk (Enterprise Requirement)
# # report_filename = f"reproducibility_report_{session_id}.md"
# # with open(report_filename, "w") as f:
# #     f.write(final_report)

# # print(f"\nâœ… Report Saved to Disk: {report_filename}")
# # print("   (You can download this file from the 'Output' tab)")
# # Cell 4: Run the Forensic Math Audit
# from IPython.display import Markdown
# import os

# # 1. Setup Services
# global_session_service = InMemorySessionService()
# session_id = global_session_service.create_session(user_id="kaggle_judge")
# generate_mcp_config()

# # 2. Define Agents
# bg_checker = EnterpriseAgent(
#     "Background_Checker", 
#     "Verify paper credibility via Wikipedia.", 
#     tools=[wiki_search]
# )
# bg_checker.session_service = global_session_service 

# paper_analyst = EnterpriseAgent(
#     "Paper_Analyst", 
#     "Extract the 'Key Algorithms' and 'Mathematical Formulas' from PDF.", 
#     tools=[read_pdf_content]
# )
# paper_analyst.session_service = global_session_service

# code_forensics = EnterpriseAgent(
#     "Code_Forensics", 
#     "You are a Mathematical Code Auditor. Your job is to find the ACTUAL PYTHON FUNCTION DEFINITION (def ...) and extract the math logic.", 
#     tools=[inspect_github_content] 
# )
# code_forensics.session_service = global_session_service

# # ğŸ› ï¸� UPDATE: Academic Report Writer Instructions
# writer = EnterpriseAgent(
#     "Report_Writer", 
#     "You are a Scientific Reviewer. Write a formal, academic-style report on the reproducibility of the paper's claims in the codebase. Use formal language, clear headings (Abstract, Methodology, Results, Discussion), and cite the provided evidence.", 
#     enable_code_execution=True
# )
# writer.session_service = global_session_service

# # --- EXECUTION ---
# pdf_path = "/kaggle/input/bert2019/bert.pdf" 
# repo_url = "https://github.com/google-research/bert"

# print(f"\nğŸš€ Starting Forensic Math Audit: {session_id}\n")

# # PHASE 1: Verify Topic
# print("ğŸŒ� Phase 1: Topic Verification")
# bg_info = bg_checker.run(session_id, f"Search for 'BERT language model'")

# # PHASE 2: Extract Algorithms
# print("ğŸ“� Phase 2: Extracting Math Specs")
# paper_math = paper_analyst.run(session_id, f"Read {pdf_path}. explicitly extract the Activation Function (Gelu/Relu) and Optimizer details.")

# research_artifact = {"source": "BERT", "algorithms": paper_math[:1000]}
# global_session_service.save_artifact(session_id, "research_math", research_artifact)
# print(f"   > Specs Captured: {paper_math[:100]}...\n")

# # PHASE 3: Forensic Code Audit (Aggressive Math Hunt)
# print("ğŸ’» Phase 3: Forensic Code Audit (Hunting for Math)")
# audit_findings = code_forensics.run(
#     session_id, 
#     f"""
#     Target Repo: {repo_url}
    
#     TASK 1: Find the file 'modeling.py'.
#     TASK 2: Read it and find the function definition `def gelu` or `def get_activation`.
#     TASK 3: Extract the EXACT 3-4 lines of code that perform the mathematical calculation.
    
#     Compare this math against the standard definition.
#     """,
#     inject_artifact=research_artifact 
# )
# print(f"   > Forensics Complete.\n")

# # PHASE 4: Technical Report (Score & Diff)
# print("âš–ï¸� Phase 4: Final Academic Forensic Report")
# final_prompt = f"""
# [EVIDENCE 1: PAPER SPECS] {paper_math}
# [EVIDENCE 2: CODE IMPLEMENTATION] {audit_findings}

# Generate a formal **Reproducibility Audit Report** in an academic style.

# **Structure:**
# 1.  **Abstract:** A brief summary of the audit's purpose, methods, and main finding (the reproducibility score).
# 2.  **Methodology:** Describe the process of auditing the paper against the codebase.
# 3.  **Results:**
#     * **Reproducibility Score:** Assign a score (1-4) with a justification.
#     * **Mathematical Verification:** Present the identified code snippet for the activation function and analyze its mathematical equivalence to the paper's claim.
#     * **Discrepancy Analysis:** Provide a table comparing the "Paper Claim" vs. "Code Reality".
# 4.  **Discussion:** Discuss the implications of the findings, including any ambiguity in the paper and the correctness of the implementation.
# 5.  **Conclusion:** A final concluding statement on the reproducibility of the audited feature.
# 6.  **References:** List the paper and repository as sources.

# **Tone:** Formal, objective, and scientific.
# """
# final_report = writer.run(session_id, final_prompt)

# display(Markdown(f"# ğŸ“œ Academic Forensic Audit Report\n\n{final_report}"))

# # ğŸ› ï¸� UPDATE: Kaggle-Friendly File Saving
# # Save to /kaggle/working/ for persistence after commit
# output_dir = "/kaggle/working/"
# os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
# report_filename = os.path.join(output_dir, f"submission_report.md")

# with open(report_filename, "w") as f:
#     f.write(final_report)

# print(f"\nâœ… Report Saved to Disk: {report_filename}")
# print("   (This file will be available in the 'Output' tab after the notebook finishes running.)")

# # Optional: Read back the report to confirm it was saved correctly
# # with open(report_filename, "r") as f:
# #     print("\n--- Verifying Saved Report Content ---")
# #     print(f.read()[:500] + "...\n(truncated)")


# # Cell 3: Enterprise Agent Class (Wikipedia + Code Execution)
# import base64
# import requests
# import json
# from pypdf import PdfReader
# from google.generativeai import protos
# # --- 1. CUSTOM TOOL: Wikipedia Search ---
# def wiki_search(query: str) -> str:
#     """Searches Wikipedia to verify if a topic is real and popular."""
#     try:
#         # Get summary of first result
#         results = wikipedia.search(query)
#         if not results: return "No Wikipedia results found."
#         summary = wikipedia.summary(results[0], sentences=3)
#         return f"Wikipedia Summary for '{results[0]}': {summary}"
#     except Exception as e: return f"Wikipedia Error: {str(e)}"

# # --- 2. SMART TOOL: GitHub Scanner + README Reader ---
# def check_github_files(repo_url: str) -> dict:
#     """Scans a repo AND reads the README.md for installation details."""
#     try:
#         clean = repo_url.rstrip("/")
#         parts = clean.split("/")
#         owner, repo = parts[-2], parts[-1]
        
#         # A. Get File Structure
#         api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
#         resp = requests.get(api_url)
#         if resp.status_code == 403: return {"error": "GitHub API Rate Limit Exceeded. Try again in 1 hour."}
#         if resp.status_code != 200: return {"error": "Repo not found"}
        
#         data = resp.json()
#         files = [i['name'] for i in data]
        
#         # B. Fetch README Content (The Upgrade)
#         readme_text = "No README found."
#         if "README.md" in files:
#             readme_url = f"https://api.github.com/repos/{owner}/{repo}/contents/README.md"
#             r_resp = requests.get(readme_url)
#             if r_resp.status_code == 200:
#                 # GitHub sends content as Base64
#                 content = r_resp.json()['content']
#                 decoded = base64.b64decode(content).decode('utf-8', errors='ignore')
#                 readme_text = decoded[:4000] # Limit to 4000 chars to save context
        
#         return {
#             "structure": files,
#             "readme_snippet": readme_text
#         }
#     except Exception as e: return {"error": str(e)}

# def read_pdf_content(pdf_path: str) -> str:
#     """Reads text from a local PDF."""
#     try:
#         reader = PdfReader(pdf_path)
#         text = ""
#         for i, page in enumerate(reader.pages):
#             if i >= 5: break
#             text += page.extract_text() + "\n"
#         return text
#     except Exception as e: return f"Error: {str(e)}"

# # --- 3. AGENT CLASS ---
# class EnterpriseAgent:
#     def __init__(self, name, instructions, tools=None, enable_code_execution=False):
#         self.name = name
#         self.instructions = instructions
#         self.session_service = InMemorySessionService() 
        
#         # Base Tools (Custom)
#         self.active_tools = tools if tools else []
        
#         # ğŸ› ï¸� FEATURE: Built-in Code Execution (Official Tool)
#         # We pass 'code_execution' as a string which the SDK maps to the official sandbox
#         if enable_code_execution:
#             self.active_tools.append("code_execution")
        
#         # CONFIG: Gemini 2.5 Flash
#         self.model = genai.GenerativeModel(
#             model_name="gemini-2.5-flash", 
#             tools=self.active_tools, 
#             system_instruction=instructions
#         )
        
#         generate_agent_card(name, ["reasoning", "tool-use"])

#     def run(self, session_id, prompt, inject_artifact=None):
#         # Context Engineering
#         if inject_artifact:
#             logger.info(f"ğŸ’‰ Injecting Context into {self.name}...")
#             dynamic_system_prompt = f"""
#             {self.instructions}
#             [CRITICAL CONTEXT]
#             {json.dumps(inject_artifact, indent=2)}
#             """
#             self.model._system_instruction = dynamic_system_prompt

#         history = self.session_service.get_history(session_id)
#         # Note: Code Execution requires function calling enabled
#         chat = self.model.start_chat(history=history, enable_automatic_function_calling=True)
        
#         logger.info(f"ğŸ¤– {self.name} executing...")
#         try:
#             response = chat.send_message(prompt)
#             self.session_service.add_turn(session_id, "user", prompt)
            
#             # Handle Text Response
#             if response.parts and response.parts[0].text:
#                 self.session_service.add_turn(session_id, "model", response.text)
#                 return response.text
            
#             # Handle Code Execution Response (Executable Code)
#             if response.parts and response.parts[0].executable_code:
#                 return "Code Executed Successfully inside Sandbox."
                
#             return "Task Completed."
                
#         except Exception as e:
#             return f"Execution Error: {e}"

# print("âœ… Enterprise Agent Class Ready (Wikipedia + Code Execution)")


# # Cell 4: Run the Robust Pipeline (Fixed for Context Passing)
# from IPython.display import Markdown

# # 1. Setup Services
# global_session_service = InMemorySessionService()
# session_id = global_session_service.create_session(user_id="kaggle_judge")
# generate_mcp_config()

# # 2. Define Agents
# bg_checker = EnterpriseAgent(
#     name="Background_Checker",
#     instructions="You are a Librarian. Verify paper credibility.",
#     tools=[wiki_search]
# )
# bg_checker.session_service = global_session_service 

# paper_analyst = EnterpriseAgent(
#     name="Paper_Analyst",
#     instructions="Extract 'Ideal Setup' from PDF. Output the raw text of what you found.",
#     tools=[read_pdf_content] 
# )
# paper_analyst.session_service = global_session_service

# code_auditor = EnterpriseAgent(
#     name="Code_Auditor",
#     instructions="Verify code files. Output a list of MISSING files and MATCHING files.",
#     tools=[check_github_files]
# )
# code_auditor.session_service = global_session_service

# writer = EnterpriseAgent(
#     name="Report_Writer", 
#     instructions="You are a Judge. Use the provided Context to write a final score and report.",
#     enable_code_execution=True 
# )
# writer.session_service = global_session_service

# # --- PIPELINE ---
# pdf_path = "/kaggle/input/2017attention/NIPS-2017-attention-is-all-you-need-Paper.pdf"
# repo_url = "https://github.com/tensorflow/tensor2tensor"

# print(f"\nğŸš€ Starting Robust Session: {session_id}\n")

# # PHASE 1: Background Check
# print("ğŸŒ� Phase 1: Search Verification")
# bg_info = bg_checker.run(session_id, f"Search for 'Attention Is All You Need' paper.")
# print(f"   > Background: {bg_info[:100]}...\n")

# # PHASE 2: Paper Analysis
# print("ğŸ“� Phase 2: Analyzing PDF")
# # We explicitly ask for a detailed summary to pass forward
# paper_findings = paper_analyst.run(session_id, f"Read {pdf_path}. List the Dataset, Model, and Hyperparameters found.")

# # Create Artifact (Context Engineering)
# research_artifact = {
#     "source": "Attention Is All You Need",
#     "findings": paper_findings[:1000] # Capture the text!
# }
# global_session_service.save_artifact(session_id, "research_specs", research_artifact)
# print(f"   > Paper Findings Captured.\n")

# # PHASE 3: Code Audit
# print("ğŸ’» Phase 3: Code Audit")
# audit_findings = code_auditor.run(
#     session_id, 
#     f"Check this repo: {repo_url}. Compare against these findings: {paper_findings}",
#     inject_artifact=research_artifact 
# )
# print(f"   > Audit Findings Captured.\n")

# # PHASE 4: Final Report (CRITICAL FIX HERE)
# print("âš–ï¸� Phase 4: Final Report")

# # We construct a Mega-Prompt containing all previous evidence
# final_prompt = f"""
# GENERATE FINAL REPRODUCIBILITY REPORT.

# [EVIDENCE 1: BACKGROUND]
# {bg_info}

# [EVIDENCE 2: PAPER SPECS]
# {paper_findings}

# [EVIDENCE 3: CODE AUDIT RESULTS]
# {audit_findings}

# TASK:
# 1. Assign a Reproducibility Score (1-4).
# 2. List 3 strengths and 3 weaknesses.
# 3. Use Python to calculate the score if needed.
# """

# final_report = writer.run(session_id, final_prompt)

# display(Markdown(f"# ğŸ“Š Robust Reproducibility Report\n\n{final_report}"))




