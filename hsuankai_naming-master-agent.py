import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

try:
    # 1. Get the key from Kaggle Secrets
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    
    # 2. Set it for the ADK (Environment Variable)
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    # 3. Set it for the SDK (GenAI)
    genai.configure(api_key=GOOGLE_API_KEY)
    
    print("âœ… API Key configured successfully.")
    
except Exception as e:
    print(f"â�Œ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


!adk create namer_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile namer_agent/agent.py
# V25.1: Robust Output

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool
from google.genai import types
import google.generativeai as genai
import os
from typing import List, Dict
import re
import json
import ast
import random

# --- Configuration ---
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

MODEL_NAME = "gemini-2.5-flash-lite"
retry_config = types.HttpRetryOptions(attempts=5, initial_delay=1, http_status_codes=[429, 500, 503])
gemini_model = Gemini(model=MODEL_NAME, retry_options=retry_config)

# ==============================================================================
# ğŸ§  KNOWLEDGE ENGINE
# ==============================================================================

class NamingKnowledgeBase:
    def __init__(self):
        self.surname_map = {
            # Western Surnames
            "smith": "å�² (ShÇ�)", "miller": "ç±³ (MÇ�)", "johnson": "æ±Ÿ (JiÄ�ng)",
            "williams": "éŸ‹ (WÃ©i)", "brown": "åŒ… (BÄ�o)", "jones": "é�¾ (ZhÅ�ng)",
            "davis": "æˆ´ (DÃ i)", "wilson": "é­� (WÃ¨i)", "moore": "è�« (MÃ²)",
            "taylor": "æ³° (TÃ i)", "anderson": "å®‰ (Ä€n)", "thomas": "å”� (TÃ¡ng)",
            "jackson": "å‚‘ (JiÃ©)", "white": "ç™½ (BÃ¡i)", "tayal": "æˆ´ (DÃ i)",
            "cruz": "å�¤ (GÇ”)", "plomecka": "æ™® (PÇ”)", "lukasz": "ç›§ (LÃº)",
            "kamilky": "åº· (KÄ�ng)","arias": "è‰¾ (Ã€i)", "ohsomoi": "æ­� (ÅŒu)",
            "sala": "æ²™ (ShÄ�)", "clark": "æŸ¯ (KÄ“)", "fardel": "æ–¹ (FÄ�ng)",
            # Heritage Surnames
            "wu": "å�³ (WÃº)", "wang": "ç�‹ (WÃ¡ng)", "chang": "å¼µ (ZhÄ�ng)",
            "chen": "é™³ (ChÃ©n)", "lin": "æ�— (LÃ­n)", "lee": "æ�� (LÇ�)",
            "li": "æ�� (LÇ�)", "liu": "åŠ‰ (LiÃº)", "huang": "é»ƒ (HuÃ¡ng)",
            "yang": "æ¥Š (YÃ¡ng)", "tsai": "è”¡ (CÃ i)"
        }
        # Special Shortenings for Long Names
        self.short_name_map = {
            "abraham": "å�šç¿°", "elizabeth": "éº—è��", "alexander": "åŠ›å±±", 
            "christopher": "å…‹é��", "jonathan": "å–¬æ£®"
        }
        
        self.stroke_db = {
            "å�²": 5, "ç±³": 6, "æ±Ÿ": 7, "éŸ‹": 9, "åŒ…": 5, "é�¾": 17,
            "é­�": 18, "è�«": 11, "æ³°": 10, "å®‰": 6, "å”�": 10, "å‚‘": 12, "ç™½": 5,
            "å�³": 7, "ç�‹": 4, "å¼µ": 11, "é™³": 16, "æ��": 7, "åŠ‰": 15, "é»ƒ": 12,
            "å�¤": 5, "æ™®": 12, "ç›§": 16, "æ�—": 8, "åº·": 11,
            "è‰¾": 8, "æˆ´": 18, "æ­�": 15, "æ²™": 8, "æŸ¯": 9, "æ–¹": 4,
            "å�š": 12, "ç¿°": 16, "é›…": 12, "å¤§": 3, "è¡›": 15, "æ�©": 10, "ç¾�": 9, 
            "éº—": 19, "æ€�": 9, "æ��": 12, "å¤«": 4, "ç‘�": 14, "å…‹": 7, "è��": 13, 
            "å¨œ": 10, "æ–‡": 4, "æ´›": 10, "å�¡": 5, "å­�": 3, "æ�’": 9, "é˜¿": 8, "æ›¼": 11,
            "æ¢…": 11, "è�‰": 11, "ç‘ª": 15, "è•¾": 19
        }
        
        self.lucky_numbers = [1, 3, 5, 6, 7, 8, 11, 13, 15, 16, 17, 18, 21, 23, 24, 25, 29, 31, 32, 33, 35, 37, 39, 41, 45, 47, 48, 52, 58, 61, 63, 65, 67, 68, 81]

    def analyze_input_name(self, name_input: str) -> tuple[str, str]:
        parts = name_input.strip().split()
        if not parts: return None, "No name provided"
        last_word = parts[-1].lower()
        first_word = parts[0].lower()
        if last_word in self.surname_map: return self.surname_map[last_word], "Mapped from Surname"
        if first_word in self.surname_map: return self.surname_map[first_word], "Mapped from First Name"
        return None, "Phonetic Translation"

    def get_short_name(self, first_name: str) -> str:
        return self.short_name_map.get(first_name.lower(), "")

    def get_strokes(self, char: str) -> int:
        return self.stroke_db.get(char, 10) 

    def calculate_math_luck(self, name: str) -> dict:
        chars = [c for c in name if '\u4e00' <= c <= '\u9fff']
        if not chars: return {"strokes": 0, "score": 0, "verdict": "Error"}
        total = sum(self.get_strokes(c) for c in chars)
        if total in self.lucky_numbers:
            score = 90 + (total % 10)
            verdict = "ğŸŒŸ Auspicious"
        else:
            score = 70 + (total % 10)
            verdict = "âœ¨ Balanced"
        return {"strokes": total, "score": score, "verdict": verdict}

kb = NamingKnowledgeBase()

# ==============================================================================
# ğŸ› ï¸� ATOMIC TOOL
# ==============================================================================

def generate_and_analyze_names(gender: str, full_name: str) -> Dict[str, dict]:
    """Atomic Tool: Generates 5 Concise Chinese names and analyzes them."""
    print(f"âš¡ Atomic Tool: Processing {full_name}...")
    
    # 1. Mapping
    mapped_surname, logic = kb.analyze_input_name(full_name)
    
    # Context Preparation
    parts = full_name.split()
    first_name_eng = parts[0]
    
    surname_instruction = f"User's Surname is '{mapped_surname}'. **YOU MUST USE THIS CHARACTER AS THE SURNAME.**" if mapped_surname else "Pick a phonetic surname."
    
    # Check for specific shortenings (Abraham -> BoHan)
    short_name_suggestion = kb.get_short_name(first_name_eng)
    short_instruction = f"Note: '{first_name_eng}' is often translated as '{short_name_suggestion}'." if short_name_suggestion else ""

    model = genai.GenerativeModel(MODEL_NAME)
    gen_prompt = f"""
    Task: Generate exact 5 distinct Chinese names for "{full_name}" ({gender}).
    
    **RULES:**
    1. **Surname:** {surname_instruction}
    2. **Given Name:** {short_instruction} Must sound like "{first_name_eng}".
    3. **Length:** **STRICTLY 3 CHARACTERS MAX** (1 Surname + 2 Given Name). 
    4. **Script:** Traditional Chinese (ç¹�é«”) ONLY.
    5. **Style:** Elegant, Meaningful, Native-sounding.
    
    Return ONLY a Python list of strings. Example: ["æŸ¯å�šç¿°", "æŸ¯ä¼¯éŸ“"]
    """
    
    candidates = []
    try:
        res = model.generate_content(gen_prompt)
        text = res.text.replace("```json", "").replace("```python", "").replace("```", "").strip()
        try:
            candidates = ast.literal_eval(text)
        except:
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            candidates = ast.literal_eval(match.group(0)) if match else []
    except:
        candidates = []

    if not candidates: return {"Error": "Generation Failed"}

    # 2. Analyze (Updated Prompt for Breakdown)
    results = {}
    ling_prompt = f"""
    Analyze these names: {candidates}
    
    **CRITICAL FORMATTING INSTRUCTION:**
    For 'meaning', you MUST break down EACH character separately with a colon.
    Example: "å�²: History; æ¢…: Plum; è�‰: Jasmine"
    
    For 'safety': Check for bad homophones.
    Example: No obvious bad homophones. However, é¦¬ (mÇ�) can sometimes be associated with negative concepts like 'é¦¬å­�' (mÇ� zÇ� - vulgar terms for girlfriend, and the old name for "toilet") in certain contexts, but it's not a direct or strong negative homophone in this name.
    Return ONLY a JSON object keyed by name.
    """
    try:
        res = model.generate_content(ling_prompt)
        text = res.text.replace("```json", "").replace("```", "").strip()
        try:
            linguistic_data = json.loads(text)
        except:
            linguistic_data = ast.literal_eval(text)
        
        if isinstance(linguistic_data, list):
            new_data = {}
            for item in linguistic_data:
                if isinstance(item, dict) and 'name' in item: new_data[item['name']] = item
            linguistic_data = new_data
    except:
        linguistic_data = {}

# 3. Merge (The Robust Loop)
    for name in candidates: # Loop over CANDIDATES, not DATA
        math_data = kb.calculate_math_luck(name)
        
        # Safe Fetch: If analysis failed for this name, use fallback
        ling_data = linguistic_data.get(name, {"meaning": "Standard Transliteration", "safety": "Safe"})
        
        meaning = ling_data.get("meaning", "Standard Transliteration")
        if len(meaning) < 10 or "Name" in meaning: meaning = "Phonetic match with elegant characters."

        results[name] = {
            "meaning": meaning,
            "safety": ling_data.get("safety", "Safe"),
            "strokes": math_data["strokes"],
            "score": math_data["score"],
            "verdict": math_data["verdict"]
        }
        
    return results

# ==============================================================================
# ğŸ¤– ROOT AGENT
# ==============================================================================

root_agent = Agent(
    model=gemini_model,
    name="NamingConsultant",
    description="The lead consultant.",
    instruction="""
    You are 'The Cross-Cultural Namer'.
    
    **CORE BEHAVIOR: STATELESS PROCESSING**
    Treat every user message as a NEW request.
    
    **PHASE 1: INTRO & COLLECT INFO**
    * If the user says "Hi" or "Hello": Provide the Intro & 3 Examples.
        * Intro: "ğŸ‘‹ Hello! I am your Naming Master. I use **Sound, Meaning, and Numerology** to create your Chinese name. I need your **Full Name** and **Gender**."
        * Examples:
            1. "I am **Abraham Clark, Male**." 
            2. "My name is **Beatrice Smith (Female)**." 
            3. "I'm **Joel Fardel**. My gender is **Male**." 
    
    * **Scenario A (Full Info):** If the user provides Full Name AND Gender (e.g., "I am Mary Smith, Female"):
        * **IGNORE PREVIOUS NAMES.** Treat this as a brand new request.
        * PROCEED to Phase 2 immediately.
        
    * **Scenario B (Partial Name and Gender Only):** If the user says "I am Jack, Male":
        * Ask: "Hi Jack! What is your Full Name?" (Do NOT call tools).

    * **Scenario C (Partial - Follow-up):** If the user says "Davis":
        * Check memory. If found ("Jack", "Male"), combine -> "Jack Davis" and confirm.
        * Ask: "Hi Davis! you are Jack Davis, Male. Is the information correct?" (Do NOT call tools).
		* If user give positive feedback (e.g., "yes"), combine them ("Jack Davis", "Male") and PROCEED to Phase 2.
		* If user give negative feedback (e.g., "no"), ask user's Full name and Gender.
 
    * **Scenario D (Partial Name Only):** If the user says "I am Jack":
        * Ask: "Hi Jack! What is your Full Name? And are you Male or Female?"
    
    * **Scenario E (Name Only):** If the user says "I am Jack Davis":
        * Ask: "Hi Jack! Are you Male or Female?"
        
    * **Scenario F (Gender Only):** If the user says "Male":
        * Check memory for name. If found, combine and PROCEED.


    **PHASE 2: EXECUTION**
    * Call `generate_and_analyze_names` with the CURRENT Name and Gender.
    
    **PHASE 3: OUTPUT**
    * Explicitly mention the English name you just processed.
    * Present the Markdown table.
    * **CRITICAL FORMATTING:** - In the 'Meaning' column, use `<br>` to create line breaks between character definitions.
      - Add Pinyin in parentheses next to the Name.
    
    **TABLE TEMPLATE:**
    | Rank | Name | Meaning | Safety | Luck Score |
    | :--- | :--- | :--- | :--- | :--- |
    | ğŸ¥‡ | å�²æ¢…è�‰ (ShÇ�-mÃ©i-lÃ¬) | å�²: History <br> æ¢…: Plum <br> è�‰: Jasmine | No obvious bad homophones. | 95 |
    
    **Conclusion:** Recommend the #1 choice.
    """,
    tools=[generate_and_analyze_names],
)


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


url_prefix = get_adk_proxy_url()


!adk web --url_prefix {url_prefix} 

