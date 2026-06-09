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


# Install the Google Generative AI SDK
!pip install -q -U google-generativeai


import google.generativeai as genai
import os
import time
from IPython.display import Markdown, display, HTML


#  CONFIGURATION (SECURE MODE) ---

import google.generativeai as genai
import os
from kaggle_secrets import UserSecretsClient

# 1. Connect to Kaggle's Secret Vault
user_secrets = UserSecretsClient()

# 2. Retrieve the key 
# IMPORTANT: Make sure your Secret Label in Add-ons is exactly "GEMINI_API_KEY"
try:
    my_secret_key = user_secrets.get_secret("GEMINI_API_KEY")
except Exception as e:
    print("â�Œ ERROR: Could not find the secret. Did you name it 'GEMINI_API_KEY' inside Add-ons -> Secrets?")
    raise e

# 3. Configure Gemini
os.environ["GOOGLE_API_KEY"] = my_secret_key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("âœ… API Configured Successfully using Kaggle Secrets!")


import google.generativeai as genai
import os
import time
from IPython.display import Markdown, display, HTML
from kaggle_secrets import UserSecretsClient

# --- PART 1: SETUP & AUTO-DETECT MODEL ---
print("âš™ï¸� Configuring API...")
try:
    # 1. Setup Key
    user_secrets = UserSecretsClient()
    my_secret_key = user_secrets.get_secret("GEMINI_API_KEY")
    os.environ["GOOGLE_API_KEY"] = my_secret_key
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    # 2. AUTO-DETECT THE CORRECT MODEL NAME
    print("ğŸ”� Searching for available models...")
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            
    # Priority logic: Try Flash -> Pro -> Standard
    model_name = None
    for m in available_models:
        if "flash" in m: # Prefer 1.5 Flash (Fastest)
            model_name = m
            break
    if not model_name:
        for m in available_models:
            if "gemini-pro" in m: # Fallback to Pro
                model_name = m
                break
    
    # Final Fallback
    if not model_name and available_models:
        model_name = available_models[0]
        
    print(f"âœ… Found and using model: {model_name}")

except Exception as e:
    print(f"â�Œ Setup Error: {e}")
    print("Ensure 'GEMINI_API_KEY' is in Add-ons > Secrets.")
    # Fallback to a hardcoded string if list_models fails (rare)
    model_name = "models/gemini-1.5-flash-001" 

# --- PART 2: DEFINE AGENTS (Using the detected model) ---

def scanner_agent(contract_text):
    """ Agent 1: Scans for Red Flags. """
    try:
        model = genai.GenerativeModel(model_name) # <--- Uses the auto-detected name
        prompt = f"""
        You are a Senior Legal Risk Analyst. 
        DOCUMENT TEXT: {contract_text[:20000]} 
        TASK: Identify top 3 dangerous/predatory clauses (Binding Arbitration, Data Selling, Hidden Fees).
        OUTPUT: Numbered list with quotes.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in Scanner Agent: {str(e)}"

def translator_agent(scanner_output):
    """ Agent 2: Translates to Plain English. """
    try:
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        You are a 'Plain English' Translator.
        INPUT: {scanner_output}
        TASK: Rewrite into simple, brutal terms. Start each point with âš ï¸�.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in Translator: {str(e)}"

def negotiator_agent(translator_output):
    """ Agent 3: Drafts the email. """
    try:
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        You are a Consumer Rights Advocate.
        RISKS: {translator_output}
        TASK: Draft a formal opt-out email. Placeholders like [Date].
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in Negotiator: {str(e)}"

# --- PART 3: MAIN APP ORCHESTRATOR ---

def run_fine_print_assassin():
    display(Markdown("# ğŸ•µï¸�â€�â™€ï¸� **The Fine Print Assassin**"))
    display(Markdown(f"*(Powered by {model_name})*"))
    display(Markdown("---"))
    
    sample_tos = """
    12. BINDING ARBITRATION. YOU AGREE THAT ANY DISPUTES SHALL BE RESOLVED BY BINDING ARBITRATION. YOU WAIVE YOUR RIGHT TO A TRIAL BY JURY.
    15. DATA USAGE. WE RESERVE THE RIGHT TO SELL YOUR BIOMETRIC DATA TO THIRD-PARTY ADVERTISERS WITHOUT CONSENT.
    22. INACTIVITY FEE. IF YOUR ACCOUNT REMAINS INACTIVE FOR 30 DAYS, WE CHARGE A $50 MAINTENANCE FEE.
    """
    
    print("ğŸ‘‡ ENTER CONTRACT TEXT BELOW (Or press Enter to use Demo Text):")
    try:
        user_input = input("Contract Text: ")
    except:
        user_input = ""

    if len(user_input) < 10:
        contract_text = sample_tos
        display(Markdown("> *â„¹ï¸� Input empty. Using **Sample Predatory Contract** for demo...*"))
    else:
        contract_text = user_input

    # --- EXECUTION ---
    print("\nğŸ¦… Agent 1 (Scanner) is working...")
    risks = scanner_agent(contract_text)
    display(Markdown(f"**Found Risks:**\n{risks}"))
    time.sleep(1)
    
    print("\nğŸ—£ï¸� Agent 2 (Translator) is working...")
    simple = translator_agent(risks)
    display(Markdown(f"**Plain English:**\n{simple}"))
    time.sleep(1)
    
    print("\nğŸ›¡ï¸� Agent 3 (Negotiator) is working...")
    email_draft = negotiator_agent(simple)
    
    email_html = f"""
    <div style="background-color: #f0f4f8; padding: 20px; border-left: 5px solid #007bff; font-family: monospace;">
    {email_draft.replace(chr(10), '<br>')}
    </div>
    """
    display(HTML(email_html))

# --- PART 4: RUN IT ---
print("ğŸŸ¢ SYSTEM READY. STARTING NOW...")
run_fine_print_assassin()


# --- MAIN APPLICATION ORCHESTRATOR ---

def run_fine_print_assassin():
    display(Markdown("# ğŸ•µï¸�â€�â™€ï¸� **The Fine Print Assassin**"))
    display(Markdown("### *Don't sign it until we read it.*"))
    display(Markdown("---"))
    
    # Sample text for demonstration purposes (if user enters nothing)
    sample_tos = """
    12. BINDING ARBITRATION. YOU AGREE THAT ANY DISPUTES ARISING FROM THIS AGREEMENT SHALL BE RESOLVED BY BINDING ARBITRATION IN THE STATE OF DELAWARE, USA. YOU HEREBY WAIVE YOUR RIGHT TO A TRIAL BY JURY.
    15. DATA USAGE. WE RESERVE THE RIGHT TO SELL, LICENSE, OR DISTRIBUTE YOUR UPLOADED CONTENT AND BIOMETRIC DATA TO THIRD-PARTY ADVERTISING PARTNERS WITHOUT FURTHER CONSENT OR COMPENSATION.
    22. INACTIVITY FEE. IF YOUR ACCOUNT REMAINS INACTIVE FOR 30 DAYS, WE AUTOMATICALLY CHARGE A $50 MAINTENANCE FEE TO YOUR FILED CREDIT CARD.
    """
    
    print("ğŸ‘‡ Paste your Terms of Service / Contract below (Press Enter to use a Demo Contract):")
    user_input = input("Contract Text: ")
    
    if len(user_input) < 10:
        contract_text = sample_tos
        display(Markdown("> *â„¹ï¸� User input empty. Using **Sample Predatory Contract** for demonstration...*"))
    else:
        contract_text = user_input

    # --- STAGE 1: SCANNING ---
    display(Markdown("---"))
    display(Markdown("### ğŸ¦… **Phase 1: The Scanner**"))
    display(Markdown("*Scanning document for legal traps...*"))
    
    risks = scanner_agent(contract_text)
    display(Markdown(f"**Found these Critical Risks:**\n{risks}"))
    time.sleep(1) # Pause for dramatic effect
    
    # --- STAGE 2: TRANSLATION ---
    display(Markdown("---"))
    display(Markdown("### ğŸ—£ï¸� **Phase 2: The Translator**"))
    display(Markdown("*Translating to Plain English...*"))
    
    simple = translator_agent(risks)
    display(Markdown(f"**What this actually means:**\n{simple}"))
    time.sleep(1)
    
    # --- STAGE 3: ACTION ---
    display(Markdown("---"))
    display(Markdown("### ğŸ›¡ï¸� **Phase 3: The Fighter**"))
    display(Markdown("*Drafting legal defense...*"))
    
    email_draft = negotiator_agent(simple)
    
    # Pretty print the email
    email_html = f"""
    <div style="background-color: #f0f4f8; padding: 20px; border-left: 5px solid #007bff; font-family: monospace;">
    {email_draft.replace(chr(10), '<br>')}
    </div>
    """
    display(HTML(email_html))

# Execute the App
if __name__ == "__main__":
    run_fine_print_assassin()

