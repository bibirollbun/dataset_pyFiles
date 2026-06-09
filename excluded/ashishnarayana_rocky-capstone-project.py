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


# --- INSTALLATION & SETUP ---
!pip install -q -U google-generativeai pandas

import google.generativeai as genai
import pandas as pd
from kaggle_secrets import UserSecretsClient

# 1. Retrieve the key from Kaggle Secrets
user_secrets = UserSecretsClient()
my_secret_key = user_secrets.get_secret("GOOGLE_API_KEY")

# 2. Authenticate Gemini
genai.configure(api_key=my_secret_key)

print("âœ… Setup Complete. Gemini is connected and ready!")


# --- STEP 1: GENERATE DUMMY DATA ---
def create_dummy_data():
    data = {
        'Lead_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Name': ['ARES Corp', 'BENLY Ltd', 'Gamma Inc', 'Delta Co', 'Epsilon LLC', 'ZOHO Group', 'Eta Sol', 'Theta Biz', 'INTRO Soft', 'Kappa Tech'],
        'Industry': ['Tech', 'Retail', 'Finance', 'Tech', 'Retail', 'Tech', 'Finance', 'Healthcare', 'Tech', 'Retail'],
        'Company_Size': [100, 200, 500, 40, 150, 1000, 300, 70, 90, 30],
        'Lead_Source': ['Webinar', 'Cold Call', 'Referral', 'Webinar', 'Webinar', 'Referral', 'Cold Call', 'Referral', 'Webinar', 'Cold Call'],
        'Deal_Value_USD': [25000, 10000, 60000, 4000, 22000, 90000, 55000, 48000, 2500, 7000],
        'Status': ['Won', 'Lost', 'New', 'New', 'Won', 'New', 'Lost', 'Won', 'New', 'Lost']
    }
    df = pd.DataFrame(data)
    df.to_csv('crm_leads.csv', index=False)
    print("âœ… 'crm_leads.csv' generated with 10 rows of dummy data.")
    return df

# Create the file immediately
df = create_dummy_data()
display(df.head())


# --- STEP 2: DEFINE TOOLS ---

# This acts as our "Python REPL" tool.
# The agent will write code as a string, and this function executes it.
def execute_python_analysis(code_string: str):
    """
    Executes Python pandas code to analyze 'crm_leads.csv'.
    The code must assign the result to a variable named 'result'.
    """
    # Clean up the code string (remove markdown wrappers if the LLM adds them)
    code_string = code_string.strip().replace("```python", "").replace("```", "")
    
    local_scope = {}
    try:
        # We pre-load pandas and the csv for the agent
        exec(f"import pandas as pd\ndf = pd.read_csv('crm_leads.csv')\n{code_string}", {}, local_scope)
        return str(local_scope.get('result', "No 'result' variable found in code."))
    except Exception as e:
        return f"Error executing code: {str(e)}"

# Register the tool with Gemini
tools_list = [execute_python_analysis]

print("âœ… Analysis Tool Created.")


# --- STEP 3: DEFINE THE AGENTS (AUTO-DETECT MODEL) ---

# 1. FIND A VALID MODEL
# We will check what your API key actually has access to.
valid_model_name = ""
print("ğŸ”� Scanning for available models...")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   - Found: {m.name}")
            # We prefer Flash 1.5, then Pro 1.5, then Pro 1.0
            if "flash" in m.name and "1.5" in m.name:
                valid_model_name = m.name
            elif "pro" in m.name and "1.5" in m.name and valid_model_name == "":
                valid_model_name = m.name
            elif "gemini-pro" in m.name and valid_model_name == "":
                valid_model_name = m.name
except Exception as e:
    print(f"âš ï¸� Could not list models (API Error): {e}")

# If we found nothing or failed, fallback to a safe default
if not valid_model_name:
    valid_model_name = "gemini-1.5-flash"
    print("âš ï¸� Could not auto-detect. Defaulting to 'gemini-1.5-flash'")
else:
    # The SDK usually wants the name WITHOUT 'models/' prefix for the constructor,
    # but sometimes with it. We'll strip it just in case.
    if valid_model_name.startswith("models/"):
        valid_model_name = valid_model_name.replace("models/", "")
    print(f"\nâœ… SELECTED BEST MODEL: {valid_model_name}")


# 2. DEFINE SYSTEM INSTRUCTIONS
analyst_system_instruction = """
You are a Senior Data Analyst.
Your goal is to analyze the 'crm_leads.csv' file to find high-value patterns.
You have access to a tool 'execute_python_analysis'.
ALWAYS use this tool to calculate numbers. Do not guess.
Your final output should be a summary of:
1. Which Industry has the highest average Deal Value?
2. Which Lead Source has the highest Win Rate (Status='Won')?
3. A specific 'Scoring Rule'.
"""

sales_system_instruction = """
You are a Top-Tier Sales Representative.
Your goal is to draft a personalized cold outreach email template based on data findings.
The email should be professional, persuasive, and under 150 words.
"""

# 3. INITIALIZE AGENTS
try:
    model_analyst = genai.GenerativeModel(
        model_name=valid_model_name,
        tools=tools_list,
        system_instruction=analyst_system_instruction
    )

    model_sales = genai.GenerativeModel(
        model_name=valid_model_name,
        system_instruction=sales_system_instruction
    )
    print("âœ… Agents Successfully Initialized.")
except Exception as e:
    print(f"ğŸ”¥ CRITICAL ERROR initializing model: {e}")


# --- STEP 3.1: DEFINE THE AGENTS (FORCED FLASH) ---

# We explicitly choose the fast, free-tier friendly model from your list
model_name_stable = 'gemini-2.5-flash' 

print(f"âœ… USING MODEL: {model_name_stable}")

# 1. THE DATA ANALYST
analyst_system_instruction = """
You are a Senior Data Analyst.
Your goal is to analyze the 'crm_leads.csv' file to find high-value patterns.
You have access to a tool 'execute_python_analysis'.
ALWAYS use this tool to calculate numbers. Do not guess.
Your final output should be a summary of:
1. Which Industry has the highest average Deal Value?
2. Which Lead Source has the highest Win Rate (Status='Won')?
3. A specific 'Scoring Rule' (e.g. "Prioritize Tech companies over 50 employees").
"""

# 2. THE SALES STRATEGIST
sales_system_instruction = """
You are a Top-Tier Sales Representative.
You will receive an analysis of high-value leads.
Your goal is to draft a personalized cold outreach email template.
The email should:
1. Reference the specific data point (e.g. "We saw success with other Tech companies...").
2. Be professional but persuasive.
3. Keep it under 150 words.
"""

# Initialize the Gemini Models
try:
    model_analyst = genai.GenerativeModel(
        model_name=model_name_stable,
        tools=tools_list,
        system_instruction=analyst_system_instruction
    )

    model_sales = genai.GenerativeModel(
        model_name=model_name_stable,
        system_instruction=sales_system_instruction
    )
    print("âœ… Agents Initialized with Flash (High Speed/Free Quota).")
except Exception as e:
    print(f"âš ï¸� Error initializing agents: {e}")


# --- STEP 4: ORCHESTRATION WITH RETRY LOGIC (FINAL) ---
import time
import warnings

# Hide messy pandas warnings for a clean video demo
warnings.filterwarnings('ignore') 

def run_agency_robust():
    print(f"ğŸ¤– AGENT 1 (Analyst): Starting analysis...")
    
    analysis_prompt = (
        "Use your python tool to analyze 'crm_leads.csv'. "
        "Calculate the average Deal Value by Industry and Win Rate by Lead Source. "
        "Summarize the best opportunities."
    )
    
    analysis_result = None
    
    # --- PHASE 1: ANALYST WITH RETRY ---
    # We try up to 3 times to get the analysis
    for attempt in range(3):
        try:
            chat_analyst = model_analyst.start_chat(enable_automatic_function_calling=True)
            print(f"   ...Attempt {attempt+1}: Asking Analyst to crunch numbers...")
            
            response_analyst = chat_analyst.send_message(analysis_prompt)
            
            # If we get here, it worked!
            analysis_result = response_analyst.text
            print(f"\nğŸ“Š ANALYST FINDINGS:\n{'-'*20}\n{analysis_result}\n{'-'*20}")
            break # Exit the loop, we have success
            
        except Exception as e:
            if "429" in str(e):
                print(f"   âš ï¸� Quota Hit (429). Waiting 30 seconds before retrying...")
                time.sleep(30)
            else:
                print(f"   âš ï¸� Unexpected Error: {e}")
                break

    # If analysis failed after 3 tries, we stop.
    if not analysis_result:
        print("â�Œ CRITICAL: Analysis failed multiple times. Stopping.")
        return

    # --- PHASE 2: SALES AGENT ---
    # Smart wait: We pause briefly to ensure we don't hit the limit for the second agent
    print("\nâ�³ Cooldown: Waiting 10 seconds to ensure Sales Agent has quota...")
    time.sleep(10)
    
    print("\nğŸ¤– AGENT 2 (Sales): Drafting email...")
    
    prompt_for_sales = f"""
    Here are the latest market findings:
    {analysis_result}
    
    Task: Write a short, professional cold outreach email to a potential client in the best-performing industry found above.
    """
    
    try:
        response_sales = model_sales.generate_content(prompt_for_sales)
        print(f"\nâœ‰ï¸� FINAL DRAFT:\n{'-'*20}\n{response_sales.text}\n{'-'*20}")
    except Exception as e:
         print(f"âš ï¸� Sales Agent Error: {e}")

# EXECUTE
run_agency_robust()

