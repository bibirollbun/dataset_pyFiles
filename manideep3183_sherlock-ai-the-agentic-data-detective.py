# Cell 1: Installation & Imports
!pip install -q -U google-generativeai

import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import traceback

# Authentication
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("âœ… API Key Loaded Successfully. The game is afoot!")
except Exception as e:
    print("â�Œ Error: Please ensure 'GOOGLE_API_KEY' is added in Add-ons -> Secrets.")


# Cell 2: Load Data
# We check for the standard Kaggle path, or fallback to a URL
try:
    df = pd.read_csv("/kaggle/input/titanic/train.csv")
    print("ğŸ“‚ Evidence loaded from Kaggle directory.")
except:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    print("ğŸŒ� Evidence loaded from External Source.")

print(f"ğŸ“Š DATASET STATUS: {df.shape[0]} Subjects | {df.shape[1]} Attributes")
df.head(3)


# Cell 3: The Mind Palace Tools

# Global list to store insights
mind_palace_storage = []

def add_clue_to_mind_palace(clue_text):
    """
    Stores a key finding or statistical fact into the agent's long-term memory.
    Use this when you discover something important (e.g., 'Females had a 74% survival rate').
    """
    mind_palace_storage.append(clue_text)
    return f"ğŸ§  Clue stored in Mind Palace: '{clue_text}'"

def retrieve_mind_palace():
    """
    Retrieves all stored clues to help form a final conclusion.
    """
    if not mind_palace_storage:
        return "The Mind Palace is empty. We need to analyze more data first."
    
    formatted_clues = "\n".join([f"- {clue}" for clue in mind_palace_storage])
    return f"ğŸ“‚ MIND PALACE CONTENTS:\n{formatted_clues}"

print("âœ… Mind Palace System Initialized.")


# Cell 4: The Interrogation Room Tool

def inspect_and_clean(column_name, action):
    """
    Inspects a column for missing values and cleans it based on user action.
    
    Args:
        column_name (str): The column to check (e.g., 'Age').
        action (str): 
            - 'report': Just count missing values.
            - 'impute_median': Fill NaN with the median value.
            - 'drop_rows': Delete rows with missing values.
    """
    if column_name not in df.columns:
        return f"â�Œ Error: Column '{column_name}' does not exist in the evidence."
    
    missing_count = df[column_name].isnull().sum()
    
    if action == "report":
        if missing_count == 0:
            return f"âœ… The column '{column_name}' is pristine. No missing values."
        else:
            return f"âš ï¸� SUSPICIOUS ACTIVITY: Found {missing_count} missing entries in '{column_name}'."
            
    elif action == "impute_median":
        if df[column_name].dtype not in ['float64', 'int64']:
            return "â�Œ Cannot calculate median on non-numeric data."
        median_val = df[column_name].median()
        df[column_name] = df[column_name].fillna(median_val)
        return f"ğŸ”§ Fixed: Imputed {missing_count} missing values with the median ({median_val})."
        
    elif action == "drop_rows":
        initial_len = len(df)
        df.dropna(subset=[column_name], inplace=True)
        dropped = initial_len - len(df)
        return f"âœ‚ï¸� Ruthless. Dropped {dropped} rows containing missing '{column_name}' data."

    return "â�Œ Unknown action."

print("âœ… Interrogation Room Ready.")


# Cell 5: Python Execution Tool

def execute_python_code(code_string):
    """
    Allows the agent to run Python code to analyze the dataframe `df`.
    Ideal for plotting graphs or calculating complex stats.
    """
    # Capture standard output to return to the agent
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    try:
        # Pre-import handy libraries for the agent
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Execute the code
        exec(code_string, globals()) 
        output = redirected_output.getvalue()
        
        if not output:
            return "âœ… Code executed successfully (Graphs should be visible above)."
        return output
    except Exception as e:
        return f"âš ï¸� ERROR in calculation: {str(e)}"
    finally:
        sys.stdout = old_stdout

print("âœ… Magnifying Glass (Code Executor) Ready.")


# Cell 6: System Instruction & Model Setup

sherlock_instruction = """
You are Sherlock Holmes, the world's first Agentic Data Detective.
Your Goal: Solve the mystery hidden in the `df` dataframe.

**YOUR TOOLKIT:**
1. `inspect_and_clean(column, action)`: Use this FIRST to check for missing data ('report'). If found, ask the user what to do. Then use 'impute_median' or 'drop_rows'.
2. `execute_python_code(code)`: Use this to plot graphs (matplotlib/seaborn) or calculate stats.
3. `add_clue_to_mind_palace(text)`: CRITICAL. Whenever you find a survival rate or correlation, save it here.
4. `retrieve_mind_palace()`: Use this before your final verdict to review evidence.

**PROTOCOL:**
1. **Voice:** Speak in 19th-century English ("Elementary!", "The game is afoot!").
2. **Process:** - Step A: Interrogate the data (Check for missing values).
   - Step B: Analyze relationships (Survival vs Class/Age/Sex).
   - Step C: Store findings in the Mind Palace.
   - Step D: Conclude.
3. **Visuals:** When asking to plot, assume `plt.show()` works.

**INTERACTION RULE:**
If you find missing values using `inspect_and_clean`, DO NOT fix them immediately. Report it to the user and ask: "Shall we impute or drop?"
"""

# Register all tools
tools_registry = [inspect_and_clean, execute_python_code, add_clue_to_mind_palace, retrieve_mind_palace]

# Initialize Model - USING YOUR AVAILABLE MODEL
# We selected 'gemini-2.0-flash' from your list
model = genai.GenerativeModel(
    'gemini-2.0-flash', 
    tools=tools_registry,
    system_instruction=sherlock_instruction
)

chat = model.start_chat(enable_automatic_function_calling=True)
print("ğŸ•µï¸�â€�â™‚ï¸� SHERLOCK IS ONLINE (Running on Gemini 2.0 Flash).")


# Cell 7: Rate-Limit Safe Chat
import time

def start_sherlock_session():
    print("=" * 60)
    print("ğŸ•µï¸�â€�â™‚ï¸� SHERLOCK HOLMES IS LISTENING...")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nğŸ‘¤ YOU: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            print("..." * 2)
            
            # --- THE FIX: WAIT 4 SECONDS BEFORE SENDING ---
            time.sleep(4) 
            
            response = chat.send_message(user_input)
            print(f"ğŸ•µï¸�â€�â™‚ï¸� SHERLOCK: {response.text}")
            
        except Exception as e:
            print(f"â�Œ Error: {e}")
            print("âš ï¸� You hit the rate limit. Waiting 20 seconds...")
            time.sleep(20) # Auto-wait if we hit an error

start_sherlock_session()

