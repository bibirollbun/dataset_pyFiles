# --- BLOCK 1: INSTALL LIBRARIES ---
!pip install -q -U google-generativeai ddgs

# --- BLOCK 2: SETUP ---
import google.generativeai as genai
from duckduckgo_search import DDGS

# !!! PASTE YOUR REAL API KEY BELOW !!!
MY_API_KEY = "INSERT_YOUR_API_KEY_HERE"
genai.configure(api_key=MY_API_KEY)

# --- THE FIX: AUTO-DETECT THE RIGHT MODEL ---
print("ğŸ”� Asking Google for available models...")
valid_model = None

try:
    # Loop through all models available to your account
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"âœ… Found: {m.name}")
            valid_model = m.name
            # We prefer flash if available, otherwise keep looking or take the last one
            if 'flash' in m.name:
                break 
    
    if valid_model:
        print(f"ğŸš€ CONNECTED TO: {valid_model}")
        model = genai.GenerativeModel(valid_model)
    else:
        print("â�Œ No text generation models found for this key.")
        
except Exception as e:
    print(f"âš ï¸� specific model error: {e}")
    # Fallback just in case
    model = genai.GenerativeModel('gemini-1.5-flash')

# --- TOOL SETUP ---
def search_tool(topic):
    print(f"ğŸ•µï¸� [TOOL ACTION] Searching the web for: {topic}...")
    try:
        results = DDGS().text(topic, max_results=3)
        summary_text = ""
        if results:
            for r in results:
                summary_text += f"- {r['body']}\n"
        else:
            summary_text = "No results found."
        return summary_text
    except Exception as e:
        return f"Search Error: {e}"

def run_student_agent(user_query):
    print(f"ğŸ�“ [USER] Asking: {user_query}")
    raw_info = search_tool(user_query)
    
    final_prompt = f"""
    You are a helpful university tutor.
    The student asked: '{user_query}'
    Here is the information I found on the web:
    {raw_info}
    Please summarize this information into clear, bullet-point exam notes.
    """
    
    print("ğŸ¤– [AGENT] Reading results and writing summary...")
    try:
        response = model.generate_content(final_prompt)
        print("\n" + "="*40)
        print("ğŸ“� EXAM NOTES")
        print("="*40)
        print(response.text)
    except Exception as e:
        print(f"â�Œ Error: {e}")

# --- TEST IT ---
if valid_model:
    run_student_agent("What is the difference between TCP and UDP?")
else:
    print("Could not run agent because no model was found.")

