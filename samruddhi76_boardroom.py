# --- PART 1: SYSTEM SETUP & TOOLS ---
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import pandas as pd
from IPython.display import display, HTML, Markdown

print("âš™ï¸� SETTING UP THE BOARDROOM...")

# 1. AUTHENTICATE
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("âœ… API Connected Successfully.")
except:
    print("â�Œ Error: Check API Key.")

# 2. CREATE KNOWLEDGE BASE (The Memory)
pd.DataFrame({'Food': ['Oatmeal', 'Banana', 'Pizza', 'Salad', 'Coffee'], 
              'Calories': [150, 105, 285, 30, 5]}).to_csv('nutrition.csv', index=False)

pd.DataFrame({'Exercise': ['Running', 'Deep Work', 'Nap', 'Yoga'], 
              'Type': ['Cardio', 'Mental', 'Rest', 'Recovery']}).to_csv('exercises.csv', index=False)
print("âœ… Knowledge Base (CSV) Created.")

# 3. DEFINE TOOLS (The Skills)

def search_food(food_name: str):
    """Returns calorie info for a food."""
    df = pd.read_csv('nutrition.csv')
    res = df[df['Food'].str.contains(food_name, case=False, na=False)]
    return res.to_string() if not res.empty else "No data."

def get_exercises(type_filter: str):
    """Returns exercises by type."""
    df = pd.read_csv('exercises.csv')
    res = df[df['Type'].str.contains(type_filter, case=False, na=False)]
    return res.to_string() if not res.empty else "No data."

# --- NEW TOOL ADDED HERE ---
def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str):
    """Calculates Basal Metabolic Rate (Calories burned at rest)."""
    if gender.lower() == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    return f"Estimated BMR: {int(bmr)} calories/day (minimum needed for survival)."

# Update the list to include the 3rd tool
my_tools = [search_food, get_exercises, calculate_bmr]
print("âœ… Tools Ready: 'search_food', 'get_exercises', and 'calculate_bmr'.")



# --- PART 2: INITIALIZING THE AGENTS ---

# 1. AUTO-DETECT BEST MODEL
valid_model = None
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        if 'flash' in m.name: valid_model = m.name; break
        if not valid_model: valid_model = m.name 
print(f"ğŸ”¹ Using Gemini Model: {valid_model}")

# 2. DEFINE PERSONAS
marcus = genai.GenerativeModel(valid_model, system_instruction="You are MARCUS (CPO). Maximize productivity. Be stern. Hate breaks.")

# UPDATED SARAH INSTRUCTION:
sarah_sys = """
You are DR. SARAH (CMO). Check health. 
Use 'search_food' to check calories.
Use 'calculate_bmr' if the user provides weight/height/age to see their survival needs.
If no weight is given, assume average and use general medical knowledge.
"""
sarah = genai.GenerativeModel(valid_model, tools=my_tools, system_instruction=sarah_sys)

boss = genai.GenerativeModel(valid_model, system_instruction="You are THE CHAIRMAN. Listen to the debate. Output a Final Schedule as a Markdown Table.")

# 3. UI HELPER (Visuals)
def print_bubble(name, text, color, icon, align):
    html = f"""
    <div style="display:flex; justify-content:{align}; margin:10px;">
        <div style="background:{color}; color: black; padding:15px; border-radius:15px; max-width:70%; border:2px solid #ccc; box-shadow: 3px 3px 5px #ddd; font-family: Arial, sans-serif;">
            <b>{icon} {name}</b><br>
            <div style="margin-top:5px;">{text.replace(chr(10), '<br>')}</div>
        </div>
    </div>
    """
    display(HTML(html))

print("âœ… The Board of Directors is assembled.")


# --- PART 3: THE BOARD MEETING ---

def start_board_meeting(user_goal):
    print_bubble("System", f"ğŸ”” NEW USER GOAL: {user_goal}", "#eee", "âš™ï¸�", "center")
    
    # Round 1: Marcus (Productivity Bid)
    print("âš¡ Marcus is thinking...")
    plan = marcus.generate_content(f"Plan for: {user_goal}. Be aggressive.").text
    print_bubble("Marcus (CPO)", plan, "#e3f2fd", "âš¡", "left")
    
    # Round 2: Sarah (Health Audit - USES TOOLS)
    print("ğŸŒ¿ Sarah is checking database...")
    chat = sarah.start_chat(enable_automatic_function_calling=True)
    critique = chat.send_message(f"Critique this: {plan}. Check food calories and exercise types.").text
    print_bubble("Dr. Sarah (CMO)", critique, "#e8f5e9", "ğŸŒ¿", "right")
    
    # Round 3: Chairman (Final Decree)
    print("ğŸ�† Chairman is finalizing...")
    final = boss.generate_content(f"Plan: {plan}\nCritique: {critique}\nCreate Final Schedule.").text
    print_bubble("The Chairman", "Here is the final approved schedule.", "#fff3e0", "ğŸ�†", "center")
    display(Markdown(final))

# --- RUN THE DEMO ---
# We added weight/height so Sarah uses the BMR Tool!
start_board_meeting("I need to finish my coding project in 2 days, but I want to lose weight. I am 25 years old, male, 75kg, 180cm.")

