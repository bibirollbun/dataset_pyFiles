# ==========================================
# ğŸ§  CELL 1: SETUP & TEXT CHAT (FIXED)
# ==========================================
import os
import time
import datetime
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# --- 1. CONNECT TO THE STABLE BRAIN ---
print("ğŸ”Œ Connecting to AI...")
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    
    # PRIORITY LIST: Try Flash first (fast/stable), then Pro.
    priority_models = [
        "gemini-1.5-flash", 
        "gemini-1.5-pro", 
        "gemini-1.0-pro", 
        "gemini-pro"
    ]
    
    active_model = None
    available = [m.name for m in genai.list_models()]
    
    # Find the best match
    for target in priority_models:
        for m in available:
            if target in m:
                active_model = m
                break
        if active_model: break
    
    if not active_model:
        active_model = "gemini-pro" # Fallback
        
    model = genai.GenerativeModel(active_model)
    print(f"âœ… Connected to: {active_model} (Stable)")
    
except Exception as e:
    print(f"âš ï¸� Connection Error: {e}")

# --- 2. THE DATABASE (Memory) ---
# I have checked the quotes below carefully to fix your syntax error.
if 'med_db' not in globals():
    med_db = {
        "Monday": "Heart Pill (Blue) at 9 AM",
        "Tuesday": "Blood Pressure Pill (Red) at 8 PM",
        "Wednesday": "Vitamin D (White) at 10 AM",
        "Thursday": "Heart Pill (Blue) at 9 AM",
        "Friday": "Iron Supplement at 2 PM",
        "Saturday": "Multivitamin at 10 AM",
        "Sunday": "Rest Day - No Meds"
    }

# --- 3. INTELLIGENT AGENTS (With Retry Logic) ---
def ask_ai(prompt):
    """Sends message to AI. If it hits a limit, it waits 10s and tries again."""
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        # Check for Quota Error (429)
        if "429" in str(e) or "ResourceExhausted" in str(e):
            print("â�³ Quota limit hit. Cooling down for 10 seconds...")
            time.sleep(10)
            try:
                # Retry once
                return model.generate_content(prompt).text.strip()
            except:
                return "I am resting right now. Please ask me in a minute."
        return f"Error: {e}"

def get_response(user_input):
    """Decides if it's the Nurse or the Friend speaking"""
    # Medical Keywords
    med_words = ["pill", "medicine", "take", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "doctor", "tablet"]
    
    if any(x in user_input.lower() for x in med_words):
        context = f"Current Med Database: {med_db}. Today is {datetime.datetime.now().strftime('%A')}."
        prompt = f"Context: {context}. User asks: '{user_input}'. Answer strictly based on the database. Do not use Markdown (no ** or ##). Keep it clean."
        return f"ğŸ‘©â€�âš•ï¸� Nurse: {ask_ai(prompt)}"
    else:
        prompt = f"Role: Warm friend 'Elder Sync'. User says: '{user_input}'. Keep it short, kind, and plain text (no bold/markdown)."
        return f"ğŸ§£ Friend: {ask_ai(prompt)}"

# --- 4. TEXT CHAT LOOP ---
print("\n" + "="*40)
print("ğŸ’¬ TEXT CHAT ACTIVE (Silent Mode)")
print("Type 'voice' to stop and go to Cell 3.")
print("="*40)

while True:
    u_in = input("\nğŸ‘¤ YOU: ")
    if u_in.lower() in ["exit", "quit", "voice"]:
        print("ğŸ‘‹ Stopping Text Mode. Run Cell 3 for Voice.")
        break
    
    # Generate and print response
    print(get_response(u_in))


# ==========================================
# âš™ï¸� CELL 2: SETTINGS DASHBOARD
# ==========================================

def show_menu():
    print("\n" + "="*40)
    print("ğŸ�¥ ELDER SYNC SETTINGS")
    print("="*40)
    print("1. View Schedule")
    print("2. Add/Edit Medication")
    print("3. View Vitals Log")
    print("4. Clear All Logs")
    print("5. Exit to Chat")
    return input("\nEnter choice (1-5): ")

while True:
    selection = show_menu()
    
    if selection == '1':
        print("\nğŸ“‹ CURRENT SCHEDULE:")
        for day, pills in med_db.items():
            print(f"   - {day}: {pills}")
            
    elif selection == '2':
        print("\nğŸ’Š EDIT MEDICATION")
        day = input("Enter Day (e.g. Monday): ").capitalize()
        pill = input(f"Enter Schedule for {day}: ")
        med_db[day] = pill
        print(f"âœ… Saved: {day} -> {pill}")
        
    elif selection == '3':
        print("\nâ�¤ï¸� VITALS LOG:")
        print(vitals_log if vitals_log else "   (No records yet)")

    elif selection == '4':
        confirm = input("ğŸ—‘ï¸� Delete all chat/health logs? (y/n): ")
        if confirm.lower() == 'y':
            chat_history.clear()
            vitals_log.clear()
            mood_log.clear()
            print("âœ… Memory Wiped.")

    elif selection == '5':
        print("ğŸ‘‹ Settings Closed. Please Run Cell 3 to Chat.")
        break
    else:
        print("â�Œ Invalid Choice.")


# ==========================================
# ğŸ’� CELL 3: VOICE MODE (CLEAN AUDIO)
# ==========================================
import re
import time
import os
from IPython.display import Audio, display

# --- 1. INSTALL VOICE LIBRARY (If missing) ---
try:
    from gtts import gTTS
except ImportError:
    print("â�³ Installing Voice Library...")
    os.system("pip install gTTS")
    from gtts import gTTS

# --- 2. THE CLEANER (Fixes "Asterisk Asterisk") ---
def clean_for_voice(text):
    """
    Removes Markdown symbols and prefixes so the voice sounds natural.
    Example: "**Hello**" becomes "Hello"
    """
    # 1. Remove the "Nurse:" and "Friend:" labels so it just speaks the message
    clean = text.replace("ğŸ‘©â€�âš•ï¸� Nurse:", "").replace("ğŸ§£ Friend:", "")
    
    # 2. Remove special symbols (*, #, _, -) using Regex
    clean = re.sub(r"[\*\#\_\-]", "", clean)
    
    return clean.strip()

# --- 3. THE SPEAKER ---
def speak_out(text):
    try:
        # Step A: Clean the text
        spoken_text = clean_for_voice(text)
        
        # Step B: Generate Audio file
        tts = gTTS(text=spoken_text, lang='en')
        filename = f"voice_{int(time.time())}.mp3"
        tts.save(filename)
        
        # Step C: Show text and Play Audio
        print(f"ğŸ”Š AI: {spoken_text}") 
        display(Audio(filename, autoplay=True))
        
    except Exception as e:
        print(f"(Voice Error: {e})")

# --- 4. MAIN VOICE LOOP ---
print("\n" + "="*40)
print("ğŸ�™ï¸� VOICE MODE ACTIVE")
print("Type 'exit' to stop.")
print("="*40)

while True:
    try:
        # 1. Get User Input
        u_in = input("\nğŸ‘¤ YOU (Voice Mode): ")
        if u_in.lower() in ["exit", "quit", "stop"]:
            print("ğŸ‘‹ Voice Mode Off.")
            break

        # 2. Get Answer from Cell 1's Brain
        # (We use the 'get_response' function defined in Cell 1)
        full_response = get_response(u_in)
        
        # 3. Speak the Answer
        speak_out(full_response)
        
        # 4. Short pause to let audio load
        time.sleep(1)

    except NameError:
        print("âš ï¸� Error: Please run CELL 1 first so the AI exists!")
        break
    except Exception as e:
        print(f"âš ï¸� Error: {e}")


# ==========================================
# ğŸ”� CELL 4: THE SECRETARY AGENT (MEMORY)
# ==========================================
import json

# 1. The Secure Vault (In-Memory for Demo)
# In a real app, this would be encrypted.
personal_vault = {
    "family": {},   # e.g., "Grandson": "Phone number..."
    "finance": {},  # e.g., "Bank": "Account ends in 4433"
    "reminders": {} # e.g., "Monday": ["Call Alice", "Buy Milk"]
}

def update_vault(category, key, value):
    """Stores info securely."""
    if category in personal_vault:
        personal_vault[category][key] = value
        return f"âœ… Saved to {category}: {key} = {value}"
    return "â�Œ Error: Category not found."

def add_reminder(day, task):
    """Adds a task to a specific day."""
    day = day.capitalize()
    if day not in personal_vault["reminders"]:
        personal_vault["reminders"][day] = []
    personal_vault["reminders"][day].append(task)
    return f"â�° Reminder set for {day}: '{task}'"

def check_daily_briefing():
    """Checks meds (from Cell 1) AND reminders (from Cell 4)"""
    today = datetime.datetime.now().strftime('%A')
    
    # Get Meds (From Global DB in Cell 1)
    meds = med_db.get(today, "No meds")
    
    # Get Reminders
    tasks = personal_vault["reminders"].get(today, [])
    task_str = ", ".join(tasks) if tasks else "No extra tasks"
    
    briefing = f"""
    ğŸ“… DAILY BRIEFING for {today}:
    -----------------------------
    ğŸ’Š Meds: {meds}
    ğŸ“� To-Do: {task_str}
    """
    return briefing

print("âœ… Secretary Agent Loaded.")


# ==========================================
# ğŸ�² CELL 5: THE GAME CENTER (ENTERTAINMENT)
# ==========================================
import random

# --- GAME 1: TIC-TAC-TOE ---
board = [" "] * 9

def print_board():
    return f"""
      {board[0]} | {board[1]} | {board[2]}
     -----------
      {board[3]} | {board[4]} | {board[5]}
     -----------
      {board[6]} | {board[7]} | {board[8]}
    """

def play_xo(user_move):
    # User Move (0-8)
    try:
        move = int(user_move) - 1
        if board[move] != " ": return "âš ï¸� Space taken!"
        board[move] = "X"
    except:
        return "âš ï¸� Invalid move. Enter 1-9."
        
    # Check Win
    if check_win("X"): return f"{print_board()}\nğŸ�‰ YOU WIN!"
    
    # AI Move (Random simple logic)
    available = [i for i, x in enumerate(board) if x == " "]
    if not available: return f"{print_board()}\nğŸ¤� It's a Draw!"
    
    ai_move = random.choice(available)
    board[ai_move] = "O"
    
    if check_win("O"): return f"{print_board()}\nğŸ¤– I WIN!"
    
    return f"{print_board()}\nYour turn (1-9)?"

def check_win(player):
    wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    return any(board[a] == board[b] == board[c] == player for a,b,c in wins)

def reset_game():
    global board
    board = [" "] * 9
    return "ğŸ”„ Board cleared! Type 1-9 to start."

# --- GAME 2: NEWS & JOKES ---
def get_entertainment(request_type):
    if "joke" in request_type:
        prompt = "Tell a short, clean, funny joke suitable for a senior."
    elif "news" in request_type:
        prompt = "Give me 3 positive, uplifting short news headlines from science or history (no politics)."
    elif "story" in request_type:
        prompt = "Tell me a very short, heartwarming story about a dog."
    else:
        return "I can tell jokes, stories, or positive news!"
        
    return ask_ai(prompt) # Uses the AI function from Cell 1

print("âœ… Game Center Loaded.")


# ==========================================
# ğŸŒŸ CELL 6: THE MASTER INTERFACE
# ==========================================

print("\n" + "="*50)
print("ğŸ¤– ELDER SYNC: ULTIMATE EDITION")
print("="*50)
print("commands: 'meds', 'remind [day] [task]', 'play xo', 'news', 'joke', 'save [info]'")

game_mode = False

while True:
    user_input = input("\nğŸ‘¤ YOU: ").strip()
    u_lower = user_input.lower()
    
    if u_lower in ["exit", "quit"]:
        print("ğŸ‘‹ Goodbye!")
        break

    # --- 1. GAME MODE HANDLING ---
    if game_mode:
        if "stop" in u_lower or "exit" in u_lower:
            game_mode = False
            print("ğŸ�® Game Over. Back to Chat.")
        else:
            # Pass input to XO Game Logic
            print(play_xo(user_input))
        continue

    # --- 2. START GAME ---
    if "play" in u_lower and ("xo" in u_lower or "game" in u_lower):
        game_mode = True
        print(reset_game())
        print(print_board())
        print("ğŸ�® TIC-TAC-TOE STARTED! Enter 1-9 to place X.")
        continue

    # --- 3. SECRETARY AGENT (Reminders & Vault) ---
    if "remind" in u_lower:
        # Simple parser: "Remind me on Monday to call Son"
        try:
            parts = u_lower.split(" to ")
            day_part = parts[0].replace("remind me on", "").strip()
            task_part = parts[1]
            print(f"ğŸ“� Secretary: {add_reminder(day_part, task_part)}")
        except:
            print("ğŸ“� Secretary: Please say format: 'Remind me on [Day] to [Task]'")
    
    elif "briefing" in u_lower or "what do i have" in u_lower:
        print(check_daily_briefing())

    elif "save" in u_lower and "bank" in u_lower:
        # Example: "Save bank details Account 123"
        print(f"ğŸ”� Secretary: {update_vault('finance', 'BankDetails', user_input)}")
        
    # --- 4. NURSE & FRIEND AGENT (Legacy from Cell 1) ---
    elif "news" in u_lower or "joke" in u_lower or "story" in u_lower:
        print(f"âœ¨ Companion: {get_entertainment(u_lower)}")
        
    else:
        # Fallback to the original Brain from Cell 1
        # It handles Meds and Small Talk
        print(get_response(user_input))

