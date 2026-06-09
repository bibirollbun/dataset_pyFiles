# --- Import necessary libraries ---
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import textwrap
import time
import sys

# --- CONFIGURATION ---
# IMPORTANT: Your Project ID
PROJECT_ID = "gen-lang-client-0903866717" 
LOCATION = "us-central1"

# ======================================================================
# âœ… --- 1. PASTE YOUR STUDY TEXT HERE --- âœ…
# ======================================================================

study_text = """
PASTE YOUR STUDY MATERIAL RIGHT HERE,
BETWEEN THE TRIPLE QUOTES.

You can paste multiple paragraphs,
just like this.
"""

# ======================================================================

# --- Initialize Vertex AI ---
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- Define the "Personality" of StudyBot ---
studybot_persona = (
    "You are 'StudyBot', a friendly, encouraging, and expert AI study partner. "
    "Your goal is to help a student learn and feel confident. "
    "You are never judgmental. You break down complex topics into simple, "
    "easy-to-understand explanations. Your tone is always helpful and positive."
)

# --- Load the Gemini Model with the Persona ---
model = GenerativeModel(
    "gemini-1.5-flash-001",
    system_instruction=studybot_persona,
    generation_config=GenerationConfig(
        temperature=0.3,
        max_output_tokens=8192
    )
)

# --- A helper for "humanized" interaction ---
def show_thinking():
    """Prints a small 'thinking' animation to feel more human."""
    print("\nGot it! Analyzing the text... ðŸ¤”", end="", flush=True)
    for _ in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\r" + " " * 50 + "\r", end="", flush=True)

# --- Define the Agent's "Skills" ---

def get_summary_and_takeaways(text):
    prompt = f"Please act as my friendly study partner. 1. First, provide a concise summary of the following material. 2. Second, pull out the 3-5 most important 'Key Takeaways' from the text. Material:\n---\n{text}\n---"
    try:
        response = model.generate_content(prompt); return response.text
    except Exception as e: return f"Error: {e}"
def get_flashcards(text):
    prompt = f"Please generate a list of 10-15 key terms and their definitions from the following text. Format them perfectly for flashcards like this:\nTerm: [The Term]\nDefinition: [The Definition]\n\n---\n\nMaterial:\n---\n{text}\n---"
    try:
        response = model.generate_content(prompt); return response.text
    except Exception as e: return f"Error: {e}"
def get_quiz_multiple_choice(text):
    prompt = f"Let's test my knowledge! Create a 5-question multiple-choice quiz based on the following text. The questions should test the main ideas. Please provide the question, 4 options (A, B, C, D), and clearly indicate the **Correct Answer**. Material:\n---\n{text}\n---"
    try:
        response = model.generate_content(prompt); return response.text
    except Exception as e: return f"Error: {e}"
def get_quiz_fill_in_blank(text):
    prompt = f"I'd like a different kind of quiz. Please create a 5-question fill-in-the-blank quiz based on the key facts in the text. Use [BLANK] for the missing word. Provide an answer key at the very end. Material:\n---\n{text}\n---"
    try:
        response = model.generate_content(prompt); return response.text
    except Exception as e: return f"Error: {e}"
def get_essay_questions(text):
    prompt = f"I need to prepare for my exam. Based on the text, what are 3 thought-provoking essay questions a professor might ask? These should require critical thinking and not just simple recall. Material:\n---\n{text}\n---"
    try:
        response = model.generate_content(prompt); return response.text
    except Exception as e: return f"Error: {e}"
def ask_a_question(text, user_question):
    prompt = f"I'm a student, and I have a specific question about the study material I've provided. Please answer it in a clear, simple, and encouraging way. If the answer isn't in the text, please say so. Study Material:\n---\n{text}\n---\n\nMy Question: \"{user_question}\"\n\nYour Answer:"
    try:
        response = model.generate_content(prompt); return response.text
    except Exception as e: return f"Error: {e}"

# --- Helper functions for the main loop ---
def print_welcome():
    """Clears screen and prints a welcome message."""
    print("=" * 60)
    print("ðŸ‘‹ Hi there! I'm StudyBot, your personal AI study partner.")
    print("My goal is to help you learn and feel confident. You've got this!")
    print("=" * 60)

def print_menu():
    """Displays the main menu of choices."""
    print("\n--- What can I help you with? ---")
    print("  1. Give me a Summary & Key Takeaways")
    print("  2. Create Flashcards (Key Terms & Definitions)")
    print("  3. Quiz me! (Multiple Choice)")
    print("  4. Quiz me! (Fill-in-the-Blank)")
    print("  5. Generate Practice Essay Questions")
    print("  6. I have a specific question about the text...")
    print("-----------------------------------")
    print("  7. Exit")
    print("\n(To load new text, paste it in the `study_text` variable at the top and re-run the cell.)")


# --- Main Program to Run the Agent ---
def main():
    """
    The main interactive loop for the StudyBot.
    """
    print_welcome()
    
    # We no longer call get_study_text(). We just use the variable from the top.
    if not study_text or "PASTE YOUR STUDY MATERIAL RIGHT HERE" in study_text:
        print("Whoops! It looks like you forgot to paste your study material.")
        print("Please paste your text into the `study_text` variable at the top of the cell and re-run.")
        return

    print(f"\nAwesome! I've read your {len(study_text)} characters of text.")
    print("I'm ready to help!")

    while True:
        print_menu()
        # This input() for the menu WILL work in Kaggle
        choice = input("Enter your choice (1-7): ")
        
        if choice == '1':
            show_thinking()
            print("Here's a breakdown of the text:")
            print("-" * 30)
            print(textwrap.fill(get_summary_and_takeaways(study_text), width=80))
            print("-" * 30)
            
        elif choice == '2':
            show_thinking()
            print("Great idea! Here are some flashcards to get you started:")
            print("-" * 30)
            print(get_flashcards(study_text))
            print("-" * 30)
            
        elif choice == '3':
            show_thinking()
            print("Okay, let's test your knowledge! Here's your quiz:")
            print("-" * 30)
            print(get_quiz_multiple_choice(study_text))
            print("-" * 30)
        
        elif choice == '4':
            show_thinking()
            print("You got it! A fill-in-the-blank quiz coming right up:")
            print("-" * 30)
            print(get_quiz_fill_in_blank(study_text))
            print("-" * 30)
        
        elif choice == '5':
            show_thinking()
            print("Preparing for the long-form questions, smart! Here you go:")
            print("-" * 30)
            print(get_essay_questions(study_text))
            print("-" * 30)
            
        elif choice == '6':
            # This input() will also work
            user_question = input("What's your question about the text?\n> ")
            if user_question:
                show_thinking()
                print("Good question! Here's what I found in the text:")
                print("-" * 30)
                print(textwrap.fill(ask_a_question(study_text, user_question), width=80))
                print("-" * 30)
            else:
                print("Looks like you didn't ask a question. Try again!")
            
        elif choice == '7':
            print("\nHappy studying! You're doing great. See you next time!")
            break
            
        else:
            print("\nOops! I didn't recognize that. Please enter a number from 1 to 7.")

# --- Run the main program ---
if __name__ == "__main__":
    main()

