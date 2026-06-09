from graphviz import Digraph

dot = Digraph(comment='NeuroBuddy Multi-Agent Pipeline', format='png')

# Nodes
dot.node('A', 'Task Orchestrator\n(Main Agent)\n- Decides workflow\n- Coordinates agents', 
         shape='box', style='filled', color='lightblue')
dot.node('B', 'Study Summary Agent\n- Processes notes/text\n- Generates explanations', 
         shape='box', style='filled', color='lightgreen')
dot.node('C', 'Memory Tracking Agent\n- Tracks topics & sessions\n- Detects knowledge gaps', 
         shape='box', style='filled', color='lightyellow')
dot.node('D', 'Quiz Generation Agent\n- Generates personalized quizzes\n- Multiple-choice / Open-ended', 
         shape='box', style='filled', color='lightpink')

# Edges
dot.edge('A', 'B')
dot.edge('A', 'C')
dot.edge('B', 'D')
dot.edge('C', 'D')

# Display in notebook
dot



!pip install openai python-dotenv
#to install packages


# --- Core NeuroBuddy Functions ---

# 1ï¸�âƒ£ Explanation agent
def explanation_agent(input_text, preferences):
    clm = preferences["cognitive_load_multiplier"]
    return f"Explanation (CLM={clm}): {input_text} explained in detail."

# 2ï¸�âƒ£ Summary agent
def summary_agent(input_text):
    return f"Summary: {input_text[:50]}..."  # truncated for display

# 3ï¸�âƒ£ Flashcard agent
def flashcard_agent(summary_text):
    return f"Flashcard Question: What is meant by '{summary_text[:30]}...?'"

# 4ï¸�âƒ£ Quiz agent
def quiz_agent(summary_text):
    return f"Quiz Question: Explain '{summary_text[:30]}...' in your own words."

# 5ï¸�âƒ£ Mock NeuroBuddy response (for testing)
def ask_neurobuddy_mock(question):
    return f"[Mock Answer] This is a study explanation for: {question}"

# 6ï¸�âƒ£ Session handler
study_sessions = []

def add_session(input_text, study_output, preferences):
    """
    Store a study session in memory.
    """
    session = {
        "input_text": input_text,
        "output": study_output,
        "preferences": preferences
    }
    study_sessions.append(session)
    print(f"âœ… Session added! Total sessions: {len(study_sessions)}")

def handle_study_input(input_text, preferences):
    """
    Pass input through the sequential pipeline: explanation â†’ summary â†’ flashcards â†’ quiz.
    """
    explanation = explanation_agent(input_text, preferences)
    summary = summary_agent(explanation)
    flashcard = flashcard_agent(summary)
    quiz = quiz_agent(summary)
    return {
        "explanation": explanation,
        "summary": summary,
        "flashcard": flashcard,
        "quiz": quiz
    }

def display_study_output(study_output):
    """
    Nicely print the outputs of the sequential pipeline with emojis and headers.
    """
    print(f"ğŸ§  ### Step 1 - Explanation\n{study_output['explanation']}\n")
    print(f"ğŸ“� ### Step 2 - Summary\n{study_output['summary']}\n")
    print(f"ğŸ�´ ### Step 3 - Flashcard\n{study_output['flashcard']}\n")
    print(f"â�“ ### Step 4 - Quiz\n{study_output['quiz']}\n")



!pip install ipywidgets --quiet



import ipywidgets as widgets
from IPython.display import display, clear_output

# Mock NeuroBuddy
def ask_neurobuddy_mock(question):
    return f"[Mock Answer] This is a study explanation for: {question}"

# Widgets
question_box = widgets.Text(
    placeholder='Type your question here...',
    description='Question:'
)
submit_button = widgets.Button(description="Ask NeuroBuddy")
output_area = widgets.Output()

def on_click(b):
    with output_area:
        clear_output()
        if question_box.value.strip() == "":
            print("Please enter a question!")
            return
        print(ask_neurobuddy_mock(question_box.value))

submit_button.on_click(on_click)

display(question_box, submit_button, output_area)



from IPython.display import clear_output

def on_submit(b):
    with output_area:
        clear_output()  # Clear previous answer
        user_question = question_box.value.strip()
        if not user_question:
            print("Please type a question!")
        else:
            print(ask_neurobuddy_mock(user_question))

submit_button.on_click(on_submit)



import ipywidgets as widgets
from IPython.display import display, clear_output

# ----------------------------
# Mock NeuroBuddy function
# ----------------------------
def ask_neurobuddy_mock(question):
    return f"[Mock Answer] This is a study explanation for: {question}"

# ----------------------------
# Widgets
# ----------------------------
question_box = widgets.Text(
    placeholder='Type your question here...',
    description='Question:',
    layout=widgets.Layout(width='70%')
)

submit_button = widgets.Button(description="Ask NeuroBuddy", button_style='success')
output_area = widgets.Output()

# ----------------------------
# Button click handler
# ----------------------------
def on_submit(b):
    with output_area:
        clear_output()  # clear previous output
        user_question = question_box.value.strip()
        if not user_question:
            print("Please type a question!")
        else:
            answer = ask_neurobuddy_mock(user_question)
            print(f"NeuroBuddy: {answer}")

# Link button to function
submit_button.on_click(on_submit)

# ----------------------------
# Display the UI
# ----------------------------
display(question_box, submit_button, output_area)



import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output

# --- Markdown explanation ---
display(Markdown("## 1. Set Your Preferences\n"
                 "You can adjust these parameters in the notebook cells:\n\n"
                 "**â€¢ Cognitive Load Multiplier (CLM):**  \n"
                 "Please choose a value from **1â€“5** to control explanation depth.  \n"
                 "(1 = simple, 5 = highly detailed)\n\n"
                 "**â€¢ Brain State:**  \n"
                 "Please select one of: **\"focused\"**, **\"tired\"**, **\"overwhelmed\"**.  \n"
                 "This adjusts tone and pacing.\n\n"
                 "**â€¢ Sensory Mode:**  \n"
                 "Please choose **\"text\"**, **\"visual\"**, or **\"interactive\"** output styles."))

# --- Interactive widgets ---
clm_slider = widgets.IntSlider(value=3, min=1, max=5, step=1, description='Cognitive Load:', style={'description_width': 'initial'})
brain_dropdown = widgets.Dropdown(options=['focused', 'tired', 'overwhelmed'], value='focused', description='Brain State:', style={'description_width': 'initial'})
sensory_dropdown = widgets.Dropdown(options=['text', 'visual', 'interactive'], value='text', description='Sensory Mode:', style={'description_width': 'initial'})
submit_btn = widgets.Button(description='Set Preferences', button_style='success')
output_area = widgets.Output()

def on_submit(b):
    with output_area:
        clear_output()
        preferences = {
            "cognitive_load_multiplier": clm_slider.value,
            "brain_state": brain_dropdown.value,
            "sensory_mode": sensory_dropdown.value
        }
        print("âœ… Preferences successfully set!")
        print(preferences)

submit_btn.on_click(on_submit)

display(clm_slider, brain_dropdown, sensory_dropdown, submit_btn, output_area)



import ipywidgets as widgets
from IPython.display import display, clear_output

# --- Mock summarization function ---
def summarize_notes_mock(notes_text):
    return f"[Mock Summary] Key points from your notes:\n- {notes_text[:100]}...\n- Continue reviewing the main ideas!"

# --- Input widget ---
notes_box = widgets.Textarea(
    placeholder='Paste your notes here...',
    description='Notes:',
    layout=widgets.Layout(width='80%', height='150px')
)

# --- Generate button ---
submit_button_summarize = widgets.Button(
    description="Summarize",
    button_style='warning',
    tooltip="Click to generate a summary"
)

# --- Output area ---
output_area_summarize = widgets.Output()

# --- Click handler ---
def on_summarize_click(b):
    with output_area_summarize:
        clear_output()
        content = notes_box.value.strip()
        if not content:
            print("âš ï¸� Please paste your notes!")
        else:
            print(summarize_notes_mock(content))

submit_button_summarize.on_click(on_summarize_click)

# --- Display widgets ---
display(notes_box, submit_button_summarize, output_area_summarize)



import ipywidgets as widgets
from IPython.display import display, clear_output

# --- Mock flashcard generator function ---
def flashcard_mock(topic_or_text):
    return (f"[Mock Flashcards] Key questions for '{topic_or_text}':\n"
            "1. What is the definition of the main concept?\n"
            "2. Explain a key example.\n"
            "3. How would you apply this in practice?\n"
            "4. List the main advantages or disadvantages.\n"
            "5. What common mistakes should be avoided?")

# --- Input widget ---
flashcard_box = widgets.Text(
    placeholder='Type a concept or topic here...',
    description='Concept/Topic:',
    layout=widgets.Layout(width='70%')
)

# --- Generate button ---
submit_button_flashcards = widgets.Button(
    description="Generate Flashcards",
    button_style='success',
    tooltip="Click to generate flashcards"
)

# --- Output area ---
output_area_flashcards = widgets.Output()

# --- Click handler ---
def on_flashcard_click(b):
    with output_area_flashcards:
        clear_output()
        content = flashcard_box.value.strip()
        if not content:
            print("âš ï¸� Please enter a concept or topic!")
        else:
            print(flashcard_mock(content))

submit_button_flashcards.on_click(on_flashcard_click)

# --- Display widgets ---
display(flashcard_box, submit_button_flashcards, output_area_flashcards)



import ipywidgets as widgets
from IPython.display import display, clear_output

# --- Mock function to generate a study plan ---
def study_schedule_mock(topic_or_course):
    return (f"[Mock Study Schedule] Suggested study schedule for '{topic_or_course}':\n"
            "Day 1: Introduction and overview\n"
            "Day 2: Key concepts\n"
            "Day 3: Practice problems\n"
            "Day 4: Review and summary\n"
            "Day 5: Mock test")

# --- Input widget ---
schedule_box = widgets.Text(
    placeholder='Type a topic or course here...',
    description='Topic/Course:',
    layout=widgets.Layout(width='70%')
)

# --- Generate button ---
submit_button_schedule = widgets.Button(
    description="Generate Schedule",
    button_style='info',
    tooltip="Click to generate a study plan"
)

# --- Output area ---
output_area_schedule = widgets.Output()

# --- Click handler ---
def on_schedule_click(b):
    with output_area_schedule:
        clear_output()
        content = schedule_box.value.strip()
        if not content:
            print("âš ï¸� Please enter a topic or course!")
        else:
            print(study_schedule_mock(content))

submit_button_schedule.on_click(on_schedule_click)

# --- Display widgets ---
display(schedule_box, submit_button_schedule, output_area_schedule)



# ----------------------------
# Import packages
# ----------------------------
import ipywidgets as widgets
from IPython.display import display, clear_output

# ----------------------------
# Mock NeuroBuddy core function
# ----------------------------
def neurobuddy_agent(task, content):
    """
    task: str - one of ['explain', 'summary', 'flashcards', 'quiz', 'schedule']
    content: str - topic, notes, or question
    """
    if task == "explain":
        return f"[Mock Explanation] Hereâ€™s a simple explanation of '{content}'."
    elif task == "summary":
        return f"[Mock Summary] Hereâ€™s a concise summary of your notes: {content[:100]}... (truncated for demo)"
    elif task == "flashcards":
        return f"[Mock Flashcards] Key points for '{content}':\n1. ...\n2. ...\n3. ..."
    elif task == "quiz":
        return f"[Mock Quiz] Quiz questions for '{content}':\nQ1: ...\nQ2: ...\nQ3: ..."
    elif task == "schedule":
        return f"[Mock Schedule] Suggested study schedule for '{content}':\nDay 1: Intro\nDay 2: Key concepts\nDay 3: Practice\nDay 4: Review\nDay 5: Mock test"
    else:
        return "[Mock] Task not recognized"

# ----------------------------
# Widgets
# ----------------------------
task_dropdown = widgets.Dropdown(
    options=[
        ("Explain Concept", "explain"),
        ("Summarize Notes", "summary"),
        ("Generate Flashcards", "flashcards"),
        ("Create Quiz", "quiz"),
        ("Study Schedule", "schedule")
    ],
    description="Task:"
)

content_box = widgets.Textarea(
    placeholder='Type your topic, notes, or question here...',
    description='Content:',
    layout=widgets.Layout(width='80%', height='150px')
)

submit_button = widgets.Button(description="Run NeuroBuddy", button_style='success')
output_area = widgets.Output()

# ----------------------------
# Button click handler
# ----------------------------
def on_click(b):
    with output_area:
        clear_output()
        task = task_dropdown.value
        content = content_box.value.strip()
        if not content:
            print("Please enter some content to process!")
        else:
            print(neurobuddy_agent(task, content))

submit_button.on_click(on_click)

# ----------------------------
# Display full interface
# ----------------------------
display(task_dropdown, content_box, submit_button, output_area)



import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown

# --- Display section header ---
display(Markdown("## ğŸ§  NeuroBuddy Study Sessions & Memory"))

# --- Store sessions ---
study_sessions = []

# --- Functions ---
def add_session(input_text, study_output, preferences):
    """Add a new study session to memory."""
    session = {
        "input_text": input_text,
        "output": study_output,
        "preferences": preferences
    }
    study_sessions.append(session)
    with output_area:
        clear_output()
        print(f"âœ… Session added! Total sessions: {len(study_sessions)}")

def view_sessions():
    """Display all previous sessions with outputs."""
    with output_area:
        clear_output()
        if not study_sessions:
            print("No sessions yet. Start studying to create your first session!")
            return
        for i, session in enumerate(study_sessions, 1):
            print(f"\nğŸ“š Session {i}")
            print(f"Input: {session['input_text']}")
            print(f"Output: {session['output']}")
            print(f"Preferences: {session['preferences']}")
            print("-" * 50)

def handle_study_input(input_text, preferences):
    """Mock function to process study input."""
    return f"[Mock Output] Processed '{input_text}' with preferences {preferences}"

# --- Widgets for adding a session ---
input_box = widgets.Textarea(
    placeholder="Type your study notes or topic here...",
    description="Study Input:",
    layout=widgets.Layout(width='80%', height='80px'),
    style={'description_width': 'initial'}
)

add_btn = widgets.Button(description="Add Session", button_style='success')
view_btn = widgets.Button(description="View All Sessions", button_style='info')
output_area = widgets.Output()

# --- Example: Using existing preferences ---
# Replace this with widget-based preferences if you have them
user_preferences = {
    "cognitive_load_multiplier": 3,
    "brain_state": "focused",
    "sensory_mode": "text"
}

# --- Callbacks ---
def on_add_clicked(b):
    study_input = input_box.value.strip()
    if not study_input:
        with output_area:
            clear_output()
            print("â�— Please enter some study input first!")
        return
    study_output = handle_study_input(study_input, user_preferences)
    add_session(study_input, study_output, user_preferences)

def on_view_clicked(b):
    view_sessions()

add_btn.on_click(on_add_clicked)
view_btn.on_click(on_view_clicked)

# --- Display widgets ---
display(input_box, widgets.HBox([add_btn, view_btn]), output_area)




# --- NeuroBuddy Sequential Agent Pipeline (Notebook Example) ---

# 1ï¸�âƒ£ Define each "agent" as a function
def explanation_agent(input_text, preferences):
    clm = preferences["cognitive_load_multiplier"]
    return f"Explanation (CLM={clm}): {input_text} explained in detail."

def summary_agent(input_text):
    return f"Summary: {input_text[:50]}..."  # truncate for display

def flashcard_agent(summary_text):
    return f"Flashcard Question: What is meant by '{summary_text[:30]}...?'"

def quiz_agent(summary_text):
    return f"Quiz Question: Explain '{summary_text[:30]}...' in your own words."

# 2ï¸�âƒ£ Example input and user preferences
user_preferences = {
    "cognitive_load_multiplier": 3,
    "brain_state": "focused",
    "sensory_mode": "text"
}

input_text = "NeuroBuddy is an AI study assistant that helps students learn efficiently."

# 3ï¸�âƒ£ Sequential pipeline execution
explanation = explanation_agent(input_text, user_preferences)
summary = summary_agent(explanation)
flashcard = flashcard_agent(summary)
quiz = quiz_agent(summary)

# 4ï¸�âƒ£ Display results with notebook-friendly formatting
print("### Step 1 - Explanation\n", explanation, "\n")
print("### Step 2 - Summary\n", summary, "\n")
print("### Step 3 - Flashcard\n", flashcard, "\n")
print("### Step 4 - Quiz\n", quiz)






# --- Interactive UI Widgets for NeuroBuddy ---
import ipywidgets as widgets
from IPython.display import display

# Cognitive Load Multiplier slider
clm_slider = widgets.IntSlider(
    value=3,
    min=1,
    max=5,
    step=1,
    description='CLM:',
    continuous_update=False
)

# Brain State dropdown
brain_state_dropdown = widgets.Dropdown(
    options=['focused', 'tired', 'overwhelmed'],
    value='focused',
    description='Brain State:'
)

# Sensory Mode dropdown
sensory_mode_dropdown = widgets.Dropdown(
    options=['text', 'visual', 'interactive'],
    value='text',
    description='Sensory Mode:'
)

# Button to confirm preferences
button = widgets.Button(description="Set Preferences")

# Function to display chosen preferences
def on_button_click(b):
    preferences = {
        "cognitive_load_multiplier": clm_slider.value,
        "brain_state": brain_state_dropdown.value,
        "sensory_mode": sensory_mode_dropdown.value
    }
    print("\nâœ… Preferences set successfully!")
    print(preferences)

button.on_click(on_button_click)

# Display all widgets
display(clm_slider, brain_state_dropdown, sensory_mode_dropdown, button)




# --- Handlers for NeuroBuddy ---

def display_study_output(study_output):
    """
    Display the pipeline outputs with headers, emojis, and clear formatting.
    """
    print(f"ğŸ§  ### Step 1 - Explanation\n{study_output['explanation']}\n")
    print(f"ğŸ“� ### Step 2 - Summary\n{study_output['summary']}\n")
    print(f"ğŸ�´ ### Step 3 - Flashcard\n{study_output['flashcard']}\n")
    print(f"â�“ ### Step 4 - Quiz\n{study_output['quiz']}\n")


# Example study output dictionary
study_output = {
    "explanation": "Explanation (CLM=3): NeuroBuddy helps students learn effectively. explained in detail.",
    "summary": "Summary: Explanation (CLM=3): NeuroBuddy helps students lea...",
    "flashcard": "Flashcard Question: What is meant by 'Summary: Explanation (CLM=3): ...?'",
    "quiz": "Quiz Question: Explain 'Summary: Explanation (CLM=3): ...' in your own words."
}

# Display it
display_study_output(study_output)




import ipywidgets as widgets
from IPython.display import display, clear_output, Javascript
import time
import threading

# --- Browser-based Text-to-Speech ---
def speak(text):
    display(Javascript(f'const msg = new SpeechSynthesisUtterance("{text}"); window.speechSynthesis.speak(msg);'))

# --- Pomodoro Timer ---
pomodoro_output = widgets.Output()
pomodoro_minutes = widgets.IntSlider(value=25, min=5, max=60, step=5, description='Pomodoro (min):')
start_pomodoro_btn = widgets.Button(description="Start Timer", button_style='info')

def start_pomodoro(b):
    def run_timer(minutes):
        with pomodoro_output:
            for i in range(minutes * 60, -1, -1):
                mins, secs = divmod(i, 60)
                clear_output(wait=True)
                print(f"â�±ï¸� Time Remaining: {mins:02d}:{secs:02d}")
                time.sleep(1)
            print("âœ… Pomodoro Complete!")
            speak("Pomodoro complete!")
    threading.Thread(target=run_timer, args=(pomodoro_minutes.value,)).start()

start_pomodoro_btn.on_click(start_pomodoro)

# --- Task / Progress Tracker ---
tasks_output = widgets.Output()
task_box = widgets.Text(description='New Task:')
add_task_btn = widgets.Button(description='Add Task', button_style='success')
tasks_list = []

def update_tasks():
    with tasks_output:
        clear_output(wait=True)
        if not tasks_list:
            print("No tasks yet. Add one above!")
        for i, t in enumerate(tasks_list):
            status = 'âœ…' if t['done'] else 'â�Œ'
            print(f"{i+1}. {status} {t['task']}")

def add_task(b):
    task = task_box.value.strip()
    if task:
        tasks_list.append({'task': task, 'done': False})
        task_box.value = ''
        update_tasks()

add_task_btn.on_click(add_task)

# --- Display all widgets ---
display(widgets.HTML("<h3>ğŸ�¯ Pomodoro Timer</h3>"), pomodoro_minutes, start_pomodoro_btn, pomodoro_output)
display(widgets.HTML("<h3>ğŸ“� Task Tracker</h3>"), task_box, add_task_btn, tasks_output)

# Show initial task list
update_tasks()



# --- Friendly closing message ---
def wrap_up_message():
    print("ğŸ�‰ Great job! Your NeuroBuddy session is complete.")
    print("ğŸ’¡ Tip: Adjust your preferences and try another topic to continue learning.\n")

wrap_up_message()



import json
import pandas as pd
from datetime import datetime

def save_neurbuddy_session(user_name,
                           preferences,
                           explanations,
                           quizzes,
                           memory_logs,
                           file_prefix="neurbuddy_session",
                           save_csv=True):
    session_data = {
        "user": user_name,
        "preferences": preferences,
        "session_explanations": explanations,
        "quiz_results": quizzes,
        "memory_logs": memory_logs,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # JSON
    json_file = f"{file_prefix}_{user_name.replace(' ', '_')}.json"
    with open(json_file, "w") as f:
        json.dump(session_data, f, indent=4)
    print(f"Session saved successfully to {json_file}")

    # CSVs
    if save_csv:
        if quizzes:
            pd.DataFrame(quizzes).to_csv(f"{file_prefix}_{user_name.replace(' ', '_')}_quizzes.csv", index=False)
            print(f"Quizzes saved to {file_prefix}_{user_name.replace(' ', '_')}_quizzes.csv")
        if memory_logs:
            pd.DataFrame(memory_logs).to_csv(f"{file_prefix}_{user_name.replace(' ', '_')}_memory_logs.csv", index=False)
            print(f"Memory logs saved to {file_prefix}_{user_name.replace(' ', '_')}_memory_logs.csv")

    return json_file



user_name = "Soumya Padma"
preferences = {
    "cognitive_load": 3,
    "brain_state": "focused",
    "sensory_mode": "visual"
}
explanations = [
    {"topic": "Neuroplasticity", "output": "The brain's ability to reorganize and form new connections."},
    {"topic": "Adaptive Learning", "output": "Learning adjusts based on the learner's state."}
]
quizzes = [
    {"question": "Define neuroplasticity", "answer": "Brain's ability to adapt", "correct": True},
    {"question": "What is adaptive learning?", "answer": "Learning adjusts based on learner state", "correct": True}
]
memory_logs = [
    {"topic": "Neuroplasticity", "gaps": "None", "recommendations": "Review weekly"},
    {"topic": "Adaptive Learning", "gaps": "Needs reinforcement", "recommendations": "Do 2 practice quizzes"}
]



save_neurbuddy_session(user_name, preferences, explanations, quizzes, memory_logs)


