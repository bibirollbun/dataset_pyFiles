!pip install langchain==0.2.10
!pip install langchain-openai==0.1.7
!pip install openai==1.35.13



# Using a simulated LLM for Kaggle notebook (no API key needed)
def call_llm(prompt):
    # This is a safe placeholder for demo purposes
    return f"(Simulated LLM reply) Prompt received: {prompt[:100]}..."

print("Simulated LLM ready!")



# --- Memory for Student Success Coach Agent ---
class SimpleMemory:
    def __init__(self):
        self.store = {
            'student_profile': {},
            'subjects': {},
            'exam_dates': {},
            'progress': {},
        }

    def read(self, key):
        return self.store.get(key)

    def write(self, key, value):
        self.store[key] = value

    def update_profile(self, **kwargs):
        profile = self.store.setdefault('student_profile', {})
        profile.update(kwargs)
        self.store['student_profile'] = profile

    def append_progress(self, subject, entry):
        self.store['progress'].setdefault(subject, []).append(entry)

    def get_progress(self, subject):
        return self.store['progress'].get(subject, [])

memory = SimpleMemory()
print("Memory setup complete.")



# --- Tools for Student Success Coach Agent ---

from datetime import datetime, timedelta

def create_study_plan(subject, start_date, exam_date, daily_minutes):
    """Create day-by-day study plan."""
    if isinstance(start_date, str):
        start = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start = start_date
    if isinstance(exam_date, str):
        end = datetime.strptime(exam_date, '%Y-%m-%d')
    else:
        end = exam_date

    days = (end - start).days
    if days <= 0:
        return []

    sessions = []
    for i in range(days):
        day = (start + timedelta(days=i)).strftime('%Y-%m-%d')
        sessions.append({'date': day, 'subject': subject, 'minutes': daily_minutes})
    return sessions

def generate_notes(topic, level='beginner'):
    """Generate notes (simulated LLM)."""
    prompt = f"Create concise study notes for '{topic}' at {level} level with 3 bullets."
    return call_llm(prompt)

def generate_quiz(topic, n=5):
    """Generate sample quiz questions."""
    questions = [f"Q{i+1}: What is a key point about {topic}? (sample question)" for i in range(n)]
    return questions

print("Tools setup complete.")



# --- Agent Classes for Student Success Coach Agent ---

class PlannerAgent:
    def plan(self, student_profile, exam_dates):
        plans = {}
        start_date = datetime.now().strftime('%Y-%m-%d')
        for subj, date in exam_dates.items():
            plan = create_study_plan(subj, start_date, date, student_profile.get('daily_available_minutes', 60))
            plans[subj] = plan
        return plans

class NotesAgent:
    def create_notes(self, topic, level='beginner'):
        return generate_notes(topic, level)

class ConceptAgent:
    def explain(self, topic, level='beginner'):
        prompt = f"Explain {topic} in simple terms for a {level} student with a short example."
        return call_llm(prompt)

class MotivationAgent:
    def message(self, name='Student'):
        prompt = f"Write a short motivational message for {name} to study today."
        return call_llm(prompt)

class StudentCoach:
    def __init__(self, memory):
        self.memory = memory
        self.planner = PlannerAgent()
        self.notes = NotesAgent()
        self.concept = ConceptAgent()
        self.motivation = MotivationAgent()

    def build_plan(self):
        profile = self.memory.read('student_profile')
        exam_dates = self.memory.read('exam_dates')
        return self.planner.plan(profile, exam_dates)

    def explain_topic(self, topic):
        return self.concept.explain(topic)

    def create_notes(self, topic):
        return self.notes.create_notes(topic)

    def get_daily_message(self):
        profile = self.memory.read('student_profile') or {}
        return self.motivation.message(profile.get('name', 'Student'))

coach = StudentCoach(memory)
print("Agent setup complete.")



# --- 8. Demo / Example Usage ---

# Initialize a sample student profile
memory.update_profile(
    name='Alex Student',
    preferred_style='short-bursts',
    daily_available_minutes=90
)

# Add exam dates
memory.write('exam_dates', {
    'Math': '2026-01-15',
    'Physics': '2026-01-20'
})

# Generate study plan
plan = coach.build_plan()
print("Study plan subjects:", list(plan.keys()))
print("\nFirst 3 Math plan entries:")
from pprint import pprint
pprint(plan['Math'][:3])

# Explain a topic
explanation = coach.explain_topic('Integration by substitution')
print("\nExplanation (simulated LLM):")
print(explanation)

# Generate notes
notes = coach.create_notes('Integration by substitution')
print("\nNotes (simulated LLM):")
print(notes)

# Generate quiz
quiz = generate_quiz('Integration by substitution', n=3)
print("\nSample quiz:")
pprint(quiz)

# Motivation message
msg = coach.get_daily_message()

# If using simulated LLM, replace with a real-looking message
if "(Simulated LLM reply)" in msg:
    msg = "Hey Alex! Keep going â€” every study session brings you closer to success! ğŸ’ªğŸ“š"

print("\nMotivation message:")
print(msg)



# Adaptive study plan: adjust sessions based on progress
for subj in plan:
    completed_today = 2  # Example: student completed 2 tasks today
    if completed_today < 3:  # if not enough tasks completed
        plan[subj].append({"Day": "Extra Session", "Topic": "Review Previous"})



# Dynamic motivation based on topic difficulty
topic_difficulty = 'hard'  # Example difficulty level
msg = f"Alex, {topic_difficulty} topics are tough, but keep going! Each step counts ğŸ’ª"
print("Motivational Message:", msg)



# Quick review flashcards
topic = 'Integration by substitution'
flashcards = ["Definition of substitution", "Steps for integration", "Common mistakes"]
print("\nFlashcard Review for", topic)
for i, card in enumerate(flashcards, 1):
    print(f"{i}. {card}")



# Sample quiz + answers
quiz_questions = ["Q1: âˆ«x dx", "Q2: âˆ«x^2 dx"]
quiz_answers = ["A1: x^2/2 + C", "A2: x^3/3 + C"]

print("\nQuiz Questions & Answers:")
for q, a in zip(quiz_questions, quiz_answers):
    print(f"{q} - {a}")



import pandas as pd

# Student progress overview
progress = pd.DataFrame({
    'Subject': ['Math', 'Physics'],
    'Total Sessions': [125, 90],
    'Completed': [10, 5],
    'Remaining': [115, 85]
})

print("\nStudent Progress Overview:")
display(progress)



import matplotlib.pyplot as plt

subjects = ['Math', 'Physics']
completed = [10, 5]
remaining = [115, 85]

plt.bar(subjects, completed, color='green', label='Completed')
plt.bar(subjects, remaining, bottom=completed, color='red', label='Remaining')
plt.ylabel("Sessions")
plt.title("Study Progress Overview")
plt.legend()
plt.show()



# Dummy Gemini function (works without installing anything)
def ask_gemini(question):
    return f"[Gemini AI simulated response] Answering: {question}"

# Example usage
user_question = "How can I improve my study habits?"
answer = ask_gemini(user_question)
print("Gemini says:", answer)





import pandas as pd

# Combine all sessions from all subjects
all_sessions = []
for subj, sessions in plan.items():
    for s in sessions:
        s_copy = s.copy()
        s_copy['subject'] = subj
        all_sessions.append(s_copy)

# Create DataFrame
df_plan = pd.DataFrame(all_sessions)

# Reset index for nicer Sr No.
df_plan.reset_index(drop=True, inplace=True)
df_plan.index += 1  # Sr No. starts from 1

# Show first 5 sessions for each subject
display_df = pd.concat([df_plan[df_plan['subject'] == 'Math'].head(5),
                        df_plan[df_plan['subject'] == 'Physics'].head(5)])
display(display_df)


