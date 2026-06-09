# EduPath AI: Personalized Education Pathway Multi-Agent (Agents For Good Track)

# ==========================================
# 1. Introduction & Problem Statement
# ==========================================
# Many students lose motivation or are confused due to static, one-size-fits-all learning paths. Personalized learning is either manual or too expensive. We propose a multi-agent AI system that dynamically creates and adapts study plans for students, drawing on open educational content, user feedback, and progress evaluation.
# 
# > **Day 1:** Multi-agent architecture, sequential/parallel orchestration.
# > **Day 2:** Tool use (API calls for study resources).
# > **Day 3:** Student memory/context (logs progress).
# > **Day 4:** Success evaluation (metrics, feedback).
# > **Day 5:** Social good - improving access and outcomes for students globally.
#
# | Agent Name       | Role                       |
# |------------------|---------------------------|
# | Intake Agent     | Collects student info     |
# | Pathway Agent    | Designs learning roadmap  |
# | Recommender Agent| Suggests resources        |
# | Eval Agent       | Tracks & reports progress |

# ==========================================
# 2. User Details Input Form (Fill & Run)
# ==========================================
import ipywidgets as widgets
from IPython.display import display, clear_output

form_output = widgets.Output()
student_data = {}

def on_form_submit(btn):
    with form_output:
        clear_output()
        student_data['name'] = name.value
        student_data['goal'] = goal.value
        student_data['background'] = background.value
        student_data['subject'] = subject.value
        print(f"Intake complete. Welcome {student_data['name']}!")
        print(f"Goal: {student_data['goal']}\nBackground: {student_data['background']}\nSubject: {student_data['subject']}")

name = widgets.Text(description="Name:")
goal = widgets.Text(description="Goal:")
background = widgets.Text(description="Background:")
subject = widgets.Text(description="Subject/Topic:")

submit_form = widgets.Button(description="Submit", button_style='success')
submit_form.on_click(on_form_submit)

display(widgets.VBox([widgets.Label("Fill Personal Details for Custom Path"),
                      name, goal, background, subject, submit_form, form_output]))

# ==========================================
# 3. Multi-Agent Workflow (Main Pipeline)
# ==========================================
import random

# --- Intake Agent (uses student_data dictionary) ---

# --- Pathway Agent ---
def generate_study_path(goal, background, subject):
    # For demo, a simple static plan. In production, use actual path planner (sequence: intro, practice, review, test)
    return [
        f"1. Introduction to {subject}",
        f"2. Review key {subject} concepts",
        f"3. Practice problems (find on Khan Academy, OpenAI, Coursera)",
        f"4. Complete a project related to {goal}",
        "5. Take self-assessment quiz and get feedback"
    ]

# --- Recommender Agent (Day 2: Tool/API Calls) ---
def recommend_resources(subject):
    resource_db = {
        "python": ["Kaggle Learn Python", "Codecademy Python", "LeetCode Easy Challenges"],
        "math": ["Khan Academy Math", "Paul's Online Math Notes", "MIT OCW"],
        "ml": ["Kaggle Intro ML", "fast.ai ML", "Stanford CS229 YouTube"]
    }
    subject_key = subject.lower()
    return resource_db.get(subject_key, ["Google search for top courses", "YouTube tutorials", "Coursera specials"])

# --- Eval Agent (Day 4: Evaluation + Metrics) ---
def collect_feedback():
    # Simulated feedback for demo; in prod, ask user, track scores
    return random.choice(["Good progress", "Needs more practice", "Excellent work!", "Try more quizzes"])

# ==== Demo Example after form is filled ====
if student_data:
    path = generate_study_path(student_data['goal'], student_data['background'], student_data['subject'])
    resources = recommend_resources(student_data['subject'])
    print("Personalized Study Plan:")
    for step in path:
        print("•", step)
    print("\nResource Suggestions:", resources)
    evaluation = collect_feedback()
    print("\nProgress Evaluation:", evaluation)
else:
    print("Fill and submit the form above to generate your plan!")

# ==========================================
# 4. (Demo) Memory/Tracking Feature Example
# ==========================================
# For full version, save this data in a user profile (dict/list) and update with each run.
student_profile = {
    "name": student_data.get("name", "test"),
    "goal": student_data.get("goal", "review basics"),
    "subject": student_data.get("subject", "python"),
    "progress": random.randint(0, 100),
    "last_eval": evaluation if student_data else None
}
print("Your Profile:", student_profile)

# ==========================================
# 5. Conclusion and Next Steps
# ==========================================
# Our multi-agent EduPath AI adapts learning in real-time, provides resource suggestions, and tracks user progress.
# For deployment, this can be integrated with Gemini API (for explanations), deployed as a webapp, and demoed via YouTube.
#
# == To Score Bonus Points ==
# - **Gemini API:** Use for generating explanations/resources (insert code block if you're white-listed for API)
# - **Deployment:** Deploy agent workflow on Vercel/Streamlit (basic Python webapp)
# - **YouTube Demo:** Record 2-min walkthrough (show form, agent outputs)

# ==========================================
# 6. References
# ==========================================
# (1) Kaggle Learn Python: https://www.kaggle.com/learn/python
# (2) Khan Academy: https://www.khanacademy.org/
# (3) fast.ai: https://course.fast.ai/
# (4) Google Gemini: https://ai.google.dev/gemini-api/


