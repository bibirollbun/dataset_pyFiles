import urllib.request

urllib.request.urlretrieve(
    "https://rohitconsultants.com/imgs/tutobot_logo_1200675.png",
    "/kaggle/working/cover.png"
)



# Run this cell once in the notebook environment.
# gspread: simple Sheets client
# oauth2client: service account credentials helper
# graphviz: draw architecture diagram
!pip install --quiet groq gspread oauth2client graphviz



# Get our api keys from kaggle secrets  
import os
from kaggle_secrets import UserSecretsClient

try:
    GROQ_API_KEY = UserSecretsClient().get_secret("GROQ_API_KEY_TUTOBOT")
    os.environ["GROQ_API_KEY_TUTOBOT"] = GROQ_API_KEY
    GCP_SERVICE_ACCOUNT_JSON = UserSecretsClient().get_secret("GCP_SERVICE_ACCOUNT_JSON")
    os.environ["GCP_SERVICE_ACCOUNT_JSON"] = GCP_SERVICE_ACCOUNT_JSON
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GROQ_API_KEY_TUTOBOT' and 'GCP_SERVICE_ACCOUNT_JSON' to your Kaggle secrets. Details: {e}"
    )


# Setup the groq client
import os
import json
from typing import Dict, Any
from dataclasses import dataclass

# Groq client (you should already have set the GROQ_API_KEY_TUTOBOT in Kaggle Secrets)
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY_TUTOBOT"))

# Google Sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Graphviz for architecture diagram
from graphviz import Digraph

# Helper
import pprint
pp = pprint.PrettyPrinter(indent=2, width=120)


def safe_json_loads(s: str, default=None):
    """
    Safely load JSON from any messy LLM response.
    Tries full string, then searches for first JSON block.
    Returns `default` if parsing fails.
    """
    if not isinstance(s, str):
        return default

    s = s.strip()
    if not s:
        return default

    # Try direct parse
    try:
        return json.loads(s)
    except:
        pass

    # Try detecting JSON inside wrapper text
    for char in ["{", "["]:
        idx = s.find(char)
        if idx != -1:
            try:
                return json.loads(s[idx:])
            except:
                continue  

    return default




# Initialize the Google sheets client 
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

def init_gsheets_from_secret(secret_name="GCP_SERVICE_ACCOUNT_JSON"):
    """
    Loads the service account JSON from a Kaggle Secret and uses it
    to authenticate with Google Sheets.

    You MUST add the JSON to Kaggle:
    Settings â†’ Secrets â†’ Add new secret â†’ name = GCP_SERVICE_ACCOUNT_JSON
    """
    
    # Load the JSON string from Kaggle Secrets
    raw_json = os.getenv(secret_name)
    if raw_json is None:
        raise ValueError(
            f"â�Œ Kaggle secret '{secret_name}' not found.\n"
            "Go to: Settings â†’ Secrets â†’ Add New Secret\n"
            "Name: GCP_SERVICE_ACCOUNT_JSON\n"
            "Value: (paste the full service account JSON)"
        )
    
    # Convert string â†’ Python dict
    sa_json = json.loads(raw_json)

    # Define scopes required for reading/writing to Sheets + Drive
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    # Build credentials directly from the dict
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_json, scopes)

    # Initialize gspread client
    client = gspread.authorize(creds)
    print("âœ… Google Sheets client initialized through Kaggle Secret")

    return client

# Initialize Sheets client securely
try:
    gs_client = init_gsheets_from_secret()
except Exception as e:
    print("âš ï¸� Error initializing Google Sheets:", e)
    gs_client = None



"""
Sheet layout recommendations (single spreadsheet with multiple sheets):
- Sheet: teacher_profiles
  Columns: teacher_id | name | email | preferences_json
- Sheet: lesson_plans
  Columns: teacher_id | timestamp | curriculum_json | lessons_json | assessments_json | evaluation_json
- Sheet: form_responses (if using Forms)
  Columns: timestamp | teacher_id | subject | grade | learning_goals_json | classes_per_week | student_needs

We provide helpers to:
- open/create sheet
- append a lesson plan record
- read latest lesson plan for teacher
- upsert teacher profile
"""

def open_or_create_spreadsheet(client, title="TutoBot-Memory-Kaggle"):
    # returns a gspread.Spreadsheet object
    try:
        ss = client.open(title)
        print(f"Opened spreadsheet: {title}")
    except Exception:
        # create new spreadsheet if not found
        ss = client.create(title)
        print(f"Created spreadsheet: {title}")
    return ss

def ensure_sheets_exist(ss):
    # Ensure the named worksheets exist and have header rows
    expected = {
        "teacher_profiles": ["teacher_id", "name", "email", "preferences_json"],
        "lesson_plans": ["teacher_id", "timestamp", "curriculum_json", "lessons_json", "assessments_json", "evaluation_json"],
        "form_responses": ["timestamp", "teacher_id", "subject", "grade", "learning_goals_json", "classes_per_week", "student_needs"]
    }
    existing = {ws.title for ws in ss.worksheets()}
    for name, headers in expected.items():
        if name not in existing:
            ws = ss.add_worksheet(title=name, rows="1000", cols="20")
            ws.append_row(headers)
            print(f"Created worksheet and headers: {name}")
        else:
            ws = ss.worksheet(name)
            # You could check headers and insert if missing â€” kept simple here.

# Example usage:
if gs_client:
    ss = open_or_create_spreadsheet(gs_client, title="TutoBot-Memory-Kaggle")
    ensure_sheets_exist(ss)
else:
    ss = None



import time
from datetime import datetime

def append_lesson_plan(ss, teacher_id: str, curriculum: dict, lessons: dict, assessments: dict, evaluation: dict):
    """
    Append a lesson plan record to lesson_plans worksheet.
    """
    ws = ss.worksheet("lesson_plans")
    row = [
        teacher_id,
        datetime.utcnow().isoformat() + "Z",
        json.dumps(curriculum, ensure_ascii=False),
        json.dumps(lessons, ensure_ascii=False),
        json.dumps(assessments, ensure_ascii=False),
        json.dumps(evaluation, ensure_ascii=False)
    ]
    ws.append_row(row)
    print(f"Appended lesson plan for {teacher_id}")

def read_latest_lesson_plan(ss, teacher_id: str) -> Dict[str, Any]:
    """
    Read the latest lesson_plan row for teacher_id and return parsed JSONs.
    If not found, returns {}
    """
    ws = ss.worksheet("lesson_plans")
    values = ws.get_all_records()
    # iterate reversed so newest appended rows come first
    for rec in reversed(values):
        if rec.get("teacher_id") == teacher_id:
            # parse JSON columns
            return {
                "curriculum": json.loads(rec.get("curriculum_json") or "{}"),
                "lessons": json.loads(rec.get("lessons_json") or "{}"),
                "assessments": json.loads(rec.get("assessments_json") or "{}"),
                "evaluation": json.loads(rec.get("evaluation_json") or "{}"),
                "timestamp": rec.get("timestamp")
            }
    return {}

def upsert_teacher_profile(ss, teacher_id: str, name: str, email: str, preferences: dict):
    """
    Add or update a teacher profile in teacher_profiles
    """
    ws = ss.worksheet("teacher_profiles")
    values = ws.get_all_records()
    row_index = None
    for i, rec in enumerate(values, start=2):  # sheet rows start at 1 with headers
        if rec.get("teacher_id") == teacher_id:
            row_index = i
            break
    serialized = json.dumps(preferences, ensure_ascii=False)
    if row_index:
        ws.update([[teacher_id, name, email, serialized]], f"A{row_index}:D{row_index}")

        print(f"Updated teacher_profile: {teacher_id}")
    else:
        ws.append_row([teacher_id, name, email, serialized])
        print(f"Inserted teacher_profile: {teacher_id}")



"""
Fully fixed â€” Groq Agent pipeline with bulletproof JSON,
Sheets persistence, and proper curriculum structure handling.
"""

# === Safety Utilities ===
def safe_json_loads(s: str, default=None):
    """
    Safely loads JSON from any messy LLM output.
    1ï¸�âƒ£ Try direct parse
    2ï¸�âƒ£ Try extracting JSON from middle
    """
    if not isinstance(s, str):
        return default

    s = s.strip()
    if not s:
        return default

    try:
        return json.loads(s)
    except:
        pass

    # Try detecting embedded JSON
    for char in ["{", "["]:
        idx = s.find(char)
        if idx != -1:
            try:
                return json.loads(s[idx:])
            except:
                continue

    return default


# === Fallback Template Plans (for safety) ===
def fallback_curriculum(teacher_input):
    weeks = teacher_input.get("weeks", 4)
    subject = teacher_input.get("subject", "Science")
    grade = teacher_input.get("grade_level", "7")
    return {
        "weeks": [
            {
                "week_number": i,
                "classes": [
                    {
                        "class_number": 1,
                        "topic": f"{subject} Key Concepts",
                        "objectives": ["Understand fundamentals"],
                        "activities": ["Teacher demo", "Guided practice"]
                    }
                ]
            }
            for i in range(1, int(weeks) + 1)
        ]
    }

def fallback_lessons(curriculum):
    return [{"msg": "LLM Failed â†’ Placeholder lessons"}]

def fallback_assessments(curriculum):
    return [{"msg": "LLM Failed â†’ Placeholder assessments"}]

def fallback_evaluation():
    return {"overall": "Solid start", "feedback": []}


# === Agent Wrapper (Groq) ===
@dataclass
class Agent:
    name: str
    instructions: str
    model: str = "llama-3.1-8b-instant"

    def call_llm(self, input_data: dict):
        prompt = f"""
You are {self.name}.
{self.instructions}

INPUT DATA:
{json.dumps(input_data, indent=2)}

Respond with valid JSON only. No markdown. No backticks.
"""

        try:
            chat = groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return chat.choices[0].message.content
        except Exception as e:
            print(f"âš ï¸� LLM call failed ({self.name}):", e)
            return ""


    def run(self, input_data: dict):
        # Call model
        raw = self.call_llm(input_data)

        # Debug print
        print(f"--- RAW OUTPUT ({self.name}, first 200 chars) ---")
        print(raw[:200])
        print("----------------------------------------------")

        # Parse JSON safely
        parsed = safe_json_loads(raw)

        return parsed


# === Init Agents ===
planner_agent = Agent("Planner Agent", "Generate curriculum JSON with a `weeks` array")
lesson_agent = Agent("Lesson Agent", "Generate weekly lessons JSON")
assessment_agent = Agent("Assessment Agent", "Generate quizzes JSON")
evaluator_agent = Agent("Evaluator Agent", "Evaluate curriculum JSON")


# === Pipeline with persistence ===
def generate_and_persist(teacher_id: str, teacher_input: dict, ss):
    print("Running Planner Agent...")
    planner_json = planner_agent.run(teacher_input)

    # Handle nested structure: { "curriculum": {...} }
    if planner_json and "curriculum" in planner_json:
        planner_json = planner_json["curriculum"]

    # Fallback if parser or model failure
    if planner_json is None or "weeks" not in planner_json:
        print("âš ï¸� Planner invalid â†’ using fallback curriculum")
        planner_json = fallback_curriculum(teacher_input)


    print("Running Lesson Agent...")
    lesson_json = lesson_agent.run({
        "teacher_input": teacher_input,
        "curriculum": planner_json
    })

    if lesson_json is None:
        print("âš ï¸� Lesson Agent failed â†’ fallback")
        lesson_json = fallback_lessons(planner_json)


    print("Running Assessment Agent...")
    assessment_json = assessment_agent.run({
        "teacher_input": teacher_input,
        "curriculum": planner_json,
        "lessons": lesson_json
    })

    if assessment_json is None:
        print("âš ï¸� Assessment fallback")
        assessment_json = fallback_assessments(planner_json)


    print("Running Evaluator Agent...")
    evaluator_json = evaluator_agent.run({
        "teacher_input": teacher_input,
        "curriculum": planner_json,
        "lessons": lesson_json,
        "assessments": assessment_json
    })

    if evaluator_json is None:
        print("âš ï¸� Evaluation fallback")
        evaluator_json = fallback_evaluation()


    # ==== Persist Everything to Sheets ====
    append_lesson_plan(
        ss,
        teacher_id,
        planner_json,
        lesson_json,
        assessment_json,
        evaluator_json
    )

    print("ğŸ“Œ Saved to Google Sheets successfully.")

    return {
        "curriculum": planner_json,
        "lessons": lesson_json,
        "assessments": assessment_json,
        "evaluation": evaluator_json
    }



# Prepare sheet object
if ss is None and gs_client:
    ss = open_or_create_spreadsheet(gs_client, title="TutoBot-Memory-Kaggle")
    ensure_sheets_exist(ss)

if ss is None:
    raise RuntimeError("Spreadsheet object not available. Check your Sheets setup (Cell C).")

# Example teacher input (same structure as earlier)
teacher_id = "teacher_demo_002"
teacher_input = {
    "subject": "Math",
    "grade": "7",
    "learning_goals": [
        "Understand fractions",
        "Add and subtract fractions",
        "Apply fractions to real life problems"
    ],
    "classes_per_week": 3,
    "student_needs": "Struggling with multiplication"
}


# learning_goals = LEARNING_GOALS_JSON.get(subject, {}).get(str(grade), ["General practice & revision"])



# Upsert teacher profile (optional)
upsert_teacher_profile(ss, teacher_id, name="Mrs. Hazel D'Mello", email="hazel.dmello@gmail.com", preferences={"preferred_output":"printable"})

# Run pipeline
result = generate_and_persist(teacher_id, teacher_input, ss)

# Print a short summary
print("\n--- Curriculum summary (first 2 weeks) ---")
pp.pprint(result["curriculum"].get("weeks", [])[:2])

print("\nSaved to Google Sheets: 'TutoBot-Memory-Kaggle' -> worksheet 'lesson_plans'.")


