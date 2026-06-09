import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… API key loaded successfully.")
except Exception as e:
    print("â�Œ Error loading API key:", e)



from google.genai import Client
import os

client = Client(api_key=os.environ["GOOGLE_API_KEY"])

def call_model(prompt: str) -> str:
    """Single helper â€“ yahi sab 'agents' ke andar use hoga."""
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    return resp.text

print("âœ… Client + helper ready")



# 1) Career Profile Agent
def career_profile_agent(user_desc: str) -> str:
    prompt = f"""
You are a "Career Profile Analyzer" for students in India.

User description:
{user_desc}

1. Summarize the user's current stage (class/college, branch, skills).
2. Identify 3â€“4 realistic goal directions (e.g., AI engineer, embedded engineer, data scientist, etc.).
3. Return in short bullet points.
"""
    return call_model(prompt)


# 2) Skill Gap Agent
def skill_gap_agent(user_desc: str, target_role: str) -> str:
    prompt = f"""
You are a "Skill Gap Analyst" for the role: {target_role}.

User background:
{user_desc}

1. List required skills for {target_role} (technical + soft skills).
2. Mark which skills the user already has, and which are missing.
3. Prioritize top 5 skills user should learn first.
Return in clean bullet points.
"""
    return call_model(prompt)


# 3) Roadmap Agent
def roadmap_agent(target_role: str, horizon_years: int = 3) -> str:
    prompt = f"""
You are a "Roadmap Planner" for career: {target_role}.

Design a step-by-step roadmap for the next {horizon_years} years:
- Split into phases (0â€“6 months, 6â€“12 months, 1â€“2 years, 2â€“3 years).
- For each phase: topics to learn, projects to build, suggested resources (YouTube type, MOOC type, books).
- Keep it realistic for a middle-class Indian student with limited money and average college.
"""
    return call_model(prompt)


# 4) Exam & College Advisor Agent
def exam_college_agent(target_role: str, country: str = "India") -> str:
    prompt = f"""
You are an "Exam & College Advisor" for students in {country}.

Target role: {target_role}

1. Suggest entrance exams, certifications or important tests relevant to this role (ex: GATE, specific online certs).
2. Suggest types of colleges/programs or online degrees that can help.
3. Add 5â€“6 practical tips on balancing self-study + college + exams.
Return in bullet points.
"""
    return call_model(prompt)


# 5) Motivation / Habit Coach Agent
def motivation_agent(user_desc: str, target_role: str) -> str:
    prompt = f"""
You are a positive, practical "Study & Motivation Coach".

User: {user_desc}
Target role: {target_role}

Give:
1. 5 daily habits (with time suggestion) to move towards the goal.
2. How to avoid procrastination (in Hindi + English mixed, friendly tone).
3. A short 5â€“6 line motivational message directly addressing the user by 'tum'.
"""
    return call_model(prompt)



def career_guidance_orchestrator(user_desc: str, target_role: str):
    print("ğŸ�¯ User goal:", target_role)
    print("\n=== Step 1: Career Profile ===\n")
    profile = career_profile_agent(user_desc)
    print(profile)

    print("\n=== Step 2: Skill Gaps ===\n")
    gaps = skill_gap_agent(user_desc, target_role)
    print(gaps)

    print("\n=== Step 3: Roadmap ===\n")
    roadmap = roadmap_agent(target_role)
    print(roadmap)

    print("\n=== Step 4: Exams & Colleges ===\n")
    exams = exam_college_agent(target_role)
    print(exams)

    print("\n=== Step 5: Motivation & Habits ===\n")
    motivation = motivation_agent(user_desc, target_role)
    print(motivation)

    # Optionally return all text also
    return {
        "profile": profile,
        "gaps": gaps,
        "roadmap": roadmap,
        "exams": exams,
        "motivation": motivation,
    }



user_desc = """
I am Divyanshu from India. I am in IIT Madras BS Electronic Systems Qualifier.
I like maths, electronics and AI. I know some Python and a bit of C.
I want a strong career in AI + Embedded / AI Engineer type role.
"""

target_role = "AI + Embedded Systems Engineer"

result = career_guidance_orchestrator(user_desc, target_role)



# ğŸ”� Example 2 â€“ Run for a Completely New User (Aditi)

user_desc_2 = """
Name: Aditi
Background: Final-year Computer Science student
Interests: AI, Data Science, ML
Goal: Become a Data Scientist at a top tech company
Challenges: Confused about roadmap, doesn't know where to start projects,
            struggling with confidence & procrastination
Preferred Language Style: English only
"""

print("ğŸ”„ Running Example 2 (Aditi)...\n")

# yaha hum target_role bhi pass kar rahe hain, error isi se aa rahi thi
result_2 = career_guidance_orchestrator(
    user_desc_2,
    target_role="Data Scientist at a top tech company"
)

print("âœ… Example 2 Output Generated Successfully!\n")
print(result_2)




