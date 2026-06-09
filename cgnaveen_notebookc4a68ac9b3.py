import pandas as pd
import random, uuid, json, re
from dataclasses import dataclass, asdict

random.seed(42)



# Generate synthetic rural candidate profiles

first_names = ["Raju","Sita","Asha","Manoj","Deepa","Kiran","Ramesh","Anita",
               "Sunil","Radha","Priya","Vikram","Geeta","Suresh","Meena",
               "Arjun","Lalitha","Kavita","Rohit","Sowmya"]

last_names = ["Kumar","Reddy","Patel","Sharma","Singh","Nair","Das","Khan",
              "Thakur","Gupta","Iyer","Naik","Varma","Joshi","Chowdhury"]

villages = ["Kurnool","Anantapur","Nellore","Kadapa","Warangal",
            "Vizag","Guntur","Vijayawada","Tirupati","Hyderabad"]

educations = ["10th","12th","Diploma","ITI","B.Sc","B.Com","BA","B.Tech","Polytechnic"]

roles = ["Data Entry Operator","Field Technician","Sales Executive","Customer Support",
         "Junior Accountant","Delivery Executive","Office Assistant","Retail Associate"]

skills_pool = ["MS Office","Basic Excel","Communication","Customer Service","Driving",
               "Two-wheeler repair","Cash Handling","Inventory Management","Telephone Etiquette",
               "Tally","Shopfloor Operations","Basic Python","Google Forms"]


def generate_profiles(n=200):
    rows = []
    for _ in range(n):
        fname = random.choice(first_names)
        lname = random.choice(last_names)
        name = f"{fname} {lname}"
        age = random.randint(18, 35)
        gender = random.choice(["Male","Female","Other"])
        education = random.choice(educations)
        location = random.choice(villages)
        exp_years = random.choice([0,0,0,1,1,2,3,4])
        desired_role = random.choice(roles)
        skills = ", ".join(random.sample(skills_pool, k=random.randint(3,6)))
        baseline_resume = (
            f"{name} from {location}. Education: {education}. "
            f"Experience: {exp_years} years. Skills: {skills}. "
            f"Looking for work as {desired_role}."
        )
        rid = str(uuid.uuid4())[:8]
        rows.append({
            "id": rid,
            "name": name,
            "age": age,
            "gender": gender,
            "education": education,
            "location": location,
            "experience_years": exp_years,
            "desired_role": desired_role,
            "skills": skills,
            "baseline_resume": baseline_resume
        })
    return pd.DataFrame(rows)

profiles_df = generate_profiles()
profiles_df.head()



# A tiny in-notebook "job postings" dataset

job_postings = [
    {
        "id": "j1",
        "title": "Sales Executive",
        "location": "Guntur",
        "description": "Sales executive, customer handling, cash handling, communication, field visits"
    },
    {
        "id": "j2",
        "title": "Data Entry Operator",
        "location": "Vijayawada",
        "description": "Data entry, MS Office, basic excel, accuracy, typing speed"
    },
    {
        "id": "j3",
        "title": "Customer Support Associate",
        "location": "Hyderabad",
        "description": "Customer service, call handling, telephone etiquette, problem solving"
    },
    {
        "id": "j4",
        "title": "Delivery Executive",
        "location": "Nellore",
        "description": "Driving, delivery, customer interaction, cash handling, time management"
    }
]

pd.DataFrame(job_postings)



# ==== Memory Store ====

class MemoryStore:
    """
    Very simple in-memory store plus dict; in a real app this could be a DB.
    """
    def __init__(self):
        self.sessions = {}

    def save_session(self, session_id: str, payload: dict):
        self.sessions[session_id] = payload

    def get_session(self, session_id: str):
        return self.sessions.get(session_id, None)

memory_store = MemoryStore()



# ==== Resume Agent ====

class ResumeAgent:
    """
    Creates a more structured, ATS-friendly resume snippet.
    Uses template logic (no external LLM) so it runs in Kaggle offline.
    """
    def build_resume(self, profile: dict) -> str:
        name = profile.get("name")
        edu = profile.get("education")
        loc = profile.get("location")
        role = profile.get("desired_role")
        skills = profile.get("skills")
        exp = profile.get("experience_years")

        header = f"{name} | {role} | {loc}"
        summary = (
            f"{name} has {exp} year(s) of experience in entry-level roles and "
            f"has completed {edu}. Based in {loc}, interested in working as {role}."
        )

        bullets = []
        for s in skills.split(","):
            s = s.strip()
            bullets.append(f"Hands-on exposure to {s} in practical situations.")

        resume = header + "\n\nSUMMARY\n" + summary + "\n\nSKILLS & EXPERIENCE\n"
        resume += "\n".join(["- " + b for b in bullets])

        return resume



# ==== Job Finder Agent ====

class JobFinderAgent:
    """
    Filters the small in-memory job_postings list based on role and location.
    """
    def __init__(self, jobs):
        self.jobs = jobs

    def find_jobs(self, profile: dict, top_k: int = 3):
        role = profile.get("desired_role", "").lower()
        loc = profile.get("location", "").lower()

        scored = []
        for job in self.jobs:
            score = 0
            title = job["title"].lower()
            desc = job["description"].lower()
            jloc = job["location"].lower()

            if role and role.split()[0] in title:
                score += 3
            if loc and loc == jloc:
                score += 2
            if any(word in desc for word in role.split()):
                score += 1

            if score > 0:
                scored.append((score, job))

        scored.sort(key=lambda x: -x[0])
        return [job for score, job in scored[:top_k]]



# ==== Skill-Gap Agent ====

class SkillGapAgent:
    """
    Compares candidate resume text with job description looking for missing skills.
    This is a simple keyword matcher for demo purposes.
    """
    def extract_job_skills(self, job_desc: str):
        tokens = [t.strip().lower() for t in re.split(r"[,.]", job_desc) if t.strip()]
        return tokens

    def extract_candidate_skills(self, resume_text: str):
        # Look for 'exposure to X' pattern
        skills = set()
        for m in re.findall(r"exposure to ([\\w\\s-]+)", resume_text, flags=re.IGNORECASE):
            skills.add(m.strip().lower())
        # Fallback: simple keywords like 'ms office', 'customer service' etc.
        base_keywords = ["ms office","customer service","cash handling","driving",
                         "basic excel","telephone etiquette","inventory management"]
        for kw in base_keywords:
            if kw in resume_text.lower():
                skills.add(kw)
        return list(skills)

    def compare(self, resume_text: str, job: dict):
        if not job:
            return {"job_skills": [], "candidate_skills": [], "missing": []}
        job_skills = self.extract_job_skills(job["description"])
        cand_skills = self.extract_candidate_skills(resume_text)
        missing = [s for s in job_skills if s not in cand_skills]
        return {
            "job_skills": job_skills,
            "candidate_skills": cand_skills,
            "missing": missing
        }



# ==== Interview Agent ====

class InterviewAgent:
    """
    Simple interview simulation for one question + scoring.
    """
    QBANK = {
        "sales executive": [
            ("Tell me about a time you handled a difficult customer.", 
             ["customer","listen","apolog","solution","follow"]),
        ],
        "data entry operator": [
            ("How do you ensure accuracy when entering data?",
             ["accuracy","double-check","verify","mistake"])
        ]
    }

    def pick_question(self, role: str):
        role_key = role.lower()
        for key, qlist in self.QBANK.items():
            if key in role_key:
                return qlist[0]
        # fallback
        return ("Tell me about yourself and why you want this job.",
                ["experience","skills","interest","job"])

    def score_answer(self, answer: str, expected_keywords=None):
        score = 0
        words = re.findall(r"\\w+", answer)
        # length
        if len(words) >= 30:
            score += 4
        elif len(words) >= 15:
            score += 2
        # keywords
        if expected_keywords:
            found = 0
            for kw in expected_keywords:
                if re.search(rf"\\b{re.escape(kw)}", answer, flags=re.IGNORECASE):
                    found += 1
            score += min(4, found)
        # multiple sentences
        sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
        if len(sentences) >= 2:
            score += 2

        score = min(score, 10)
        feedback = ("Good answer: clear and specific." 
                    if score >= 7 else 
                    "Try to give more detail, examples, and show how you handled the situation.")
        return score, feedback



class Coordinator:
    """
    Orchestrates all agents for a single candidate session.
    """
    def __init__(self, jobs):
        self.resume_agent = ResumeAgent()
        self.job_agent = JobFinderAgent(jobs)
        self.skill_agent = SkillGapAgent()
        self.interview_agent = InterviewAgent()
        self.memory = memory_store

    def run_session(self, profile: dict):
        session_id = profile["id"]

        # 1) Build improved resume
        improved_resume = self.resume_agent.build_resume(profile)

        # 2) Find suitable jobs (top 1 for skill-gap)
        jobs = self.job_agent.find_jobs(profile, top_k=3)
        top_job = jobs[0] if jobs else None

        # 3) Skill gap analysis
        gaps = self.skill_agent.compare(improved_resume, top_job)

        # 4) Prepare interview question
        question, expected_keywords = self.interview_agent.pick_question(
            profile["desired_role"]
        )

        # Save to "memory"
        snapshot = {
            "profile": profile,
            "improved_resume": improved_resume,
            "jobs": jobs,
            "skill_gaps": gaps,
            "interview_question": question
        }
        self.memory.save_session(session_id, snapshot)
        return snapshot, expected_keywords

coordinator = Coordinator(job_postings)



# Pick a sample rural candidate
sample = profiles_df.iloc[0].to_dict()
sample



snapshot, expected_keywords = coordinator.run_session(sample)

print("=== IMPROVED RESUME ===")
print(snapshot["improved_resume"])
print("\n=== JOB RECOMMENDATIONS ===")
for job in snapshot["jobs"]:
    print(f"- {job['title']} @ {job['location']}: {job['description']}")
print("\n=== SKILL GAP ANALYSIS ===")
print("Job skills:       ", snapshot["skill_gaps"]["job_skills"])
print("Candidate skills: ", snapshot["skill_gaps"]["candidate_skills"])
print("Missing skills:   ", snapshot["skill_gaps"]["missing"])
print("\n=== INTERVIEW QUESTION ===")
print(snapshot["interview_question"])



demo_answer = """
In my previous role, I had a customer who was very upset about a delayed delivery.
First, I listened carefully to understand the problem, then I apologized for the delay.
I checked the order status, coordinated with the delivery person and updated the customer.
Finally, I followed up after delivery to make sure they were satisfied.
"""

score, feedback = coordinator.interview_agent.score_answer(
    demo_answer, expected_keywords=expected_keywords
)

print("=== CANDIDATE ANSWER ===")
print(demo_answer.strip())
print("\nScore (0-10):", score)
print("Feedback:", feedback)



results = []
for i in range(20):  # first 20 candidates
    profile = profiles_df.iloc[i].to_dict()
    snapshot, expected = coordinator.run_session(profile)

    # pretend each candidate gives a short generic answer
    generic_answer = (
        "I handled a customer by listening to them, apologising, "
        "and trying to give a proper solution based on their problem."
    )
    score, _ = coordinator.interview_agent.score_answer(generic_answer, expected)
    missing_count = len(snapshot["skill_gaps"]["missing"])

    results.append({
        "id": profile["id"],
        "name": profile["name"],
        "role": profile["desired_role"],
        "interview_score": score,
        "missing_skills": missing_count
    })

eval_df = pd.DataFrame(results)
eval_df.head()



print("Average interview score:", eval_df["interview_score"].mean())
print("Average missing skills:", eval_df["missing_skills"].mean())


