# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Hiring & Talent Development Agent

This notebook is a **toy prototype** of a multi-agent "Hiring & Talent Development Agent".

We will:

- Simulate candidate data
- Implement simple "agents":
  - IntakeAgent
  - ScreeningAgent
  - EvaluationAgent
  - SchedulerAgent
  - DevelopmentAgent
- Run a full pipeline:
  1. Load candidates
  2. Screen them
  3. Give them a simple evaluation
  4. "Schedule" interviews
  5. Generate a basic development plan for hired candidates

This is not production code â€“ it's a **conceptual, runnable demo** for experimentation and extension.



import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



random.seed(42)
np.random.seed(42)

roles = ["Backend Engineer", "Data Scientist", "Frontend Engineer"]
skills_pool = {
    "Backend Engineer": ["Python", "Django", "SQL", "APIs", "Docker"],
    "Data Scientist": ["Python", "Pandas", "ML", "SQL", "Statistics"],
    "Frontend Engineer": ["JavaScript", "React", "HTML", "CSS", "TypeScript"],
}
locations = ["Bangalore", "Remote", "Mumbai", "SF", "London"]

def generate_candidate(idx: int) -> Dict[str, Any]:
    role = random.choice(roles)
    skills = random.sample(skills_pool[role], k=random.randint(2, 5))
    years_exp = np.round(np.random.uniform(0.5, 8.0), 1)
    expected_salary = int(np.random.uniform(6, 35) * 1e5)  # example: 600,000+
    return {
        "candidate_id": idx,
        "name": f"Candidate_{idx}",
        "role_applied": role,
        "years_experience": years_exp,
        "skills": skills,
        "expected_salary": expected_salary,
        "location": random.choice(locations),
    }

candidates_raw = [generate_candidate(i) for i in range(1, 21)]
candidates_df = pd.DataFrame(candidates_raw)
candidates_df


job_spec = {
    "role_title": "Backend Engineer",
    "min_experience": 2.0,
    "must_have_skills": ["Python", "SQL"],
    "nice_to_have_skills": ["Django", "APIs", "Docker"],
    "min_budget": 800000,   # in whatever currency
    "max_budget": 2500000,
}

job_spec


@dataclass
class CandidateProfile:
    candidate_id: int
    name: str
    role_applied: str
    years_experience: float
    skills: List[str]
    expected_salary: int
    location: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreeningResult:
    candidate: CandidateProfile
    score: float
    decision: str
    reasons: List[str]


@dataclass
class EvaluationResult:
    candidate: CandidateProfile
    task_description: str
    score: float
    passed: bool
    feedback: str


@dataclass
class InterviewSlot:
    candidate: CandidateProfile
    interviewer: str
    slot_time: str
    mode: str  # e.g. "Online", "Onsite"


@dataclass
class DevelopmentPlan:
    candidate: CandidateProfile
    strengths: List[str]
    improvement_areas: List[str]
    recommended_actions: List[str]



class IntakeAgent:
    """Simulates the intake & parsing agent."""
    
    def __init__(self):
        self.candidates: Dict[int, CandidateProfile] = {}
    
    def load_from_dataframe(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            profile = CandidateProfile(
                candidate_id=row["candidate_id"],
                name=row["name"],
                role_applied=row["role_applied"],
                years_experience=row["years_experience"],
                skills=row["skills"],
                expected_salary=row["expected_salary"],
                location=row["location"],
            )
            self.candidates[profile.candidate_id] = profile
    
    def get_all_candidates(self) -> List[CandidateProfile]:
        return list(self.candidates.values())


intake_agent = IntakeAgent()
intake_agent.load_from_dataframe(candidates_df)
len(intake_agent.get_all_candidates())



class ScreeningAgent:
    """Scores candidate fit vs job spec."""

    def __init__(self, job_spec: Dict[str, Any]):
        self.job_spec = job_spec

    def score_candidate(self, candidate: CandidateProfile) -> ScreeningResult:
        reasons = []
        score = 0.0
        
        # Role match
        if candidate.role_applied == self.job_spec["role_title"]:
            score += 20
        else:
            reasons.append(f"Role mismatch: applied for {candidate.role_applied}")
        
        # Experience
        if candidate.years_experience >= self.job_spec["min_experience"]:
            score += 20
        else:
            reasons.append(f"Low experience: {candidate.years_experience} yrs")
        
        # Budget fit
        if self.job_spec["min_budget"] <= candidate.expected_salary <= self.job_spec["max_budget"]:
            score += 20
        else:
            reasons.append(f"Salary {candidate.expected_salary} outside budget")
        
        # Skills
        skills = set(candidate.skills)
        must_have = set(self.job_spec["must_have_skills"])
        nice_to_have = set(self.job_spec["nice_to_have_skills"])
        
        must_matches = len(skills & must_have)
        nice_matches = len(skills & nice_to_have)
        
        score += must_matches * 15  # each must-have is valuable
        score += nice_matches * 5
        
        if must_matches < len(must_have):
            reasons.append("Missing some must-have skills")
        
        # Simple decision thresholds
        if score >= 70:
            decision = "strong_pass"
        elif 50 <= score < 70:
            decision = "borderline_pass"
        else:
            decision = "reject"
            if not reasons:
                reasons.append("Overall fit is low")
        
        return ScreeningResult(candidate=candidate, score=score, decision=decision, reasons=reasons)

    def batch_screen(self, candidates: List[CandidateProfile]) -> List[ScreeningResult]:
        return [self.score_candidate(c) for c in candidates]


screening_agent = ScreeningAgent(job_spec=job_spec)
screening_results = screening_agent.batch_screen(intake_agent.get_all_candidates())

# Show top 10 by score
screen_df = pd.DataFrame([
    {
        "candidate_id": r.candidate.candidate_id,
        "name": r.candidate.name,
        "role_applied": r.candidate.role_applied,
        "score": r.score,
        "decision": r.decision,
        "reasons": "; ".join(r.reasons),
    }
    for r in screening_results
]).sort_values(by="score", ascending=False)

screen_df.head(10)



class EvaluationAgent:
    """
    Simulates a skill evaluation task.
    In a real system, this would:
    - generate a coding problem, writing task, case study, etc.
    - evaluate solution using code execution / LLMs.
    Here we just simulate with some randomness + skill match.
    """
    
    def __init__(self, job_spec: Dict[str, Any]):
        self.job_spec = job_spec
    
    def generate_task(self, candidate: CandidateProfile) -> str:
        # Very simple task description
        return textwrap.dedent(f"""
        Implement a simple API in {', '.join(self.job_spec['must_have_skills'])}
        that supports CRUD operations for a 'User' resource.
        """).strip()
    
    def evaluate(self, candidate: CandidateProfile) -> EvaluationResult:
        task_description = self.generate_task(candidate)
        
        # Simulated performance: base on number of must-have and nice-to-have skills + some noise
        skills = set(candidate.skills)
        must_have = set(self.job_spec["must_have_skills"])
        nice_to_have = set(self.job_spec["nice_to_have_skills"])
        
        base_score = 40 + 10 * len(skills & must_have) + 5 * len(skills & nice_to_have)
        noise = np.random.normal(0, 10)
        score = max(0, min(100, base_score + noise))
        
        passed = score >= 60
        feedback = (
            "Great job, strong fundamentals."
            if passed else
            "Needs improvement on core backend concepts."
        )
        
        return EvaluationResult(
            candidate=candidate,
            task_description=task_description,
            score=score,
            passed=passed,
            feedback=feedback,
        )

    def batch_evaluate(self, candidates: List[CandidateProfile]) -> List[EvaluationResult]:
        return [self.evaluate(c) for c in candidates]


# We only evaluate candidates who were not rejected
shortlisted_candidates = [
    r.candidate for r in screening_results if r.decision != "reject"
]

eval_agent = EvaluationAgent(job_spec=job_spec)
evaluation_results = eval_agent.batch_evaluate(shortlisted_candidates)

eval_df = pd.DataFrame([
    {
        "candidate_id": r.candidate.candidate_id,
        "name": r.candidate.name,
        "eval_score": r.score,
        "passed": r.passed,
        "feedback": r.feedback,
    }
    for r in evaluation_results
]).sort_values(by="eval_score", ascending=False)

eval_df



class SchedulerAgent:
    """
    Simulates interview scheduling.
    In reality this would integrate with Google/Outlook calendar APIs.
    """
    
    def __init__(self, interviewer_pool: Optional[List[str]] = None):
        if interviewer_pool is None:
            interviewer_pool = ["Interviewer_A", "Interviewer_B", "Interviewer_C"]
        self.interviewers = interviewer_pool
    
    def schedule(self, candidate: CandidateProfile) -> InterviewSlot:
        interviewer = random.choice(self.interviewers)
        # Just pick a simple pseudo-slot
        slot_time = f"2025-01-{random.randint(10, 20)} 10:{random.choice(['00','30'])}"
        mode = random.choice(["Online", "Onsite"])
        return InterviewSlot(
            candidate=candidate,
            interviewer=interviewer,
            slot_time=slot_time,
            mode=mode,
        )

    def batch_schedule(self, candidates: List[CandidateProfile]) -> List[InterviewSlot]:
        return [self.schedule(c) for c in candidates]


# Only schedule for candidates who passed evaluation
to_schedule = [
    r.candidate for r in evaluation_results if r.passed
]

scheduler_agent = SchedulerAgent()
interview_slots = scheduler_agent.batch_schedule(to_schedule)

schedule_df = pd.DataFrame([
    {
        "candidate_id": s.candidate.candidate_id,
        "name": s.candidate.name,
        "interviewer": s.interviewer,
        "slot_time": s.slot_time,
        "mode": s.mode,
    }
    for s in interview_slots
])

schedule_df



class DevelopmentAgent:
    """
    Creates a simple development plan based on:
    - Job spec
    - Candidate's current skills
    - Evaluation feedback
    """
    
    def __init__(self, job_spec: Dict[str, Any]):
        self.job_spec = job_spec
    
    def build_plan(
        self, 
        candidate: CandidateProfile, 
        screening_result: ScreeningResult, 
        evaluation_result: EvaluationResult
    ) -> DevelopmentPlan:
        
        skills = set(candidate.skills)
        must_have = set(self.job_spec["must_have_skills"])
        nice_to_have = set(self.job_spec["nice_to_have_skills"])
        
        strengths = list(skills & (must_have | nice_to_have))
        missing_must = list(must_have - skills)
        missing_nice = list(nice_to_have - skills)
        
        improvement_areas = missing_must + missing_nice
        
        recommended_actions = []
        for skill in missing_must:
            recommended_actions.append(f"Complete a structured course on {skill}.")
        for skill in missing_nice:
            recommended_actions.append(f"Practice {skill} through side projects.")
        
        if not recommended_actions:
            recommended_actions.append("Take on a stretch project leading a backend feature end-to-end.")
        
        return DevelopmentPlan(
            candidate=candidate,
            strengths=strengths,
            improvement_areas=improvement_areas,
            recommended_actions=recommended_actions,
        )
    
    def build_for_hired(
        self,
        hired_candidates: List[CandidateProfile],
        screening_results: List[ScreeningResult],
        evaluation_results: List[EvaluationResult],
    ) -> List[DevelopmentPlan]:
        # Index results by candidate_id for quick lookup
        screen_map = {r.candidate.candidate_id: r for r in screening_results}
        eval_map = {r.candidate.candidate_id: r for r in evaluation_results}
        
        plans = []
        for c in hired_candidates:
            s_res = screen_map[c.candidate_id]
            e_res = eval_map[c.candidate_id]
            plans.append(self.build_plan(c, s_res, e_res))
        return plans


# Let's assume "hired" = passed eval and had strong_pass or borderline_pass
hired_candidates = to_schedule  # from earlier step

dev_agent = DevelopmentAgent(job_spec=job_spec)
dev_plans = dev_agent.build_for_hired(
    hired_candidates=hired_candidates,
    screening_results=screening_results,
    evaluation_results=evaluation_results,
)

# Show one example plan
example_plan = dev_plans[0]
example_plan


