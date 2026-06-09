import re
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import json

# =======================================================
# 1. SETUP: MOCK DATA (Inputs)
# =======================================================

BASE_RESUME_TEXT = """
Name: Jane Doe | Phone: (555) 123-4567
Summary: Dedicated Python developer with 5 years experience in web applications and database management. Proven track record of improving application performance by 15%. Seeking new challenges.
Experience:
- Senior Developer at TechCorp (2020-Present): Developed REST APIs using Flask/SQLAlchemy. Managed cloud deployments on AWS. Reduced latency by 20%.
- Developer at StartupX (2018-2020): Built front-end components using React. Participated in daily standups and sprint planning.
Skills: Python, Flask, SQL, AWS, JavaScript, HTML, CSS.
"""

JOB_DESCRIPTION_TEXT = """
Title: Senior Backend Engineer - Python/Django
We are seeking a Senior Backend Engineer proficient in Python, **Django**, and **PostgreSQL**.
Must have experience with microservices architecture and **unit testing** (pytest).
Responsibilities include optimizing database queries and contributing to CI/CD pipelines.
Keywords: Python, Django, PostgreSQL, microservices, unit testing, optimization, CI/CD.
"""

# =======================================================
# 2. TOOL DEFINITIONS (The Agent's Skills)
# =======================================================

class CustomTools:
    """
    Encapsulates the specialized skills (tools) used by the agents.
    These tools provide objective, quantitative data to the reasoning agents.
    """

    @staticmethod
    def extract_keywords(text: str) -> list:
        """
        Tool 1: Extracts key technical terms and concepts using regex and common tech terms.
        """
        # A robust regex/set for common tech terms, simulating an advanced NLP model.
        keywords = set(re.findall(
            r'[A-Z][a-z]+|Python|Django|SQL|AWS|React|CI/CD|PostgreSQL|microservices|unit testing|pytest|Flask|SQLAlchemy',
            text
        ))
        return list(keywords)

    @staticmethod
    def calculate_match_score(resume_keywords: list, jd_keywords: list) -> tuple:
        """
        Tool 2: Calculates the keyword overlap and identifies missing terms.
        This emulates the ATS (Applicant Tracking System) score.
        """
        jd_set = set(jd_keywords)
        resume_set = set(resume_keywords)
        overlap = jd_set.intersection(resume_set)
        
        # Simple overlap percentage score
        score = (len(overlap) / len(jd_set)) * 100 if len(jd_set) > 0 else 0
        
        missing = jd_set - resume_set
        
        return round(score, 2), list(missing)

# =======================================================
# 3. THE SEQUENTIAL MULTI-AGENT SYSTEM
# =======================================================

class BaseAgent:
    """Base class providing common logging functionality (Observability)."""
    
    def log(self, prefix: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp} | {prefix}] {message}")

class AnalystAgent(BaseAgent):
    """Agent 1: Extracts and analyzes the core components."""
    
    def run(self, resume: str, job_description: str) -> dict:
        self.log("ANALYST �洫申", "Starting analysis of job requirements and base resume.")
        
        # 1. Use Tool 1 to extract keywords
        resume_kws = CustomTools.extract_keywords(resume)
        jd_kws = CustomTools.extract_keywords(job_description)
        
        # 2. Use Tool 2 to calculate the initial score
        match_score, missing_kws = CustomTools.calculate_match_score(resume_kws, jd_kws)

        # Extract current summary text for revision
        summary_match = re.search(r'Summary: (.*?)Experience:', resume, re.DOTALL)
        current_summary = summary_match.group(1).strip() if summary_match else ""
        
        # Pass data forward (Session State/Memory)
        analysis_report = {
            "initial_score": match_score,
            "missing_keywords": missing_kws,
            "target_keywords": jd_kws,
            "current_summary": current_summary
        }
        
        self.log("ANALYST �洫申", f"Initial Match Score: {analysis_report['initial_score']}% (Missing: {len(missing_kws)})")
        return analysis_report

class RevisionAgent(BaseAgent):
    """Agent 2: Rewrites sections based on the Analyst's report."""
    
    def run(self, analysis_report: dict, base_resume: str) -> str:
        self.log("REVISION 笨搾ｸ十", "Starting resume revision using LLM emulation.")
        
        # --- LLM EMULATION: Reasoning and Text Generation ---
        # In a real agent, this section would be an LLM API call (e.g., GPT-4)
        # prompted with the analysis_report.
        
        revised_resume = base_resume
        missing = analysis_report['missing_keywords']
        new_summary = analysis_report['current_summary']
        
        # Rule 1: Inject primary missing framework (Django) into the summary.
        if "Django" in missing:
            self.log("REVISION 笨搾ｸ十", "Tailoring summary: Injecting 'Django' expertise.")
            new_summary = new_summary.replace(
                "Flask/SQLAlchemy.", "Python/Django framework. Expertise in RESTful API development and database optimization."
            )
            # Update the resume string with the new summary
            revised_resume = revised_resume.replace(analysis_report['current_summary'], new_summary)


        # Rule 2: Inject primary missing methodology (unit testing) into experience.
        if "unit testing" in missing:
            self.log("REVISION 笨搾ｸ十", "Tailoring experience: Adding 'unit testing' and 'PostgreSQL' context.")
            # Find the most relevant experience bullet point to revise
            revised_resume = revised_resume.replace(
                "Reduced latency by 20%.", 
                "Reduced latency by 20%. Implemented **pytest** for **unit testing** coverage and optimized **PostgreSQL** queries."
            )

        self.log("REVISION 笨搾ｸ十", "Revision complete. Passed new version to Manager for validation.")
        return revised_resume

class ManagerAgent(BaseAgent):
    """Agent 3 (Finalizer): Verifies the changes and provides a final log/score (Evaluation)."""
    
    def run(self, revised_resume: str, job_description: str) -> dict:
        self.log("MANAGER 笨�", "Starting final evaluation and synthesis.")
        
        # 1. Rerun the Analyst's core logic on the final output
        final_kws = CustomTools.extract_keywords(revised_resume)
        jd_kws = CustomTools.extract_keywords(job_description)

        final_score, final_missing = CustomTools.calculate_match_score(final_kws, jd_kws)
        
        # Calculate improvement (Memory required the initial score which is fetched by re-running initial analysis)
        initial_score, _ = CustomTools.calculate_match_score(
            CustomTools.extract_keywords(BASE_RESUME_TEXT), jd_kws
        )
        
        improvement = final_score - initial_score
        
        final_report = {
            "initial_score": initial_score,
            "final_score": final_score,
            "score_improvement": round(improvement, 2),
            "final_missing": final_missing,
            "revised_resume": revised_resume
        }
        
        self.log("MANAGER 笨�", f"Final Score: {final_report['final_score']}%. Improvement: +{final_report['score_improvement']} points.")
        return final_report

# =======================================================
# 4. EXECUTION WORKFLOW (Orchestration)
# =======================================================

def run_auto_cv_agent(base_resume: str, job_description: str):
    """Orchestrates the sequential flow of the three agents."""
    print("\n" + "="*50)
    print("      --- Auto-CV Agent Initiated ---")
    print("="*50 + "\n")
    
    # 1. Analyst Agent runs (Input: Raw data)
    analysis_report = AnalystAgent().run(base_resume, job_description)
    
    print("-" * 50)
    
    # 2. Revision Agent runs (Input: Analysis Report + Base Resume)
    revised_resume = RevisionAgent().run(analysis_report, base_resume)
    
    print("-" * 50)

    # 3. Manager Agent runs (Input: Revised Resume + JD for final scoring)
    final_report = ManagerAgent().run(revised_resume, job_description)
    
    print("\n" + "="*50)
    print("        --- FINAL ASSESSMENT ---")
    print("="*50)
    print(f"Initial Keyword Match Score: {final_report['initial_score']}%")
    print(f"**Final Tailored Match Score:** {final_report['final_score']}%")
    print(f"Score Improvement: +{final_report['score_improvement']} percentage points")
    print(f"Still Missing Keywords: {final_report['final_missing']}")
    
    print("\n--- REVISED RESUME (Output for Submission) ---")
    print("-------------------------------------------------")
    print(final_report['revised_resume'].strip())
    print("-------------------------------------------------")
    
    return final_report

# --- RUN THE AGENT ---
if __name__ == '__main__':
    # We call the main orchestration function
    report = run_auto_cv_agent(BASE_RESUME_TEXT, JOB_DESCRIPTION_TEXT)

