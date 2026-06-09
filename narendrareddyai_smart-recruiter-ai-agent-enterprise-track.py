# Install required packages
!pip install -q google-generativeai PyPDF2

import os
import json
import re
from datetime import datetime
from typing import Dict, List
import google.generativeai as genai

# Configure Gemini API (you'll need to add your API key in secrets)
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
except:
    print("Note: Add GOOGLE_API_KEY to Kaggle Secrets for full functionality")
    print("This demo will use simulated responses")

print("âœ… Environment setup complete!")


# Agent Base Class and Multi-Agent Orchestrator

class BaseAgent:
    """Base class for all agents with logging and metrics"""
    def __init__(self, name: str):
        self.name = name
        self.execution_log = []
        
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {self.name}: {message}"
        self.execution_log.append(log_entry)
        print(log_entry)
        
class CVParserAgent(BaseAgent):
    """Agent 1: Parses CV and extracts structured information"""
    def __init__(self):
        super().__init__("CV Parser")
        
    def parse_cv(self, cv_text: str) -> Dict:
        self.log("Parsing CV...")
        
        # Simulated parsing (in production, would use NLP/OCR)
        skills = re.findall(r'\b(?:Python|Java|JavaScript|React|Node\.js|AWS|Docker|Kubernetes|SQL|MongoDB|Machine Learning|AI|Data Science)\b', cv_text, re.IGNORECASE)
        years_exp_match = re.search(r'(\d+)\+?\s*years?\s*(?:of\s*)?experience', cv_text, re.IGNORECASE)
        years_exp = int(years_exp_match.group(1)) if years_exp_match else 0
        
        education_match = re.search(r'\b(B\.?Tech|M\.?Tech|Bachelor|Master|PhD)\b.*?(Computer Science|Engineering|CS|CSE)', cv_text, re.IGNORECASE)
        education = education_match.group(0) if education_match else "Not specified"
        
        parsed_data = {
            "skills": list(set([s.title() for s in skills])),
            "years_experience": years_exp,
            "education": education,
            "total_skills": len(set(skills))
        }
        
        self.log(f"Extracted {len(parsed_data['skills'])} skills, {years_exp} years experience")
        return parsed_data

class SkillMatcherAgent(BaseAgent):
    """Agent 2: Matches candidate skills with job requirements"""
    def __init__(self):
        super().__init__("Skill Matcher")
        
    def match_skills(self, candidate_data: Dict, job_requirements: Dict) -> Dict:
        self.log("Matching candidate skills with job requirements...")
        
        required_skills = set([s.lower() for s in job_requirements.get('required_skills', [])])
        candidate_skills = set([s.lower() for s in candidate_data.get('skills', [])])
        
        matched_skills = required_skills.intersection(candidate_skills)
        missing_skills = required_skills.difference(candidate_skills)
        
        match_score = (len(matched_skills) / len(required_skills) * 100) if required_skills else 0
        
        # Experience match
        exp_required = job_requirements.get('min_experience', 0)
        exp_candidate = candidate_data.get('years_experience', 0)
        exp_match = min(exp_candidate / max(exp_required, 1), 1.0) * 100
        
        overall_score = (match_score * 0.7 + exp_match * 0.3)
        
        result = {
            "matched_skills": list(matched_skills),
            "missing_skills": list(missing_skills),
            "skill_match_percentage": round(match_score, 2),
            "experience_match_percentage": round(exp_match, 2),
            "overall_score": round(overall_score, 2),
            "recommendation": "Strong Match" if overall_score >= 70 else "Moderate Match" if overall_score >= 50 else "Weak Match"
        }
        
        self.log(f"Overall match score: {result['overall_score']}% - {result['recommendation']}")
        return result

print("âœ… Agent classes defined successfully!")


# Multi-Agent Orchestrator and Demo

class RecruitmentOrchestrator:
    """Coordinates all agents in the recruitment pipeline"""
    def __init__(self):
        self.cv_parser = CVParserAgent()
        self.skill_matcher = SkillMatcherAgent()
        self.execution_log = []
        
    def process_candidate(self, cv_text: str, job_requirements: Dict) -> Dict:
        """Main orchestration method - runs all agents sequentially"""
        print("âœ¨" * 40)
        print("      SMART RECRUITER AI AGENT - PROCESSING CANDIDATE")
        print("âœ¨" * 40 + "\n")
        
        # Step 1: Parse CV
        parsed_data = self.cv_parser.parse_cv(cv_text)
        print(f"\nğŸ“‹ Parsed CV Data:\n{json.dumps(parsed_data, indent=2)}\n")
        
        # Step 2: Match skills
        match_result = self.skill_matcher.match_skills(parsed_data, job_requirements)
        print(f"ğŸ�¯ Skill Matching Results:\n{json.dumps(match_result, indent=2)}\n")
        
        # Compile final result
        final_result = {
            "candidate_profile": parsed_data,
            "job_match": match_result,
            "next_steps": self._determine_next_steps(match_result),
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "âœ…" * 40)
        print(f"      PROCESSING COMPLETE - {match_result['recommendation']}")
        print("âœ…" * 40)
        
        return final_result
    
    def _determine_next_steps(self, match_result: Dict) -> List[str]:
        """Determines next actions based on match score"""
        score = match_result['overall_score']
        
        if score >= 70:
            return [
                "Schedule technical interview",
                "Send coding challenge",
                "Notify hiring manager"
            ]
        elif score >= 50:
            return [
                "Conduct phone screening",
                "Review portfolio/projects",
                "Request additional information"
            ]
        else:
            return [
                "Send rejection email with feedback",
                "Add to talent pool for future opportunities"
            ]

print("âœ… Orchestrator created successfully!")


# Demo: Process a Sample Candidate

# Sample CV text
sample_cv = """
John Smith
Senior Full-Stack Developer

EXPERIENCE:
- 5 years of experience in full-stack development
- Led development of microservices architecture using Python, Node.js, and React
- Implemented CI/CD pipelines with Docker and Kubernetes
- Database expertise: MongoDB, SQL
- Cloud platforms: AWS

EDUCATION:
B.Tech Computer Science Engineering

SKILLS:
Python, JavaScript, React, Node.js, Docker, Kubernetes, AWS, MongoDB, SQL, 
Machine Learning, Data Science, Java
"""

# Job requirements
job_requirements = {
    "title": "Senior Full-Stack Engineer",
    "required_skills": ["Python", "JavaScript", "React", "Node.js", "Docker", "AWS", "SQL"],
    "min_experience": 4,
    "nice_to_have": ["Kubernetes", "MongoDB", "Machine Learning"]
}

# Create orchestrator and process candidate
orchestrator = RecruitmentOrchestrator()
result = orchestrator.process_candidate(sample_cv, job_requirements)

# Display next steps
print("\nğŸš€ RECOMMENDED NEXT STEPS:")
for i, step in enumerate(result['next_steps'], 1):
    print(f"   {i}. {step}")

print("\n\nğŸ�‰ Demo Complete! The Smart Recruiter AI Agent successfully processed the candidate.")

