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


# Install required packages
!pip install -q google-adk python-dotenv requests beautifulsoup4


# Import necessary libraries
import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import requests
from bs4 import BeautifulSoup

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Data structures for Sessions & Memory
@dataclass
class UserProfile:
    """User profile for personalized job matching"""
    name: str
    skills: List[str]
    interests: List[str]
    education: str
    location: str
    preferred_companies: List[str] = None
    
@dataclass
class InternshipPosting:
    """Internship posting data structure"""
    job_id: str
    title: str
    company: str
    location: str
    description: str
    required_skills: List[str]
    deadline: str
    url: str
    match_score: float = 0.0
    
@dataclass
class Application:
    """Application tracking data structure"""
    app_id: str
    job_id: str
    company: str
    position: str
    applied_date: str
    status: str  # pending, interview, rejected, accepted
    deadline: str
    notes: str = ""

# In-Memory Session Service for state management
class SessionService:
    """Manages user sessions and application state"""
    def __init__(self):
        self.user_profile = None
        self.internships = []
        self.applications = []
        logger.info("SessionService initialized")
    
    def set_user_profile(self, profile: UserProfile):
        """Store user profile in session"""
        self.user_profile = profile
        logger.info(f"User profile set for {profile.name}")
    
    def add_internship(self, posting: InternshipPosting):
        """Add internship to session memory"""
        self.internships.append(posting)
        logger.info(f"Added internship: {posting.title} at {posting.company}")
    
    def add_application(self, application: Application):
        """Track new application"""
        self.applications.append(application)
        logger.info(f"Application tracked: {application.position} at {application.company}")
    
    def get_matching_internships(self, threshold: float = 0.6) -> List[InternshipPosting]:
        """Get internships above match threshold"""
        return [job for job in self.internships if job.match_score >= threshold]


# Custom Tools for the agents
class JobSearchTool:
    """Custom tool for searching internships"""
    
    def search_internships(self, query: str, location: str = "Remote") -> List[Dict]:
        """Mock search function - in production, would scrape job boards"""
        logger.info(f"Searching internships for: {query} in {location}")
        
        # Mock data - in production, this would scrape LinkedIn, Indeed, etc.
        mock_internships = [
            {
                "job_id": "INT001",
                "title": "AI/ML Intern",
                "company": "Google",
                "location": "Remote",
                "description": "Work on machine learning projects, Python, TensorFlow",
                "required_skills": ["Python", "Machine Learning", "TensorFlow"],
                "deadline": "2025-12-15",
                "url": "https://careers.google.com/jobs/123"
            },
            {
                "job_id": "INT002",
                "title": "Data Science Intern",
                "company": "Microsoft",
                "location": "Hybrid",
                "description": "Analyze data, build models, SQL, Python, Power BI",
                "required_skills": ["Python", "SQL", "Data Analysis", "Power BI"],
                "deadline": "2025-12-20",
                "url": "https://careers.microsoft.com/jobs/456"
            },
            {
                "job_id": "INT003",
                "title": "Cloud Engineering Intern",
                "company": "Amazon",
                "location": "Remote",
                "description": "Work with AWS, build scalable systems",
                "required_skills": ["AWS", "Python", "Cloud Architecture"],
                "deadline": "2025-12-10",
                "url": "https://amazon.jobs/intern/789"
            }
        ]
        
        return mock_internships


# Multi-Agent System Implementation

class SearchAgent:
    """Agent responsible for finding internship postings"""
    
    def __init__(self, session: SessionService):
        self.session = session
        self.search_tool = JobSearchTool()
        logger.info("SearchAgent initialized")
    
    def execute(self, query: str) -> List[InternshipPosting]:
        """Search for internships and add to session"""
        logger.info(f"SearchAgent executing query: {query}")
        
        # Use custom tool to search
        results = self.search_tool.search_internships(query)
        
        # Convert to InternshipPosting objects
        postings = []
        for result in results:
            posting = InternshipPosting(**result)
            postings.append(posting)
            self.session.add_internship(posting)
        
        logger.info(f"SearchAgent found {len(postings)} internships")
        return postings

class AnalyzerAgent:
    """Agent for analyzing and matching jobs to user profile"""
    
    def __init__(self, session: SessionService):
        self.session = session
        logger.info("AnalyzerAgent initialized")
    
    def calculate_match_score(self, job_skills: List[str], user_skills: List[str]) -> float:
        """Calculate skill match percentage"""
        if not job_skills:
            return 0.0
        
        matches = sum(1 for skill in job_skills if skill in user_skills)
        score = matches / len(job_skills)
        return round(score, 2)
    
    def execute(self) -> List[InternshipPosting]:
        """Analyze and rank internships based on user profile"""
        logger.info("AnalyzerAgent analyzing internships")
        
        if not self.session.user_profile:
            logger.warning("No user profile found")
            return []
        
        # Analyze each internship
        for posting in self.session.internships:
            score = self.calculate_match_score(
                posting.required_skills,
                self.session.user_profile.skills
            )
            posting.match_score = score
            logger.info(f"Match score for {posting.title}: {score}")
        
        # Return sorted by match score
        return sorted(self.session.internships, key=lambda x: x.match_score, reverse=True)


# Sequential Agent Orchestrator - demonstrates sequential agent execution
class InternHunterOrchestrator:
    """Main orchestrator that coordinates all agents in sequence"""
    
    def __init__(self):
        # Initialize session service for state management
        self.session = SessionService()
        
        # Initialize all agents
        self.search_agent = SearchAgent(self.session)
        self.analyzer_agent = AnalyzerAgent(self.session)
        
        logger.info("InternHunterOrchestrator initialized with all agents")
    
    def set_user_profile(self, profile: UserProfile):
        """Set user profile for personalized search"""
        self.session.set_user_profile(profile)
    
    def run(self, search_query: str) -> Dict[str, Any]:
        """Execute the full agent pipeline sequentially"""
        logger.info("="*50)
        logger.info("Starting InternHunter Agent System")
        logger.info("="*50)
        
        # Step 1: Search Agent finds internships
        logger.info("\n[Step 1/2] Executing SearchAgent...")
        search_results = self.search_agent.execute(search_query)
        
        # Step 2: Analyzer Agent matches and ranks
        logger.info("\n[Step 2/2] Executing AnalyzerAgent...")
        ranked_results = self.analyzer_agent.execute()
        
        # Prepare results
        top_matches = self.session.get_matching_internships(threshold=0.5)
        
        logger.info("\n" + "="*50)
        logger.info(f"Pipeline complete! Found {len(top_matches)} matching internships")
        logger.info("="*50)
        
        return {
            "total_found": len(search_results),
            "top_matches": top_matches,
            "all_results": ranked_results
        }


# Demo Execution - Test the InternHunter Agent System
print("\n" + "="*70)
print("INTERNHUNTER AI AGENT DEMO")
print("="*70)

# Create user profile (simulating a high school student looking for internships)
user = UserProfile(
    name="Student",
    skills=["Python", "Machine Learning", "SQL", "Data Analysis"],
    interests=["AI/ML", "Data Science", "Cloud Computing"],
    education="High School Final Year",
    location="India",
    preferred_companies=["Google", "Microsoft", "Amazon"]
)

print(f"\nUser Profile:")
print(f"  Name: {user.name}")
print(f"  Skills: {', '.join(user.skills)}")
print(f"  Interests: {', '.join(user.interests)}")
print(f"  Education: {user.education}")

# Initialize the orchestrator
orchestrator = InternHunterOrchestrator()
orchestrator.set_user_profile(user)

# Run the agent system
print("\n" + "="*70)
results = orchestrator.run("AI ML Data Science internships")
print("="*70)

# Display results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nTotal internships found: {results['total_found']}")
print(f"Top matches (>50% skill match): {len(results['top_matches'])}")

print("\n" + "-"*70)
print("TOP MATCHING INTERNSHIPS:")
print("-"*70)

for i, job in enumerate(results['top_matches'], 1):
    print(f"\n{i}. {job.title} at {job.company}")
    print(f"   Match Score: {job.match_score*100:.0f}%")
    print(f"   Location: {job.location}")
    print(f"   Required Skills: {', '.join(job.required_skills)}")
    print(f"   Deadline: {job.deadline}")
    print(f"   URL: {job.url}")

print("\n" + "="*70)
print("DEMO COMPLETE!")
print("="*70)

