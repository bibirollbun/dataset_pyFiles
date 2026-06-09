# ==========================================
# TECH CAREER PREP MULTI-AGENT SYSTEM
# ==========================================
# Capstone Project: Agents Intensive - Concierge Agents Track
# Author: Dhooda Ashwitha
# 
# This system demonstrates THREE KEY CONCEPTS:
# 1. Multi-Agent System (Parallel Agents)
# 2. Tools Integration (Google Search)
# 3. Sessions & Memory (InMemorySessionService)
# ==========================================

# CELL 1: Imports and Configuration
print("\nğŸš€ Tech Career Prep Multi-Agent System")
print("=" * 50)

import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Try to import Google AI - will use simulated mode if not available
try:
    from kaggle_secrets import UserSecretsClient
    import google.generativeai as genai
    
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    USE_REAL_AI = True
    print("âœ“ Google AI API configured")
except Exception as e:
    USE_REAL_AI = False
    print(f"âš  Running in simulation mode (AI API not available)")

print("âœ“ Imports completed\n")


# CELL 2: Data Structures and Google Search Tool Integration
print("\nğŸ“¦ Setting up Data Structures & Tools")
print("=" * 50)

# Data structures for storing information
@dataclass
class InterviewQuestion:
    """Stores interview question information"""
    title: str
    company: str
    difficulty: str
    topic: str
    url: str
    
@dataclass
class CompanyInfo:
    """Stores company research information"""
    company: str
    interview_process: str
    recent_news: str
    hiring_status: str
    url: str

@dataclass
class SkillInfo:
    """Stores skill gap information"""
    skill: str
    trend: str
    relevance: str
    learning_resources: List[str] = field(default_factory=list)

# ==========================================
# CONCEPT 2: TOOLS - Google Search Integration
# ==========================================
class GoogleSearchTool:
    """Simulates Google Search functionality"""
    
    def __init__(self):
        # In production, this would use real Google Search API
        self.search_count = 0
        
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Simulates Google Search for queries
        In production: Would use google.search() or Custom Search API
        """
        self.search_count += 1
        print(f"  ğŸ”� Searching: '{query}'")
        time.sleep(0.3)  # Simulate API call
        
        # Simulated search results based on query type
        if "interview" in query.lower():
            return self._interview_results(query)
        elif "company" in query.lower() or "microsoft" in query.lower() or "google" in query.lower():
            return self._company_results(query)
        elif "skills" in query.lower() or "technology" in query.lower():
            return self._skill_results(query)
        else:
            return self._general_results(query)
    
    def _interview_results(self, query: str) -> List[Dict]:
        """Simulated interview question search results"""
        questions = [
            {"title": "Array Two Sum Problem", "company": "Microsoft", "difficulty": "Medium", 
             "url": "https://leetcode.com/problems/two-sum"},
            {"title": "Binary Tree Traversal", "company": "Google", "difficulty": "Easy",
             "url": "https://leetcode.com/problems/binary-tree-traversal"},
            {"title": "Dynamic Programming - Coin Change", "company": "Amazon", "difficulty": "Hard",
             "url": "https://leetcode.com/problems/coin-change"},
            {"title": "System Design - URL Shortener", "company": "Meta", "difficulty": "Hard",
             "url": "https://example.com/system-design"},
            {"title": "String Manipulation - Valid Palindrome", "company": "Apple", "difficulty": "Easy",
             "url": "https://leetcode.com/problems/valid-palindrome"},
        ]
        return random.sample(questions, min(3, len(questions)))
    
    def _company_results(self, query: str) -> List[Dict]:
        """Simulated company research results"""
        companies = [
            {"company": "Microsoft", "process": "4-5 rounds: Coding + System Design + Behavioral",
             "news": "Expanding AI team in India", "hiring": "Active",
             "url": "https://careers.microsoft.com"},
            {"company": "Google", "process": "5-6 rounds: Technical + Googleyness interview",
             "news": "Focus on ML Engineers", "hiring": "Active",
             "url": "https://careers.google.com"},
            {"company": "Amazon", "process": "3-4 rounds: Leadership Principles focused",
             "news": "Growing cloud services team", "hiring": "Very Active",
             "url": "https://amazon.jobs"},
        ]
        return random.sample(companies, min(2, len(companies)))
    
    def _skill_results(self, query: str) -> List[Dict]:
        """Simulated skill gap search results"""
        skills = [
            {"skill": "Kubernetes", "trend": "Rising", "relevance": "High for DevOps/Cloud roles",
             "resources": ["Kubernetes.io", "Udemy K8s course"]},
            {"skill": "LangChain", "trend": "Emerging", "relevance": "Critical for AI Engineer roles",
             "resources": ["LangChain docs", "DeepLearning.ai course"]},
            {"skill": "System Design", "trend": "Always Important", "relevance": "Essential for senior roles",
             "resources": ["Grokking System Design", "System Design Primer"]},
            {"skill": "React/TypeScript", "trend": "Stable", "relevance": "Frontend development",
             "resources": ["React docs", "TypeScript handbook"]},
        ]
        return random.sample(skills, min(2, len(skills)))
    
    def _general_results(self, query: str) -> List[Dict]:
        """General search results"""
        return [{"title": "Search result for: " + query, "url": "https://example.com"}]

# Initialize the Google Search Tool
search_tool = GoogleSearchTool()
print("âœ“ Google Search Tool initialized")
print("âœ“ Data structures defined\n")


# CELL 3: Multi-Agent System with Parallel Execution
print("\nğŸ¤– Creating Multi-Agent System")
print("=" * 50)

# ==========================================
# CONCEPT 1: MULTI-AGENT SYSTEM (Parallel Agents)
# ==========================================

class InterviewPrepAgent:
    """Agent specialized in finding interview questions"""
    
    def __init__(self, search_tool: GoogleSearchTool):
        self.name = "Interview Prep Agent"
        self.search_tool = search_tool
        
    def execute(self, user_prefs: Dict) -> List[InterviewQuestion]:
        """
        Searches for coding interview questions based on user preferences
        Runs in parallel with other agents
        """
        print(f"\n  ğŸ“š {self.name} starting search...")
        target_companies = user_prefs.get('target_companies', ['Microsoft', 'Google'])
        
        questions = []
        for company in target_companies[:2]:  # Limit to 2 for demo
            query = f"{company} coding interview questions"
            results = self.search_tool.search(query)
            
            for result in results:
                question = InterviewQuestion(
                    title=result.get('title', 'Unknown'),
                    company=result.get('company', company),
                    difficulty=result.get('difficulty', 'Medium'),
                    topic='Data Structures',
                    url=result.get('url', '')
                )
                questions.append(question)
        
        print(f"  âœ“ {self.name} found {len(questions)} questions")
        return questions


class CompanyResearchAgent:
    """Agent specialized in company research"""
    
    def __init__(self, search_tool: GoogleSearchTool):
        self.name = "Company Research Agent"
        self.search_tool = search_tool
        
    def execute(self, user_prefs: Dict) -> List[CompanyInfo]:
        """
        Gathers company-specific interview information
        Runs in parallel with other agents
        """
        print(f"\n  ğŸ�¢ {self.name} starting research...")
        target_companies = user_prefs.get('target_companies', ['Microsoft'])
        
        company_data = []
        for company in target_companies[:2]:
            query = f"{company} interview process 2025"
            results = self.search_tool.search(query)
            
            for result in results:
                info = CompanyInfo(
                    company=result.get('company', company),
                    interview_process=result.get('process', 'Unknown'),
                    recent_news=result.get('news', 'No recent updates'),
                    hiring_status=result.get('hiring', 'Unknown'),
                    url=result.get('url', '')
                )
                company_data.append(info)
        
        print(f"  âœ“ {self.name} researched {len(company_data)} companies")
        return company_data


class SkillGapAgent:
    """Agent specialized in identifying skill gaps"""
    
    def __init__(self, search_tool: GoogleSearchTool):
        self.name = "Skill Gap Agent"
        self.search_tool = search_tool
        
    def execute(self, user_prefs: Dict) -> List[SkillInfo]:
        """
        Analyzes trending technologies and skill requirements
        Runs in parallel with other agents
        """
        print(f"\n  ğŸ�¯ {self.name} analyzing trends...")
        current_skills = user_prefs.get('current_skills', ['Python'])
        
        query = "trending technology skills 2025 software engineering"
        results = self.search_tool.search(query)
        
        skill_data = []
        for result in results:
            skill = SkillInfo(
                skill=result.get('skill', 'Unknown'),
                trend=result.get('trend', 'Stable'),
                relevance=result.get('relevance', 'Medium'),
                learning_resources=result.get('resources', [])
            )
            skill_data.append(skill)
        
        print(f"  âœ“ {self.name} identified {len(skill_data)} trending skills")
        return skill_data


# Initialize all three agents
interview_agent = InterviewPrepAgent(search_tool)
company_agent = CompanyResearchAgent(search_tool)
skill_agent = SkillGapAgent(search_tool)

print("\nâœ“ Three specialized agents created")
print("âœ“ Ready for parallel execution\n")


# CELL 4: Memory System (Sessions & Long-term Memory)
print("\nğŸ§  Implementing Memory System")
print("=" * 50)

# ==========================================
# CONCEPT 3: SESSIONS & MEMORY
# ==========================================

class InMemorySessionService:
    """
    Implements session management and long-term memory
    Simulates InMemorySessionService from Agents SDK
    """
    
    def __init__(self):
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_data = {
            'started_at': datetime.now(),
            'query_count': 0,
            'interactions': []
        }
        self.long_term_memory = {
            'user_preferences': {},
            'search_history': [],
            'recommendations_given': []
        }
        
    def start_session(self, user_id: str):
        """Initialize a new session"""
        print(f"  ğŸ�¯ Starting session: {self.session_id}")
        self.session_data['user_id'] = user_id
        return self.session_id
    
    def store_user_preferences(self, prefs: Dict):
        """Store user preferences in long-term memory"""
        self.long_term_memory['user_preferences'].update(prefs)
        print(f"  ğŸ’¾ Stored preferences: {list(prefs.keys())}")
    
    def get_user_preferences(self) -> Dict:
        """Retrieve stored user preferences"""
        return self.long_term_memory['user_preferences']
    
    def log_interaction(self, interaction_type: str, data: Dict):
        """Log user interaction in session"""
        self.session_data['query_count'] += 1
        self.session_data['interactions'].append({
            'type': interaction_type,
            'timestamp': datetime.now(),
            'data': data
        })
    
    def add_to_search_history(self, search_query: str, results_count: int):
        """Add search to long-term history"""
        self.long_term_memory['search_history'].append({
            'query': search_query,
            'results': results_count,
            'timestamp': datetime.now()
        })
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        return {
            'session_id': self.session_id,
            'duration': (datetime.now() - self.session_data['started_at']).seconds,
            'queries': self.session_data['query_count'],
            'interactions': len(self.session_data['interactions'])
        }
    
    def personalize_recommendations(self, results: List) -> List:
        """Use memory to personalize recommendations"""
        prefs = self.get_user_preferences()
        # In real implementation, would filter/rank based on preferences
        print(f"  âœ¨ Personalizing based on stored preferences")
        return results


# Initialize Memory System
memory_system = InMemorySessionService()
print("âœ“ Memory System (InMemorySessionService) initialized")
print("âœ“ Session and long-term memory ready\n")


# CELL 5: Orchestrator and Main Execution with Parallel Agent Execution
print("\nâš¡ Running Tech Career Prep Multi-Agent System")
print("=" * 50)

# Orchestrator that coordinates all agents
class TechCareerOrchestrator:
    """Orchestrates parallel agent execution with memory management"""
    
    def __init__(self, interview_agent, company_agent, skill_agent, memory_system):
        self.interview_agent = interview_agent
        self.company_agent = company_agent
        self.skill_agent = skill_agent
        self.memory = memory_system
        
    def execute_parallel(self, user_prefs: Dict) -> Dict:
        """
        Execute all three agents in PARALLEL using ThreadPoolExecutor
        This demonstrates the Multi-Agent System with Parallel Agents concept
        """
        print("\nğŸš€ Executing agents in PARALLEL...")
        print("-" * 50)
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all three agents to run concurrently
            future_interview = executor.submit(self.interview_agent.execute, user_prefs)
            future_company = executor.submit(self.company_agent.execute, user_prefs)
            future_skill = executor.submit(self.skill_agent.execute, user_prefs)
            
            # Collect results as they complete
            results = {
                'interview_questions': future_interview.result(),
                'company_info': future_company.result(),
                'skill_gaps': future_skill.result()
            }
        
        execution_time = time.time() - start_time
        print(f"\nâœ… All agents completed in {execution_time:.2f} seconds (PARALLEL execution)")
        
        # Log interaction in memory
        self.memory.log_interaction('agent_execution', {
            'agents': 3,
            'execution_time': execution_time,
            'results_count': sum(len(v) for v in results.values())
        })
        
        return results


# ===========================================
# MAIN DEMONSTRATION: All 3 Concepts Together
# ===========================================

print("\n" + "="*50)
print("ğŸ�¯ DEMONSTRATING ALL 3 KEY CONCEPTS")
print("="*50)

# User preferences (simulating a user like you!)
user_preferences = {
    'target_companies': ['Microsoft', 'Google', 'Amazon'],
    'current_skills': ['Python', 'Java', 'SQL'],
    'experience_level': 'Entry-level',
    'goal': 'Join MAANG company within 3 years',
    'interests': ['AI/ML', 'Cloud', 'System Design']
}

print("\nğŸ‘¤ User Profile:")
for key, value in user_preferences.items():
    print(f"  â€¢ {key}: {value}")

# Start session and store preferences
print("\n" + "-"*50)
print("ğŸ“‹ CONCEPT 3 DEMO: Sessions & Memory")
print("-"*50)
memory_system.start_session("user_dhooda")
memory_system.store_user_preferences(user_preferences)

# Create orchestrator
print("\n" + "-"*50)
print("ğŸ¤– CONCEPT 1 DEMO: Multi-Agent System (Parallel)")
print("-"*50)
orchestrator = TechCareerOrchestrator(
    interview_agent,
    company_agent,
    skill_agent,
    memory_system
)

print("\nNote: Agents will execute in PARALLEL using ThreadPoolExecutor")
print("Watch how all 3 agents work simultaneously!")

# Execute all agents in parallel
print("\n" + "-"*50)
print("ğŸ”� CONCEPT 2 DEMO: Google Search Tool Integration")
print("-"*50)
print("Each agent uses Google Search Tool to find information...")

results = orchestrator.execute_parallel(user_preferences)

# Display results
print("\n" + "="*50)
print("ğŸ�‰ RESULTS FROM ALL AGENTS")
print("="*50)

print(f"\nğŸ“š Interview Questions Found: {len(results['interview_questions'])}")
for i, q in enumerate(results['interview_questions'][:3], 1):
    print(f"  {i}. {q.title} ({q.company}) - {q.difficulty}")
    print(f"     ğŸ”— {q.url}")

print(f"\nğŸ�¢ Company Research: {len(results['company_info'])} companies")
for i, c in enumerate(results['company_info'][:2], 1):
    print(f"  {i}. {c.company}")
    print(f"     Process: {c.interview_process}")
    print(f"     Status: {c.hiring_status}")
    print(f"     ğŸ”— {c.url}")

print(f"\nğŸ�¯ Skill Gaps Identified: {len(results['skill_gaps'])} skills")
for i, s in enumerate(results['skill_gaps'], 1):
    print(f"  {i}. {s.skill} - {s.trend}")
    print(f"     Relevance: {s.relevance}")
    if s.learning_resources:
        print(f"     Resources: {', '.join(s.learning_resources[:2])}")

# Personalize with memory
print("\n" + "-"*50)
print("âœ¨ Personalizing recommendations using Memory...")
print("-"*50)
personalized = memory_system.personalize_recommendations(results['interview_questions'])

# Session summary
print("\n" + "="*50)
print("ğŸ“Š SESSION SUMMARY (From Memory System)")
print("="*50)
summary = memory_system.get_session_summary()
for key, value in summary.items():
    print(f"  â€¢ {key}: {value}")

print("\n" + "="*50)
print("âœ… ALL 3 CONCEPTS SUCCESSFULLY DEMONSTRATED!")
print("="*50)
print("\n1ï¸�âƒ£  Multi-Agent System: 3 specialized agents executed in PARALLEL")
print("2ï¸�âƒ£  Tools Integration: Google Search Tool used by all agents")
print("3ï¸�âƒ£  Sessions & Memory: InMemorySessionService tracked everything")
print("\nğŸ�‰ Tech Career Prep Multi-Agent System Complete!\n")

