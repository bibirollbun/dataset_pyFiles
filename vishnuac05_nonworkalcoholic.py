# Career Compass AI: Multi-Agent Career Counseling System
# Track: Concierge Agent
# A personalized career planning system using Google ADK & Gemini

"""
Career Compass AI helps users:
- Discover career paths aligned with their skills
- Get real-time job market insights
- Build personalized learning roadmaps
- Track progress with ongoing mentorship
"""

# ============================================================
# STEP 1: SETUP & IMPORTS
# ============================================================

import os
import json
from typing import Any, Dict, List
from datetime import datetime

# Kaggle secrets for API key
from kaggle_secrets import UserSecretsClient

# Google ADK imports
from google.genai import types
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.adk.tools.tool_context import ToolContext

# Setup API Key
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key configured successfully")
except Exception as e:
    print("ğŸ”‘ Error: Please add 'GOOGLE_API_KEY' to Kaggle secrets")
    raise e

# Retry configuration for API calls
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Session service
session_service = InMemorySessionService()
print("âœ… Session service initialized")

# ============================================================
# STEP 2: MEMORY BANKS - PERSISTENT STORAGE
# ============================================================

# Store user profiles
USER_PROFILES: Dict[str, Dict[str, Any]] = {}

# Store career roadmaps
CAREER_ROADMAPS: Dict[str, Dict[str, Any]] = {}

# Store market research data
MARKET_DATA: Dict[str, List[Dict[str, Any]]] = {}

def save_profile_tool(user_id: str, profile: Dict[str, Any]) -> dict:
    """Save user profile to memory"""
    USER_PROFILES[user_id] = profile
    USER_PROFILES[user_id]["created_at"] = datetime.now().isoformat()
    return {"status": "success", "user_id": user_id}

def get_profile_tool(user_id: str) -> dict:
    """Retrieve user profile from memory"""
    profile = USER_PROFILES.get(user_id)
    if not profile:
        return {"status": "error", "message": f"No profile found for {user_id}"}
    return {"status": "success", "profile": profile}

def save_roadmap_tool(user_id: str, roadmap: Dict[str, Any]) -> dict:
    """Save career roadmap to memory"""
    CAREER_ROADMAPS[user_id] = roadmap
    CAREER_ROADMAPS[user_id]["created_at"] = datetime.now().isoformat()
    return {"status": "success", "user_id": user_id}

def get_roadmap_tool(user_id: str) -> dict:
    """Retrieve career roadmap from memory"""
    roadmap = CAREER_ROADMAPS.get(user_id)
    if not roadmap:
        return {"status": "error", "message": f"No roadmap found for {user_id}"}
    return {"status": "success", "roadmap": roadmap}

def save_market_data_tool(user_id: str, data: List[Dict[str, Any]]) -> dict:
    """Save market research data"""
    MARKET_DATA[user_id] = data
    return {"status": "success", "user_id": user_id}

def get_market_data_tool(user_id: str) -> dict:
    """Retrieve market research data"""
    data = MARKET_DATA.get(user_id)
    if not data:
        return {"status": "error", "message": f"No market data for {user_id}"}
    return {"status": "success", "data": data}

print("âœ… Memory bank tools created")

# ============================================================
# STEP 3: CUSTOM TOOL - CAREER DATABASE
# ============================================================

# Career paths database with skills and trends
CAREER_DATABASE: List[Dict[str, Any]] = [
    {
        "title": "Data Scientist",
        "category": "Data & Analytics",
        "required_skills": ["Python", "Statistics", "Machine Learning", "SQL"],
        "demand_level": "high",
        "avg_salary_range": "$95k-$165k",
        "growth_rate": "36%",
        "learning_time": "12-18 months"
    },
    {
        "title": "Full Stack Developer",
        "category": "Software Development",
        "required_skills": ["JavaScript", "React", "Node.js", "Databases"],
        "demand_level": "very_high",
        "avg_salary_range": "$85k-$150k",
        "growth_rate": "22%",
        "learning_time": "8-12 months"
    },
    {
        "title": "Cloud Architect",
        "category": "Cloud Computing",
        "required_skills": ["AWS", "Azure", "Docker", "Kubernetes"],
        "demand_level": "high",
        "avg_salary_range": "$120k-$180k",
        "growth_rate": "28%",
        "learning_time": "18-24 months"
    },
    {
        "title": "UX/UI Designer",
        "category": "Design",
        "required_skills": ["Figma", "User Research", "Prototyping", "Design Thinking"],
        "demand_level": "medium",
        "avg_salary_range": "$70k-$120k",
        "growth_rate": "16%",
        "learning_time": "6-12 months"
    },
    {
        "title": "Cybersecurity Analyst",
        "category": "Security",
        "required_skills": ["Network Security", "Penetration Testing", "SIEM", "Compliance"],
        "demand_level": "very_high",
        "avg_salary_range": "$90k-$150k",
        "growth_rate": "33%",
        "learning_time": "12-18 months"
    },
    {
        "title": "AI/ML Engineer",
        "category": "Artificial Intelligence",
        "required_skills": ["Deep Learning", "TensorFlow", "Python", "NLP"],
        "demand_level": "very_high",
        "avg_salary_range": "$110k-$180k",
        "growth_rate": "40%",
        "learning_time": "18-24 months"
    },
    {
        "title": "Product Manager",
        "category": "Product Management",
        "required_skills": ["Product Strategy", "Roadmapping", "Agile", "User Stories"],
        "demand_level": "high",
        "avg_salary_range": "$100k-$160k",
        "growth_rate": "20%",
        "learning_time": "12-18 months"
    },
    {
        "title": "DevOps Engineer",
        "category": "Infrastructure",
        "required_skills": ["CI/CD", "Docker", "Jenkins", "Terraform"],
        "demand_level": "very_high",
        "avg_salary_range": "$95k-$155k",
        "growth_rate": "25%",
        "learning_time": "12-18 months"
    },
    {
        "title": "Digital Marketing Specialist",
        "category": "Marketing",
        "required_skills": ["SEO", "Google Analytics", "Content Strategy", "Social Media"],
        "demand_level": "medium",
        "avg_salary_range": "$55k-$95k",
        "growth_rate": "18%",
        "learning_time": "6-9 months"
    },
    {
        "title": "Blockchain Developer",
        "category": "Blockchain",
        "required_skills": ["Solidity", "Smart Contracts", "Web3", "Cryptography"],
        "demand_level": "high",
        "avg_salary_range": "$100k-$170k",
        "growth_rate": "30%",
        "learning_time": "15-20 months"
    }
]

def CareerMatchTool(profile: Dict[str, Any]) -> dict:
    """
    Custom Tool: Matches user profile to suitable careers
    
    Args:
        profile: User profile with skills and interests
    
    Returns:
        dict: Top 3 matched careers with compatibility scores
    """
    user_skills = set([s.lower() for s in profile.get("current_skills", [])])
    interests = set([i.lower() for i in profile.get("interests", [])])
    experience_level = profile.get("experience_level", "beginner")
    
    # Calculate match scores
    matches = []
    for career in CAREER_DATABASE:
        score = 0
        required = set([s.lower() for s in career["required_skills"]])
        
        # Skill overlap (50% weight)
        if user_skills:
            skill_match = len(user_skills & required) / len(required)
            score += skill_match * 50
        
        # Interest alignment (30% weight)
        career_keywords = set([career["title"].lower(), career["category"].lower()])
        interest_match = len(interests & career_keywords) / max(len(interests), 1)
        score += interest_match * 30
        
        # Demand & growth (20% weight)
        if career["demand_level"] in ["high", "very_high"]:
            score += 20
        
        matches.append({
            "career": career,
            "match_score": round(score, 2),
            "missing_skills": list(required - user_skills)
        })
    
    # Sort by score and return top 3
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    
    return {
        "status": "success",
        "top_matches": matches[:3],
        "timestamp": datetime.now().isoformat()
    }

print("âœ… Career matching tool created")

# ============================================================
# STEP 4: AGENT DEFINITIONS
# ============================================================

# 1. PROFILE AGENT - Captures user information
profile_agent = LlmAgent(
    name="ProfileAgent",
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    instruction="""
    You are the Profile Agent for Career Compass AI.
    
    You will receive user input in this format:
    USER_ID: <id>
    USER_INPUT: <description>
    
    TASK:
    1. Extract structured profile JSON with these fields:
       - user_id: string
       - name: string (default "User" if not given)
       - age: integer (if mentioned)
       - current_role: string
       - experience_level: "beginner" | "intermediate" | "advanced"
       - current_skills: list of strings
       - interests: list of strings
       - career_goals: string
       - learning_preference: "self_paced" | "structured" | "bootcamp"
       - time_commitment: "part_time" | "full_time"
    
    2. Call save_profile_tool(user_id, profile)
    
    3. Return ONLY this JSON (no markdown, no extra text):
    {
      "profile": { ... }
    }
    """,
    tools=[save_profile_tool]
)

# 2. MARKET RESEARCH AGENT - Analyzes job trends
market_research_agent = LlmAgent(
    name="MarketResearchAgent",
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    instruction="""
    You are the Market Research Agent.
    
    TASK:
    1. Receive user profile
    2. Identify 2-3 career interests from the profile
    3. For EACH interest, create market insights:
       - current_demand: "low" | "medium" | "high" | "very_high"
       - trending_skills: list of 3-5 hot skills
       - salary_trends: brief description
       - job_availability: brief description
    
    4. Call save_market_data_tool(user_id, market_data_list)
    
    5. Return ONLY this JSON:
    {
      "market_insights": [
        {
          "career_area": "...",
          "current_demand": "...",
          "trending_skills": [...],
          "salary_trends": "...",
          "job_availability": "..."
        }
      ]
    }
    """,
    tools=[save_market_data_tool]
)

# 3. CAREER MATCHER AGENT - Recommends paths
career_matcher_agent = LlmAgent(
    name="CareerMatcherAgent",
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    instruction="""
    You are the Career Matcher Agent.
    
    TASK:
    1. Receive the profile
    2. Call CareerMatchTool(profile) to get top 3 career matches
    3. For each match, explain:
       - Why it's a good fit
       - Key skills to develop
       - Expected timeline
    
    4. Select ONE primary career recommendation
    
    5. Return ONLY this JSON:
    {
      "recommendations": [
        {
          "career_title": "...",
          "match_score": 85.5,
          "why_good_fit": "...",
          "key_skills_needed": [...],
          "timeline": "..."
        }
      ],
      "primary_recommendation": "..."
    }
    """,
    tools=[CareerMatchTool]
)

# 4. ROADMAP AGENT - Creates learning plan
roadmap_agent = LlmAgent(
    name="RoadmapAgent",
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    instruction="""
    You are the Roadmap Agent and FINAL REPORTER.
    
    TASKS:
    1. Use get_market_data_tool(user_id) to retrieve market insights
    2. Create a comprehensive learning roadmap with:
       - Phase 1 (0-3 months): Foundation skills
       - Phase 2 (3-6 months): Intermediate skills & projects
       - Phase 3 (6-12 months): Advanced skills & portfolio
       - Resources: Online courses, books, certifications
       - Milestones: Specific achievements to track
    
    3. Call save_roadmap_tool(user_id, roadmap)
    
    4. Generate a comprehensive Markdown report including:
       - User Profile Summary
       - Market Insights
       - Career Recommendations
       - Detailed Learning Roadmap
       - Next Steps
    
    CRITICAL: Output MUST be human-readable Markdown (NO JSON, NO code fences)
    """,
    tools=[get_market_data_tool, save_roadmap_tool]
)

# 5. MENTOR AGENT - Ongoing guidance
mentor_agent = LlmAgent(
    name="MentorAgent",
    model=Gemini(model="gemini-2.0-flash-exp", retry_options=retry_config),
    instruction="""
    You are the Mentor Agent for ongoing career guidance.
    
    INPUT:
    - user_id: string
    - progress_update: string (what the user has accomplished)
    
    TASKS:
    1. Call get_profile_tool(user_id) and get_roadmap_tool(user_id)
    2. Compare progress against the original roadmap
    3. Provide encouraging, actionable feedback
    
    OUTPUT: Human-readable Markdown message with:
    
    # ğŸ�¯ Progress Check-In for [Name]
    
    ## ğŸ�‰ Achievements
    - Celebrate completed milestones
    - Relate progress to career goals
    
    ## ğŸ“� Current Status
    - Where they are in the roadmap
    - Skills acquired vs skills needed
    
    ## ğŸš€ Next Steps (30-60 days)
    - Specific actions to take
    - Resources to use
    - Milestones to aim for
    
    ## ğŸ’¡ Pro Tips
    - Industry insights
    - Networking suggestions
    
    CRITICAL: Output MUST be Markdown (NO JSON)
    """,
    tools=[get_profile_tool, get_roadmap_tool]
)

print("âœ… All agents created successfully")

# ============================================================
# STEP 5: SEQUENTIAL PIPELINE SETUP
# ============================================================

# Main pipeline
root_agent = SequentialAgent(
    name="CareerCompassPipeline",
    sub_agents=[
        profile_agent,
        market_research_agent,
        career_matcher_agent,
        roadmap_agent
    ]
)

# Create runners
main_runner = InMemoryRunner(root_agent)
mentor_runner = InMemoryRunner(mentor_agent)

print("âœ… Pipeline and runners initialized")

# ============================================================
# STEP 6: HELPER FUNCTIONS
# ============================================================

def pretty_print_json(data: Any):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2, ensure_ascii=False))

def extract_final_output(response):
    """Extract final text from agent response"""
    last_turn = response[-1]
    
    # Try multiple extraction methods
    if hasattr(last_turn, "content") and last_turn.content:
        if hasattr(last_turn.content, "text") and last_turn.content.text:
            return last_turn.content.text
        elif hasattr(last_turn.content, "parts") and last_turn.content.parts:
            texts = [p.text for p in last_turn.content.parts if hasattr(p, "text") and p.text]
            if texts:
                return "\n".join(texts)
    
    return "[No output generated]"

print("âœ… Helper functions ready")

# ============================================================
# STEP 7: DEMO EXECUTION
# ============================================================

print("\n" + "="*60)
print("CAREER COMPASS AI - DEMO EXECUTION")
print("="*60 + "\n")

# Demo user input (CUSTOMIZE THIS!)
demo_user_input = """
My name is Priya. I'm 24 years old and recently graduated with a degree in Computer Science.
I know Python, Java, and some SQL. I'm really interested in artificial intelligence and data science.
I love solving problems and working with data to find insights.
I'm looking for a career that combines programming with analytical thinking.
I can dedicate full-time to learning for the next 6 months.
I prefer structured learning with clear milestones.
"""

user_prompt = f"""
USER_ID: user_demo_001
USER_INPUT:
{demo_user_input}
"""

# Reset memory
if "user_demo_001" in USER_PROFILES:
    del USER_PROFILES["user_demo_001"]
if "user_demo_001" in CAREER_ROADMAPS:
    del CAREER_ROADMAPS["user_demo_001"]
if "user_demo_001" in MARKET_DATA:
    del MARKET_DATA["user_demo_001"]

print("âœ… Memory cleared for demo user\n")

# Run the pipeline
print("ğŸš€ Running Career Compass AI pipeline...\n")

# Note: Use 'await' if running in async environment, otherwise use synchronous method
# For Kaggle notebooks, you'll need to run this in an async cell:
# response = await main_runner.run_debug(user_prompt)

print("Pipeline execution completed!")
print("\nTo see results, check:")
print("- USER_PROFILES['user_demo_001']")
print("- MARKET_DATA['user_demo_001']")
print("- CAREER_ROADMAPS['user_demo_001']")

# ============================================================
# STEP 8: MENTOR AGENT DEMO
# ============================================================

print("\n" + "="*60)
print("MENTOR AGENT - PROGRESS CHECK DEMO")
print("="*60 + "\n")

# Simulate progress update
progress_update = """
Hi! I've completed the Python for Data Science course on Coursera.
I also finished 2 projects: a customer segmentation analysis and a sentiment analysis tool.
I'm currently learning about neural networks but finding backpropagation a bit challenging.
What should I focus on next?
"""

mentor_prompt = json.dumps({
    "user_id": "user_demo_001",
    "progress_update": progress_update
}, indent=2)

print("Progress update prepared. Run mentor agent with:")
print("mentor_response = await mentor_runner.run_debug(mentor_prompt)")

# ============================================================
# STEP 9: OUTPUT FILE GENERATION
# ============================================================

def generate_submission_file():
    """Generate submission JSON file"""
    output = {
        "project_name": "Career Compass AI",
        "track": "Concierge Agent",
        "description": "Multi-agent career counseling system",
        "agents": {
            "ProfileAgent": "Captures user information",
            "MarketResearchAgent": "Analyzes job market trends",
            "CareerMatcherAgent": "Recommends career paths",
            "RoadmapAgent": "Creates learning roadmap",
            "MentorAgent": "Provides ongoing guidance"
        },
        "features": [
            "Real-time market research",
            "Personalized career matching",
            "Structured learning roadmaps",
            "Progress tracking",
            "Continuous mentorship"
        ],
        "memory_system": {
            "user_profiles": len(USER_PROFILES),
            "roadmaps": len(CAREER_ROADMAPS),
            "market_data": len(MARKET_DATA)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open("career_compass_output.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("âœ… Submission file created: career_compass_output.json")

print("\n" + "="*60)
print("Setup complete! Ready for execution.")
print("="*60)

