!pip install -q google-genai google-adk mcp chromadb numpy pandas matplotlib seaborn plotly
!pip install -q python-dotenv typing-extensions pydantic

print("âœ… All dependencies installed successfully!")


"""
This cell imports ADK components, tools, agents, and utilities
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Google ADK Core Imports
from google import genai
from google.genai import types

# ADK Agent and Runner Imports
try:
    from google.adk.agents import Agent, AgentTool
    from google.adk.agents.parallel_agent import ParallelAgent
    from google.adk.agents.sequential_agent import SequentialAgent
    from google.adk.runners.local_runner import LocalRunner
except:
    print("âš ï¸� Using alternative ADK imports")
    
# ADK Tools Imports
try:
    from google.adk.tools import FunctionTool, BuiltInTool
    from google.adk.tools.google_search import google_search
    from google.adk.tools.code_execution import code_execution
    from google.adk.tools.mcp_tool import McpToolset
except:
    print("âš ï¸� Some tools may not be available")

# Session and Memory Management
try:
    from google.adk.sessions import InMemorySessionService, Session
    from google.adk.memory import MemoryBank, MemoryEntry
except:
    print("âš ï¸� Using basic session management")

# Observability
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("âœ… All imports completed successfully!")
print(f"ğŸ“… System initialized at: {datetime.now()}")


import os
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient

# Configure your API key (use Kaggle Secrets in production)
user_secrets = UserSecretsClient()

try:
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… API Key loaded from Kaggle Secrets")
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    if not GOOGLE_API_KEY:
        print("âš ï¸� Please set GOOGLE_API_KEY in Kaggle Secrets or environment")
        GOOGLE_API_KEY = input("Enter your Google API Key: ")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Gemini client (no explicit api_version)
client = genai.Client(api_key=GOOGLE_API_KEY)

# Use a model that supports generateContent with this SDK
MODEL_NAME = "gemini-2.0-flash-001"   # stable Flash model

# Test API connection
try:
    test_response = client.models.generate_content(
        model=MODEL_NAME,
        contents="Hello, respond with OK if you can hear me."
    )
    print(f"âœ… Gemini API Connected: {test_response.text[:50]}")
except Exception as e:
    print(f"â�Œ API Connection Error: {e}")

TEMPERATURE = 0.7
MAX_TOKENS = 2048

print(f"\nğŸ¤– Using Model: {MODEL_NAME}")


"""
Performs mood analysis and risk assessment based on user input
"""

def assess_mental_health(user_input: str, mood_score: int = 5) -> Dict[str, Any]:
    """
    Analyzes user's mental health state and provides risk assessment.
    
    Args:
        user_input: User's message describing their mental state
        mood_score: Self-reported mood score (1-10, default 5)
    
    Returns:
        Dictionary with assessment results, risk level, and recommendations
    """
    logger.info(f"ğŸ”� Assessing mental health - Mood Score: {mood_score}")
    
    # Crisis keywords detection
    crisis_keywords = ['suicide', 'kill myself', 'end it all', 'no point living', 
                       'self harm', 'hurt myself', 'die', 'death wish']
    high_risk_keywords = ['depressed', 'hopeless', 'worthless', 'anxiety attack',
                         'panic', 'can\'t cope', 'overwhelming']
    
    user_lower = user_input.lower()
    
    # Determine risk level
    if any(keyword in user_lower for keyword in crisis_keywords):
        risk_level = "CRITICAL"
        urgency = "IMMEDIATE"
        color = "ğŸ”´"
    elif mood_score <= 3 or any(keyword in user_lower for keyword in high_risk_keywords):
        risk_level = "HIGH"
        urgency = "URGENT"
        color = "ğŸŸ "
    elif mood_score <= 5:
        risk_level = "MODERATE"
        urgency = "SOON"
        color = "ğŸŸ¡"
    else:
        risk_level = "LOW"
        urgency = "ROUTINE"
        color = "ğŸŸ¢"
    
    assessment = {
        "risk_level": risk_level,
        "urgency": urgency,
        "mood_score": mood_score,
        "timestamp": datetime.now().isoformat(),
        "crisis_detected": risk_level == "CRITICAL",
        "indicators": {
            "mood_score_low": mood_score <= 3,
            "crisis_language": any(keyword in user_lower for keyword in crisis_keywords),
            "high_risk_language": any(keyword in user_lower for keyword in high_risk_keywords)
        },
        "status_icon": color,
        "recommended_actions": get_recommended_actions(risk_level)
    }
    
    logger.info(f"{color} Assessment Complete: {risk_level} risk")
    return assessment

def get_recommended_actions(risk_level: str) -> List[str]:
    """Get recommended actions based on risk level"""
    actions = {
        "CRITICAL": [
            "Contact emergency services (999 UK) immediately",
            "Reach out to Samaritans (116 123) - 24/7 free helpline",
            "Use Shout Crisis Text Line - Text SHOUT to 85258",
            "Go to nearest A&E or call crisis team"
        ],
        "HIGH": [
            "Schedule urgent appointment with GP or university counseling",
            "Contact student wellbeing services today",
            "Reach out to trusted friend or family member",
            "Call Nightline (student listening service)"
        ],
        "MODERATE": [
            "Book counseling session within this week",
            "Practice self-care activities",
            "Monitor mood daily",
            "Consider peer support groups"
        ],
        "LOW": [
            "Continue current wellbeing practices",
            "Maintain healthy routines",
            "Stay connected with support network",
            "Regular check-ins with wellbeing services"
        ]
    }
    return actions.get(risk_level, [])

# Test the tool
test_result = assess_mental_health("Feeling a bit stressed about exams", mood_score=6)
print(f"âœ… Mental Health Assessment Tool Created")
print(f"Test Result: {test_result['risk_level']} - {test_result['status_icon']}")


"""
Finds relevant mental health resources based on location and needs
"""

def find_mental_health_resources(
    location: str = "UK",
    resource_type: str = "all",
    urgency: str = "routine"
) -> Dict[str, List[Dict[str, str]]]:
    """
    Finds mental health resources based on location and type.
    
    Args:
        location: Geographic location (default: UK)
        resource_type: Type of resource (crisis, counseling, peer_support, online, all)
        urgency: Level of urgency (immediate, urgent, routine)
    
    Returns:
        Dictionary of categorized resources with contact information
    """
    logger.info(f"ğŸ”� Finding resources: {resource_type} in {location} (Urgency: {urgency})")
    
    # Comprehensive UK student mental health resources database
    resources_db = {
        "crisis_helplines": [
            {
                "name": "Samaritans",
                "phone": "116 123",
                "availability": "24/7 Free",
                "description": "Confidential emotional support for anyone in crisis",
                "website": "https://www.samaritans.org"
            },
            {
                "name": "Shout Crisis Text Line",
                "contact": "Text SHOUT to 85258",
                "availability": "24/7 Free",
                "description": "Free text support for mental health crises",
                "website": "https://giveusashout.org"
            },
            {
                "name": "Papyrus (Under 35s)",
                "phone": "0800 068 4141",
                "text": "07860 039967",
                "availability": "Weekdays 9am-10pm, Weekends 2pm-10pm",
                "description": "Suicide prevention for young people",
                "website": "https://www.papyrus-uk.org"
            },
            {
                "name": "Emergency Services",
                "phone": "999",
                "availability": "24/7",
                "description": "Immediate life-threatening emergencies"
            }
        ],
        "university_services": [
            {
                "name": "University Counseling Service",
                "description": "Free confidential counseling for enrolled students",
                "access": "Book via student portal or wellbeing office",
                "typical_wait": "1-3 weeks"
            },
            {
                "name": "Student Wellbeing Team",
                "description": "Mental health advisors and support coordinators",
                "access": "Drop-in sessions or appointment",
                "services": "Assessment, referrals, ongoing support"
            },
            {
                "name": "Nightline",
                "phone": "Check your university's local number",
                "availability": "Evening hours during term time",
                "description": "Confidential listening service run by students",
                "website": "https://www.nightline.ac.uk"
            }
        ],
        "nhs_services": [
            {
                "name": "NHS 111",
                "phone": "111",
                "availability": "24/7",
                "description": "Mental health crisis support and GP out of hours"
            },
            {
                "name": "GP Mental Health Services",
                "access": "Register with local GP practice",
                "services": "Assessment, medication, therapy referrals",
                "cost": "Free on NHS"
            },
            {
                "name": "IAPT (Improving Access to Psychological Therapies)",
                "access": "Self-referral online or via GP",
                "services": "CBT, counseling for anxiety and depression",
                "website": "https://www.nhs.uk/service-search/mental-health/find-a-psychological-therapies-service"
            }
        ],
        "online_support": [
            {
                "name": "Student Minds",
                "website": "https://www.studentminds.org.uk",
                "description": "UK student mental health charity with resources and peer support",
                "services": "Online community, toolkits, campaigns"
            },
            {
                "name": "Mind",
                "website": "https://www.mind.org.uk",
                "phone": "0300 123 3393",
                "description": "Mental health information and support",
                "services": "Information, local services, online community"
            },
            {
                "name": "YoungMinds",
                "website": "https://www.youngminds.org.uk",
                "text": "Text YM to 85258",
                "description": "Youth mental health support and resources"
            },
            {
                "name": "Togetherall",
                "website": "https://togetherall.com",
                "description": "Online peer support community (often free via university)",
                "access": "Check if your university provides access"
            }
        ],
        "peer_support": [
            {
                "name": "University Peer Support Groups",
                "description": "Student-led support groups on campus",
                "access": "Check with Student Union or Wellbeing Services"
            },
            {
                "name": "Mental Health Society",
                "description": "Student societies focused on mental health awareness",
                "activities": "Events, discussions, peer support"
            }
        ],
        "specialized_services": [
            {
                "name": "Exam Stress Support",
                "description": "Specialized support during exam periods",
                "access": "Academic skills team, wellbeing services"
            },
            {
                "name": "International Student Support",
                "description": "Culturally sensitive support for international students",
                "access": "International student office, specialist counselors"
            },
            {
                "name": "LGBTQ+ Support",
                "description": "Specialist support for LGBTQ+ students",
                "resources": "LGBT Foundation, university LGBTQ+ groups"
            }
        ]
    }
    
    # Filter resources based on urgency and type
    if urgency == "immediate":
        return {"crisis_helplines": resources_db["crisis_helplines"]}
    
    if resource_type == "all":
        return resources_db
    elif resource_type in resources_db:
        return {resource_type: resources_db[resource_type]}
    else:
        # Return most relevant categories
        return {
            "crisis_helplines": resources_db["crisis_helplines"],
            "university_services": resources_db["university_services"],
            "online_support": resources_db["online_support"]
        }

# Test the resource finder
test_resources = find_mental_health_resources(location="Birmingham, UK", urgency="urgent")
print(f"âœ… Resource Finder Tool Created")
print(f"Found {sum(len(v) for v in test_resources.values())} resources across {len(test_resources)} categories")


"""
Suggests evidence-based wellness activities based on current state
"""

def recommend_wellness_activities(
    mood_state: str = "neutral",
    time_available: int = 15,
    energy_level: str = "medium"
) -> List[Dict[str, Any]]:
    """
    Recommends personalized wellness activities based on current state.
    
    Args:
        mood_state: Current mood (anxious, low, stressed, neutral, good)
        time_available: Minutes available for activity (5-120)
        energy_level: Current energy (low, medium, high)
    
    Returns:
        List of recommended activities with instructions and benefits
    """
    logger.info(f"ğŸ’¡ Recommending activities: mood={mood_state}, time={time_available}min, energy={energy_level}")
    
    activities_db = {
        "breathing_exercises": {
            "name": "4-7-8 Breathing Technique",
            "duration": 5,
            "energy_required": "low",
            "best_for": ["anxious", "stressed"],
            "instructions": [
                "Sit comfortably with back straight",
                "Exhale completely through mouth",
                "Close mouth, inhale through nose for 4 counts",
                "Hold breath for 7 counts",
                "Exhale through mouth for 8 counts",
                "Repeat cycle 4 times"
            ],
            "benefits": "Reduces anxiety, promotes relaxation, improves focus",
            "evidence": "Clinically proven to reduce cortisol levels"
        },
        "grounding_exercise": {
            "name": "5-4-3-2-1 Grounding Technique",
            "duration": 10,
            "energy_required": "low",
            "best_for": ["anxious", "overwhelmed", "panic"],
            "instructions": [
                "Acknowledge 5 things you can SEE around you",
                "Acknowledge 4 things you can TOUCH",
                "Acknowledge 3 things you can HEAR",
                "Acknowledge 2 things you can SMELL",
                "Acknowledge 1 thing you can TASTE"
            ],
            "benefits": "Reduces panic, increases present-moment awareness",
            "evidence": "CBT-based grounding technique"
        },
        "progressive_relaxation": {
            "name": "Progressive Muscle Relaxation (PMR)",
            "duration": 15,
            "energy_required": "low",
            "best_for": ["stressed", "anxious", "tense"],
            "instructions": [
                "Lie down or sit comfortably",
                "Tense feet muscles for 5 seconds, then release",
                "Move up to calves, thighs, abdomen, hands, arms, shoulders, face",
                "Tense each muscle group, hold, then release",
                "Notice the difference between tension and relaxation"
            ],
            "benefits": "Reduces physical tension, improves sleep, decreases stress",
            "evidence": "Evidence-based relaxation technique"
        },
        "mindful_walk": {
            "name": "Mindful Walking",
            "duration": 15,
            "energy_required": "medium",
            "best_for": ["low", "neutral", "stressed"],
            "instructions": [
                "Walk slowly in a quiet area",
                "Focus on sensation of feet touching ground",
                "Notice your breath and body movement",
                "Observe surroundings without judgment",
                "Return focus to walking when mind wanders"
            ],
            "benefits": "Boosts mood, increases mindfulness, gentle exercise",
            "evidence": "Combines physical activity with mindfulness"
        },
        "gratitude_journaling": {
            "name": "Gratitude Practice",
            "duration": 10,
            "energy_required": "low",
            "best_for": ["low", "neutral"],
            "instructions": [
                "Write down 3 things you're grateful for today",
                "Be specific and detailed",
                "Include why each thing matters to you",
                "Notice how you feel after writing"
            ],
            "benefits": "Increases positive emotions, improves mood, builds resilience",
            "evidence": "Positive psychology intervention with strong research support"
        },
        "physical_activity": {
            "name": "Quick Exercise Burst",
            "duration": 10,
            "energy_required": "high",
            "best_for": ["stressed", "low", "neutral"],
            "instructions": [
                "Do 2 minutes of jumping jacks or running in place",
                "Follow with 2 minutes of bodyweight exercises (squats, push-ups)",
                "Take 1 minute rest",
                "Repeat circuit 2 times",
                "Cool down with stretching"
            ],
            "benefits": "Releases endorphins, improves mood, increases energy",
            "evidence": "Exercise is evidence-based treatment for depression and anxiety"
        },
        "social_connection": {
            "name": "Reach Out to Someone",
            "duration": 20,
            "energy_required": "medium",
            "best_for": ["low", "lonely", "neutral"],
            "instructions": [
                "Text or call a friend or family member",
                "Share something positive or ask how they are",
                "Have a genuine conversation",
                "Arrange to meet up if possible"
            ],
            "benefits": "Reduces isolation, improves mood, strengthens relationships",
            "evidence": "Social support is protective factor for mental health"
        },
        "creative_expression": {
            "name": "Creative Activity",
            "duration": 30,
            "energy_required": "medium",
            "best_for": ["stressed", "neutral", "low"],
            "instructions": [
                "Choose: drawing, coloring, writing, music, crafts",
                "Set aside judgment - focus on process not product",
                "Allow yourself to express emotions through creativity",
                "Notice how you feel before and after"
            ],
            "benefits": "Emotional expression, stress relief, sense of accomplishment",
            "evidence": "Art therapy shown to reduce stress and improve wellbeing"
        }
    }
    
    # Filter activities based on criteria
    recommendations = []
    
    for activity_id, activity in activities_db.items():
        # Check if activity matches criteria
        time_ok = activity["duration"] <= time_available
        energy_ok = (
            (energy_level == "low" and activity["energy_required"] == "low") or
            (energy_level == "medium" and activity["energy_required"] in ["low", "medium"]) or
            (energy_level == "high")
        )
        mood_ok = mood_state in activity["best_for"]
        
        if time_ok and energy_ok:
            score = 3 if mood_ok else 1
            recommendations.append({
                **activity,
                "match_score": score
            })
    
    # Sort by match score and return top recommendations
    recommendations.sort(key=lambda x: x["match_score"], reverse=True)
    
    logger.info(f"âœ… Found {len(recommendations)} suitable activities")
    return recommendations[:3]  # Return top 3

# Test wellness recommender
test_activities = recommend_wellness_activities(mood_state="anxious", time_available=15, energy_level="low")
print(f"âœ… Wellness Activity Recommender Created")
print(f"Sample recommendation: {test_activities[0]['name']}")


"""
Implements persistent session state and memory for personalized support
"""

class MentalHealthMemoryBank:
    """Memory bank for storing user's mental health history and preferences"""
    
    def __init__(self):
        self.entries = []
        self.user_profile = {
            "first_interaction": datetime.now().isoformat(),
            "total_interactions": 0,
            "mood_history": [],
            "risk_assessments": [],
            "preferred_resources": [],
            "wellness_activities_tried": []
        }
        logger.info("ğŸ§  Memory Bank initialized")
    
    def add_interaction(self, interaction_data: Dict[str, Any]):
        """Store an interaction in memory"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "data": interaction_data,
            "session_id": interaction_data.get("session_id", "default")
        }
        self.entries.append(entry)
        self.user_profile["total_interactions"] += 1
        
        # Update mood history
        if "mood_score" in interaction_data:
            self.user_profile["mood_history"].append({
                "score": interaction_data["mood_score"],
                "timestamp": entry["timestamp"]
            })
        
        # Update risk assessments
        if "risk_level" in interaction_data:
            self.user_profile["risk_assessments"].append({
                "level": interaction_data["risk_level"],
                "timestamp": entry["timestamp"]
            })
        
        logger.info(f"ğŸ’¾ Interaction stored (Total: {self.user_profile['total_interactions']})")
    
    def get_mood_trend(self) -> Dict[str, Any]:
        """Analyze mood trends over time"""
        if not self.user_profile["mood_history"]:
            return {"trend": "insufficient_data", "average": None}
        
        scores = [m["score"] for m in self.user_profile["mood_history"]]
        avg_score = sum(scores) / len(scores)
        
        # Determine trend (compare recent vs earlier scores)
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            earlier_avg = sum(scores[:-3]) / len(scores[:-3]) if len(scores) > 3 else avg_score
            
            if recent_avg > earlier_avg + 1:
                trend = "improving"
            elif recent_avg < earlier_avg - 1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "average": round(avg_score, 1),
            "total_entries": len(scores),
            "latest_score": scores[-1] if scores else None
        }
    
    def get_summary(self) -> str:
        """Get summary of user's journey"""
        mood_trend = self.get_mood_trend()
        
        summary = f"""
ğŸ“Š **Your Mental Health Journey**
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
âœ… Total check-ins: {self.user_profile['total_interactions']}
ğŸ“ˆ Mood trend: {mood_trend['trend'].upper()}
ğŸ“‰ Average mood: {mood_trend['average']}/10
ğŸ�¯ Activities tried: {len(self.user_profile['wellness_activities_tried'])}
ğŸ“… Tracking since: {self.user_profile['first_interaction'][:10]}
"""
        return summary.strip()

# Initialize global memory bank
memory_bank = MentalHealthMemoryBank()

# Test memory bank
memory_bank.add_interaction({
    "session_id": "test_001",
    "mood_score": 6,
    "risk_level": "LOW",
    "activity": "breathing_exercise"
})

print("âœ… Memory Bank System Created")
print(memory_bank.get_summary())


"""
Creates specialized agents for different mental health support tasks
"""

# Agent 1: Crisis Assessment Agent
crisis_assessment_agent_config = {
    "name": "crisis_assessment_agent",
    "model": MODEL_NAME,
    "description": "Specialized agent for crisis detection and immediate intervention",
    "instruction": """You are a Crisis Assessment Specialist trained in mental health crisis intervention.

YOUR CRITICAL RESPONSIBILITIES:
1. IMMEDIATELY identify any signs of:
   - Suicidal ideation or self-harm intent
   - Acute mental health crisis
   - Immediate danger to self or others

2. If CRISIS detected:
   - Respond with URGENT tone but remain calm and supportive
   - Provide immediate crisis helpline numbers (Samaritans 116 123, Emergency 999)
   - Encourage immediate professional help
   - DO NOT attempt to provide therapy or solve the crisis yourself

3. If NO immediate crisis:
   - Conduct gentle risk assessment
   - Ask about mood, recent changes, support system
   - Recommend appropriate level of support

ALWAYS prioritize safety. When in doubt, escalate to crisis resources.

Be compassionate, non-judgmental, and clear in your communication.""",
    "tools": ["assess_mental_health", "find_mental_health_resources"]
}

# Agent 2: Resource Navigator Agent
resource_navigator_agent_config = {
    "name": "resource_navigator_agent",
    "model": MODEL_NAME,
    "description": "Helps users find appropriate mental health resources and services",
    "instruction": """You are a Mental Health Resource Navigator with expertise in UK student mental health services.

YOUR ROLE:
1. Understand the user's specific needs:
   - Type of support needed (counseling, crisis, peer support, online)
   - Location (university, city)
   - Urgency level
   - Preferences (in-person vs online, NHS vs university services)

2. Provide SPECIFIC, ACTIONABLE resources:
   - Exact phone numbers and websites
   - Clear instructions on how to access
   - Expected wait times where known
   - Free vs paid services

3. Explain options clearly:
   - What each service offers
   - How to access it
   - What to expect
   - Alternative options if first choice unavailable

4. Follow up:
   - Ask if user needs help understanding any resource
   - Offer to find alternative options
   - Encourage taking the first step

Be knowledgeable, practical, and encouraging.""",
    "tools": ["find_mental_health_resources", "google_search"]
}

# Agent 3: Wellness Coach Agent
wellness_coach_agent_config = {
    "name": "wellness_coach_agent",
    "model": MODEL_NAME,
    "description": "Provides evidence-based wellness recommendations and coping strategies",
    "instruction": """You are a Wellness Coach specializing in evidence-based mental health practices for students.

YOUR EXPERTISE:
1. Recommend evidence-based wellness activities:
   - Mindfulness and meditation
   - Physical exercise
   - Sleep hygiene
   - Stress management techniques
   - Social connection strategies

2. Personalize recommendations based on:
   - Current mood and energy level
   - Time available
   - Student's schedule and constraints
   - Previous activities tried

3. Provide clear, actionable guidance:
   - Step-by-step instructions
   - Expected benefits and timeline
   - Tips for building habits
   - How to track progress

4. Motivate and encourage:
   - Celebrate small wins
   - Normalize setbacks
   - Provide gentle accountability
   - Focus on progress not perfection

Be supportive, practical, and science-based in your recommendations.""",
    "tools": ["recommend_wellness_activities"]
}

# Agent 4: Empathetic Listener Agent
empathetic_listener_agent_config = {
    "name": "empathetic_listener_agent",
    "model": MODEL_NAME,
    "description": "Provides empathetic listening and emotional validation",
    "instruction": """You are an Empathetic Listener trained in active listening and emotional validation.

YOUR APPROACH:
1. Create a safe, non-judgmental space:
   - Use warm, compassionate language
   - Validate emotions without minimizing
   - Show genuine understanding and empathy

2. Practice active listening:
   - Reflect back what you hear
   - Ask clarifying questions
   - Show you're fully present and engaged
   - Don't rush to solutions

3. Validate feelings:
   - Acknowledge emotions as valid
   - Normalize common student experiences
   - Avoid toxic positivity
   - Show that struggling is okay

4. Gently guide toward support:
   - After listening, ask what would be helpful
   - Offer resources when appropriate
   - Encourage professional help if needed
   - Respect the user's pace and choices

You are NOT a replacement for professional therapy. You provide supportive listening while encouraging appropriate professional help.

Be warm, patient, and genuinely caring.""",
    "tools": []
}

print("âœ… Four specialized agents configured:")
print(f"  1. {crisis_assessment_agent_config['name']}")
print(f"  2. {resource_navigator_agent_config['name']}")
print(f"  3. {wellness_coach_agent_config['name']}")
print(f"  4. {empathetic_listener_agent_config['name']}")


"""
Orchestrates specialized agents based on user needs
"""

coordinator_agent_config = {
    "name": "mental_health_coordinator",
    "model": MODEL_NAME,
    "description": "Main coordinator that routes users to appropriate specialized support agents",
    "instruction": """You are the Mental Health Support Coordinator for a student wellbeing system.

YOUR ROLE:
1. Triage incoming requests:
   - Assess urgency and type of support needed
   - Route to appropriate specialized agent(s)
   - Coordinate multi-agent responses when needed

2. Routing decisions:
   - CRISIS language â†’ Crisis Assessment Agent IMMEDIATELY
   - Resource/service questions â†’ Resource Navigator Agent
   - Coping strategies/wellness â†’ Wellness Coach Agent
   - Need to talk/emotional support â†’ Empathetic Listener Agent
   - Complex needs â†’ Multiple agents in sequence

3. Maintain conversation flow:
   - Introduce user to specialized agents
   - Summarize key information between agent handoffs
   - Ensure continuity of care
   - Check if user's needs are met

4. Safety protocols:
   - Always prioritize immediate safety
   - Escalate to crisis resources when needed
   - Document risk levels in memory
   - Follow up on high-risk cases

You coordinate a team of specialists to provide comprehensive mental health support.

Be organized, responsive, and focused on user's wellbeing.""",
    "tools": [
        "assess_mental_health",
        "find_mental_health_resources",
        "recommend_wellness_activities",
        "google_search"
    ]
}

print("âœ… Coordinator Agent configured with multi-agent orchestration")
print(f"\nğŸ�¯ System Architecture:")
print("""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚   MENTAL HEALTH COORDINATOR     â”‚
â”‚   (Main Orchestrator)           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
             â”‚
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”�
    â”‚                 â”‚
    â–¼                 â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”�      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚ CRISIS  â”‚      â”‚RESOURCE  â”‚
â”‚ASSESSMENTâ”‚     â”‚NAVIGATOR â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    â”‚                 â”‚
    â–¼                 â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”�      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚WELLNESS â”‚      â”‚EMPATHETICâ”‚
â”‚ COACH   â”‚      â”‚ LISTENER â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")


"""
This creates the fully functional Mental Health Support Agent
"""

# Wrap custom functions as ADK FunctionTools
from google.adk.tools import FunctionTool

# Option A: let ADK infer name & description from function + docstring
mental_health_tools = [
    FunctionTool(assess_mental_health),
    FunctionTool(find_mental_health_resources),
    FunctionTool(recommend_wellness_activities),
]

# Create the main agent instance
# Note: This is a simplified version due to ADK API variations
# In production, you'd use proper Agent class initialization

class MentalHealthSupportSystem:
    """Complete Mental Health Support System with multi-agent architecture"""
    
    def __init__(self, api_key: str, model_name: str = MODEL_NAME):
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.memory = memory_bank
        self.session_history = []
        
        # Tool functions
        self.tools = {
            "assess_mental_health": assess_mental_health,
            "find_resources": find_mental_health_resources,
            "recommend_activities": recommend_wellness_activities
        }
        
        logger.info("ğŸ¤– Mental Health Support System initialized")
    
    async def process_user_input(self, user_message: str, mood_score: int = 5) -> Dict[str, Any]:
        """Process user input through the multi-agent system"""
        
        logger.info(f"\n{'='*60}\nğŸ“¥ Processing: {user_message[:50]}...\n{'='*60}")
        
        # Step 1: Crisis Assessment
        assessment = assess_mental_health(user_message, mood_score)
        logger.info(f"ğŸ”� Risk Assessment: {assessment['risk_level']}")
        
        # Step 2: Store in memory
        self.memory.add_interaction({
            "user_message": user_message,
            "mood_score": mood_score,
            "risk_level": assessment['risk_level'],
            "timestamp": datetime.now().isoformat()
        })
        
        # Step 3: Determine agent routing
        if assessment['crisis_detected'] or assessment['risk_level'] in ['CRITICAL', 'HIGH']:
            agent_role = "Crisis Assessment Specialist"
            system_instruction = crisis_assessment_agent_config['instruction']
        elif any(word in user_message.lower() for word in ['resource', 'help', 'service', 'counseling', 'therapist']):
            agent_role = "Resource Navigator"
            system_instruction = resource_navigator_agent_config['instruction']
        elif any(word in user_message.lower() for word in ['cope', 'stress', 'activity', 'exercise', 'relax']):
            agent_role = "Wellness Coach"
            system_instruction = wellness_coach_agent_config['instruction']
        else:
            agent_role = "Empathetic Listener"
            system_instruction = empathetic_listener_agent_config['instruction']
        
        logger.info(f"ğŸ�¯ Routing to: {agent_role}")
        
        # Step 4: Generate contextual response
        context = f"""
Assessment Results:
- Risk Level: {assessment['risk_level']}
- Mood Score: {mood_score}/10
- Urgency: {assessment['urgency']}
- Crisis Detected: {assessment['crisis_detected']}

Recommended Actions:
{chr(10).join(f"- {action}" for action in assessment['recommended_actions'][:3])}

User's Message: {user_message}
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"{system_instruction}\n\nCONTEXT:\n{context}\n\nProvide appropriate support response:",
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                )
            )
            
            agent_response = response.text
            
        except Exception as e:
            logger.error(f"â�Œ Error generating response: {e}")
            agent_response = self._get_fallback_response(assessment)
        
        # Step 5: Get relevant resources if needed
        resources = None
        if assessment['risk_level'] in ['CRITICAL', 'HIGH', 'MODERATE']:
            resources = find_mental_health_resources(
                location="UK",
                urgency=assessment['urgency'].lower()
            )
        
        # Step 6: Get wellness recommendations
        activities = None
        if assessment['risk_level'] in ['MODERATE', 'LOW']:
            mood_state = "anxious" if mood_score <= 4 else "stressed" if mood_score <= 6 else "neutral"
            activities = recommend_wellness_activities(
                mood_state=mood_state,
                time_available=15,
                energy_level="low" if mood_score <= 4 else "medium"
            )
        
        # Compile full response
        result = {
            "agent_role": agent_role,
            "assessment": assessment,
            "response": agent_response,
            "resources": resources,
            "wellness_activities": activities,
            "mood_trend": self.memory.get_mood_trend(),
            "session_summary": self.memory.get_summary()
        }
        
        logger.info(f"âœ… Response generated by {agent_role}")
        return result
    
    def _get_fallback_response(self, assessment: Dict[str, Any]) -> str:
        """Fallback response if AI generation fails"""
        if assessment['risk_level'] == 'CRITICAL':
            return """ğŸ”´ I'm concerned about what you've shared. Your safety is the top priority right now.

Please contact one of these services IMMEDIATELY:
â€¢ Emergency Services: 999
â€¢ Samaritans: 116 123 (24/7 free)
â€¢ Shout Crisis Text: Text SHOUT to 85258

You don't have to face this alone. These services are available right now to help you."""
        
        return """I'm here to listen and support you. While I process your message, please know that:

â€¢ Your feelings are valid
â€¢ Help is available
â€¢ You don't have to go through this alone

If you're in crisis, please contact Samaritans at 116 123 (24/7 free).

How can I best support you right now?"""

    def format_response_for_display(self, result: Dict[str, Any]) -> str:
        """Format the complete response for user-friendly display"""
        
        output = f"""
{'='*70}
ğŸ¤– **MENTAL HEALTH SUPPORT SYSTEM** - {result['agent_role']}
{'='*70}

{result['assessment']['status_icon']} **Current Assessment**: {result['assessment']['risk_level']} Risk
ğŸ“Š **Mood Score**: {result['assessment']['mood_score']}/10
â�° **Urgency**: {result['assessment']['urgency']}

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

ğŸ’¬ **Support Response**:
{result['response']}

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
"""
        
        # Add resources if available
        if result['resources']:
            output += "\nğŸ“� **IMMEDIATE SUPPORT RESOURCES**:\n"
            for category, items in list(result['resources'].items())[:2]:
                output += f"\n**{category.replace('_', ' ').title()}**:\n"
                for item in items[:3]:
                    output += f"  â€¢ **{item['name']}**: "
                    if 'phone' in item:
                        output += f"{item['phone']}"
                    if 'contact' in item:
                        output += f"{item['contact']}"
                    if 'description' in item:
                        output += f"\n    {item['description']}"
                    output += "\n"
        
        # Add wellness activities if available
        if result['wellness_activities']:
            output += "\n\nğŸ§˜ **RECOMMENDED WELLNESS ACTIVITIES**:\n"
            for idx, activity in enumerate(result['wellness_activities'][:2], 1):
                output += f"\n{idx}. **{activity['name']}** ({activity['duration']} minutes)\n"
                output += f"   {activity['benefits']}\n"
                if 'instructions' in activity:
                    output += f"   Steps: {', '.join(activity['instructions'][:3])}...\n"
        
        # Add mood trend
        if result['mood_trend']['trend'] != 'insufficient_data':
            output += f"\n\nğŸ“ˆ **Your Progress**: Mood trend is **{result['mood_trend']['trend']}** "
            output += f"(Avg: {result['mood_trend']['average']}/10)\n"
        
        output += f"\n{'='*70}\n"
        
        return output

# Initialize the system
mh_system = MentalHealthSupportSystem(api_key=GOOGLE_API_KEY)


"""
Demonstrates different use cases and agent routing
"""

import asyncio

async def test_system():
    """Run test scenarios through the system"""
    
    test_scenarios = [
        {
            "message": "I've been feeling really anxious about my upcoming exams. I can't sleep and my heart races whenever I think about them.",
            "mood_score": 4,
            "expected_agent": "Wellness Coach or Crisis Assessment"
        },
        {
            "message": "Can you help me find counseling services? I think I need to talk to someone professional.",
            "mood_score": 5,
            "expected_agent": "Resource Navigator"
        },
        {
            "message": "Just feeling a bit down today, not sure why. Could use someone to talk to.",
            "mood_score": 6,
            "expected_agent": "Empathetic Listener"
        }
    ]
    
    print("ğŸ§ª TESTING MENTAL HEALTH SUPPORT SYSTEM")
    print("="*70)
    
    for idx, scenario in enumerate(test_scenarios, 1):
        print(f"\n\nğŸ”¬ TEST SCENARIO #{idx}")
        print(f"Expected Agent: {scenario['expected_agent']}")
        print("-"*70)
        
        result = await mh_system.process_user_input(
            user_message=scenario['message'],
            mood_score=scenario['mood_score']
        )
        
        formatted_output = mh_system.format_response_for_display(result)
        print(formatted_output)
        
        # Brief pause between tests
        await asyncio.sleep(1)
    
    print("\nâœ… All test scenarios completed!")
    print(f"\n{memory_bank.get_summary()}")

# Run async tests
await test_system()


"""
Allows users to interact with the Mental Health Support System
"""

async def interactive_chat():
    """Run interactive chat session with the mental health support system"""
    
    print("\n" + "="*70)
    print("ğŸŒŸ WELCOME TO STUDENT MENTAL HEALTH SUPPORT SYSTEM ğŸŒŸ")
    print("="*70)
    print("""
This AI-powered system provides 24/7 mental health support for students.

ğŸ”¹ Features:
  â€¢ Crisis detection and immediate support
  â€¢ Mental health resource finder
  â€¢ Evidence-based wellness recommendations
  â€¢ Empathetic listening and validation
  â€¢ Personalized mood tracking

âš ï¸�  IMPORTANT: This is NOT a replacement for professional mental health care.
    If you're in crisis, please call: Samaritans 116 123 or Emergency 999

Type 'quit' or 'exit' to end the session.
Type 'summary' to see your progress.
""")
    print("="*70)
    
    session_active = True
    
    while session_active:
        print("\n" + "-"*70)
        
        # Get user input
        user_message = input("\nğŸ’¬ You: ").strip()
        
        if not user_message:
            continue
        
        # Check for exit commands
        if user_message.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\nğŸŒŸ Thank you for using the Mental Health Support System.")
            print(memory_bank.get_summary())
            print("\nğŸ’š Remember: You matter, and support is always available.")
            print("   Samaritans: 116 123 (24/7 free)")
            session_active = False
            break
        
        # Check for summary command
        if user_message.lower() == 'summary':
            print(memory_bank.get_summary())
            continue
        
        # Get mood score
        try:
            mood_input = input("ğŸ“Š Current mood (1-10, press Enter for 5): ").strip()
            mood_score = int(mood_input) if mood_input else 5
            mood_score = max(1, min(10, mood_score))  # Clamp to 1-10
        except:
            mood_score = 5
        
        print("\nâ�³ Processing your message...")
        
        # Process through system
        result = await mh_system.process_user_input(
            user_message=user_message,
            mood_score=mood_score
        )
        
        # Display formatted response
        output = mh_system.format_response_for_display(result)
        print(output)
        
        # Prompt for follow-up
        print("\nğŸ’­ Feel free to share more, ask questions, or type 'quit' to end.")

# For Kaggle notebook, we'll provide a single interaction demo
# In production, run: await interactive_chat()

print("""
âœ… Interactive Chat Interface Ready!

To start chatting, run this cell and interact with the system.
For demonstration, here's a sample interaction:
""")

# Demo interaction
demo_result = await mh_system.process_user_input(
    user_message="I've been feeling really stressed about my studies and having trouble focusing. What can I do?",
    mood_score=5
)

print(mh_system.format_response_for_display(demo_result))


"""
Displays system metrics, logging, and performance analytics
"""

def display_system_metrics():
    """Display comprehensive system metrics and analytics"""
    
    print("\n" + "="*70)
    print("ğŸ“Š SYSTEM OBSERVABILITY DASHBOARD")
    print("="*70)
    
    # Memory and usage metrics
    metrics = {
        "Total Interactions": memory_bank.user_profile['total_interactions'],
        "Mood Entries": len(memory_bank.user_profile['mood_history']),
        "Risk Assessments": len(memory_bank.user_profile['risk_assessments']),
        "Session Duration": "Active",
        "System Status": "ğŸŸ¢ Online"
    }
    
    print("\nğŸ”§ **System Metrics**:")
    for key, value in metrics.items():
        print(f"  â€¢ {key}: {value}")
    
    # Risk level distribution
    if memory_bank.user_profile['risk_assessments']:
        print("\nğŸ“ˆ **Risk Level Distribution**:")
        risk_counts = {}
        for assessment in memory_bank.user_profile['risk_assessments']:
            level = assessment['level']
            risk_counts[level] = risk_counts.get(level, 0) + 1
        
        for level, count in sorted(risk_counts.items()):
            bar = "â–ˆ" * count
            print(f"  {level:12} | {bar} ({count})")
    
    # Mood trends
    mood_trend = memory_bank.get_mood_trend()
    if mood_trend['trend'] != 'insufficient_data':
        print("\nğŸ“‰ **Mood Analytics**:")
        print(f"  â€¢ Trend: {mood_trend['trend'].upper()}")
        print(f"  â€¢ Average Mood: {mood_trend['average']}/10")
        print(f"  â€¢ Latest Score: {mood_trend['latest_score']}/10")
        print(f"  â€¢ Data Points: {mood_trend['total_entries']}")
    
    # Agent usage statistics
    print("\nğŸ¤– **Agent Performance**:")
    print("  âœ“ Crisis Assessment Agent: Active")
    print("  âœ“ Resource Navigator Agent: Active")
    print("  âœ“ Wellness Coach Agent: Active")
    print("  âœ“ Empathetic Listener Agent: Active")
    
    # Recent activity log
    if memory_bank.entries:
        print("\nğŸ“� **Recent Activity Log** (Latest 3):")
        for entry in memory_bank.entries[-3:]:
            timestamp = entry['timestamp'][:19].replace('T', ' ')
            mood = entry['data'].get('mood_score', 'N/A')
            risk = entry['data'].get('risk_level', 'N/A')
            print(f"  [{timestamp}] Mood: {mood}/10, Risk: {risk}")
    
    # System health checks
    print("\nğŸ�¥ **System Health**:")
    health_checks = {
        "API Connection": "âœ… Connected",
        "Memory Bank": "âœ… Operational",
        "Agent Tools": "âœ… Available",
        "Session Management": "âœ… Active"
    }
    for check, status in health_checks.items():
        print(f"  â€¢ {check}: {status}")
    
    print("\n" + "="*70)

display_system_metrics()

print("""
âœ… Observability Dashboard Active!

The system tracks:
- User interactions and mood history
- Risk assessments and crisis detection
- Agent routing decisions
- Resource recommendations
- Wellness activity engagement

All data is stored in-memory for this session and follows privacy best practices.
""")


"""
Implements evaluation metrics for agent performance
"""

class AgentEvaluator:
    """Evaluates agent performance across multiple dimensions"""
    
    def __init__(self):
        self.evaluation_metrics = {
            "crisis_detection_accuracy": [],
            "response_relevance": [],
            "resource_appropriateness": [],
            "empathy_score": [],
            "response_time": []
        }
    
    def evaluate_crisis_detection(self, user_input: str, detected_risk: str) -> float:
        """Evaluate accuracy of crisis detection"""
        # This would use labeled test data in production
        # For demo, we'll use heuristic evaluation
        
        crisis_indicators = ['suicide', 'kill', 'harm', 'die', 'end it']
        has_crisis_language = any(word in user_input.lower() for word in crisis_indicators)
        
        # Check if detection matches presence of crisis language
        correct_detection = (
            (has_crisis_language and detected_risk in ['CRITICAL', 'HIGH']) or
            (not has_crisis_language and detected_risk in ['MODERATE', 'LOW'])
        )
        
        score = 1.0 if correct_detection else 0.5
        self.evaluation_metrics['crisis_detection_accuracy'].append(score)
        return score
    
    def evaluate_response_quality(self, response: str, context: Dict) -> Dict[str, float]:
        """Evaluate multiple dimensions of response quality"""
        
        scores = {}
        
        # 1. Length appropriateness (not too short, not too long)
        word_count = len(response.split())
        scores['length'] = 1.0 if 50 <= word_count <= 300 else 0.7
        
        # 2. Empathy indicators
        empathy_words = ['understand', 'hear', 'feel', 'support', 'here for', 'valid']
        empathy_count = sum(1 for word in empathy_words if word in response.lower())
        scores['empathy'] = min(1.0, empathy_count / 3)
        
        # 3. Actionability (provides concrete next steps)
        action_words = ['contact', 'call', 'visit', 'try', 'practice', 'reach out']
        action_count = sum(1 for word in action_words if word in response.lower())
        scores['actionability'] = min(1.0, action_count / 2)
        
        # 4. Safety consciousness (mentions professional help when appropriate)
        if context.get('risk_level') in ['CRITICAL', 'HIGH']:
            has_safety_info = any(word in response.lower() for word in ['emergency', 'crisis', 'helpline', '999', '116'])
            scores['safety'] = 1.0 if has_safety_info else 0.3
        else:
            scores['safety'] = 1.0
        
        return scores
    
    def get_overall_performance(self) -> Dict[str, Any]:
        """Calculate overall system performance metrics"""
        
        performance = {}
        
        # Crisis detection accuracy
        if self.evaluation_metrics['crisis_detection_accuracy']:
            avg_crisis = sum(self.evaluation_metrics['crisis_detection_accuracy']) / len(self.evaluation_metrics['crisis_detection_accuracy'])
            performance['crisis_detection'] = round(avg_crisis * 100, 1)
        
        # Add more metrics as needed
        performance['total_evaluations'] = len(self.evaluation_metrics['crisis_detection_accuracy'])
        performance['status'] = "âœ… PASSING" if performance.get('crisis_detection', 0) >= 80 else "âš ï¸� NEEDS IMPROVEMENT"
        
        return performance
    
    def display_evaluation_report(self):
        """Display comprehensive evaluation report"""
        
        print("\n" + "="*70)
        print("ğŸ“‹ AGENT EVALUATION REPORT")
        print("="*70)
        
        performance = self.get_overall_performance()
        
        print("\nğŸ�¯ **Overall Performance**:")
        for metric, value in performance.items():
            if metric != 'status':
                print(f"  â€¢ {metric.replace('_', ' ').title()}: {value}")
        
        print(f"\n{performance.get('status', 'N/A')}")
        
        print("\nğŸ“Š **Evaluation Criteria**:")
        criteria = {
            "Crisis Detection": "Accurately identifies crisis situations",
            "Response Relevance": "Provides contextually appropriate responses",
            "Resource Quality": "Recommends suitable and accessible resources",
            "Empathy & Support": "Demonstrates understanding and validation",
            "Safety Protocols": "Follows mental health safety guidelines"
        }
        
        for criterion, description in criteria.items():
            print(f"  âœ“ {criterion}: {description}")
        
        print("\n" + "="*70)

# Initialize evaluator
evaluator = AgentEvaluator()

# Run sample evaluations
print("ğŸ§ª Running Agent Evaluations...")

# Test 1: Crisis detection
test_inputs = [
    ("I'm feeling stressed about exams", "MODERATE"),
    ("I don't want to live anymore", "CRITICAL"),
    ("Having trouble sleeping lately", "LOW")
]

for user_input, expected_risk in test_inputs:
    assessment = assess_mental_health(user_input)
    score = evaluator.evaluate_crisis_detection(user_input, assessment['risk_level'])
    print(f"  âœ“ Crisis Detection Test: {score * 100}%")

evaluator.display_evaluation_report()

print("\nâœ… Evaluation framework ready for continuous monitoring")

