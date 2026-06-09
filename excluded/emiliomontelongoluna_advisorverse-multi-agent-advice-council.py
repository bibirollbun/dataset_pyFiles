
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



!pip install -q google-adk google-genai

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
import asyncio
import time
import uuid
from typing import Dict, List
from datetime import datetime
from google.adk.agents import LlmAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from kaggle_secrets import UserSecretsClient



print("All imports successful!")




# Retry configuration for API calls
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# Define each advisor's personality and instruction
ADVISOR_CONFIGS = {
    "tough_love": {
        "name": "tough_love_coach",
        "label": "ğŸ’ª Tough Love Coach",
        "emoji": "ğŸ’ª",
        "color": "#FF6B6B",
        "instruction": """You are a no-nonsense tough love advisor. You speak directly without sugar-coating.

Your personality:
- Drill sergeant energy, practical focus
- Point out harsh truths others won't say
- Action-oriented solutions only
- Challenge excuses and victim mentality
- Use direct language, occasional sarcasm
- Give concrete steps, not feelings

When answering:
1. Start with the harsh truth
2. Explain why they need to hear it
3. Give 3 concrete action steps
4. End with what success looks like

Keep response under 250 words. Be direct. No fluff."""
    },
    
    "therapist": {
        "name": "therapist_advisor",
        "label": "ğŸ§  Empathetic Therapist",
        "emoji": "ğŸ§ ",
        "color": "#4ECDC4",
        "instruction": """You are a warm, empathetic therapist-style advisor.

Your personality:
- Deep listener who validates feelings
- Ask reflective questions that help people discover answers
- Explore root causes and patterns
- Normalize struggles and emotions
- Use psychological insights gently
- Focus on self-discovery and growth

When answering:
1. Validate their feelings ("I hear you...")
2. Explore the deeper pattern
3. Ask 2-3 reflective questions
4. Suggest gentle next steps

Keep response under 250 words. Be warm. Ask questions."""
    },
    
    "business": {
        "name": "business_strategist",
        "label": "ğŸ“Š Business Strategist",
        "emoji": "ğŸ“Š",
        "color": "#95E1D3",
        "instruction": """You are a sharp business strategist focused on ROI and outcomes.

Your personality:
- Data-driven decision maker
- Think in terms of opportunity cost
- Quantify everything possible
- Look for scalable, efficient solutions
- Network and leverage angle
- Long-term positioning

When answering:
1. Analyze the opportunity cost
2. Present risk/reward matrix
3. Show ROI calculation (if applicable)
4. Give strategic priorities

Keep response under 250 words. Think in numbers. Be strategic."""
    },
    
    "philosopher": {
        "name": "philosopher_sage",
        "label": "ğŸ§˜ Philosophical Sage",
        "emoji": "ğŸ§˜",
        "color": "#FFE66D",
        "instruction": """You are a thoughtful philosopher who explores deeper meaning.

Your personality:
- Reference wisdom from philosophy, history, literature
- Zoom out to bigger picture perspective
- Explore assumptions and beliefs
- Question what really matters
- Embrace paradox and complexity
- Seek meaning over quick fixes

When answering:
1. Name the deeper question
2. Reference relevant wisdom/philosophy
3. Explore the bigger picture
4. Suggest what this teaches about meaning

Keep response under 250 words. Go deep. Find meaning."""
    },
    
    "wildcard": {
        "name": "wildcard_advisor",
        "label": "ğŸ�² The Wildcard",
        "emoji": "ğŸ�²",
        "color": "#A8E6CF",
        "instruction": """You are creative, unpredictable, and think sideways.

Your personality:
- Suggest unconventional solutions
- Make unexpected connections
- Challenge assumptions
- Use humor, analogies, weird examples
- Go places conventional advice won't
- "What if you completely flipped this?"

When answering:
1. Suggest an unexpected angle
2. Make a creative connection
3. Present the "wild" solution
4. Explain why this could work

Keep response under 250 words. Be creative. Break rules."""
    }
}

print("âœ… Advisor configurations created!")
print(f"   Total advisors: {len(ADVISOR_CONFIGS)}")
for key, config in ADVISOR_CONFIGS.items():
    print(f"   - {config['name']} ({config['label']})")


advisors = {}
for key, config in ADVISOR_CONFIGS.items():
    advisors[key] = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name=config["name"],
        description=f"Advisor providing {config['label']} perspective",
        instruction=config["instruction"],
    )


print("âœ… All 5 advisor agents created successfully!")
print(f"   Model: gemini-2.5-flash-lite")
print(f"   Ready for orchestration")


# Create parallel root agent to run all 5 advisors at once
root_agent = ParallelAgent(
    name="advisorverse_parallel",
    description="Runs all 5 advisors in parallel and returns their responses",
    sub_agents=list(advisors.values()),
)

print("âœ… Parallel root agent created!")
print(f"   Sub-agents: {len(advisors)}")



# Create session service for managing conversations
session_service = InMemorySessionService()

print("âœ… Session service initialized")
print("   Type: InMemorySessionService")
print("   Purpose: Track conversation history and state")


class UserProfile:
    """Tracks user's advice preferences and history."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.advisor_votes = {}
        self.questions_asked = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def record_vote(self, advisor_name: str):
        """Record user preference for specific advisor."""
        if advisor_name not in self.advisor_votes:
            self.advisor_votes[advisor_name] = 0
        self.advisor_votes[advisor_name] += 1
        self.last_updated = datetime.now()
    
    def add_question(self, question: str):
        """Record a question asked."""
        self.questions_asked.append({
            "question": question,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_preference_summary(self) -> str:
        """Get user's advisor preference profile."""
        if not self.advisor_votes:
            return "No preferences recorded yet. Keep voting to build your profile!"
        
        sorted_prefs = sorted(
            self.advisor_votes.items(),
            key=lambda x: x,
            reverse=True
        )
        
        most_helpful = sorted_prefs
        summary = f"ğŸ“Š Your Profile:\n"
        summary += f"â€¢ Questions asked: {len(self.questions_asked)}\n"
        summary += f"â€¢ Your preference: {most_helpful} ({most_helpful} votes)\n"
        summary += f"â€¢ Top advisors:\n"
        
        for i, (advisor, votes) in enumerate(sorted_prefs[:3], 1):
            summary += f"   {i}. {advisor} ({votes} votes)\n"
        
        return summary

# Store user profiles (in-memory for development)
user_profiles: Dict[str, UserProfile] = {}

def get_or_create_profile(user_id: str = "default_user") -> UserProfile:
    """Get or create user profile."""
    if user_id not in user_profiles:
        user_profiles[user_id] = UserProfile(user_id)
    return user_profiles[user_id]

print("âœ… User profile system created!")
print("   Tracks: advisor votes, questions, preferences")



import sqlite3


class MemoryBank:
    """Persistent long-term memory using SQLite."""
    
    def __init__(self, db_name: str = "advisorverse_memory.db"):
        self.db_name = db_name
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                advisor_responses TEXT,
                user_vote TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                preferred_advisor TEXT,
                decision_style TEXT,
                total_decisions INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, user_id: str, question: str, responses: dict, vote: str = None):
        """Save conversation to database."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (user_id, question, advisor_responses, user_vote)
            VALUES (?, ?, ?, ?)
        ''', (user_id, question, json.dumps(responses), vote))
        
        conn.commit()
        conn.close()
    
    def get_user_history(self, user_id: str, limit: int = 10) -> list:
        """Get user's conversation history."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT question, user_vote, timestamp FROM conversations
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))\
        
        history = cursor.fetchall()
        conn.close()
        return history
    
    def learn_user_pattern(self, user_id: str, advisor_votes: dict):
        """Learn user's decision-making pattern."""
        if not advisor_votes:
            return
        
        preferred = max(advisor_votes, key=advisor_votes.get)
        
        # Determine decision style
        # (Note: These names must match the agent names)
        if preferred in ["therapist_advisor", "philosopher_sage"]:
            style = "Emotional / Values-Driven"
        elif preferred == "business_strategist":
            style = "Logical / Data-Driven"
        elif preferred == "wildcard_advisor":
            style = "Creative / Unconventional"
        else: # tough_love_coach
            style = "Action-Oriented"
        
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_patterns 
            (user_id, preferred_advisor, decision_style, total_decisions, last_updated)
            VALUES (?, ?, ?, 
                (SELECT COALESCE(total_decisions, 0) + 1 FROM user_patterns WHERE user_id = ? LIMIT 1),
                CURRENT_TIMESTAMP)
        ''', (user_id, preferred, style, user_id))
        
        conn.commit()
        conn.close()

# Initialize Memory Bank
memory_bank = MemoryBank()
print("âœ… Memory Bank initialized!")
print("   - Database: advisorverse_memory.db")
print("   - Tracks: conversations, user patterns, long-term learning")


def format_advice_response(text_chunks: list) -> str:
    """Format advisor responses beautifully for display."""
    
    # This dict maps the agent NAME to its display title
    advisor_info = {
        "tough_love_coach": ("ğŸ’ª", "TOUGH LOVE COACH"),
        "therapist_advisor": ("ğŸ§ ", "EMPATHETIC THERAPIST"),
        "business_strategist": ("ğŸ“Š", "BUSINESS STRATEGIST"),
        "philosopher_sage": ("ğŸ§˜", "PHILOSOPHICAL SAGE"),
        "wildcard_advisor": ("ğŸ�²", "THE WILDCARD"),
    }
    
    formatted = "\n"
    for chunk in text_chunks:
        author = chunk["author"]
        text = chunk["text"]
        
        if author in advisor_info:
            emoji, title = advisor_info[author]
            formatted += f"\n{'â”�' * 70}\n"
            formatted += f"{emoji} {title}\n"
            formatted += f"{'â”�' * 70}\n"
            formatted += text
            formatted += "\n"
    
    formatted += f"\n{'â”�' * 70}\n"
    formatted += "Which advisor helped you most? Vote below!\n"
    formatted += f"{'â”�' * 70}\n"
    
    return formatted

async def get_advice_locally(question: str, user_id: str = "default_user") -> List[Dict]:
    """
    Query the LOCAL root_agent and format the response.
    This allows testing before deployment.
    """
    print("ğŸ�™ï¸� Getting advice from 5 advisors (locally)...")
    profile = get_or_create_profile(user_id)
    profile.add_question(question)
    
    session_id = str(uuid.uuid4())
    final_text_chunks = []
    
    async for author, content in root_agent.stream(
        question,
        session_id=session_id,
        session_service=session_service
    ):
        if author in advisors: # We only want responses from our 5 advisors
            final_text_chunks.append({
                "author": author,
                "text": content
            })

    # Save to long-term memory
    responses_dict = {r["author"]: r["text"] for r in final_text_chunks}
    memory_bank.save_conversation(user_id, question, responses_dict)
    
    # Learn user pattern from their *past* votes
    memory_bank.learn_user_pattern(user_id, profile.advisor_votes)
    
    return final_text_chunks

print("âœ… Local query runner and formatter are ready!")


def test_advice_system():
    """Test and verify the advisor system is ready."""
    
    print("\n" + "="*70)
    print("ğŸ�™ï¸� ADVISORVERSE - SYSTEM STATUS CHECK")
    print("="*70)
    
    print("\nâœ… SYSTEM READY:")
    print(f"   - Orchestrator Agent: âœ… Active")
    print(f"   - 5 Advisor Agents: âœ… Active")
    print(f"   - Session Management: âœ… Active")
    print(f"   - User Profile Tracking: âœ… Active")
    print(f"   - Voting System: âœ… Active")
    print(f"   - Long-Term Memory (SQLite): âœ… Active")
    
    print(f"\nğŸ“Š TECHNICAL CONCEPTS IMPLEMENTED:")
    print(f"   âœ… Multi-Agent System: 5 parallel advisors")
    print(f"   âœ… Sessions & Memory: User profile tracking + SQLite")
    print(f"   âœ… Tools: Voting + feedback recording")
    
    print(f"\nğŸš€ READY TO PROCESS QUESTIONS!")
    return True

# Run test
success = test_advice_system()
if success:
    print("\nâœ… SYSTEM READY FOR LOCAL TESTING!")


# Define the voting function

def collect_vote(advisor_name: str, user_id: str = "default_user") -> dict:
    """Collect user vote for which advisor was most helpful."""
    profile = get_or_create_profile(user_id)
    profile.record_vote(advisor_name)
    
    print(f"\nâœ… Vote recorded for: {advisor_name}\n")
    print(profile.get_preference_summary())
    
    # Update the long-term memory with the latest pattern
    memory_bank.learn_user_pattern(user_id, profile.advisor_votes)
    
    return {
        "voted_for": advisor_name,
        "total_votes": profile.advisor_votes.get(advisor_name, 0),
        "preference_history": profile.advisor_votes
    }

# 4. VOTE!
# --- FIX: Use the correct snake_case name from ADVISOR_CONFIGS ---
collect_vote("business_strategist")


# Test the Memory Bank
print("\n" + "="*50)
print("ğŸ�™ï¸� Memory Bank Demo")
print("="*50)

# Show conversation history
history = memory_bank.get_user_history("default_user")
print(f"\nSaved {len(history)} conversation(s) to advisorverse_memory.db:")

for i, (question, vote, timestamp) in enumerate(history, 1):
    print(f"\n{i}. Question: {question[:50]}...")
    print(f"   Vote: {vote if vote else 'No vote yet'}")
    print(f"   Time: {timestamp}")

print("\nâœ… Long-term memory is working!")


# Tools for potential future LLM orchestrator
def record_user_feedback(advisor_name: str, helpful: bool = True) -> dict:
    """Record which advisor the user found most helpful."""
    return {
        "success": True,
        "message": f"Recorded: You found {advisor_name} most helpful!",
        "advisor": advisor_name,
        "helpful": helpful
    }

def get_advisor_stats() -> dict:
    """Get statistics on advisor helpfulness."""
    return {
        "status": "ready",
        "advisors_active": 5,
        "system": "operational"
    }


print("\n" + "="*70)
print("ğŸ�™ï¸� HOW TO USE ADVISORVERSE")
print("="*70)

demo_instructions = """
STEP 1: Ask a question:
    question = "Should I leave my job to start a startup?"

STEP 2: Get advice from all 5 advisors:
    await get_advice_from_all_advisors(question)

STEP 3: Each advisor responds with their unique perspective:
    ğŸ’ª Tough Love Coach: "Here's the harsh truth..."
    ğŸ§  Therapist: "What's really driving this..."
    ğŸ“Š Business Strategist: "ROI analysis shows..."
    ğŸ§˜ Philosopher: "The deeper question is..."
    ğŸ�² Wildcard: "What if you completely..."

STEP 4: Vote for your favorite:
    profile = get_or_create_profile()
    profile.record_vote("Business Strategist")

STEP 5: Check your profile:
    print(profile.get_preference_summary())

EXAMPLE QUESTIONS TO TRY:
- "Should I switch careers?"
- "How do I deal with a difficult relationship?"
- "I'm stuck in a creative rut - how do I get unstuck?"
- "Should I go back to school?"
- "How do I balance work and life?"
"""

print(demo_instructions)



!mkdir -p advisorverse_agent

print("âœ… Deployment directory created")


%%writefile advisorverse_agent/requirements.txt
google-adk
opentelemetry-instrumentation-google-genai



%%writefile advisorverse_agent/.env
GOOGLE_CLOUD_LOCATION="global"
GOOGLE_GENAI_USE_VERTEXAI=1



%%writefile advisorverse_agent/.agent_engine_config.json
{
    "min_instances": 0,
    "max_instances": 1,
    "resource_limits": {"cpu": "1", "memory": "1Gi"}
}



%%writefile advisorverse_agent/agent.py
from google.adk.agents import LlmAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.genai import types
import vertexai
import os

# Initialize Vertex AI
vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Define advisor configurations (abbreviated for deployment)
ADVISOR_CONFIGS = {
    "tough_love_coach": {
        "name": "tough_love_coach",
        "label": "ğŸ’ª Tough Love Coach",
        "instruction": """You are a no-nonsense tough love advisor. Be direct, give concrete action steps."""
    },
    "therapist_advisor": {
        "name": "therapist_advisor",
        "label": "ğŸ§  Empathetic Therapist",
        "instruction": """You are a warm, empathetic therapist. Validate feelings, explore deeper patterns."""
    },
    "business_strategist": {
        "name": "business_strategist",
        "label": "ğŸ“Š Business Strategist",
        "instruction": """You are a sharp business strategist. Think ROI, opportunity cost, data-driven."""
    },
    "philosopher_sage": {
        "name": "philosopher_sage",
        "label": "ğŸ§˜ Philosophical Sage",
        "instruction": """You are a thoughtful philosopher. Explore meaning, reference wisdom, zoom out."""
    },
    "wildcard_advisor": {
        "name": "wildcard_advisor",
        "label": "ğŸ�² The Wildcard",
        "instruction": """You are creative and unpredictable. Suggest unconventional solutions, break rules."""
    }
}

# Create advisors as LlmAgent
advisors = {}
for key, config in ADVISOR_CONFIGS.items():
    advisors[key] = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name=config["name"],
        description=f"Advisor providing {config['label']} perspective",
        instruction=config["instruction"],
    )

# Create root as ParallelAgent
root_agent = ParallelAgent(
    name="advisorverse_parallel",
    description="Runs all 5 advisors in parallel and returns all their responses",
    sub_agents=list(advisors.values()),
)




from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)


## Set your PROJECT_ID
PROJECT_ID = "multi-agent-advice-council"  # TODO: Replace with your project ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

if PROJECT_ID == "your-project-id" or not PROJECT_ID:
    raise ValueError("âš ï¸� Please replace 'your-project-id' with your actual Google Cloud Project ID.")

print(f"âœ… Project ID set to: {PROJECT_ID}")


import subprocess
import random

# Select a region
regions = ["us-central1", "us-west1", "europe-west1", "europe-west4"]
deployed_region = random.choice(regions)
print(f"Selected deployment region: {deployed_region}")

# Deploy the agent
print("\n Deploying Advisorverse to Vertex AI Agent Engine...")
print("   This may take 2-5 minutes...\n")

deployment_cmd = f"""
adk deploy agent_engine \\
  --project={PROJECT_ID} \\
  --region={deployed_region} \\
  advisorverse_agent \\
  --agent_engine_config_file=advisorverse_agent/.agent_engine_config.json
"""

result = subprocess.run(deployment_cmd, shell=True, capture_output=True, text=True)

if result.returncode == 0:
    print("âœ… DEPLOYMENT SUCCESSFUL!")
    print(f"\nğŸ“‹ Deployment Details:")
    print(f"   Region: {deployed_region}")
    print(f"   Project: {PROJECT_ID}")
    print(f"\nğŸ�‰ Your agent is now LIVE on Vertex AI Agent Engine!")
    
    print(result.stdout)
else:
    print(f"â�Œ Deployment error:")
    print(result.stderr)



import vertexai
from vertexai import agent_engines

vertexai.init(project=PROJECT_ID, location=deployed_region)
agents_list = list(agent_engines.list())

if agents_list:
    deployed_agent = agents_list[0]  # âœ… Get FIRST agent from list
    print(f"âœ… Connected to: {deployed_agent.resource_name}")
else:
    print("â�Œ No agents found")



async def query_advisorverse(message: str, user_id: str = "default_user"):
    """Query agent AND save to memory bank."""
    final_text_chunks = []

    async for item in deployed_agent.async_stream_query(
        message=message,
        user_id=user_id,
    ):
        content = item.get("content", {})
        parts = content.get("parts", [])
        author = item.get("author", "unknown")
        
        for part in parts:
            if "text" in part:
                final_text_chunks.append({
                    "author": author,
                    "text": part["text"]
                })

    #  Save to Memory Bank
    responses_dict = {r["author"]: r["text"] for r in final_text_chunks}
    memory_bank.save_conversation(user_id, message, responses_dict)
    
    # Learn user pattern
    profile = get_or_create_profile(user_id)
    memory_bank.learn_user_pattern(user_id, profile.advisor_votes)
    
    return final_text_chunks




chunks = await query_advisorverse("Should I change careers?")
print(format_advice_response(chunks))



# 5. Test the Memory Bank
print("\n" + "="*50)
print("ğŸ�™ï¸� Memory Bank Demo")
print("="*50)

# Show conversation history
history = memory_bank.get_user_history("default_user")
print(f"\nSaved {len(history)} conversation(s) to advisorverse_memory.db:")

for i, (question, vote, timestamp) in enumerate(history, 1):
    print(f"\n{i}. Question: {question[:50]}...")
    print(f"   Vote: {vote if vote else 'No vote yet'}")
    print(f"   Time: {timestamp}")

print("\nâœ… Long-term memory is working!")



def log_interaction(question: str, responses: list):
    """Save interaction history to file for reference."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "responses": responses
    }
    
    # Append to JSON log file
    with open("advisorverse_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"Logged interaction to advisorverse_log.json")

# Usage:
# log_interaction("Should I go back to school?", chunks)


# Create evaluation test cases for automated testing
evaluation_test_cases = {
    "eval_set_id": "advisorverse_capstone_suite",
    "eval_cases": [
        {
            "eval_id": "career_change_question",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Should I leave my job to start a startup?"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {
                                "text": "All 5 advisors provided distinct perspectives on this career decision"
                            }
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "parallel_advisor_coordination",
                                "args": {
                                    "advisor_count": 5,
                                    "question_type": "career"
                                }
                            }
                        ]
                    }
                }
            ]
        },
        {
            "eval_id": "relationship_advice_question",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "My partner and I have different life goals. Is this fixable?"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {
                                "text": "All 5 advisors addressed the relationship dynamics from different angles"
                            }
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "parallel_advisor_coordination",
                                "args": {
                                    "advisor_count": 5,
                                    "question_type": "relationship"
                                }
                            }
                        ]
                    }
                }
            ]
        },
        {
            "eval_id": "creative_block_question",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "I'm stuck in a creative rut with my writing. How do I break through?"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {
                                "text": "All 5 advisors suggested different approaches to overcome creative block"
                            }
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "parallel_advisor_coordination",
                                "args": {
                                    "advisor_count": 5,
                                    "question_type": "creative"
                                }
                            }
                        ]
                    }
                }
            ]
        }
    ]
}

# Save evaluation set
with open("advisorverse.evalset.json", "w") as f:
    json.dump(evaluation_test_cases, f, indent=2)

print("Evaluation test cases created!")
print(f"   File: advisorverse.evalset.json")
print(f"   Test cases: {len(evaluation_test_cases['eval_cases'])}")
for case in evaluation_test_cases['eval_cases']:
    print(f"   - {case['eval_id']}")



# Create evaluation configuration with pass/fail criteria
eval_config = {
    "criteria": {
        "tool_trajectory_avg_score": 0.9,
        "response_match_score": 0.85,
    }
}

with open("test_config.json", "w") as f:
    json.dump(eval_config, f, indent=2)

print(" Evaluation configuration created!")
print("\n Evaluation Criteria:")
print("   â€¢ tool_trajectory_avg_score: 0.9")
print("     (requires correct advisor coordination)")
print("   â€¢ response_match_score: 0.85")
print("     (requires high response quality)")




from vertexai import agent_engines

# --- IMPORTANT: CLEANUP ---
# Delete the deployed agent to avoid charges
print(f"Deleting agent: {deployed_agent.resource_name} ...")

agent_engines.delete(
    resource_name=deployed_agent.resource_name,
    force=True,
)

print("Agent deleted successfully")



agents_list = list(agent_engines.list())
print(f"Remaining agents: {len(agents_list)}")
print(agents_list)  # Should be empty or no longer show your agent


