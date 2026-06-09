


# Install Google ADK (Agent Development Kit)
!pip install google-adk google-genai --quiet

print("ADK installed successfully!")


# YouTube Automation Agent - Multi-Agent System with ADK
import os
from google import genai
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
from datetime import datetime
import json

# Configure Gemini API (use your API key from Kaggle secrets)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'your-api-key-here')
client = genai.Client(api_key=GOOGLE_API_KEY)

# ===== CUSTOM TOOLS =====

# Tool 1: Get Trending Topics
def get_trending_topics(niche: str) -> dict:
    """Fetches trending topics for a given YouTube niche."""
    # Simulated trending data (in production, use YouTube API)
    trends = {
        "tech": ["AI Agents Tutorial", "GPT-5 Features", "Coding with Gemini"],
        "gaming": ["GTA 6 Leaks", "Best Gaming Setup 2025", "Speedrun Records"],
        "lifestyle": ["Morning Routine", "Productivity Hacks", "Budget Travel"]
    }
    return {"niche": niche, "trending_topics": trends.get(niche.lower(), ["General Content Ideas"])}

trending_tool = FunctionTool(get_trending_topics)

# Tool 2: SEO Keyword Generator
def generate_seo_keywords(title: str) -> dict:
    """Generates SEO keywords for a video title."""
    keywords = title.lower().split() + ["tutorial", "2025", "how to", "best"]
    return {"title": title, "keywords": keywords[:10]}

seo_tool = FunctionTool(generate_seo_keywords)

# Tool 3: Schedule Optimizer
def get_optimal_schedule(timezone: str = "IST") -> dict:
    """Returns optimal upload times based on timezone."""
    schedules = {
        "IST": {"best_days": ["Saturday", "Sunday"], "best_time": "6:00 PM"},
        "PST": {"best_days": ["Tuesday", "Thursday"], "best_time": "3:00 PM"},
        "UTC": {"best_days": ["Wednesday", "Friday"], "best_time": "2:00 PM"}
    }
    return schedules.get(timezone, schedules["UTC"])

schedule_tool = FunctionTool(get_optimal_schedule)

print("Custom tools defined successfully!")


# ===== AGENT DEFINITIONS =====
# Using Kaggle Secrets for API Key
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# Initialize Gemini model
MODEL_ID = "gemini-2.0-flash"

# Agent 1: Idea Generator Agent
idea_agent = Agent(
    name="idea_generator",
    model=MODEL_ID,
    instruction="""You are a YouTube content idea generator. 
    Given a niche, generate 3 creative and trending video ideas.
    Use the trending_tool to get current trends.
    Be specific and engaging with your suggestions.""",
    tools=[trending_tool]
)

# Agent 2: Script Writer Agent  
script_agent = Agent(
    name="script_writer",
    model=MODEL_ID,
    instruction="""You are a YouTube script writer.
    Take a video idea and create a compelling 2-minute script outline.
    Include: Hook (10 sec), Main Content (90 sec), CTA (20 sec).
    Make it engaging and viewer-friendly."""
)

# Agent 3: Metadata Agent
metadata_agent = Agent(
    name="metadata_generator",
    model=MODEL_ID,
    instruction="""You are a YouTube SEO expert.
    Generate optimized metadata for the video:
    - Catchy title (under 60 chars)
    - Description (150 words)
    - 10 relevant tags
    Use the seo_tool for keyword suggestions.""",
    tools=[seo_tool]
)

# Agent 4: Scheduler Agent
scheduler_agent = Agent(
    name="upload_scheduler",
    model=MODEL_ID,
    instruction="""You are a YouTube upload scheduler.
    Recommend the best upload time based on the timezone.
    Use the schedule_tool to get optimal times.
    Provide specific day and time recommendations.""",
    tools=[schedule_tool]
)

print("All 4 agents created successfully!")


# ===== SESSIONS & MEMORY =====
# Create session service for memory persistence
session_service = InMemorySessionService()

# Store channel preferences in memory
channel_memory = {
    "channel_name": "TechWithTabish",
    "niche": "tech",
    "timezone": "IST",
    "preferred_style": "educational",
    "past_videos": ["Python Tutorial", "AI Basics", "Web Dev Tips"]
}

print("Session & Memory initialized!")
print(f"Channel: {channel_memory['channel_name']}")
print(f"Niche: {channel_memory['niche']}")
print(f"Timezone: {channel_memory['timezone']}")


# ===== DEMO: SEQUENTIAL AGENT WORKFLOW =====
# This demonstrates the multi-agent system in action

def run_youtube_automation(niche: str, timezone: str = "IST"):
    """
    Run the complete YouTube automation workflow.
    Demonstrates: Multi-Agent System + Custom Tools + Memory
    """
    print("="*60)
    print("ğŸ�¬ YOUTUBE AUTOMATION AGENT - DEMO")
    print("="*60)
    
    # Step 1: Get trending topics using custom tool
    print("\nğŸ“Š Step 1: Finding Trending Topics...")
    trends = get_trending_topics(niche)
    print(f"Niche: {trends['niche']}")
    print(f"Trending: {trends['trending_topics']}")
    
    # Step 2: Generate SEO keywords using custom tool
    print("\nğŸ”� Step 2: Generating SEO Keywords...")
    video_title = trends['trending_topics'][0] if trends['trending_topics'] else "Tutorial Video"
    seo_data = generate_seo_keywords(video_title)
    print(f"Title: {seo_data['title']}")
    print(f"Keywords: {seo_data['keywords']}")
    
    # Step 3: Get optimal schedule using custom tool
    print("\nğŸ“… Step 3: Finding Best Upload Time...")
    schedule = get_optimal_schedule(timezone)
    print(f"Best Days: {schedule['best_days']}")
    print(f"Best Time: {schedule['best_time']}")
    
    # Summary
    print("\n" + "="*60)
    print("âœ… AUTOMATION COMPLETE!")
    print("="*60)
    print(f"\nğŸ“¹ Video Topic: {video_title}")
    print(f"ğŸ�·ï¸� Tags: {', '.join(seo_data['keywords'][:5])}")
    print(f"â�° Upload: {schedule['best_days'][0]} at {schedule['best_time']}")
    
    return {
        "topic": video_title,
        "keywords": seo_data['keywords'],
        "schedule": schedule
    }

# Run the demo
result = run_youtube_automation(niche="tech", timezone="IST")
print("\nğŸ�‰ Demo completed successfully!")

