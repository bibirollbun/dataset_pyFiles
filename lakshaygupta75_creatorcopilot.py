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


# Step 1: Install Google ADK (Agent Development Kit)
%pip install -q -U google-adk



%pip show google-adk



from google.adk.agents import Agent

print("ADK imported OK")




import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ["GOOGLE_GENAI_API_KEY"] = user_secrets.get_secret("GOOGLE_GENAI_API_KEY")
print("API key loaded.")





print("OK — API key loaded.")



from google.genai import types



from google.genai import types
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search, AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

# Retry config (from course pattern)
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

def build_model():
    return Gemini(
        model=MODEL_NAME,
        retry_options=retry_config
    )



from typing import List, Dict

def score_ideas_tool(ideas: List[str]) -> Dict:
    """
    Scores content ideas with a simple heuristic so the planner
    can prioritize them.

    Args:
        ideas: List of idea descriptions.

    Returns:
        dict with status and a list of {idea, score}.
    """
    if not ideas:
        return {"status": "error", "error_message": "No ideas provided."}

    scored = []
    for idea in ideas:
        # Very dumb heuristic: longer + has strong words = higher score
        length_score = min(len(idea) / 80, 1.0)
        bonus_words = ["story", "secret", "mistake", "myths", "hack", "vs"]
        bonus = sum(w.lower() in idea.lower() for w in bonus_words) * 0.2
        score = round(min(length_score + bonus, 1.0), 2)
        scored.append({"idea": idea, "score": score})

    return {"status": "success", "ideas": scored}



from google.adk.models import Gemini

def build_model():
    # Simple model config, no extra retry options
    return Gemini(
        model="gemini-1.5-flash"   # you can also use "gemini-1.5-pro" if you want
    )



# 1) Agent: builds/updates user creator profile
profile_agent = Agent(
    name="profile_agent",
    model=build_model(),
    description="Builds a concise creator profile.",
    instruction="""
You are a content strategy expert.

From the user's description, extract this profile:
- niche (e.g. JEE prep, mental health, fashion, gaming)
- target audience (age, country, situation)
- main platforms (Reels, Shorts, YouTube, Blog, etc.)
- tone (motivational, educational, funny, emotional)
Return a short JSON-like summary, not code.
Example:
{"niche": "...", "audience": "...", "platforms": "...", "tone": "..."}
"""
)

# 2) Agent: trend research, uses built-in Google Search tool
trend_agent = Agent(
    name="trend_research_agent",
    model=build_model(),
    description="Finds trending questions/topics.",
    instruction="""
You research trending topics for the creator's niche and audience.
Use Google Search when helpful.

Input: creator profile.
Output: 5-10 short 'trend seeds' as bullet points that
reflect current or evergreen interests in that niche.
""",
    tools=[google_search],
)

# 3) Agent: idea generator
idea_agent = Agent(
    name="idea_generator_agent",
    model=build_model(),
    description="Generates concrete content ideas.",
    instruction="""
You are a creative content brainstormer.

Given:
- creator profile
- trend seeds

Generate 10-15 specific content ideas.
Group them by platform if helpful (Reels, YouTube, Blog).
Make each idea concrete, not generic. Avoid repetition.
Return a numbered list of ideas as plain text.
"""
)

# 4) Agent: hook & title generator
hook_agent = Agent(
    name="hook_and_title_agent",
    model=build_model(),
    description="Writes hooks, titles, and CTAs.",
    instruction="""
You turn brainstormed content ideas into:
- 1 viral-style hook (max 1 line)
- 1 platform-appropriate title
- 1-2 CTAs (e.g. save, share, comment, follow)

Input: numbered list of ideas.
Output: a structured list where each idea has:
Idea #: ...
Hook: ...
Title: ...
CTA: ...
"""
)

# 5) Agent: planner, uses our custom scoring tool
planner_agent = Agent(
    name="planner_agent",
    model=build_model(),
    description="Creates a 7-day content plan.",
    instruction="""
You are a planning agent.

You receive:
- creator profile
- ideas with hooks & titles

First, extract a plain list of idea descriptions and call
the `score_ideas_tool` tool to get engagement scores.
Use the scores to pick a balanced top set of ideas.

Then create a 7-day schedule:
- each day has 1-2 pieces of content
- vary formats and topics
- display plan as a Markdown table with columns:
  [Day, Idea, Hook/Title, Why it works]

Focus on clarity and variety.
""",
    tools=[score_ideas_tool],
)



# Wrap sub-agents as tools
profile_tool = AgentTool(agent=profile_agent)
trend_tool = AgentTool(agent=trend_agent)
idea_tool = AgentTool(agent=idea_agent)
hook_tool = AgentTool(agent=hook_agent)
planner_tool = AgentTool(agent=planner_agent)

# Root multi-agent system
root_agent = Agent(
    name="creator_copilot",
    model=build_model(),
    description="Multi-agent creative companion for content creators.",
    instruction="""
You are CreatorCopilot, a multi-agent creative companion for
content creators (Reels, Shorts, YouTube, Blogs).

You have these tools:
- profile_agent: build/update creator profile
- trend_research_agent: get trend seeds using search
- idea_generator_agent: generate content ideas
- hook_and_title_agent: create hooks/titles/CTAs
- planner_agent: create a 7-day posting plan using scoring tool

When the user gives you their niche/audience/platforms:
1. Call profile_agent to build/update the creator profile.
2. Call trend_research_agent using that profile.
3. Call idea_generator_agent with profile + trends.
4. Call hook_and_title_agent on the ideas.
5. Call planner_agent to build a 7-day schedule.
6. Return:
   - a short summary of the creator profile,
   - 8-12 best ideas with hooks/titles/CTAs,
   - the 7-day posting plan as a table.

Keep output compact and actionable.
If user asks follow-up like 'give me more ideas', reuse the
existing profile and only re-run steps 2-5.
""",
    tools=[profile_tool, trend_tool, idea_tool, hook_tool, planner_tool],
)



# Session and memory services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# Runner with memory & sessions
runner = Runner(
    agent=root_agent,
    app_name="creator_copilot_app",
    session_service=session_service,
    memory_service=memory_service,
)



import asyncio

USER_ID = "demo_user"

async def run_creator_copilot(message: str, session_name: str = "session1"):
    # Make sure the session exists
    session = await session_service.create_session(
        app_name=runner.app_name,
        user_id=USER_ID,
        session_id=session_name,
    )

    print(f"\nUSER > {message}\n")
    last_text = ""

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=message)]
        ),
    ):
        # stream-like printing
        if event.content and event.content.parts:
            part_text = event.content.parts[0].text
            if part_text and part_text != last_text:
                print("ASSISTANT >", part_text)
                last_text = part_text

    return last_text



from google.adk.sessions import InMemorySessionService

# Unique ID for you (can be any string)
USER_ID = "user_1"

# In-memory session storage used by the runner
session_service = InMemorySessionService()



test_prompt = """
I am a college student in India making Instagram Reels and YouTube Shorts
about study motivation and productivity for JEE/NEET aspirants
(ages 16-19). My tone is motivational, honest, and slightly funny.

Please give me ideas and a 7-day plan.
"""

from google.genai import types

async def run_creator_copilot(message: str, session_name: str = "session1"):
    # 1. Create or reuse session
    try:
        session = await session_service.create_session(
            app_name=runner.app_name,
            user_id=USER_ID,
            state={},
            session_id=session_name,
        )
    except Exception:
        # If it already exists, just get it
        session = await session_service.get_session(
            app_name=runner.app_name,
            user_id=USER_ID,
            session_id=session_name,
        )

    # 2. Wrap your string into a Content object (what ADK expects)
    content = types.Content(
        role="user",
        parts=[types.Part(text=message)]
    )

    final_text = ""

    # 3. Run the agent
    events = runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=content,
    )

    async for event in events:
        # You can print all events for debugging if you want:
        # print(event)

        # ADK pattern: check for final response
        if hasattr(event, "is_final_response") and event.is_final_response():
            if getattr(event, "content", None) and event.content.parts:
                final_text = event.content.parts[0].text

    return final_text




    

         



from google.genai import types

USER_ID = "user_1"  # any string is fine

async def run_creator_copilot(message: str):
    # 1) Wrap the user message in Content (what ADK expects)
    content = types.Content(
        role="user",
        parts=[types.Part(text=message)]
    )

    final_text = ""

    # 2) Call the runner WITHOUT a session_id (let ADK handle sessions)
    events = runner.run_async(
        user_id=USER_ID,
        session_id=None,        # <--- important: no manual session
        new_message=content,
    )

    async for event in events:
        if getattr(event, "partial", False):
            continue

        # Try common ways to read final text
        if hasattr(event, "text") and event.text:
            final_text = event.text
        elif hasattr(event, "output_text") and event.output_text:
            final_text = event.output_text
        elif hasattr(event, "content") and event.content and event.content.parts:
            # Join all text parts if present
            parts_text = []
            for p in event.content.parts:
                if hasattr(p, "text") and p.text:
                    parts_text.append(p.text)
            if parts_text:
                final_text = "".join(parts_text)

    return final_text



test_prompt = """
I am a college student in India making Instagram Reels and YouTube Shorts
about study motivation and productivity for JEE/NEET aspirants
(ages 16-19). My tone is motivational, honest, and slightly funny.

Please give me ideas and a 7-day plan.
"""

sample_output = """
[Creator Copilot – 7-Day Content Plan]

User profile:
- Niche: JEE/NEET study motivation & productivity
- Audience: 16–19-year aspirants in India
- Platforms: Instagram Reels, YouTube Shorts
- Tone: motivational, honest, slightly funny

DAY 1 – Reality Check + Hope
Hook: "If your last test went trash, this video is for you — not toppers."
Idea: Talk honestly about bad tests, that one bad score ≠ whole life, and how real toppers bounce back.
CTA: "Comment 'RESET' if you're restarting from today."

DAY 2 – Realistic Morning Routine
Hook: "This is my ‘non-Instagram’ morning routine as a JEE/NEET student."
Idea: Show waking up, first study block, no-phone rule, and admit you don’t follow it perfectly every day.
CTA: "Save this reel as your morning checklist."

DAY 3 – 10-Minute Focus Hack
Hook: "If you can scroll for 40 minutes, you can study for 10 minutes. Try this."
Idea: Explain a 10-min focus rule: one topic, timer on, no decoration, just questions.
CTA: "Type '10 MIN' if you're trying this today."

DAY 4 – Meme + Lesson: The 5-Minute Break
Hook: "POV: You took a 5-minute break… 3 hours ago."
Idea: Skit of '5-min break' turning into scrolling, then show how you now keep phone in another room & use alarms.
CTA: "Tag that one friend always on a 5-min break."

DAY 5 – Notes & Active Recall
Hook: "Stop making 'Instagram notes'. Start making topper notes."
Idea: Show before/after notes: aesthetic vs short question-based notes and a 'mistake notebook' for wrong questions.
CTA: "Comment 'NOTES' if you want a full video on my notes."

DAY 6 – Study With Me (Realistic)
Hook: "Study with me (I also want to check my phone every 2 mins)."
Idea: Time-lapse of 20–25 min focused study, with overlay: 'Discipline = sitting even when you’re not motivated'.
CTA: "Use this reel as background and study for 25 mins."

DAY 7 – Sunday Reset & Planning
Hook: "If today is Sunday, watch this before you waste it."
Idea: Show weekly reset: write what went wrong, pick 3 chapters, block 3 study slots per day.
CTA: "Comment 'PLAN' if you want my weekly planning template."
"""

print("USER PROMPT:\n", test_prompt)
print("\n--- SAMPLE AGENT OUTPUT ---\n")
print(sample_output)


