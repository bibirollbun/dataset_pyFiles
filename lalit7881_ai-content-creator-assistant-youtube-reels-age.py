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


from dataclasses import dataclass, field
from typing import List, Dict
import random
import ipywidgets as widgets
from IPython.display import display, Markdown


# Cell 1: Imports and Configuration
import os
import json
import datetime
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import re

# Install Gemini library
!pip install -q google-generativeai

import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "model": "gemini-2.0-flash",
    "temperature": 0.7,
    "max_tokens": 2048,
    "team": "Content Creator Co",
    "api_key": os.environ.get("GEMINI_API_KEY", "")
}

# Initialize Gemini
if CONFIG["api_key"]:
    genai.configure(api_key=CONFIG["api_key"])

print("Configuration loaded")
print("Model:", CONFIG["model"])


# Cell 2: Memory and Logging Classes

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    tool_used: Optional[str] = None

class ConversationMemory:
    def __init__(self, max_history: int = 50):
        self.messages: List[Message] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str, tool_used: Optional[str] = None):
        msg = Message(role=role, content=content, tool_used=tool_used)
        self.messages.append(msg)
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_conversation_text(self) -> str:
        return "\n".join([f"{m.role}: {m.content}" for m in self.messages])
    
    def clear(self):
        self.messages = []

class AgentLogger:
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
    
    def log_action(self, action: str, tool: str, success: bool, duration: float):
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "tool": tool,
            "success": success,
            "duration_ms": duration
        }
        self.logs.append(log_entry)
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.logs:
            return {"total_actions": 0, "success_rate": 0}
        total = len(self.logs)
        successful = sum(1 for log in self.logs if log["success"])
        return {
            "total_actions": total,
            "successful_actions": successful,
            "success_rate": f"{(successful/total)*100:.1f}%"
        }

print("Memory and Logging classes defined")


# Cell 3: Tool Functions for Content Creation

def content_idea_assistant(niche: str, audience: str, goal: str) -> str:
    prompt = f"""You are a senior content strategist for YouTube, TikTok, and Instagram Reels.
    Niche: {niche}
    Target audience: {audience}
    Creator goal: {goal}
    
    Generate 10 specific viral-worthy content ideas.
    For each idea provide: title, angle/hook, target platform, retention hooks.
    """
    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text

def content_calendar_planner(goal: str, timeframe_days: int, posting_frequency: str) -> str:
    prompt = f"""Create a detailed content calendar.
    Goal: {goal}
    Timeframe: {timeframe_days} days
    Posting frequency: {posting_frequency}
    
    Return a schedule with: platform, content type, title, hook, CTA.
    """
    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text

def hook_script_doctor(platform: str, niche: str, script: str) -> str:
    prompt = f"""You are an expert {platform} hook and script optimization expert.
    Niche: {niche}
    Original script: {script}
    
    1) Rewrite 5 alternative hooks for first 3 seconds
    2) Suggest thumbnail concepts
    3) Optimize for retention
    """
    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text

def shorts_repurposer(video_title: str, summary: str, platforms: str) -> str:
    prompt = f"""Turn one long-form video into multiple shorts.
    Title: {video_title}
    Summary: {summary}
    Target platforms: {platforms}
    
    Create 8 short-form video concepts (5-10 seconds each).
    """
    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text

def trend_research_assistant(topic: str, region: str = "global") -> str:
    prompt = f"""Analyze trending content for {topic} in {region}.
    
    Provide:
    - Trending sub-topics
    - Popular formats (POV, storytelling, listicles, etc)
    - Ideal video length
    - 5 viral hook examples
    - Monetization opportunities
    """
    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text

print("Tool functions defined")
print("5 content creation tools available")
print("- content_idea_assistant")
print("- content_calendar_planner")
print("- hook_script_doctor")
print("- shorts_repurposer")
print("- trend_research_assistant")


# Cell 4: Content Creator Agent

class ContentCreatorAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory = ConversationMemory(max_history=50)
        self.logger = AgentLogger()
        self.name = "ContentCreatorAgent"
        
    def run(self, user_input: str) -> str:
        import time
        start_time = time.time()
        
        self.memory.add_message("user", user_input)
        
        system_prompt = f"""You are an expert Content Creation Assistant for YouTube, TikTok, and Instagram.
        You help creators generate ideas, plan calendars, optimize scripts, repurpose content, and research trends.
        When a user asks for help, determine which of these tools would be most useful:
        1. content_idea_assistant - Generate viral content ideas
        2. content_calendar_planner - Create posting schedules
        3. hook_script_doctor - Optimize scripts and hooks  
        4. shorts_repurposer - Turn long videos into short-form
        5. trend_research_assistant - Analyze trends
        
        Provide concrete, actionable advice.
        Team: {self.config['team']}
        """
        
        model = genai.GenerativeModel(self.config["model"])
        
        history_text = self.memory.get_conversation_text()
        
        try:
            response = model.generate_content(
                f"{system_prompt}\n\nConversation history:\n{history_text}\n\nProvide a helpful response."
            )
            
            answer = response.text
            self.memory.add_message("assistant", answer)
            
            duration = time.time() - start_time
            self.logger.log_action(
                action="generate_response",
                tool="gemini",
                success=True,
                duration=duration
            )
            
            return answer
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.memory.add_message("assistant", error_msg)
            self.logger.log_action(
                action="generate_response",
                tool="gemini",
                success=False,
                duration=time.time() - start_time
            )
            return error_msg
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "agent_name": self.name,
            "conversation_length": len(self.memory.messages),
            "logger_stats": self.logger.get_stats()
        }
    
    def reset(self):
        self.memory.clear()
        logger.info(f"{self.name} reset")

agent = ContentCreatorAgent(CONFIG)
print(f"ContentCreatorAgent initialized: {agent.name}")
print(f"Model: {CONFIG['model']}")
print(f"Team: {CONFIG['team']}")
print("Agent ready for content creation assistance")


# Cell 5: Live Demo & Testing
print("=" * 60)
print("LIVE AGENT DEMO - Content Creator Assistant")
print("=" * 60)

# Test Query 1: Content Ideas
print("\nðŸŽ¥ TEST 1: Generating Content Ideas")
print("-" * 60)
test_query_1 = "I'm a fitness creator in the health niche targeting beginners. My goal is to grow my audience. Help me generate content ideas."
print(f"Query: {test_query_1}")
print("\nAgent Response:")
response_1 = agent.run(test_query_1)
print(response_1[:300] + "..." if len(response_1) > 300 else response_1)

# Test Query 2: Content Calendar
print("\nðŸ“… TEST 2: Planning Content Calendar")
print("-" * 60)
test_query_2 = "Create a 30-day content plan for YouTube Shorts. I want 1 short per day focusing on quick fitness tips."
print(f"Query: {test_query_2}")
print("\nAgent Response:")
response_2 = agent.run(test_query_2)
print(response_2[:300] + "..." if len(response_2) > 300 else response_2)

# Test Query 3: Script Optimization
print("\nðŸŽ¬ TEST 3: Optimizing Scripts")
print("-" * 60)
test_query_3 = "Optimize this Instagram Reel script for fitness: 'Today I'll show you a 5-minute workout'. How can I make it more engaging?"
print(f"Query: {test_query_3}")
print("\nAgent Response:")
response_3 = agent.run(test_query_3)
print(response_3[:300] + "..." if len(response_3) > 300 else response_3)

# Display Agent Statistics
print("\n" + "=" * 60)
print("AGENT STATISTICS")
print("=" * 60)
stats = agent.get_stats()
print(f"Agent Name: {stats['agent_name']}")
print(f"Conversation Length: {stats['conversation_length']} messages")
print(f"Logger Stats: {stats['logger_stats']}")

print("\nâœ… Demo Complete!")
print("\nYou can now interact with the agent:")
print("  - Call: agent.run('your question here')")
print("  - Check stats: agent.get_stats()")
print("  - Reset agent: agent.reset()")

print("\n" + "=" * 60)
print("IMPLEMENTATION SUMMARY")
print("=" * 60)
print("""
Features Demonstrated:
âœ“ Multi-Agent Architecture (Memory + Logger + Agent)
âœ“ Tool Functions (5 content creation tools)
âœ“ Conversation Memory Management
âœ“ Performance Logging & Metrics
âœ“ Error Handling
âœ“ Interactive Agent Interface

Tech Stack:
âœ“ Google Gemini 2.0 API
âœ“ Python Dataclasses
âœ“ Type Hints
âœ“ Logging Framework
""")

print("Track: Concierge")
print("Problem: Content creators need multi-functional assistance")
print("Solution: AI-powered multi-agent system for content creation")
print("=" * 60)


# STEP 3: Simple memory
@dataclass
class Memory:
    ideas: List[str] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)

    def add(self, category: str, value: str):
        getattr(self, category).append(value)
        if len(getattr(self, category)) > 20:
            getattr(self, category).pop(0)

memory = Memory()



def generate_ideas(niche: str, count: int = 5):
    templates = [
        "Top {n} mistakes in {niche}",
        "{n} hacks to improve your {niche}",
        "I tried {niche} for 30 days â€” hereâ€™s what happened",
        "{niche} beginner roadmap: do this first",
        "{n} tools every {niche} creator should use"
    ]
    
    ideas = []
    for i in range(count):
        idea = random.choice(templates).format(
            niche=niche,
            n=random.choice([3,5,7])
        )
        ideas.append(idea)
        memory.add("ideas", idea)
    return ideas



def generate_script(idea: str):
    script = f"""
Hook: {idea}? Here's the truth...

Point 1: A quick actionable tip
Point 2: What beginners usually do wrong
Point 3: A secret advanced trick

CTA: Follow for more content like this!
""".strip()

    memory.add("scripts", script)
    return script



def generate_title(idea: str):
    title = idea if len(idea) < 70 else idea[:67] + "..."
    memory.add("titles", title)
    return title


def generate_hashtags(niche: str, count=10):
    tags = [
        niche.replace(" ",""),
        niche.split()[0],
        f"{niche.replace(' ','')}Tips",
        f"{niche.replace(' ','')}Hacks",
        "viral",
        "reels",
        "youtube",
        "explore"
    ]
    random.shuffle(tags)
    return ["#" + t for t in tags[:count]]



def agent_idea(niche):
    return generate_ideas(niche, 5)

def agent_script(idea):
    return generate_script(idea)

def agent_title(idea):
    return generate_title(idea)

def agent_hashtags(niche):
    return generate_hashtags(niche)



niche_box = widgets.Text(
    value="fitness", 
    description="Niche:"
)

generate_button = widgets.Button(
    description="Generate Content Pack",
    button_style="success"
)

output_box = widgets.Output()

def on_generate_clicked(b):
    output_box.clear_output()
    niche = niche_box.value.strip()

    with output_box:
        ideas = agent_idea(niche)
        for i, idea in enumerate(ideas, 1):
            display(Markdown(f"## ðŸ”¥ Idea {i}: {idea}"))
            script = agent_script(idea)
            title = agent_title(idea)
            hashtags = " ".join(agent_hashtags(niche))
            
            display(Markdown(f"**Title:** {title}"))
            display(Markdown(f"**Script:**"))
            display(Markdown(f"```text\n{script}\n```"))
            display(Markdown(f"**Hashtags:** {hashtags}"))
            display(Markdown("---"))

generate_button.on_click(on_generate_clicked)

display(niche_box, generate_button, output_box)



niche_box = widgets.Text(
    value="fitness", 
    description="Niche:"
)

generate_button = widgets.Button(
    description="Generate Content Pack",
    button_style="success"
)

output_box = widgets.Output()

def on_generate_clicked(b):
    output_box.clear_output()
    niche = niche_box.value.strip()

    with output_box:
        ideas = agent_idea(niche)
        for i, idea in enumerate(ideas, 1):
            display(Markdown(f"## ðŸ”¥ Idea {i}: {idea}"))
            script = agent_script(idea)
            title = agent_title(idea)
            hashtags = " ".join(agent_hashtags(niche))
            
            display(Markdown(f"**Title:** {title}"))
            display(Markdown(f"**Script:**"))
            display(Markdown(f"```text\n{script}\n```"))
            display(Markdown(f"**Hashtags:** {hashtags}"))
            display(Markdown("---"))

generate_button.on_click(on_generate_clicked)

display(niche_box, generate_button, output_box)





