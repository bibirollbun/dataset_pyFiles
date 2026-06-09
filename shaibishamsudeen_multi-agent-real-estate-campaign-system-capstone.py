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


!pip install google-adk google-generativeai -q

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService, Session
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from kaggle_secrets import UserSecretsClient
import os
import requests
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List
import uuid
from datetime import datetime
import time

print("âœ… All imports successful")


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


try:
    PERPLEXITY_API_KEY = UserSecretsClient().get_secret("PERPLEXITY_API_KEY")
    os.environ["PERPLEXITY_API_KEY"] = PERPLEXITY_API_KEY
    print("âœ… Perplexity API key loaded")
except Exception as e:
    print(f"âš ï¸� Error loading Perplexity key: {e}")


from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("âœ… ADK components imported successfully.")


from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

APP_NAME = "real_estate_campaign_concierge"
USER_ID = "demo_user"

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

print("âœ… ADK Session & Memory services initialized")
print(f"   App: {APP_NAME}")
print(f"   User: {USER_ID}")


def _now():
    return datetime.utcnow().isoformat()

@dataclass
class SessionMemory:
    brand_voice: str = "trust-building, safety-first, data-backed"
    platforms: List[str] = field(default_factory=lambda: ["instagram", "linkedin", "medium", "official_site"])

class CampaignMemory:
    def __init__(self):
        self._campaigns: Dict[str, Dict[str, Any]] = {}

    def create_campaign(self, idea: str) -> str:
        cid = str(uuid.uuid4())
        self._campaigns[cid] = {"idea": idea, "created_at": _now(), "data": {}}
        return cid

    def set(self, campaign_id: str, key: str, value: Any):
        self._campaigns[campaign_id]["data"][key] = value

    def get(self, campaign_id: str, key: str, default=None):
        return self._campaigns[campaign_id]["data"].get(key, default)

class CrossCampaignMemory:
    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def add_entry(self, entry: Dict[str, Any]):
        self._history.append(entry)

    def get_relevant(self, persona: str, category: str, limit: int = 3):
        candidates = [h for h in self._history
                      if h.get("buyer_persona") == persona and h.get("buyer_category") == category]
        return candidates[-limit:]

@dataclass
class QueueItem:
    id: str
    campaign_id: str
    item_type: str          # "social_post" | "blog" | "video_package"
    target_platform: str    # "instagram" | "linkedin" | "medium" | "official_site" | "video"
    platform_handle: str
    content_payload: Dict[str, Any]
    status: str = "PENDING"
    reviewer_comments: str = ""
    edit_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

class ApprovalQueue:
    def __init__(self):
        self._items: Dict[str, QueueItem] = {}

    def add_item(self, campaign_id, item_type, target_platform, platform_handle, payload):
        qid = str(uuid.uuid4())
        self._items[qid] = QueueItem(
            id=qid,
            campaign_id=campaign_id,
            item_type=item_type,
            target_platform=target_platform,
            platform_handle=platform_handle,
            content_payload=payload,
        )
        return qid

    def list_pending(self, campaign_id=None):
        return [i for i in self._items.values()
                if i.status == "PENDING" and (campaign_id is None or i.campaign_id == campaign_id)]



session_mem = SessionMemory()
campaign_mem = CampaignMemory()
cross_mem = CrossCampaignMemory()
approval_queue = ApprovalQueue()

print("âœ… Project memory + queue initialised.")



# from google import adk
# import os
# import requests
# import json

def fetch_property_data(project_query: str) -> dict:
    """
    Use Perplexity API to retrieve structured info about a real estate project.
    `project_query` can be a URL, or 'project name + location', or both.
    Returns a normalized JSON dict for downstream agents.
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    # Ask Perplexity to respond with strict JSON in a fixed schema
    system_prompt = (
        "You are a data extraction agent for real estate projects. "
        "Given the query below, return ONLY a JSON object with these keys:\n"
        "name (string), location (string), price_range (string or null), "
        "unit_types (array of strings), amenities (array of strings), "
        "developer (string or null), project_type (string or null), "
        "launch_status (string or null), source_urls (array of strings).\n"
        "If a field is unknown, set it to null or an empty array.\n\n"
        f"Query: {project_query}"
    )

    payload = {
        "model": "sonar",          # or your preferred Perplexity model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": project_query}
        ],
        "max_tokens": 512,
        "temperature": 0.1,
        # "response_format": {"type": "json_object"}
    }

    
    url = "https://api.perplexity.ai/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=25)
    if resp.status_code != 200:
        # Helpful debug print
        print("Perplexity error:", resp.status_code, resp.text[:500])
        resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Try to parse JSON; if model returned text, try to extract JSON
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # very simple fallback: wrap raw content
            parsed = {"raw_content": content}
    else:
        parsed = content

    # Normalize to your internal schema
    return {
        "name": parsed.get("name"),
        "location": parsed.get("location"),
        "price_range": parsed.get("price_range"),
        "unit_types": parsed.get("unit_types", []),
        "amenities": parsed.get("amenities", []),
        "developer": parsed.get("developer"),
        "project_type": parsed.get("project_type"),
        "launch_status": parsed.get("launch_status"),
        "source_urls": parsed.get("source_urls", []),
        "query": project_query,
    }
def search_web_info(query: str) -> dict:
    """
    Search the web for specific information using Perplexity.
    Use this for detailed queries like 'Altair 52 Dubai South price range',
    'Altair 52 payment plans', 'schools near Dubai South', etc.
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "user", "content": query}
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }
    
    url = "https://api.perplexity.ai/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=25)
    
    if resp.status_code != 200:
        print("Perplexity error:", resp.status_code, resp.text[:500])
        resp.raise_for_status()
    
    data = resp.json()
    return {
        "answer": data["choices"][0]["message"]["content"],
        "query": query
    }


from google.adk.agents import LlmAgent

# Initialize Gemini model
gemini_model = Gemini(model_id="gemini-2.0-flash")


research_agent = LlmAgent(
    name="ResearchAgent",
    model=gemini_model,
    instruction=(
        "You are a research assistant for Dubai real estate. "
        "Use fetch_property_data to get structured project info. "
        "Use search_web_info to find specific details like pricing, payment plans, "
        "schools nearby, ROI projections, etc. "
        "Return comprehensive property_data and market_trends."
    ),
    tools=[fetch_property_data, search_web_info],
)

# SegmentationAgent - determines buyer category and persona
segmentation_agent = LlmAgent(
    name="SegmentationAgent",
    model=gemini_model,
    instruction=(
        "You are a buyer segmentation expert for Dubai real estate. "
        "Based on property price range and features, determine:\n"
        "1. buyer_category: HNI (>5M AED), HighCaliber (2-5M AED), "
        "MiddleRange (500K-2M AED), or LowRange (<500K AED)\n"
        "2. buyer_persona: SafeFutureForKidsInvestor (families with children), "
        "SafetyFocusedWealthPreserver (high net worth security focused), "
        "SafeHavenRetirementPlanner (retirees), SecondPassportSecuritySeeker (citizenship seekers), "
        "or SafetySeekingWealthMigrant (relocating professionals)\n"
        "Return ONLY a JSON object: {\"buyer_category\": \"...\", \"buyer_persona\": \"...\", \"rationale\": \"...\"}"
    ),
)

# PositioningAgent - creates marketing positioning
positioning_agent = LlmAgent(
    name="PositioningAgent",
    model=gemini_model,
    instruction=(
        "You are a strategic marketing positioning expert. "
        "Create compelling positioning based on property data, buyer category, and persona. "
        "Return ONLY a JSON object with:\n"
        "{\n"
        "  \"core_promise\": \"one sentence value proposition\",\n"
        "  \"key_hooks\": [\"hook1\", \"hook2\", \"hook3\"],\n"
        "  \"proof_points\": [\"data point 1\", \"data point 2\"],\n"
        "  \"emotional_triggers\": [\"emotion1\", \"emotion2\"],\n"
        "  \"cta\": \"clear call to action\"\n"
        "}"
    ),
)

# ContentAgent - creates social media posts
content_agent = LlmAgent(
    name="ContentAgent",
    model=gemini_model,
    instruction=(
        "You are a social media content creator for Dubai real estate. "
        "Create engaging posts for Instagram and LinkedIn based on positioning. "
        "Return ONLY a JSON object:\n"
        "{\n"
        "  \"instagram\": {\"headline\": \"...\", \"body\": \"...\", \"hashtags\": [...], \"cta\": \"...\"},\n"
        "  \"linkedin\": {\"headline\": \"...\", \"body\": \"...\", \"hashtags\": [...], \"cta\": \"...\"}\n"
        "}"
    ),
)

# EvaluationAgent - scores content quality
evaluation_agent = LlmAgent(
    name="EvaluationAgent",
    model=gemini_model,
    instruction=(
        "You are a content quality evaluator. "
        "Score content on: persona_fit (1-5), data_grounding (1-5), "
        "clarity (1-5), cta_effectiveness (1-5). "
        "Return ONLY a JSON object:\n"
        "{\n"
        "  \"scores\": {\"persona_fit\": X, \"data_grounding\": X, \"clarity\": X, \"cta_effectiveness\": X},\n"
        "  \"overall\": X.X,\n"
        "  \"issues\": [...],\n"
        "  \"suggestions\": [...]\n"
        "}"
    ),
)

print("âœ… All agents defined successfully with LlmAgent + Gemini")


import time
from datetime import datetime

class SimpleTracer:
    def __init__(self):
        self.traces = []
    
    def log(self, agent_name, action, details=None):
        self.traces.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "action": action,
            "details": details
        })
    
    def print_summary(self):
        print("\n" + "="*60)
        print("ğŸ”� EXECUTION TRACE")
        print("="*60)
        for i, trace in enumerate(self.traces, 1):
            print(f"{i}. [{trace['timestamp'].split('T')[1][:8]}] {trace['agent']}: {trace['action']}")
            if trace['details']:
                print(f"   â””â”€ {trace['details']}")
        print("="*60 + "\n")

# Initialize the tracer
tracer = SimpleTracer()
print("âœ… Tracer initialized and ready for logging")



from google.adk.runners import Runner
from google.adk.sessions import Session
import json
import time

async def run_complete_campaign_with_sessions(property_query: str, session_id: str = None):
    """
    Full campaign: Research â†’ Segmentation â†’ Content Generation â†’ Evaluation. Sequential campaign workflow with human-in-the-loop approval before publishing
    """
    import time
    from google.genai import types
    
    print(f"\nğŸš€ Starting campaign for: {property_query}\n")
    
    # Generate session ID if not provided
    if not session_id:
        session_id = f"campaign_{int(time.time())}"

    print(f"Starting campaign for: {property_query}")
    print(f"Session ID: {session_id}")

    # AWAIT the async create_session method
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    print(f"âœ… Created session: {session_id}")
    
    # Verify it was saved
    check = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id
    )
    if check is None:
        raise RuntimeError(f"Failed to save session {session_id}")
    print(f"âœ… Verified session exists")
    
    # Create campaign in custom memory
    campaign_id = campaign_mem.create_campaign(property_query)
    tracer.log("Orchestrator", "Campaign created", f"ID: {campaign_id}, Session: {session_id}")
    
    # Step 1: RESEARCH with session tracking
    # === RESEARCH PHASE ===
    user_message = types.Content(
        role="user",
        parts=[types.Part(text=f"Research this property: {property_query}")]
    )
    tracer.log("ResearchAgent", "Starting research", property_query)
    start = time.time()
    print("\nğŸ”� Starting Research Phase...")
   
    research_result = None
    async for event in research_runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=user_message,
    ):
        if hasattr(event, 'content') and event.content:
            research_result = event.content
            print(f"âœ… Research complete: {str(research_result)[:200]}...")
    
    print(f"âœ… Research complete")
    campaign_mem.set(campaign_id, "research", research_result)
    
    tracer.log("ResearchAgent", "Research complete", f"Duration: {time.time()-start:.1f}s")
    campaign_mem.set(campaign_id, "research", str(research_result))
    
    # === PHASE 2: SEGMENTATION ===
    tracer.log("SegmentationAgent", "Analyzing buyer segment")
    start = time.time()
    
    print("\nğŸ�¯ Phase 2: Segmentation")
    segment_msg = types.Content(
        role="user",
        parts=[types.Part(text=f"Analyze target segments for: {research_result}")]
    )
    
    segment_result = None
    async for event in segmentation_runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=segment_msg,
    ):
        if hasattr(event, 'content') and event.content:
            segment_result = event.content
    
    print(f"âœ… Segmentation complete")
    campaign_mem.set(campaign_id, "segmentation", segment_result)
    
    tracer.log("SegmentationAgent", "Segmentation complete", f"Duration: {time.time()-start:.1f}s")
    
    try:
        segment_data = json.loads(str(segment_result))
    except:
        segment_data = {"buyer_category": "MiddleRange", "buyer_persona": "SafeFutureForKidsInvestor"}
    
    
    # === PHASE 3: POSITIONING ===
    tracer.log("PositioningAgent", "Creating positioning strategy")
    start = time.time()
    
    print("\nğŸ�¨ Phase 3: Positioning")
    positioning_msg = types.Content(
        role="user",
        parts=[types.Part(text=f"Create positioning strategy based on research: {research_result} and segments: {segment_result}")]
    )
    
    positioning_result = None
    async for event in positioning_runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=positioning_msg,
    ):
        if hasattr(event, 'content') and event.content:
            positioning_result = event.content
    
    print(f"âœ… Positioning complete")
    campaign_mem.set(campaign_id, "positioning", positioning_result)
    
    tracer.log("PositioningAgent", "Positioning complete", f"Duration: {time.time()-start:.1f}s")
    
    try:
        positioning_data = json.loads(str(positioning_result))
    except:
        positioning_data = {"core_promise": "Secure your family's future", "key_hooks": [], "proof_points": []}
    
    # === PHASE 4: CONTENT GENERATION ===
    tracer.log("ContentAgent", "Generating social content")
    start = time.time()
    
    print("\nâœ�ï¸� Phase 4: Content Generation")
    content_msg = types.Content(
        role="user",
        parts=[types.Part(text=f"Generate social media content based on positioning: {positioning_result}")]
    )
    
    content_result = None
    async for event in content_runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content_msg,
    ):
        if hasattr(event, 'content') and event.content:
            content_result = event.content
    
    print(f"âœ… Content generation complete")
    campaign_mem.set(campaign_id, "content", content_result)
    
    tracer.log("ContentAgent", "Content created", f"Duration: {time.time()-start:.1f}s")
    
    try:
        content_data = json.loads(str(content_result))
    except:
        content_data = {"instagram": {}, "linkedin": {}}

    # === PHASE 5: EVALUATION ===
    print("\nâ­� Phase 5: Evaluation")
    eval_msg = types.Content(
        role="user",
        parts=[types.Part(text=f"Evaluate this content for quality and brand alignment: {content_result}")]
    )
    
    eval_result = None
    async for event in evaluation_runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=eval_msg,
    ):
        if hasattr(event, 'content') and event.content:
            eval_result = event.content
    
    print(f"âœ… Evaluation complete")
    campaign_mem.set(campaign_id, "evaluation", eval_result)
    
    # Step 5: QUEUE FOR APPROVAL
    tracer.log("ApprovalQueue", "Queueing content for approval")
    
    if "instagram" in content_data:
        approval_queue.add_item(campaign_id, "social_post", "instagram", "INSTAGRAM_HANDLE", content_data["instagram"])
    if "linkedin" in content_data:
        approval_queue.add_item(campaign_id, "social_post", "linkedin", "LINKEDIN_HANDLE", content_data["linkedin"])
    
    # Step 6: EVALUATION
    tracer.log("EvaluationAgent", "Evaluating content quality")
    start = time.time()
    
    eval_msg = types.Content(
        role="user",
        parts=[types.Part(text=f"Evaluate this content for quality and brand alignment: {content_result}")]
    )
    
    eval_result = None
    async for event in evaluation_runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=eval_msg,
    ):
        if hasattr(event, 'content') and event.content:
            eval_result = event.content
    
    tracer.log("EvaluationAgent", "Evaluation complete", f"Duration: {time.time()-start:.1f}s")
    
    try:
        eval_data = json.loads(str(eval_result))
    except:
        eval_data = {"scores": {}, "overall": 4.0, "issues": [], "suggestions": []}
    
    campaign_mem.set(campaign_id, "evaluation", eval_data)

    # === HUMAN-IN-THE-LOOP: ADD TO APPROVAL QUEUE ===
    print("\nâ�¸ï¸� Phase 6: Awaiting Human Approval")
    
    # Add Instagram post to approval queue
    instagram_payload = {
        "content": content_result,
        "evaluation": eval_result,
        "positioning": positioning_result,
    }
    
    instagram_id = approval_queue.add_item(
        campaign_id,              # campaign_id
        "social_post",            # item_type
        "instagram",              # target_platform
        "@safefutureforkids",     # platform_handle (your Instagram handle)
        instagram_payload         # payload
    )
    print(f"   Added Instagram post to queue: {instagram_id}")
    
    # Add LinkedIn post to approval queue
    linkedin_payload = {
        "content": content_result,
        "evaluation": eval_result,
        "positioning": positioning_result,
    }
    
    linkedin_id = approval_queue.add_item(
        campaign_id,
        "social_post",
        "linkedin",
        "safe-future-for-kids",   # platform_handle (your LinkedIn page)
        linkedin_payload
    )
    print(f"   Added LinkedIn post to queue: {linkedin_id}")
    
    print(f"\nâœ… Campaign paused for approval. Pending items: {len(approval_queue.list_pending())}")
    print(f"   Use approval_queue.approve_item(item_id) to approve")
    print(f"   Use approval_queue.reject_item(item_id, reason) to reject")

    
    # SAVE SESSION TO MEMORY for long-term recall
    session = session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id
    )
    
    if session:
        memory_service.add_session_to_memory(session)
        tracer.log("MemoryService", "Session saved to long-term memory", session_id)
    
    # Print trace summary
    tracer.print_summary()
    
    # Return results
    return {
        "campaign_id": campaign_id,
        "session_id": session_id,
        "segment": segment_data,
        "positioning": positioning_data,
        "content": content_data,
        "evaluation": eval_data,
        "pending_approvals": len(approval_queue.list_pending(campaign_id))
    }

print("âœ… Session-aware orchestrator ready")

async def publish_approved_content(campaign_id: str):
    """
    Publishes content after human approval
    Called AFTER user approves items in the queue
    """
    print(f"\nğŸ“¤ Publishing approved content for campaign: {campaign_id}")
    
    # Get approved items from queue
    approved_items = [
        item for item in approval_queue.items 
        if item["campaign_id"] == campaign_id and item["status"] == "APPROVED"
    ]
    
    if not approved_items:
        print("â�Œ No approved items found for this campaign")
        return
    
    published = []
    for item in approved_items:
        print(f"   Publishing to {item['channel']}...")
        
        # Simulate publishing (in real system, this would call social media APIs)
        published.append({
            "channel": item["channel"],
            "content_type": item["content_type"],
            "status": "PUBLISHED",
            "timestamp": time.time(),
        })
        
        # Mark as published in queue
        approval_queue.mark_published(item["id"])
    
    print(f"âœ… Published {len(published)} items")
    campaign_mem.set(campaign_id, "published", published)
    
    return published



# ğŸ”§ Initialize All Memory Systems

# Initialize custom application memory
session_mem = SessionMemory()
campaign_mem = CampaignMemory()
cross_mem = CrossCampaignMemory()
approval_queue = ApprovalQueue()

print("âœ… Custom memory systems initialized:")
print(f"   - SessionMemory: {type(session_mem).__name__}")
print(f"   - CampaignMemory: {type(campaign_mem).__name__}")
print(f"   - CrossCampaignMemory: {type(cross_mem).__name__}")
print(f"   - ApprovalQueue: {type(approval_queue).__name__}")


from google.adk.runners import Runner

# Create runners for each agent, all using the SAME session_service
research_runner = Runner(
    agent=research_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

segmentation_runner = Runner(
    agent=segmentation_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

positioning_runner = Runner(
    agent=positioning_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

content_runner = Runner(
    agent=content_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

evaluation_runner = Runner(
    agent=evaluation_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… All runners created successfully")
print(f"All runners share same session_service: {id(session_service)}")



# Diagnostic check
print("=== SESSION SERVICE DIAGNOSTIC ===")
print(f"session_service type: {type(session_service)}")
print(f"session_service id: {id(session_service)}")
print(f"research_runner session_service id: {id(research_runner.session_service)}")  # no underscore
print(f"Same instance? {research_runner.session_service is session_service}")

# List all sessions
print(f"\nAll sessions in session_service:")
try:
    all_sessions = session_service._sessions  # InMemorySessionService stores in ._sessions dict
    print(f"  Total sessions: {len(all_sessions)}")
    for key in all_sessions:
        print(f"    {key}")
except Exception as e:
    print(f"  (Could not access internal storage: {e})")


# Run the complete campaign with session tracking
result = await run_complete_campaign_with_sessions(
    "Altair 52 by Acube Abode Realty - official website https://altair52.com - "
    "located in Dubai South. Get all property details: location coordinates, "
    "price range, amenities, unit types, and launch status to promote this 2BR "
    "family tower for Safe Future for Kids investors."
)


print("\n" + "="*60)
print("ğŸ“Š CAMPAIGN RESULTS SUMMARY")
print("="*60)
# print(f"\nğŸ“‹ Campaign Status: {result['status']}")
print(f"Pending approvals: {result['pending_approvals']}")

print(f"Campaign ID: {result['campaign_id']}")
print(f"Session ID: {result['session_id']}")
print(f"\nğŸ�¯ Buyer Segment:")
print(f"  Category: {result['segment']['buyer_category']}")
print(f"  Persona: {result['segment']['buyer_persona']}")
print(f"\nğŸ“� Positioning:")
print(f"  Promise: {result['positioning'].get('core_promise', 'N/A')}")
print(f"\nâœ�ï¸� Content Created:")
print(f"  Instagram: {'âœ“' if result['content'].get('instagram') else 'âœ—'}")
print(f"  LinkedIn: {'âœ“' if result['content'].get('linkedin') else 'âœ—'}")
print(f"\nâ­� Evaluation Score: {result['evaluation'].get('overall', 'N/A')}/5.0")
print(f"\nğŸ“‹ Pending Approvals: {result['pending_approvals']}")
print("="*60)


# Step 2: Human reviews and approves
pending = approval_queue.list_pending()
print(f"\nğŸ“‹ Pending Approvals ({len(pending)} items):\n")

for item in pending:
    # Use dot notation instead of dict access
    print(f"Item ID: {item.id}")
    print(f"  Platform: {item.target_platform}")
    print(f"  Handle: {item.platform_handle}")
    print(f"  Type: {item.item_type}")
    print(f"  Status: {item.status}")
    print(f"  Content preview: {str(item.content_payload)[:100]}...")
    print()

# To approve an item:
# approval_queue.approve_item(item.id)

# To reject an item:
# approval_queue.reject_item(item.id, "reason here")



import pandas as pd

# Get pending items
pending = approval_queue.list_pending()

# Convert to list of dicts for DataFrame
data = []
for item in pending:
    data.append({
        "Item ID": item.id[:8] + "...",  # Shortened for readability
        "Campaign ID": item.campaign_id[:8] + "...",
        "Type": item.item_type,
        "Platform": item.target_platform,
        "Handle": item.platform_handle,
        "Status": item.status,
        "Content Preview": str(item.content_payload.get('content', ''))[:100] + "..." if isinstance(item.content_payload, dict) else str(item.content_payload)[:100] + "...",
    })

# Create DataFrame and display
df = pd.DataFrame(data)
print("\nğŸ“‹ APPROVAL QUEUE")
print("=" * 120)
display(df)  # Use display() in Jupyter/Kaggle for better formatting

# Or use print if display() doesn't work:
# print(df.to_string(index=False))



def get_approval_data_full():
    """Get approval queue data with full text extraction"""
    pending = approval_queue.list_pending()
    data = []
    
    for item in pending:
        row = {
            "Item ID": item.id[:12],
            "Platform": item.target_platform,
            "Handle": item.platform_handle,
            "Type": item.item_type,
        }
        
        # Extract full text from payload
        if isinstance(item.content_payload, dict):
            for key, value in item.content_payload.items():
                # Extract text from Content objects
                if hasattr(value, 'parts'):
                    text_parts = []
                    for part in value.parts:
                        if hasattr(part, 'text'):
                            text_parts.append(part.text)
                    row[key] = '\n'.join(text_parts) if text_parts else str(value)
                elif hasattr(value, 'text'):
                    row[key] = value.text
                else:
                    row[key] = str(value)
        
        data.append(row)
    
    return pd.DataFrame(data)

# Get full data
df_full = get_approval_data_full()

# Display specific columns
print("\nğŸ“‹ APPROVAL QUEUE - FULL CONTENT\n")
for idx, row in df_full.iterrows():
    print(f"\n{'='*100}")
    print(f"ITEM {idx+1}: {row['Platform']} - {row['Handle']}")
    print(f"{'='*100}")
    for col in df_full.columns:
        if col not in ['Item ID', 'Platform', 'Handle', 'Type']:
            print(f"\n{col.upper()}:")
            print(row[col])
    print()


