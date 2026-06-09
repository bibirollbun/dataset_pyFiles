import os
import json
import asyncio
import time
import logging
import uuid
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, field
from kaggle_secrets import UserSecretsClient

# Basic setup
try:
    _usc = UserSecretsClient()
    _gkey = _usc.get_secret("GOOGLE_API_KEY")
    try:
        os.environ["SEARCH_ENGINE_ID"] = _usc.get_secret("SEARCH_ENGINE_ID") 
    except: pass
    os.environ["GOOGLE_API_KEY"] = _gkey
    print("GOOGLE_API_KEY loaded.")
except Exception as e:
    if "GOOGLE_API_KEY" not in os.environ:
         raise RuntimeError(f"Credentials missing: {e}")

# Imports
import google.generativeai as genai 
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdvancedBlogAgent")

# Config
GEMINI_MODEL = "gemini-2.5-flash" 

def get_model(system_instruction: str):
    """Returns a configured Gemini model instance with system prompting."""
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_instruction,
        generation_config=genai.GenerationConfig(
            temperature=0.4,
            response_mime_type="application/json"
        )
    )

# Share state mgmt
@dataclass
class BlogState:
    """The 'Brain' of the operation. Passed between agents."""
    request_id: str
    topic: str
    target_audience: str
    tone: str
    
    # Artifacts built over time
    research_notes: List[str] = field(default_factory=list)
    outline: Dict = field(default_factory=dict)
    first_draft: str = ""
    editor_critique: str = ""
    final_draft: str = ""
    
    # Workflow control
    status: Literal["planning", "researching", "writing", "reviewing", "publishing", "completed"] = "planning"
    revision_count: int = 0
    max_revisions: int = 2


# Tooling - just a place holder for now
class Tools:
    @staticmethod
    def mock_web_search(query: str) -> List[Dict]:
        """
        Replaces the live API for demo reliability. 
        In production, swap this with your SerpAPI/Google Search code.
        """
        logger.info(f"SEARCHING: {query}")
        # sim search results
        time.sleep(1)
        return [
            {"title": f"Guide to {query}", "snippet": f"This is a comprehensive guide about {query}. It covers key automation strategies.", "url": "example.com/1"},
            {"title": "Recent Trends in AI", "snippet": "Agents are moving from linear pipelines to cyclic graphs. Efficiency is up 40%.", "url": "example.com/2"},
            {"title": "Common Pitfalls", "snippet": "Avoid infinite loops in agent logic. Always have a max_retry count.", "url": "example.com/3"}
        ]

# Agents
class BaseAgent:
    def __init__(self, name):
        self.name = name

    async def generate_json(self, model, prompt):
        try:
            response = await model.generate_content_async(prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text)
        except Exception as e:
            logger.error(f"{self.name} failed to generate JSON: {e}")
            return {}

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Planner")
        self.model = get_model(
            "You are a Content Strategist. Analyze the topic and create a plan. "
            "Output JSON with keys: 'angle', 'research_questions' (list of strings)."
        )

    async def execute(self, state: BlogState) -> BlogState:
        logger.info("PLANNER: Developing strategy...")
        prompt = f"Topic: {state.topic}\nAudience: {state.target_audience}\nTone: {state.tone}"
        result = await self.generate_json(self.model, prompt)
        state.research_notes.append(f"Strategic Angle: {result.get('angle', 'General Overview')}")
        state.current_research_questions = result.get('research_questions', [f"facts about {state.topic}"])
        state.status = "researching"
        return state

class ResearcherAgent(BaseAgent):
    def __init__(self):
        super().__init__("Researcher")
        self.model = get_model(
            "You are a Senior Researcher. Summarize search results into key insights. "
            "Output JSON with key: 'summary'."
        )

    async def execute(self, state: BlogState) -> BlogState:
        logger.info("ğŸ•µï¸� RESEARCHER: Gathering intelligence...")
        raw_data = []
        questions = getattr(state, 'current_research_questions', [state.topic])
        
        for q in questions[:3]: 
            results = Tools.mock_web_search(q)
            raw_data.extend([r['snippet'] for r in results])
        
        context = "\n".join(raw_data)
        prompt = f"Synthesize these search snippets into coherent notes for a writer.\nContext: {context[:5000]}"
        result = await self.generate_json(self.model, prompt)
        state.research_notes.append(f"Findings: {result.get('summary', 'No summary generated')}")
        state.status = "writing"
        return state

class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__("Writer")
        self.model = get_model(
            "You are a Professional Blogger. Write a structured blog post. "
            "Output JSON with keys: 'outline' (dict), 'draft_text' (markdown string)."
        )

    async def execute(self, state: BlogState) -> BlogState:
        logger.info(f"WRITER: Drafting (Revision {state.revision_count})...")
        research_context = "\n".join(state.research_notes)
        critique_context = f"Previous Feedback: {state.editor_critique}" if state.editor_critique else ""
        prompt = f"Topic: {state.topic}\nAudience: {state.target_audience}\nResearch: {research_context}\n{critique_context}\nTask: Create a blog post."
        
        result = await self.generate_json(self.model, prompt)
        state.outline = result.get('outline', {})
        state.first_draft = result.get('draft_text', "# Draft\nContent generation failed.")
        state.status = "reviewing"
        return state

class EditorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Editor")
        self.model = get_model(
            "You are a Chief Editor. Critique the blog post for flow, tone, and depth. "
            "Output JSON with keys: 'score' (1-10), 'critique' (string), 'decision' ('approve' or 'revise')."
        )

    async def execute(self, state: BlogState) -> BlogState:
        logger.info("EDITOR: Reviewing draft...")
        prompt = f"Draft: {state.first_draft[:10000]}\nTarget Tone: {state.tone}"
        result = await self.generate_json(self.model, prompt)
        
        score = result.get('score', 5)
        decision = result.get('decision', 'revise')
        critique = result.get('critique', 'No critique provided')
        
        logger.info(f"--- Draft Score: {score}/10 ---")
        logger.info(f"--- Decision: {decision} ---")
        
        if decision == 'approve' or state.revision_count >= state.max_revisions:
            state.final_draft = state.first_draft
            state.status = "publishing"
        else:
            state.editor_critique = critique
            state.revision_count += 1
            state.status = "writing" 
        return state

class PublisherAgent(BaseAgent):
    """Finalizer: Formats and outputs the result."""
    def __init__(self):
        # This was missing in the previous version
        super().__init__("Publisher")

    def execute(self, state: BlogState) -> BlogState:
        logger.info("PUBLISHER: Formatting final output.")
        state.status = "completed"
        return state


# Orchestrator
class BlogOrchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        self.editor = EditorAgent()
        self.publisher = PublisherAgent()

    async def run_workflow(self, initial_request: Dict):
        state = BlogState(
            request_id=str(uuid.uuid4()),
            topic=initial_request['topic'],
            target_audience=initial_request.get('audience', 'General'),
            tone=initial_request.get('tone', 'Neutral')
        )
        
        logger.info(f"Starting Workflow for: {state.topic}")
        
        while state.status != "completed":
            if state.status == "planning":
                state = await self.planner.execute(state)
            
            elif state.status == "researching":
                state = await self.researcher.execute(state)
                
            elif state.status == "writing":
                state = await self.writer.execute(state)
                
            elif state.status == "reviewing":
                state = await self.editor.execute(state)
                
            elif state.status == "publishing":
                state = self.publisher.execute(state)
                
            await asyncio.sleep(1) 
            
        return state

# Main
from IPython.display import display, Markdown

async def run_demo():
    # Request
    request = {
        "topic": "The Future of AI Agents in Enterprise Automation",
        "audience": "CTOs and Tech Leaders",
        "tone": "Professional, Insightful, slightly provocative"
    }

    print(f"Starting Workflow for: '{request['topic']}'...\n")

    # Run
    orchestrator = BlogOrchestrator()
    final_state = await orchestrator.run_workflow(request)

    # Output
    print("\n" + "="*50)
    print("WORKFLOW COMPLETED")
    print("="*50)
    print(f"Revisions made: {final_state.revision_count}")
    print("-" * 20)
    
    display(Markdown(f"# {final_state.topic}"))
    display(Markdown(final_state.final_draft))

# Execute
await run_demo()

