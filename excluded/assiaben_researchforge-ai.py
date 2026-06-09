!pip install -q faiss-cpu
!pip install nest_asyncio --quiet


# âš™ï¸� Just to Suppress unnecessary warnings for a cleaner notebook
# ğŸš« Disable GPU backend to prevent cuDNN/cuBLAS warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      #  CPU mode
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # Hide TF backend logs
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="  # Disable XLA GPU checks


# âš™ï¸� Clean Notebook Start â€” Suppress Unnecessary Warnings
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("google.protobuf").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)



# âš›ï¸� ResearchForge AI - Loading Dependencies
print("âš›ï¸� ResearchForge AI - ğŸ�—ï¸� Initializing...")
print("=" * 48)

#  Protobuf Compatibility (Safe Downgrade)
import sys
import subprocess

try:
    import google.protobuf
    version = google.protobuf.__version__
    major = int(version.split(".")[0])

    if major >= 4:
        print(f"ğŸ“¦ Adjusting protobuf version ({version}) for compatibility...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "protobuf==3.20.3", "--quiet", "--no-warn-conflicts"
        ])
        import importlib
        importlib.reload(google.protobuf)
        print("ğŸ”½ Protobuf downgraded to 3.20.3\n")
    else:
        print(f"ğŸŸ¢ Protobuf {version} - OK")

except Exception as e:
    print(f"âš ï¸� Protobuf check skipped: {e}")

#  Core Libraries
import asyncio
import json
import uuid
import time
import requests
from datetime import datetime
from collections import defaultdict
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from typing import Union

import pandas as pd
import numpy as np

#  Embeddings & Similarity
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

#  Google ADK Multi-Agent Framework
try:
    from google.genai import types
    from google.adk.agents import Agent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import FunctionTool, AgentTool
    print("\nğŸŒŸ Google ADK imported successfully")
except Exception as e:
    print(f"ğŸ”´ Google ADK import failed: {e}")

#  Security + Validation
from pydantic import BaseModel, Field
from kaggle_secrets import UserSecretsClient

#  UI Components
import ipywidgets as widgets
from IPython.display import display, clear_output

#  Async Fix
import nest_asyncio
nest_asyncio.apply()

print("\nğŸ†— All dependencies loaded successfully â€” Ready.")
print("=" * 48)



last_request_time = 0
REQUEST_DELAY = 3.0  # 3 seconds between API calls

def rate_limit():
    """Enforce minimum delay between API calls"""
    global last_request_time
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < REQUEST_DELAY:
        sleep_time = REQUEST_DELAY - elapsed
        time.sleep(sleep_time)
        print(f"â�³ Rate limiting: waited {sleep_time:.1f}s")
    last_request_time = time.time()


# Configure Gemini API for agent intelligence

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Gemini API configured successfully âœ”ï¸�")
except Exception as e:
    print(f"âœ— Failed to configure API: {e}")
    print("  â†’ Add GOOGLE_API_KEY to Kaggle Secrets")


# Configure retry strategy for API resilience
# Handles rate limits and temporary failures gracefully

retry_config = types.HttpRetryOptions(
    attempts=5,           # Retry up to 5 times
    exp_base=7,          # Exponential backoff base
    initial_delay=1,     # Start with 1 second delay
    http_status_codes=[429, 500, 503, 504]  # Retry on these errors
)
print("API retry configuration initialized âœ”ï¸�")


# 1. CONFIGURE LOGGING


# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create loggers for each agent
logger_main = logging.getLogger('ResearchForge')
logger_data = logging.getLogger('DataScout')
logger_match = logging.getLogger('MatchEngine')
logger_proposal = logging.getLogger('ProposalGen')
logger_outreach = logging.getLogger('OutreachAgent')

# Suppress external library noise
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger_main.info("ğŸ”� Observability system initialized")

# ========================================================
# 2. METRICS TRACKER
class AgentMetrics:
    """Track performance metrics for all agents"""
    
    def __init__(self):
        self.metrics = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'total_time': 0.0,
            'agent_calls': defaultdict(int),
            'tool_calls': defaultdict(int),
            'model_usage': defaultdict(int)
        }
        self.start_time = time.time()
    
    def record_request(self, agent_name: str, success: bool, duration: float):
        """Record a request"""
        self.metrics['requests'] += 1
        if success:
            self.metrics['successes'] += 1
        else:
            self.metrics['failures'] += 1
        self.metrics['total_time'] += duration
        self.metrics['agent_calls'][agent_name] += 1
        
        logger_main.info(
            f"ğŸ“Š {agent_name} | Success: {success} | Duration: {duration:.2f}s"
        )
    
    def record_tool(self, tool_name: str):
        """Record tool usage"""
        self.metrics['tool_calls'][tool_name] += 1
        logger_main.info(f"ğŸ”§ Tool called: {tool_name}")
    
    def record_model(self, model_name: str):
        """Record model usage"""
        self.metrics['model_usage'][model_name] += 1
    
    def get_report(self) -> dict:
        """Generate metrics report"""
        uptime = time.time() - self.start_time
        success_rate = (
            (self.metrics['successes'] / self.metrics['requests'] * 100)
            if self.metrics['requests'] > 0 else 0
        )
        avg_time = (
            self.metrics['total_time'] / self.metrics['requests']
            if self.metrics['requests'] > 0 else 0
        )
        
        return {
            'uptime_seconds': round(uptime, 2),
            'total_requests': self.metrics['requests'],
            'success_rate': f"{success_rate:.1f}%",
            'avg_response_time': f"{avg_time:.2f}s",
            'agent_calls': dict(self.metrics['agent_calls']),
            'tool_calls': dict(self.metrics['tool_calls']),
            'model_usage': dict(self.metrics['model_usage'])
        }
    
    def print_report(self):
        """Print formatted metrics report"""
        report = self.get_report()
        
        print("\n" + "="*45)
        print("ğŸ“Š RESEARCHFORGE AI - OBSERVABILITY METRICS")
        print("="*45)
        print(f"â�±ï¸�  System Uptime: {report['uptime_seconds']}s")
        print(f"ğŸ“ˆ Total Requests: {report['total_requests']}")
        print(f"ğŸŒŸ Success Rate: {report['success_rate']}")
        print(f"âš¡ Avg Response Time: {report['avg_response_time']}")
        
        if report['agent_calls']:
            print(f"\nğŸ¤– Agent Calls:")
            for agent, count in report['agent_calls'].items():
                print(f"   â€¢ {agent}: {count}")
        
        if report['tool_calls']:
            print(f"\nğŸ”§ Tool Usage:")
            for tool, count in report['tool_calls'].items():
                print(f"   â€¢ {tool}: {count}")
        
        if report['model_usage']:
            print(f"\nğŸ§  Model Usage:")
            for model, count in report['model_usage'].items():
                print(f"   â€¢ {model}: {count}")
        
        print("="*45 + "\n")

# Initialize global metrics tracker
metrics = AgentMetrics()

logger_main.info("âœ”ï¸� Metrics system ready")
print("ğŸ“Š Observability: Logging & Metrics âœ”ï¸�")


# ResearchInterest: Represents a specific research topic with metadata

# Used to capture researcher expertise and depth in various fields
class ResearchInterest(BaseModel):
    topic: str                                    # Research area name
    expertise_level: float = Field(ge=0, le=1)  # Proficiency (0=novice, 1=expert)
    years_experience: int                         # Years working in this area
    publications_count: int                       # Number of publications in this topic


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ResearcherProfile: Complete researcher identity and capabilities
# Central entity for matching and collaboration recommendations

class ResearcherProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique identifier
    name: str                                     # Full name
    institution: str                              # Affiliated university/org
    department: str                               # Academic department
    position: str                                 # Academic rank (Prof, PostDoc, etc)
    research_interests: List[ResearchInterest]    # Multiple research areas
    skills: List[str]                             # Technical & methodological skills
    publications: List[str]                       # Publication titles/DOIs
    citation_count: int                           # Total citations received
    h_index: int                                  # Hirsch index (research impact)
    email: Optional[str] = None                   # Contact email
    collaboration_history: List[str] = []         # Past collaboration records
    geolocation: Optional[str] = None             # Country/region
    funding_sources: List[str] = []               # Grant sources
    availability: str = "Open to collaboration"   # Current status
    embedding: Optional[List[float]] = None       # 768-dim semantic vector
    

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ResearchProject: Research opportunity/need requiring collaboration
# Represents active projects seeking team members
class ResearchProject(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str                                    # Project title
    abstract: str                                 # Project description
    research_areas: List[str]                     # Relevant research domains
    required_skills: List[str]                    # Must-have technical skills
    desired_expertise: List[str]                  # Preferred additional expertise
    timeline_months: int                          # Project duration
    budget_range: str                             # Funding available
    funding_status: str                           # "Funded" | "Seeking" | "Pending"
    collaboration_type: str                       # "Academic" | "Industry" | "International"
    difficulty_level: str = "Intermediate"        # Entry bar for participants
    embedding: Optional[List[float]] = None       # 768-dim semantic vector

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MatchResult: Compatibility analysis between researcher and project
# Contains multi-dimensional scoring and explanation
class MatchResult(BaseModel):
    researcher_id: str
    project_id: str
    overall_score: float = Field(ge=0, le=100)   # Composite compatibility score
    skill_match: float                            # Technical skill alignment
    interest_match: float                         # Research interest overlap
    complementary_score: float                    # Novel expertise contribution
    geographic_compatibility: float               # Location synergy
    career_synergy: float                         # Career stage fit
    explanation: str                              # Human-readable justification
    confidence_interval: str                      # Statistical confidence

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CollaborationProposal: Auto-generated research proposal document
# Professional output ready for grant submission
class CollaborationProposal(BaseModel):
    title: str                                    # Proposal title
    abstract: str                                 # Executive summary
    research_question: str                        # Core question being addressed
    methodology: str                              # Research approach
    expected_outcomes: str                        # Anticipated results
    timeline: str                                 # Project schedule
    budget_breakdown: str                         # Financial allocation
    collaboration_plan: str                       # Team coordination strategy
    evaluation_metrics: str                       # Success measurement criteria


# PROJECT SPECIFICATION FOR ML MATCHING
# ------------------------------------
# Structured project data for compatibility analysis
# Contains core fields required for similarity scoring

class ProjectRequirement(BaseModel):
    """
    Project specification for ML-driven compatibility analysis
    Contains essential fields for multi-dimensional similarity matching
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique identifier
    title: str = Field(default="Research Project")              # Project name
    description: str = Field(description="Project summary/description")  # Core content
    required_skills: List[str] = Field(default_factory=list)    # Technical skills needed
    research_areas: List[str] = Field(default_factory=list)     # Research domains
    duration_months: int = Field(default=12)                    # Project timeline
    location: str = Field(default="Academic Collaboration")     # Geographic/context info
    
    # REQUIRED FOR AdvancedMatchingEngine.calculate_advanced_match():
    collaboration_type: str = Field(default="remote")           # remote/hybrid/in-person
    funding_available: bool = Field(default=True)               # Funding status
    start_date: str = Field(default="flexible")                 # Start timeline
    institution: str = Field(default="International")           # Institution name
    
    # Optional embedding for semantic search
    embedding: Optional[List[float]] = None                     # Vector representation

print("ProjectRequirement class added for ML matching âœ”ï¸�")
print("Data models initialized (5 entities defined) âœ”ï¸�")


print("ğŸ§  Initializing ML Matching Engine...")
print("=" * 38)

# Initialize Sentence Transformer Model
# This converts text into 768-dimensional vectors for semantic similarity
print("âš¡ Loading sentence transformer model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("âœ“ Embedding model loaded (384-dim vectors)")



# AdvancedMatchingEngine: Multi-dimensional compatibility scoring system
class AdvancedMatchingEngine:
    """
    Intelligent matching system using ML and multi-factor analysis
    
    Evaluates researcher-project compatibility across 6 dimensions:
    1. Skill alignment (technical capabilities match)
    2. Research interest overlap (topic similarity)
    3. Complementary expertise (unique value addition)
    4. Geographic compatibility (location synergy)
    5. Career synergy (experience level fit)
    6. Semantic similarity (deep contextual understanding)
    """
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.researcher_index = None  # FAISS index for researchers
        self.project_index = None     # FAISS index for projects
        self.researchers = []          # Stored researcher profiles
        self.projects = []             # Stored project profiles
        
    def build_similarity_index(self, researchers: List[ResearcherProfile], 
                               projects: List[ResearchProject]):
        """
        Build FAISS indices for fast similarity search
        
        Args:
            researchers: List of researcher profiles to index
            projects: List of research projects to index
        """
        print("ğŸ”¨ Building FAISS similarity indices...")
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Generate embeddings for researchers
        # Combine name, skills, and research interests into single text
        researcher_texts = []
        for r in researchers:
            # Create rich text representation for embedding
            interests = " ".join([ri.topic for ri in r.research_interests])
            skills = " ".join(r.skills)
            text = f"{r.name} {r.institution} {interests} {skills}"
            researcher_texts.append(text)
        
        # Convert to embeddings (384-dim vectors)
        researcher_embeddings = self.embedding_model.encode(researcher_texts)
        
        # Store embeddings in researcher objects
        for i, r in enumerate(researchers):
            r.embedding = researcher_embeddings[i].tolist()
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Build FAISS index for researchers (L2 distance)
        # FAISS enables sub-linear time similarity search
        dimension = researcher_embeddings.shape[1]
        self.researcher_index = faiss.IndexFlatL2(dimension)
        self.researcher_index.add(researcher_embeddings)
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Generate embeddings for projects
        project_texts = []
        for p in projects:
            areas = " ".join(p.research_areas)
            skills = " ".join(p.required_skills)
            text = f"{p.title} {p.abstract} {areas} {skills}"
            project_texts.append(text)
        
        project_embeddings = self.embedding_model.encode(project_texts)
        
        for i, p in enumerate(projects):
            p.embedding = project_embeddings[i].tolist()
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Build FAISS index for projects
        self.project_index = faiss.IndexFlatL2(dimension)
        self.project_index.add(project_embeddings)
        
        # Store references
        self.researchers = researchers
        self.projects = projects
        
        print(f"âœ“ Indexed {len(researchers)} researchers and {len(projects)} projects")
        
    def calculate_advanced_match(self, researcher: ResearcherProfile, 
                             project: Union[ResearchProject, 
                                 ProjectRequirement]) -> MatchResult:
        """
        Calculate comprehensive compatibility score using 6 factors
    
        Args:
            researcher: Researcher profile to evaluate
            project: Project/requirement to match against (accepts both types)
        
        Returns:
            MatchResult with scores and explanation
        
        """
        # Handle both ResearchProject and ProjectRequirement
        if isinstance(project, ProjectRequirement):
            # Convert ProjectRequirement fields to ResearchProject format
            project_title = project.title
            project_required_skills = project.required_skills
            project_research_areas = project.research_areas
            project_collaboration_type = project.collaboration_type
            project_difficulty_level = "Intermediate"  # Default for ProjectRequirement
        else:
            # Use ResearchProject fields directly
            project_title = project.title
            project_required_skills = project.required_skills
            project_research_areas = project.research_areas
            project_collaboration_type = project.collaboration_type
            project_difficulty_level = project.difficulty_level
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # 1. SKILL MATCH
        researcher_skills = set([s.lower() for s in researcher.skills])
        project_skills = set([s.lower() for s in project_required_skills])
        
        if len(researcher_skills.union(project_skills)) > 0:
            skill_match = len(researcher_skills.intersection(project_skills)) / \
                         len(researcher_skills.union(project_skills)) * 100
        else:
            skill_match = 0.0
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # 2. INTEREST MATCH
        if not researcher.research_interests or not project_research_areas:
            interest_match = 0.0
        else:
            # Convert to lowercase
            researcher_topics = [ri.topic.lower() for ri in researcher.research_interests]
            project_topics = [pa.lower() for pa in project_research_areas]
            
            # Check for partial matches
            matches = 0
            total = len(researcher_topics)
            
            for r_topic in researcher_topics:
                for p_topic in project_topics:
                    # Partial match: either contains the other
                    if p_topic in r_topic or r_topic in p_topic:
                        matches += 1
                        break
                    
                    # Word-level match: share common words
                    r_words = set(r_topic.split())
                    p_words = set(p_topic.split())
                    common_words = r_words.intersection(p_words)
                    # Exclude common filler words
                    meaningful_common = common_words - {'and', 'the', 'of', 'in', 'for', 'with'}
                    if len(meaningful_common) > 0:
                        matches += 1
                        break
            interest_match = (matches / total * 100) if total > 0 else 0.0
    
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # 3. COMPLEMENTARY SCORE
        unique_skills = researcher_skills - project_skills
        complementary_score = min(len(unique_skills) / max(len(researcher_skills), 1) * 100, 100)
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # 4. GEOGRAPHIC COMPATIBILITY
        if project_collaboration_type == "International" or project_collaboration_type == "remote":
            geographic_compatibility = 85.0
        elif researcher.geolocation:
            geographic_compatibility = 70.0
        else:
            geographic_compatibility = 50.0
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # 5. CAREER SYNERGY
        avg_experience = np.mean([ri.years_experience for ri in researcher.research_interests])
        
        if project_difficulty_level == "Advanced" and avg_experience >= 7:
            career_synergy = 90.0
        elif project_difficulty_level == "Intermediate" and 3 <= avg_experience < 10:
            career_synergy = 85.0
        elif project_difficulty_level == "Beginner" and avg_experience < 5:
            career_synergy = 80.0
        else:
            career_synergy = 60.0
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # 6. SEMANTIC SIMILARITY
        if researcher.embedding and project.embedding:
            r_emb = np.array(researcher.embedding).reshape(1, -1)
            p_emb = np.array(project.embedding).reshape(1, -1)
            semantic_sim = cosine_similarity(r_emb, p_emb)[0][0]
            semantic_score = max(0, semantic_sim * 100)
        else:
            semantic_score = 50.0
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # OVERALL SCORE
        overall_score = (
            skill_match * 0.25 +
            interest_match * 0.25 +
            complementary_score * 0.15 +
            geographic_compatibility * 0.10 +
            career_synergy * 0.10 +
            semantic_score * 0.15
        )
        
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Generate explanation
        explanation = self._generate_explanation(
            overall_score, skill_match, interest_match, 
            complementary_score, researcher, project_title
        )
        
        confidence = "High" if researcher.citation_count > 100 else "Medium"
        confidence_interval = f"{confidence} (Â±{5 if confidence == 'High' else 10} points)"
        
        return MatchResult(
            researcher_id=researcher.id,
            project_id=project.id,
            overall_score=round(overall_score, 1),
            skill_match=round(skill_match, 1),
            interest_match=round(interest_match, 1),
            complementary_score=round(complementary_score, 1),
            geographic_compatibility=round(geographic_compatibility, 1),
            career_synergy=round(career_synergy, 1),
            explanation=explanation,
            confidence_interval=confidence_interval
        )

    def _generate_explanation(self, overall_score: float, skill_match: float,
                            interest_match: float, complementary_score: float,
                            researcher: ResearcherProfile, 
                            project_title: str) -> str:
        """Generate human-readable match explanation"""
        
        if overall_score >= 80:
            quality = "excellent"
            emoji = "â­�"
        elif overall_score >= 65:
            quality = "strong"
            emoji = "âœ“"
        elif overall_score >= 50:
            quality = "moderate"
            emoji = "~"
        else:
            quality = "weak"
            emoji = "â—‹"
        
        return f"{emoji} {quality.upper()} match ({overall_score:.0f}/100): " \
               f"Skills align at {skill_match:.0f}%, research interests overlap {interest_match:.0f}%, " \
               f"brings {complementary_score:.0f}% complementary expertise. " \
               f"{researcher.name} from {researcher.institution} would add significant value to {project_title}."
    
    


# Initialize matching engine
matching_engine = AdvancedMatchingEngine(embedding_model)
print("âœ“ Matching engine initialized")


print("\nğŸ“Š Generating research ecosystem...")

def generate_researchers(count: int = 50) -> List[ResearcherProfile]:
    """
    Generate researcher profiles for demonstration
    
    Args:
        count: Number of researchers to generate
        
    Returns:
        List of ResearcherProfile objects with data
    """
    # Institutional diversity - top research universities globally
    institutions = [
        "Stanford University", "MIT", "Harvard University", "UC Berkeley",
        "Carnegie Mellon", "University of Washington", "University of Toronto",
        "ETH Zurich", "University of Cambridge", "Oxford University",
        "National University of Singapore", "Tsinghua University"
    ]
    
    # Research domains with specific sub-areas
    research_areas = {
        "AI in Healthcare": ["Medical Imaging", "Drug Discovery", "Clinical NLP", "Precision Medicine"],
        "Natural Language Processing": ["Transformers", "Multilingual NLP", "Text Generation", "Question Answering"],
        "Computer Vision": ["Object Detection", "Image Segmentation", "3D Vision", "Video Understanding"],
        "Robotics": ["Manipulation", "Navigation", "Human/Robot Interaction", "Autonomous Systems"],
        "Machine Learning Theory": ["Optimization", "Deep Learning Theory", "Causality", "Fairness"]
    }
    
    # Technical skill sets relevant to AI research
    skills_pool = [
        "Python", "PyTorch", "TensorFlow", "Scikit-learn", "JAX",
        "AWS", "Docker", "Kubernetes", "Statistical Analysis",
        "Experimental Design", "Data Visualization", "Research Methodology"
    ]
    
    researchers = []
    
    for i in range(count):

        # Select random main research area and specializations
        main_area = np.random.choice(list(research_areas.keys()))
        sub_areas = np.random.choice(research_areas[main_area], 
                                     size=min(2, len(research_areas[main_area])), 
                                     replace=False)
        
       
        # Generate research interests with expertise levels
        interests = [
            ResearchInterest(
                topic=main_area,
                expertise_level=np.random.uniform(0.75, 0.95),
                years_experience=np.random.randint(5, 15),
                publications_count=np.random.randint(10, 50)
            )
        ]
        
        for sub in sub_areas:
            interests.append(
                ResearchInterest(
                    topic=sub,
                    expertise_level=np.random.uniform(0.6, 0.85),
                    years_experience=np.random.randint(3, 10),
                    publications_count=np.random.randint(5, 30)
                )
            )
        
        # Create complete researcher profile
        researcher = ResearcherProfile(
            name=f"Dr. {chr(65 + i % 26)}. Researcher{i}",
            institution=np.random.choice(institutions),
            department=f"{main_area} Department",
            position=np.random.choice(["Assistant Professor", "Associate Professor", 
                                       "Full Professor", "Postdoctoral Researcher"]),
            research_interests=interests,
            skills=list(np.random.choice(skills_pool, 
                                         size=np.random.randint(4, 7), 
                                         replace=False)),
            publications=[f"Paper on {interests[0].topic}" for _ in range(3)],
            citation_count=np.random.randint(100, 2000),
            h_index=np.random.randint(10, 45),
            email=f"researcher{i}@university.edu",
            collaboration_history=[f"Collaboration with {np.random.choice(institutions)}"],
            geolocation=np.random.choice(["North America", "Europe", "Asia"]),
            funding_sources=["NIH", "NSF", "Industry Partner"][:np.random.randint(1, 3)]
        )
        
        researchers.append(researcher)
    
    return researchers


def generate_projects(count: int = 30) -> List[ResearchProject]:
    """
    Generate research project opportunities
    
    Args:
        count: Number of projects to generate
        
    Returns:
        List of ResearchProject objects
    """

    # Project templates for different research domains
    project_templates = [
        {
            "title_prefix": "AI-Powered Early Detection of",
            "areas": ["Medical Imaging", "Healthcare AI", "Diagnostic Tools"],
            "skills": ["Deep Learning", "Medical Data Analysis", "Python", "PyTorch"]
        },
        {
            "title_prefix": "Advanced Natural Language Processing for",
            "areas": ["NLP", "Text Mining", "Information Extraction"],
            "skills": ["Transformers", "Python", "TensorFlow", "Linguistic Analysis"]
        },
        {
            "title_prefix": "Robotic Systems for",
            "areas": ["Robotics", "Automation", "Human-Robot Interaction"],
            "skills": ["ROS", "Python", "Computer Vision", "Control Systems"]
        }
    ]
    
    domains = ["Healthcare", "Education", "Climate Science", "Manufacturing", "Transportation"]
    
    projects = []
    
    for i in range(count):
        template = np.random.choice(project_templates)
        domain = np.random.choice(domains)
        
        project = ResearchProject(
            title=f"{template['title_prefix']} {domain}",
            abstract=f"This project aims to develop innovative AI solutions for {domain.lower()}. "
                     f"We will address key challenges using cutting-edge techniques and create "
                     f"impactful outcomes that advance the field.",
            research_areas=list(template["areas"]),
            required_skills=list(template["skills"]),
            desired_expertise=[domain, "Research Methodology"],
            timeline_months=np.random.randint(12, 36),
            budget_range=np.random.choice(["$200K-$500K", "$500K-$1M", "$1M+"]),
            funding_status=np.random.choice(["Funded", "Grant Pending", "Seeking Funding"]),
            collaboration_type=np.random.choice(["Academic", "Industry", "International"]),
            difficulty_level=np.random.choice(["Intermediate", "Advanced"])
        )
        
        projects.append(project)
    
    return projects



# Generate the research ecosystem
RESEARCHERS = generate_researchers(50)
PROJECTS = generate_projects(30)

print(f"âœ“ Generated {len(RESEARCHERS)} researcher profiles")
print(f"âœ“ Generated {len(PROJECTS)} research projects")


# Build FAISS indices for fast similarity search
matching_engine.build_similarity_index(RESEARCHERS, PROJECTS)

print("\n âœ“ ML Matching ğŸ”— Engine Ready ğŸŸ¢!")
print("=" * 34)


# AGENT 1: ğŸ•µï¸�â€�â™€ï¸� DataScout - Academic Data Discovery
print("ğŸ¤– Initializing 8 Specialized Agents...")
print("=" * 39)

def advanced_arxiv_search(
    query: str, 
    category: str = "cs.CV,cs.LG,q-bio.QM,cs.AI",
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Search arXiv for REAL STEM research papers
    """
    start_time = time.time()
    logger_data.info(f"ğŸ”� arXiv: '{query}'")
    
    try:
        metrics.record_tool('arxiv_search')
        base_url = "http://export.arxiv.org/api/query"
        
        # ğŸ�¯ SMART CATEGORY SELECTION WITH BETTER MEDICAL DETECTION
        categories = [cat.strip() for cat in category.split(",")]
        
        # AUTO-OPTIMIZE - MORE COMPREHENSIVE
        # query_lower = query.lower()
        query_lower = query.lower()

        # Define category mappings (ordered by specificity)
        category_mapping = {
            # Medical/Health
            'medical': {
                'keywords': ['medical', 'clinical', 'health', 'healthcare', 'cancer', 'tumor', 
                            'alzheimer', 'dementia', 'mri', 'pet', 'ct', 'scan', 'imaging', 
                            'radiology', 'diagnosis', 'treatment', 'patient', 'disease'],
                'categories': ['cs.CV', 'q-bio.QM', 'eess.IV', 'physics.med-ph']
            },
            
            # Physics/Quantum
            'physics': {
                'keywords': ['quantum', 'quant', 'qubit', 'entanglement', 'superposition',
                            'physics', 'particle', 'relativity', 'cosmology', 'gravitational'],
                'categories': ['quant-ph', 'cond-mat', 'physics', 'hep-th', 'gr-qc']
            },
            
            # Astronomy/Space
            'astronomy': {
                'keywords': ['astronomy', 'astro', 'astrophysics', 'cosmology', 'planet', 
                            'mars', 'space', 'galaxy', 'star', 'telescope', 'nasa', 'exoplanet'],
                'categories': ['astro-ph', 'physics']
            },
            
            # Biology/Life Sciences
            'biology': {
                'keywords': ['biology', 'genomics', 'protein', 'dna', 'rna', 'gene', 
                            'molecular', 'cell', 'organism', 'evolution'],
                'categories': ['q-bio', 'cs.CE', 'physics.bio-ph']
            },
            
            # Robotics/Control
            'robotics': {
                'keywords': ['robot', 'robotic', 'autonomous', 'control', 'manipulation', 
                            'drone', 'automation'],
                'categories': ['cs.RO', 'cs.SY', 'cs.AI']
            },
            
            # NLP/Language
            'nlp': {
                'keywords': ['nlp', 'language', 'text', 'translation', 'sentiment', 
                            'chatbot', 'dialogue', 'speech'],
                'categories': ['cs.CL', 'cs.AI', 'cs.LG']
            },
            
            # Computer Vision
            'vision': {
                'keywords': ['vision', 'image', 'video', 'visual', 'detection', 
                            'segmentation', 'recognition', 'face'],
                'categories': ['cs.CV', 'cs.AI', 'eess.IV']
            },
            
            # Economics/Finance
            'economics': {
                'keywords': ['economics', 'finance', 'economy', 'financial', 'market', 
                            'trading', 'investment', 'monetary'],
                'categories': ['econ', 'q-fin', 'stat.AP']
            },
            
            # Mathematics
            'math': {
                'keywords': ['mathematics', 'math', 'theorem', 'proof', 'topology', 
                            'algebra', 'geometry', 'calculus'],
                'categories': ['math', 'math.CO', 'math.NT']
            },
            
            # Art/Design/Creative
            'art': {
                'keywords': ['art', 'music', 'design', 'creative', 'aesthetic', 
                            'painting', 'sculpture', 'artist', 'audio', 'sound'],
                'categories': ['cs.SD', 'cs.GR', 'cs.MM', 'cs.HC']
            },
            
            # Climate/Environment
            'climate': {
                'keywords': ['climate', 'environment', 'weather', 'atmospheric', 'ocean', 
                            'ecology', 'sustainability'],
                'categories': ['physics.ao-ph', 'physics.geo-ph', 'stat.AP']
            },
            
            # Chemistry
            'chemistry': {
                'keywords': ['chemistry', 'chemical', 'molecule', 'compound', 'reaction', 
                            'synthesis', 'catalyst'],
                'categories': ['physics.chem-ph', 'cond-mat', 'q-bio.BM']
            }
        }
        
        # Smart category detection with scoring
        detected_domain = None
        max_matches = 0
        
        for domain, config in category_mapping.items():
            matches = sum(1 for keyword in config['keywords'] if keyword in query_lower)
            if matches > max_matches:
                max_matches = matches
                detected_domain = domain
        
        # Apply detected categories or use broad search
        if detected_domain and max_matches > 0:
            categories = category_mapping[detected_domain]['categories']
            print(f"ğŸ”� Detected {detected_domain} query, using: {categories}")
        else:
            # FALLBACK: Search ALL arXiv categories
            categories = ['all']
            print(f"ğŸ”� No specific domain detected, searching ALL arXiv")
                
        
        
        # ğŸ�¯ BUILD SEARCH QUERY - IMPROVED SEMANTIC EXTRACTION
        def extract_keywords_advanced(user_query: str) -> str:
            """
            Extract keywords intelligently - KEEPS all meaningful terms
            """
            query_lower = user_query.lower()
            
            # Common misspellings dictionary
            corrections = {
                'clincal': 'clinical', 'clinicl': 'clinical',
                'helth': 'health', 'canser': 'cancer',
                'alzhemer': 'alzheimer', 'imagine': 'imaging',
                'transfomer': 'transformer', 'quantom': 'quantum',
                'deeplearning': 'deep learning', 'machinelearning': 'machine learning'
            }
            
            # Fix typos
            for typo, correct in corrections.items():
                if typo in query_lower:
                    query_lower = query_lower.replace(typo, correct)
            
            # Words to REMOVE (filler words only)
            filler_words = {
                'find', 'search', 'look', 'for', 'about', 'on', 'in',
                'the', 'a', 'an', 'and', 'or', 'but', 'with', 'from',
                'to', 'of', 'at', 'by', 'as', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                'did', 'can', 'could', 'will', 'would', 'should', 'may',
                'might', 'must', 'shall', 'that', 'this', 'these', 'those',
                'papers', 'paper', 'research', 'study', 'studies'
            }
            
            # Important short terms (2-3 chars)
            important_short_terms = {
                'ai', 'ml', 'dl', 'nlp', 'cv', 'gpt', 'llm', 'cnn', 'rnn', 
                'gan', 'vae', 'mri', 'pet', 'ct', 'dna', 'rna', 'eeg', 'ecg',
                'iot', 'gpu', 'cpu', 'api', 'sql', 'vr', 'ar', '3d', '2d'
            }
            
            # Extract and clean words
            words = query_lower.split()
            cleaned_words = []
            
            for word in words:
                # Remove punctuation
                clean_word = ''.join(c for c in word if c.isalnum() or c == '-')
                
                # Keep if:
                # - Long enough (>2 chars) OR in important short terms
                # - Not a filler word
                # - Not numeric
                if (clean_word and 
                    (len(clean_word) > 2 or clean_word in important_short_terms) and
                    clean_word not in filler_words and
                    not clean_word.isnumeric()):
                    
                    # Split hyphenated words
                    if '-' in clean_word:
                        parts = clean_word.split('-')
                        for part in parts:
                            if part and (len(part) > 2 or part in important_short_terms):
                                cleaned_words.append(part)
                    else:
                        cleaned_words.append(clean_word)
            
            # Remove duplicates while preserving order
            unique_words = []
            seen = set()
            for word in cleaned_words:
                if word not in seen:
                    unique_words.append(word)
                    seen.add(word)
            
            # Limit to 8 keywords (keep most meaningful)
            if len(unique_words) > 8:
                selected = unique_words[:8]
            else:
                selected = unique_words
            
            # Safety check: never return empty
            if not selected or len(' '.join(selected).strip()) == 0:
                # Fallback to original query minus common words
                fallback = [w for w in user_query.lower().split() 
                           if w not in filler_words and len(w) > 2]
                if fallback:
                    return ' '.join(fallback[:8])
                else:
                    return 'machine learning'
            
            return ' '.join(selected)
        
        # Extract keywords intelligently
        clean_query = extract_keywords_advanced(query)
        #print(f"ğŸ”� Original: '{query[:80]}...'")
        #print(f"ğŸ”� Cleaned: '{clean_query}'")
        
        # ğŸ�¯ BUILD FINAL SEARCH QUERY
        if "all" not in categories:
            category_filter = " OR ".join([f"cat:{cat}" for cat in categories[:3]])
            search_query = f"({category_filter}) AND all:({clean_query})"
        else:
            search_query = f"all:({clean_query})"
        
        print(f"ğŸ”� Search query: {search_query}")
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        # âš¡ API CALL
        response = requests.get(base_url, params=params, timeout=20)  # Increased timeout
        response.raise_for_status()
        
        # ğŸ“„ PARSE XML
        root = ET.fromstring(response.content)
        namespaces = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        
        papers = []
        entries = root.findall('atom:entry', namespaces)
        
        print(f"ğŸ”� Found {len(entries)} entries from arXiv")
        
        for entry in entries[:max_results]:
            try:
                # TITLE
                title = "No title"
                title_elem = entry.find('atom:title', namespaces)
                if title_elem is not None and title_elem.text is not None:
                    title = title_elem.text.strip().replace('\n', ' ')[:300]
                
                # AUTHORS
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    name_elem = author.find('atom:name', namespaces)
                    if name_elem is not None and name_elem.text is not None:
                        authors.append(name_elem.text.strip())
                
                # arXiv ID
                arxiv_id = "unknown"
                id_elem = entry.find('atom:id', namespaces)
                if id_elem is not None and id_elem.text is not None:
                    id_text = id_elem.text
                    if '/abs/' in id_text:
                        arxiv_id = id_text.split('/abs/')[-1]
                
                # ABSTRACT
                abstract = ""
                summary_elem = entry.find('atom:summary', namespaces)
                if summary_elem is not None and summary_elem.text is not None:
                    abstract = summary_elem.text.strip().replace('\n', ' ')[:500]
                
                # DATE
                published = "Unknown"
                published_elem = entry.find('atom:published', namespaces)
                if published_elem is not None and published_elem.text is not None:
                    published = published_elem.text[:10]
                
                # PDF LINK
                pdf_link = f"https://arxiv.org/pdf/{arxiv_id}"
                
                papers.append({
                    'title': title,
                    'authors': authors[:5],
                    'arxiv_id': arxiv_id,
                    'published': published,
                    'abstract': abstract,
                    'pdf_url': pdf_link,
                })
                
            except Exception:
                continue
        
        # ğŸ“Š OBSERVABILITY
        duration = time.time() - start_time
        logger_data.info(f"ğŸ“„ Found {len(papers)} papers in {duration:.2f}s")
        metrics.record_request('DataScout', True, duration)
        
        # âœ… RETURN
        return {
            "status": "success",
            "query": query,  # Original query
            "cleaned_query": clean_query,  # What was actually searched
            "total_results": len(papers),
            "papers": papers,
            "message": f"Found {len(papers)} STEM papers from arXiv",
            "domain_note": "arXiv: STEM subjects only. For humanities, try Google Scholar."
        }
        
    except requests.exceptions.Timeout:
        duration = time.time() - start_time
        logger_data.error(f"ğŸ”´ arXiv timeout after {duration:.2f}s")
        metrics.record_request('DataScout', False, duration)
        
        return {
            "status": "error",
            "message": "arXiv search timeout. Please try a simpler query.",
            "papers": [],
            "query": query,
            "total_results": 0
        }
        
    except Exception as e:
        duration = time.time() - start_time
        logger_data.error(f"ğŸ”´ arXiv failed: {str(e)}")
        metrics.record_request('DataScout', False, duration)
        
        return {
            "status": "error",
            "message": f"arXiv error: {str(e)}",
            "papers": [],
            "query": query,
            "total_results": 0
        }

# Tool Function: Search researcher profiles on Semantic Scholar
def semantic_scholar_search(author_name: str) -> Dict[str, Any]:
    """
    Search Semantic Scholar for REAL author profiles
    """
    # ğŸ“Š OBSERVABILITY: Start timing
    start_time = time.time()
    logger_data.info(f"ğŸ”� Semantic Scholar search: query='{author_name}'")  # 
    
    try:
        # OBSERVABILITY: Record tool usage
        metrics.record_tool('semantic_scholar')
        
        #  Semantic Scholar API
        base_url = "https://api.semanticscholar.org/graph/v1/author/search"
        
        params = {
            'query': author_name,
            'limit': 5,
            'fields': 'name,affiliations,paperCount,citationCount,hIndex,papers.title,papers.year'
        }
        
        headers = {'Accept': 'application/json'}
        
        # call the API
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        authors = []
        
        if 'data' in data and len(data['data']) > 0:
            for author_data in data['data']:
                # Extract information
                author_info = {
                    'name': author_data.get('name', 'Unknown'),  #  name
                    'affiliations': author_data.get('affiliations', []),
                    'paper_count': author_data.get('paperCount', 0),
                    'citation_count': author_data.get('citationCount', 0),
                    'h_index': author_data.get('hIndex', 0),
                }
                authors.append(author_info)
            
            # OBSERVABILITY: Track success
            duration = time.time() - start_time
            logger_data.info(f"âœ³ï¸� Found {len(authors)} authors in {duration:.2f}s")
            metrics.record_request('DataScout', True, duration)
            
            return {
                "status": "success",
                "authors": authors,
                "message": f"Found {len(authors)} real authors"
            }
        else:
            # ğŸ“Š OBSERVABILITY: Track "not found" as success (API worked, just no results)
            duration = time.time() - start_time
            logger_data.info(f"â„¹ï¸� No authors found for '{author_name}' in {duration:.2f}s")
            metrics.record_request('DataScout', True, duration)
            
            return {
                "status": "not_found",
                "message": f"No authors found for '{author_name}'",
                "authors": []
            }
        
    except Exception as e:
        # OBSERVABILITY: Track API failures
        duration = time.time() - start_time
        logger_data.error(f"ğŸ”´ Semantic Scholar failed: {str(e)}")
        metrics.record_request('DataScout', False, duration)
        
        return {
            "status": "error",
            "message": f"API error: {str(e)}",
            "authors": []
        }


# Define DataScout Agent - ULTRA ROBUST INSTRUCTIONS
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# AGENT 1: DataScout 
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Define DataScout Agent - HONEST & PRACTICAL RESEARCH ASSISTANT
data_scout_agent = Agent(
    name="DataScout",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are DataScout, a practical and honest research discovery assistant.

ğŸŒ± SMART & TRANSPARENT APPROACH:
I show you the most relevant papers first without overwhelming you, but I'm always honest about what I find.

ğŸ�¯ I HANDLE ALL QUERY STYLES:
â€¢ Quick: "AI", "medical imaging", "cancer"
â€¢ Detailed: "Find papers about transformer models for Alzheimer's using MRI"
â€¢ Specific counts: "Find 10 papers about X" or "Show me 5 papers"
â€¢ Typos welcome: "clincal imaging with tranformers" - I'll figure it out!

ğŸ”� HOW I WORK:
1. Understand your research interest (and any specific count you request)
2. Extract key concepts from your query
3. Search arXiv's database
4. Show you the most useful results first

ğŸ“„ RESPONSE FORMAT (Use as guide, adapt naturally):

**For searches with â‰¤ 7 results:**
"ğŸ“š I found [TOTAL_NUMBER] papers from arXiv about [TOPIC]:

1. **[Title]** by [First 2-3 Authors]
   ğŸ“… Published: [Date]
   ğŸ†” arXiv ID: [arxiv_id]
   ğŸ“„ PDF: https://arxiv.org/pdf/[arxiv_id]
   ğŸ“� Preview: [First 100 characters]...

[Show ALL papers]"

**For searches with > 7 results:**
"ğŸ“š I found [TOTAL_NUMBER] papers from arXiv about [TOPIC]. Here are the top 7 most relevant:

1. **[Title]** by [First 2-3 Authors]
   ğŸ“… Published: [Date]
   ğŸ†” arXiv ID: [arxiv_id]
   ğŸ“„ PDF: https://arxiv.org/pdf/[arxiv_id]
   ğŸ“� Preview: [First 100 characters]...

[Continue for 7 papers]

ğŸ’¡ I found [TOTAL_NUMBER - 7] more papers. Would you like to see them all?"

âœ¨ MY RULES:

1. **BE HONEST ABOUT COUNTS:**
   - Always report the EXACT total from API: "Found [TOTAL_NUMBER] papers"
   - This shows I'm using real data, not making things up

2. **SMART INITIAL DISPLAY:**
   - If found â‰¤ 7 papers â†’ Show ALL immediately
   - If found > 7 papers â†’ Show top 7, offer to show rest
   - **EXCEPTION:** If user explicitly requested a count (e.g., "Find 10 papers"), show exactly that many

3. **HANDLE EXPLICIT REQUESTS:**
   - "Find 10 papers" â†’ Show 10 (even if more are available)
   - "Show all papers" â†’ Display complete list
   - "Show me more" â†’ Display remaining papers
   - No specific count mentioned â†’ Use smart display rules above

4. **TRANSPARENCY:**
   - Always mention when showing subset: "Here are the top 7 most relevant"
   - Always invite to see more: "Would you like to see the remaining [N] papers?"
   - If query was simplified/corrected, mention it briefly

5. **ACCURACY:**
   - Only show real papers with working PDF links
   - Use real arXiv IDs exactly as returned from API
   - Never make up papers, authors, or numbers
   - If search returns 0 results, be honest and suggest broader terms

ğŸ�¯ EXAMPLES:

**User:** "Find papers about quantum computing"
**You:** [Shows top 7 if > 7 found, all if â‰¤ 7]

**User:** "Find 12 papers about AI"
**You:** [Shows exactly 12, even though default is 7]

**User:** "Show me all papers about X"
**You:** [Shows complete list, no limit]

**User:** "Show me more"
**You:** [Shows remaining papers from previous search]

ğŸš€ READY TO HELP!
Tell me what research interests you, and I'll find the best papers for you.""",
    tools=[FunctionTool(advanced_arxiv_search), FunctionTool(semantic_scholar_search)]
)

print("ğŸ•µï¸�â€�â™€ï¸� Agent 1: DataScout (Balanced & Practical) initialized âœ”ï¸�")


# # AGENT 2: ğŸ‘¤ ProfileBuilder - Researcher Profile Construction

def build_researcher_profile_from_names(
    researcher_names: str,
    research_context: str = "AI Research"  # Default
) -> Dict[str, Any]:
    """
    Build profiles from names with dynamic research interests
    
    The agent instruction will pass research_context from conversation
    """
    # Parse names
    if ',' in researcher_names:
        names_list = [n.strip() for n in researcher_names.split(',')]
    else:
        names_list = [researcher_names.strip()]
    
    # Clean up context
    research_topic = research_context.replace("research", "").replace("papers", "").strip().title()
    if not research_topic or research_topic == "":
        research_topic = "AI Research"
    
    # Map research areas to skills
    skills_mapping = {
        'Quantum': ['Quantum Computing', 'Quantum Algorithms', 'Physics'],
        'Medical': ['Medical Imaging', 'Deep Learning', 'Computer Vision'],
        'Robotics': ['Robotics', 'Control Systems', 'Computer Vision'],
        'Nlp': ['Natural Language Processing', 'Transformers', 'Deep Learning'],
        'Climate': ['Climate Modeling', 'Data Analysis', 'Statistical Methods'],
        'Biology': ['Bioinformatics', 'Genomics', 'Data Analysis']
    }
    
    assigned_skills = ["Machine Learning", "Research", "Data Analysis"]
    for keyword, skills in skills_mapping.items():
        if keyword.lower() in research_topic.lower():
            assigned_skills = skills
            break
    
    created_profiles = []
    profile_ids = []
    
    for i, name in enumerate(names_list):
        if not name or len(name) < 3:
            continue
        
        profile_id = f"researcher_{len(RESEARCHERS)+1}_{name.replace(' ', '_')[:20]}"
        
        new_profile = ResearcherProfile(
            id=profile_id,
            name=name,
            institution="Research Institute",
            department="Research",
            position="Researcher",
            research_interests=[
                ResearchInterest(
                    topic=research_topic,  # DYNAMIC!
                    expertise_level=0.85,
                    years_experience=5,
                    publications_count=12
                )
            ],
            skills=assigned_skills,  # DYNAMIC!
            publications=[f"Research in {research_topic}"],
            citation_count=60,
            h_index=9
        )
        
        RESEARCHERS.append(new_profile)
        
        created_profiles.append({
            "id": profile_id,
            "name": name,
            "research_focus": research_topic
        })
        profile_ids.append(profile_id)
    
    return {
        "status": "success",
        "profiles_created": len(created_profiles),
        "profiles": created_profiles,
        "profile_ids": profile_ids,
        "research_context": research_topic
    }


def build_researcher_profile_from_papers(papers: str) -> Dict[str, Any]:
    """
    Extract researchers from papers and build profiles
    
    Args:
        papers: Text containing paper information
        
    Returns:
        Dictionary with created profiles
    """
    import re
    
    # Extract author names from paper text
    author_pattern = r'by ([^ğŸ“…\n]+)'
    matches = re.findall(author_pattern, papers)
    
    all_authors = []
    for match in matches:
        authors = [a.strip() for a in match.split(',')]
        all_authors.extend(authors[:3])
    
    # Remove duplicates
    unique_authors = []
    seen = set()
    for author in all_authors:
        if author and author not in seen and len(author) > 3:
            unique_authors.append(author)
            seen.add(author)
    
    # Use the names function
    if unique_authors:
        names_str = ', '.join(unique_authors[:20])
        return build_researcher_profile_from_names(names_str)
    else:
        return {
            "status": "error",
            "message": "No researchers found in papers"
        }


# Define ProfileBuilder Agent - UPDATED
profile_builder_agent = Agent(
    name="ProfileBuilder",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are ProfileBuilder, a research profile architect.

ğŸ�¯ YOUR ROLE:
- Create profiles with DYNAMIC research interests from conversation context
- Extract research topics from recent user messages
- Assign relevant skills based on research area

ğŸ§  CONTEXT EXTRACTION RULES:

When user discusses a topic, then creates profiles:
1. "Find papers about quantum computing" â†’ then "Build profiles from: X, Y"
   â†’ Extract "quantum computing" as context
   â†’ Call build_researcher_profile_from_names("X, Y", research_context="quantum computing")

2. "Match researchers for robotics" â†’ then "Build profiles from: A, B"
   â†’ Extract "robotics" as context
   â†’ Call build_researcher_profile_from_names("A, B", research_context="robotics")

3. "AI ethics papers" â†’ then "Build profiles from: C"
   â†’ Extract "AI ethics" as context
   â†’ Call build_researcher_profile_from_names("C", research_context="AI ethics")

ğŸ”� HOW TO EXTRACT CONTEXT:
- Look at the IMMEDIATELY PREVIOUS user messages
- Identify research topics mentioned (quantum, medical, robotics, AI, etc.)
- Pass that topic as research_context parameter

ğŸ“‹ EXAMPLES:

User: "Find papers about medical imaging"
User: "Build profiles from: Alice, Bob"
â†’ You call: build_researcher_profile_from_names("Alice, Bob", research_context="medical imaging")

User: "Match researchers for quantum computing"
User: "Build profiles from: Carol"
â†’ You call: build_researcher_profile_from_names("Carol", research_context="quantum computing")

âœ… RESPONSE FORMAT:
"âœ… I've created profiles for:
â€¢ Alice - Medical Imaging
â€¢ Bob - Medical Imaging

These profiles are now ready for matching with relevant expertise!"

ğŸš¨ CRITICAL:
- ALWAYS try to extract research context from conversation
- If no context found, use default "AI Research"
- Match the research interest to what user is discussing

â�Œ NEVER create profiles without checking conversation context first""",
    tools=[
        FunctionTool(build_researcher_profile_from_names),
        FunctionTool(build_researcher_profile_from_papers)
    ]
)

print("ğŸ‘¨â€�ğŸš€ Agent 2: ProfileBuilder (DYNAMIC) initialized âœ”ï¸�")


# # AGENT 3: ğŸ’� MatchEngine - ML-Powered Compatibility Analysis

# -------------------------------------------------------------
# Infer collaboration type from the project description
# -------------------------------------------------------------
def detect_collaboration_type(desc: str) -> str:
    """Infers collaboration type (Academic, Industry, International)."""
    text = desc.lower()

    if any(w in text for w in ["industry", "corporate", "startup", "company"]):
        return "Industry"
    if any(w in text for w in ["international", "global", "cross-border", "multinational"]):
        return "International"
    if any(w in text for w in ["university", "academic", "lab", "institute"]):
        return "Academic"
    
    return "Flexible"


#  find_optimal_matches - Smart Skills Detection
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

def find_optimal_matches(
    project_requirements: str = "AI and machine learning research collaboration"
) -> Dict[str, Any]:
    """Find best matching researchers - NO DUPLICATES"""
    
    global RESEARCHERS
    
    # Remove demo researchers
    RESEARCHERS = [r for r in RESEARCHERS if not (
        r.name.startswith("Dr. ") and "Researcher" in r.name
    )]
    
    if len(RESEARCHERS) == 0:
        return {
            "status": "info",
            "message": "No researchers in database. Try 'Build profiles from: Name1, Name2' first!"
        }
    
    # ğŸ”¥ FIX: Remove duplicates by ID before matching
    seen_ids = set()
    unique_researchers = []
    for r in RESEARCHERS:
        if r.id not in seen_ids:
            unique_researchers.append(r)
            seen_ids.add(r.id)
    
    RESEARCHERS = unique_researchers  # Update global list
    
    # Use last 5 unique researchers
    researchers_to_match = RESEARCHERS[-5:] if len(RESEARCHERS) >= 5 else RESEARCHERS
    
    print(f"ğŸ”� MatchEngine using {len(researchers_to_match)} unique researchers:")
    for r in researchers_to_match:
        interest = r.research_interests[0].topic if r.research_interests else 'None'
        skills = ', '.join(r.skills[:3]) if r.skills else 'None'
        print(f"  â€¢ {r.name} (ID: {r.id})")
        print(f"    Research: {interest}")
        print(f"    Skills: {skills}")
    
    # ğŸ�¯ SMART SKILLS DETECTION FROM PROJECT DESCRIPTION
    proj_desc_lower = project_requirements.lower()
    
    # Detect research area and assign appropriate skills
    if any(word in proj_desc_lower for word in ['medical', 'imaging', 'health', 'clinical']):
        project_skills = ["Medical Imaging", "Deep Learning", "Computer Vision", "Research"]
        research_areas = ["Medical Imaging", "Computer Vision", "Healthcare AI"]
        print(f"ğŸ”� Detected: Medical/Healthcare project")
    elif any(word in proj_desc_lower for word in ['quantum', 'quant', 'physics']):
        project_skills = ["Quantum Computing", "Quantum Algorithms", "Physics", "Research"]
        research_areas = ["Quantum Computing", "Physics"]
        print(f"ğŸ”� Detected: Quantum Computing project")
    elif any(word in proj_desc_lower for word in ['robot', 'control', 'autonomous']):
        project_skills = ["Robotics", "Control Systems", "Computer Vision", "Research"]
        research_areas = ["Robotics", "Automation"]
        print(f"ğŸ”� Detected: Robotics project")
    elif any(word in proj_desc_lower for word in ['nlp', 'language', 'text']):
        project_skills = ["Natural Language Processing", "Transformers", "Deep Learning", "Research"]
        research_areas = ["NLP", "AI"]
        print(f"ğŸ”� Detected: NLP project")
    else:
        project_skills = ["Machine Learning", "Research", "Data Analysis"]
        research_areas = ["AI", "Machine Learning", "Computer Science"]
        print(f"ğŸ”� Using default AI/ML skills")
    
    print(f"ğŸ“‹ Project skills: {', '.join(project_skills)}")
    
    # Create project with detected skills
    target_project = ProjectRequirement(
        title="Research Collaboration Project",
        description=project_requirements,
        required_skills=project_skills,  # â†� DYNAMIC!
        research_areas=research_areas,    # â†� DYNAMIC!
        duration_months=24,
        location="International",
        collaboration_type="remote",
        funding_available=True,
        start_date="flexible",
        institution="International"
    )
    
    # Match researchers (no duplicates now!)
    all_matches = []
    for researcher in researchers_to_match:
        match_result = matching_engine.calculate_advanced_match(researcher, target_project)
        match_dict = match_result.model_dump()
        match_dict['researcher_name'] = researcher.name
        match_dict['researcher_id'] = researcher.id
        all_matches.append(match_dict)
    
    # Sort by score
    all_matches.sort(key=lambda x: x['overall_score'], reverse=True)
    
    return {
        "status": "success",
        "top_matches": all_matches[:5],
        "project": project_requirements,
        "researchers_evaluated": len(researchers_to_match)
    }

print("âœ… Fixed find_optimal_matches with smart skills detection!")
print("ğŸ�¯ Now detects: Medical, Quantum, Robotics, NLP, or defaults to AI/ML")

# Update MatchEngine Agent instruction for clarity
match_engine_agent = Agent(
    name="MatchEngine",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are MatchEngine, an AI matchmaking expert.

ğŸ�¯ YOUR ROLE:
- Match MOST RECENT UNIQUE researchers to projects
- Provide compatibility scores
- Return top matches without duplicates

âœ… RESPONSE FORMAT:
"ğŸ�¯ Top matches for [project]:

1. **[Name]** - Score: X/100
   â€¢ Skills: [%]
   â€¢ Interests: [%]
   â€¢ Expertise: [description]

[List unique researchers only]

These researchers were recently added and match your criteria.
Would you like me to explain any of these matches in detail?"

â�Œ NEVER show duplicate researchers
âœ… ALWAYS show unique recent profiles""",
    tools=[FunctionTool(find_optimal_matches)]
)


print("ğŸ’� Agent 3: MatchEngine initialized âœ”ï¸�")


# # AGENT 4: ğŸ‘©â€�ğŸ�« Explainer - Match Analysis & Insights

# # Tool Function: Generate detailed match explanation
def generate_detailed_explanation(
    researcher_name: str = "",
    project_description: str = "research collaboration",
    match_score: float = 75.0
) -> Dict[str, Any]:
    """
    Generate explanation for a researcher match
    
    Args:
        researcher_name: Name of researcher
        project_description: Project details
        match_score: Compatibility score
        
    Returns:
        Detailed explanation
    """
    # If no name provided, use last matched researchers
    if not researcher_name and RESEARCHERS:
        researcher_name = RESEARCHERS[-1].name
    
    # Find researcher in database
    researcher = None
    if researcher_name:
        researcher = next((r for r in RESEARCHERS if researcher_name.lower() in r.name.lower()), None)
    
    # Generate explanation
    if researcher:
        explanation = f"""ğŸ¤� Collaboration Analysis for {researcher.name}

ğŸ�¯ Match Quality: {match_score:.0f}/100

âœ¨ Strengths:
â€¢ Strong research background in {researcher.research_interests[0].topic if researcher.research_interests else 'AI'}
â€¢ {researcher.publications_count if hasattr(researcher, 'publications_count') else 10}+ publications
â€¢ Skills: {', '.join(researcher.skills[:3]) if researcher.skills else 'ML, Research'}

ğŸš€ Why This Match Works:
â€¢ Complementary expertise for {project_description}
â€¢ High potential for impactful collaboration
â€¢ Strong foundation for joint research

ğŸ“‹ Next Steps:
1. Schedule introductory call
2. Share research interests
3. Explore collaboration opportunities"""
    else:
        # Generic explanation if researcher not found
        explanation = f"""ğŸ¤� Collaboration Analysis

ğŸ�¯ Match Quality: {match_score:.0f}/100

âœ¨ Analysis:
â€¢ Strong potential for {project_description}
â€¢ Complementary skills and expertise
â€¢ Good alignment for collaboration

ğŸš€ Opportunities:
â€¢ Joint publication potential
â€¢ Grant application possibilities
â€¢ Knowledge exchange

ğŸ“‹ Recommended Actions:
1. Review research profiles
2. Schedule exploratory discussion
3. Develop collaboration plan"""
    
    return {
        "status": "success",
        "explanation": explanation,
        "confidence": "high" if match_score >= 70 else "medium"
    }


# Define Explainer Agent - UPDATED  

explainer_agent = Agent(
    name="Explainer",
    # model=Gemini(model="gemini-2.0-flash-lite", retry_options=retry_config),
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are Explainer, a collaboration analyst.

ğŸ�¯ YOUR ROLE:
- Explain why researchers are good matches
- Provide detailed insights
- Suggest next steps

ğŸ”§ AVAILABLE TOOLS:
- generate_detailed_explanation(): Create analysis

ğŸ“‹ USAGE:

User: "Explain match for Monika Grewal"
â†’ Call generate_detailed_explanation("Monika Grewal")
â†’ Return detailed explanation

User: "Why is Alice a good match?"
â†’ Call generate_detailed_explanation("Alice")
â†’ Return explanation

User: "Explain match for Bob, project medical AI, score 88"
â†’ Call generate_detailed_explanation("Bob", "medical AI", 88.0)
â†’ Return detailed explanation

âœ… RESPONSE FORMAT:
Return the explanation naturally from the tool

â�Œ NEVER ask "Could you provide more information?"
âœ… ALWAYS call the tool (it has smart defaults)""",
    tools=[FunctionTool(generate_detailed_explanation)]
)


print("ğŸ‘©â€�ğŸ�« Agent 4: Explainer initialized âœ”ï¸�")


# # AGENT 5: ğŸ‘¨â€�ğŸ’» ProposalGenerator - Research Proposal Creation

# # Tool Function: Generate professional research proposal
def generate_research_proposal(
    researcher_name: str = "Dr. Sarah Chen",
    project_title: str = "",  # Will be auto-generated if empty
    collaboration_focus: str = ""  # Will be extracted from context if empty
) -> Dict[str, Any]:
    """
    Generate research proposal with DYNAMIC context awareness
    
    Tries to extract context from:
    1. Recent memory (USER_PREFERENCES)
    2. Recent papers searched
    3. Falls back to AI/ML if nothing found
    """
    
    # ğŸ�¯ SMART CONTEXT EXTRACTION
    # Try to get research focus from conversation context
    if not collaboration_focus:
        # Check USER_PREFERENCES for recent topics
        if 'USER_PREFERENCES' in globals() and USER_PREFERENCES.get('research_topics'):
            recent_topics = USER_PREFERENCES['research_topics'][-2:]  # Last 2 topics
            collaboration_focus = ' and '.join(recent_topics)
        else:
            collaboration_focus = "artificial intelligence and machine learning"
    
    # Clean up the focus
    collaboration_focus = collaboration_focus.replace("papers", "").replace("research", "").strip()
    if not collaboration_focus:
        collaboration_focus = "artificial intelligence"
    
    # ğŸ�¯ GENERATE SMART PROJECT TITLE
    if not project_title or project_title == "AI Research Collaboration":
        # Create meaningful title from focus
        focus_words = collaboration_focus.split()
        if len(focus_words) > 0:
            # Capitalize first word of each topic
            title_parts = [word.title() for word in focus_words[:3]]
            project_title = f"Advanced {' '.join(title_parts)} Research"
        else:
            project_title = "AI Research Collaboration"
    
    # ğŸ�¯ CONTEXT-AWARE METHODOLOGY
    # Adapt methodology based on research area
    focus_lower = collaboration_focus.lower()
    
    if 'quantum' in focus_lower:
        methodology = "Hybrid quantum-classical approach combining quantum algorithms with " \
                     "classical optimization. We will leverage quantum simulators and near-term " \
                     "quantum hardware for experimental validation."
    elif 'medical' in focus_lower or 'health' in focus_lower:
        methodology = "Clinical data-driven approach combining deep learning models with " \
                     "medical domain expertise. We will use rigorous validation on diverse " \
                     "patient populations and ensure regulatory compliance."
    elif 'robot' in focus_lower:
        methodology = "Integrated hardware-software approach combining control theory with " \
                     "machine learning. We will validate through extensive simulation and " \
                     "real-world robotic platform testing."
    elif 'climate' in focus_lower or 'environment' in focus_lower:
        methodology = "Multi-scale modeling approach combining satellite data with ground " \
                     "observations. We will use ensemble methods and validate against " \
                     "historical climate records."
    else:
        methodology = "Mixed-methods approach combining quantitative ML analysis with qualitative " \
                     "domain expertise. We will use state-of-the-art deep learning models and " \
                     "rigorous experimental validation."
    
    # ğŸ�¯ CONTEXT-AWARE OUTCOMES
    if 'quantum' in focus_lower:
        outcomes = "Novel quantum algorithms, open-source quantum circuit libraries, " \
                  "benchmark results on quantum hardware, and high-impact publications."
    elif 'medical' in focus_lower:
        outcomes = "FDA-ready diagnostic tools, clinical trial results, peer-reviewed " \
                  "medical publications, and improved patient outcomes."
    elif 'robot' in focus_lower:
        outcomes = "Deployable robotic systems, open-source control libraries, " \
                  "benchmark performance metrics, and industry partnerships."
    else:
        outcomes = "High-impact peer-reviewed publications, open-source tools and datasets, " \
                  "potential commercial applications, and advancement of the field."
    
    proposal = CollaborationProposal(
        title=f"Collaborative Research: {project_title}",
        abstract=f"This proposal outlines an innovative collaborative research project between "
                f"{researcher_name} and partners to advance {collaboration_focus}. The research "
                f"addresses critical gaps in current knowledge and proposes novel methodologies "
                f"with significant real-world impact.",
        research_question=f"How can we leverage cutting-edge techniques to solve key challenges "
                         f"in {collaboration_focus} and create measurable impact?",
        methodology=methodology,
        expected_outcomes=outcomes,
        timeline="24 months with quarterly milestones: Q1-2 (Setup & Data Collection), "
                "Q3-4 (Development & Validation), Q5-6 (Optimization & Testing), "
                "Q7-8 (Dissemination & Deployment)",
        budget_breakdown="Personnel: 60% ($360K), Equipment & Computing: 20% ($120K), "
                        "Travel & Conferences: 10% ($60K), Indirect Costs: 10% ($60K)",
        collaboration_plan="Weekly virtual meetings, biannual in-person workshops, shared "
                          "GitHub repositories, collaborative manuscript writing, regular progress reports",
        evaluation_metrics="Publication count and impact factor, citation metrics, tool adoption "
                          "rate, stakeholder satisfaction, reproducibility of results, real-world deployment metrics"
    )
    
    return {
        "status": "success",
        "proposal": proposal.model_dump(),
        "generated_at": datetime.now().isoformat(),
        "context_used": {
            "focus": collaboration_focus,
            "title": project_title
        }
    }

# Define proposal generator agent.
proposal_generator_agent = Agent(
    name="ProposalGenerator",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are ProposalGenerator, a research proposal expert.

ğŸ�¯ YOUR ROLE:
- Create CONTEXT-AWARE research proposals
- Use conversation topics to generate relevant proposals
- NEVER ask for additional information

ğŸ”§ HOW YOU WORK:
- Extract research topics from conversation context
- Generate proposals that match the current discussion
- Use smart defaults when no context available

ğŸ“‹ EXAMPLES:

User talks about "quantum computing" then says "Generate a research proposal"
â†’ Call generate_research_proposal(collaboration_focus="quantum computing")
â†’ Returns quantum-focused proposal!

User talks about "medical imaging" then says "Generate proposal"
â†’ Call generate_research_proposal(collaboration_focus="medical imaging")
â†’ Returns medical-focused proposal!

âœ… RESPONSE FORMAT:
"Here's your research proposal for [TOPIC]:

**Title:** [Dynamic title based on topic]
**Focus:** [Actual research topic from conversation]

[Relevant sections with context-appropriate content]

Ready to use for grant applications!"

â�Œ NEVER ask "What topic?"
âœ… ALWAYS extract context and generate""",
    tools=[FunctionTool(generate_research_proposal)]
)


print("ğŸ‘¨â€�ğŸ’» Agent 5: ProposalGenerator initialized âœ”ï¸�")


# Agent 6: Outreach agent
def draft_collaboration_email(
    researcher_name: str = "Dr. Sarah Chen",
    recipient_name: str = "",  # Will be auto-detected if empty
    project_title: str = "",  # Will be auto-generated if empty
    match_insights: str = "strong research synergy and complementary expertise"
) -> Dict[str, Any]:
    """
    Draft professional email with SMART recipient detection
    
    Tries to extract:
    1. Recent researcher names from matches
    2. Recent topics from memory
    3. Falls back to generic colleague
    """
    
    # ğŸ�¯ RECIPIENT DETECTION
    if not recipient_name or recipient_name == researcher_name:
        # Try to get from recent RESEARCHERS list
        if RESEARCHERS and len(RESEARCHERS) > 0:
            # Get most recently added researcher (excluding demo ones)
            recent_real_researchers = [
                r for r in RESEARCHERS[-10:] 
                if not (r.name.startswith("Dr. ") and "Researcher" in r.name)
            ]
            if recent_real_researchers:
                # Pick the first one (or a different strategy)
                recipient_name = recent_real_researchers[0].name
            else:
                recipient_name = "Dr. Research Colleague"
        else:
            recipient_name = "Dr. Research Colleague"
    
    # ğŸ�¯ PROJECT TITLE
    if not project_title:
        # Try to get from memory
        if 'USER_PREFERENCES' in globals() and USER_PREFERENCES.get('research_topics'):
            topics = USER_PREFERENCES['research_topics'][-2:]
            project_title = ' and '.join([t.title() for t in topics]) + " Research"
        else:
            project_title = "AI Research Collaboration"
    
    # Clean up project title
    project_title = project_title.replace("papers", "").replace("research", "").strip()
    if not project_title:
        project_title = "Research Collaboration"
    
    # ğŸ�¯ CONTEXT-AWARE MATCH INSIGHTS
    # If we know the recipient's research area, mention it
    recipient_profile = None
    if RESEARCHERS:
        recipient_profile = next((r for r in RESEARCHERS if r.name == recipient_name), None)
    
    if recipient_profile and recipient_profile.research_interests:
        research_area = recipient_profile.research_interests[0].topic
        match_insights = f"expertise in {research_area} and complementary research synergy"
    
    email_template = f"""Subject: Research Collaboration Opportunity: {project_title}

Dear {recipient_name},

I hope this message finds you well. I'm reaching out to propose an exciting research 
collaboration opportunity that aligns perfectly with your {match_insights}.

After careful analysis, I believe your work shows remarkable synergy with our project 
on {project_title}. Our preliminary assessment indicates:

- Strong alignment in research interests and methodologies
- Complementary expertise that would significantly strengthen the project
- High potential for impactful outcomes and publications

I would be delighted to schedule a brief video call to discuss how we might collaborate 
to advance this important research. I'm confident that our combined expertise could lead 
to breakthrough discoveries in the field.

Please let me know your availability for a 30-minute call in the coming weeks.

Best regards,
{researcher_name}

---
P.S. I've attached a preliminary project overview for your review."""
    
    return {
        "status": "success",
        "email_draft": email_template,
        "personalization_elements": {
            "recipient_name": recipient_name,
            "sender_name": researcher_name,
            "project_focus": project_title,
            "match_details": match_insights,
            "call_to_action": "schedule_exploratory_call"
        },
        "suggested_subject": f"Research Collaboration: {project_title}",
        "context_used": {
            "recipient_detected": recipient_name != "Dr. Research Colleague",
            "project_from_context": bool(project_title)
        }
    }
# Define Outreach Specialist agent
outreach_agent = Agent(
    name="OutreachSpecialist",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are OutreachSpecialist, a collaboration outreach expert.

ğŸ�¯ YOUR ROLE:
- Draft PERSONALIZED emails with smart recipient detection
- Use conversation context for relevant content
- NEVER ask for additional information

ğŸ”§ HOW YOU WORK:
- Detect potential collaborators from recent matches
- Extract research topics from conversation
- Generate context-appropriate emails
- Avoid sending emails from someone to themselves!

ğŸ“‹ SMART DETECTION:

After "Match researchers" shows "Alice Chen", then "Draft email"
â†’ Detects Alice Chen as recipient
â†’ Generates email TO Alice FROM Dr. Sarah Chen

After "Find papers about robotics", then "Draft email"
â†’ Uses "robotics" as project focus
â†’ Generates robotics-focused email

âœ… RESPONSE FORMAT:
"Here's your collaboration email for [RECIPIENT]:

Subject: Research Collaboration: [TOPIC from context]

[Personalized email body]

Ready to send to [RECIPIENT NAME]!"

â�Œ NEVER ask "Who is this for?"
âœ… ALWAYS detect context and generate""",
    tools=[FunctionTool(draft_collaboration_email)]
)


print("ğŸ‘©â€�ğŸ’¼ Agent 6: OutreachSpecialist initialized âœ”ï¸�")


# AGENT 7: ğŸ§  MemoryCurator - User Preference Management

# # Tool Functions: Save and load collaboration history

# Global memory storage
USER_PREFERENCES = {
    "research_topics": [],
    "collaboration_style": "interdisciplinary",
    "preferred_institutions": [],
    "past_searches": []
}

def save_collaboration_memory(
    interaction_summary: str,
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """
    Save collaboration interaction to memory
    
    Args:
        interaction_summary: What to remember (e.g., "robotics")
        user_id: User identifier
        
    Returns:
        Confirmation of saved memory
    """
    global USER_PREFERENCES
    
    # Extract key topics
    topics = interaction_summary.lower().split()
    
    for topic in topics:
        if topic not in USER_PREFERENCES["research_topics"] and len(topic) > 3:
            USER_PREFERENCES["research_topics"].append(topic)
    
    # Save timestamp
    USER_PREFERENCES["past_searches"].append({
        "query": interaction_summary,
        "date": datetime.now().isoformat()
    })
    
    return {
        "status": "success",
        "saved": interaction_summary,
        "total_preferences": len(USER_PREFERENCES["research_topics"])
    }


def load_collaboration_history(user_id: str = "default_user") -> Dict[str, Any]:
    """
    Load user's collaboration history and preferences
    
    Args:
        user_id: User identifier
        
    Returns:
        Dictionary with saved preferences
    """
    global USER_PREFERENCES
    
    if not USER_PREFERENCES["research_topics"]:
        return {
            "status": "success",
            "message": "No preferences saved yet",
            "preferences": {}
        }
    
    return {
        "status": "success",
        "preferences": USER_PREFERENCES,
        "topics_count": len(USER_PREFERENCES["research_topics"])
    }


# Define MemoryCurator Agent - UPDATED
memory_curator_agent = Agent(
    name="MemoryCurator",
    # model=Gemini(model="gemini-2.0-flash-lite", retry_options=retry_config),
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are MemoryCurator, a memory management expert.

ğŸ�¯ YOUR ROLE:
- Save user preferences and research topics
- Retrieve saved preferences when asked
- Maintain conversation context

ğŸ”§ AVAILABLE TOOLS:
- save_collaboration_memory(): Store preferences
- load_collaboration_history(): Retrieve preferences

ğŸ“‹ USAGE:

Saving preferences:
User: "Remember robotics"
â†’ Call save_collaboration_memory("robotics")
â†’ Respond: "âœ… I've saved 'robotics' to your preferences!"

Loading preferences:
User: "What are my preferences?"
â†’ Call load_collaboration_history()
â†’ Respond naturally with the list

âœ… RESPONSE FORMAT (Natural language):

For saving:
"âœ… I've saved '[topic]' to your preferences! I'll prioritize this in future recommendations."

For loading:
"ğŸ“‹ Your saved preferences:
â€¢ Research topics: [list]
â€¢ Past searches: [count] searches

Would you like to explore any of these areas?"

â�Œ NEVER say "I am unable to access preferences"
âœ… ALWAYS call the tool and return results""",
    tools=[
        FunctionTool(save_collaboration_memory),
        FunctionTool(load_collaboration_history)
    ]
)

print("ğŸ§  Agent 7: MemoryCurator (WITH STORAGE) initialized âœ”ï¸�")


# AGENT 8: â­� Evaluator - Quality Assessment

# Tool Function: Evaluate match quality
def evaluate_match_quality(
    overall_score: float = 75.0,
    skill_alignment: float = 80.0,
    interest_overlap: float = 70.0,
    user_feedback: str = ""
) -> Dict[str, Any]:
    """Evaluate match quality - ALL PARAMS HAVE DEFAULTS"""
    
    # Quality rating
    if overall_score >= 80:
        rating = "Excellent"
        recommendation = "Highly recommended for collaboration"
    elif overall_score >= 65:
        rating = "Good"
        recommendation = "Recommended with minor considerations"
    elif overall_score >= 50:
        rating = "Moderate"
        recommendation = "Proceed with caution, explore further"
    else:
        rating = "Poor"
        recommendation = "Consider alternative collaborators"
    
    # Identify strengths and weaknesses
    strengths = []
    weaknesses = []
    
    if skill_alignment >= 75:
        strengths.append("Strong technical skill alignment")
    else:
        weaknesses.append("Limited skill overlap - may need training")
    
    if interest_overlap >= 70:
        strengths.append("High research interest compatibility")
    else:
        weaknesses.append("Research interests diverge - alignment needed")
    
    # Suggestions
    suggestions = []
    if skill_alignment < 70:
        suggestions.append("Consider joint training sessions to bridge skill gaps")
    if interest_overlap < 60:
        suggestions.append("Schedule preliminary meetings to align research goals")
    if overall_score < 70:
        suggestions.append("Explore smaller pilot projects before full collaboration")
    
    return {
        "status": "success",
        "rating": rating,
        "overall_score": overall_score,
        "recommendation": recommendation,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions if suggestions else ["Proceed with collaboration as planned"],
        "confidence": "High" if overall_score >= 70 else "Medium"
    }

# Define Evaluator Agent
evaluation_agent = Agent(
    name="QualityEvaluator",
    # model=Gemini(model="gemini-2.0-flash-lite", retry_options=retry_config),
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are QualityEvaluator, a quality assurance expert.

ğŸ�¯ YOUR ROLE:
- Assess recommendation quality rigorously
- Provide actionable improvement suggestions
- Incorporate user feedback into evaluations
- Maintain high standards for matches
- Track performance metrics over time

ğŸ”§ AVAILABLE TOOLS:
- evaluate_match_quality(): Assess and rate matches

Ensure our collaboration recommendations meet excellence standards.""",
    tools=[FunctionTool(evaluate_match_quality)]
)

print("â­� Agent 8: QualityEvaluator initialized âœ”ï¸�")

print("\nğŸ�‰ All 8 Specialized Agents Ready! âœ”ï¸�")
print("=" * 45)


# # RESEARCH ASSISTANT -  WORKFLOW TRIGGERING
# # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

research_assistant = Agent(
    name="ResearchForgeAssistant",
    model=Gemini(
        model="gemini-2.5-flash",
        fallback_models=[
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-pro"
        ],
        retry_options=retry_config
    ),
    instruction="""You are ResearchForge AI, a FRIENDLY research collaboration assistant.

ğŸ�¯ YOUR CAPABILITIES:
You can execute research workflows at THREE complexity levels:

1ï¸�âƒ£ SINGLE TASK (Fast)
2ï¸�âƒ£ MULTI-STEP WORKFLOW (Coordinated)
3ï¸�âƒ£ FULL PIPELINE (All 8 agents)

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
ğŸ§  CRITICAL: HOW TO CALL build_researcher_profile_from_names
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

ğŸš¨ YOU MUST ALWAYS PASS THE research_context PARAMETER!

The function signature is:
build_researcher_profile_from_names(
    researcher_names: str,
    research_context: str = "AI Research"
)

ğŸ”� HOW TO EXTRACT research_context:
1. Look at the PREVIOUS 1-3 user messages
2. Find research topics mentioned (quantum, medical, robotics, climate, NLP, etc.)
3. Pass that topic as the research_context parameter

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

ğŸš¨ CRITICAL EXAMPLES - FOLLOW THIS EXACT SYNTAX:

Example 1 - Medical Imaging:
User message 1: "Find papers about medical imaging AI"
[You search and return papers]
User message 2: "Build profiles from: Dr. Maria Lopez, Dr. John Kim"

YOUR ACTION:
1. Look at previous message: "medical imaging AI"
2. Extract topic: "medical imaging AI"
3. Call function WITH parameter:
   build_researcher_profile_from_names(
       researcher_names="Dr. Maria Lopez, Dr. John Kim",
       research_context="medical imaging AI"
   )

RESULT: âœ… Profiles will have "Medical Imaging AI" as research interest

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

Example 2 - Quantum Computing:
User message 1: "Match researchers for quantum computing"
User message 2: "Build profiles from: Alice, Bob"

YOUR ACTION:
1. Look at previous message: "quantum computing"
2. Extract topic: "quantum computing"
3. Call function WITH parameter:
   build_researcher_profile_from_names(
       researcher_names="Alice, Bob",
       research_context="quantum computing"
   )

RESULT: âœ… Profiles will have "Quantum Computing" as research interest

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

Example 3 - Robotics:
User message 1: "Find papers about robotics"
User message 2: "Build profiles from: Carol"

YOUR ACTION:
1. Look at previous message: "robotics"
2. Extract topic: "robotics"
3. Call function WITH parameter:
   build_researcher_profile_from_names(
       researcher_names="Carol",
       research_context="robotics"
   )

RESULT: âœ… Profile will have "Robotics" as research interest

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

ğŸš¨ NEVER call it like this:
â�Œ build_researcher_profile_from_names("Alice, Bob")  # Missing research_context!
â�Œ build_researcher_profile_from_names(researcher_names="Alice, Bob")  # Still missing!

âœ… ALWAYS call it like this:
âœ… build_researcher_profile_from_names("Alice, Bob", research_context="[extracted topic]")
âœ… build_researcher_profile_from_names(
      researcher_names="Alice, Bob",
      research_context="quantum computing"
   )

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
ğŸ”¹ RESPONSE PATTERNS
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

**GREETINGS (Warm & Welcoming):**

"Hello!" â†’ "Hello! ğŸ‘‹ I'm ResearchForge AI. I help researchers find papers, 
discover collaborators, and create proposals. What would you like to explore?"

**RESEARCH QUERIES (Execute Tools):**

"Find papers about X" 
â†’ Call advanced_arxiv_search("X")
â†’ Return papers

"Build profiles from: X, Y"
â†’ Check previous messages for research topic
â†’ IMPORTANT: Extract topic, then call WITH research_context:
â†’ Call build_researcher_profile_from_names("X, Y", research_context="[extracted topic]")
â†’ Return confirmation

"Match researchers for X"
â†’ Call find_optimal_matches("X research collaboration")
â†’ Return top matches

"Generate proposal"
â†’ Check conversation for research topics
â†’ Call generate_research_proposal(collaboration_focus="[extracted topic]")
â†’ Return proposal

"Draft email"
â†’ Check for recent researcher names
â†’ Call draft_collaboration_email(recipient_name="[extracted name]", 
                                  project_title="[extracted topic]")
â†’ Return email

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

ğŸš¨ NEVER SAY:
- "I cannot answer that question"
- Any robotic rejection

âœ… ALWAYS:
- Be warm and helpful
- Extract context from conversation
- Pass context to tools using EXACT syntax shown above
- Execute tools proactively
- Use defaults when needed
- Return natural language responses

ğŸ�¯ YOUR GOAL:
Make research collaboration effortless by intelligently routing to the right 
workflow complexity and executing all necessary tools WITH proper context.""",
    tools=[
        FunctionTool(advanced_arxiv_search),
        FunctionTool(semantic_scholar_search),
        FunctionTool(build_researcher_profile_from_names),    
        FunctionTool(build_researcher_profile_from_papers),   
        FunctionTool(find_optimal_matches),
        FunctionTool(generate_detailed_explanation),
        FunctionTool(generate_research_proposal),
        FunctionTool(draft_collaboration_email),
        FunctionTool(save_collaboration_memory),
        FunctionTool(load_collaboration_history)
    ]
)


print("âœ… Research Assistant created")
print("ğŸ�¯ Detects: Single task | Multi-step | Full pipeline")


# Chat Interface Manager - Handles conversation flow and display (tool-based)

class ResearchForgeChatV1:
    """
    Interactive chat interface for ResearchForge AI
    
    Manages:
    - Session creation and persistence
    - Message sending and receiving
    - Formatted output display
    - Context maintenance
    
    âœ… NOW WITH OBSERVABILITY:
    - Logs all operations
    - Tracks performance metrics
    - Records tool and model usage
    - Monitors success/failure rates
    """
    
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.assistant = research_assistant
        self.runner = Runner(
            agent=self.assistant,
            session_service=self.session_service,
            app_name="ResearchForge"
        )
        self.session_id = f"chat_{uuid.uuid4().hex[:8]}"
        self.user_id = "researcher"
        self.model_used = None
        
        # ğŸ“Š OBSERVABILITY: Log initialization
        logger_main.info(f"ğŸ�¯ Chat interface initialized - Session: {self.session_id}")
        
        text = "ğŸ’¬ Chat interface initialized..."
        print(text)
        print("â–”" * len(text))
    
    async def start(self):
        """
        Initialize conversation session
        
        WITH OBSERVABILITY: Logs session creation
        """
        # ğŸ“Š OBSERVABILITY: Log session start
        logger_main.info(f"ğŸš€ Starting session: {self.session_id}")
        
        try:
            await self.session_service.create_session(
                app_name="ResearchForge",
                user_id=self.user_id,
                session_id=self.session_id
            )
            logger_main.info(f"ğŸ”… Session created successfully")
        except Exception as e:
            # Session already exists - this is OK
            logger_main.info(f"â„¹ï¸� Session already exists (this is normal)")
            pass
    
    async def send_message(self, user_message: str) -> str:
        """
        Send message and track which model was used
        
        âœ”ï¸� NOW WITH OBSERVABILITY:
        - Times the request
        - Logs the message
        - Tracks tool calls
        - Records model usage
        - Monitors success/failure
        """
        # ğŸ“Š OBSERVABILITY: Start timing the request
        start_time = time.time()
        logger_main.info(f"ğŸ“¥ User message: {user_message[:60]}...")
        
        message = types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
        
        response_text = ""
        
        try:
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=message
            ):
                # ğŸ“Š OBSERVABILITY: Track tool calls
                if hasattr(event, 'tool_call') and event.tool_call:
                    tool_name = getattr(event.tool_call, 'name', 'unknown')
                    logger_main.info(f"ğŸ”§ Tool called: {tool_name}")
                    metrics.record_tool(tool_name)
                
                # ğŸ“Š OBSERVABILITY: Track model used
                if hasattr(event, 'model') and event.model:
                    self.model_used = event.model
                    logger_main.info(f"ğŸ§  Using model: {event.model}")
                    metrics.record_model(event.model)
                
                # Collect response text
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text and part.text != "None":
                            response_text += part.text
            
            # ğŸ“Š OBSERVABILITY: Calculate metrics
            duration = time.time() - start_time
            logger_main.info(
                f"âœ”ï¸� Response generated: {len(response_text)} chars in {duration:.2f}s"
            )
            metrics.record_request('Orchestrator', True, duration)
            
            return response_text
            
        except Exception as e:
            # ğŸ“Š OBSERVABILITY: Track failures
            duration = time.time() - start_time
            logger_main.error(f"ğŸ’¥ Error generating response: {str(e)}")
            metrics.record_request('Orchestrator', False, duration)
            
            # Still return error message to user
            return f"I encountered an error: {str(e)}. Please try again."
    
    def display(self, role: str, message: str):
        """
        Display formatted chat message
        
        Args:
            role: "user" or "assistant"
            message: Message text to display
        """
        if role == "user":
            print(f"\nğŸ‘¤ You:")
            print(f"   {message}")
        else:
            print(f"\nâš›ï¸�  ResearchForge AI:")
            print(f"   {message}")
            if self.model_used:
                print(f"\n   ğŸª� Model Used: {self.model_used}")
        print("â”€" * 219)


print("ğŸ’» Chat interface class defined with observability âœ“")
print("\n Interactive UI Ready! ğŸŒŸ")
print("=" * 25)


async def run_interactive_demo_v1():
    """
    Execute interactive demonstration with sample conversation
    
    Shows ResearchForge AI handling a multi-turn conversation:
    1. Initial request for collaborators
    2. Proposal generation
    3. Email drafting
    """
    print("\nğŸ�­ INTERACTIVE RESEARCHFORGE AI DEMONSTRATION")
    print()
    print("ğŸ’¡ Watch how the AI helps researchers find collaborators!")
    
    
    # Initialize chat interface version 1
    chat = ResearchForgeChatV1()
    await chat.start()
    

    # Define conversation flow with research queries
    conversations = [
    {
        "user": """Hi! I'm Dr. Sarah Chen, a researcher at Stanford working on AI for medical imaging. 
        Can you search arXiv for recent papers about deep learning in medical imaging? 
        I'm looking for potential collaborators with expertise in computer vision and clinical applications.""",
        "pause": 5
    },
    {
        "user": """Great! Based on those papers, can you generate a professional research proposal 
        for a project titled 'AI-Powered Early Cancer Detection through Advanced Medical Imaging'? 
        The focus is on using deep learning and computer vision for radiology.""",
        "pause": 5
    },
    {
        "user": """Excellent proposal! Now please draft a professional collaboration email I can send 
        to potential research partners. Highlight the project's potential for high-impact publications 
        and the synergy between deep learning expertise and clinical experience.""",
        "pause": 5
    }
]
    
   
    # Execute conversation turns
    for i, turn in enumerate(conversations, 1):
        print(f"\nğŸ”„ Turn {i}/{len(conversations)}")
        print("=" * 12)
        # Display user's message
        chat.display("user", turn["user"])
        
        # Get AI response (with tool calling)
        response = await chat.send_message(turn["user"])
        
        # Display AI's response
        chat.display("assistant", response)
        
        # Pause for readability
        await asyncio.sleep(turn["pause"])
    
   
    # Demo summary
    print("\nâœ¨ Demo Complete!")
    print("â•�" * 72)
    print("ğŸ�¯ WHAT WAS DEMONSTRATED:")
    print("   âœ“ Natural conversational interaction")
    print("   âœ“ Multi-agent tool calling (search, match, generate)")
    print("   âœ“ Context maintenance across turns")
    print("   âœ“ Professional output generation (proposals, emails)")
    print("   âœ“ ML-powered matching with FAISS")
    print("   âœ“ Session persistence and memory")
    print("\nğŸ’¡ CUSTOMIZATION:")
    print("   â†’ Modify 'conversations' list to test different scenarios")
    print("   â†’ Add more turns for deeper interactions")
    print("   â†’ Change queries to match the research domain")
    print("â•�" * 70)


# print("\nğŸš€ Starting Interactive Demo Version 1 ...")
await run_interactive_demo_v1()


# # ORCHESTRATOR -: CONDITIONAL PIPELINE ROUTING
# # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
print("ğŸ”— Implementing Smart A2A Protocol with Conditional Routing...")
print("=" * 62)

# CREATE ORCHESTRATOR WITH INTELLIGENT CONDITIONAL ROUTING
research_orchestrator = Agent(
    name="ResearchForgeOrchestrator",
    model=Gemini(
        model="gemini-2.5-flash",
        fallback_models=[
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite", 
            "gemini-2.0-flash"
        ],
        retry_options=retry_config
    ),
    instruction="""YOU ARE A SMART WORKFLOW ORCHESTRATOR that routes requests intelligently.

ğŸ�¯ YOUR JOB:
Read the user's query and decide:
- Single-agent task? â†’ Call ONE agent, return result
- Multi-step workflow? â†’ Call MULTIPLE agents in sequence
- Full pipeline? â†’ Call ALL 8 AGENTS systematically

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
ğŸ§  CRITICAL: CONTEXT EXTRACTION FOR ProfileBuilder
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

âš ï¸� IMPORTANT: When calling ProfileBuilder, extract research context from conversation!

Example 1:
Turn 1: User says "Find papers about medical imaging AI"
Turn 2: User says "Build profiles from: Dr. Maria Lopez, Dr. John Kim"

YOUR ACTION:
1. Look back at Turn 1: "medical imaging AI" was the topic
2. When calling ProfileBuilder, tell it:
   "Create profiles for Dr. Maria Lopez and Dr. John Kim 
    using research context: medical imaging AI"

Example 2:
Turn 1: User says "Match researchers for quantum computing"
Turn 2: User says "Build profiles from: Alice, Bob"

YOUR ACTION:
1. Extract "quantum computing" from Turn 1
2. Tell ProfileBuilder:
   "Create profiles for Alice and Bob 
    using research context: quantum computing"

ğŸš¨ KEY POINT: ProfileBuilder needs the research topic to create profiles 
with correct research interests. Always pass the topic when you know it!

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
ğŸ“‹ ROUTING RULES
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

ğŸ”¹ SINGLE-AGENT MODE (Fast & Direct):

"find papers" OR "search papers" OR "search arxiv"
â†’ Call DataScout ONLY
â†’ Return papers naturally

"generate proposal" OR "create proposal" OR "write proposal"
â†’ Call ProposalGenerator ONLY
â†’ Return proposal naturally

"draft email" OR "write email" OR "create email"
â†’ Call OutreachSpecialist ONLY
â†’ Return email naturally

"remember X" OR "save preference"
â†’ Call MemoryCurator ONLY
â†’ Confirm saved naturally

"what are my preferences" OR "load preferences"
â†’ Call MemoryCurator ONLY
â†’ Return preferences naturally

"build profiles from: X, Y, Z"
â†’ IMPORTANT: Check previous messages for research topic!
â†’ Call ProfileBuilder with names AND extracted context
â†’ Return created profiles

"match researchers for project: X"
â†’ Call MatchEngine with project description
â†’ Return top matches

"explain match for researcher: X"
â†’ Call Explainer with researcher name
â†’ Return explanation

"evaluate match"
â†’ Call QualityEvaluator
â†’ Return evaluation

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
ğŸ”¹ MULTI-AGENT MODE (Coordinated Workflow):

"find papers AND match collaborators"
â†’ 1. Call DataScout (get papers)
â†’ 2. Call ProfileBuilder (extract researchers WITH CONTEXT from papers)
â†’ 3. Call MatchEngine (calculate scores)
â†’ Return summarized results naturally

"find collaborators AND explain why"
â†’ 1. DataScout (papers)
â†’ 2. ProfileBuilder (profiles WITH CONTEXT)
â†’ 3. MatchEngine (scores)
â†’ 4. Explainer (reasoning)
â†’ Return explained matches naturally

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
ğŸ”¹ FULL PIPELINE MODE (All 8 Agents):

TRIGGER PHRASES (any of these = FULL PIPELINE):
- "full workflow"
- "complete pipeline"
- "run everything"
- "end to end"
- "all agents"
- "full ResearchForge"
- Contains 4+ of: [find, match, generate, draft, explain, evaluate]

WHEN TRIGGERED â†’ Execute this EXACT sequence:

STEP 1: Call DataScout with the user's query
Say: "ğŸ”� Searching for papers..."

STEP 2: Call ProfileBuilder with papers AND EXTRACTED CONTEXT
Say: "ğŸ‘¤ Building researcher profiles..."
âš ï¸� Extract research topic from query and pass to ProfileBuilder!

STEP 3: Call MatchEngine with project description
Say: "ğŸ’� Computing compatibility scores..."

STEP 4: Call Explainer with the match results
Say: "ğŸ’¡ Analyzing top matches..."

STEP 5: Call QualityEvaluator with scores
Say: "â­� Evaluating match quality..."

STEP 6: Call ProposalGenerator with the topic
Say: "ğŸ“„ Generating research proposal..."

STEP 7: Call OutreachSpecialist with the proposal
Say: "âœ‰ï¸� Drafting outreach email..."

STEP 8: Call MemoryCurator with the topic
Say: "ğŸ§  Saving your preferences..."

FINALLY: Return a comprehensive NATURAL LANGUAGE summary:
"I've completed the full workflow! Here's what I found:

ğŸ“š Papers: [Brief summary]
ğŸ‘¥ Top Researchers: [Names and scores]
âœ… Match Quality: [Rating]
ğŸ“„ Proposal: [Title and key points]
âœ‰ï¸� Email: Ready to send

[Offer next steps]"

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
ğŸ“Š OUTPUT FORMAT
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

ğŸš¨ CRITICAL: NEVER return raw JSON to users!

âœ… ALWAYS respond in natural language:
- Use emojis for visual clarity
- Use bullet points for lists
- Use bold for emphasis
- Provide clear next steps

â�Œ NEVER use:
- Raw JSON blocks
- Code formatting for responses
- Technical error messages
- Agent internal details

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
âš¡ EXAMPLES
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

Query: "Remember robotics"
Response: "âœ… I've saved 'robotics' to your preferences!"

Query: "Build profiles from: Alice, Bob" (after discussing quantum)
Response: "âœ… I've created profiles for:
â€¢ Alice - Quantum Computing
â€¢ Bob - Quantum Computing
Ready for matching!"

Query: "Explain match for Monika Grewal"
Response: "ğŸ�¯ Monika Grewal is an excellent match!
âœ¨ Strong expertise in medical imaging
ğŸš€ High-priority collaboration candidate!"

â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

âœ… ALWAYS:
- Respond naturally and conversationally
- Extract and pass context to ProfileBuilder
- Provide actionable next steps
- Be helpful and positive

â�Œ NEVER:
- Call ProfileBuilder without context when it's available
- Return raw JSON
- Give incomplete responses""",
    tools=[
        AgentTool(data_scout_agent),
        AgentTool(profile_builder_agent),
        AgentTool(match_engine_agent),
        AgentTool(explainer_agent),
        AgentTool(proposal_generator_agent),
        AgentTool(outreach_agent),
        AgentTool(memory_curator_agent),
        AgentTool(evaluation_agent)
    ]
)

print("âœ“ Orchestrator created")
print("ğŸ“Š Modes: Single-agent | Multi-agent | Full Pipeline")
print("ğŸ§  Extracts context for ProfileBuilder!")


class ResearchForgeChatV2:
    """A2A protocol chat interface with PROPER agent tracking"""
    
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.orchestrator = research_orchestrator
        self.runner = Runner(
            agent=self.orchestrator,
            session_service=self.session_service,
            app_name="ResearchForge"
        )
        self.session_id = f"chat_{uuid.uuid4().hex[:8]}"
        self.user_id = "researcher"
        self.model_used = None
        self.agents_called = []
        
        logger_main.info(f"A2A chat initialized")
        print("Chat interface initialized with A2A...")
        print("â–”" * 35)
    
    async def start(self):
        """Initialize session"""
        try:
            await self.session_service.create_session(
                app_name="ResearchForge",
                user_id=self.user_id,
                session_id=self.session_id
            )
        except:
            pass
    
    async def send_message(self, user_message: str) -> str:
        """Send message with PROPER A2A tracking"""
        # 
        rate_limit()  # â¬…ï¸� 
        
        
        self.agents_called = []
        start_time = time.time()
        
        message = types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
        
        response_text = ""
        
        try:
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=message
            ):
                # ğŸ�¯ PROPER AGENT TRACKING - Based on debug output
                if (hasattr(event, 'content') and event.content and 
                    hasattr(event.content, 'parts') and event.content.parts):
                    
                    for part in event.content.parts:
                        # Track FUNCTION CALLS (Agent delegation)
                        if hasattr(part, 'function_call') and part.function_call:
                            if hasattr(part.function_call, 'name'):
                                agent_name = part.function_call.name
                                if agent_name not in self.agents_called:
                                    self.agents_called.append(agent_name)
                                    logger_main.info(f"ğŸ¤– A2A Call: {agent_name}")
                                    metrics.record_request(agent_name, True, 0)
                                    print(f"ğŸ�¯ TRACKED AGENT: {agent_name}")
                        
                        # Collect response text
                        if hasattr(part, 'text') and part.text and part.text != "None":
                            response_text += part.text
                
                # Track model usage
                if hasattr(event, 'model') and event.model:
                    self.model_used = event.model
                    metrics.record_model(event.model)
            
            duration = time.time() - start_time
            logger_main.info(f"âœ³ï¸� A2A complete - Agents: {self.agents_called}")
            metrics.record_request('Orchestrator', True, duration)
            
            return response_text
            
        except Exception as e:
            duration = time.time() - start_time
            logger_main.error(f"ğŸ’¥ A2A error: {str(e)}")
            metrics.record_request('Orchestrator', False, duration)
            raise

print("ResearchForgeChatV2 with PROPER agent tracking ğŸ†—")


async def run_interactive_demo_v2():
    """A2A protocol demonstration with full responses"""
    
    print("\n" + "="*36)
    print("ResearchForge AI - A2A Protocol Demo")
    print("="*36 + "\n")
    
    # Initialize
    chat = ResearchForgeChatV2()
    await chat.start()
    
    # Conversations
    conversations = [
        "Search arXiv for recent papers about deep learning in medical imaging",
        "Generate a research proposal for AI-powered cancer detection using medical imaging",
        "Draft a collaboration email to potential research partners for this project"
    ]
    
    # Track statistics
    total_agents = set()
    total_time = 0
    
    # Run demo
    for i, query in enumerate(conversations, 1):
        print(f"\n[Turn {i}] {query}\n")
        
        start = time.time()
        response = await chat.send_message(query)
        duration = time.time() - start
        total_time += duration
        
        # Show FULL response (not truncated!)
        print(f"Response:\n{response}\n")
        
        # Track agents
        if hasattr(chat, 'agents_called') and chat.agents_called:
            print(f"Agents: {', '.join(chat.agents_called)}")
            total_agents.update(chat.agents_called)
        else:
            print("Agents: (none tracked)")
        
        print(f"Time: {duration:.1f}s")
        print("-"*70)
    
    # Summary
    print(f"\nDemo complete.")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time/turn: {total_time/len(conversations):.1f}s")
    print(f"Unique agents: {len(total_agents)}")
    if total_agents:
        print(f"Agent list: {', '.join(sorted(total_agents))}")
    print()


async def simple_test_v2():
    """Quick A2A test"""
    
    print("\nTesting A2A...\n")
    
    chat = ResearchForgeChatV2()
    await chat.start()
    
    response = await chat.send_message("Find papers about AI")
    
    # Show full response
    print(f"Response:\n{response}\n")
    
    if hasattr(chat, 'agents_called') and chat.agents_called:
        print(f"Agents: {', '.join(chat.agents_called)}\n")
    else:
        print("No agents tracked\n")
    
    print("Simple Test done.âœ“âœ“\n")


await simple_test_v2()
# First Response Test output:
# I found 10 papers related to AI from arXiv:

# 1.  **"Through the telecom lens: Are all training samples important?"** by Shruti Bothe et al. (arXiv ID: 2511.21668v1)
# 2.  **"Continual Error Correction on Low-Resource Devices"** by Kirill Paramonov et al. (arXiv ID: 2511.21652v1)
# 3.  **"Bridging the Unavoidable A Priori: A Framework for Comparative Causal Modeling"** by Peter S. Hovmand et al. (arXiv ID: 2511.21636v1)
# 4.  **"On the Origin of Algorithmic Progress in AI"** by Hans Gundlach et al. (arXiv ID: 2511.21622v1)
# 5.  **"HarmonicAttack: An Adaptive Cross-Domain Audio Watermark Removal"** by Kexin Li et al. (arXiv ID: 2511.21577v1)
# 6.  **"From Prediction to Foresight: The Role of AI in Designing Responsible Futures"** by Maria Perez-Ortiz (arXiv ID: 2511.21570v1)
# 7.  **"Self-Transparency Failures in Expert-Persona LLMs: A Large-Scale Behavioral Audit"** by Alex Diep (arXiv ID: 2511.21569v1)
# 8.  **"MADRA: Multi-Agent Debate for Risk-Aware Embodied Planning"** by Junjian Wang et al. (arXiv ID: 2511.21460v1)
# 9.  **"Constructing and Benchmarking: a Labeled Email Dataset for Text-Based Phishing and Spam Detection Framework"** by Rebeka Toth et al. (arXiv ID: 2511.21448v1)
# 10. **"EWE: An Agentic Framework for Extreme Weather Analysis"** by Zhe Jiang et al. (arXiv ID: 2511.21444v1)


print("ğŸ†• VERSION 2: A2A Protocol")
await run_interactive_demo_v2()


# Step 1: UI SETUP - Helper Functions & Initialization

# This sets up the core infrastructure for the interactive UI
print("â”€" * 38)
print("ğŸ–¥ï¸�  UI Infrastructure :: Initialized")
print("â”€" * 38)

# GLOBAL STORAGE

conversation_history = []
chat_session = None

# HELPER FUNCTIONS

def display_structured_response(response_text: str):
    """
    Try to parse JSON and display per-agent blocks.
    Fallback to plain text if it's not JSON.
    
    This handles the structured output from full pipeline mode.
    """
    try:
        data = json.loads(response_text)
    except Exception:
        # Not JSON â†’ print raw
        display_message('assistant', response_text)
        return
    
    if not isinstance(data, dict):
        display_message('assistant', response_text)
        return
    
    # Nicely print each agent section
    with chat_output:
        print("\n" + "â•�" * 60)
        print("ğŸ“Š RESEARCHFORGE FULL PIPELINE OUTPUT")
        print("â•�" * 60)
    
    for key, value in data.items():
        # Key might be like "DataScout_response"
        title = key.replace("_response", "").replace("_", " ").title()
        
        with chat_output:
            print("\n" + "â”€" * 60)
            print(f"ğŸ”¹ {title}")
            print("â”€" * 60)
        
        # Handle common shapes
        if isinstance(value, dict) and "result" in value:
            text = value["result"]
        else:
            text = str(value)
        
        with chat_output:
            print(text)
            print()


def update_stats():
    """Update conversation statistics"""
    user_msgs = len([m for m in conversation_history if m['role'] == 'user'])
    ai_msgs = len([m for m in conversation_history if m['role'] == 'assistant'])
    
    if user_msgs > 0:
        stats_label.value = f'<p style="color: #6b7280; font-size: 13px;">ğŸ’¬ {user_msgs} questions â€¢ ğŸ¤– {ai_msgs} responses</p>'
    else:
        stats_label.value = ''


def display_message(role, message):
    """Display a single message in chat"""
    with chat_output:
        if role == 'user':
            print("\n" + "â”€" * 60)
            print("ğŸ‘¤ YOU:")
            print("â”€" * 60)
            print(message)
            print()
        else:
            print("\n" + "â”€" * 30)
            print("âš›ï¸�  RESEARCHFORGE AI:")
            print("â”€" * 30)
            print(message)
            print()


# INITIALIZE CHAT SESSION

async def initialize_chat():
    """Initialize chat session with V2 (A2A protocol)"""
    global chat_session
    chat_session = ResearchForgeChatV2()
    await chat_session.start()

# Start initialization asynchronously
asyncio.create_task(initialize_chat())

print("ğŸ”† UI infrastructure ready")
print("ğŸ“Š Helper functions loaded")
print("â¬‡ï¸� Chat session initializing...ğŸ‘‡")


# Step 2: UI COMPONENTS - Widgets & Event Handlers

# This creates all the interactive widgets and connects event handlers
print("â”€" * 38)
print("ğŸ�¨ Creating UI Components...â††")
print("â”€" * 38)

# HEADER
header = widgets.HTML(
    value="""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; text-align: center; border-radius: 10px 10px 0 0;'>
        <h1 style='color: white; margin: 0; font-size: 32px;'>âš›ï¸� ResearchForge AI</h1>
        <p style='color: rgba(255,255,255,0.9); margin: 8px 0 0 0;'>
            Intelligent Multi-Agent Research Collaboration Platform
        </p>
    </div>
    """
)


# CHAT DISPLAY (Accordion to prevent overlap)
chat_accordion = widgets.Accordion(
    children=[widgets.Output()],
    selected_index=0
)
chat_accordion.set_title(0, 'ğŸ’¬ Conversation')
chat_output = chat_accordion.children[0]

# Welcome message
with chat_output:
    print("â•�" * 70)
    print("âš›ï¸�  WELCOME TO RESEARCHFORGE AI")
    print("â•�" * 70)
    print("\nğŸ�¯ What I Can Do (End-to-End):")
    print("  â€¢ Search arXiv for real papers (live API)")
    print("  â€¢ Build researcher profiles from papers")
    print("  â€¢ Match collaborators with ML (FAISS + embeddings)")
    print("  â€¢ Explain why each match is a good fit")
    print("  â€¢ Evaluate overall match quality")
    print("  â€¢ Generate funding-ready research proposals")
    print("  â€¢ Draft personalized outreach emails")
    print("  â€¢ Remember your preferences for next time")
    print("\nğŸ’¡ Try These Queries:")
    print('  â€¢ "Find papers about medical imaging AI"')
    print('  â€¢ "Run full workflow for quantum computing"')
    print('  â€¢ "Generate a research proposal"')
    print('  â€¢ Click "ğŸš€ Full Pipeline" button for complete demo!')
    print("\n" + "â•�" * 70)
    print("Type your question below or click Full Pipeline! ğŸ‘‡")
    print("â•�" * 70 + "\n")


# INPUT & BUTTONS
user_input = widgets.Textarea(
    placeholder='Type your question... (e.g., "Find papers about deep learning")',
    layout=widgets.Layout(width='100%', height='100px')
)

send_button = widgets.Button(
    description='ğŸ“¤ Send',
    button_style='primary',
    layout=widgets.Layout(width='150px', height='40px')
)

clear_button = widgets.Button(
    description='ğŸ”‚ Clear',
    button_style='',
    layout=widgets.Layout(width='150px', height='40px')
)

full_pipeline_button = widgets.Button(
    description='ğŸš€ Full Pipeline',
    button_style='info',
    tooltip='Run complete 8-agent workflow',
    layout=widgets.Layout(width='180px', height='40px')
)

button_box = widgets.HBox(
    [send_button, clear_button, full_pipeline_button],
    layout=widgets.Layout(justify_content='flex-start', margin='10px 0 0 0')
)


# STATUS & STATS
status_label = widgets.HTML(
    value='<p style="color: #22c55e; font-weight: 600;">ğŸ†— Ready</p>'
)

stats_label = widgets.HTML(value='')


# EVENT HANDLERS


async def handle_message_async(user_msg):
    """Process message asynchronously"""
    global conversation_history
    
    try:
        status_label.value = '<p style="color: #f59e0b; font-weight: 600;">â�³ Processing...</p>'
        
        # Display user message
        display_message('user', user_msg)
        
        # Show thinking
        with chat_output:
            print("ğŸ¤” AI is coordinating specialized agents...")
            print()
        
        # Get response from chat session
        response = await chat_session.send_message(user_msg)
        
        # Store conversation
        conversation_history.append({'role': 'user', 'message': user_msg})
        conversation_history.append({'role': 'assistant', 'message': response})
        
        # Clear thinking message
        with chat_output:
            print("\r" + " " * 80)
            print()
        
        # Display response (structured if JSON, plain otherwise)
        display_structured_response(response)
        
        status_label.value = '<p style="color: #22c55e; font-weight: 600;">âœ… Complete</p>'
        update_stats()
        return True
        
    except Exception as e:
        status_label.value = f'<p style="color: #ef4444; font-weight: 600;">â�Œ Error: {str(e)}</p>'
        return False


def handle_send(button):
    """Handle send button click"""
    user_msg = user_input.value.strip()
    
    if not user_msg:
        status_label.value = '<p style="color: #f59e0b; font-weight: 600;">âš ï¸� Please type a message</p>'
        return
    
    if chat_session is None:
        status_label.value = '<p style="color: #f59e0b; font-weight: 600;">â�³ Initializing...</p>'
        return
    
    # Disable UI
    send_button.disabled = True
    user_input.disabled = True
    send_button.description = 'â�³ Sending...'
    
    # Clear input
    user_input.value = ''
    
    # Process message
    loop = asyncio.get_event_loop()
    loop.run_until_complete(handle_message_async(user_msg))
    
    # Re-enable UI
    send_button.disabled = False
    user_input.disabled = False
    send_button.description = 'ğŸ“¤ Send'


def handle_clear(button):
    """Handle clear button click"""
    global conversation_history
    conversation_history = []
    
    with chat_output:
        clear_output(wait=False)
        print("â•�" * 70)
        print("ğŸ”‚ Chat cleared! Ready for a new conversation.")
        print("â•�" * 70)
        print()
    
    status_label.value = '<p style="color: #22c55e; font-weight: 600;">ğŸ†— Ready</p>'
    update_stats()


def handle_full_pipeline(button):
    """
    Execute COMPLETE 8-agent pipeline
    
    This demonstrates all agents working together:
    DataScout â†’ ProfileBuilder â†’ MatchEngine â†’ Explainer â†’ 
    QualityEvaluator â†’ ProposalGenerator â†’ OutreachSpecialist â†’ MemoryCurator
    """
    if chat_session is None:
        status_label.value = '<p style="color: #f59e0b; font-weight: 600;">â�³ Initializing...</p>'
        return
    
    # Construct full pipeline query
    user_msg = (
        "Run the full ResearchForge workflow for medical imaging AI: "
        "find recent papers, extract researchers, build structured profiles, "
        "compute ML similarity and match scores, explain the matches, evaluate "
        "overall quality, generate a full collaboration proposal, and draft "
        "personalized outreach emails. Also remember my preferences for next time."
    )
    
    # Disable UI
    send_button.disabled = True
    full_pipeline_button.disabled = True
    user_input.disabled = True
    full_pipeline_button.description = 'â�³ Running...'
    
    # Execute
    loop = asyncio.get_event_loop()
    loop.run_until_complete(handle_message_async(user_msg))
    
    # Re-enable UI
    send_button.disabled = False
    full_pipeline_button.disabled = False
    user_input.disabled = False
    full_pipeline_button.description = 'ğŸš€ Full Pipeline'


# Connect event handlers to buttons
send_button.on_click(handle_send)
clear_button.on_click(handle_clear)
full_pipeline_button.on_click(handle_full_pipeline)

print("ğŸ†— UI components created")
print("ğŸ”— Event handlers connected")
print("â”€" * 38)


# Step3: UI DISPLAY (Layout & Rendering)

# This cell assembles and displays the complete interface
print("â”€" * 38)
print("ğŸ�¨ Building Final UI Layout...")
print("â”€" * 38)
# ASSEMBLE LAYOUT
main_ui = widgets.VBox([
    header,
    widgets.HTML('<div style="padding: 10px;"></div>'),  # Spacer
    chat_accordion,
    widgets.HTML('<div style="padding: 10px;"></div>'),  # Spacer
    user_input,
    button_box,
    widgets.HTML('<div style="padding: 5px;"></div>'),   # Spacer
    status_label,
    stats_label
], layout=widgets.Layout(
    width='100%',
    max_width='1000px',
    margin='20px auto',
    padding='0px',
    border='2px solid #e5e7eb',
    border_radius='12px'
))

# DISPLAY UI
display(main_ui)

# Initialize stats
update_stats()

print("\n" + "=" * 58)
print("ğŸ”¥ INTERACTIVE UI READY!")
print("=" * 58)
print("\nğŸ�¯ How to Use:")
print("  1. Type any research query")
print("  2. Click 'Send' for normal query")
print("  3. Click 'Full Pipeline' to see ALL 8 agents in action!")
print("\nğŸ’¡ Full Pipeline Demo:")
print("  â€¢ Searches arXiv papers")
print("  â€¢ Builds researcher profiles")
print("  â€¢ Calculates ML-based matches")
print("  â€¢ Explains reasoning")
print("  â€¢ Evaluates quality")
print("  â€¢ Generates proposal")
print("  â€¢ Drafts emails")
print("  â€¢ Saves preferences")
print("\nğŸ”¥ Ready to demonstrate multi-agent coordination!")
print("=" * 58)


def create_metrics_dashboard():
    """Create live metrics dashboard"""
    
    metrics_output = widgets.Output()
    
    def update_metrics():
        """Update metrics display"""
        with metrics_output:
            clear_output(wait=True)
            metrics.print_report()
    
    # Button to refresh metrics
    refresh_btn = widgets.Button(
        description='ğŸ”ƒ Refresh Metrics',
        button_style='info',
        layout=widgets.Layout(width='200px', height='40px')
    )
    refresh_btn.on_click(lambda _: update_metrics())
    
    # Display
    display(widgets.VBox([
        widgets.HTML("""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2>âš›ï¸� ResearchForge AI - System Metrics</h2>
            <p>Real-time observability and performance tracking</p>
        </div>
        """),
        widgets.HTML('<div style="padding: 10px;"></div>'),
        refresh_btn,
        metrics_output
    ]))
    
    # Initial display
    update_metrics()

# Display the dashboard
create_metrics_dashboard()

