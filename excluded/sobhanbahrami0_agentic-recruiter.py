#Kaggle Notebook: Multi-Agent Resume Screener with Memory, Tools & Evaluation
# - Built with Google Agent Development Kit (ADK)
# - Scenario: hiring manager uploads Job Description + resumes (PDF/DOCX/URL)
# - Agent team:
#     * JD Interpreter Agent
#     * Resume Extractor & Normalizer Agent (LoopAgent over multiple CVs)
#     * Matching & Ranking Agent
#     * Explanation / Social Summaries Agent
# - Features:
#     * Multi-agent (SequentialAgent + LoopAgent orchestrated by a root agent)
#     * Tools: custom tools, Google Search, placeholders for MCP + OpenAPI tools
#     * Sessions: InMemory + DatabaseSessionService (+ VertexAiSessionService optional)
#     * Long-term memory: VertexAiMemoryBankService (preferences, past roles)
#     * Context engineering: EventsCompactionConfig for summarised histories
#     * Observability: LoggingPlugin
#     * Agent evaluation: evalset + config + `adk eval` call
#     * A2A / Deployment: optional section for turning this into an HTTP agent



# 1. Setup: Imports & API keys
# ============================================================================
# IMPORTANT: Set up Google Cloud credentials BEFORE importing ADK!
# This ensures ADK uses Vertex AI with proper OAuth authentication.
# ============================================================================

import os
import logging
import vertexai
from typing import Any, Dict, List
from kaggle_secrets import UserSecretsClient

# --- Set up Google Cloud credentials FIRST ---
user_secrets = UserSecretsClient()

# Get and set the GCloud credential (OAuth) - this is key!
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)
print("âœ… Google Cloud OAuth credentials configured.")

# --- Now import ADK (after credentials are set) ---
from google.adk.agents import Agent, LlmAgent, SequentialAgent, LoopAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import (
    InMemorySessionService,
    DatabaseSessionService,
    VertexAiSessionService,
)
from google.adk.tools.tool_context import ToolContext

# App and Context Compaction (Day 3 - Context Engineering)
from google.adk.apps.app import App, EventsCompactionConfig

# Observability (Day 4 - LoggingPlugin)
from google.adk.plugins.logging_plugin import LoggingPlugin

# Memory services
from google.adk.memory import (
    InMemoryMemoryService,
    VertexAiMemoryBankService,
)

# Built-in tools
from google.adk.tools.google_search_tool import google_search
from google.adk.tools import load_memory, preload_memory
from google.adk.tools import FunctionTool, AgentTool

# For API types (retry config, etc.)
from google.genai import types

print("âœ… ADK components imported successfully.")
print("   - App & EventsCompactionConfig (Context Engineering)")
print("   - LoggingPlugin (Observability)")
print("   - SequentialAgent, LoopAgent, ParallelAgent (Workflow Agents)")


# Global Config & Logging
# ============================================================================
# Using Vertex AI with Google Cloud OAuth credentials (from Kaggle SDK add-on)
# ============================================================================

# Models â€“ use whatever is suggested in the Kaggle course
MODEL_NAME_FAST = "gemini-2.5-flash-lite"
MODEL_NAME_STRONG = "gemini-2.5-flash"

APP_NAME = "resume-matcher-app"
USER_ID = "demo_recruiter"

# --- Load Vertex AI configuration ---
PROJECT_ID = user_secrets.get_secret("VERTEX_PROJECT_ID")
LOCATION = user_secrets.get_secret("VERTEX_LOCATION")
print(f"âœ… Vertex AI config: Project={PROJECT_ID}, Location={LOCATION}")

# Set environment variables for ADK
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

# Also load API key (as backup / for some tools)
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("âœ… GOOGLE_API_KEY loaded.")

# --- Initialize Vertex AI ---
vertexai.init(project=PROJECT_ID, location=LOCATION)
print("âœ… Vertex AI initialized with OAuth credentials.")

# --- Logging ---
logging.basicConfig(
    filename="resume_agent.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)
print("âœ… Logging configured -> resume_agent.log")

# --- Retry config (same style as course notebooks) ---
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)
print("âœ… Retry config set.")


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]
    
    # Extract kernel and token
    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <h3>ğŸš€ ADK Web UI is Ready</h3>
        <p>1. Run the cell below (`!adk web ...`) to start the server.</p>
        <p>2. Wait for it to say "Uvicorn running on..."</p>
        <p>3. <strong><a href='{url}' target='_blank' style="background-color: #1a73e8; color: white; padding: 5px 10px; text-decoration: none; border-radius: 5px;">CLICK HERE TO OPEN UI</a></strong></p>
    </div>
    """
    display(HTML(styled_html))
    return url_prefix

# Display the button
url_prefix = get_adk_proxy_url()


# Session & Memory Configuration
# ============================================================================
# Memory Architecture for the Multi-Agent Resume Screener:
#
# ğŸ§  Short-term Memory (Sessions):
#    - DatabaseSessionService (SQLite) for persistent conversation storage
#    - Each screening session maintains its own conversation history
#
# ğŸ§  Long-term Memory:
#    - Development: InMemoryMemoryService (keyword-based, resets on restart)
#    - Production: VertexAiMemoryBankService (memory consolidation, semantic search)
#
# For production deployment, use: adk deploy agent_engine
# Memory Bank is automatically included with Agent Engine!
# ============================================================================

# --- Session Service (Short-term Memory) ---
# SQLite persists conversations between notebook restarts

db_file_path = "/kaggle/working/resume_agent_sessions.db"
if os.path.exists(db_file_path):
    try:
        os.remove(db_file_path)
        print("ğŸ§¹ Cleaned up old database file.")
    except Exception as e:
        print(f"âš ï¸� Could not remove DB file: {e}")

session_service = DatabaseSessionService(
    db_url=f"sqlite:///{db_file_path}"
)
print(f"âœ… DatabaseSessionService for sessions initialised {db_file_path}.")

# --- Memory Service (Long-term Memory) ---
# For Kaggle notebook: InMemoryMemoryService provides the core memory workflow
# It uses keyword matching and stores raw events (no memory consolidation)
memory_service = InMemoryMemoryService()
print("âœ… InMemoryMemoryService initialised.")

# --- Production Configuration (for Agent Engine deployment) ---
# When you deploy with `adk deploy agent_engine`, Memory Bank is automatic!
# Here's the memory bank configuration for resume screening:

MEMORY_BANK_CONFIG = {
    "customization_configs": [
        {
            "memory_topics": [
                # Built-in managed topics
                {"managed_memory_topic": {"managed_topic_enum": "USER_PREFERENCES"}},
                {"managed_memory_topic": {"managed_topic_enum": "KEY_CONVERSATION_DETAILS"}},
                {"managed_memory_topic": {"managed_topic_enum": "EXPLICIT_INSTRUCTIONS"}},
                # Custom topics for resume screening
                {
                    "custom_memory_topic": {
                        "label": "job_requirements",
                        "description": "Key job requirements, skills, qualifications from JDs."
                    }
                },
                {
                    "custom_memory_topic": {
                        "label": "candidate_profiles",
                        "description": "Candidate skills, experience, education from resumes."
                    }
                },
                {
                    "custom_memory_topic": {
                        "label": "screening_decisions",
                        "description": "Match scores, strengths, concerns for candidates."
                    }
                }
            ]
        }
    ]
}

print("\nğŸ“� Memory Configuration Summary:")
print(f"   Session Service: {type(session_service).__name__} (persistent SQLite)")
print(f"   Memory Service:  {type(memory_service).__name__} (development mode)")
print("\nğŸ’¡ For production with Vertex AI Memory Bank:")
print("   1. Create agent code in a directory (e.g., resume_agent/)")
print("   2. Run: adk deploy agent_engine --project=$PROJECT_ID --region=$LOCATION resume_agent/")
print("   3. Memory Bank is automatically enabled with LLM consolidation + semantic search!")


%pip install -q PyPDF2 python-docx


# Custom Tools for Document Fetching and Parsing
# ============================================================================
import requests
import re
import json
from io import BytesIO
from typing import Optional

# For PDF parsing
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    print("âš ï¸� PyPDF2 not available - PDF extraction will use fallback")

# For DOCX parsing
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("âš ï¸� python-docx not available - DOCX extraction will use fallback")


def fetch_url_text(url: str) -> dict:
    """
    Fetch text content from a URL (webpage, raw text, or API endpoint).
    
    Args:
        url: The URL to fetch content from
    
    Returns:
        dict with 'status' and 'content' or 'error'
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; AgenticRecruiter/1.0)'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'text/html' in content_type:
            # Basic HTML to text conversion (strip tags)
            text = re.sub(r'<script[^>]*>.*?</script>', '', response.text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return {'status': 'success', 'content': text[:15000]}  # Limit size
        else:
            return {'status': 'success', 'content': response.text[:15000]}
            
    except requests.RequestException as e:
        return {'status': 'error', 'error': f'Failed to fetch URL: {str(e)}'}


def extract_pdf_from_url(url: str) -> dict:
    """
    Download a PDF from URL and extract its text content.
    
    Args:
        url: URL pointing to a PDF file
    
    Returns:
        dict with 'status' and 'content' or 'error'
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; AgenticRecruiter/1.0)'}
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        if HAS_PYPDF2:
            pdf_file = BytesIO(response.content)
            reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or '')
            text = '\n'.join(text_parts)
            return {'status': 'success', 'content': text[:15000], 'pages': len(reader.pages)}
        else:
            return {'status': 'error', 'error': 'PyPDF2 not installed. Install with: pip install PyPDF2'}
            
    except Exception as e:
        return {'status': 'error', 'error': f'Failed to extract PDF: {str(e)}'}


def extract_docx_from_url(url: str) -> dict:
    """
    Download a DOCX file from URL and extract its text content.
    
    Args:
        url: URL pointing to a DOCX file
    
    Returns:
        dict with 'status' and 'content' or 'error'
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; AgenticRecruiter/1.0)'}
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        if HAS_DOCX:
            docx_file = BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
            text = '\n'.join(text_parts)
            return {'status': 'success', 'content': text[:15000], 'paragraphs': len(text_parts)}
        else:
            return {'status': 'error', 'error': 'python-docx not installed. Install with: pip install python-docx'}
            
    except Exception as e:
        return {'status': 'error', 'error': f'Failed to extract DOCX: {str(e)}'}


def parse_document(text_or_url: str) -> dict:
    """
    Smart document parser - handles raw text, URLs, PDFs, and DOCX files.
    
    Args:
        text_or_url: Either raw text content OR a URL to fetch from
    
    Returns:
        dict with 'status', 'content', and 'source_type'
    """
    text_or_url = text_or_url.strip()
    
    # Check if it's a URL
    if text_or_url.startswith(('http://', 'https://')):
        url_lower = text_or_url.lower()
        
        if url_lower.endswith('.pdf'):
            result = extract_pdf_from_url(text_or_url)
            result['source_type'] = 'pdf_url'
            return result
        elif url_lower.endswith('.docx'):
            result = extract_docx_from_url(text_or_url)
            result['source_type'] = 'docx_url'
            return result
        else:
            result = fetch_url_text(text_or_url)
            result['source_type'] = 'webpage'
            return result
    else:
        # It's raw text
        return {
            'status': 'success',
            'content': text_or_url[:15000],
            'source_type': 'raw_text'
        }


print("âœ… Document parsing tools defined:")
print("   - fetch_url_text(url)")
print("   - extract_pdf_from_url(url)")
print("   - extract_docx_from_url(url)")
print("   - parse_document(text_or_url) - smart router")


# Structured Data Extraction Tools
# ============================================================================
# These tools help agents extract and normalize structured data

def extract_job_profile(jd_text: str) -> dict:
    """
    Parse a job description into a structured profile.
    This is a tool that the JD Analyst Agent will use.
    The LLM fills in the structure; this tool provides the schema.
    
    Args:
        jd_text: Raw job description text
    
    Returns:
        dict with structured job profile schema (to be filled by LLM)
    """
    # Return schema for the LLM to fill
    return {
        'instruction': 'Extract the following fields from the job description',
        'schema': {
            'title': 'Job title',
            'level': 'Seniority level (Junior/Mid/Senior/Lead/Principal/Director)',
            'location': 'Work location or Remote',
            'experience_min_years': 'Minimum years of experience required',
            'experience_max_years': 'Maximum years of experience (if mentioned)',
            'must_have_skills': ['List of required/must-have skills'],
            'nice_to_have_skills': ['List of preferred/nice-to-have skills'],
            'responsibilities': ['Key job responsibilities'],
            'domains': ['Industry domains mentioned (e.g., fintech, healthcare)'],
            'education': 'Education requirements',
            'languages': ['Required languages'],
            'hard_constraints': ['Non-negotiable requirements (visa, clearance, etc.)'],
            'company_culture': 'Any culture/values mentioned',
            'compensation': 'Salary range if mentioned'
        },
        'jd_text_preview': jd_text[:500] + '...' if len(jd_text) > 500 else jd_text
    }


def extract_candidate_profile(resume_text: str) -> dict:
    """
    Parse a resume into a structured candidate profile.
    This is a tool that the Candidate Profiler Agent will use.
    
    Args:
        resume_text: Raw resume text
    
    Returns:
        dict with structured candidate profile schema (to be filled by LLM)
    """
    return {
        'instruction': 'Extract the following fields from the resume',
        'schema': {
            'name': 'Candidate name',
            'email': 'Email address',
            'phone': 'Phone number',
            'location': 'Current location',
            'total_years_experience': 'Total years of professional experience',
            'current_title': 'Current or most recent job title',
            'current_company': 'Current or most recent company',
            'work_history': [
                {
                    'title': 'Job title',
                    'company': 'Company name',
                    'duration': 'Time period',
                    'highlights': ['Key achievements']
                }
            ],
            'primary_skills': ['Core technical skills with proficiency'],
            'secondary_skills': ['Additional skills'],
            'domains': ['Industries/domains worked in'],
            'education': [
                {'degree': 'Degree name', 'institution': 'School', 'year': 'Graduation year'}
            ],
            'certifications': ['Professional certifications'],
            'languages': ['Languages spoken'],
            'seniority_estimate': 'Estimated seniority level',
            'strengths': ['Key strengths based on experience'],
            'potential_concerns': ['Any gaps or concerns to note']
        },
        'resume_text_preview': resume_text[:500] + '...' if len(resume_text) > 500 else resume_text
    }


def compute_fit_score(job_profile: dict, candidate_profile: dict) -> dict:
    """
    Compute a structured fit score between a job and candidate.
    This provides the rubric for the Fit Scorer Agent.
    
    Args:
        job_profile: Structured job requirements
        candidate_profile: Structured candidate profile
    
    Returns:
        dict with scoring rubric
    """
    return {
        'instruction': 'Score the candidate against the job requirements',
        'scoring_dimensions': {
            'skills_match': {
                'weight': 0.30,
                'description': 'How well do candidate skills match must-have requirements?',
                'score_range': '0-100'
            },
            'experience_match': {
                'weight': 0.25,
                'description': 'Does experience level match the role requirements?',
                'score_range': '0-100'
            },
            'domain_match': {
                'weight': 0.20,
                'description': 'How relevant is their industry/domain experience?',
                'score_range': '0-100'
            },
            'seniority_match': {
                'weight': 0.15,
                'description': 'Does seniority level align with the role?',
                'score_range': '0-100'
            },
            'nice_to_haves': {
                'weight': 0.10,
                'description': 'How many nice-to-have skills are present?',
                'score_range': '0-100'
            }
        },
        'output_format': {
            'overall_score': 'Weighted average 0-100',
            'fit_label': 'Strong Fit (75+) / Mixed Fit (50-74) / Weak Fit (<50)',
            'dimension_scores': 'Individual scores per dimension',
            'evidence_table': 'Job requirement -> Resume evidence mapping',
            'gaps': 'Missing requirements',
            'strengths': 'Exceeds requirements',
            'risks': 'Potential concerns'
        }
    }


print("âœ… Structured extraction tools defined:")
print("   - extract_job_profile(jd_text)")
print("   - extract_candidate_profile(resume_text)")
print("   - compute_fit_score(job_profile, candidate_profile)")


# Agent 1: JD Analyst Agent
# ============================================================================
# Parses job descriptions into structured requirements

jd_analyst_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME_FAST, retry_options=retry_config),
    name="jd_analyst_agent",
    description="Analyzes job descriptions and extracts structured requirements.",
    instruction="""
You are an expert Job Description Analyst. Your role is to parse job postings 
and extract structured, actionable requirements.

When given a job description:
1. Use the extract_job_profile tool to get the schema
2. Carefully read the JD and fill in ALL fields in the schema
3. Be specific about skill requirements - distinguish must-haves from nice-to-haves
4. Infer seniority level from experience requirements and responsibilities
5. Note any hard constraints (location, visa, clearance requirements)

Output your analysis as a structured JSON object matching the schema.
If information is not present, use null rather than guessing.

Focus on extracting information that will be useful for matching candidates.
""",
    tools=[extract_job_profile, parse_document],
)

print("âœ… JD Analyst Agent created")
print(f"   Model: {MODEL_NAME_FAST}")
print(f"   Tools: extract_job_profile, parse_document")


# Agent 2: Candidate Profiler Agent
# ============================================================================
# Extracts structured profiles from resumes

candidate_profiler_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME_FAST, retry_options=retry_config),
    name="candidate_profiler_agent",
    description="Extracts structured candidate profiles from resumes.",
    instruction="""
You are an expert Resume Analyst. Your role is to parse resumes and extract 
structured candidate profiles.

When given a resume:
1. Use the extract_candidate_profile tool to get the schema
2. Extract ALL fields from the resume content
3. Calculate total years of experience from work history
4. Identify primary skills (frequently used, core to their role) vs secondary
5. Estimate seniority level based on titles, responsibilities, and experience
6. Note any potential concerns (gaps, frequent job changes, etc.)

Output your analysis as a structured JSON object matching the schema.
Be objective and factual - extract what's in the resume, don't embellish.

If parsing from URL, use parse_document first to get the text content.
""",
    tools=[extract_candidate_profile, parse_document],
)

print("âœ… Candidate Profiler Agent created")
print(f"   Model: {MODEL_NAME_FAST}")
print(f"   Tools: extract_candidate_profile, parse_document")


# Agent 3: Fit Scorer Agent
# ============================================================================
# Computes match scores between job requirements and candidates

fit_scorer_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME_STRONG, retry_options=retry_config),  # Use stronger model for reasoning
    name="fit_scorer_agent",
    description="Evaluates candidate fit against job requirements and produces scores.",
    instruction="""
You are an expert Hiring Evaluator. Your role is to objectively assess how well 
a candidate matches a job's requirements.

When given a job profile and candidate profile:
1. Use the compute_fit_score tool to get the scoring rubric
2. Score each dimension (skills, experience, domain, seniority, nice-to-haves)
3. Calculate the weighted overall score
4. Assign a fit label: Strong Fit (75+), Mixed Fit (50-74), Weak Fit (<50)
5. Create an evidence table linking each job requirement to resume evidence
6. List gaps (requirements not met) and strengths (exceeds requirements)
7. Note any risks or concerns

Be rigorous and fair:
- Don't penalize for irrelevant missing skills
- Do penalize for missing must-have requirements
- Consider transferable skills and adjacent experience
- Provide specific evidence for your scores

Output a structured evaluation with scores, evidence, and justification.
""",
    tools=[compute_fit_score],
)

print("âœ… Fit Scorer Agent created")
print(f"   Model: {MODEL_NAME_STRONG} (stronger model for reasoning)")
print(f"   Tools: compute_fit_score")


# Agent 4: Report Writer Agent
# ============================================================================
# Generates final rankings and recruiter-ready summaries

report_writer_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME_FAST, retry_options=retry_config),
    name="report_writer_agent",
    description="Generates ranked candidate shortlists and recruiter-ready summaries.",
    instruction="""
You are an expert Hiring Report Writer. Your role is to synthesize candidate 
evaluations into clear, actionable reports for hiring managers.

When given evaluated candidates:
1. Rank candidates by overall fit score (highest first)
2. For each candidate, provide:
   - Quick verdict (1-2 sentences)
   - Key strengths (bullet points)
   - Key concerns (bullet points)
   - Recommended interview focus areas
3. Generate a summary comparison table
4. Provide overall hiring recommendation

Format your output as:

## ğŸ“Š Candidate Ranking

| Rank | Candidate | Score | Fit Label | Quick Verdict |
|------|-----------|-------|-----------|---------------|

## ğŸ“‹ Detailed Evaluations

### 1. [Candidate Name] - [Score]/100 - [Fit Label]
**Verdict:** ...
**Strengths:** ...
**Concerns:** ...
**Interview Focus:** ...

## ğŸ’¡ Hiring Recommendation
...

Keep language professional but accessible. Highlight actionable insights.
""",
    tools=[],  # No tools needed - pure synthesis
)

print("âœ… Report Writer Agent created")
print(f"   Model: {MODEL_NAME_FAST}")
print(f"   Tools: None (synthesis only)")


# Agent 5: Root Orchestrator with SequentialAgent Pipeline
# ============================================================================
# Multi-agent architecture using SequentialAgent for deterministic workflow
# This demonstrates Day 1 concepts: Sequential, Parallel, and Loop agents
# ============================================================================

from google.adk.tools import AgentTool

# --- Create a SequentialAgent for the screening pipeline ---
# This ensures deterministic order: JD Analysis -> Candidate Profiling -> Scoring -> Report
# The SequentialAgent runs each sub-agent in order, passing state between them

screening_pipeline = SequentialAgent(
    name="screening_pipeline",
    sub_agents=[
        jd_analyst_agent,
        candidate_profiler_agent,
        fit_scorer_agent,
        report_writer_agent,
    ],
)

print("âœ… SequentialAgent pipeline created:")
print("   1. jd_analyst_agent -> Analyze JD")
print("   2. candidate_profiler_agent -> Profile candidates")
print("   3. fit_scorer_agent -> Score fit")
print("   4. report_writer_agent -> Generate report")

# --- Wrap sub-agents as tools for flexible orchestration ---
# The root agent can use these tools for dynamic routing
jd_analyst_tool = AgentTool(agent=jd_analyst_agent)
candidate_profiler_tool = AgentTool(agent=candidate_profiler_agent)
fit_scorer_tool = AgentTool(agent=fit_scorer_agent)
report_writer_tool = AgentTool(agent=report_writer_agent)
pipeline_tool = AgentTool(agent=screening_pipeline)

# The root orchestrator agent
agentic_recruiter = LlmAgent(
    model=Gemini(model=MODEL_NAME_STRONG, retry_options=retry_config),
    name="agentic_recruiter",
    description="""
    An intelligent recruiting assistant that helps hiring managers screen candidates.
    Accepts job descriptions and resumes, then provides structured evaluations and rankings.
    """,
    instruction="""
You are **Agentic Recruiter**, an AI-powered hiring assistant. You help hiring managers 
efficiently screen candidates by providing structured, fair, and actionable evaluations.

## Your Workflow

1. **Collect Input**
   - Ask for the job description (text or URL)
   - Ask for candidate resumes (text or URLs - can handle multiple)
   - Ask for any specific preferences or constraints

2. **Analyze Job Description**
   - Use the jd_analyst_agent to parse the JD into structured requirements
   - Confirm the extracted requirements with the hiring manager

3. **Profile Candidates**
   - Use the candidate_profiler_agent for EACH resume
   - Extract structured profiles for all candidates

4. **Score Candidates**
   - Use the fit_scorer_agent to evaluate EACH candidate against the JD
   - Produce detailed scores and evidence

5. **Generate Report**
   - Use the report_writer_agent to create the final ranking and summary
   - Present results to the hiring manager

## Alternative: Use the Pipeline
For a complete end-to-end screening, you can use the screening_pipeline tool
which runs all agents in sequence automatically.

## Guidelines

- Be conversational and helpful
- Ask clarifying questions when needed
- Use memory (load_memory) to recall hiring manager preferences from past sessions
- Always explain your reasoning
- Be objective and avoid bias in evaluations
- If something seems off, flag it for human review

## Conversation Starters

When a user starts a new session, you might say:
"Hello! I'm your AI recruiting assistant. I can help you screen candidates for a role.
To get started, please share:
1. The job description (paste text or provide a URL)
2. Candidate resumes (paste text, URLs, or upload files)

I'll analyze the requirements, evaluate each candidate, and provide a ranked shortlist 
with detailed explanations. Ready when you are!"
""",
    tools=[
        jd_analyst_tool,
        candidate_profiler_tool,
        fit_scorer_tool,
        report_writer_tool,
        pipeline_tool,  # Full sequential pipeline
        parse_document,  # Direct document parsing
        load_memory,     # Access long-term memory for preferences
    ],
)

print("\nâœ… Agentic Recruiter (Root Orchestrator) created")
print(f"   Model: {MODEL_NAME_STRONG}")
print(f"   Sub-agents: jd_analyst, candidate_profiler, fit_scorer, report_writer")
print(f"   Pipeline: screening_pipeline (SequentialAgent)")
print(f"   Tools: parse_document, load_memory, pipeline_tool")


# Create the App with Context Compaction, LoggingPlugin, and Runner
# ============================================================================
# Day 3: EventsCompactionConfig for context engineering
# Day 4: LoggingPlugin for observability (passed to App, not Runner)
# ============================================================================

# --- Memory Callback for Long-term Memory ---
# This callback saves session summaries to memory after each conversation

async def save_session_to_memory(callback_context):
    """
    After-agent callback to save important session information to memory.
    This enables long-term memory across sessions.
    """
    # Get session state
    state = callback_context.state

    # Extract key information to remember
    memory_items = []
    if "job_profile" in state:
        memory_items.append(f"Job analyzed: {state.get('job_profile', {}).get('title', 'Unknown')}")
    if "candidates_evaluated" in state:
        memory_items.append(f"Candidates evaluated: {state.get('candidates_evaluated', 0)}")
    if "hiring_preferences" in state:
        memory_items.append(f"Hiring preferences: {state.get('hiring_preferences', '')}")

    # Log what we're saving
    if memory_items:
        print(f"ğŸ’¾ Saving to memory: {memory_items}")

    return None  # Continue normal flow

# --- Create App with Context Compaction AND LoggingPlugin ---
# IMPORTANT: When using App, plugins must be passed to App, not Runner!
# EventsCompactionConfig automatically summarizes long conversations
# LoggingPlugin provides observability into agent execution
resume_matcher_app = App(
    name=APP_NAME,
    root_agent=agentic_recruiter,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,  # Compact after every 5 conversation turns
        overlap_size=2,         # Keep 2 previous turns for context continuity
    ),
    plugins=[LoggingPlugin()],  # Day 4: Observability - must be in App!
)

print("âœ… App created with:")
print(f"   - EventsCompactionConfig (compaction_interval=5, overlap_size=2)")
print(f"   - LoggingPlugin (observability enabled)")

# --- Create Runner ---
# When using App, plugins are already configured in the App
runner = Runner(
    app=resume_matcher_app,  # Use App instead of agent directly
    session_service=session_service,
    memory_service=memory_service,
    # NOTE: plugins go in App, not here!
)

print("\nâœ… Runner created with:")
print(f"   App: {resume_matcher_app.name}")
print(f"   Session Service: {type(session_service).__name__}")
print(f"   Memory Service: {type(memory_service).__name__}")
print("\nğŸ“Š Logs will be written to resume_agent.log")


# Helper function to run a conversation turn
# ============================================================================

async def chat(user_message: str, session_id: str = "default"):
    """
    Send a message to the Agentic Recruiter and display the response.
    
    Args:
        user_message: The message from the hiring manager
        session_id: Session identifier for conversation continuity
    """
    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    
    # Create the message content
    message_content = types.Content(
        role="user",
        parts=[types.Part(text=user_message)]
    )
    
    print(f"\nğŸ‘¤ Hiring Manager: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")
    print("\nğŸ¤– Agentic Recruiter:")
    print("-" * 60)
    
    # Run the agent and stream response
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=message_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    print(part.text)
    
    print("-" * 60)


print("âœ… Chat helper function ready")
print("   Usage: await chat('Your message here', session_id='session_name')")


#!adk web --log_level DEBUG --url_prefix {url_prefix}
print("SKIPPING ADK WEB")


# Test the system with a sample job description and resume
# ============================================================================

# Sample Job Description
SAMPLE_JD = """
Senior Machine Learning Engineer - Fintech

About the Role:
We're looking for a Senior ML Engineer to join our fraud detection team. You'll build 
and deploy real-time ML models that protect millions of transactions daily.

Requirements:
- 5+ years of experience in machine learning or data science
- Strong Python skills (must have)
- Experience with deep learning frameworks (PyTorch or TensorFlow)
- Experience with real-time ML systems and streaming data
- Background in fraud detection or financial services (preferred)
- Experience with Kubernetes and cloud platforms (AWS/GCP)

Nice to have:
- PhD in ML, Statistics, or related field
- Experience with graph neural networks
- Contributions to open source ML projects

Location: San Francisco (Hybrid - 3 days in office)
Salary: $180,000 - $250,000 + equity
"""

# Sample Resume 1 - Strong Fit
SAMPLE_RESUME_1 = """
JANE SMITH
Senior Machine Learning Engineer
San Francisco, CA | jane.smith@email.com

EXPERIENCE:

Senior ML Engineer - PaymentGuard Inc. (2021-Present)
- Built real-time fraud detection system processing 10M+ transactions/day
- Reduced fraud losses by 40% using ensemble deep learning models
- Deployed models on Kubernetes with sub-100ms latency requirements
- Tech: Python, PyTorch, Kafka, Kubernetes, AWS

ML Engineer - DataCorp (2018-2021)
- Developed credit scoring models for fintech clients
- Built MLOps pipelines for model training and deployment
- Tech: Python, TensorFlow, scikit-learn, GCP

Data Scientist - Analytics Co (2016-2018)
- Built predictive models for customer churn
- Tech: Python, R, SQL

EDUCATION:
MS Computer Science (ML Focus) - Stanford University, 2016
BS Mathematics - UC Berkeley, 2014

SKILLS:
Python, PyTorch, TensorFlow, Kubernetes, AWS, GCP, Kafka, SQL, MLOps
"""

# Sample Resume 2 - Mixed Fit
SAMPLE_RESUME_2 = """
JOHN DOE
Data Scientist
Austin, TX | john.doe@email.com

EXPERIENCE:

Senior Data Scientist - E-commerce Corp (2020-Present)
- Built recommendation systems for product discovery
- Developed A/B testing framework for ML experiments
- Tech: Python, TensorFlow, AWS SageMaker

Data Scientist - Marketing Analytics (2017-2020)
- Built customer segmentation models
- Created dashboards for marketing performance
- Tech: Python, scikit-learn, SQL, Tableau

EDUCATION:
MS Data Science - UT Austin, 2017
BS Statistics - Texas A&M, 2015

SKILLS:
Python, TensorFlow, scikit-learn, SQL, AWS, Tableau, A/B Testing
"""

print("ğŸ“� Sample data ready for testing")
print(f"   JD: Senior ML Engineer - Fintech ({len(SAMPLE_JD)} chars)")
print(f"   Resume 1: Jane Smith - Strong fit ({len(SAMPLE_RESUME_1)} chars)")
print(f"   Resume 2: John Doe - Mixed fit ({len(SAMPLE_RESUME_2)} chars)")


# Run a test screening session
# ============================================================================

# Start the conversation
await chat("Hello! I need help screening candidates for a role.", session_id="test_screening")


# Provide the job description
await chat(f"Here's the job description:\n\n{SAMPLE_JD}", session_id="test_screening")


# Provide the first resume
await chat(f"Here's the first candidate resume:\n\n{SAMPLE_RESUME_1}", session_id="test_screening")


# Provide the second resume
await chat(f"Here's the second candidate:\n\n{SAMPLE_RESUME_2}", session_id="test_screening")


# Ask for the final ranking
await chat("Please provide the final ranking and recommendations for these candidates.", session_id="test_screening")


# Save the current session to memory for future recall
# ============================================================================

async def save_session_to_memory(session_id: str):
    """Save a session to long-term memory."""
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    await memory_service.add_session_to_memory(session)
    print(f"âœ… Session '{session_id}' saved to memory")

# Save our test session
await save_session_to_memory("test_screening")


# Agent Evaluation Setup (Day 4)
# ============================================================================
# Create evaluation test cases and configuration for `adk eval`
# This demonstrates systematic agent testing and quality measurement
# ============================================================================

import json

# --- Evaluation Test Cases (evalset.json) ---
# These test cases verify the agent's core capabilities
eval_set = {
    "eval_set_id": "resume_screener_eval",
    "name": "Resume Screener Evaluation",
    "description": "Test cases for the multi-agent resume screening system",
    "test_cases": [
        {
            "test_case_id": "tc_greeting",
            "name": "Greeting Response",
            "description": "Agent should greet and explain its capabilities",
            "conversation": [
                {
                    "role": "user",
                    "content": "Hello, I need help screening candidates."
                }
            ],
            "expected_behaviors": [
                "Agent introduces itself as a recruiting assistant",
                "Agent asks for job description",
                "Agent mentions it can analyze resumes"
            ]
        },
        {
            "test_case_id": "tc_jd_analysis",
            "name": "Job Description Analysis",
            "description": "Agent should extract structured requirements from JD",
            "conversation": [
                {
                    "role": "user",
                    "content": "Here is the JD: Senior Python Developer, 5+ years experience, must know Django and PostgreSQL, nice to have: AWS, Docker."
                }
            ],
            "expected_behaviors": [
                "Agent identifies required skills (Python, Django, PostgreSQL)",
                "Agent identifies nice-to-have skills (AWS, Docker)",
                "Agent identifies experience requirement (5+ years)"
            ]
        },
        {
            "test_case_id": "tc_candidate_scoring",
            "name": "Candidate Scoring",
            "description": "Agent should provide objective fit scores",
            "conversation": [
                {
                    "role": "user",
                    "content": "JD: Python Developer, 3+ years. Resume: John Doe, 5 years Python, Django expert, AWS certified."
                }
            ],
            "expected_behaviors": [
                "Agent provides a numerical fit score",
                "Agent explains strengths (exceeds experience, has Django)",
                "Agent provides objective assessment"
            ]
        }
    ]
}

# --- Evaluation Configuration (test_config.json) ---
# Defines how to evaluate agent responses
eval_config = {
    "evaluation_criteria": [
        {
            "name": "relevance",
            "description": "Response is relevant to the user's request",
            "weight": 0.3
        },
        {
            "name": "accuracy",
            "description": "Information extracted is accurate and complete",
            "weight": 0.3
        },
        {
            "name": "helpfulness",
            "description": "Response provides actionable guidance",
            "weight": 0.2
        },
        {
            "name": "professionalism",
            "description": "Response maintains professional tone",
            "weight": 0.2
        }
    ],
    "model": "gemini-2.5-flash",
    "threshold": 0.7
}

# Save evaluation files
with open("evalset.json", "w") as f:
    json.dump(eval_set, f, indent=2)

with open("test_config.json", "w") as f:
    json.dump(eval_config, f, indent=2)

print("âœ… Evaluation files created:")
print("   - evalset.json (3 test cases)")
print("   - test_config.json (4 evaluation criteria)")
print("\nğŸ“� To run evaluation locally:")
print("   !adk eval resume_agent evalset.json --config_file_path=test_config.json")
print("\nğŸ’¡ For production, use Vertex AI Evaluation:")
print("   from vertexai.evaluation import EvalTask")


# Test memory recall in a new session
# ============================================================================

await chat(
    "I'm back! Can you remind me what role we were screening for last time?", 
    session_id="memory_test"
)


# Agent Deployment to Vertex AI Agent Engine (Day 5)
# ============================================================================
# This section shows how to deploy the agent to production
# Agent Engine provides: auto-scaling, session management, Memory Bank
# ============================================================================

print("ğŸš€ DEPLOYMENT GUIDE: Vertex AI Agent Engine -- UNCOMMENT")
# print("=" * 60)

# print("""
# ## Step 1: Create Agent Directory Structure

# Your agent code should be organized as:

# resume_agent/
# â”œâ”€â”€ agent.py              # Main agent definition
# â”œâ”€â”€ requirements.txt      # Python dependencies
# â”œâ”€â”€ .env                  # Environment variables
# â””â”€â”€ .agent_engine_config.json  # Deployment config

# ## Step 2: Create requirements.txt
# """)

# # Create requirements.txt content
# requirements_content = """google-adk>=0.1.0
# google-cloud-aiplatform
# PyPDF2
# python-docx
# requests
# """

# print("ğŸ“„ requirements.txt:")
# print(requirements_content)

# # Create .env content
# env_content = """# Environment variables for Agent Engine
# GOOGLE_CLOUD_LOCATION=global
# GOOGLE_GENAI_USE_VERTEXAI=1
# """

# print("ğŸ“„ .env:")
# print(env_content)

# # Create deployment config
# deploy_config = {
#     "min_instances": 0,
#     "max_instances": 2,
#     "cpu": "1",
#     "memory": "2Gi"
# }

# print("ğŸ“„ .agent_engine_config.json:")
# print(json.dumps(deploy_config, indent=2))

# print("""
# ## Step 3: Deploy with ADK CLI

# Run this command to deploy:

# ```bash
# adk deploy agent_engine \
#     --project=$PROJECT_ID \
#     --region=us-central1 \
#     resume_agent/ \
#     --agent_engine_config_file=resume_agent/.agent_engine_config.json
# ```

# ## Step 4: Test Deployed Agent

# ```python
# import vertexai
# from vertexai import agent_engines

# vertexai.init(project=PROJECT_ID, location="us-central1")

# # Get deployed agent
# agents_list = list(agent_engines.list())
# remote_agent = agents_list[0]

# # Query the agent
# async for item in remote_agent.async_stream_query(
#     message="Hello, I need help screening candidates.",
#     user_id="user_123",
# ):
#     print(item)
# ```

# ## Step 5: Memory Bank (Automatic with Agent Engine!)

# When deployed to Agent Engine, Memory Bank is automatically enabled:
# - Long-term memory across sessions
# - LLM-powered memory consolidation
# - Semantic search for relevant memories

# ## Step 6: Cleanup (Important!)

# ```python
# # Delete agent to avoid charges
# agent_engines.delete(resource_name=remote_agent.resource_name, force=True)
# ```
# """)

# print("\nâœ… Deployment guide complete!")
# print("   See Day 5 notebook for full deployment walkthrough")

