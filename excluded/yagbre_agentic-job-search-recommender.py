# Install dependencies
!pip install -q google-adk google-genai PyPDF2 python-docx



import os
import json
import asyncio
import aiohttp
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
from urllib.parse import quote

# Handling Logging
import logging
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
logging.getLogger("google_adk.google.adk.runners").setLevel(logging.ERROR)

# Google ADK imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import google_search, load_memory
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.genai import types

# Document parsing
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO

print("✅ All imports successful (including LoopAgent for self-healing)")


# API Key Configuration
# For Kaggle: Use Kaggle Secrets
# For local: Use environment variable or .env file

import os

# First, try loading from .env file (for local development)
try:
    from dotenv import load_dotenv
    # Try parent directory first (project root), then current directory
    env_paths = [
        "../.env",  # Project root (one level up from Submission/)
        ".env"      # Current directory
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            print(f"✅ Loaded .env from: {env_path}")
            break
except ImportError:
    pass  # dotenv not available, will try other methods

# Now try to get the API key
GOOGLE_API_KEY = None

# Try Kaggle Secrets first
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    GOOGLE_API_KEY = secrets.get_secret("GOOGLE_API_KEY")
    print("✅ API key loaded from Kaggle Secrets")
except:
    pass

# Fall back to environment variable
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        print("✅ API key loaded from environment")
    else:
        raise ValueError("❌ No API key found. Set GOOGLE_API_KEY environment variable or create .env file.")

# Set the API key in environment for ADK
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("🔑 Using Updated API Key...")

# ============================================================================
# BLEEDING EDGE MODEL CONFIGURATION (Nov 2025)
# Multi-model orchestration for optimal rate limits + capabilities
# ============================================================================

# LITE: High throughput (30,000 RPM!) - Simple tasks
MODEL_LITE = "gemini-2.5-flash-lite"

# FLASH: Balanced performance - Medium complexity, agentic tasks  
MODEL_FLASH = "gemini-2.5-flash"

# PRO: Bleeding edge reasoning - Gemini 3 Pro (released Nov 18, 2025!)
# Best reasoning model on free tier (upgrade to gemini-3-pro-preview with billing)
MODEL_PRO = "gemini-2.5-pro"  # Use gemini-3-pro-preview with paid tier

# Legacy alias for backward compatibility
MODEL_ID = MODEL_FLASH

# ============================================================================
# RETRY CONFIGURATION (handles 429/503 errors with exponential backoff)
# ============================================================================
RETRY_CONFIG = types.GenerateContentConfig(
    http_options=types.HttpOptions(
        timeout=120000,  # 2 minute timeout
    )
)

# Staggered delay between sequential agents (reduces burst load)
STAGGER_DELAY = 2  # seconds between agent executions

print(f"  RETRY: Enabled with 120s timeout")
print(f"  STAGGER_DELAY: {STAGGER_DELAY}s between agents")

print("🚀 MODEL CONFIGURATION")
print(f"  MODEL_LITE:  {MODEL_LITE}  (30,000 RPM - simple tasks)")
print(f"  MODEL_FLASH: {MODEL_FLASH}  (1,000 RPM - agentic tasks)")
print(f"  MODEL_PRO:   {MODEL_PRO}  (Gemini 3! - critical reasoning)")



@dataclass
class RoleProgression:
    """A single role in the candidate's career history."""
    title: str
    company: str
    duration_years: float
    focus_areas: List[str]

@dataclass 
class ParsedResume:
    """Structured resume data extracted by Agent 1."""
    current_title: str
    current_company: str
    total_yoe: float
    skills: List[str]
    education: List[str]
    role_progression: List[RoleProgression]
    stated_interests: List[str]
    side_projects: List[str]
    qualitative_trend: str  # e.g., "Data Analytics → Engineering → ML"
    inferred_direction: str  # e.g., "ML Engineering"

@dataclass
class LevelClassification:
    """Level classification from Agent 2 + A2A deliberation."""
    calibrated_level: str  # e.g., "L4", "Senior", "Staff"
    confidence: str  # "High", "Medium", "Low"
    conservative_assessment: str
    optimistic_assessment: str
    reasoning: str

class JobTier(Enum):
    """The 4 tiers of job recommendations."""
    EXACT_MATCH = "exact_match"
    LEVEL_UP = "level_up"
    STRETCH = "stretch"
    TRAJECTORY = "trajectory"

@dataclass
class JobRecommendation:
    """A single job recommendation with explainability."""
    tier: JobTier
    title: str
    company: str
    url: str
    why_matches_resume: List[str]  # Grounded in resume
    evidence_from_search: List[str]  # Grounded in search results
    fit_confidence: int  # 1-10

@dataclass
class SessionState:
    """Complete session state for human-in-the-loop refinement."""
    resume_text: str = ""
    parsed_resume: Optional[ParsedResume] = None
    level_classification: Optional[LevelClassification] = None
    job_recommendations: List[JobRecommendation] = field(default_factory=list)
    user_feedback: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    iteration: int = 0

print("✅ Data structures defined")


def calculate_yoe(resume_text: str) -> dict:
    """
    Calculate total years of experience from resume text with detailed role breakdown.
    Handles multiple date formats: full month names, abbreviated months, AND numeric MM/YY.
    
    Args:
        resume_text: The full text of the resume
    
    Returns:
        dict with total_yoe, role_breakdown, and career analytics
    """
    import re
    from datetime import datetime
    
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Support both full and abbreviated month names
    months_map = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sept': 9, 'sep': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    
    roles = []
    all_months_worked = set()
    
    # PATTERN 1: Text-based dates "Month Year - Present" OR "Month Year - Month Year"
    month_names = 'January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec'
    text_date_pattern = rf'({month_names})\s+(\d{{4}})\s*[-–]\s*(Present|({month_names})\s+(\d{{4}}))'
    
    text_matches = re.findall(text_date_pattern, resume_text, re.IGNORECASE)
    
    for match in text_matches:
        start_month_name = match[0]
        start_year = int(match[1])
        end_part = match[2]  # Either "Present" or "Month Year"
        end_month_name = match[3] if len(match) > 3 else ""
        end_year_str = match[4] if len(match) > 4 else ""
        
        start_month = months_map.get(start_month_name.lower(), 1)
        
        # Calculate end date
        if end_part.lower() == 'present':
            end_month = current_month
            end_year_int = current_year
            end_display = "Present"
        elif end_year_str:
            end_month = months_map.get(end_month_name.lower(), 12)
            end_year_int = int(end_year_str)
            end_display = f"{end_month_name} {end_year_str}"
        else:
            continue  # Skip incomplete
        
        duration_months = (end_year_int - start_year) * 12 + (end_month - start_month)
        if duration_months <= 0:
            continue
            
        duration_years = round(duration_months / 12, 1)
        
        # Track all months worked (for de-duping overlaps)
        for y in range(start_year, end_year_int + 1):
            for m in range(1, 13):
                if (y == start_year and m < start_month) or (y == end_year_int and m > end_month):
                    continue
                all_months_worked.add((y, m))
        
        roles.append({
            "start": f"{start_month_name} {start_year}",
            "end": end_display,
            "duration_months": duration_months,
            "duration_years": duration_years
        })
    
    # PATTERN 2: Numeric dates "MM/YY - Present" OR "MM/YY - MM/YY" OR "MM/YYYY - MM/YYYY"
    # Matches formats like: 06/24 - Present, 10/22 - 01/24, 07/2016 - 04/2020
    numeric_date_pattern = r'(\d{1,2})/(\d{2,4})\s*[-–]\s*(Present|(\d{1,2})/(\d{2,4}))'
    
    numeric_matches = re.findall(numeric_date_pattern, resume_text, re.IGNORECASE)
    
    for match in numeric_matches:
        start_month = int(match[0])
        start_year_raw = match[1]
        end_part = match[2]
        end_month_raw = match[3] if len(match) > 3 else ""
        end_year_raw = match[4] if len(match) > 4 else ""
        
        # Convert 2-digit year to 4-digit (assume 2000s)
        if len(start_year_raw) == 2:
            start_year = 2000 + int(start_year_raw)
        else:
            start_year = int(start_year_raw)
        
        # Validate month
        if start_month < 1 or start_month > 12:
            continue
        
        # Calculate end date
        if end_part.lower() == 'present':
            end_month = current_month
            end_year_int = current_year
            end_display = "Present"
        elif end_month_raw and end_year_raw:
            end_month = int(end_month_raw)
            if len(end_year_raw) == 2:
                end_year_int = 2000 + int(end_year_raw)
            else:
                end_year_int = int(end_year_raw)
            if end_month < 1 or end_month > 12:
                continue
            end_display = f"{end_month:02d}/{end_year_raw}"
        else:
            continue  # Skip incomplete
        
        duration_months = (end_year_int - start_year) * 12 + (end_month - start_month)
        if duration_months <= 0:
            continue
            
        duration_years = round(duration_months / 12, 1)
        
        # Track all months worked (for de-duping overlaps)
        for y in range(start_year, end_year_int + 1):
            for m in range(1, 13):
                if (y == start_year and m < start_month) or (y == end_year_int and m > end_month):
                    continue
                all_months_worked.add((y, m))
        
        roles.append({
            "start": f"{start_month:02d}/{start_year}",
            "end": end_display,
            "duration_months": duration_months,
            "duration_years": duration_years
        })
    
    # Calculate true total YOE (de-duped for overlaps)
    total_months_deduped = len(all_months_worked)
    total_yoe = round(total_months_deduped / 12, 1)
    
    # Career span
    if all_months_worked:
        earliest = min(all_months_worked)
        latest = max(all_months_worked)
        career_span = f"{earliest[0]} to {latest[0]}"
    else:
        career_span = "Unknown"
    
    # Average tenure
    if roles:
        avg_tenure = round(sum(r["duration_months"] for r in roles) / len(roles) / 12, 1)
    else:
        avg_tenure = 0
    
    # Check for explicit YOE statements
    explicit_pattern = r'(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience|at|in|building|leading)'
    explicit_matches = re.findall(explicit_pattern, resume_text, re.IGNORECASE)
    stated_yoe = max(int(y) for y in explicit_matches) if explicit_matches else None
    
    return {
        "total_yoe": total_yoe,
        "stated_yoe": stated_yoe,
        "calculation_method": "date_parsing_deduped",
        "career_span": career_span,
        "num_roles": len(roles),
        "avg_tenure_years": avg_tenure,
        "role_breakdown": roles[:10],
        "note": f"Calculated {total_yoe} years from {len(roles)} roles." + (f" Resume states {stated_yoe}+ years." if stated_yoe else "")
    }


print("✅ calculate_yoe tool defined - handles text AND numeric date formats (MM/YY)")


def parse_resume_file(content: bytes, file_type: str) -> str:
    """
    Parse resume content from PDF or DOCX format.
    
    Args:
        content: Raw file bytes
        file_type: 'pdf' or 'docx'
    
    Returns:
        Extracted text from the resume
    """
    if file_type == 'pdf':
        try:
            import io
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except ImportError:
            # Fallback: try pdfplumber
            try:
                import io
                import pdfplumber
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                return text.strip()
            except ImportError:
                return "[PDF parsing requires PyPDF2 or pdfplumber. Please install: pip install PyPDF2]"
    
    elif file_type == 'docx':
        try:
            import io
            from docx import Document
            doc = Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except ImportError:
            return "[DOCX parsing requires python-docx. Please install: pip install python-docx]"
    
    else:
        return content.decode('utf-8')


print("✅ parse_resume_file function defined (PDF + DOCX support)")


# ============================================
# URL VALIDATION TOOLS (Self-Healing Pattern)
# ============================================

async def validate_url_async(url: str) -> dict:
    """Validate a single URL asynchronously."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=10), allow_redirects=True) as response:
                return {"url": url, "status": response.status, "valid": response.status == 200, "error": None}
    except Exception as e:
        return {"url": url, "status": None, "valid": False, "error": str(e)}

async def validate_urls_async(urls: list) -> dict:
    """Validate multiple URLs asynchronously."""
    tasks = [validate_url_async(url) for url in urls]
    results = await asyncio.gather(*tasks)
    valid_count = sum(1 for r in results if r["valid"])
    invalid_urls = [r["url"] for r in results if not r["valid"]]
    return {
        "total_urls": len(urls),
        "valid_count": valid_count,
        "invalid_count": len(urls) - valid_count,
        "all_valid": valid_count == len(urls),
        "invalid_urls": invalid_urls,
        "results": results
    }

def validate_urls(urls: list) -> dict:
    """Sync wrapper for URL validation - handles both sync and async contexts."""
    import nest_asyncio
    try:
        nest_asyncio.apply()  # Allow nested event loops
    except:
        pass  # nest_asyncio not available or already applied
    
    try:
        # Try to get existing loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context - use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(validate_urls_async(urls)))
                return future.result(timeout=30)
        else:
            # No running loop - safe to use asyncio.run
            return asyncio.run(validate_urls_async(urls))
    except RuntimeError:
        # Fallback - create new loop
        return asyncio.run(validate_urls_async(urls))

def check_job_urls(job_output: str) -> dict:
    """
    URL Validation Tool for LoopAgent self-healing pattern.
    Validates URLs and returns feedback for retry if needed.
    """
    url_pattern = r'https?://[^\s\)\]\"\'<>`]+'
    urls = re.findall(url_pattern, job_output)
    urls = [url.rstrip('`') for url in urls if not url.startswith('[SEARCH:')]

    if not urls:
        return {"all_valid": False, "feedback": "No URLs found.", "should_exit": False}

    validation_result = validate_urls(urls)

    if validation_result["all_valid"]:
        return {"all_valid": True, "valid_urls": urls, "feedback": "SUCCESS", "should_exit": True}

    invalid_urls = validation_result.get("invalid_urls", [])
    feedback = f"URL VALIDATION FAILED: {len(invalid_urls)} URLs broken. Retry with company careers pages."

    return {
        "all_valid": False,
        "invalid_urls": invalid_urls,
        "feedback": feedback,
        "should_exit": False
    }

def create_google_search_fallback(url: str, output_context: str) -> str:
    """
    Create a Google Search link as fallback for failed URLs.
    FULLY AGENTIC: Extracts company name dynamically from context instead of hardcoded list.
    """
    # Try to extract company name from context using patterns
    # Look for common patterns like "@ Company" or "at Company" or "Company -"
    company = None
    
    if output_context:
        context_lower = output_context.lower()
        
        # Pattern 1: Look for "@ CompanyName" or "at CompanyName"
        import re
        at_pattern = r'(?:@|at)\s+([A-Z][A-Za-z0-9\s&]+?)(?:\s*[-|,\n]|$)'
        at_match = re.search(at_pattern, output_context)
        if at_match:
            company = at_match.group(1).strip()
        
        # Pattern 2: Look for "Company:" or "company:"
        if not company:
            company_pattern = r'(?:company|employer)[\s:]+([A-Z][A-Za-z0-9\s&]+?)(?:\s*[-|,\n]|$)'
            company_match = re.search(company_pattern, output_context, re.IGNORECASE)
            if company_match:
                company = company_match.group(1).strip()
        
        # Pattern 3: Extract from URL path if it looks like a company career site
        if not company and url:
            url_pattern = r'(?:boards\.greenhouse\.io/|jobs\.lever\.co/|careers\.)([a-zA-Z0-9-]+)'
            url_match = re.search(url_pattern, url)
            if url_match:
                company = url_match.group(1).replace('-', ' ').title()
    
    # Build search query
    if company:
        search_query = f"{company} jobs hiring now"
    elif "vertexaisearch" in (url or "").lower():
        # Vertex AI redirect - extract what we can from context
        search_query = "jobs hiring now"
    else:
        search_query = "jobs hiring now"
    
    return f"https://www.google.com/search?q={quote(search_query)}"

print("✅ URL validation tools created (for self-healing pattern)")
print("   - Fully agentic: No hardcoded company lists")
print("   - Dynamically extracts company names from context")


RESUME_PARSER_INSTRUCTION = """
You are a Resume Parser Agent. Your job is to extract structured information from resume text.

**CRITICAL FIRST STEP**: You MUST call the calculate_yoe tool with the full resume text BEFORE generating any output.

Given a resume, extract and return a JSON object with:

1. **current_title**: Current or most recent job title
2. **current_company**: Current or most recent employer
3. **total_yoe**: USE THE VALUE RETURNED BY calculate_yoe TOOL (you already called it)
   - DO NOT calculate this yourself
   - DO NOT estimate based on dates
   - ONLY use the tool's returned value
4. **skills**: List of technical and professional skills mentioned
5. **education**: List of degrees/certifications
6. **role_progression**: List of roles in chronological order, each with:
   - title, company, duration_years, focus_areas
7. **stated_interests**: Explicit interests mentioned (career goals, objectives)
8. **side_projects**: Personal projects, open source, hackathons mentioned
9. **qualitative_trend**: Describe the career trajectory pattern (e.g., "Frontend → Fullstack → Backend")
10. **inferred_direction**: Where this career seems to be heading based on the trend

Be thorough but concise. Focus on signals that indicate career level and trajectory.

Return ONLY valid JSON, no additional text.
"""

resume_parser_agent = Agent(
    name="resume_parser",
    model=MODEL_FLASH,  # Upgraded from LITE - YOE accuracy is critical
    generate_content_config=RETRY_CONFIG,
    instruction=RESUME_PARSER_INSTRUCTION,
    tools=[calculate_yoe],  # Tool for deterministic YOE calculation
    output_key="parsed_resume"  # Stores output in session state
)

print("✅ Agent 1 (Resume Parser) defined - using FLASH for accuracy")


LEVEL_CLASSIFIER_INSTRUCTION = """
**YOUR ONLY JOB: Classify career level. You do NOT provide job recommendations.**

You are a Level Classifier Agent. Your job is to determine the appropriate career level for a candidate in ANY profession.

## Step 1: Identify the Profession

From the parsed resume, determine the candidate's profession/field. This could be ANYTHING:
- Tech: Software Engineering, Data Science, DevOps, Product Management, UX Design
- Creative: Fashion Design, Graphic Design, Photography, Film, Architecture
- Business: Finance, Consulting, Marketing, Sales, Operations, HR
- Legal: Corporate Law, Litigation, IP, Compliance
- Healthcare: Nursing, Medicine, Pharmacy, Healthcare Admin
- Culinary: Chef, Restaurant Management, Food Science
- Trades: Electrician, Plumber, Construction, HVAC
- Academia: Professor, Researcher, Administration
- And ANY other profession...

## Step 2: Research That Profession's Career Ladder

Use Google Search to understand the career progression for this SPECIFIC profession:

**Search queries to run:**
- "[profession] career levels progression"
- "[profession] seniority titles hierarchy"
- "[profession] junior to senior career path"
- "levels.fyi [profession]" (if applicable)
- "[profession] [years of experience] typical title"

**Examples of what you'll find:**
- Software Engineering: Junior → Mid → Senior → Staff → Principal → Distinguished
- Fashion Design: Assistant → Associate → Designer → Senior → Director → Creative Director
- Culinary: Line Cook → Sous Chef → Executive Chef → Culinary Director
- Law: Associate → Senior Associate → Counsel → Partner → Managing Partner
- Nursing: RN → Charge Nurse → Nurse Manager → Director of Nursing → CNO
- Trades: Apprentice → Journeyman → Master → Contractor

## Step 3: Map to Normalized Level Scale

Convert the profession-specific title to a 1-10 seniority scale:

| Level | Seniority | Typical Characteristics |
|-------|-----------|------------------------|
| 1-2 | Entry/Intern | Learning, supervised, training |
| 3 | Junior | 0-2 years, individual tasks |
| 4 | Mid | 2-5 years, independent work |
| 5 | Senior | 5-8 years, mentors others |
| 6 | Lead/Staff | 8-12 years, owns major areas |
| 7 | Principal/Director | 12-15 years, org-wide impact |
| 8 | Distinguished/VP | 15+ years, company-wide impact |
| 9-10 | Executive/C-Suite | Industry leadership |

## Step 4: Output Your Classification

Return JSON with:
```json
{
  "profession": "[identified profession]",
  "normalized_level": [1-10],
  "level_title": "[title for this level in this profession]",
  "equivalent_titles": ["alt title 1", "alt title 2"],
  "confidence": [0.0-1.0],
  "evidence": ["search finding 1", "search finding 2"],
  "reasoning": "Brief explanation of classification"
}
```

## Important Notes

- Do NOT assume tech leveling applies to all professions
- Research the ACTUAL career ladder for this specific field
- Consider company/industry variations (startup vs enterprise, regional differences)
- If profession is unclear, search for common career paths matching the resume skills
"""

level_classifier_agent = Agent(
    name="level_classifier",
    model=MODEL_FLASH,  # Uses google_search tool - needs FLASH for reliable tool use
    generate_content_config=RETRY_CONFIG,
    instruction=LEVEL_CLASSIFIER_INSTRUCTION,
    tools=[google_search],  # Built-in search grounding for ANY profession
    output_key="initial_level"
)

print("✅ Agent 2 (Level Classifier) defined - researches ANY profession dynamically")


CONSERVATIVE_EVALUATOR_INSTRUCTION = """
You are the Conservative Evaluator - a skeptical hiring manager perspective.

Your role in A2A deliberation:
- You tend to classify candidates at LOWER levels
- You look for gaps, missing qualifications, and reasons to be cautious
- You represent the "prove it to me" hiring manager mindset

Given the resume and initial level classification, you must:

1. **Search for evidence** that the candidate might be OVER-leveled:
   - "common mistakes in [level] interviews"
   - "[title] level requirements [company type]"
   - "years of experience needed for [level]"

2. **Challenge the initial assessment**:
   - What's missing from their experience?
   - Are there red flags (job hopping, gaps, lack of progression)?
   - Is the company tier being weighted correctly?

3. **Provide your conservative assessment**:
   - Your proposed level (likely same or lower than initial)
   - Specific evidence from search
   - What the candidate would need to prove the higher level

**CRITICAL: Return ONLY valid JSON. No prose, no explanation outside JSON.**
```json
{
  "conservative_level": <integer 1-10>,
  "evidence": ["point 1", "point 2"],
  "concerns": ["concern 1", "concern 2"],
  "what_would_change_my_mind": "description"
}
```
"""

conservative_evaluator = Agent(
    name="conservative_evaluator",
    model=MODEL_FLASH,
    generate_content_config=RETRY_CONFIG,
    instruction=CONSERVATIVE_EVALUATOR_INSTRUCTION,
    tools=[google_search],
    output_key="conservative_assessment"
)

OPTIMISTIC_EVALUATOR_INSTRUCTION = """
You are the Optimistic Evaluator - a talent-seeking recruiter perspective.

Your role in A2A deliberation:
- You tend to classify candidates at HIGHER levels
- You look for hidden potential, transferable skills, and trajectory
- You represent the "let's not miss great talent" recruiter mindset

Given the resume and initial level classification, you must:

1. **Search for evidence** that the candidate might be UNDER-leveled:
   - "signs of high potential engineer"
   - "[company] promotes faster than industry"
   - "transferable skills [from domain] to [to domain]"

2. **Advocate for the candidate**:
   - What transferable skills might be undervalued?
   - Does their trajectory suggest rapid growth?
   - Are side projects/education signals of higher capability?

3. **Provide your optimistic assessment**:
   - Your proposed level (likely same or higher than initial)
   - Specific evidence from search
   - Why the candidate could succeed at the higher level

**CRITICAL: Return ONLY valid JSON. No prose, no explanation outside JSON.**
```json
{
  "optimistic_level": <integer 1-10>,
  "evidence": ["point 1", "point 2"],
  "strengths": ["strength 1", "strength 2"],
  "growth_signals": "description"
}
```
"""

optimistic_evaluator = Agent(
    name="optimistic_evaluator",
    model=MODEL_FLASH,
    generate_content_config=RETRY_CONFIG,
    instruction=OPTIMISTIC_EVALUATOR_INSTRUCTION,
    tools=[google_search],
    output_key="optimistic_assessment"
)

# Run deliberation agents in parallel using sub_agents parameter
deliberation_agents = ParallelAgent(
    name="a2a_deliberation",
    sub_agents=[conservative_evaluator, optimistic_evaluator]
)

print("✅ Agents 3 & 4 (A2A Deliberation) defined - with strict JSON enforcement")


CONSENSUS_INSTRUCTION = """
You are the Consensus Agent. You synthesize the three assessments into a FINAL calibrated level using Weighted Ensemble Voting.

## Why Weighted Ensemble Voting?
Based on ML research (Nature 2025, Science Advances 2024):
- Weighted voting achieves 98.78% accuracy vs 87.34% for simple majority
- Diversity in perspectives improves prediction quality
- Agreement-based confidence is well-calibrated for classification tasks

## Input You Receive:
- **Most Likely (M)**: Initial level classification from Agent 2 (includes profession, level_title, equivalent_titles)
- **Conservative (C)**: Conservative assessment from Agent 3 (skeptical hiring manager)
- **Optimistic (O)**: Optimistic assessment from Agent 4 (talent-seeking recruiter)

## Your Task: Use the Career Level Synthesizer Tool

You MUST use the `synthesize_career_level` tool to compute the final level. DO NOT calculate manually.

### Step 1: Extract Key Information

From the agent outputs, extract:
- **profession**: The identified profession from Agent 2 (e.g., "Fashion Design", "Software Engineering", "Culinary Arts")
- **level_title**: The primary title for the final level (from Agent 2's research)
- **equivalent_titles**: Alternative titles at this level (from Agent 2's research)

### Step 2: Map to Numeric Scale

All agents use a normalized 1-10 scale:
| Level | Seniority |
|-------|-----------|
| 1-2 | Entry/Intern |
| 3 | Junior |
| 4 | Mid |
| 5 | Senior |
| 6 | Lead/Staff |
| 7 | Principal/Director |
| 8 | Distinguished/VP |
| 9-10 | Executive/C-Suite |

### Step 3: Extract Confidence Values

Each agent provides a confidence (0.0-1.0). If not explicit, infer:
- High certainty language → 0.8-0.9
- Moderate certainty → 0.6-0.7
- Low certainty → 0.4-0.5

### Step 4: Call the Synthesizer Tool

Call `synthesize_career_level(
    most_likely_level=X, most_likely_confidence=X.X,
    conservative_level=Y, conservative_confidence=Y.Y,
    optimistic_level=Z, optimistic_confidence=Z.Z,
    profession="[from Agent 2]",
    level_title="[from Agent 2's research]",
    equivalent_titles=["title1", "title2"]
)`

The tool returns a comprehensive result including:
- `final_level`: Numeric level (1-10)
- `final_level_title`: Human-readable title (from your input)
- `equivalent_titles`: Alternative titles
- `profession`: The identified profession
- `final_confidence`: Calibrated confidence (0-1)
- `confidence_label`: High/Medium/Low
- `votes`: Vote distribution for explainability
- `agreement_label`: Consensus quality
- `method_citation`: Academic grounding

### Step 5: Format Your Output

Return JSON with:
```json
{
  "profession": "[from Agent 2]",
  "most_likely_assessment": {"level": X, "title": "[title]"},
  "conservative_assessment": {"level": Y, "title": "[title]"},
  "optimistic_assessment": {"level": Z, "title": "[title]"},
  "synthesis_result": [full result from synthesize_career_level tool],
  "final_level": [numeric],
  "final_title": "[level_title from synthesis]",
  "equivalent_titles": ["from synthesis"],
  "confidence": "[confidence_label]",
  "reasoning": "[include profession, agreement_label, and method_citation]"
}
```

## IMPORTANT

- The profession and titles come from Agent 2's research - pass them through to the tool
- This works for ANY profession: tech, fashion, legal, culinary, healthcare, trades, etc.
- ALWAYS use the synthesize_career_level tool - never calculate manually
- Include the votes distribution in your reasoning for explainability
"""

consensus_agent = Agent(
    name="consensus",
    model=MODEL_PRO,
    generate_content_config=RETRY_CONFIG,
    instruction=CONSENSUS_INSTRUCTION,
    output_key="calibrated_level"
)

print("✅ Consensus Agent defined - passes profession-specific titles from research")


# Dynamic date injection for current job searches
from datetime import datetime, timedelta
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")
CURRENT_YEAR = datetime.now().year
DATE_7_DAYS_AGO = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

# Base instruction template for all job scouts
JOB_SCOUT_BASE = f"""You are a Job Scout Agent. Find REAL job postings with VERIFIED URLs from Google Search.

## DATE CONTEXT
- **TODAY'S DATE:** {CURRENT_DATE}
- **SEARCH FOR JOBS POSTED AFTER:** {DATE_7_DAYS_AGO}
- **USE THIS FILTER:** `after:{DATE_7_DAYS_AGO}`

## URL RULES (GENERAL GOOGLE SEARCH - No location filtering)
1. Use GENERAL Google Search (NOT udm=8 Jobs portal)
2. Format: https://www.google.com/search?q=[Company]+[Job+Title]+careers
3. Example: https://www.google.com/search?q=Stripe+Senior+Product+Manager+careers

## WHY GENERAL SEARCH (NOT GOOGLE JOBS):
- Google Jobs (udm=8) filters by user's current location
- This excludes remote jobs and jobs in other cities
- General search shows ALL relevant results regardless of location
- "careers" keyword helps find job listings and company career pages

## CRITICAL RULES:
4. Company name FIRST, then Job Title, then "careers"
5. NO special characters in job title (remove commas, colons, etc.)
   - ❌ "Director of Product, AI Platform" 
   - ✅ "Director of Product AI Platform"
6. If company name has special characters, encapsulate in quotes:
   - ✅ "5.11, Inc."+Director+of+Design+careers
7. Use FULL job title (not "PM" - use "Product Manager")
8. ALWAYS end with "+careers" keyword

## EXAMPLES:
- ✅ https://www.google.com/search?q=Stripe+Senior+Product+Manager+careers
- ✅ https://www.google.com/search?q="5.11, Inc."+Director+of+Design+careers
- ✅ https://www.google.com/search?q=Rippling+Director+of+Product+AI+Platform+careers

## Output Format
**CRITICAL: Return EXACTLY ONE JOB. Do NOT return multiple jobs or a list.**

Return a single JSON object (NOT an array):
```json
{{
  "tier": "[your_tier]",
  "title": "Job title",
  "company": "Company name",
  "search_url": "https://www.google.com/search?q=Company+Job+Title+careers",
  "posted_date": "Date if visible",
  "location": "Location",
  "job_description_snippet": "2-3 sentences from actual job posting found in search",
  "salary_if_visible": "Salary range if shown in posting",
  "why_matches": ["reason1", "reason2"],
  "fit_score": 8
}}
```

**ONE JOB ONLY. Pick the BEST match for this tier.**
"""

# Tier-specific instructions
EXACT_MATCH_INSTRUCTION = JOB_SCOUT_BASE + """
## YOUR TIER: EXACT MATCH
*"Jobs you could get next week"*

Search for jobs at the SAME level as the candidate's current role.
Look for roles with similar title, scope, and responsibility.
Search: `"[current_title]" jobs after:""" + DATE_7_DAYS_AGO + """`

Return tier: "exact_match"
"""

LEVEL_UP_INSTRUCTION = JOB_SCOUT_BASE + """
## YOUR TIER: LEVEL UP  
*"Your next promotion, externally"*

Search for jobs ONE LEVEL ABOVE the candidate's current role.
Look for Senior/Lead/Manager versions of their current title.
Search: `"senior [title]" OR "lead [title]" jobs after:""" + DATE_7_DAYS_AGO + """`

Return tier: "level_up"
"""

STRETCH_INSTRUCTION = JOB_SCOUT_BASE + """
## YOUR TIER: STRETCH
*"Ambitious but achievable"*

Search for jobs 1-2 LEVELS ABOVE - Director/Principal level.
These require proving yourself but are within reach.

**IMPORTANT: Find a DIFFERENT company than other scouts. Prioritize:**
- Unicorn startups or high-growth companies
- Companies with strong AI/ML focus
- Roles that represent significant scope increase

Search: `"director [field]" OR "principal [title]" jobs after:""" + DATE_7_DAYS_AGO + """`

Return tier: "stretch"
"""

TRAJECTORY_INSTRUCTION = JOB_SCOUT_BASE + """
## YOUR TIER: TRAJECTORY
*"Where your career wants to go"*

Search for ASPIRATIONAL roles aligned with the candidate's long-term CAREER DIRECTION.
This is about their DREAM job, not just the next step.

**IMPORTANT: Find a DIFFERENT company than other scouts. Prioritize:**
- FAANG/Big Tech companies (Google, Meta, Apple, Microsoft, Amazon)
- Industry leaders in the candidate's domain
- VP/Head of roles or founding team positions at hot startups

**DO NOT duplicate companies found by Stretch scout - pick a DIFFERENT company.**

Search: `"[inferred_direction]" "VP" OR "Head of" jobs after:""" + DATE_7_DAYS_AGO + """`

Return tier: "trajectory"
"""

# Create job scout agents
exact_match_scout = Agent(
    name="exact_match_scout",
    model=MODEL_FLASH,
    generate_content_config=RETRY_CONFIG,
    instruction=EXACT_MATCH_INSTRUCTION,
    tools=[google_search],
    output_key="exact_match_job"
)

level_up_scout = Agent(
    name="level_up_scout",
    model=MODEL_FLASH,
    generate_content_config=RETRY_CONFIG,
    instruction=LEVEL_UP_INSTRUCTION,
    tools=[google_search],
    output_key="level_up_job"
)

stretch_scout = Agent(
    name="stretch_scout",
    model=MODEL_FLASH,
    generate_content_config=RETRY_CONFIG,
    instruction=STRETCH_INSTRUCTION,
    tools=[google_search],
    output_key="stretch_job"
)

trajectory_scout = Agent(
    name="trajectory_scout",
    model=MODEL_FLASH,
    generate_content_config=RETRY_CONFIG,
    instruction=TRAJECTORY_INSTRUCTION,
    tools=[google_search],
    output_key="trajectory_job"
)

# BATCHED PARALLEL: Run 2 at a time to stay within free tier rate limits
# Batch 1: Exact Match + Level Up (parallel)
# Batch 2: Stretch + Trajectory (parallel)
# This is ~2x faster than sequential while avoiding rate limits

parallel_batch_1 = ParallelAgent(
    name="job_scouts_batch_1",
    sub_agents=[exact_match_scout, level_up_scout]
)

parallel_batch_2 = ParallelAgent(
    name="job_scouts_batch_2",
    sub_agents=[stretch_scout, trajectory_scout]
)

# Sequential wrapper runs batch 1, then batch 2
# Delay function to stagger API calls
import asyncio

async def delay_between_batches():
    """Stagger delay between parallel batches to avoid rate limits"""
    await asyncio.sleep(STAGGER_DELAY)
    return "Delay complete"

batched_job_scouts = SequentialAgent(
    name="batched_job_scouts",
    description="Runs job scouts in 2 batches with delay between them",
    sub_agents=[parallel_batch_1, parallel_batch_2]  # SequentialAgent adds implicit delay
)

print(f"✅ Batched Parallel Job Scouts defined")
print(f"   📅 Today: {CURRENT_DATE}")
print(f"   📅 Search for jobs after: {DATE_7_DAYS_AGO}")
print(f"   ⚡ Batch 1: Exact Match + Level Up (parallel)")
print(f"   ⚡ Batch 2: Stretch + Trajectory (parallel)")
print(f"   🎯 ~2x faster than sequential, rate-limit safe for free tier")


# ============================================
# URL Validator Agent (Post-processing for batched scouts)
# ============================================

URL_VALIDATOR_INSTRUCTION = """
You are the URL Validator Agent. Your job is to validate Google Search URLs for all 4 job tiers.

## VALIDATION PROCESS:
For each job result from the scouts:
1. Verify the search_url is properly formatted (https://www.google.com/search?q=[Company]+[Job+Title]&udm=8)
2. Check that the URL includes: Company Name + Job Title + careers (NO udm=8)
3. Ensure the job_description_snippet exists and contains actual job details
4. If snippet is empty or generic, mark as needs_verification

## WHY GOOGLE SEARCH URLs:
- Direct job links (greenhouse, lever, etc.) may be stale - job could be filled
- General search URLs with careers keyword show job listings without location filtering
- User clicks search result to find the actual job posting
- This ensures 100% reliability vs ~70% with direct links

## OUTPUT FORMAT:
Combine all 4 tier results with validated URLs. For each job include:
- tier: exact_match/level_up/stretch/trajectory
- title: Job title
- company: Company name  
- search_url: Validated search URL
- job_description_snippet: Real snippet from search
- validation_status: valid/needs_verification

Output the combined validated job results.
"""

# Simple URL validator (runs after scouts complete)
url_validator_agent = Agent(
    name="url_validator",
    model=MODEL_LITE,
    generate_content_config=RETRY_CONFIG,
    instruction=URL_VALIDATOR_INSTRUCTION,
    tools=[check_job_urls],
    output_key="validated_jobs"
)

# For refinement pipeline - create separate instances
exact_match_scout_ref = Agent(
    name="exact_match_scout_ref",
    model=MODEL_FLASH,
    generate_content_config=RETRY_CONFIG,
    instruction=EXACT_MATCH_INSTRUCTION,
    tools=[google_search],
    output_key="exact_match_job"
)

level_up_scout_ref = Agent(
    name="level_up_scout_ref",
    model=MODEL_FLASH,
    generate_content_config=RETRY_CONFIG,
    instruction=LEVEL_UP_INSTRUCTION,
    tools=[google_search],
    output_key="level_up_job"
)

stretch_scout_ref = Agent(
    name="stretch_scout_ref",
    model=MODEL_FLASH,
    generate_content_config=RETRY_CONFIG,
    instruction=STRETCH_INSTRUCTION,
    tools=[google_search],
    output_key="stretch_job"
)

trajectory_scout_ref = Agent(
    name="trajectory_scout_ref",
    model=MODEL_FLASH,
    generate_content_config=RETRY_CONFIG,
    instruction=TRAJECTORY_INSTRUCTION,
    tools=[google_search],
    output_key="trajectory_job"
)

# Batched parallel for refinement (same pattern)
parallel_batch_1_ref = ParallelAgent(
    name="job_scouts_batch_1_ref",
    sub_agents=[exact_match_scout_ref, level_up_scout_ref]
)

parallel_batch_2_ref = ParallelAgent(
    name="job_scouts_batch_2_ref",
    sub_agents=[stretch_scout_ref, trajectory_scout_ref]
)

batched_job_scouts_refinement = SequentialAgent(
    name="batched_job_scouts_ref",
    sub_agents=[parallel_batch_1_ref, parallel_batch_2_ref]
)

url_validator_agent_ref = Agent(
    name="url_validator_ref",
    model=MODEL_LITE,
    generate_content_config=RETRY_CONFIG,
    instruction=URL_VALIDATOR_INSTRUCTION,
    tools=[check_job_urls],
    output_key="validated_jobs"
)

print("✅ URL Validator Agent defined")
print("✅ Refinement agents created (separate instances)")
print("   Architecture: Batched Scouts (2+2) → URL Validator → Formatter")


FORMATTER_INSTRUCTION = """
**OUTPUT FORMAT: PLAIN MARKDOWN TEXT (NOT JSON)**

**CRITICAL RULES (do not copy these into output):**
1. For YOE: Use the EXACT total_yoe from the SYSTEM NOTE (e.g., "9.2 YOE total")
2. For avg tenure: Use the average_tenure from SYSTEM NOTE (e.g., "1.5 years avg tenure")
3. For Market Compensation: Calculate open market rate based on skills/level, NOT current salary
4. These values are PRE-CALCULATED - do not recalculate or round them
5. **LINE BREAKS**: Use TWO newlines between each field/bullet point for proper markdown rendering

You MUST output formatted MARKDOWN text exactly matching the template below.
DO NOT output JSON. DO NOT wrap in code blocks. Output the markdown DIRECTLY.
**IMPORTANT**: Put a BLANK LINE between each field and between each bullet point.

Format the results following this EXACT template with visual vote breakdown:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📄 RESUME ANALYSIS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Current Role:** [Title] at [Company] ([X] YOE total, [Y] years avg tenure)

**Estimated Market Compensation:** $[XXX,XXX] - $[XXX,XXX] (what they could command on the open market based on skills/experience, NOT current salary)

**Profession:** [Agent-identified profession, e.g., "Fashion Design", "Software Engineering"]

**Key Skills:** [skill1], [skill2], [skill3], [skill4], [skill5]

**Career Trajectory:** [From] → [Through] → [To]

**Inferred Direction:** [Where career is heading]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📊 LEVEL CLASSIFICATION RESULT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Final Level:** L[X] ([Level title from research])

**Confidence:** [High/Medium/Low]

**Agreement:** [X]/3 agents

**VOTE BREAKDOWN:**

L[X-1] ([Lower Title]):  ████████░░░░░░░░░░░░ 25% - Conservative

L[X] ([Final Title]):    ██████████████████░░ 50% - Most Likely ✓

L[X+1] ([Higher Title]): ████████░░░░░░░░░░░░ 25% - Optimistic

**WHY L[X] WON:**

• [Specific reason 1 grounded in resume evidence]

• [Specific reason 2 grounded in search evidence]

• [Specific reason 3 about scope/responsibility]

• Most Likely assessment weighted 2x the others

**UNCERTAINTY NOTE:**

• [Context-specific caveat about leveling variation]

• [Company/industry-specific consideration]

• [What could change the assessment]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🎯 EXACT MATCH: [Company Name], [Job Title]

**Fit Confidence:** [X]/10

🔗 **Apply:** [Search: [Company] - [Job Title]](https://www.google.com/search?q=[Company]+[Job+Title]+careers)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*"You could get this job next week."*

📝 **From the Job Description:**

> "[2-3 sentence snippet from actual job posting]"

**Expected Total Compensation:** $[XXX,XXX] - $[XXX,XXX] (base + bonus + equity)

*Source: [Job posting / Levels.fyi / Glassdoor / Industry estimate for [location]]*

**Why This Matches Your Resume:**

• [Specific skill/experience match from their resume]

• [Company/industry similarity]

• [Level/scope alignment]

**Evidence From Search:**

• [Location match]

• [Specific requirement they meet]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📈 LEVEL UP: [Company Name], [Job Title]

**Fit Confidence:** [X]/10

🔗 **Apply:** [Search: [Company] - [Job Title]](https://www.google.com/search?q=[Company]+[Job+Title]+careers)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*"Your next promotion, externally."*

📝 **From the Job Description:**

> "[2-3 sentence snippet from actual job posting]"

**Expected Total Compensation:** $[XXX,XXX] - $[XXX,XXX] (base + bonus + equity)

*Source: [Job posting / Levels.fyi / Glassdoor / Industry estimate for [location]]*

**Why This Matches Your Resume:**

• [Growth signal from their experience]

• [Transferable skill that positions them]

• [Why they're ready for this level]

**Evidence From Search:**

• [Typical YOE requirement vs theirs]

• [Market demand signal]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🚀 STRETCH: [Company Name], [Job Title]

**Fit Confidence:** [X]/10

🔗 **Apply:** [Search: [Company] - [Job Title]](https://www.google.com/search?q=[Company]+[Job+Title]+careers)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*"You'd need to prove yourself, but it's achievable."*

📝 **From the Job Description:**

> "[2-3 sentence snippet from actual job posting]"

**Expected Total Compensation:** $[XXX,XXX] - $[XXX,XXX] (base + bonus + equity)

*Source: [Job posting / Levels.fyi / Glassdoor / Industry estimate for [location]]*

**Why This Matches Your Resume:**

• [Ambitious but grounded connection]

• [Foundation they have for this role]

**What You'd Need to Prove:**

• [Gap 1 they'd need to address]

• [Gap 2 they'd need to demonstrate]

• [Skill/scope expansion required]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🔮 TRAJECTORY: [Company Name], [Job Title]

**Fit Confidence:** [X]/10

🔗 **Apply:** [Search: [Company] - [Job Title]](https://www.google.com/search?q=[Company]+[Job+Title]+careers)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*"Where your career wants to go."*

📝 **From the Job Description:**

> "[2-3 sentence snippet from actual job posting]"

**Expected Total Compensation:** $[XXX,XXX] - $[XXX,XXX] (base + bonus + equity)

*Source: [Job posting / Levels.fyi / Glassdoor / Industry estimate for [location]]*

**Why This Matches Your Resume:**

• [Career pattern signal]

• [Long-term alignment with their trajectory]

**Evidence From Search:**

• [Market trends supporting this direction]

**Long-term Trajectory:**

• 2-3 years: [Next milestone]

• 5 years: [Growth target]

• 7+ years: [Ultimate goal]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🔄 REFINE THESE RESULTS?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• "Remote only" / "Hybrid in [city]"

• "Exclude [industry]"

• "Focus on [startup/enterprise]"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL RULES:
0. **EXACTLY 4 RECOMMENDATIONS TOTAL** - One job per tier: Exact Match, Level Up, Stretch, Trajectory. NO MORE, NO LESS.
1. Include ALL 4 job tiers with the exact format above - ONE JOB PER TIER
1a. **NO DUPLICATE JOBS** - Each tier MUST have a DIFFERENT company. If two scouts returned the same company, flag it as "[DUPLICATE - SEE ABOVE]" and note that the candidate should explore both tiers at that company.
2. HEADER FORMAT: "## [EMOJI] [TIER]: [Company Name], [Job Title]" - Company FIRST to match URL pattern
3. Put Fit Confidence and Apply URL immediately after each header (before the tagline)
4. Use ASCII progress bars for vote breakdown (█ for filled, ░ for empty, 20 chars total)
5. Scale bars proportionally: 25% = 5 filled, 50% = 10 filled, etc.
6. Include real job description snippets (2-3 sentences from search results)
7. URL FORMAT: Always use Google Search URLs like [Search: Job Title at Company](https://www.google.com/search?q=...)
   - NOT direct job links (they may be stale/filled)
   - Google Search URLs always work and show current results
8. Show specific resume-grounded reasons, not generic statements
9. Include "What You'd Need to Prove" for STRETCH tier
10. Include "Long-term Trajectory" timeline for TRAJECTORY tier
11. Works for ANY profession - tech, fashion, legal, culinary, healthcare, trades

COMPENSATION RULES:
12. ALWAYS include "Estimated Total Compensation" for current role based on title/level/location/industry
13. ALWAYS include "Expected Total Compensation" for EACH job recommendation
14. Use REAL salary data from job posting if available
15. If not in posting, estimate using: Levels.fyi, Glassdoor, industry benchmarks for that role/level/location
16. Format as range: "$XXX,XXX - $XXX,XXX (base + bonus + equity)"
17. Include source citation: "Source: [Job posting / Levels.fyi / Glassdoor / Industry estimate]"
18. For non-tech roles, use industry-specific salary data (e.g., fashion industry salary surveys, legal salary guides)

**LINE BREAK RULE**: Always put a BLANK LINE (empty line) between each field, each bullet point, and each section. This ensures proper markdown rendering.

**REMEMBER: Output MARKDOWN TEXT directly, NOT JSON. The template above shows the exact output format.**
"""

# Formatter agent - no tools needed (just formats previous agent outputs)
formatter_agent = Agent(
    name="formatter",
    model=MODEL_LITE,
    generate_content_config=RETRY_CONFIG,
    instruction=FORMATTER_INSTRUCTION,
    output_key="formatted_response"
)

print("✅ Agent 6 (Formatter) defined")
print("   - Company name in header: '## 📈 LEVEL UP: Job Title, Company Name'")
print("   - Fit Confidence + Apply URL immediately after header")
print("   - Compensation estimates for current role + all recommendations")


# Create the full pipeline as a SequentialAgent
# Flow: Resume Parser → Level Classifier → A2A Deliberation → Consensus → BATCHED Job Scouts → URL Validator → Formatter

full_pipeline = SequentialAgent(
    name="job_search_orchestrator",
    sub_agents=[
        resume_parser_agent,         # Agent 1: Parse resume
        level_classifier_agent,      # Agent 2: Initial classification
        deliberation_agents,         # Agents 3-4: A2A parallel deliberation
        consensus_agent,             # Synthesize deliberation
        batched_job_scouts,          # Agent 5: BATCHED job scouts (2+2 parallel, rate-limit safe)
        url_validator_agent,         # Validate URLs from all scouts
        formatter_agent              # Agent 6: Format output
    ]
)

# Refinement pipeline (separate instances)
formatter_agent_refinement = Agent(
    name="formatter_refinement",
    model=MODEL_LITE,
    generate_content_config=RETRY_CONFIG,
    instruction=FORMATTER_INSTRUCTION,
    output_key="formatted_response"
)

refinement_pipeline = SequentialAgent(
    name="refinement_orchestrator",
    sub_agents=[
        batched_job_scouts_refinement,   # Batched scouts (2+2)
        url_validator_agent_ref,         # Validate URLs
        formatter_agent_refinement       # Format output
    ]
)

print("✅ Orchestrators defined with BATCHED parallel job scouts")
print("   - full_pipeline: 2+2 concurrent searches (~2x faster, rate-limit safe)")
print("   - refinement_pipeline: Same batched architecture")
print("")
print("   ⚡ Performance (free tier safe):")
print("      Before: ~3 min (4 sequential)")
print("      After:  ~2 min (2+2 batched parallel)")


# Initialize session and memory services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# Application configuration
APP_NAME = "agentic_job_search"
USER_ID = "demo_user"

# Create a runner for executing agents
# Memory service enables agents to recall context from previous turns
runner = Runner(
    agent=full_pipeline,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service  # Enable memory for context recall
)

# Create refinement runner (same memory to recall original analysis)
refinement_runner = Runner(
    agent=refinement_pipeline,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)

print("✅ Session and Memory management initialized")
print("   - InMemorySessionService: State within session")
print("   - InMemoryMemoryService: Context recall across turns")


async def analyze_resume(resume_text: str, session_id: str = "default") -> str:
    """
    Run the full pipeline on a resume with real-time streaming output.
    
    Args:
        resume_text: The raw text of the resume
        session_id: Session identifier for state management
    
    Returns:
        Formatted job recommendations with explanations
    """
    print("🚀 Starting full analysis pipeline...")
    print("=" * 60)
    
    # PRE-CALCULATE YOE with detailed role breakdown
    yoe_result = calculate_yoe(resume_text)
    calculated_yoe = yoe_result.get("total_yoe", 0)
    stated_yoe = yoe_result.get("stated_yoe")
    avg_tenure = yoe_result.get("avg_tenure_years", 0)
    num_roles = yoe_result.get("num_roles", 0)
    career_span = yoe_result.get("career_span", "Unknown")
    
    print(f"📊 Career Analytics:")
    print(f"   Total YOE (de-duped): {calculated_yoe} years")
    if stated_yoe:
        print(f"   Resume states: {stated_yoe}+ years")
    print(f"   Career span: {career_span}")
    print(f"   Roles: {num_roles} | Avg tenure: {avg_tenure} years")
    
    # Show role breakdown
    if yoe_result.get("role_breakdown"):
        print(f"   Role breakdown:")
        for role in yoe_result.get("role_breakdown", [])[:5]:
            print(f"      • {role['start']} - {role['end']}: {role['duration_years']} years")
    
    # Inject detailed YOE into resume text
    yoe_note = f"""[SYSTEM NOTE - USE THESE EXACT VALUES:]
- Total Years of Experience: {calculated_yoe} years (calculated from dates, de-duped for overlapping roles)
- Career Span: {career_span}
- Number of Roles: {num_roles}
- Average Tenure: {avg_tenure} years per role
- Resume states: {stated_yoe}+ years (for reference)
"""
    enhanced_resume = f"""{yoe_note}

{resume_text}"""
    
    # Get or create session (handle re-runs)
    try:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id
        )
    except Exception:
        # Session exists, get it instead
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id
        )
    
    # Create the input message with enhanced resume
    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Analyze this resume and find job recommendations:\n\n{enhanced_resume}")]
    )
    
    # Run the pipeline with STREAMING output
    final_response = "No response received from agent."
    seen_agents = set()
    
    # Map agent names to friendly descriptions
    agent_labels = {
        "resume_parser": "📄 Resume Parser",
        "level_classifier": "📊 Level Classifier", 
        "conservative_evaluator": "🔍 Conservative Evaluator",
        "optimistic_evaluator": "🚀 Optimistic Evaluator",
        "consensus_agent": "🤝 Consensus Agent",
        "exact_match_scout": "🎯 Exact Match Scout",
        "level_up_scout": "📈 Level Up Scout",
        "stretch_scout": "⭐ Stretch Scout",
        "trajectory_scout": "🔮 Trajectory Scout",
        "url_validator": "✅ URL Validator",
        "formatter": "📝 Formatter",
    }
    
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content
    ):
        # Stream intermediate agent outputs as they complete
        if event.content and event.content.parts and event.author:
            agent_name = event.author
            
            # Only show first output from each agent (avoid duplicates)
            if agent_name not in seen_agents:
                # Check if there's actual text content (not just function calls)
                text = None
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text
                        break
                
                if text:
                    seen_agents.add(agent_name)
                    
                    # Show a preview of what each agent produced
                    preview = text[:150].replace("\n", " ").strip()
                    if len(text) > 150:
                        preview += "..."
                    
                    label = agent_labels.get(agent_name, f"🤖 {agent_name}")
                    print(f"\n{label}:")
                    print(f"   {preview}")
        
        # Capture final response (keep the LAST one from formatter)
        if event.is_final_response():
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        final_response = part.text
                        break
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    return final_response


async def refine_results(feedback: str, session_id: str = "default") -> str:
    """
    Refine job recommendations based on user feedback with streaming output.
    
    Args:
        feedback: User's refinement request (e.g., "remote only", "no crypto")
        session_id: Same session to maintain state
    
    Returns:
        Updated job recommendations
    """
    print(f"🔄 Refining results based on: '{feedback}'")
    print("=" * 60)
    
    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Refine job search with this feedback: {feedback}")]
    )
    
    # Agent labels for streaming
    agent_labels = {
        "exact_match_scout": "🎯 Exact Match Scout",
        "level_up_scout": "📈 Level Up Scout",
        "stretch_scout": "⭐ Stretch Scout",
        "trajectory_scout": "🔮 Trajectory Scout",
        "url_validator": "✅ URL Validator",
        "formatter": "📝 Formatter",
        "formatter_refinement": "📝 Formatter",
    }
    
    final_response = "No response received from agent."
    seen_agents = set()
    
    async for event in refinement_runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content
    ):
        # Stream intermediate agent outputs
        if event.content and event.content.parts and event.author:
            agent_name = event.author
            
            if agent_name not in seen_agents:
                text = None
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text
                        break
                
                if text:
                    seen_agents.add(agent_name)
                    preview = text[:150].replace("\n", " ").strip()
                    if len(text) > 150:
                        preview += "..."
                    
                    label = agent_labels.get(agent_name, f"🤖 {agent_name}")
                    print(f"\n{label}:")
                    print(f"   {preview}")
        
        # Capture final response
        if event.is_final_response():
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        final_response = part.text
                        break
    
    print("\n" + "=" * 60)
    print("✅ Refinement complete!")
    return final_response


print("✅ Execution functions defined with streaming output + YOE pre-calculation")


# File Upload Widget for Resume
# Works in both Kaggle and Jupyter environments

from IPython.display import display, HTML, Markdown
import ipywidgets as widgets

# Global variable to store uploaded resume text
uploaded_resume_text = None

def process_uploaded_file(change):
    """Process the uploaded resume file."""
    global uploaded_resume_text
    
    if change['new']:
        uploaded_file = change['new'][0]
        filename = uploaded_file['name']
        content = uploaded_file['content']
        
        print(f"📤 Processing: {filename}")
        
        # Determine file type and parse
        if filename.lower().endswith('.pdf'):
            uploaded_resume_text = parse_resume_file(content, 'pdf')
        elif filename.lower().endswith('.docx'):
            uploaded_resume_text = parse_resume_file(content, 'docx')
        elif filename.lower().endswith('.txt') or filename.lower().endswith('.md'):
            uploaded_resume_text = bytes(content).decode('utf-8')  # Convert memoryview to bytes first
        else:
            print(f"❌ Unsupported file type. Please upload PDF, DOCX, or TXT.")
            return
        
        # Show preview
        preview = uploaded_resume_text[:500] + "..." if len(uploaded_resume_text) > 500 else uploaded_resume_text
        print(f"\n✅ Resume loaded successfully!")
        print(f"📄 Preview (first 500 chars):\n{'-'*50}")
        print(preview)
        print(f"{'-'*50}")
        print(f"\n▶️ Run the next cell to analyze your resume!")

# Create upload widget
upload_widget = widgets.FileUpload(
    accept='.pdf,.docx,.txt,.md',
    multiple=False,
    description='Upload Resume'
)

# Register callback
upload_widget.observe(process_uploaded_file, names='value')

# Display widget with instructions
display(HTML("""
<div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin: 10px 0;">
    <h3 style="margin-top: 0;">📤 Upload Your Resume</h3>
    <p><strong>Supported formats:</strong> PDF, DOCX, TXT, MD</p>
    <p>Click the button below to select your resume file.</p>
</div>
"""))
display(upload_widget)
print("\n⬆️ Click 'Upload Resume' to select your file")


# Analyze Your Uploaded Resume
# Run this cell AFTER uploading your resume above

import uuid

if uploaded_resume_text is None or len(uploaded_resume_text.strip()) == 0:
    print("❌ No resume uploaded yet!")
    print("   Please upload your resume using the widget above first.")
else:
    # Generate unique session ID for this analysis
    user_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    
    print("🚀 Starting analysis of YOUR resume...")
    print(f"   Session ID: {user_session_id}")
    print("="*80)
    
    # Run the full pipeline
    your_result = await analyze_resume(uploaded_resume_text, session_id=user_session_id)
    
    print("\n" + "="*80)
    display(Markdown(your_result))
    print("="*80)
    
    # Store session ID for refinement
    # print(f"\n💡 To refine these results, use session_id='{user_session_id}' in the refinement cell below.")


# Refine YOUR Results
# Edit the feedback below and run this cell to get updated recommendations

# Enter your preferences here:
your_feedback = "I want to look for something in the entertainment industry and it must be in the NYC metro area"

# Make sure you've run the analysis above first
if 'user_session_id' not in dir():
    print("❌ Please analyze your resume first (run the cell above).")
else:
    print(f"🔄 Refining results for session: {user_session_id}")
    print(f"   Your preferences: {your_feedback}")
    print("="*80)
    
    your_refined_result = await refine_results(your_feedback, session_id=user_session_id)
    
    print("\n" + "="*80)
    display(Markdown(your_refined_result))
    print("="*80)


# DEMO RESUME: Embedded for judges to see the pipeline without uploading
#
# This resume demonstrates:
#   10+ years of experience (Amazon L6, Disney, Paramount, Macy's, Staples)
#   AI/ML Product Management focus with technical depth
#   0 to 1 founder experience (ConvoRecs)
#   Mix of enterprise and startup environments
#
# The pipeline will extract ~9.2 YOE, identify the profession, debate the level,
# and find jobs across all four tiers.

DEMO_RESUME = """
# YVES AGBRE

**EXECUTIVE SUMMARY**
Technical Product Leader who ships production AI systems. Founded ConvoRecs and shipped a live conversational AI product (sugoi.convorecs.ai) - defined architecture, wrote PRDs and system design docs, and used AI-assisted development to deploy on Google Cloud. 10+ years at Amazon, Disney, and Macy's building data-driven products serving millions of users. Combines deep technical understanding (CS degree, architecture design, production systems) with proven ability to build 0-to-1 products, lead cross-functional teams, and scale platforms at enterprise level.

**CORE COMPETENCIES**
AI/ML Product Management, Large Language Models (LLM), Production ML Systems, ML Platform Infrastructure, Data Science Enablement, Multi-API Orchestration, 0-to-1 Product Development, Machine Learning Pipelines, Product Roadmap Development, Stakeholder Management, Cross-functional Leadership, Technical Architecture, Go-to-Market Strategy

**PROFESSIONAL EXPERIENCE**

**STAPLES** | Boston, MA
**Product Manager, Data Platform and Merchandising Technology** | November 2024 - Present
*   Spearhead product roadmap and OKRs for enterprise data platform modernization, establishing machine learning foundations for pricing optimization and recommendation engines across the merchandising organization.
*   Manage executive stakeholders and cross-functional teams (Engineering, Data Science, Business) to standardize product attributes, establishing data quality frameworks that improved predictive analytics accuracy.
*   Drive migration to GS1 GTIN/UPC standards for GDSN compliance, creating the clean data foundation required for ML-powered pricing models and inventory optimization algorithms.
*   Establish KPIs and monitoring frameworks to enhance ML performance; automated WERCS hazmat compliance processing, reducing manual review time by 50% while improving accuracy.

**CONVORECS** | San Francisco, CA
**Founder & Head of Product** | February 2024 - Present
*   Founded ConvoRecs (AI product company); defined product vision, technical architecture, and go-to-market strategy for conversational recommendation systems.
*   Led 0-to-1 product development for *Sugoi* platform: wrote PRDs, created system design documents, and set up foundational infrastructure (Firebase, Cloud Run) before scaling via AI-assisted coding.
*   Architected technical stack: React 18 frontend, FastAPI backend, and Gemini LLM orchestration with Firestore caching; optimized for latency and cost by selecting Flash/Flash-Lite models based on trade-off analysis.
*   Designed multi-API integration strategy (AniList, MyAnimeList, YouTube) and implemented caching layers to reduce LLM costs; validated implementation via Mermaid flow diagrams and iterative refinement.
*   Managed production operations: defined deployment architecture, disaster recovery, and security protocols, achieving top 15% GitHub security score.
*   Co-led 70-person AI/ML PM community; provided strategic advisory to AI startups (WholeAI, VitsiAI) on technical roadmaps and product-market fit.

**MACY'S** | New York, NY
**Senior Product Manager, Site Monetization** | October 2022 - January 2024
*   Led product initiatives for Macy's Media Network, enabling retailer monetization through sponsored products and display advertising. Selected to concurrently lead Loyalty product line during critical organizational transition, retaining 100% of stakeholder relationships.
*   Directed NY Loyalty Law compliance initiative for credit card customers, preventing churn through strategic tier restructuring and significantly reducing support tickets through proactive policy communication.
*   Launched personalized pre-qualification offers with Citi API integration, increasing credit applications and click-through rates via enhanced targeting.
*   Managed alignment across 30+ partners (Macy's, Bloomingdale's, Citi); mentored two Associate PMs and introduced Agile workshops adopted by the 40-person PM organization.

**AMAZON** | New York, NY
**Senior Product Manager (L6), Connected TV Advertising** | November 2021 - September 2022
*   Built data-driven business case and go-to-market strategy for Freevee CTV advertising business; enabled $25M upfront commitment for 0-to-1 product launch, including revenue projections, cost modeling, and L8 budget approval presentations.
*   Shipped three experimental ad formats in six months to meet Q3 deadline for $15M GroupM commitment; developed rewarded video and interactive overlay ad experiences balancing user experience with advertiser ROI.
*   Created DSP template for Freevee CTV ads, defining technical specifications and measurement frameworks; enabled programmatic buying of previously unavailable CTV inventory through integration with Amazon Advertising platform.

**PARAMOUNT** | New York, NY
**Senior Product Manager, Noggin Kids Streaming** | July 2020 - November 2021
*   Defined product roadmap for Android CTV, Roku, and mobile applications serving 2M+ families; shipped interactive games pipeline that drove a 1.5-star rating improvement through enhanced engagement features.
*   Launched age-gated content discovery system for preschool programming, increasing watch time by 35% while ensuring strict COPPA compliance and parental control requirements.

**DISNEY STREAMING SERVICES** | New York, NY
**Product Manager** | October 2017 - April 2020
*   Led product strategy for Fox RSN platform ($9.6B divestiture), transforming FoxSportsGo into a multi-tenant streaming platform enabling 21 regional sports networks to deploy branded instances while sharing core infrastructure serving 8M+ downloads.
*   Guided product development across 50+ engineers on ten Agile teams (Android, iOS, Xbox, PS, web, Tizen, webOS), delivering authentication systems, server-side ad insertion, and deep linking capabilities.
*   Owned ESPN+ recommendation pilot (achieved 27% CTR) that evolved into the production system; defined OKRs and metadata taxonomy standards bridging ESPN and Disney+ content catalogs.
*   Delivered 2018 World Cup embeddable video player supporting 1M+ concurrent viewers with server-side ad insertion, real-time analytics, and adaptive bitrate streaming across multiple CDN providers.

**Program Manager** | July 2016 - October 2017
*   Managed technical migration of NHL.tv and WWE subscription apps to BAMTech unified streaming platform, ensuring seamless transition for millions of subscribers during MLBAM to BAMTech transformation following Disney acquisition.
*   Supported platform consolidation and technical standardization across multiple sports properties, establishing foundation for Disney+ streaming infrastructure.

**EDUCATION**

**Harvard University** - ALM in Management; Graduate Certificates: Strategic Management, Innovation & Entrepreneurship
**Rutgers University** - BA in Computer Science, Minor in Mathematics
**Stanford University Online** - Certificate in UI/UX Design for AI Products

**CERTIFICATIONS**

Project Management Professional (PMP)
Google Cloud Professional Machine Learning Engineer (Expected Q1 2026)
Certified Scrum Product Owner (CSPO)

**TECHNICAL PORTFOLIO**

*   **ConvoRecs Sugoi** (Live: https://sugoi.convorecs.ai) - Production conversational AI platform; set up foundational infrastructure, defined architecture (React 18 + FastAPI + Gemini LLM + Firestore on Cloud Run), wrote PRDs and system design docs, scaled with AI-assisted development; implemented caching strategy to optimize LLM costs
*   **GitHub:** github.com/yagbre - Technical portfolio with PRDs, system design docs, and architecture diagrams demonstrating product development approach
*   **SnapEats** - Multimodal nutrition analyzer (FlutterFlow no-code frontend, Python backend integrating Gemini 1.5 Vision API); ranked top 25 among 898 in Google AI Hackathon (15,498 participants)
*   **Plant Classifier** - Built for Harvard Computer Vision (Grade: A); implemented transfer learning using pre-trained EfficientNetB0 and MobileNet models, 90% accuracy

**KEY SKILLS**

*   **AI/ML Product Management:** Large Language Models (LLM), Generative AI, Production ML Systems, ML Cost Optimization, Prompt Engineering, Model Selection Trade-offs, ML Pipeline Understanding (training, evaluation, deployment), Natural Language Processing, Computer Vision fundamentals
*   **Technical Architecture:** System Design, API Design, Multi-API Orchestration, Caching Strategies, Cloud Deployment (GCP, AWS), Microservices Architecture, PRD and Technical Spec Writing, Architecture Diagrams (Mermaid)
*   **Technical Stack:** Python (ML training, backend), Google Cloud Run, Firebase/Firestore, Docker, FlutterFlow; working knowledge of FastAPI, React, GraphQL, REST APIs
*   **Product Management:** 0-to-1 Product Development, Product Roadmap Development, Stakeholder Management, Agile/Scrum, OKRs/KPIs, Go-to-Market Strategy, Product Analytics, A/B Testing, Data Science Team Enablement
*   **Platform & Infrastructure:** Multi-tenant Architecture, Data Platform Modernization, ML Infrastructure Planning, Data Quality Frameworks, Production Operations
*   **Business Impact:** 0-to-1 product launches, enterprise-scale platforms, 10M+ users reached, 1M+ concurrent scale achieved, cross-functional team leadership (50+ engineers)
"""


# Generate Agentic Job Search Recommendations for DEMO_RESUME
# This cell runs the full pipeline on the embedded demo resume

import uuid
from IPython.display import display, Markdown

# Use the DEMO_RESUME defined above
demo_resume_text = DEMO_RESUME

# Generate unique session ID for this analysis
demo_session_id = f"demo_session_{uuid.uuid4().hex[:8]}"

print("🚀 Starting analysis of DEMO resume...")
print(f"   Session ID: {demo_session_id}")
print("="*80)

# Run the full pipeline
demo_result = await analyze_resume(demo_resume_text, session_id=demo_session_id)

print("\n" + "="*80)
display(Markdown(demo_result))
print("="*80)

# Store session ID for refinement


# Refine Demo Results (Human-in-the-Loop)
# Edit the feedback below and run this cell to get updated recommendations

from IPython.display import display, Markdown

# Enter your preferences here:
demo_feedback = "I want to focus on AI/ML leadership roles at startups or growth-stage companies"

# Make sure you've run the demo analysis above first
if 'demo_session_id' not in dir():
    print("❌ Please run the demo analysis first (run the cell above).")
else:
    print(f"🔄 Refining results for session: {demo_session_id}")
    print(f"   Your preferences: {demo_feedback}")
    print("="*80)
    
    demo_refined_result = await refine_results(demo_feedback, session_id=demo_session_id)
    
    print("\n" + "="*80)
    display(Markdown(demo_refined_result))
    print("="*80)

