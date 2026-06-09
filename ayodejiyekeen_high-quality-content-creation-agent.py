!pip install -q tavily-python


import os
import re
import asyncio
from typing import Dict, List, Any

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool, google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin

from google.genai import types

from kaggle_secrets import UserSecretsClient

from tavily import TavilyClient


user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
OPENAI_API_KEY = user_secrets.get_secret("OPENAI_API_KEY")
TAVILY_API_KEY = user_secrets.get_secret("TAVILY_API_KEY")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"]= "FALSE"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Verify API keys
if not GOOGLE_API_KEY:
    print("âš ï¸� Warning: GOOGLE_API_KEY not found.")
if not OPENAI_API_KEY:
    print("âš ï¸� Warning: OPENAI_API_KEY not found.")
if not TAVILY_API_KEY:
    print("âš ï¸� Warning: TAVILY_API_KEY not found.")


tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# Configure retry options
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)
print("âœ… Retry configuration set.")


# ============================================================================
# CUSTOM TOOLS (with ADK best practices: status/data format + error handling)
# ============================================================================
def extract_key_information(query: str, search_results: str) -> Dict[str, Any]:
    """Extract key information, facts, and statistics from search results.
    
    Args:
        query: The search query that was used
        search_results: The search results text to analyze
    
    Returns:
        Dictionary with status and extracted data or error message
    """
    try:
        if not query or not search_results:
            return {
                "status": "error",
                "error_message": "Both query and search_results are required"
            }
        
        # Simple extraction for demo (in production, use NLP)
        return {
            "status": "success",
            "data": {
                "query": query,
                "key_facts": "Extracted facts from results",
                "statistics": "Relevant stats",
                "insights": "Key insights",
                "summary": f"Summary for: {query}"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to extract key information: {str(e)}"
        }

def analyze_keyword_opportunity(topic: str) -> Dict[str, Any]:
    """Analyze keyword opportunities for a given topic.
    
    Args:
        topic: The topic to analyze for keywords
    
    Returns:
        Dictionary with status and keyword data or error message
    """
    try:
        if not topic:
            return {
                "status": "error",
                "error_message": "Topic is required"
            }
        
        keywords = topic.lower().split()
        primary = " ".join(keywords[:3]) if len(keywords) >= 3 else topic
        
        return {
            "status": "success",
            "data": {
                "primary_keyword": primary,
                "secondary_keywords": keywords[:5],
                "long_tail_keywords": [f"{primary} guide", f"best {primary}", f"{primary} tips"],
                "difficulty": "medium",
                "recommendation": f"Focus on '{primary}'"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to analyze keywords: {str(e)}"
        }

def check_readability(text: str) -> Dict[str, Any]:
    """Calculate readability scores for content.
    
    Args:
        text: The text to analyze
    
    Returns:
        Dictionary with status and readability metrics or error message
    """
    try:
        if not text.strip():
            return {"status": "error", "error_message": "Text required"}
        
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.split()
        
        if not sentences or not words:
            return {"status": "error", "error_message": "Invalid text"}
        
        avg_sentence = len(words) / len(sentences)
        avg_word = sum(len(w) for w in words) / len(words)
        score = max(0, min(100, 100 - (avg_sentence * 2) - (avg_word * 3)))
        
        return {
            "status": "success",
            "data": {
                "readability_score": round(score, 2),
                "avg_sentence_length": round(avg_sentence, 2),
                "avg_word_length": round(avg_word, 2),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "recommendation": "Good" if score > 70 else "Improve"
            }
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def check_keyword_density(content: str, keywords: List[str]) -> Dict[str, Any]:
    """Check keyword density in content.
    
    Args:
        content: The content to analyze
        keywords: List of keywords to check
    
    Returns:
        Dictionary with status and keyword density metrics or error message
    """
    try:
        if not content.strip() or not keywords:
            return {"status": "error", "error_message": "Content and keywords required"}
        
        content_lower = content.lower()
        word_count = len(content.split())
        
        if word_count == 0:
            return {"status": "error", "error_message": "No words in content"}
        
        densities = {kw: {"count": content_lower.count(kw.lower()), "density": round((content_lower.count(kw.lower()) / word_count * 100), 2)} for kw in keywords}
        natural = all(d["density"] < 3 for d in densities.values())
        
        return {
            "status": "success",
            "data": {
                "word_count": word_count,
                "keyword_densities": densities,
                "is_natural": natural,
                "recommendation": "Natural" if natural else "Reduce frequency"
            }
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def optimize_meta_tags(content: str, primary_keyword: str) -> Dict[str, Any]:
    """Generate optimized meta title and description.
    
    Args:
        content: The content to create meta tags for
        primary_keyword: The primary keyword to include
    
    Returns:
        Dictionary with status and optimized meta tags or error message
    """
    try:
        if not content.strip() or not primary_keyword.strip():
            return {"status": "error", "error_message": "Content and keyword required"}
        
        first_sentence = content.split('.')[0][:150] if '.' in content else content[:150]
        title = f"{primary_keyword.title()}: Complete Guide"[:60]
        desc = f"Learn about {primary_keyword}. {first_sentence}"[:160]
        
        return {
            "status": "success",
            "data": {
                "meta_title": title,
                "meta_description": desc,
                "title_length": len(title),
                "description_length": len(desc),
                "includes_keyword": primary_keyword.lower() in title.lower()
            }
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def check_quality_metrics(content: str, objectives: List[str]) -> Dict[str, Any]:
    """Check if content meets quality objectives.
    
    Args:
        content: The content to evaluate
        objectives: List of content objectives to check
    
    Returns:
        Dictionary with status and quality scores or error message
    """
    try:
        if not content.strip():
            return {"status": "error", "error_message": "Content required"}
        
        word_count = len(content.split())
        has_headings = bool(re.search(r'^#+\s', content, re.MULTILINE))
        has_lists = bool(re.search(r'^[\*\-\+]', content, re.MULTILINE))
        
        score = 0
        if word_count >= 500: score += 25
        if word_count >= 1000: score += 10
        if has_headings: score += 20
        if has_lists: score += 15
        if objectives: score += 30
        
        final = min(100, score)
        
        return {
            "status": "success",
            "data": {
                "quality_score": final,
                "word_count": word_count,
                "has_structure": has_headings,
                "has_lists": has_lists,
                "objectives_count": len(objectives) if objectives else 0,
                "recommendation": "High quality" if final >= 75 else "Improve"
            }
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def exit_loop() -> Dict[str, Any]:
    """Call this ONLY when quality report indicates APPROVED.
    
    Returns:
        Dictionary indicating approval to exit
    """
    return {
        "status": "success",
        "data": {
            "action": "exit_loop",
            "approved": True,
            "message": "Content approved. Exiting loop."
        }
    }


def custom_web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Custom web search using Tavily API.
    
    Args:
        query: Search query
        max_results: Maximum number of results
    
    Returns:
        Dictionary with search results
    """
    try:
        response = tavily_client.search(query=query, max_results=max_results)
        return {
            "status": "success",
            "data": {
                "query": query,
                "results": response.get("results", []),
                "answer": response.get("answer", "")
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Search failed: {str(e)}"
        }


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================
# Research Agent
research_agent = LlmAgent(
    name="ResearchAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Gathers information using web search and extraction",
    instruction="""Research specialist: Use google_search and extract_key_information.
Output: Key findings, stats, sources, trends.""",
    tools=[FunctionTool(custom_web_search), FunctionTool(extract_key_information)],
    output_key="research_findings"
)

# Keyword Agent
keyword_agent = LlmAgent(
    name="KeywordAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Identifies optimal keywords",
    instruction="""SEO specialist: Use analyze_keyword_opportunity.
Output: Primary/secondary/long-tail keywords, strategy.""",
    tools=[FunctionTool(custom_web_search), FunctionTool(analyze_keyword_opportunity)],
    output_key="keyword_strategy"
)

# Strategy Agent
strategy_agent = LlmAgent(
    name="StrategyAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Creates content plans",
    instruction="""Strategy specialist: Use {research_findings} and {keyword_strategy}.
Output: Objectives, audience, value prop, messages.""",
    output_key="content_strategy"
)

# Outline Agent
outline_agent = LlmAgent(
    name="OutlineAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Creates detailed outlines",
    instruction="""Outline specialist: Use {research_findings}, {keyword_strategy}, {content_strategy}.
Output: Headline, intro, sections with points, conclusion.""",
    output_key="content_outline"
)

# Writer Agent
writer_agent = LlmAgent(
    name="WriterAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Writes engaging content",
    instruction="""Writer: Use {content_outline}, {research_findings}, {content_strategy}.
Output: Markdown draft.""",
    output_key="content_draft"
)

# Editor Agent
editor_agent = LlmAgent(
    name="EditorAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Improves clarity and flow",
    instruction="""Editor: Use check_readability on {content_draft}.
Output: Edited content.""",
    tools=[FunctionTool(check_readability)],
    output_key="edited_content"
)

# SEO Agent
seo_agent = LlmAgent(
    name="SEOAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Optimizes for SEO",
    instruction="""SEO specialist: Use optimize_meta_tags and check_keyword_density on {edited_content}, {keyword_strategy}.
Output: Optimized content with meta tags.""",
    tools=[FunctionTool(optimize_meta_tags), FunctionTool(check_keyword_density)],
    output_key="seo_optimized_content"
)

# Quality Checker Agent
quality_checker_agent = LlmAgent(
    name="QualityCheckerAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Verifies quality standards",
    instruction="""QA specialist: Use check_readability and check_quality_metrics on {seo_optimized_content}, {content_strategy}.
End with VERDICT: APPROVED or NEEDS_IMPROVEMENT.""",
    tools=[FunctionTool(check_readability), FunctionTool(check_quality_metrics)],
    output_key="quality_report"
)

# Refiner Agent
refiner_agent = LlmAgent(
    name="RefinerAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Refines based on feedback",
    instruction="""Refiner: If {quality_report} is APPROVED, call exit_loop. Else refine {seo_optimized_content}.""",
    tools=[FunctionTool(exit_loop)],
    output_key="refined_content"
)

# Reviewer Agent
reviewer_agent = LlmAgent(
    name="ReviewerAgent",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Final approval",
    instruction="""Reviewer: Review {refined_content}, {quality_report}, {content_strategy}.
Output: Final content with summary.""",
    output_key="final_content"
)


# ============================================================================
# PIPELINE CONSTRUCTION
# ============================================================================
# Phase 1: Parallel Research
research_phase = ParallelAgent(
    name="ResearchPhase",
    sub_agents=[research_agent, keyword_agent]
)

# Phase 2: Sequential Creation
creation_phase = SequentialAgent(
    name="CreationPhase",
    sub_agents=[strategy_agent, outline_agent, writer_agent]
)

# Phase 3: Optimization Loop (max 3 iterations)
optimization_loop = LoopAgent(
    name="OptimizationLoop",
    sub_agents=[editor_agent, seo_agent, quality_checker_agent, refiner_agent],
    max_iterations=3
)

# Phase 4: Final Review
finalization_phase = SequentialAgent(
    name="FinalizationPhase",
    sub_agents=[reviewer_agent]
)

# Root Pipeline
root_agent = SequentialAgent(
    name="ContentCreationPipeline",
    sub_agents=[research_phase, creation_phase, optimization_loop, finalization_phase]
)


# ============================================================================
# RUNNER SETUP WITH OBSERVABILITY
# ============================================================================
# Session service for state management
session_service = InMemorySessionService()
app_name = "MyContentApp"

# Create runner with logging plugin for observability
runner = Runner(
    agent=root_agent,
    app_name=app_name,
    session_service=session_service,
    plugins=[LoggingPlugin()]
)


async def run_pipeline(topic: str, user_id: str = "default_user") -> Dict[str, Any]:
    """Execute the content creation pipeline."""
    session_id = f"content_session_{id(topic)}"
    
    try:
        # Create session
        await session_service.create_session(
            user_id=user_id,
            session_id=session_id,
            app_name=app_name
        )
        
        # Create user message
        user_message = types.Content(
            role='user', 
            parts=[types.Part(text=topic)]
        )
        
        print(f"\nğŸš€ Starting pipeline for: '{topic}'")
        print("="*80)
        
        # Run pipeline with streaming events
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message
        ):
            if hasattr(event, 'content') and event.content:
                print(f"\n[{event.author}]:")
                for part in event.content.parts:
                    # Handle text output
                    if hasattr(part, 'text') and part.text:
                        print(part.text)
                    # Handle function calls
                    elif hasattr(part, 'function_call') and part.function_call is not None:
                        func_name = getattr(part.function_call, 'name', 'unknown')
                        print(f"ğŸ”§ Tool call: {func_name}")
                    # Handle function responses
                    elif hasattr(part, 'function_response') and part.function_response is not None:
                        func_name = getattr(part.function_response, 'name', 'unknown')
                        print(f"âœ“ Tool response: {func_name}")
        
        print("\n" + "="*80)
        print("âœ… Pipeline completed successfully!")
        
        return {"status": "success", "message": "Pipeline completed"}
        
    except Exception as e:
        print(f"\nâ�Œ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error_message": str(e)}


# Run the pipeline
result = await run_pipeline('AI Agent Frameworks')

