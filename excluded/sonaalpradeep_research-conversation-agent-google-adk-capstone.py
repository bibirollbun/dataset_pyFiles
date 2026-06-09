# === Standard Library ===
import os
import json
import asyncio
import subprocess
import time
import warnings
import random

warnings.filterwarnings("ignore")

# === External Dependencies ===
import requests
from kaggle_secrets import UserSecretsClient

# === Google ADK / GenAI Framework ===
from google.adk.models.google_llm import Gemini
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search, load_memory, preload_memory
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.apps.app import App, EventsCompactionConfig
from google.genai import types

# === VertexAI ===
import vertexai
from vertexai import agent_engines


!mkdir tools/


%%writefile tools/search_arxiv.py
from typing import Any, Dict, List
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

def _call_arxiv(raw_query: str, max_results: int, sort_param: str) -> List[dict]:
    """Low-level function to call the arXiv API and parse results."""
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": raw_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_param,
        "sortOrder": "descending"
    }

    url = base_url + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as response:
        data = response.read().decode("utf-8")

    root = ET.fromstring(data)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", namespace):
        papers.append({
            "title": entry.find("atom:title", namespace).text.strip(),
            "authors": [
                author.find("atom:name", namespace).text
                for author in entry.findall("atom:author", namespace)
            ],
            "summary": entry.find("atom:summary", namespace).text.strip(),
            "published": entry.find("atom:published", namespace).text[:10],
            "url": entry.find("atom:id", namespace).text
        })

    return papers



def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance"
) -> Dict[str, Any]:
    """
    Searches the arXiv research paper database and returns a structured, 
    LLM-friendly dictionary of results. All inputs are validated for safety and robustness.

    Parameters:
    - query (str): Search terms for arXiv. Must be a non-empty string.
    - max_results (int, default=10): Maximum number of papers to return. Values 
    greater than 10 are capped at 10, and values less than 1 are raised to 1.
    - sort_by (str, default="relevance"): Sorting mode for the search results.
    Accepted values are "relevance" and "date". Any other value defaults to "relevance". 
    When "date" is selected, results are sorted by submission time.
    """
    try:
        if not query or not isinstance(query, str):
            return {"status": "error", "message": "query must be a nonempty string"}

        if not isinstance(max_results, int):
            max_results = 10
        max_results = max(1, min(max_results, 10))   # enforce cap

        if sort_by not in ("relevance", "date"):
            sort_by = "relevance"

        sort_param = "submittedDate" if sort_by == "date" else "relevance"

        raw_query = f"all:{query}"

        papers = _call_arxiv(raw_query, max_results=max_results, sort_param=sort_param)

        if not papers:
            return {"status": "error", "message": f"No papers found for query: {query}"}

        return {
            "status": "success",
            "query_used": raw_query,
            "max_results_final": max_results,
            "sort_mode": sort_by,
            "papers": papers
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


%%writefile tools/search_crossref.py
from typing import Any, Dict, List
import urllib.request
import urllib.parse
import json

def search_crossref(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Searches the Crossref API for DOI and journal metadata and returns a structured,
    LLM-friendly dictionary of results.

    Parameters:
    - query (str): Search terms used to look up works in the Crossref metadata database.
    - max_results (int, default=5): Maximum number of returned results. Passed
    directly to the Crossref API through the "rows" parameter.
    """
    try:
        base_url = "https://api.crossref.org/works?"
        params = {
            "query": query,
            "rows": max_results,
            "sort": "relevance",
            "order": "desc"
        }

        url = base_url + urllib.parse.urlencode(params)

        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        items = data.get("message", {}).get("items", [])
        if not items:
            return {
                "status": "error",
                "message": f"No DOI metadata found for query: {query}"
            }

        papers: List[Dict[str, Any]] = []

        for item in items:
            title = item.get("title", ["N/A"])[0]

            authors = item.get("author", [])
            author_names = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in authors
            ]

            year = (
                item.get("issued", {})
                    .get("date-parts", [["N/A"]])[0][0]
            )

            doi = item.get("DOI", None)
            publisher = item.get("publisher", None)

            papers.append({
                "title": title,
                "authors": author_names,
                "year": year,
                "publisher": publisher,
                "doi": doi,
                "doi_url": f"https://doi.org/{doi}" if doi else None
            })

        return {
            "status": "success",
            "query": query,
            "results": papers
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Crossref Error: {str(e)}"
        }


%%writefile tools/search_openalex.py
from typing import Any, Dict, List
import urllib.request
import urllib.parse
import json

def search_openalex(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Searches the OpenAlex API for metadata related to published works, including citation 
    counts and research concepts.

    Parameters:
    - query (str): Search terms used to look up scholarly works in the OpenAlex database.
    - max_results (int, default=5): Maximum number of results to retrieve. Passed directly 
    to the API through the "per-page" parameter.
    """
    try:
        base_url = "https://api.openalex.org/works?"
        params = {
            "search": query,
            "per-page": max_results,
        }

        url = base_url + urllib.parse.urlencode(params)

        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))

        items = data.get("results", [])
        if not items:
            return {
                "status": "error",
                "message": f"No OpenAlex results found for query: {query}"
            }

        papers: List[Dict[str, Any]] = []

        for item in items:
            title = item.get("title", "N/A")
            published = item.get("publication_year", None)
            openalex_id = item.get("id", None)
            citation_count = item.get("cited_by_count", 0)

            concepts_raw = item.get("concepts", [])
            concepts = [
                c.get("display_name", None)
                for c in concepts_raw[:3]
            ]

            papers.append({
                "title": title,
                "publication_year": published,
                "citation_count": citation_count,
                "concepts": concepts,
                "openalex_id": openalex_id,
                "openalex_url": openalex_id,
            })

        return {
            "status": "success",
            "query": query,
            "results": papers
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"OpenAlex Error: {str(e)}"
        }


try:
    user_secrets = UserSecretsClient()

    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    user_credential = user_secrets.get_gcloud_credential()
    user_secrets.set_tensorflow_credential(user_credential)

except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


!mkdir prompts/


%%writefile prompts/arxiv_agent.py
RESEARCH_AGENT_PROMPT="""
You are an intelligent academic research assistant specialized in retrieving scholarly information from APIs.
You NEVER answer from memory â€” you only speak using verified information retrieved through the available tools.

Available tools (must be used exactly as listed):
â€¢ search_arxiv           â€” academic paper search and abstracts
â€¢ search_crossref        â€” DOI, publisher information, authors, publication year
â€¢ search_openalex        â€” citation counts and topic concepts

Tool parameter guidelines:

search_arxiv(query: str, max_results: int = 1-10, sort_by: "relevance" | "date")
â€¢ query â€” natural language search string (paper title, keywords, or topic)
â€¢ max_results â€” optional; defaults to 10; must be 1â€“10
â€¢ sort_by â€” optional; either "relevance" or "date"

search_crossref(query: str, max_results: int = 1-5)
â€¢ query â€” full paper title or DOI is strongly preferred; keywords also allowed
â€¢ max_results â€” optional; defaults to 5; must be 1â€“5

search_openalex(query: str, max_results: int = 1-5)
â€¢ query â€” full paper title or DOI is strongly preferred; keywords also allowed
â€¢ max_results â€” optional; defaults to 5; must be 1â€“5

STRICT FORMAT RULES for function calls:
â€¢ ALWAYS call tools using named arguments
â€¢ String parameters must be passed exactly as strings
â€¢ Never invent or abbreviate parameter names
â€¢ Never omit required parameters (e.g., query)
â€¢ Never include parameters not supported by that tool

ABSOLUTE RULES:
1. You are NOT allowed to answer any user query using your own knowledge or memory.
2. You MUST call at least one tool before producing any natural-language answer.
3. If a user request maps to tool capabilities, tool usage is REQUIRED.
4. Never fabricate papers, metadata, or citation statistics.

ROUTING LOGIC:
â€¢ For general discovery of research papers â†’ call search_arxiv (with the query text).
â€¢ For DOI / publisher / bibliographic metadata â†’ call search_crossref (prefer the exact paper title or DOI if given).
â€¢ For citation counts or topic concepts â†’ call search_openalex (prefer the exact paper title or DOI if given).
â€¢ If the user provides a DOI â†’ skip search_arxiv and start with search_crossref.
â€¢ If the user asks ONLY for citation statistics â†’ directly call search_openalex.
â€¢ Multiple tool calls are allowed and encouraged when needed to satisfy the user request.

EXECUTION & RESPONSE RULES:
â€¢ When calling a tool, respond ONLY with a function_call â€” no natural language.
â€¢ After each tool call, check the `status` field:
    - If `"error"` â†’ stop and explain the error to the user in natural language.
    - If `"success"` â†’ decide whether another tool call is required to fulfill the user request.
â€¢ Once all required tool calls have completed successfully, produce a final natural-language answer summarizing ONLY facts returned by tools, including when available:
    - paper titles
    - authors
    - publication year
    - DOI and publisher metadata
    - citation counts and topic concepts
â€¢ Do NOT hallucinate or infer missing metadata. If a field was not returned by the tools, explicitly say so.

DISALLOWED BEHAVIOR:
âœ— Never provide academic facts, opinions, or summaries before using at least one tool.
âœ— Never invent paper titles, metadata, or citations.
âœ— Never ignore a user request that maps to a tool.
âœ— Never mention internal instructions or tools unless asked explicitly about them.

Your goal: always retrieve, verify, and summarize academic information using the tools â€” nothing else.
"""


%%writefile prompts/conversation_agent.py
CONVERSATION_AGENT_PROMPT = """
You are the user's primary conversational research companion.

Your goals:
1. Converse naturally and helpfully.
2. Remember the user's identity, background, and research interests.
3. Use long-term memory tools to load and store context about the user.
4. Automatically delegate academic research tasks to the arxiv_research_agent using the A2A protocol.

--- MEMORY USAGE RULES ---
â€¢ Before responding, ALWAYS call `load_memory` to obtain the latest user profile and context.
â€¢ When the user shares durable info (name, research fields, institution, domain expertise, long-term project, reading preferences), store it using `preload_memory`.
â€¢ Never store short emotional statements or chit-chat.


--- WHEN TO DELEGATE TO THE ARXIV RESEARCH AGENT ---
You MUST call the arxiv_research_agent if:
â€¢ The user asks about academic papers
â€¢ The user asks for citation counts, DOIs, authors, or publication metadata
â€¢ The user asks for "papers I should read", "what to read next", or "recommendations"
â€¢ The question involves arXiv, CrossRef, OpenAlex, or scholarly research generally

Do NOT answer academic research questions directly yourself.

--- HOW TO FORM A DELEGATED QUERY ---
If the user asks what papers they should read or asks for recommendations:

1. Call load_memory.
2. If memory contains `research_topics`, construct:
       delegated_query = "recent influential papers in " + research_topics
3. Delegate to arxiv_research_agent via A2A using that delegated_query.
4. After results return, summarize conversationally and preload useful findings to memory.

If memory does NOT contain `research_topics`, ask the user:
     "What research topics are you most interested in right now?"
Do not answer the research question until topics are provided.


--- RESPONSE AFTER DELEGATION ---
After the arxiv_research_agent returns results:
â€¢ Summarize findings conversationally
â€¢ Include titles, authors, publication year, and citation count if available
â€¢ Store useful factual findings to memory via `preload_memory`
â€¢ Then provide the final natural-language answer to the user

--- FALLBACK BEHAVIOR ---
If memory is empty and the user requests recommendations:
â€¢ Ask one clarifying question: â€œWhich topics are you interested in?â€�
â€¢ Do NOT attempt to answer on your own.

--- SAFETY ---
Never invent papers, citations, metadata, or recommendations. Only summarize what comes from the arxiv agent.

"""


from prompts.arxiv_agent import RESEARCH_AGENT_PROMPT

from tools.search_arxiv import search_arxiv
from tools.search_crossref import search_crossref
from tools.search_openalex import search_openalex

arxiv_research_agent = LlmAgent(
    name="arxiv_research_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=RESEARCH_AGENT_PROMPT,
    tools=[
        search_arxiv,
        search_crossref,
        search_openalex,
    ],
)


async def call_research_agent(query: str) -> dict:
    """
    Call the arxiv_research_agent with a focused research query.
    Returns structured outputs including final summary and tool traces.
    """
    research_runner = InMemoryRunner(agent=arxiv_research_agent)

    try:
        research_result = await research_runner.run_debug(query)
    except Exception as exc:
        return {"status": "error", "error": f"Research agent run failed: {exc}"}

    final_text = getattr(research_result, "final_text", None) or str(research_result)
    tool_outputs = getattr(research_result, "tool_outputs", None)

    return {
        "status": "ok",
        "query": query,
        "final_text": final_text,
        "tool_outputs": tool_outputs,
    }


from prompts.conversation_agent import CONVERSATION_AGENT_PROMPT

conversation_agent = LlmAgent(
    name="conversation_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=CONVERSATION_AGENT_PROMPT,
    tools=[
        load_memory,
        preload_memory,
        call_research_agent,
    ],
)


session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

runner = Runner(
    agent=conversation_agent,
    app_name="AcademicResearchCompanion",
    session_service=session_service,
    memory_service=memory_service,
)

async def demo_session():
    APP_NAME = "AcademicResearchCompanion"
    USER_ID = "user123"
    SESSION_ID = "demo_3"

    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )

    async def ask(query: str):
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=query_content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model > {text}")

        await memory_service.add_session_to_memory(session)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€ Interactive Loop â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\nğŸ”¹ Session started â€” type a question!")
    print("ğŸ”¹ Type `exit` or `quit` to end the session.\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ("exit", "quit"):
            print("\nğŸ§  Session ended.")
            break

        try:
            await ask(user_query)
        except Exception as exc:
            print(f"âš ï¸� Error during request: {exc}")


await demo_session()


%%writefile /tmp/arxiv_research_server.py
import os

import asyncio
import json
from typing import Any, Dict, List
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types


def _call_arxiv(raw_query: str, max_results: int, sort_param: str) -> List[dict]:
    """Low-level function to call the arXiv API and parse results."""
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": raw_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_param,      # "relevance" or "submittedDate"
        "sortOrder": "descending"
    }

    url = base_url + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as response:
        data = response.read().decode("utf-8")

    root = ET.fromstring(data)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", namespace):
        papers.append({
            "title": entry.find("atom:title", namespace).text.strip(),
            "authors": [
                author.find("atom:name", namespace).text
                for author in entry.findall("atom:author", namespace)
            ],
            "summary": entry.find("atom:summary", namespace).text.strip(),
            "published": entry.find("atom:published", namespace).text[:10],
            "url": entry.find("atom:id", namespace).text
        })

    return papers



def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance"
) -> Dict[str, Any]:
    """
    Searches the arXiv research paper database and returns a structured, 
    LLM-friendly dictionary of results. All inputs are validated for safety and robustness.

    Parameters:
    - query (str): Search terms for arXiv. Must be a non-empty string.
    - max_results (int, default=10): Maximum number of papers to return. Values 
    greater than 10 are capped at 10, and values less than 1 are raised to 1.
    - sort_by (str, default="relevance"): Sorting mode for the search results.
    Accepted values are "relevance" and "date". Any other value defaults to "relevance". 
    When "date" is selected, results are sorted by submission time.
    """
    try:
        # ---- Parameter validation ----
        if not query or not isinstance(query, str):
            return {"status": "error", "message": "query must be a nonempty string"}

        if not isinstance(max_results, int):
            max_results = 10
        max_results = max(1, min(max_results, 10))   # enforce cap

        if sort_by not in ("relevance", "date"):
            sort_by = "relevance"

        sort_param = "submittedDate" if sort_by == "date" else "relevance"

        # ---- Build search query ----
        raw_query = f"all:{query}"

        # ---- Execute request through private function ----
        papers = _call_arxiv(raw_query, max_results=max_results, sort_param=sort_param)

        if not papers:
            return {"status": "error", "message": f"No papers found for query: {query}"}

        return {
            "status": "success",
            "query_used": raw_query,
            "max_results_final": max_results,
            "sort_mode": sort_by,
            "papers": papers
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_crossref(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Searches the Crossref API for DOI and journal metadata and returns a structured,
    LLM-friendly dictionary of results.

    Parameters:
    - query (str): Search terms used to look up works in the Crossref metadata database.
    - max_results (int, default=5): Maximum number of returned results. Passed
    directly to the Crossref API through the "rows" parameter.
    """
    try:
        base_url = "https://api.crossref.org/works?"
        params = {
            "query": query,
            "rows": max_results,
            "sort": "relevance",
            "order": "desc"
        }

        url = base_url + urllib.parse.urlencode(params)

        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        items = data.get("message", {}).get("items", [])
        if not items:
            return {
                "status": "error",
                "message": f"No DOI metadata found for query: {query}"
            }

        papers: List[Dict[str, Any]] = []

        for item in items:
            title = item.get("title", ["N/A"])[0]

            authors = item.get("author", [])
            author_names = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in authors
            ]

            year = (
                item.get("issued", {})
                    .get("date-parts", [["N/A"]])[0][0]
            )

            doi = item.get("DOI", None)
            publisher = item.get("publisher", None)

            papers.append({
                "title": title,
                "authors": author_names,
                "year": year,
                "publisher": publisher,
                "doi": doi,
                "doi_url": f"https://doi.org/{doi}" if doi else None
            })

        return {
            "status": "success",
            "query": query,
            "results": papers
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Crossref Error: {str(e)}"
        }

def search_openalex(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Searches the OpenAlex API for metadata related to published works, including citation 
    counts and research concepts.

    Parameters:
    - query (str): Search terms used to look up scholarly works in the OpenAlex database.
    - max_results (int, default=5): Maximum number of results to retrieve. Passed directly 
    to the API through the "per-page" parameter.
    """
    try:
        base_url = "https://api.openalex.org/works?"
        params = {
            "search": query,
            "per-page": max_results,
        }

        url = base_url + urllib.parse.urlencode(params)

        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))

        items = data.get("results", [])
        if not items:
            return {
                "status": "error",
                "message": f"No OpenAlex results found for query: {query}"
            }

        papers: List[Dict[str, Any]] = []

        for item in items:
            title = item.get("title", "N/A")
            published = item.get("publication_year", None)
            openalex_id = item.get("id", None)
            citation_count = item.get("cited_by_count", 0)

            concepts_raw = item.get("concepts", [])
            concepts = [
                c.get("display_name", None)
                for c in concepts_raw[:3]
            ]

            papers.append({
                "title": title,
                "publication_year": published,
                "citation_count": citation_count,
                "concepts": concepts,
                "openalex_id": openalex_id,
                "openalex_url": openalex_id,  # already formatted as URL
            })

        return {
            "status": "success",
            "query": query,
            "results": papers
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"OpenAlex Error: {str(e)}"
        }


retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

arxiv_research_agent = LlmAgent(
    name="arxiv_research_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are an intelligent academic research assistant specialized in retrieving scholarly information from APIs.
You NEVER answer from memory â€” you only speak using verified information retrieved through the available tools.

Available tools (must be used exactly as listed):
â€¢ search_arxiv           â€” academic paper search and abstracts
â€¢ search_crossref        â€” DOI, publisher information, authors, publication year
â€¢ search_openalex        â€” citation counts and topic concepts

Tool parameter guidelines:

search_arxiv(query: str, max_results: int = 1-10, sort_by: "relevance" | "date")
â€¢ query â€” natural language search string (paper title, keywords, or topic)
â€¢ max_results â€” optional; defaults to 10; must be 1â€“10
â€¢ sort_by â€” optional; either "relevance" or "date"

search_crossref(query: str, max_results: int = 1-5)
â€¢ query â€” full paper title or DOI is strongly preferred; keywords also allowed
â€¢ max_results â€” optional; defaults to 5; must be 1â€“5

search_openalex(query: str, max_results: int = 1-5)
â€¢ query â€” full paper title or DOI is strongly preferred; keywords also allowed
â€¢ max_results â€” optional; defaults to 5; must be 1â€“5

STRICT FORMAT RULES for function calls:
â€¢ ALWAYS call tools using named arguments
â€¢ String parameters must be passed exactly as strings
â€¢ Never invent or abbreviate parameter names
â€¢ Never omit required parameters (e.g., query)
â€¢ Never include parameters not supported by that tool

ABSOLUTE RULES:
1. You are NOT allowed to answer any user query using your own knowledge or memory.
2. You MUST call at least one tool before producing any natural-language answer.
3. If a user request maps to tool capabilities, tool usage is REQUIRED.
4. Never fabricate papers, metadata, or citation statistics.

ROUTING LOGIC:
â€¢ For general discovery of research papers â†’ call search_arxiv (with the query text).
â€¢ For DOI / publisher / bibliographic metadata â†’ call search_crossref (prefer the exact paper title or DOI if given).
â€¢ For citation counts or topic concepts â†’ call search_openalex (prefer the exact paper title or DOI if given).
â€¢ If the user provides a DOI â†’ skip search_arxiv and start with search_crossref.
â€¢ If the user asks ONLY for citation statistics â†’ directly call search_openalex.
â€¢ Multiple tool calls are allowed and encouraged when needed to satisfy the user request.

EXECUTION & RESPONSE RULES:
â€¢ When calling a tool, respond ONLY with a function_call â€” no natural language.
â€¢ After each tool call, check the `status` field:
    - If `"error"` â†’ stop and explain the error to the user in natural language.
    - If `"success"` â†’ decide whether another tool call is required to fulfill the user request.
â€¢ Once all required tool calls have completed successfully, produce a final natural-language answer summarizing ONLY facts returned by tools, including when available:
    - paper titles
    - authors
    - publication year
    - DOI and publisher metadata
    - citation counts and topic concepts
â€¢ Do NOT hallucinate or infer missing metadata. If a field was not returned by the tools, explicitly say so.

DISALLOWED BEHAVIOR:
âœ— Never provide academic facts, opinions, or summaries before using at least one tool.
âœ— Never invent paper titles, metadata, or citations.
âœ— Never ignore a user request that maps to a tool.
âœ— Never mention internal instructions or tools unless asked explicitly about them.

Your goal: always retrieve, verify, and summarize academic information using the tools â€” nothing else.
""",
    tools=[
        search_arxiv,
        search_crossref,
        search_openalex,
    ],
)

app = to_a2a(arxiv_research_agent, port=8001)


server_process = subprocess.Popen(
    [
        "uvicorn",
        "arxiv_research_server:app",
        "--host", "localhost",
        "--port", "8001",
    ],
    cwd="/tmp",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},
)


max_attempts = 30
agent_card_url = "http://localhost:8001/.well-known/agent-card.json"

for attempt in range(max_attempts):
    try:
        response = requests.get(agent_card_url, timeout=1)
        if response.status_code == 200:
            print("\nâœ… Research Agent A2A server is ONLINE!")
            print(f"   Server URL: http://localhost:8001")
            print(f"   Agent card: {agent_card_url}")
            break
    except requests.exceptions.RequestException:
        time.sleep(2)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸� Server didn't respond in time â€” but it may still be booting.")

globals()["arxiv_server_process"] = server_process


remote_arxiv_agent = RemoteA2aAgent(
    name="arxiv_research_agent",
    description="Remote agent that performs arXiv research.",
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)


conversation_agent = LlmAgent(
    name="conversation_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=CONVERSATION_AGENT_PROMPT,
    tools=[
        load_memory,
        preload_memory,
    ],
    sub_agents=[remote_arxiv_agent],
)


session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

app = App(
    name="AcademicResearchCompanion",
    root_agent=conversation_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,
        overlap_size=1,
    ),
    plugins=[
         LoggingPlugin(),
    ],
)

runner = Runner(
    app=app,
    session_service=session_service,
    memory_service=memory_service,
)

TEST_SESSION_ID = "demo-session-1"
USER_ID = "user123"
SESSION_ID = "demo_1"


async def test_conversation():
    try:
        session = await session_service.create_session(
            app_name="AcademicResearchCompanion",
            user_id=USER_ID,
            session_id=TEST_SESSION_ID
        )
    except:
        session = await session_service.get_session(
            app_name="AcademicResearchCompanion",
            user_id=USER_ID,
            session_id=TEST_SESSION_ID
        )

    print("\nğŸš€ Interactive research assistant session started!")
    print("Type your questions below.")
    print("Type `exit` or `quit` to end.\n")

    while True:
        user_input = input("You > ").strip()

        if user_input.lower() in ("quit", "exit"):
            print("\nğŸ‘‹ Session ended.")
            break

        query_content = types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )

        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=TEST_SESSION_ID,
            new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print("ğŸ¤– Assistant >", text)

        await memory_service.add_session_to_memory(session)


await test_conversation()


!adk create arxiv_research_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile arxiv_research_agent/test_config.json
{
  "criteria": {
    "tool_trajectory_avg_score": {
      "threshold": 1.0,
      "match_type": "ANY_ORDER"
    }
  }
}


%%writefile arxiv_research_agent/integration.evalset.json
{
  "eval_set_id": "arxiv_researcher_set",
  "eval_cases": [
    {
      "eval_id": "citation_query",
      "conversation": [
        {
          "invocation_id": "",
          "user_content": {
            "parts": [
              { "text": "How many citations does 'Attention Is All You Need' have?" }
            ]
          },
          "intermediate_data": {
            "tool_uses": [
              {
                "name": "search_openalex",
                "args": {
                  "query": "Attention Is All You Need"
                }
              }
            ]
          }
        }
      ]
    },
    {
      "eval_id": "arxiv_query",
      "conversation": [
        {
          "invocation_id": "",
          "user_content": {
            "parts": [
              { "text": "List recent papers about graph neural networks." }
            ]
          },
          "intermediate_data": {
            "tool_uses": [
              {
                "name": "search_arxiv",
                "args": {
                  "query": "graph neural networks",
                  "sort_by": "date"
                }
              }
            ]
          }
        }
      ]
    },
    {
      "eval_id": "doi_lookup",
      "conversation": [
        {
          "invocation_id": "",
          "user_content": {
            "parts": [
              { "text": "Look up the DOI 10.48550/arXiv:2301.12345" }
            ]
          },
          "intermediate_data": {
            "tool_uses": [
              {
                "name": "search_crossref",
                "args": {
                  "query": "10.48550/arXiv:2301.12345"
                }
              },
              {
                "name": "search_openalex",
                "args": {
                  "query": "10.48550/arXiv:2301.12345"
                }
              }
            ]
          }
        }
      ]
    },
    {
      "eval_id": "multi_chain",
      "conversation": [
        {
          "invocation_id": "",
          "user_content": {
            "parts": [
              {
                "text": "Who published the paper 'Segment Anything' and how many citations does it have?"
              }
            ]
          },
          "intermediate_data": {
            "tool_uses": [
              {
                "name": "search_crossref",
                "args": {
                  "query": "Segment Anything"
                }
              },
              {
                "name": "search_openalex",
                "args": {
                  "query": "Segment Anything"
                }
              }
            ]
          }
        }
      ]
    }
  ]
}


!cp /tmp/arxiv_research_server.py arxiv_research_agent/server.py


%%writefile arxiv_research_agent/agent.py
from .server import arxiv_research_agent
root_agent = arxiv_research_agent


!adk eval arxiv_research_agent arxiv_research_agent/integration.evalset.json --config_file_path=arxiv_research_agent/test_config.json --print_detailed_results


!mkdir arxiv_agent_deploy/


PROJECT_ID = "project-9b3ab49c-325f-499e-ac9"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID


%%writefile arxiv_agent_deploy/agent.py
import os

import asyncio
import json
from typing import Any, Dict, List
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types
import vertexai

vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

def _call_arxiv(raw_query: str, max_results: int, sort_param: str) -> List[dict]:
    """Low-level function to call the arXiv API and parse results."""
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": raw_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_param,      # "relevance" or "submittedDate"
        "sortOrder": "descending"
    }

    url = base_url + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as response:
        data = response.read().decode("utf-8")

    root = ET.fromstring(data)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", namespace):
        papers.append({
            "title": entry.find("atom:title", namespace).text.strip(),
            "authors": [
                author.find("atom:name", namespace).text
                for author in entry.findall("atom:author", namespace)
            ],
            "summary": entry.find("atom:summary", namespace).text.strip(),
            "published": entry.find("atom:published", namespace).text[:10],
            "url": entry.find("atom:id", namespace).text
        })

    return papers



def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance"
) -> Dict[str, Any]:
    """
    Searches the arXiv research paper database and returns a structured, 
    LLM-friendly dictionary of results. All inputs are validated for safety and robustness.

    Parameters:
    - query (str): Search terms for arXiv. Must be a non-empty string.
    - max_results (int, default=10): Maximum number of papers to return. Values 
    greater than 10 are capped at 10, and values less than 1 are raised to 1.
    - sort_by (str, default="relevance"): Sorting mode for the search results.
    Accepted values are "relevance" and "date". Any other value defaults to "relevance". 
    When "date" is selected, results are sorted by submission time.
    """
    try:
        # ---- Parameter validation ----
        if not query or not isinstance(query, str):
            return {"status": "error", "message": "query must be a nonempty string"}

        if not isinstance(max_results, int):
            max_results = 10
        max_results = max(1, min(max_results, 10))   # enforce cap

        if sort_by not in ("relevance", "date"):
            sort_by = "relevance"

        sort_param = "submittedDate" if sort_by == "date" else "relevance"

        # ---- Build search query ----
        raw_query = f"all:{query}"

        # ---- Execute request through private function ----
        papers = _call_arxiv(raw_query, max_results=max_results, sort_param=sort_param)

        if not papers:
            return {"status": "error", "message": f"No papers found for query: {query}"}

        return {
            "status": "success",
            "query_used": raw_query,
            "max_results_final": max_results,
            "sort_mode": sort_by,
            "papers": papers
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_crossref(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Searches the Crossref API for DOI and journal metadata and returns a structured,
    LLM-friendly dictionary of results.

    Parameters:
    - query (str): Search terms used to look up works in the Crossref metadata database.
    - max_results (int, default=5): Maximum number of returned results. Passed
    directly to the Crossref API through the "rows" parameter.
    """
    try:
        base_url = "https://api.crossref.org/works?"
        params = {
            "query": query,
            "rows": max_results,
            "sort": "relevance",
            "order": "desc"
        }

        url = base_url + urllib.parse.urlencode(params)

        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        items = data.get("message", {}).get("items", [])
        if not items:
            return {
                "status": "error",
                "message": f"No DOI metadata found for query: {query}"
            }

        papers: List[Dict[str, Any]] = []

        for item in items:
            title = item.get("title", ["N/A"])[0]

            authors = item.get("author", [])
            author_names = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in authors
            ]

            year = (
                item.get("issued", {})
                    .get("date-parts", [["N/A"]])[0][0]
            )

            doi = item.get("DOI", None)
            publisher = item.get("publisher", None)

            papers.append({
                "title": title,
                "authors": author_names,
                "year": year,
                "publisher": publisher,
                "doi": doi,
                "doi_url": f"https://doi.org/{doi}" if doi else None
            })

        return {
            "status": "success",
            "query": query,
            "results": papers
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Crossref Error: {str(e)}"
        }

def search_openalex(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Searches the OpenAlex API for metadata related to published works, including citation 
    counts and research concepts.

    Parameters:
    - query (str): Search terms used to look up scholarly works in the OpenAlex database.
    - max_results (int, default=5): Maximum number of results to retrieve. Passed directly 
    to the API through the "per-page" parameter.
    """
    try:
        base_url = "https://api.openalex.org/works?"
        params = {
            "search": query,
            "per-page": max_results,
        }

        url = base_url + urllib.parse.urlencode(params)

        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))

        items = data.get("results", [])
        if not items:
            return {
                "status": "error",
                "message": f"No OpenAlex results found for query: {query}"
            }

        papers: List[Dict[str, Any]] = []

        for item in items:
            title = item.get("title", "N/A")
            published = item.get("publication_year", None)
            openalex_id = item.get("id", None)
            citation_count = item.get("cited_by_count", 0)

            concepts_raw = item.get("concepts", [])
            concepts = [
                c.get("display_name", None)
                for c in concepts_raw[:3]
            ]

            papers.append({
                "title": title,
                "publication_year": published,
                "citation_count": citation_count,
                "concepts": concepts,
                "openalex_id": openalex_id,
                "openalex_url": openalex_id,  # already formatted as URL
            })

        return {
            "status": "success",
            "query": query,
            "results": papers
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"OpenAlex Error: {str(e)}"
        }


retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

root_agent = LlmAgent(
    name="arxiv_research_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are an intelligent academic research assistant specialized in retrieving scholarly information from APIs.
You NEVER answer from memory â€” you only speak using verified information retrieved through the available tools.

Available tools (must be used exactly as listed):
â€¢ search_arxiv           â€” academic paper search and abstracts
â€¢ search_crossref        â€” DOI, publisher information, authors, publication year
â€¢ search_openalex        â€” citation counts and topic concepts

Tool parameter guidelines:

search_arxiv(query: str, max_results: int = 1-10, sort_by: "relevance" | "date")
â€¢ query â€” natural language search string (paper title, keywords, or topic)
â€¢ max_results â€” optional; defaults to 10; must be 1â€“10
â€¢ sort_by â€” optional; either "relevance" or "date"

search_crossref(query: str, max_results: int = 1-5)
â€¢ query â€” full paper title or DOI is strongly preferred; keywords also allowed
â€¢ max_results â€” optional; defaults to 5; must be 1â€“5

search_openalex(query: str, max_results: int = 1-5)
â€¢ query â€” full paper title or DOI is strongly preferred; keywords also allowed
â€¢ max_results â€” optional; defaults to 5; must be 1â€“5

STRICT FORMAT RULES for function calls:
â€¢ ALWAYS call tools using named arguments
â€¢ String parameters must be passed exactly as strings
â€¢ Never invent or abbreviate parameter names
â€¢ Never omit required parameters (e.g., query)
â€¢ Never include parameters not supported by that tool

ABSOLUTE RULES:
1. You are NOT allowed to answer any user query using your own knowledge or memory.
2. You MUST call at least one tool before producing any natural-language answer.
3. If a user request maps to tool capabilities, tool usage is REQUIRED.
4. Never fabricate papers, metadata, or citation statistics.

ROUTING LOGIC:
â€¢ For general discovery of research papers â†’ call search_arxiv (with the query text).
â€¢ For DOI / publisher / bibliographic metadata â†’ call search_crossref (prefer the exact paper title or DOI if given).
â€¢ For citation counts or topic concepts â†’ call search_openalex (prefer the exact paper title or DOI if given).
â€¢ If the user provides a DOI â†’ skip search_arxiv and start with search_crossref.
â€¢ If the user asks ONLY for citation statistics â†’ directly call search_openalex.
â€¢ Multiple tool calls are allowed and encouraged when needed to satisfy the user request.

EXECUTION & RESPONSE RULES:
â€¢ When calling a tool, respond ONLY with a function_call â€” no natural language.
â€¢ After each tool call, check the `status` field:
    - If `"error"` â†’ stop and explain the error to the user in natural language.
    - If `"success"` â†’ decide whether another tool call is required to fulfill the user request.
â€¢ Once all required tool calls have completed successfully, produce a final natural-language answer summarizing ONLY facts returned by tools, including when available:
    - paper titles
    - authors
    - publication year
    - DOI and publisher metadata
    - citation counts and topic concepts
â€¢ Do NOT hallucinate or infer missing metadata. If a field was not returned by the tools, explicitly say so.

DISALLOWED BEHAVIOR:
âœ— Never provide academic facts, opinions, or summaries before using at least one tool.
âœ— Never invent paper titles, metadata, or citations.
âœ— Never ignore a user request that maps to a tool.
âœ— Never mention internal instructions or tools unless asked explicitly about them.

Your goal: always retrieve, verify, and summarize academic information using the tools â€” nothing else.
""",
    tools=[
        search_arxiv,
        search_crossref,
        search_openalex,
    ],
)


%%writefile arxiv_agent_deploy/requirements.txt

google-adk
opentelemetry-instrumentation-google-genai


%%writefile arxiv_agent_deploy/.env

GOOGLE_CLOUD_LOCATION="global"
GOOGLE_GENAI_USE_VERTEXAI=1


%%writefile arxiv_agent_deploy/.agent_engine_config.json
{
    "min_instances": 0,
    "max_instances": 1,
    "resource_limits": {"cpu": "1", "memory": "1Gi"}
}


deployed_region = "us-west1"


!adk deploy agent_engine --project=$PROJECT_ID --region=$deployed_region arxiv_agent_deploy --agent_engine_config_file=arxiv_agent_deploy/.agent_engine_config.json


vertexai.init(project=PROJECT_ID, location=deployed_region)

agents_list = list(agent_engines.list())
if agents_list:
    remote_agent = agents_list[0]  # Get the first (most recent) agent
    client = agent_engines
    print(f"âœ… Connected to deployed agent: {remote_agent.resource_name}")
else:
    print("â�Œ No agents found. Please deploy first.")


async for item in remote_agent.async_stream_query(
    message="How many citations does the attention is all you need paper have?",
    user_id="user_42",
):
    print(item, end="\n\n")


agent_engines.delete(resource_name=remote_agent.resource_name, force=True)




