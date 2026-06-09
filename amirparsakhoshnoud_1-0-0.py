!pip install google-adk --quiet

import asyncio
import json
import uuid
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search, load_memory
from google.genai import types



import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


APP_NAME = "adk_factchecker_notebook"
MODEL = "gemini-2.0-flash"        # Required for google_search tool
USER_ID = "notebook_user"

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()



def create_search_agent():
    instruction = """
You specialize in web fact-checking. Use the google_search tool to look up the claim.
Return ONLY a JSON array (no extra text). Up to 6 items, each with:

{
  "title": "...",
  "snippet": "...",
  "url": "...",
  "site": "...",
  "published": "..."
}
"""

    return LlmAgent(
        name="SearchAgent",
        model=MODEL,
        instruction=instruction,
        tools=[google_search],
    )

def create_analyzer_agent():
    instruction = """
You are a fact-check analyst. Input includes:
1. Claim text
2. JSON array of search results

Return ONLY a JSON object:
{
  "verdict": "likely true | likely false | insufficient evidence",
  "confidence": 0-100,
  "reasons": ["...", "..."],
  "top_sources": [{"url": "...", "label": "news|fact-check|blog|unknown"}]
}
"""

    return LlmAgent(
        name="AnalyzerAgent",
        model=MODEL,
        instruction=instruction,
    )

search_agent = create_search_agent()
analyzer_agent = create_analyzer_agent()



async def run_agent(agent, user_text: str):
    session_id = str(uuid.uuid4())
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id
    )

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service
    )

    content = types.Content(role="user", parts=[types.Part(text=user_text)])
    events = runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content
    )

    final_text = ""
    async for ev in events:
        if ev.is_final_response():
            final_text = ev.content.parts[0].text
            break

    return final_text.strip()



async def fact_check_claim(claim: str):
    # 1. Run SearchAgent
    search_prompt = f"Claim: {claim}\nReturn JSON only."
    raw_search = await run_agent(search_agent, search_prompt)

    try:
        search_results = json.loads(raw_search)
        if not isinstance(search_results, list):
            raise ValueError("Expected array")
    except:
        search_results = [{
            "title": "raw_output",
            "snippet": raw_search,
            "url": "",
            "site": "",
            "published": ""
        }]

    # 2. Run AnalyzerAgent
    analyzer_prompt = (
        f"Claim: {claim}\n\n"
        f"SearchResults: {json.dumps(search_results, ensure_ascii=False)}\n\n"
        "Return JSON only."
    )
    raw_analysis = await run_agent(analyzer_agent, analyzer_prompt)

    try:
        verdict = json.loads(raw_analysis)
    except:
        verdict = {
            "verdict": "insufficient evidence",
            "confidence": 30,
            "reasons": ["Analyzer did not return valid JSON"],
            "top_sources": []
        }

    return {
        "claim": claim,
        "search_results": search_results,
        "verdict": verdict
    }



claim = "The UK government announced a nationwide ban on smartphones on trains starting next month."

result = await fact_check_claim(claim)
result



from pprint import pprint
pprint(result)


