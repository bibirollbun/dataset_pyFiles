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



APP_NAME = 'movie_rec_app'
USER_ID = 'test_user_123'
MODEL_NAME = 'gemini-2.5-flash-lite'


from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.tools.transfer_to_agent_tool import transfer_to_agent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams

from google.genai import types


class ParsedQuery(BaseModel):
  raw_query: str = Field(description = "Original user query")
  query_type: Literal[
      "title_keyword",
      "non_title_keyword",
      "image",
  ] = Field(description = "How the user expressed the query")

  title_keywords: List[str] = Field(default_factory = list)
  content_types: List[str] = Field(default_factory = list)
  genres: List[str] = Field(default_factory = list)
  topics: List[str] = Field(default_factory = list)
  themes: List[str] = Field(default_factory = list)
  cast: List[str] = Field(default_factory = list)
  crew: List[str] = Field(default_factory = list)
  languages: List[str] = Field(default_factory = list)

  include_kids_safe: Optional[bool] = None
  minimum_rating: Optional[float] = None
  freshness_pref: Optional[Literal["classic", "recent", "any"]] = None
  popularity_pref: Optional[Literal["popular", "niche", "any"]] = None

  has_image: bool = False



class MovieCandidate(BaseModel):
  imdb_id: str
  title: str
  year: Optional[int] = None
  genres: List[str] = Field(default_factory=list)
  score: float = 0.0
  rerank_score: float = 0.0
  explanation: Optional[str] = None



class RankResult(BaseModel):
  parsed_query: ParsedQuery
  candidates: List[MovieCandidate]


class FinalAnswer(BaseModel):
  movies: List[MovieCandidate]
  reasoning: str
  coverage_note: Optional[str] = None


google_search_tool = GoogleSearchTool()


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


query_understanding_agent = Agent(
  name = "query_understanding_agent",
  model = Gemini(
      model = MODEL_NAME,
      retry_config = retry_config
  ),
  description = (
    "Parse user movie queries (keywords, natural language, or image) into"
    "a normalized ParsedQuery object capturing facets and intent."
  ),
  include_contents = "none",
  instruction = (
    "You are a query understanding component fro a movie-recommendation system.\n"
    "Input: a single user message describing what movies they want.\n\n"
    "Output: a JSON object strictly matching the ParsedQuery schema.\n"
    "- Detect whether the user provided keywords, natural language, or an image.\n"
    "- Extract possible title keywords, genres, topics/themes, cast&crew names, "
    "  languages, maturity/kid-friendliness, rating thresholds, and recency preferences.\n"
    "- Set query_type to 'image' if the user attached a still frame or cover art.\n"
    "- Set has_image=true when an image is present.\n"
    "Do NOT recommend movies here; only parse and normalize the request."
  ),
  output_schema = ParsedQuery,
  output_key = "parsed_query",
)



catalog_retrieval_agent = Agent(
    name = "catalog_retrieval_agent",
    model = MODEL_NAME,
    description = (
        "Uses Google search to retrieve a candidate set of movies "
        "that match the ParsedQuery."
    ),
    instruction = (
        "You are the retrieval layer for a movie recommender system using Google search.\n\n"
        "You recieve the user's ParsedQuery in session.state['parsed_query'].\n"
        "Goals:\n"
        "1. Use 'google_search_tool' to retrieve ~20 candidates using:\n"
        "   - keyword / title search\n"
        "   - genre/topic/theme filters\n"
        "   - cast&crew filters\n"
        "   - language filters\n"
        "2. If parsed_query.has_image is true, use your image understanding capability to identify:\n"
        "   - possible movie titles or franchises from the frame/cover art\n"
        "   - visual attributes (era, tone, color palette, animation vs live-action)\n"
        "   Then feed those as extra constraints into Google search.\n"
        "3. Write the final candidate list (each with title, year, cast&crew, genres and "
        "   a retrieval score) as JSON object to the output.\n"
        "4. If nothing is found, explain why in a short note in the JSON object.\n"
        "Do NOT craft final user-facing recommendations here; just return structured candidates."
    ),
    tools = [google_search_tool],
    #output_schema = RetrievalResult,
    output_key = "retrieval_result",
)



rerank_agent = Agent(
    name = "rerank_agent",
    model = MODEL_NAME,
    description = (
        "Rerank candidate titles using query-item semantic match, diversity, "
        "and other business constraints such as popularity and recency of titles."
    ),
    include_contents = "none",
    instruction = (
        "You are a reranking layer of a movie recommender.\n\n"
        "Inputs:\n"
        "-  session.state['retrieval_result']: str with candidate.\n"
        "-  session.state['parsed_query']: ParsedQuery.\n\n"
        "Tasks:\n"
        "1. Re-score candidates using a mixture of\n"
        "   - semantic relevance to parsed_query.raw_query\n"
        "   - alignment with inferred facets (genres, topics, themes, cast&crew, languages)\n"
        "   - personalization hints if present in session.state['user:movie_prefs']\n"
        "   - favoring more popular and recent titles when other factors are similar\n"
        "   - minimal diversity (avoid multiple near-identical sequels unless explicitly requested).\n"
        "2. Produce a JSON object matching RankResult, but with 'rerank_score' filled "
        "   for each candidate and 'candidates' sorted descending by rerank_score.\n"
        #"3. Do NOT drop all candidates unless none were retrieved.\n"
        "3. Do NOT generate new movies that are not in the retrieval candidate list."
    ),
    output_schema = RankResult,
    output_key = "reranked_result",
)


answer_agent = Agent(
    name = 'answer_agent',
    model = "gemini-2.5-flash-lite",
    description =(
        "Turns the reranked result into a user-facing recommendation list. "
        "If titles are clearly not matching request, return an empty list and provide a clear explanation as to why."
    ),
    instruction = (
        "You are the final user-facing movie recommendation agent.\n\n"
        "Inputs:\n"
        "-  session.state['parsed_query']\n"
        "-  session.state['reranked_result']\n"
        "Tasks:\n"
        "1. Inspect parsed_query.raw_query for any *specific* titles or franchises "
        "   the user explicitly names.\n"
        "   - If those titles are missing from reranked_result.candidates, use tools to:\n"
        "     a) confirm they cannot be found.\n"
        "     b) If still not available, acknowledge it and suggest closest titles.\n"
        "2. Select the top 5 reranked candidates as recommendations.\n"
        "3. For each recommended movie, populate 'explanation' with a short justification:\n"
        "   - Which aspects of the query it matches (genre, mood, cast, theme, visual style).\n"
        "4. Fill the FinalAnswer JSON:\n"
        "   - 'movies': the selected candidates with filled explanations.\n"
        "   - 'reasoning': 2-4 sentences summarizing how you matched the user's request.\n"
        "   - 'coverage_note': explain any gaps (e.g., requested titles not found, "
        "     used web search to approximate, or unable to fully satisfy a request).\n"
        "5. Never halluninate fake movies; if a movie does not exist, only mention "
        "   it textually in reasoning/coverage_note.\n"
        "Respond ONLY with JSON conforming to FinalAnswer."
    ),
    #tools = [
    #    google_search_tool,
    #],
    output_schema = FinalAnswer,
    output_key = 'final_answer',
)


root_agent = SequentialAgent(
    name = 'movie_rec_pipeline',
    description = "This is the orchestrator agent for a movie recommendation system.",
    sub_agents = [
        query_understanding_agent,
        catalog_retrieval_agent,
        rerank_agent,
        answer_agent,
    ]
)



# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


session_service = InMemorySessionService()

runner = Runner(
    agent = root_agent,
    app_name = APP_NAME,
    session_service = session_service,
)



await run_session(
    runner,
    [
        "I want movies similar to Rio 1 and 2.",

    ],
    'a-newuser-session',
)


await run_session(
    runner,
    [
        "I want recent popular fun and visually stunning movies for family movie night.\n",
        "Narrow down to only non-animated movies.\n",
        "Can you recommend sci-fi movies like Interstellar instead? They sitll need to be fun and visually stunning.\n",
        #"I want movies similar to Matilda the musical.",
        #"I want movies similar to Kpop Demon Hunters.",
        "What kinds of movies did I request initially? Which movie was ranked lowest in your recommendation?\n",
    ],
    'stateful-agentic-session',
)




