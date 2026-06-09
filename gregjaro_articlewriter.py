%pip install google-adk arxiv


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("GOOGLE_API_KEY key setup complete.")
except Exception as e:
    print(f"Authentication Error")


from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.tools import FunctionTool, google_search
from typing import Any, Dict, List
import time
import uuid
import arxiv
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types
import logging
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import LoggingPlugin
import warnings

warnings.filterwarnings("ignore")


class ObservabilityPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="observability_plugin")
        self.agent_calls = 0
        self.model_calls = 0
        self.tool_calls = 0
        self.errors = 0
        self.trace_id = uuid.uuid4().hex

    async def before_agent_callback(self, *, agent, callback_context):
        self.agent_calls += 1
        callback_context.state["agent_start_time"] = time.time()
        logging.info(f"[TRACE {self.trace_id}] AGENT_START: {agent.name} (total={self.agent_calls})")

    async def after_agent_callback(self, *, agent, callback_context):
        start = callback_context.state.get("agent_start_time")
        elapsed = time.time() - start if start else -1
        logging.info(f"[TRACE {self.trace_id}] AGENT_END: {agent.name} (elapsed={elapsed:.3f}s)")

    async def before_model_callback(self, *, callback_context, llm_request):
        self.model_calls += 1
        callback_context.state["model_start"] = time.time()
        logging.info(f"[TRACE {self.trace_id}] MODEL_START: model={llm_request.model}")

    async def after_model_callback(self, *, callback_context, llm_response):
        start = callback_context.state.get("model_start")
        elapsed = time.time() - start if start else -1
        logging.info(f"[TRACE {self.trace_id}] MODEL_END: elapsed={elapsed:.3f}s")


    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        self.errors += 1
        logging.error(f"[TRACE {self.trace_id}] MODEL_ERROR: {type(error).__name__}: {error}")

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        self.tool_calls += 1
        tool_context.state["tool_start"] = time.time()
        logging.info(f"[TRACE {self.trace_id}] TOOL_START: {tool.name} (call #{self.tool_calls})")

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        start = tool_context.state.get("tool_start")
        elapsed = time.time() - start if start else -1
        logging.info(f"[TRACE {self.trace_id}] TOOL_END: {tool.name} (elapsed={elapsed:.3f}s)")


retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


def arxiv_search(query: str) -> dict:
    """Returns top 5 arXiv papers."""
    if not query:
        return {"papers": []}
    try:
        results = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = []
        for r in results.results():
            papers.append({"title": r.title, "summary": r.summary})
        return {"papers": papers}
    except Exception:
        return {"papers": [{"title": "Mock Paper", "summary": "Arxiv fetch failed."}]}


def edit_article(article_title: str, arxiv_papers: list[dict], google_data: list[str]) -> dict:
    """
    Generate a more engaging draft scientific article by synthesizing sources into
    narrative form, adding examples, and simplifying technical explanations.
    """
    text = f"=== ARTICLE: {article_title} ===\n\n"

    # Introduction with hook
    text += "## Introduction\n"
    text += (
        f"{article_title} is a fascinating field at the intersection of science and technology. "
        "We will explore its principles, recent discoveries, and practical applications, highlighting insights from both the web and scientific research.\n\n"
    )

    # Insights from Google – make it narrative
    text += "## Insights from Google\n"
    if google_data:
        for i, snippet in enumerate(google_data, start=1):
            snippet_short = snippet if len(snippet) < 3000 else snippet[:3000] + "..."
            text += f"- {snippet_short}\n"
        text += "\nThese findings highlight the growing interest and practical impact of the topic in the wider world.\n\n"
    else:
        text += "No Google search data available.\n\n"

    # Insights from ArXiv – summarize & explain significance
    text += "## Insights from ArXiv\n"
    if arxiv_papers:
        for i, p in enumerate(arxiv_papers, start=1):
            title = p.get("title", "Unknown")
            summary = p.get("summary", "")
            summary_short = summary if len(summary) < 3000 else summary[:3000] + "..."
            text += (
                f"{i}. **{title}**\n"
                f"   Key ideas: {summary_short}\n"
                "   Significance: This research advances our understanding of how quantum computing can solve complex problems, and may inspire new applications.\n\n"
            )
    else:
        text += "No ArXiv papers available.\n\n"

    # Conclusion – engage reader
    text += "## Conclusion\n"
    text += (
        "By combining general web insights with cutting-edge scientific research, "
        f"we get a richer picture of {article_title}. The field continues to evolve rapidly, "
        "and ongoing studies promise exciting developments that could transform technology, industry, and society.\n"
    )

    return {"article_text": text}


google_search_agent = LlmAgent(
    name="GoogleSearchAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are a generic web researcher.

    Your goal is to search specifically on requested domains using the 'site:' operator.

    INSTRUCTIONS:
    1. Read 'user_topic' and 'site_filters' from context.
    2. Construct a SINGLE query string.
       - IF site_filters are provided (e.g. "wikipedia, medium"), append them to the topic using OR and site operator.
       - Example: "Quantum computing site:wikipedia.org OR site:medium.com"
    3. Call the google_search tool with this single 'query' string.
       - DO NOT pass 'site_filters' as a separate argument to the tool.
    4. Return a list of strings (summaries).
    """,
    tools=[google_search],
    output_key="google_data"
)

arxiv_search_agent = LlmAgent(
    name="ArxivSearchAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are a scientific researcher using ArXiv.

    INPUT HANDLING RULES:
    1. You will receive an input containing 'user_topic' and potentially 'site_filters'.
    2. **CRITICAL:** 'site_filters' are intended for the Web Search Agent, NOT for you.
    3. **IGNORE** any instructions regarding specific websites (like wikipedia, medium, etc.).
    4. Your ONLY task is to extract the 'user_topic' and search for it on ArXiv.

    ACTION:
    - Call `arxiv_search` with query=user_topic.
    - For each paper, include the following fields:
        1. "title" - full title of the paper
        2. "summary" - abstract or short description
        3. "key_points" - 3–5 bullet points summarizing the main contributions or findings
    - Ensure the summaries and key_points are concise but informative, suitable for drafting a scientific article.
    """,
    tools=[FunctionTool(arxiv_search)],
    output_key="arxiv_data"
)

research_squad = ParallelAgent(
    name="ResearchSquad",
    sub_agents=[google_search_agent, arxiv_search_agent],
)


article_writer_agent = LlmAgent(
    name="ArticleWriterAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are an Article Editor. Your task is to create a draft scientific article based on:

    1. `user_topic` (the article title)
    2. `google_data` (list of web snippets)
    3. `arxiv_data` (list of papers with title and summary)

    Steps:
    - Call the function `edit_article(article_title, arxiv_papers, google_data)`.
    - Map the inputs:
        - article_title -> user_topic
        - arxiv_papers -> arxiv_data
        - google_data -> google_data
    - Return the 'article_text' field from the function's output.
    - Format the article with sections: Introduction, Insights from Google, Insights from ArXiv, Conclusion.
    - Turn raw data into engaging, human-readable text.
    - Add examples, metaphors, or analogies where appropriate.
    - Avoid just listing source snippets; synthesize them into coherent paragraphs.
    - Ensure smooth transitions between sections and maintain reader interest.
    """,
    tools=[FunctionTool(edit_article)],
    output_key="article_text"
)

pipeline = SequentialAgent(
    name="ArticlePipeline",
    sub_agents=[research_squad, article_writer_agent]
)

runner = InMemoryRunner(
    app_name="agents",
    agent=pipeline,
    plugins=[LoggingPlugin(), ObservabilityPlugin()]
)


from ipykernel.kernelbase import StdinNotImplementedError

try:
    user_topic = input("Enter Article Title (e.g. Quasicrystal): ") or "Quasicrystal"
    user_filters = input("Enter preferred domains (comma separated): ") or "nature.com, science.org"
except StdinNotImplementedError:
    # Fallback for Kaggle commit
    user_topic = "Quantum computing"
    user_filters = "wikipedia.com, medium.com,"

print(f"Configuration set: Topic='{user_topic}', Filters='{user_filters}'")


prompt = f"""START WORKFLOW.
Context Variables:
user_topic: {user_topic}
site_filters: {user_filters}"""

response = await runner.run_debug(prompt)

print(response)


for event in response:
    if event.content and hasattr(event.content, "parts"):
        for part in event.content.parts:
            if hasattr(part, "function_response") and part.function_response:
                if part.function_response.name == "edit_article":
                    article_text = part.function_response.response.get("article_text")
                    print(article_text)

