from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

import os
os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")


# Summarizer Tool (simple LLM tool)
def summarize_text_tool(text: str) -> str:
    if len(text) < 100:
        return text
    return text[:500] + "..."

# PII redactor example
def redact_pii_tool(text: str) -> str:
    return text.replace("@", "[redacted]").replace("+91", "[redacted]")


market_research_agent = Agent(
    name="market_research_agent",
    model="gemini-2.0-flash",
    description="Analyzes market trends, competitors, and industry insights.",
    instruction="Use web search heavily. Summarize results. Focus on trends and competitors.",
    tools=[google_search]
)


persona_builder_agent = Agent(
    name="persona_builder_agent",
    model="gemini-2.0-flash",
    description="Generates audience personas from market insights.",
    instruction="Build clear ICPs, demographic profiles, motivations, and purchasing behavior."
)


content_agent = Agent(
    name="content_agent",
    model="gemini-2.0-flash",
    description="Creates inbound and outbound marketing messages.",
    instruction="Write crisp emails, hooks, CTAs, and ad copy. Use insights from personas."
)


strategy_agent = Agent(
    name="strategy_decision",
    model="gemini-2.0-flash",
    description="Produce final marketing strategy",
    instruction="Combine all agents outputs and produce a unified, clear, actionable marketing strategy.",
    tools=[]
)


decision_agent = Agent(
    name="strategy_decision_agent",
    model="gemini-2.0-flash",
    description="Combines inputs from all agents and finalizes a marketing plan.",
    instruction="Read session memory. Produce a final structured strategy."
)


security_agent = Agent(
    name="security_agent",
    model="gemini-2.0-flash",
    description="Redacts sensitive info and ensures compliance.",
    instruction="Remove PII. Check text for safety issues. Keep output clean."
)


marketing_workflow = SequentialAgent(
    name="marketing_workflow",
    description="Full agentic pipeline for marketing strategy generation.",
    sub_agents=[
        market_research_agent,
        persona_builder_agent,
        content_agent,
        decision_agent,
        security_agent
    ]
)


APP_NAME = "marketing_system"
USER_ID = "random_user"
SESSION_ID = "marketing_session_01"

async def init_session():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    runner = Runner(agent=marketing_workflow,
                    app_name=APP_NAME,
                    session_service=session_service)
    return session, runner


import os
import re

def save_markdown_file(query: str, content: str, folder="outputs"):
    os.makedirs(folder, exist_ok=True)

    # Sanitize filename (remove special characters & spaces)
    safe_query = re.sub(r'[^a-zA-Z0-9_-]', '_', query.strip())

    existing = [
        f for f in os.listdir(folder)
        if f.startswith(safe_query) and f.endswith(".md")
    ]

    index = len(existing)
    filename = f"{safe_query}_{index}.md"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


from IPython.display import display, Markdown

async def run_marketing_system(topic):
    session, runner = await init_session()

    message = types.Content(
        role="user",
        parts=[types.Part(text=f"Generate a marketing strategy for: {topic}")]
    )

    events = runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message
    )

    final_output = ""
    async for event in events:
        if event.is_final_response():
            final_output = event.content.parts[0].text
            break

    # ✅ Display nicely as Markdown instead of plain print
    print("\n=== Rendered Marketing Strategy (Markdown View) ===\n")
    display(Markdown(final_output))

    # ✅ Save to file
    file_path = save_markdown_file(topic, final_output)

    print(f"\n✅ Markdown saved as: {file_path}")



query = "Build the world’s best NGO dedicated to improving children’s lives"


await run_marketing_system(query)


import os
import re

def sanitize_filename(text: str):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text.strip())


def list_query_versions(query, folder="outputs"):
    safe_query = sanitize_filename(query)

    if not os.path.exists(folder):
        print("❌ No output folder found.")
        return []

    files = [
        f for f in os.listdir(folder)
        if f.startswith(safe_query + "_") and f.endswith(".md")
    ]

    if not files:
        print(f"❌ No files found for query: {query}")
        return []

    # Sort by index number correctly
    files_sorted = sorted(files, key=lambda x: int(x.split("_")[-1].replace(".md", "")))

    print(f"\n✅ Available versions for '{query}':\n")
    for f in files_sorted:
        idx = f.split("_")[-1].replace(".md", "")
        print(f"- Version {idx}")

    return files_sorted


def show_markdown_by_query(query, index, folder="outputs"):
    safe_query = sanitize_filename(query)
    filename = f"{safe_query}_{index}.md"
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"\n✅ Displaying file: {filepath}\n")
    display(Markdown(content))


available = list_query_versions(query)
index = input("Enter version index (0, 1, 2, ...): ")

show_markdown_by_query(query, index)

