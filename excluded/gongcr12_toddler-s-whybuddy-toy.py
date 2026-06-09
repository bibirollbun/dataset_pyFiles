# pip install google-adk


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


import logging
import re
from typing import AsyncGenerator, Dict, Any, List

from google.adk.agents import LlmAgent, Agent, SequentialAgent, ParallelAgent, LoopAgent, BaseAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService, VertexAiMemoryBankService
from google.adk.tools import load_memory, preload_memory, AgentTool, FunctionTool, google_search, exit_loop
from google.genai import types
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

import asyncio
from concurrent.futures import ThreadPoolExecutor

# A single-thread executor for running blocking input() safely
input_executor = ThreadPoolExecutor(max_workers=1)

async def async_input(prompt: str, timeout: int = None) -> str:
    """Run input() in a thread and support asyncio timeout."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(input_executor, input, prompt),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return None  # Indicates timeout


APP_NAME = "WhysieApp"
USER_ID = "demo_user"


# This agent runs ONCE at the beginning to create the first draft.
initial_answer_writer_agent = Agent(
    name="InitialAnswerWriterAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a friendly toy answering a toddler's "why" and "what" questions.

Conversation and memory behavior:
- Use the entire chat history in this session when answering.
- Treat most questions as follow-ups in a little story, not as isolated.
- If the child uses words like "they", "it", "those", or "the green plants",
  assume they probably refer to something we just talked about.
  For example, if we said "Canada geese eat green plants" and the child asks
  "What are the green plants?", you should explain the kinds of plants geese
  like to eat (grass, clover, pond weeds), not just give a generic
  definition of green plants.

If memory from earlier sessions is provided, you may also use it to keep
answers consistent with what you told this child before, but be brief.

Answer style:
1. Use very simple, friendly English for a toddler under 4.
2. At most 6 sentences, 3â€“4 sentences preferred.
3. Gentle, kind, and age-appropriate.
4. Answer directly; do not show any reasoning or inner thoughts.

Output only the answer text you would say to the child.""",
    tools=[PreloadMemoryTool()],
    output_key="current_answer",  # Stores the first draft in the state.
)

print("âœ… initial_answer_writer_agent created.")

# This agent's only job is to provide feedback or the approval signal. It has no tools.
check_agent = Agent(
    name="CheckAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a professional children's media reviewer. Review the answer provided below.
    Answer: {current_answer}
    
    Evaluate whether current_answer is clearly safe and age-appropriate
    for children under 4:
   - Gentle, kind, non-violent.
   - No scary or disturbing content.
   - No adult topics.
   - Simple, toddler-friendly language.
   
    - If the answer is safe & appropriate, you MUST respond with the exact phrase: "APPROVED"
    - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
    tools=[PreloadMemoryTool()],
    output_key="media_check_feedback",  # Stores the feedback in the state.
)

print("âœ… check_agent created.")


# This agent refines the story based on critique OR calls the exit_loop function.
correction_agent = Agent(
    name="CorrectionAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are an answer refiner for a toddler Q&A toy.

You have:
- Answer Draft: {current_answer}
- Feedback: {media_check_feedback}

Use both the conversation so far and any loaded memories to keep things
consistent for the child.

Your task:
- If Feedback is EXACTLY "APPROVED":
    - Treat Answer Draft as the final answer text for the toddler.
    - Call the `exit_loop` function so that no further refinement is done.
    - Still output the Answer Draft as plain answer text (for the child).

- OTHERWISE:
    - Rewrite the Answer Draft to fully incorporate the Feedback.
    - Keep it consistent with earlier answers in this conversation.
    - Return the improved answer text as the final answer for the toddler.

Answer style:
- Keep it simple, gentle, and age-appropriate.
- 3â€“6 short sentences.
- No reasoning or explanation of what you are doing, just the final answer text.""",
    output_key="current_answer",
    tools=[
        FunctionTool(exit_loop),
        PreloadMemoryTool(),
    ],  # The tool is now correctly initialized with the function reference.
)

print("âœ… correction_agent created.")

# The LoopAgent contains the agents that will run repeatedly: Critic -> Refiner.
answer_refinement_loop = LoopAgent(
    name="AnswerRefinementLoop",
    sub_agents=[check_agent, correction_agent],
    max_iterations=3,  # Prevents infinite loops
)

memory_service = (
    InMemoryMemoryService()
)  # ADK's built-in Memory Service for development and testing

# Create Session Service
session_service = InMemorySessionService()  # Handles conversations

# The root agent is a SequentialAgent that defines the overall workflow: Initial Write -> Refinement Loop.
root_agent = SequentialAgent(
    name="StoryPipeline",
    sub_agents=[initial_answer_writer_agent, answer_refinement_loop],
)

print("âœ… Loop and Sequential Agents created.")
print("âœ… Session added to memory!")
# Create runner with BOTH services
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,  # Memory service is now available!
)

print("âœ… Agent and Runner created with memory support!")


# -------------------------------------------------------------------
# Helpers for sessions & single turns
# -------------------------------------------------------------------

async def get_or_create_session(
    session_service: InMemorySessionService,
    app_name: str,
    user_id: str,
    session_id: str,
):
    """Create a new session or return an existing one."""
    try:
        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        session = await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
    return session


async def run_one_turn(
    runner_instance: Runner,
    session,
    user_query: str,
) -> str:
    """
    Run the full answer pipeline once for a single toddler question.
    Returns the final answer text.
    """
    # We no longer print the question here, to avoid double "Toddler >" lines.
    query_content = types.Content(
        role="user",
        parts=[types.Part(text=user_query)],
    )

    final_text = None

    # Stream agent response
    async for event in runner_instance.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=query_content,
    ):
        if not (event.is_final_response() and event.content and event.content.parts):
            continue

        # Collect all text parts (ignoring tool/function_call parts)
        text_parts = [
            getattr(p, "text", "")
            for p in event.content.parts
            if getattr(p, "text", None)
        ]
        candidate = " ".join(text_parts).strip()

        if not candidate:
            continue

        # Ignore pure "APPROVED" messages from CheckAgent
        if candidate.strip().upper() == "APPROVED":
            continue

        # Otherwise, treat this as a potential final answer.
        final_text = candidate

    if final_text:
        print(f"Toy     > {final_text}")

    return final_text or ""


# -------------------------------------------------------------------
# Long-running toddler chat loop (Human-in-the-Loop)
# -------------------------------------------------------------------

async def toddler_chat_loop(
    runner_instance: Runner,
    session_service: InMemorySessionService,
    app_name: str,
    user_id: str,
    session_id: str = "conversation-02",
):
    """
    Long-running loop:
    - Reuses the same session_id
    - Each user input runs through the full answer pipeline
    - Stops when toddler says "bye"/"bye bye"/"goodbye"/"stop"
    """
    session = await get_or_create_session(
        session_service=session_service,
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    print(f"\n### Starting toddler chat session: {session_id}")
    print("Say 'bye' when you want to stop.\n")

    while True:
        # In a real app, replace input() with your UI / API layer.
        # Wait up to 3 minutes (180 seconds) for toddler input
        user_query = await async_input("Toddler > ", timeout=180)
        # ---- TIMEOUT CONDITION ----
        if user_query is None:  
            goodbye = "Bye-bye! ðŸ‘‹ We stopped because nobody talked for a while."
            print(f"Toy     > {goodbye}")
            break

        user_query = user_query.strip()
        
        if not user_query:
            continue
        lower = user_query.lower()
        words = lower.split()
        if "bye" in lower and len(words) <= 4:
            goodbye = "Bye-bye! ðŸ‘‹ See you next time!"
            print(f"Toy     > {goodbye}")
            break

        await run_one_turn(
            runner_instance=runner_instance,
            session=session,
            user_query=user_query,
        )

    # Optional: after the chat ends, store session into memory for parent review
    await memory_service.add_session_to_memory(session)
    print("âœ… Session stored to memory for parent review.")



# -------------------------------------------------------------------
# Parent review log
# -------------------------------------------------------------------

async def get_parent_review_log(
    session_service: InMemorySessionService,
    app_name: str,
    user_id: str,
):
    """
    Return all conversations (all sessions) for a given user_id in a simple,
    parent-readable format.

    Output format:
    [
        {
            "session_id": "...",
            "messages": [
                {"role": "user", "text": "..."},
                {"role": "toy",  "text": "..."},
                ...
            ]
        },
        ...
    ]
    """

    # 1. Get list of sessions for this user
    list_resp = await session_service.list_sessions(
        app_name=app_name,
        user_id=user_id,
    )

    sessions = list_resp.sessions or []
    parent_log: list[dict] = []

    # 2. For each session metadata, load the full session (with events)
    for session_meta in sessions:
        session = await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_meta.id,
        )

        session_data = {"session_id": session.id, "messages": []}
        messages = session_data["messages"]

        # 3. Walk through events and extract user/assistant text
        for event in session.events:
            if not getattr(event, "content", None) or not event.content.parts:
                continue

            author = getattr(event, "author", None)

            # For parent view:
            # - "user" stays "user"
            # - everything else we treat as "toy" (the AI toy / app)
            if isinstance(author, str) and author == "user":
                role = "user"
            else:
                role = "toy"

            # collect only text parts (ignore tool/function_call parts)
            text_parts = [
                getattr(p, "text", "")
                for p in event.content.parts
                if getattr(p, "text", None)
            ]
            message_text = " ".join(text_parts).strip()
            if not message_text:
                continue

            # ---- FILTERING LAYER ----

            # 1) Hide safety verdicts like "APPROVED"
            if role == "toy" and message_text.strip().upper() == "APPROVED":
                continue

            # 2) Deduplicate consecutive identical toy messages
            if (
                role == "toy"
                and messages
                and messages[-1]["role"] == "toy"
                and messages[-1]["text"] == message_text
            ):
                # Skip exact duplicate toy output
                continue

            messages.append({"role": role, "text": message_text})

        parent_log.append(session_data)

    return parent_log



# -------------------------------------------------------------------
# Example usage (in an async-capable environment like notebook)
# -------------------------------------------------------------------

# 1) Run the toddler chat loop interactively:

await toddler_chat_loop(
    runner_instance=runner,
    session_service=session_service,
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id="conversation-02",
)




# 2) Later, as a parent, fetch the review log:

from pprint import pprint
log = await get_parent_review_log(
    session_service=session_service,
    app_name=APP_NAME,
    user_id=USER_ID,
)
pprint(log)




