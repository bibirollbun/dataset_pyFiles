#Movie recommender
#using multiple agents to secure movies that - have not been watched, - are sutiable to the mood

#imports
import os
import asyncio
import traceback
import uuid
import logging
from typing import List, Dict

print ("imports complete")


from google.adk.agents import Agent
from google.adk.tools import FunctionTool, AgentTool, google_search
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.genai import types

print("google imports complete")


logging.getLogger("google_genai.types").setLevel(logging.ERROR)


#API Key integration (usually done earlier in the Kaggle notebooks, but it just feels mire traditional here)
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


MODEL_NAME = "gemini-2.5-flash"


def get_genre_by_mood(mood: str) -> str:
    """
    Determines the best movie genre based on the user's current mood.

    WHY this exists as a tool:
    - This mapping is deterministic, simple, and doesnâ€™t need LLM reasoning.
    - We *delegate* this work to a tool so:
        * it's testable in isolation
        * it keeps the agent focused on high-level workflow / conversation
        * we demonstrate how agents can call tools for small, reliable operations.
    """
    
    mood_map = {
        "happy": "Action",
        "sad": "Comedy",
        "angry": "Thriller",
        "tired": "Documentary",
        "bored": "Sci-Fi",
        "romantic": "Romance",
    }
    # Using .get(..., "Drama") guarantees we always return a genre, even for
    # unrecognized moods. "Drama" is the default catch-all. 
    # this is done because Drama seems to me to be the most diverse group
    # A lot of the map is based on my own opinion, not really scienctifically backed. Could be improved later...
    return mood_map.get(mood.lower(), "Drama")

print ("mood by genre tool has been defined")


genre_tool = FunctionTool(get_genre_by_mood)


researcher_agent = Agent(
    model=Gemini(model=MODEL_NAME),  # Uses the Gemini model for reasoning and summarization.
    tools=[google_search],           # This agent can call Google Search to fetch up-to-date info.
    name="MovieResearcher",          # Name improves logs/observability (â€œwhich agent did whatâ€�).
    instruction=(
        # This system prompt tightly defines what this agent should do.
        # WHY this is verbose:
        #   - We want the model to stay in the lane of â€œmovie expert + web searchâ€�.
        #   - We specify format and constraints to make the output consistent and scannable.
        "You are a movie expert. "
        "The user or another agent will give you a target genre (e.g., Action, Comedy). "
        "Use Google Search to find 3â€“5 movies in that genre. "
        "Return a concise list in the format:\n"
        "Title (Year) â€“ 1â€“2 sentence plot summary.\n"
        "IMDB rating \n"
        "Availability on Netflix: \n"
        "If the user has asked to exclude certain movies, do not include them."
    ),
)

print("Research agent defined")


concierge_agent = Agent(
    model=Gemini(model=MODEL_NAME),  # Same base model, but driven by a different role/instructions.
    tools=[
        genre_tool,                      # Tool: mood â†’ genre
        AgentTool(agent=researcher_agent),  # Tool: delegate a sub-task to MovieResearcher
    ],
    name="MovieConcierge",
    instruction=(
        # This prompt describes the high-level instruction for the conversation.
        # WHY so step-by-step:
        #   - LLMs respond well to explicit step instructions.
        #   - It makes the agent more predictable (important for evaluation).
        "You are a helpful Movie Concierge.\n"
        "Your job:\n"
        "1. Ask the user how they are feeling (their mood) if it is not already known.\n"
        "2. Call the `get_genre_by_mood` tool to map their mood to a movie genre.\n"
        "3. Delegate to the `MovieResearcher` agent using the AgentTool to find movies.\n"
        "4. Present 3â€“5 movie options, clearly formatted and easy to scan.\n"
        "5. If the user rejects a movie (e.g., 'I have already seen that'), offer alternatives.\n"
        "6. Keep your answers short, friendly, and conversational."
    ),
)

print("Conciergy agent defined")


def extract_text_from_event(event) -> str:
    """
    Extract only user-visible text from an ADK event.

    WHY we need this:
    - The runner emits a stream of "events" that may contain:
        - text parts,
        - tool calls,
        - internal reasoning,
        - metadata.
    - We want to:
        * show only text to the user (no JSON, no raw tool calls),
        * avoid warnings from trying to stringify non-text parts.
    """
    chunks = []

    # We guard accesses carefully, because events might not have content or parts.
    if event.content and getattr(event.content, "parts", None):
        for part in event.content.parts:
            # The text parts are what we want to display.
            if hasattr(part, "text") and part.text:
                chunks.append(part.text)
            # We intentionally ignore non-text parts (e.g., function_call, thought_signature).
            # They are still logged by LoggingPlugin, but not shown as raw objects to users.
    # We join into one string, which becomes the final visible response segment for that event.
    return "".join(chunks)


runner = InMemoryRunner(
    agent=concierge_agent,
    app_name="MovieConciergeApp",   # Name is used for grouping and analytics.
    plugins=[LoggingPlugin()],      # Observability is critical when working with agents.
)

# A simple fixed user ID for the demo. In a real-world app, this would come
# from authentication / user management, allowing different users to have
# separate histories and personalized recommendations.
user_id = "user_001"


# A simple fixed user ID for the demo. In a real-world app, this would come
# from authentication / user management, allowing different users to have
# separate histories and personalized recommendations.
user_id = "user_001"


async def create_session() -> str:
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
    )
    return session.id


async def run_observability_demo():
    
    print(" Running observability demo with LoggingPlugin + run_debug \n")

    response = await runner.run_debug(
        "I'm feeling happy and a bit bored. Recommend a movie."
    )

    print("\n Debug run complete.")
    # The response object includes structured info, not just text. We show it so
    # a developer can inspect the full detail if desired.
    print("Raw response object:", response)



TEST_CASES: List[Dict] = [
    {
        "description": "Happy â†’ should map to Action or similar energetic genre",
        "user_input": "I am feeling happy today.",
        "expected_keywords": ["Action"],
    },
    {
        "description": "Sad â†’ should map to Comedy",
        "user_input": "Honestly, I feel a bit sad.",
        "expected_keywords": ["Comedy"],
    },
    {
        "description": "Bored â†’ should map to Sci-Fi",
        "user_input": "I'm super bored right now.",
        "expected_keywords": ["Sci-Fi", "science fiction"],
    },
]




async def run_basic_evaluation():
    print("\n Running basic evaluation of MovieConcierge...\n")

    total = len(TEST_CASES)
    passed = 0

    for i, tc in enumerate(TEST_CASES, start=1):
        print(f"--- Test {i}/{total}: {tc['description']} ---")

        # Each test uses a fresh session so that state from previous tests
        # doesnâ€™t influence the current one. This mimics independent user runs.
        session_id = await create_session()

        # The runner expects messages as 'types.Content' with 'parts'.
        # This extra structure is used for multi-modal input, but here it's just text.
        wrapped_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=tc["user_input"])],
        )

        final_text = ""

        # We stream events to:
        #   * support partial responses,
        #   * capture all emitted text parts.
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=wrapped_content,
        ):
            final_text += extract_text_from_event(event)

        print("Response:\n", final_text, "\n")

        # Simple keyword checking: if the expected genre appears in the response, we treat it as pass.
        # This is a crude metric but enough for a quick sanity check.
        text_lower = final_text.lower()
        hit = any(kw.lower() in text_lower for kw in tc["expected_keywords"])

        if hit:
            print(" PASS\n")
            passed += 1
        else:
            print("FAIL â€“ expected one of:", tc["expected_keywords"], "\n")

    print(f" Evaluation summary: {passed}/{total} tests passed.\n")


async def main():
    print("\n AI Movie Concierge Started")
    print("-------------------------------------------------------")
    print("Agent: Hello! I can recommend a movie based on your mood. How are you feeling today?")

    # We create one persistent session for this whole interaction so:
    #   - the agent remembers prior messages,
    #   - mood / previously suggested movies can inform future responses.
    active_session_id = await create_session()

    # demo_inputs:
    #   - Kaggle sometimes doesnâ€™t allow interactive input.
    #   - Instead of crashing, we simulate a short conversation.
    demo_inputs = ["I am feeling happy!", "I have already seen that.", "quit"]
    demo_index = 0

    while True:
        try:
            # Normal path: read from user via stdin.
            user_input = input("User: ")
        except EOFError:
            # Fallback path: if input isn't available (e.g., Kaggle), we inject
            # pre-defined demo inputs so the codeâ€™s behavior can still be observed.
            if demo_index < len(demo_inputs):
                print(f"\n Input unavailable. Running Demo Input #{demo_index + 1}...")
                user_input = demo_inputs[demo_index]
                print(f"User: {user_input}")
                demo_index += 1
            else:
                # Once we exhaust demo inputs, we end the loop gracefully.
                break

        # User can manually end the conversation.
        if user_input.lower() in ["quit", "exit"]:
            print("Agent: Goodbye! ")
            break

        print("\n--- One second please ---")

        try:
            # Wrap the user message into the structured format expected by the runner.
            wrapped_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )

            final_text = ""

            # We use run_async to:
            #   - Handle streaming (chunked responses).
            #   - Capture all events from the agent, including intermediate tool use.
            async for event in runner.run_async(
                user_id=user_id,
                session_id=active_session_id,
                new_message=wrapped_content,
            ):
                final_text += extract_text_from_event(event)

            # Sometimes an agent might produce only tool calls (no direct text).
            # We guard against showing a blank by providing a placeholder.
            if not final_text.strip():
                final_text = "(No text response from agent.)"

            print(f"Agent: {final_text}\n")
            print("Done! \n")

        except Exception as e:
            # Catch-all error handler for the loop. This ensures we see the stack trace
            # instead of silently failing (which is painful when debugging agents).
            print(f" Error: {e}")
            traceback.print_exc()
            break


if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        asyncio.create_task(main())
    else:
        asyncio.run(main())

