#pip install google-adk


#pip install google.adk.session



import os
import asyncio

from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… GOOGLE_API_KEY set successfully.")
except Exception as e:
    print("â�Œ Could not load GOOGLE_API_KEY from Kaggle secrets.")
    print("   Go to: Add-ons â†’ Secrets â†’ add key 'GOOGLE_API_KEY'")
    raise e

# Core ADK imports (agents, models, sessions, runner, app)
from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner

from google.adk.sessions import InMemorySessionService, DatabaseSessionService 

from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig 

from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.code_executors import BuiltInCodeExecutor
from google.genai import types


print("âœ… ADK imports loaded. Weâ€™re good to go ğŸš€")



def extract_json_block(text: str) -> str:
    """
    Pull out the first JSON-looking block from a model response.
    Handles cases like:
    ```json
    { ... }
    ```
    or plain `{ ... }`.
    """
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("{") or candidate.startswith("["):
                return candidate
    # If there are no fences, assume the whole thing is JSON-ish
    return text


async def call_agent_and_get_text(agent: LlmAgent, prompt: str) -> str:
    """
    Convenience helper:
    - Send one prompt to an LlmAgent
    - Collect the final text response
    - Return it as a single string
    """
    runner = InMemoryRunner(agent=agent)
    content = types.Content(parts=[types.Part(text=prompt)])

    chunks: List[str] = []
    async for event in runner.run_async(
        user_id="builder",
        session_id="builder-session",
        new_message=content,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    chunks.append(part.text)

    return "".join(chunks).strip()


async def run_chat_turn(
    runner: Runner,
    user_id: str,
    session_id: str,
    user_text: str,
) -> None:
    """
    Helper to send a single message to the Game Master and print the reply.
    """
    print(f"\nğŸ§‘ You: {user_text}\n")
    content = types.Content(parts=[types.Part(text=user_text)])

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content:
            print("ğŸ¤– Game Master:")
            print("-" * 60)
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text)
            print("-" * 60)





def normalize_answer(answer: str, mode: str) -> str:
    """
    Normalizes answers so checking is fair.
    We can decide per puzzle whether to:
    - lowercase
    - just strip
    - or require exact match
    """
    answer = answer.strip()
    mode = (mode or "").lower()

    if mode == "lowercase":
        return answer.lower()
    elif mode == "strip":
        return answer
    elif mode == "exact":
        return answer
    else:
        # Reasonable default: lowercased + stripped
        return answer.lower()


from google.genai import types

# This tells Gemini how to retry on transient errors (429, 500, etc.)
retry_config = types.HttpRetryOptions(
    attempts=5,           # how many times to retry
    exp_base=7,           # backoff multiplier
    initial_delay=1,      # seconds before first retry
    http_status_codes=[   # which errors should trigger retries
        429,  # too many requests
        500,  # internal server error
        503,  # service unavailable
        504,  # gateway timeout
    ],
)

print("âœ… retry_config defined and ready to use.")


# a) Story / outline agent: builds the story + puzzle skeleton
story_agent = LlmAgent(
    name="escape_story_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Designs structured escape-room stories with puzzle outlines.",
    instruction=textwrap.dedent(
        """
        You are a game designer who builds text-only escape rooms.
        
        The user will give you:
        - a theme
        - a difficulty level (easy, medium, hard)
        - a number of puzzles
        
        You MUST respond with a single valid JSON object (no extra text), with this schema:
        
        {
          "title": "short catchy title",
          "setting": "1-2 paragraph description of the location and atmosphere",
          "puzzles": [
            {
              "id": "P1",
              "role": "opening|gate|mechanism|final",
              "recommended_type": "riddle|code|logic|pattern|word",
              "notes": "short natural language notes for what this puzzle is about"
            },
            ...
          ]
        }
        
        Rules:
        - Number of puzzles in "puzzles" MUST exactly match the requested number.
        - Puzzle IDs must be unique: P1, P2, ...
        - NO markdown, NO comments, just JSON.
        """
    ),
)

print("âœ… story_agent ready.")


# b) Puzzle agent: turns outline entries into fully specified puzzles
puzzle_agent = LlmAgent(
    name="escape_puzzle_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Expands puzzle outlines into full puzzle definitions.",
    instruction=textwrap.dedent(
        """
        You are a puzzle designer for escape rooms.
        
        Input:
        - An outline JSON describing the escape room (title, setting, puzzles skeleton)
        - A specific puzzle ID to expand
        
        You MUST respond with a single JSON object describing ONE puzzle:
        
        {
          "id": "P1",
          "type": "riddle|code|logic|pattern|word",
          "prompt": "What the player sees / the puzzle text.",
          "answer": "the correct answer as a short word/phrase/number",
          "answer_normalization": "lowercase|strip|exact",
          "notes": "internal notes for GM, optional"
        }
        
        Requirements:
        - The puzzle MUST be solvable from the prompt alone, no external data or web.
        - The answer must be clear and unambiguous.
        - Riddles/word puzzles: prefer short answers ("ORION", "ECLIPSE").
        - Numeric/logic puzzles: answer is a single integer or short code.
        - No markdown, only valid JSON.
        """
    ),
)

print("âœ… puzzle_agent ready.")


# c) Hint agent: builds 3-level hints for each puzzle
hint_agent = LlmAgent(
    name="escape_hint_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Generates tiered hints for a given escape-room puzzle.",
    instruction=textwrap.dedent(
        """
        You are a hint writer for escape-room puzzles.
        
        Given a puzzle JSON (id, type, prompt, answer, notes), you must output
        a JSON object with exactly three hints of increasing strength:
        
        {
          "id": "P1",
          "hints": [
            "very subtle nudge, no spoilers",
            "stronger direction pointing toward the mechanism",
            "almost reveals the answer but not the exact word/number"
          ]
        }
        
        Rules:
        - Hints must clearly relate to the puzzle prompt.
        - Do NOT include the exact answer string inside any hint.
        - No markdown, only valid JSON.
        """
    ),
)

print("âœ… hint_agent ready.")





import re
import json
import textwrap
from typing import List, Dict, Any
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.code_executors import BuiltInCodeExecutor
from google.genai import types


# Helper: JSON extractor


def extract_json_block(text: str) -> str:
    """
    Extracts the JSON object from a model response.
    Handles:
      ```json
      { ... }
      ```
      ``` 
      { ... }
      ```
    or just raw { ... } text.
    """

    # 1) ```json ... ``` fenced block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) Any ``` ... ``` fenced block
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3) Fallback: from first '{' to last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    # 4) Last resort: return stripped text
    return text.strip()


# Helper: call an agent and get its final text reply


async def call_agent_and_get_text(agent: LlmAgent, prompt: str) -> str:
    """
    Fire a single prompt at an agent and get its final text reply.
    Uses runner.run_debug(), so we don't manage sessions manually.
    """
    runner = InMemoryRunner(agent=agent)

    events = await runner.run_debug(prompt)

    chunks: List[str] = []
    for event in events:
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                txt = getattr(part, "text", None)
                if txt:
                    chunks.append(txt)

    return "".join(chunks).strip()



# Helper: auto-fix broken puzzle JSON 

def _strip_notes_value(bad_json: str) -> str:
    """
    Very targeted fixer:
    - Finds the "notes": "...." field
    - Nukes the inner text and keeps it as empty "" to avoid unescaped quotes.
    Assumes 'notes' is the LAST field in the object (which is true in our schema).
    """
    key_idx = bad_json.find('"notes"')
    if key_idx == -1:
        return bad_json  # nothing to do

    colon_idx = bad_json.find(":", key_idx)
    if colon_idx == -1:
        return bad_json

    first_quote = bad_json.find('"', colon_idx)
    if first_quote == -1:
        return bad_json

    # Look for the last quote before the final '}' of this object
    last_brace = bad_json.rfind("}")
    if last_brace == -1:
        last_brace = len(bad_json) - 1

    last_quote = bad_json.rfind('"', key_idx, last_brace)
    if last_quote == -1 or last_quote <= first_quote:
        return bad_json

    # Build: keep everything up to the opening quote of value,
    # then immediately jump to the closing quote and beyond.
    # Result: "notes": ""}...
    fixed = bad_json[: first_quote + 1] + bad_json[last_quote:]
    return fixed


def safe_json_loads_puzzle(s: str, pid: str = "") -> Dict[str, Any]:
    """
    JSON loader for puzzle objects:
    1) Try normal json.loads
    2) If it fails, auto-strip the notes field (where the LLM tends to put
       unescaped double quotes) and try again.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        label = pid or "puzzle"
        print(f"   âš ï¸� JSON error for {label}; trying to auto-fix notes field:", e)
        fixed = _strip_notes_value(s)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e2:
            print(f"   â�Œ Still failed to parse puzzle JSON for {label}: {e2}")
            print("Raw puzzle JSON:\n", s)
            raise



# stats agent using code execution


stats_code_agent = LlmAgent(
    name="escape_stats_code_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Writes and runs Python code to compute stats about an escape-room game_def.",
    instruction=textwrap.dedent(
        """
        You are a code generator that ONLY returns Python code blocks.
        
        Input: A JSON representation of game_def with keys:
          - title (str)
          - setting (str)
          - puzzles (list of puzzles with type, answer, etc.)
        
        Your job:
          - Parse the JSON into a Python dict.
          - Compute:
              - number of puzzles
              - counts of puzzles by type
              - average answer length (characters)
          - Print a dictionary with these stats to stdout.
        
        Rules:
        - Output MUST be ONLY a Python code block, no explanation before or after.
        - The code MUST print the stats dict as the final line (e.g., print(stats)).
        """
    ),
    code_executor=BuiltInCodeExecutor(),
)

print("âœ… stats_code_agent ready.")


async def compute_game_stats_with_code(game_def: Dict[str, Any]) -> None:
    """
    Uses stats_code_agent + BuiltInCodeExecutor to compute and print some stats
    about the generated game. Mostly here to show off code execution.
    """
    runner = InMemoryRunner(agent=stats_code_agent)

    prompt = (
        "Here is the game_def JSON. Please write Python code to compute stats.\n\n"
        + json.dumps(game_def, ensure_ascii=False)
    )

    print("ğŸ“Š Running stats_code_agent with code execution...\n")

    events = await runner.run_debug(prompt)

    for event in events:
        if event.content:
            for part in event.content.parts:
                # For code execution, the result comes back in function_response
                if hasattr(part, "function_response") and part.function_response:
                    resp = part.function_response.response
                    if isinstance(resp, dict) and "result" in resp:
                        print(resp["result"])



# End-to-end generator pipeline


async def generate_escape_room(
    theme: str,
    difficulty: str = "medium",
    num_puzzles: int = 4,
) -> Dict[str, Any]:
    """
    High-level generator:
      1) story_agent â†’ outline (title, setting, puzzle skeletons)
      2) puzzle_agent â†’ full puzzles
      3) hint_agent   â†’ 3-tier hints
      4) assemble final game_def
    """

    # --- 1) Ask for story outline + puzzle skeletons ---
    outline_prompt = textwrap.dedent(
        f"""
        Theme: {theme}
        Difficulty: {difficulty}
        Number of puzzles: {num_puzzles}
        """
    )

    outline_raw = await call_agent_and_get_text(story_agent, outline_prompt)
    outline_json_str = extract_json_block(outline_raw)

    try:
        outline = json.loads(outline_json_str)
    except json.JSONDecodeError as e:
        print("â�Œ Failed to parse outline JSON:", e)
        print("Raw response:\n", outline_raw)
        raise

    puzzle_skeletons = outline.get("puzzles", [])
    print(f"ğŸ§± Story created: {outline.get('title')}")
    print(f"   Puzzles in outline: {len(puzzle_skeletons)}")

    full_puzzles: List[Dict[str, Any]] = []

    # --- 2) Expand each puzzle skeleton ---
    for skeleton in puzzle_skeletons:
        pid = skeleton.get("id", "PX")
        print(f"\nğŸ”§ Building puzzle {pid} ...")

        puzzle_prompt = textwrap.dedent(
            f"""
            Here is the escape-room outline JSON:
            {json.dumps(outline, ensure_ascii=False)}

            Expand ONLY puzzle with id "{pid}" into a full puzzle JSON 
            using the required schema.
            """
        )

        puzzle_raw = await call_agent_and_get_text(puzzle_agent, puzzle_prompt)
        puzzle_json_str = extract_json_block(puzzle_raw)

        # Use safe loader that can auto-fix broken notes
        puzzle = safe_json_loads_puzzle(puzzle_json_str, pid)

        # --- 3) Generate hints for this puzzle ---
        hints_prompt = textwrap.dedent(
            f"""
            Here is the full puzzle JSON:
            {json.dumps(puzzle, ensure_ascii=False)}

            Generate hints JSON as specified.
            """
        )
        hints_raw = await call_agent_and_get_text(hint_agent, hints_prompt)
        hints_json_str = extract_json_block(hints_raw)

        try:
            hints_obj = json.loads(hints_json_str)
        except json.JSONDecodeError as e:
            print(f"â�Œ Failed to parse hints JSON for {pid}:", e)
            print("Raw hints response:\n", hints_raw)
            raise

        hints_list = hints_obj.get("hints", [])
        puzzle["hints"] = hints_list

        full_puzzles.append(puzzle)
        print(f"   âœ… Puzzle {pid} built with {len(hints_list)} hints.")

    # --- 4) Assemble final game_def ---
    game_def: Dict[str, Any] = {
        "title": outline.get("title", f"{theme} Escape"),
        "setting": outline.get("setting", ""),
        "puzzles": full_puzzles,
    }

    print("\nğŸ�‰ Escape room generated successfully!")
    print(f"   Title: {game_def['title']}")
    print(f"   Total puzzles: {len(game_def['puzzles'])}")

    return game_def



# Generate one game instance

async def build_game_and_stats() -> Dict[str, Any]:
    theme = "Abandoned mountain observatory under a stormy sky"
    difficulty = "medium"
    num_puzzles = 4

    game_def = await generate_escape_room(
        theme=theme,
        difficulty=difficulty,
        num_puzzles=num_puzzles,
    )

    await compute_game_stats_with_code(game_def)
    return game_def


game_def = await build_game_and_stats()


from typing import Dict, Any
import textwrap

from google.adk.runners import InMemoryRunner

def normalize_answer(text: str, mode: str = "lowercase") -> str:
    """
    Normalize answers using the mode specified in puzzle["answer_normalization"].
    Modes handled:
      - lowercase: case-insensitive compare
      - uppercase: uppercase compare
      - strip: trim spaces but keep case
      - exact: trim spaces, case-sensitive
    """
    if text is None:
        return ""
    s = str(text).strip()

    if mode == "lowercase":
        return s.lower()
    if mode == "uppercase":
        return s.upper()
    if mode == "strip":
        return s
    if mode == "exact":
        return s

    # Fallback: be forgiving and compare lowercase
    return s.lower()



# Global game state + gameplay helpers


GAME_STATE: Dict[str, Any] = {
    "game_def": game_def,
    "current_index": 0,       # index into game_def["puzzles"]
    "solved": set(),          # puzzle IDs that are solved
    "failed_attempts": {},    # puzzle_id -> wrong answer count
}


def get_current_puzzle() -> Dict[str, Any]:
    """
    Return the puzzle the player is currently on.
    If the index is out of range, returns {}.
    """
    idx = GAME_STATE["current_index"]
    puzzles = GAME_STATE["game_def"]["puzzles"]
    if idx < 0 or idx >= len(puzzles):
        return {}
    return puzzles[idx]


def move_to_next_puzzle() -> Dict[str, Any]:
    """
    Jump to the next unsolved puzzle.
    If everything is solved, returns {}.
    """
    puzzles = GAME_STATE["game_def"]["puzzles"]
    solved_ids = GAME_STATE["solved"]

    # Try moving forward from current position
    for i in range(GAME_STATE["current_index"] + 1, len(puzzles)):
        if puzzles[i]["id"] not in solved_ids:
            GAME_STATE["current_index"] = i
            return puzzles[i]

    # If none forward, wrap from start
    for i in range(len(puzzles)):
        if puzzles[i]["id"] not in solved_ids:
            GAME_STATE["current_index"] = i
            return puzzles[i]

    # Everything is solved
    return {}


def check_answer_for_current_puzzle(user_answer: str) -> Dict[str, Any]:
    """
    Compare user's answer with the current puzzle's answer.
    Updates GAME_STATE.solved and GAME_STATE.failed_attempts.
    """
    puzzle = get_current_puzzle()
    if not puzzle:
        return {
            "correct": False,
            "puzzle_id": None,
            "message": "No active puzzle.",
        }

    pid = puzzle["id"]
    norm_mode = puzzle.get("answer_normalization", "lowercase")
    correct_answer = puzzle["answer"]

    user_norm = normalize_answer(user_answer, norm_mode)
    sol_norm = normalize_answer(correct_answer, norm_mode)

    is_correct = (user_norm == sol_norm)

    if is_correct:
        GAME_STATE["solved"].add(pid)
        msg = f"Nice! Puzzle {pid} is solved."
    else:
        GAME_STATE.setdefault("failed_attempts", {})
        GAME_STATE["failed_attempts"][pid] = GAME_STATE["failed_attempts"].get(pid, 0) + 1
        msg = "That doesn't unlock it yet. Try again or ask for a hint."

    return {
        "correct": is_correct,
        "puzzle_id": pid,
        "normalized_user": user_norm,
        "normalized_solution": sol_norm,
        "message": msg,
    }


def get_hint_for_current_puzzle() -> Dict[str, Any]:
    """
    Pick a hint tier based on how many times the player missed this puzzle.
    0â€“1 misses â†’ hint 0
    2â€“3 misses â†’ hint 1
    â‰¥4 misses â†’ hint 2 (or last available)
    """
    puzzle = get_current_puzzle()
    if not puzzle:
        return {"puzzle_id": None, "hint": "No active puzzle."}

    pid = puzzle["id"]
    hints = puzzle.get("hints", [])
    if not hints:
        return {"puzzle_id": pid, "hint": "No hints available for this puzzle."}

    failures = GAME_STATE.get("failed_attempts", {}).get(pid, 0)

    if failures <= 1:
        idx = 0
    elif failures <= 3:
        idx = 1
    else:
        idx = 2 if len(hints) > 2 else len(hints) - 1

    return {"puzzle_id": pid, "hint": hints[idx]}



# Wrap helpers as ADK tools


def tool_get_current_puzzle() -> Dict[str, Any]:
    """
    Tool: return the current puzzle in a friendly dict.
    """
    puzzle = get_current_puzzle()
    if not puzzle:
        return {"status": "error", "message": "No active puzzle."}
    return {"status": "success", "puzzle": puzzle}


def tool_move_to_next_puzzle() -> Dict[str, Any]:
    """
    Tool: advance to next unsolved puzzle.
    """
    puzzle = move_to_next_puzzle()
    if not puzzle:
        return {"status": "done", "message": "All puzzles are solved!"}
    return {"status": "success", "puzzle": puzzle}


def tool_check_answer(answer: str) -> Dict[str, Any]:
    """
    Tool: check the user's guess for the current puzzle.
    """
    result = check_answer_for_current_puzzle(answer)
    return {"status": "success", "result": result}


def tool_get_hint() -> Dict[str, Any]:
    """
    Tool: get a hint for the current puzzle.
    """
    return {"status": "success", "result": get_hint_for_current_puzzle()}


print("âœ… Gameplay tools registered.")



# Game Master agent

gamemaster_agent = LlmAgent(
    name="escape_gamemaster",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Runs the EscapeSmith game: puzzles, hints, and story.",
    instruction=textwrap.dedent(
        """
        You are the Game Master for a text-based escape room called EscapeSmith.
        
        You have tools to:
        - get the current puzzle
        - move to the next puzzle
        - check the player's answer
        - provide hints
        
        Behaviour:
        - When the player says "start" (or similar), introduce the setting and the first puzzle.
        - When they send a guess, call the answer-checking tool and react based on the result.
        - Offer hints when they ask explicitly or when they seem stuck.
        - When a puzzle is solved, move to the next one. If all are solved, announce that they escaped.
        - You can re-show the puzzle text if the player asks to repeat.
        
        Constraints:
        - Actually use the tools; do not invent puzzles or answers.
        - Do NOT reveal the exact answer until the puzzle is marked solved.
        - Be immersive but reasonably concise.
        """
    ),
    tools=[
        tool_get_current_puzzle,
        tool_move_to_next_puzzle,
        tool_check_answer,
        tool_get_hint,
    ],
)

print("âœ… gamemaster_agent ready.")



# Simple in-memory (no DB, no sessions API)


runner = InMemoryRunner(agent=gamemaster_agent)
print("âœ… InMemoryRunner ready.")



# Helper: one chat turn using run_debug


async def run_chat_turn(runner: InMemoryRunner, user_text: str) -> None:
    """
    Send one user message into the EscapeSmith Game Master and print the reply.
    Uses runner.run_debug(), so no manual session handling is needed.
    """
    print(f"\nğŸ‘¤ You: {user_text}\n")
    print("ğŸ�® Game Master:")
    print("-" * 40)

    # run_debug returns a list of events (including tool calls)
    events = await runner.run_debug(user_text)

    # Grab the final LLM response text and print it nicely
    for event in events:
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                txt = getattr(part, "text", None)
                if txt:
                    print(txt)

    print("-" * 40)



# Try playing a mini session


await run_chat_turn(runner, "start the escape room")


await run_chat_turn(runner, "I think the answer is eclipse.")


await run_chat_turn(runner, "I'm stuck, can I get a hint?")


def validate_game_def(game_def):
    issues = []

    if "title" not in game_def:
        issues.append("Missing title.")
    if "puzzles" not in game_def or not isinstance(game_def["puzzles"], list):
        issues.append("Missing or invalid 'puzzles' list.")
        return issues

    for i, p in enumerate(game_def["puzzles"], start=1):
        pid = p.get("id", f"<no-id-{i}>")

        for field in ["type", "prompt", "answer"]:
            if not p.get(field):
                issues.append(f"Puzzle {pid}: missing {field}.")

        hints = p.get("hints", [])
        if not hints:
            issues.append(f"Puzzle {pid}: has no hints.")
        elif len(hints) < 2:
            issues.append(f"Puzzle {pid}: only {len(hints)} hint(s).")

    return issues

issues = validate_game_def(game_def)
if issues:
    print("âš ï¸� Found issues:")
    for line in issues:
        print("-", line)
else:
    print("âœ… game_def looks structurally fine.")

