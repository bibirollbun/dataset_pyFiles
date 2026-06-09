# ==========================================================
#   1. BASIC SETUP (Imports + Config)
# ==========================================================
!pip install openai --quiet

import os
import json
import time
from openai import OpenAI

# Put your key here
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
client = OpenAI()

# Agent Settings
SYSTEM_PROMPT = """
You are an intelligent autonomous agent. 
You must strictly follow the user's goals, think step-by-step,
use tools when required, stay factual, and produce final answers clearly.
"""


# ==========================================================
#   2. TOOL FUNCTIONS (Add as many tools as needed)
# ==========================================================
def tool_calculator(expression: str):
    """Safely evaluate arithmetic expressions."""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation."

def tool_search(query: str):
    """Dummy search tool â€“ replace this with real API."""
    return f"Search results for query: {query} (dummy placeholder)"

def tool_file_write(filename: str, content: str):
    """Write output to a file inside Kaggle notebook."""
    with open(filename, "w") as f:
        f.write(content)
    return f"File '{filename}' saved successfully."

TOOLS = {
    "calculator": tool_calculator,
    "search": tool_search,
    "file_write": tool_file_write,
}


# ==========================================================
#   3. AGENT LOOP (CORE ENGINE)
# ==========================================================
def agent_step(user_input, memory=""):
    """Single reasoning step with tool support."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    if memory:
        messages.append({"role": "assistant", "content": f"[MEMORY]\n{memory}"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=600
    ).choices[0].message.content

    return response


def run_agent(user_goal: str, max_steps: int = 5, memory_enabled=True):
    print("ðŸ”· STARTING AGENT")
    memory = ""
    last_output = ""

    for step in range(1, max_steps + 1):
        print(f"\n=== STEP {step} ===")

        agent_output = agent_step(user_goal, memory)
        print("Agent output:", agent_output)

        # TOOL FORMAT:
        # use_tool:tool_name{"arg":"value"}
        if "use_tool:" in agent_output:
            try:
                tool_name = agent_output.split("use_tool:")[1].split("{")[0].strip()
                args_json = "{" + agent_output.split("{", 1)[1].rsplit("}", 1)[0] + "}"
                args = json.loads(args_json)

                tool_fn = TOOLS.get(tool_name)
                if tool_fn:
                    tool_result = tool_fn(**args)
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                print("Tool result:", tool_result)

                if memory_enabled:
                    memory += f"\n[TOOL RESULT]\n{tool_result}"

                last_output = tool_result

            except Exception as e:
                print("Tool parsing error:", e)

        else:
            # No tool used â€” final answer
            last_output = agent_output
            break

    print("\nðŸ”· AGENT FINISHED")
    return last_output

