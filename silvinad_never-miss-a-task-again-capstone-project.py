# -------------------------------------------------------------
# GOOGLE-STYLE AGENT — WORKS DIRECTLY INSIDE JUPYTER NOTEBOOK
# -------------------------------------------------------------

from typing import Dict, Any
import datetime
import math

# -------------------------------------------------------------
# 1. Simple TOOL FUNCTIONS for the agent
# -------------------------------------------------------------

def tool_search(query: str) -> str:
    """
    Dummy search tool.
    In real environments, connect to SERP API or Google Search API.
    """
    return f"[Search Results for: {query}] — (Dummy output for demo)"

def tool_calculator(expression: str) -> str:
    """
    Safe calculator using Python's math library.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, math.__dict__)
        return str(result)
    except Exception:
        return "Error: Invalid math expression."

def tool_time_now() -> str:
    """
    Returns current date/time.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = {
    "search": tool_search,
    "calculator": tool_calculator,
    "time_now": tool_time_now,
}

# -------------------------------------------------------------
# 2. BASIC LLM AGENT (Google-like REACT structure)
# -------------------------------------------------------------

class SimpleAgent:
    def __init__(self):
        self.history = []

    def think(self, prompt: str) -> str:
        """
        Agent decides what tool to call.
        Simple logic for demonstration.
        """
        lower = prompt.lower()

        # Choose tool automatically
        if "search" in lower or "who" in lower or "what is" in lower:
            return "TOOL: search"
        if any(op in lower for op in ["+", "-", "/", "*", "calculate"]):
            return "TOOL: calculator"
        if "time" in lower or "date" in lower or "now" in lower:
            return "TOOL: time_now"
        
        return "TOOL: none"

    def run(self, prompt: str) -> str:
        self.history.append(prompt)
        decision = self.think(prompt)

        if decision.startswith("TOOL:"):
            tool_name = decision.split(":")[1].strip()

            if tool_name == "none":
                return "I'm not sure which tool to use. Please clarify."

            tool = TOOLS.get(tool_name)
            if tool:
                # Extract expression for calculator
                if tool_name == "calculator":
                    expression = prompt.replace("calculate", "").strip()
                    return tool(expression)
                
                # Extract query for search
                if tool_name == "search":
                    query = prompt.replace("search", "").strip()
                    return tool(query)
                
                return tool()

        return "Could not decide a tool."


# -------------------------------------------------------------
# 3. RUN THE AGENT
# -------------------------------------------------------------

agent = SimpleAgent()

print("Agent is ready! Try commands like:\n")
print("• search who is Sundar Pichai")
print("• calculate 22/7")
print("• what is the time now?\n")

# Example run
result = agent.run("calculate 22/7")
result



# -------------------------------------------------------------
# GOOGLE-STYLE AGENT — INTERACTIVE VERSION (ASKS INPUT RUNTIME)
# -------------------------------------------------------------

from typing import Dict, Any
import datetime
import math

# -------------------------------------------------------------
# 1. TOOL FUNCTIONS
# -------------------------------------------------------------

def tool_search(query: str) -> str:
    """Dummy search tool."""
    return f"[Search Results for: {query}] (demo output)"

def tool_calculator(expression: str) -> str:
    """Safe calculator tool."""
    try:
        result = eval(expression, {"__builtins__": {}}, math.__dict__)
        return str(result)
    except:
        return "Error: invalid math expression"

def tool_time_now() -> str:
    """Returns current date/time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOLS = {
    "search": tool_search,
    "calculator": tool_calculator,
    "time_now": tool_time_now,
}

# -------------------------------------------------------------
# 2. SIMPLE AGENT
# -------------------------------------------------------------

class SimpleAgent:
    def think(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        if "search" in prompt_lower or "who" in prompt_lower or "what is" in prompt_lower:
            return "search"

        if any(op in prompt_lower for op in ["+", "-", "*", "/", "calculate"]):
            return "calculator"

        if "time" in prompt_lower or "date" in prompt_lower or "now" in prompt_lower:
            return "time_now"

        return "none"

    def run(self, prompt: str) -> str:
        tool_name = self.think(prompt)

        if tool_name == "none":
            return "I don't know which tool to use. Try saying search/calculate/time."

        if tool_name == "calculator":
            expression = prompt.replace("calculate", "").strip()
            return TOOLS["calculator"](expression)

        if tool_name == "search":
            query = prompt.replace("search", "").strip()
            return TOOLS["search"](query)

        if tool_name == "time_now":
            return TOOLS["time_now"]()

# -------------------------------------------------------------
# 3. INTERACTIVE LOOP
# -------------------------------------------------------------

agent = SimpleAgent()

print("Google Agent Ready!")
print("Try:")
print("• search who is Sundar Pichai")
print("• calculate 22/7")
print("• time now")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Agent stopped.")
        break

    response = agent.run(user_input)
    print("Agent:", response)



# -------------------------------------------------------------
# GOOGLE + CHATGPT AGENT (NOTEBOOK RUNTIME INTERACTIVE)
# -------------------------------------------------------------

import requests
import openai
import datetime
import math

# -------------------------------------------------------------
# 1. YOUR API KEYS
# -------------------------------------------------------------

SERPAPI_KEY = "YOUR_SERPAPI_KEY"        # For Google Search
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" # For ChatGPT

openai.api_key = OPENAI_API_KEY

# -------------------------------------------------------------
# 2. TOOL FUNCTIONS
# -------------------------------------------------------------

def google_search(query: str) -> str:
    """Real Google search via SerpAPI."""
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY
    }
    
    response = requests.get(url, params=params).json()
    
    try:
        answer = response["organic_results"][0]["snippet"]
        return f"Google says: {answer}"
    except:
        return "Google search failed or returned no results."

def chatgpt_search(query: str) -> str:
    """ChatGPT reasoning-based search."""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Search and answer: {query}"}]
    )
    return response["choices"][0]["message"]["content"]

def current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculator(expr: str):
    try:
        return str(eval(expr, {"__builtins__": {}}, math.__dict__))
    except:
        return "Invalid math expression."

# -------------------------------------------------------------
# 3. INTELLIGENT AGENT DECISION MAKER
# -------------------------------------------------------------

class SmartAgent:
    def think(self, prompt: str):
        p = prompt.lower()

        if any(word in p for word in ["time", "date", "now"]):
            return "time"

        if any(op in p for op in ["+", "-", "*", "/", "calculate"]):
            return "calculator"
        
        if "google" in p:
            return "google"

        if "chatgpt" in p or "explain" in p or "why" in p:
            return "chatgpt"

        # Default behaviour
        return "google"

    def run(self, prompt: str):
        tool = self.think(prompt)

        if tool == "time":
            return current_time()
        if tool == "calculator":
            expr = prompt.replace("calculate", "")
            return calculator(expr)
        if tool == "google":
            query = prompt.replace("google", "")
            return google_search(query)
        if tool == "chatgpt":
            query = prompt.replace("chatgpt", "")
            return chatgpt_search(query)

        return "I could not decide how to search."

# -------------------------------------------------------------
# 4. INTERACTIVE LOOP
# -------------------------------------------------------------

agent = SmartAgent()

print("Google + ChatGPT Agent Ready!")
print("Examples:")
print("• google who is Sundar Pichai")
print("• chatgpt explain quantum computing")
print("• calculate 22/7")
print("• what is the time now\n")

while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit"]:
        print("Agent stopped.")
        break

    answer = agent.run(query)
    print("Agent:", answer)


