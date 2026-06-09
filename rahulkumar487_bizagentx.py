# ============================
#Â  Â AI MULTI-AGENT SYSTEM (Google Gemini)
# ============================

!pip install google-generativeai --quiet

import os
import json
import asyncio
import logging
import google.generativeai as genai
# Note: UserSecretsClient is specific to Kaggle/Colab and needs the environment to be set up.
from kaggle_secrets import UserSecretsClient


# ----------------------------
# 0. SETUP LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("ai-agents")


# ----------------------------
# 1. INIT GOOGLE GEMINI CLIENT
# ----------------------------
user_secrets = UserSecretsClient()
# This line assumes a secret named 'GOOGLE_API_KEY' exists in your environment.
google_key = user_secrets.get_secret("GOOGLE_API_KEY")

genai.configure(api_key=google_key)
# Using the stable, public alias for the Flash model
model = genai.GenerativeModel("gemini-2.5-flash")


# ----------------------------
# 2. TOOLS
# ----------------------------

def run_python(code: str):
    try:
        local_vars = {}
        # Execute code in a constrained environment
        exec(code, {"__builtins__": None}, local_vars) 
        return {"output": str(local_vars)} # Convert output to string for safe representation
    except Exception as e:
        return {"error": str(e)}

def analyze_sales(data):
    total = sum(item["amount"] for item in data)
    avg = total / len(data)
    return {
        "total_sales": total,
        "average_transaction": avg,
        "transaction_count": len(data)
    }

def faq_lookup(question):
    kb = {
        "refund policy": "Refunds are available within 30 days.",
        "shipping": "Shipping takes 3â€“5 business days.",
        "support hours": "Support is available 24/7.",
    }
    for key in kb:
        if key in question.lower():
            return kb[key]
    return "I donâ€™t have information on that."

TOOLS = {
    "run_python": run_python,
    "analyze_sales": analyze_sales,
    "faq_lookup": faq_lookup,
}


# ----------------------------
# 3. A2A PROTOCOL
# ----------------------------
def a2a(role, content):
    """Agent-to-Agent communication protocol (simple JSON wrapper)."""
    return json.dumps({"role": role, "content": content})


# ----------------------------
# 4. MEMORY
# ----------------------------
memory = {
    "sales_data": [
        {"amount": 120},
        {"amount": 200},
        {"amount": 310},
        {"amount": 150},
    ],
    "history": []
}

def compact_context(history):
    # Keep only the last 5 messages for context
    return history[-5:]


# ----------------------------
# 5. BASE AGENT CLASS
# ----------------------------
class Agent:
    def __init__(self, name):
        self.name = name

    async def run(self, query, memory):
        raise NotImplementedError


# ----------------------------
# 6. ANALYTICS AGENT
# ----------------------------
class AnalyticsAgent(Agent):
    async def run(self, query, memory):
        if "sales" in query.lower() or "analyze" in query.lower():
            result = TOOLS["analyze_sales"](memory["sales_data"])
            return a2a("analytics", f"Sales Analysis: {result}")
        return None


# ----------------------------
# 7. CUSTOMER SUPPORT AGENT
# ----------------------------
class SupportAgent(Agent):
    async def run(self, query, memory):
        answer = TOOLS["faq_lookup"](query)
        # Only return a response if a specific FAQ was found
        if "I donâ€™t have information on that" not in answer:
            return a2a("support", f"Support Response: {answer}")
        return None


# ----------------------------
# 8. CODE EXECUTION AGENT
# ----------------------------
class CodeAgent(Agent):
    async def run(self, query, memory):
        if query.startswith("run:"):
            code = query.replace("run:", "").strip()
            result = TOOLS["run_python"](code)
            return a2a("code", f"Code Output: {result}")
        return None


# ----------------------------
# 9. COORDINATOR AGENT (Gemini)
# ----------------------------
class CoordinatorAgent(Agent):
    async def run(self, query, memory):
        messages = compact_context(memory["history"])
        
        # Prompting Gemini to act as an orchestrator
        coordination_prompt = (
            "CONTEXT: You are the central Coordinator Agent. "
            "Review the user query and provide a brief, high-level summary "
            "of the task and which specialty agents (Analytics, Support, Code) "
            "should be engaged to answer it. Do not attempt to answer the query yourself.\n\n"
            f"USER QUERY: {query}"
        )
        
        messages.append({"role": "user", "content": coordination_prompt})

        full_prompt = "\n".join([m["content"] for m in messages])

        response = model.generate_content(full_prompt)
        answer = response.text

        # Return the coordination response
        return a2a("coordinator", answer)


# ----------------------------
# 10. MULTI-AGENT PIPELINE
# ----------------------------
async def multi_agent_pipeline(query, memory):
    log.info("Running pipeline...")

    agents = [
        CoordinatorAgent("coordinator"),
        AnalyticsAgent("analytics"),
        SupportAgent("support"),
        CodeAgent("code"),
    ]

    # 1. Coordinator runs first
    coordinator_answer = await agents[0].run(query, memory)

    # 2. All other specialty agents run concurrently
    results = await asyncio.gather(*[
        a.run(query, memory) for a in agents[1:]
    ])

    # Filter out None results and combine with coordinator's answer
    return [coordinator_answer] + [r for r in results if r]

# ----------------------------
# 11. STRUCTURED PRESENTATION
# ----------------------------
def present_results(raw_results, query):
    """Parses raw A2A JSON output and formats it for human readability."""
    structured_output = {"query": query, "responses": {}}
    
    # 1. Parse and aggregate results
    for r in raw_results:
        try:
            data = json.loads(r)
            role = data.get("role", "unknown").upper()
            content = data.get("content", "No content provided.")
            structured_output["responses"][role] = content
        except json.JSONDecodeError:
            structured_output["responses"]["ERROR"] = f"Failed to decode JSON: {r}"

    # 2. Format for presentation
    print("=========================================")
    print("âœ¨ MULTI-AGENT SYSTEM RESPONSE âœ¨")
    print("=========================================")
    print(f"**USER QUERY:** {structured_output['query']}\n")
    
    # Print Coordinator first
    coordinator_response = structured_output['responses'].pop('COORDINATOR', 'N/A')
    print("--- ðŸ§  COORDINATOR AGENT (Gemini) ---")
    print(coordinator_response)
    print("\n-----------------------------------------\n")

    # Print other agents
    for role, content in structured_output['responses'].items():
        if role == 'ANALYTICS':
            print("--- ðŸ“Š ANALYTICS AGENT RESPONSE ---")
            print(content)
        elif role == 'SUPPORT':
            print("--- ðŸ“ž SUPPORT AGENT RESPONSE ---")
            print(content)
        elif role == 'CODE':
            print("--- ðŸ’» CODE AGENT RESPONSE ---")
            print(content)
        else:
            print(f"--- {role} AGENT RESPONSE ---")
            print(content)
        print("\n")
    

# ----------------------------
# 12. RUN DEMO
# ----------------------------
# Example query designed to engage all three specialty agents
query = "Please analyze the sales and also tell me the refund policy. Also, run: print('Calculation done')"

# 1. Run the multi-agent pipeline
results = await multi_agent_pipeline(query, memory)

# 2. Present the results in a structured way
present_results(results, query)

