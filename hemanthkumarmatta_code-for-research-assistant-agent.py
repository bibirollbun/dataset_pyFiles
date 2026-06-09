# ============================================
# CELL 1 — Install Libraries
# ============================================
!pip install -q google-generativeai requests beautifulsoup4


# ============================================
# CELL 2 — Imports & Gemini Configuration
# ============================================
from google import generativeai as genai
import json, time, logging, requests
from bs4 import BeautifulSoup

# Configure Gemini API Key
genai.configure(api_key="YOUR_API_KEY_HERE")   # <-- Replace this

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")


# ============================================
# CELL 3 — Define Tools
# ============================================

# -------- TOOL 1: Search Tool (Simulated because Kaggle blocks internet) --------
def search_tool(query):
    simulated = [
        {"title": f"{query} - Article A", "snippet": f"Insight A about {query}", "url": "https://example.com/A"},
        {"title": f"{query} - Article B", "snippet": f"Insight B about {query}", "url": "https://example.com/B"},
        {"title": f"{query} - Article C", "snippet": f"Insight C about {query}", "url": "https://example.com/C"},
    ]

    logger.info(f"[TOOL] search_tool called for: {query}")
    return simulated


# -------- TOOL 2: Summarizer Tool (Gemini-powered) --------
def summarizer_tool(texts, instruction="Summarize into 5 bullet points with sources."):
    input_block = "\n".join(
        [f"{x['title']} — {x['snippet']} ({x['url']})" for x in texts]
    )

    prompt = f"{instruction}\n\nSources:\n{input_block}\n\nSummary:"

    result = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)

    summary = result.text
    logger.info("[TOOL] summarizer_tool executed")
    return summary


# -------- TOOL 3: Save Tool --------
def save_tool(text, filename="research_note.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"[TOOL] save_tool wrote file: {filename}")
    return filename


# ============================================
# CELL 4 — Simple Agent Class
# ============================================

class SimpleAgent:
    def __init__(self):
        self.logs = []
        self.tools = {
            "search": search_tool,
            "summarize": summarizer_tool,
            "save": save_tool
        }

    def call_tool(self, tool_name, *args, **kwargs):
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        logger.info(f"[AGENT] Calling tool: {tool_name}")

        output = tool(*args, **kwargs)

        # MCP-style logs
        log_entry = {
            "tool": tool_name,
            "args": args,
            "kwargs": kwargs,
            "output": output,
            "timestamp": time.time()
        }

        self.logs.append(log_entry)
        return output

    def run(self, query):
        results = self.call_tool("search", query)
        summary = self.call_tool("summarize", results)

        return {
            "results": results,
            "summary": summary,
            "logs": self.logs
        }


# ============================================
# CELL 5 — Run the agent
# ============================================

agent = SimpleAgent()

query = "Advancements in AI Agents in 2025"   # you can change this
output = agent.run(query)

print("===== SUMMARY =====\n")
print(output["summary"])


# ============================================
# CELL 6 — Logs
# ============================================
print("\n===== TOOL CALL LOGS =====\n")
print(json.dumps(output["logs"], indent=2))


# ============================================
# CELL 7 — LRO Simulation (Approval)
# ============================================

approval = "yes"   # change to "no" to cancel

if approval.lower() == "yes":
    filename = agent.call_tool("save", output["summary"])
    print("Saved as:", filename)
else:
    print("Publish cancelled.")


# ============================================
# CELL 8


