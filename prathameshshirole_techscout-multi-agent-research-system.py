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


%%capture
# Cell 1: Installation
# Installs Google GenAI and Search tools. 
# The '%%capture' magic command hides the long output to keep your notebook clean.
!pip install -q -U google-generativeai duckduckgo-search


# Cell 3: System Architecture (With Spam Filter)
import os
import time
import json
import google.generativeai as genai
from IPython.display import display, Markdown
from kaggle_secrets import UserSecretsClient

# --- CONFIGURATION ---
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
except Exception as e:
    print("âš ï¸� Error: GOOGLE_API_KEY not found in Secrets. Please add it!")

# --- 1. OBSERVABILITY ---
class AgentLogger:
    def __init__(self):
        self.logs = []
    
    def log(self, agent, action, content):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append({"timestamp": timestamp, "agent": agent, "type": action, "content": content})
        print(f"[{timestamp}] **{agent}** [{action}]: {content[:100]}..." if len(str(content)) > 100 else f"[{timestamp}] **{agent}** [{action}]: {content}")

    def get_trace(self):
        return json.dumps(self.logs, indent=2)

logger = AgentLogger()

# --- 2. SMART SEARCH TOOL ---
class SearchTool:
    def search(self, query: str) -> str:
        """Searches web, but falls back to backup if results are ads/irrelevant."""
        try:
            # 1. Random delay to mimic human behavior
            time.sleep(2)
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=4))
            
            # 2. Convert to string
            if results:
                result_str = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
                
                # 3. SPAM FILTER: Check if results are actually about AI
                # The cloud IP often gets insurance ads. We filter them out here.
                valid_keywords = ["ai", "agent", "intelligence", "bot", "future", "tech"]
                if any(keyword in result_str.lower() for keyword in valid_keywords):
                    return result_str
                else:
                    print("   âš ï¸� Search results looked like ads/irrelevant. Rejecting...")
                    raise Exception("Irrelevant results (Spam Filter)")
                    
        except Exception as e:
            print(f"   (Search switch: {e})")

        # 4. BACKUP KNOWLEDGE BASE (The "Golden" Data)
        print("   âš ï¸� Switching to Backup Knowledge Base.")
        return """
        [BACKUP KNOWLEDGE RETRIEVED]
        Topic: Agentic AI Trends 2025
        1. **From Chat to Action**: LLMs are evolving into "Action Models" (LAMs) that can click buttons, write code, and book flights, not just talk.
        2. **Multi-Agent Orchestration**: Systems where multiple specialized agents (e.g., a Coder, a Tester, and a Manager) collaborate to solve complex problems.
        3. **Standardized Protocols**: The rise of protocols like MCP (Model Context Protocol) allows agents to connect to any data source universally.
        4. **Self-Correction**: Agents are developing "System 2" thinkingâ€”the ability to pause, reason, check their own errors, and fix them before answering.
        """

search_tool = SearchTool()

# --- 3. AGENT LOGIC ---
class BaseAgent:
    def __init__(self, name, model="gemini-2.0-flash"):
        self.name = name
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt):
        for attempt in range(3):
            try:
                time.sleep(2) # Prevent rate limits
                return self.model.generate_content(prompt).text
            except Exception as e:
                print(f"   âš ï¸� Retry {attempt+1}...")
                time.sleep(5)
                # Fallback to older model if 2.0 is missing
                if "404" in str(e): self.model = genai.GenerativeModel("gemini-1.5-flash")
        return "Error: Generation failed."

class ResearchAgent(BaseAgent):
    def __init__(self): super().__init__("Researcher")
    
    def run(self, topic):
        logger.log(self.name, "THOUGHT", f"Researching: {topic}")
        # Generate query
        q = self.generate(f"Write 1 search query for: {topic}. Output ONLY the query.").strip().replace('"','')
        logger.log(self.name, "ACTION", f"Searching for: {q}")
        # Search
        data = search_tool.search(q)
        logger.log(self.name, "TOOL_OUTPUT", "Data retrieved.")
        # Summarize
        summary = self.generate(f"Summarize these trends:\n{data}")
        logger.log(self.name, "OUTPUT", summary)
        return summary

class WriterAgent(BaseAgent):
    def __init__(self): super().__init__("Writer")

    def run(self, data, topic):
        logger.log(self.name, "THOUGHT", "Drafting report...")
        prompt = f"Write a professional tech report on '{topic}' using this data:\n{data}\nStructure: Headline, Key Trends, Summary."
        report = self.generate(prompt)
        logger.log(self.name, "OUTPUT", "Report generated.")
        return report

class EvaluatorAgent(BaseAgent):
    def __init__(self): super().__init__("Evaluator")

    def evaluate(self, report):
        res = self.generate(f"Rate this report (1-5) and explain why.\nREPORT:\n{report}")
        logger.log(self.name, "EVALUATION", res)
        print(f"\nğŸ“Š AGENT EVALUATION:\n{res}")

# --- 4. ORCHESTRATION ---
def run_project():
    topic = "The future of Agentic AI in 2025"
    print(f"ğŸš€ STARTING AGENT WORKFLOW: {topic}\n" + "="*40)
    
    researcher = ResearchAgent()
    writer = WriterAgent()
    evaluator = EvaluatorAgent()
    
    # Execution Chain
    findings = researcher.run(topic)
    print("-" * 40)
    final_report = writer.run(findings, topic)
    print("="*40 + "\nâœ… FINAL REPORT:\n")
    display(Markdown(final_report))
    print("-" * 40)
    evaluator.evaluate(final_report)
    return logger.get_trace()


# Cell 4: Execution
# Run the complete Multi-Agent System
trace_data = run_project()

# Optional: Print the full system log to show the "Thought Process"
print("\nğŸ”� FULL SYSTEM TRACE (JSON):")
print(trace_data)

