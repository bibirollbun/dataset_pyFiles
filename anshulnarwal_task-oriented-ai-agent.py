# ============================================================
#            GOOGLE AI AGENT INTENSIVE â€“ CAPSTONE
#                      Agents for Good Track
#               Offline AI Education Assistant Agent
# ============================================================

import json
import time
import traceback
from textwrap import dedent

# ============================================================
# 1. MOCK LLM (Offline)
# ============================================================

class MockLLM:
    """A simple offline LLM that returns template-based responses."""
    
    def generate(self, prompt):
        if "plan" in prompt.lower():
            return (
                "1. Understand the user query.\n"
                "2. Search the knowledge base for relevant information.\n"
                "3. Break the topic into simple learning steps.\n"
                "4. Provide easy explanations.\n"
                "5. Run code examples if needed.\n"
                "6. Return final answer."
            )
        
        elif "summarize" in prompt.lower():
            return "Here is a simple summary based on provided content."
        
        return "This is a mock response (LLM offline mode)."


# ============================================================
# 2. DOCUMENT RETRIEVER (Local Search)
# ============================================================

class DocumentRetriever:
    """Searches the local documents for matching text."""
    
    def __init__(self, documents):
        self.documents = documents
        
    def search(self, query):
        results = []
        for title, text in self.documents.items():
            if query.lower() in text.lower():
                results.append((title, text))
        return results


# Sample mini knowledge base
DOCUMENTS = {
    "education_ai.txt": """
        AI in education helps students understand complex topics,
        provides step-by-step explanations, retrieves learning material,
        and makes education accessible for underserved communities.
    """,
    "computer_basics.txt": """
        Computers accept input, process data, store information,
        and generate output. They execute instructions using algorithms.
    """
}


# ============================================================
# 3. CODE EXECUTION TOOL
# ============================================================

class CodeExecutor:
    """Executes Python code safely inside the notebook."""
    
    def run(self, code):
        try:
            local_env = {}
            exec(code, {}, local_env)
            return local_env
        except Exception:
            return {"error": traceback.format_exc()}


# ============================================================
# 4. TASK PLANNER
# ============================================================

class TaskPlanner:
    """Creates an action plan using the LLM."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_plan(self, query):
        plan_prompt = f"Create a step-by-step plan to answer: {query}"
        return self.llm.generate(plan_prompt)


# ============================================================
# 5. MAIN AGENT CONTROLLER
# ============================================================

class EducationAgent:
    
    def __init__(self):
        self.llm = MockLLM()
        self.retriever = DocumentRetriever(DOCUMENTS)
        self.planner = TaskPlanner(self.llm)
        self.executor = CodeExecutor()
    
    def run(self, query):
        print("ğŸ”¹ User Query:", query)
        print("--------------------------------------------------\n")
        
        # 1. PLAN
        plan = self.planner.create_plan(query)
        print("ğŸ“Œ PLAN:\n", plan)
        print("\n--------------------------------------------------\n")
        
        # 2. RETRIEVE INFO
        print("ğŸ”� SEARCH RESULTS:")
        results = self.retriever.search(query)
        if not results:
            print("No documents found.\n")
        else:
            for title, text in results:
                print(f"\nğŸ“„ {title}:\n{text}\n")
        
        print("--------------------------------------------------\n")
        
        # 3. GENERATE EXPLANATION
        explanation_prompt = f"Summarize and explain: {query}"
        explanation = self.llm.generate(explanation_prompt)
        print("ğŸ§  EXPLANATION:\n", explanation)
        print("\n--------------------------------------------------\n")
        
        # 4. OPTIONAL CODE DEMO
        if "python" in query.lower() or "code" in query.lower():
            demo_code = """
result = 5 + 5
"""
            print("â–¶ï¸� Running example code...\n")
            print(self.executor.run(demo_code))
            print("--------------------------------------------------\n")
        
        print("âœ… AGENT WORKFLOW COMPLETE.")


# ============================================================
# 6. RUN AGENT WITH SAMPLE QUERY
# ============================================================

agent = EducationAgent()

# Try your own queries here
agent.run("Explain how AI helps students learn")


