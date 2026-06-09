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


"""
==============================================================
     MULTI-AGENT INTELLIGENT SYSTEM (Professional Version)
     Demonstrating:

     ✔ LLM-Powered Agent
     ✔ Sequential Agent
     ✔ Parallel Agents
     ✔ Loop Agent
     ✔ Custom Tools
     ✔ Built-in (simulated) Code Execution Tool
     ✔ In-memory Session & State Management
     ✔ Observability (Logging)
---------------------------------------------------------------
     Author  : <Your Name>
     Course  : Advanced AI Systems
     File    : main.py
==============================================================
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor

# For LLM-powered agent
import openai


# ==============================================================
# 1. LOGGING SETUP (Observability)
# ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("System")


# ==============================================================
# 2. SESSION & MEMORY MANAGEMENT
# ==============================================================

class InMemorySession:
    """Simple key-value memory store for agents."""

    def __init__(self):
        self.data = {}

    def save(self, key, value):
        logger.info(f"[MEMORY] Saved '{key}'")
        self.data[key] = value

    def load(self, key):
        logger.info(f"[MEMORY] Loaded '{key}'")
        return self.data.get(key)


SESSION = InMemorySession()


# ==============================================================
# 3. TOOLS
# ==============================================================

# ---- Custom Math Tool ----
def math_tool(a, b, op):
    logger.info(f"[TOOL] Math Tool called with: {a} {op} {b}")

    if op == "add": return a + b
    if op == "sub": return a - b
    if op == "mul": return a * b
    if op == "div": return a / b

    return "Unsupported operation"


# ---- Built-in Tool Simulation: Code Executor ----
def code_executor(expression):
    logger.info(f"[TOOL] Code Executor running: {expression}")

    try:
        return eval(expression)
    except Exception as e:
        return f"Error: {str(e)}"


# ==============================================================
# 4. BASE AGENT CLASS
# ==============================================================

class BaseAgent:
    def __init__(self, name):
        self.name = name

    def log(self, msg):
        logger.info(f"[{self.name}] {msg}")

    def run(self):
        raise NotImplementedError


# ==============================================================
# 5. AGENTS
# ==============================================================

# ---- Sequential Agent ----
class SequentialAgent(BaseAgent):
    def run(self, user):
        self.log("Generating greeting...")
        output = f"Hello {user}, I am a Sequential Agent handling the first task."
        SESSION.save("greeting", output)
        return output


# ---- Parallel Agents ----
class SquareAgent(BaseAgent):
    def run(self, n):
        self.log(f"Calculating square of {n}")
        return n * n


class CubeAgent(BaseAgent):
    def run(self, n):
        self.log(f"Calculating cube of {n}")
        return n * n * n


# ---- Loop Agent ----
class LoopAgent(BaseAgent):
    def run(self, count):
        self.log("Starting countdown task...")
        result = []

        for i in range(count, 0, -1):
            result.append(i)
            time.sleep(0.25)

        return result


# ---- LLM-Powered Agent ----
class LLMAgent(BaseAgent):
    def __init__(self, name="LLM_Agent", model="gpt-4.1-mini"):
        super().__init__(name)
        self.model = model

        # Your API key
        openai.api_key = "YOUR_OPENAI_API_KEY"

    def run(self, prompt):
        self.log("Sending request to LLM...")

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response["choices"][0]["message"]["content"]
            self.log("LLM responded successfully.")
            return result

        except Exception as e:
            return f"LLM Error: {str(e)}"


# ==============================================================
# 6. MAIN SYSTEM EXECUTION
# ==============================================================

def main():

    print("""
==============================================================
         MULTI-AGENT INTELLIGENT SYSTEM — EXECUTION
==============================================================
    """)

    # ---------------------------------------------
    # 1. SEQUENTIAL AGENT
    # ---------------------------------------------
    seq = SequentialAgent("SequentialAgent")
    seq_output = seq.run("Arijit")
    print("▶ Sequential Agent Output:")
    print("  ", seq_output)
    print("--------------------------------------------------------------")

    # ---------------------------------------------
    # 2. PARALLEL AGENTS
    # ---------------------------------------------
    sq_agent = SquareAgent("SquareAgent")
    cb_agent = CubeAgent("CubeAgent")

    print("▶ Running Parallel Agents (Square & Cube)...")

    with ThreadPoolExecutor() as executor:
        f1 = executor.submit(sq_agent.run, 6)
        f2 = executor.submit(cb_agent.run, 6)

        square_output = f1.result()
        cube_output = f2.result()

    print("  Square of 6 =", square_output)
    print("  Cube of 6   =", cube_output)
    print("--------------------------------------------------------------")

    # ---------------------------------------------
    # 3. LOOP AGENT
    # ---------------------------------------------
    loop = LoopAgent("LoopAgent")
    loop_output = loop.run(5)

    print("▶ Loop Agent Countdown:")
    print("  ", loop_output)
    print("--------------------------------------------------------------")

    # ---------------------------------------------
    # 4. CUSTOM TOOL
    # ---------------------------------------------
    print("▶ Math Tool (7 * 8):")
    print("  ", math_tool(7, 8, "mul"))
    print("--------------------------------------------------------------")

    # ---------------------------------------------
    # 5. CODE EXECUTION TOOL
    # ---------------------------------------------
    print("▶ Code Executor (10 + 90):")
    print("  ", code_executor("10 + 90"))
    print("--------------------------------------------------------------")

    # ---------------------------------------------
    # 6. LLM-POWERED AGENT
    # ---------------------------------------------
    llm = LLMAgent()
    print("▶ LLM Agent Output:")
    llm_output = llm.run("Explain multi-agent systems in 3 simple bullet points.")
    print("  ", llm_output)
    print("--------------------------------------------------------------")

    # ---------------------------------------------
    # 7. MEMORY READBACK
    # ---------------------------------------------
    print("▶ Loading Saved Memory:")
    print("  ", SESSION.load("greeting"))
    print("--------------------------------------------------------------")

    print("\n=============== END OF EXECUTION ================")


# ==============================================================
# Run main()
# ==============================================================
if __name__ == "__main__":
    main()


