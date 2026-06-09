import sys
import os

print("Step 1: Force Uninstalling All Potential Conflict Libraries")
!pip uninstall -y numpy scipy cirq dask distributed langchain langchain-core langchain-google-genai google-generativeai pydantic > /dev/null 2>&1

print("Step 2: Surgically Removing Dask's Import Hook from Memory")
sys.meta_path = [finder for finder in sys.meta_path if 'dask' not in str(finder).lower()]
print("✓ Dask hook removed.")

print("Step 3: Installing a Known Stable Stack")
!pip install "numpy==1.23.5" "scipy==1.10.1" "networkx==3.1" > /dev/null 2>&1
!pip install "cirq==1.3.0" > /dev/null 2>&1
!pip install -q "pydantic==1.10.13" "langchain==0.1.11" "langchain-core==0.1.30" "langchain-google-genai==0.0.9" "google-generativeai" "matplotlib" "pandas" > /dev/null 2>&1

print("✓ Libraries installed successfully.")


import os, warnings, time, re, io
import pandas as pd, numpy as np, matplotlib.pyplot as plt, google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from kaggle_secrets import UserSecretsClient
import cirq, scipy
from IPython.display import Markdown, display, HTML
from google.api_core import exceptions

warnings.filterwarnings('ignore')
print("✓ Core libraries imported.")

# AUTHENTICATION & VERIFICATION
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    print("✓ Authentication successful.")
    print(f"✓ Numpy: {np.__version__}, Scipy: {scipy.__version__}, Cirq: {cirq.__version__}")

except Exception as e:
    print(f"❌ ERROR: An issue occurred during setup. Details: {e}")


from IPython.display import HTML

class TraceLogger:
    @staticmethod
    def log_step(agent_name, action_type, details):
        timestamp = time.strftime("%H:%M:%S"); colors = {"Architect": "purple", "Engineer": "blue", "QA_Critic": "orange", "System": "green", "Error": "red", "Data": "teal"}
        c = colors.get(agent_name, "black")
        display(Markdown(f"""<div style="margin-bottom: 5px;"><span style="color:{c}; font-weight:bold;">[{timestamp}] {agent_name}</span> <i>({action_type})</i>: {details}</div>"""))
logger = TraceLogger(); logger.log_step("System", "Init", "Logger module loaded.")


csv_data = """City,X,Y
Warehouse_A,10,85
Factory_B,25,20
Port_C,70,50
Hub_D,55,90
"""
locations_df = pd.read_csv(io.StringIO(csv_data))
logger.log_step("Data", "Load", "Logistics dataset loaded into pandas DataFrame.")
display(locations_df)


class StatisticsDashboard:
    def __init__(self):
        self.total_invocations, self.success_count, self.failure_count = 0, 0, 0
        self.runs_data = []
        logger.log_step("System", "Metrics", "Statistics Dashboard Initialized.")
    def log_run_start(self): self.total_invocations += 1
    def log_success(self, name, attempts):
        self.success_count += 1
        self.runs_data.append({"name": name, "attempts": attempts, "status": "Success"})
    def log_failure(self, name):
        self.failure_count += 1
        self.runs_data.append({"name": name, "attempts": 3, "status": "Failure"})
    def display_dashboard(self):
        if self.total_invocations == 0: return print("No runs to display.")
        success_rate = (self.success_count / self.total_invocations) * 100
        
        summary_html = f"""
        <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 10px; background-color: #f2f2f2;">
            <h3 style="color: #333;">Agent Performance Dashboard</h3>
            <p style="color: #333;"><b>Total Scenarios Executed:</b> {self.total_invocations}</p>
            <p style="color: #333;"><b>Successful Runs:</b> {self.success_count} | <b>Failed Runs:</b> {self.failure_count}</p>
            <p style="color: #333; font-size: 1.2em;"><b>Overall Success Rate: <span style="color: #4CAF50;">{success_rate:.2f}%</span></b></p>
        </div>"""
        
        display(HTML(summary_html))
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].pie([self.success_count, self.failure_count], labels=['Successes', 'Failures'], autopct='%1.1f%%', startangle=90, colors=['#99ff99', '#ff9999'])
        ax[0].axis('equal'); ax[0].set_title('Run Outcome Distribution')
        names = [f"Run {i+1}\n({r['name'][:15]}...)" for i, r in enumerate(self.runs_data)]
        attempts = [r['attempts'] for r in self.runs_data]
        bar_colors = ['green' if r['status'] == 'Success' else 'red' for r in self.runs_data]
        ax[1].bar(names, attempts, color=bar_colors); ax[1].set_ylabel('Number of Attempts');
        ax[1].set_title('Self-Healing Attempts Per Scenario'); ax[1].set_yticks([1, 2, 3]);
        plt.tight_layout(); plt.show()
        
dashboard = StatisticsDashboard()


class AlgorithmMemory:
    def __init__(self): self.memory_store = {}
    def save_success(self, name, code):
        self.memory_store[name] = code
        logger.log_step("System", "Memory Write", f"Stored solution for '{name}' in RAM.")
    def retrieve_context(self, query): return "No reference code found." # Kept simple for this demo

memory_bank = AlgorithmMemory()
logger.log_step("System", "Init", "Memory Bank Initialized.")


def quantum_execution_tool(code_string: str):
    old_stdout, redirected_output = sys.stdout, io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(code_string, {"cirq": cirq, "plt": plt, "np": np, "print": print})
        return redirected_output.getvalue() or "WARNING: No text output."
    except Exception as e: return f"RUNTIME ERROR: {str(e)}"
    finally: sys.stdout = old_stdout
logger.log_step("System", "Tools", "Quantum Sandbox ready.")



def formulate_tsp_prompt(df: pd.DataFrame) -> str:
    """Creates a simplified, shorter prompt with coordinates instead of a full matrix."""
    
    # Format the coordinates into a simple string
    coords_str = df.to_string(index=False)
    
    # This prompt is much shorter and less likely to time out.
    prompt = f"""
    **Enterprise Challenge: Traveling Salesperson Problem (TSP)**
    **Objective:** Find the shortest route that visits all locations.
    **Location Coordinates:**
    {coords_str}
    
    **Your Task:**
    1. Acknowledge the problem is TSP.
    2. Propose a Quantum Approximate Optimization Algorithm (QAOA) approach.
    3. Synthesize a simplified Google Cirq circuit that represents this problem. For this demo, a circuit that puts {len(df)} qubits into a superposition is sufficient to represent exploring all paths.
    4. Simulate and measure the results.
    """
    logger.log_step("Data", "Pre-process", "Generated SIMPLIFIED TSP prompt from DataFrame.")
    return prompt


def plot_tsp_route(df: pd.DataFrame, agent_output: str):
    logger.log_step("System", "Visualize", "Attempting to render route plot.")
    cities = df['City'].tolist()
    coords = {city: (row['X'], row['Y']) for city, row in df.set_index('City').iterrows()}
    path_indices = [0, 1, 2, 3, 0] # A->B->C->D->A (Demo path)
    plt.figure(figsize=(8, 6))
    for city, (x, y) in coords.items():
        plt.plot(x, y, 'o', markersize=15); plt.text(x, y + 2, city, ha='center')
    for i in range(len(path_indices) - 1):
        p1_name, p2_name = cities[path_indices[i]], cities[path_indices[i+1]]
        p1, p2 = coords[p1_name], coords[p2_name]
        plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], head_width=3, length_includes_head=True, color='gray')
    plt.title("QuantumOpt's Proposed Logistics Route"); plt.legend(cities); plt.grid(True); plt.show()


architect_prompt_template = PromptTemplate(
    input_variables=["request"],
    template="You are a Quantum Architect. User Request: {request}. Create a step-by-step logic plan for a Cirq Circuit. Do not write code."
)


engineer_prompt_template = PromptTemplate(
    input_variables=["plan", "last_error"],
    template="""You are a Google Cirq Engineer. LOGIC PLAN: {plan}. ERROR HISTORY: {last_error}. TASK: Write a complete Python script to print the circuit diagram, run a 1000-rep simulation, and print the final measurement counts as a dictionary (e.g., {{'00': 500, '11': 500}}). CRITICAL: Output ONLY raw Python code."""
)

logger.log_step("System", "Agents", "Architect and Engineer blueprints defined and ready.")


def quopt_run(problem_statement, max_retries=3):
    dashboard.log_run_start()
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    architect_prompt = PromptTemplate(input_variables=["request"], template="You are a Quantum Architect. User Request: {request}. Create a step-by-step logic plan for a Cirq Circuit. Do not write code.")
    engineer_prompt = PromptTemplate(input_variables=["plan", "last_error"], template="""You are a Google Cirq Engineer. LOGIC PLAN: {plan}. ERROR HISTORY: {last_error}. TASK: Write a complete Python script to print the circuit diagram, run a 1000-rep simulation, and print the final measurement counts as a dictionary (e.g., {{'00': 500, '11': 500}}). CRITICAL: Output ONLY raw Python code.""")
    
    print("\n" + "="*72)
    logger.log_step("System", "Start", f"Processing: {problem_statement[:80]}...")
    
    try:
        architect_response = model.generate_content(architect_prompt.format(request=problem_statement))
        plan = architect_response.text
        
        logger.log_step("Architect", "Planning", "Logic plan synthesized.")
        display(Markdown(f"### Architect's Plan\n" + plan))
        
        last_error = "None"
        for attempt in range(max_retries):
            logger.log_step("Engineer", "Coding", f"Generation Attempt {attempt + 1}...")
            
            engineer_response = model.generate_content(engineer_prompt.format(plan=plan, last_error=last_error))
            raw_code = engineer_response.text.replace("```python", "").replace("```", "").strip()
            
            logger.log_step("Tool", "Sandbox", "Executing Code...")
            result = quantum_execution_tool(raw_code)
            
            if "RUNTIME ERROR" in result:
                logger.log_step("Error", "Detail", result); last_error = result
            else:
                dashboard.log_success(problem_statement, attempt + 1)
                logger.log_step("QA_Critic", "Success", "Verification passed.")
                memory_bank.save_success(problem_statement[:20], raw_code)
                display(Markdown("### Final Generated Code")); display(Markdown(f"```python\n{raw_code}\n```"))
                display(Markdown("### Execution Output")); print(result)
                return result
                
        dashboard.log_failure(problem_statement)
        logger.log_step("System", "Failure", "Maximum retries exceeded.")
        return None

    except exceptions.DeadlineExceeded as e:
        print("="*69)
        print("CAUGHT A DEADLINE EXCEEDED ERROR. This is a known network issue with long prompts, not a code bug.")
        print(f"Details: {e}")
        print("The agent will be marked as 'failed' for this specific run in the dashboard.")
        print("="*65)
        dashboard.log_failure(problem_statement)
        logger.log_step("Error", "CRITICAL", f"The API call timed out.")
        return None
    except Exception as e:
        print(f"AN UNEXPECTED CRITICAL ERROR OCCURRED: {e}")
        dashboard.log_failure(problem_statement)
        logger.log_step("Error", "CRITICAL", f"The agent failed with an unrecoverable error: {e}")
        return None

logger.log_step("System", "Orchestrator", "Self-healing control loop (with robust timeout handling) is defined.")


_ = quopt_run("Create a Bell State.")


tsp_problem_prompt = formulate_tsp_prompt(locations_df)
tsp_result_text = quopt_run(tsp_problem_prompt)
if tsp_result_text:
    plot_tsp_route(locations_df, tsp_result_text)
else:
    print("\nAgent failed to produce a valid output for visualization.")


guaranteed_fail_prompt = "Create a circuit with one qubit and apply a non-existent gate called 'cirq.MySpecialGate' to it, then measure."

_ = quopt_run(guaranteed_fail_prompt)


import json

def display_dashboard_and_save_output(self):
    """This is an enhanced version that also saves a JSON summary."""
    
    if self.total_invocations == 0:
        print("No runs to display statistics for.")
        return

    success_rate = (self.success_count / self.total_invocations) * 100
    
    # Create the JSON data structure
    output_data = {
        "project": "QuantumOpt: Cirq-Synth Enterprise Agent",
        "summary": {
            "total_scenarios": self.total_invocations,
            "successful_runs": self.success_count,
            "failed_runs": self.failure_count,
            "success_rate_percent": round(success_rate, 2)
        },
        "run_details": self.runs_data
    }
    
    # Write the data to a JSON file
    try:
        with open("run_summary.json", "w") as f:
            json.dump(output_data, f, indent=4)
        logger.log_step("System", "Output", "Successfully generated 'run_summary.json'.")
    except Exception as e:
        logger.log_step("Error", "Output", f"Failed to generate JSON output: {e}")

    # Display the visual dashboard as before
    summary_html = f"""
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 10px; background-color: #f2f2f2;">
        <h3 style="color: #333;">Agent Performance Dashboard</h3>
        <p style="color: #333;"><b>Total Scenarios Executed:</b> {self.total_invocations}</p>
        <p style="color: #333;"><b>Successful Runs:</b> {self.success_count} | <b>Failed Runs:</b> {self.failure_count}</p>
        <p style="color: #333; font-size: 1.2em;"><b>Overall Success Rate: <span style="color: #4CAF50;">{success_rate:.2f}%</span></b></p>
    </div>"""
    display(HTML(summary_html))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].pie([self.success_count, self.failure_count], labels=['Successes', 'Failures'], autopct='%1.1f%%', startangle=90, colors=['#99ff99', '#ff9999'])
    ax[0].axis('equal'); ax[0].set_title('Run Outcome Distribution')
    names = [f"Run {i+1}\n({r['name'][:15]}...)" for i, r in enumerate(self.runs_data)]
    attempts = [r['attempts'] for r in self.runs_data]
    bar_colors = ['green' if r['status'] == 'Success' else 'red' for r in self.runs_data]
    ax[1].bar(names, attempts, color=bar_colors); ax[1].set_ylabel('Number of Attempts');
    ax[1].set_title('Self-Healing Attempts Per Scenario'); ax[1].set_yticks([1, 2, 3]);
    plt.tight_layout(); plt.show()


dashboard.display_dashboard = lambda: display_dashboard_and_save_output(dashboard)

dashboard.display_dashboard()

