!pip install -U google-generativeai
!pip install google-generativeai --quiet


import pandas as pd
import logging
import time
from google import generativeai as genai
from kaggle_secrets import UserSecretsClient
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Load Google API Key securely from Colab Secrets

# The UserSecretsClient allows you to securely store and access
# private keys in Google Colab without exposing them in the notebook.
user_secrets = UserSecretsClient()

# Retrieve the API key stored under the name "Gemini_LLM".
# This keeps the key hidden and prevents accidental leaks.
GOOGLE_API_KEY = user_secrets.get_secret("Gemini_LLM")

# Configure the Gemini client with the retrieved API key.
# This enables all subsequent LLM calls to authenticate using this key.
genai.configure(api_key=GOOGLE_API_KEY)


# Lists all LLM models available in your Google Generative AI account
models = genai.list_models()

# Prints each model name
for model in models:
    print(model.name)


# Model LLM
llm = genai.GenerativeModel("models/gemini-2.5-flash")


# Test prompt 1
resp = llm.generate_content("Explain RNA-seq in 2 lines.")
print(resp.text)


# Test prompt 2
resp2 = llm.generate_content("Explain RNA-seq in 2 lines with temporal dilation.")
print(resp2.text)


# Database
df = pd.read_csv("/kaggle/input/covid19-clinical-trials-dataset/COVID clinical trials.csv")

# View database
df.head()


df.columns.tolist()


class Agent:
    def __init__(self, name, description, model):
        # Store agent metadata
        self.name = name
        self.description = description
        self.model = model

    def run(self, state):
        # Must be implemented in subclasses
        raise NotImplementedError("Implement in subclasses")


class IntakeAgent(Agent):
    def run(self, state):
        # Read user description from the pipeline state
        user_text = state["user_description"]

        # Prompt instructing the model to convert free-text into JSON
        prompt = f"""
You are a medical intake assistant.

Convert the following patient description into a JSON object with the fields:
- age (number)
- sex (string)
- condition (string)
- symptoms (list of strings)
- comorbidities (list of strings)
- location (string)

Return ONLY valid JSON with no markdown.
User description: {user_text}
"""

        # Call the LLM
        resp = llm.generate_content(prompt)

        # Clean model output in case markdown fences appear
        cleaned = resp.text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "")

        # Attempt to parse JSON safely
        try:
            profile = json.loads(cleaned)
        except:
            # Fallback to raw text if parsing fails
            profile = {"raw_profile": cleaned}

        # Store parsed patient profile back into state
        state["patient_profile"] = profile

        # Return updated state
        return state


# Base agent class that all agents inherit from
class Agent:
    # Initialize the agent with name, description, and model
    def __init__(self, name: str, description: str, model):
        self.name = name
        self.description = description
        self.model = model

    # Main execution method, must be implemented by subclasses
    def run(self, state: dict) -> dict:
        raise NotImplementedError("Subclasses must implement run().")


# Configure basic logging behavior for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Simple in-memory key-value store used to pass information between agents
class MemoryStore:
    def __init__(self):
        # Internal dictionary to hold stored values
        self.store = {}

    # Save a value under a specific key
    def save(self, key, value):
        self.store[key] = value

    # Retrieve a value associated with a key
    def get(self, key):
        return self.store.get(key)

# Create a global memory store instance
memory = MemoryStore()


# Base class for all agents in the pipeline
class Agent:
    def __init__(self, name, description, model):
        # Store agent metadata
        self.name = name
        self.description = description
        self.model = model

    # Method that must be implemented by subclasses
    def run(self, state):
        raise NotImplementedError


# Load clinical trials dataset
df = pd.read_csv("/kaggle/input/covid19-clinical-trials-dataset/COVID clinical trials.csv")

# Remove rows where essential fields are missing
df = df.dropna(subset=["Title", "Conditions"])

# Create a small sample for quick testing
df_small = df.sample(300, random_state=42).reset_index(drop=True)

# Display preview
df_small.head()


def count_trials_by_condition(condition: str):
    """
    Count how many clinical trials match a given condition.

    Returns:
        A dictionary containing:
        - condition: searched condition
        - total_trials: number of trials found
        - examples: first 3 matching trials with key fields
    """
    # Filter trials where the condition appears in the "Conditions" column
    subset = df_small[df_small["Conditions"].str.contains(condition, case=False, na=False)]

    # Build the result payload
    return {
        "condition": condition,
        "total_trials": len(subset),
        "examples": subset[["NCT Number", "Title", "Phases", "Status"]]
                       .head(3)
                       .to_dict(orient="records")
    }

# Tool registry used by the agent
TOOLS = {
    "count_trials_by_condition": count_trials_by_condition
}


# IntakeAgent: extract structured patient data
class IntakeAgent(Agent):
    def run(self, state):
        logging.info("Running IntakeAgent")

        user_text = state["user_text"]

        prompt = f"""
Extract the following fields and return valid JSON only:
- age
- sex
- condition
- symptoms
- comorbidities
- location

Text:
{user_text}
"""

        response = self.model.generate_content(prompt)
        profile_json = response.text

        state["patient_profile"] = profile_json
        memory.save("patient_profile", profile_json)

        return state


# RetrievalAgent: retrieve clinical trials for a condition
class RetrievalAgent(Agent):
    def run(self, state):
        logging.info("Running RetrievalAgent")

        condition = "COVID"

        import numpy as np

        subset = df_small[df_small["Conditions"].str.contains(condition, case=False, na=False)]
        subset = subset.replace({np.nan: None})

        examples = (
            subset[["NCT Number", "Title", "Phases", "Status"]]
            .head(3)
            .to_dict(orient="records")
        )

        tool_output = {
            "condition": condition,
            "total_trials": len(subset),
            "examples": examples
        }

        state["tool_result"] = tool_output
        memory.save("tool_result", tool_output)

        if "plot_trial_status" in TOOLS:
            try:
                plot_path = TOOLS["plot_trial_status"]()
                state["status_plot_path"] = plot_path
                memory.save("status_plot_path", plot_path)
            except Exception as e:
                logging.error(f"Plot tool failed: {e}")
                state["status_plot_path"] = None

        return state


# ExplainerAgent: generates a simple-language explanation
class ExplainerAgent(Agent):
    def run(self, state):
        logging.info("Running ExplainerAgent")

        # Retrieve previous tool output
        tool_info = memory.get("tool_result")

        # Build instruction for the LLM
        prompt = f"""
Explain the following clinical trial information in simple, patient-friendly language:

{tool_info}
"""

        response = self.model.generate_content(prompt)
        explanation = response.text

        # Save explanation back to state and memory
        state["explanation"] = explanation
        memory.save("explanation", explanation)

        return state


class ReportAgent(Agent):
    def run(self, state):
        logging.info("Running ReportAgent")

        # Load stored components
        patient = memory.get("patient_profile")
        tool_res = memory.get("tool_result")
        explanation = memory.get("explanation")
        plot_path = memory.get("status_plot_path")

        import json

        # Parse patient JSON
        try:
            cleaned = (
                patient.replace("```json", "")
                       .replace("```", "")
                       .strip()
            )
            patient_json = json.dumps(json.loads(cleaned), indent=2)
        except:
            patient_json = "{}"

        # Parse tool result JSON
        try:
            tool_json = json.dumps(tool_res, indent=2)
        except:
            tool_json = "{}"

        # Build the final report
        final_report = (
            "# ğŸ§­ Clinical Trials Navigator â€” Final Report\n\n"
            "## Patient Profile\n"
            "```json\n"
            f"{patient_json}\n"
            "```\n\n"
            "## Trials Found\n"
            "```json\n"
            f"{tool_json}\n"
            "```\n\n"
            "## Explanation\n"
            f"{explanation}\n\n"
        )

        # Attach plot information if available
        if plot_path:
            final_report += (
                "## Trial Status Chart\n"
                f"Plot saved at: `{plot_path}`\n\n"
            )

        # Add disclaimer
        final_report += (
            "## Medical Disclaimer\n"
            "This report is generated by an AI agent and is not medical advice.\n"
            "Always consult a licensed healthcare professional.\n"
        )

        # Save report
        state["final_report"] = final_report
        memory.save("final_report", final_report)

        return state



# Runs all agents and returns the final state
def run_pipeline(user_text):
    # Initialize shared state with the user text
    state = {"user_text": user_text}

    # List of agents executed in sequence
    agents = [
        IntakeAgent("intake", "Extract patient info", llm),
        RetrievalAgent("retrieval", "Search dataset with tools", llm),
        ExplainerAgent("explainer", "Explain results", llm),
        ReportAgent("report", "Generate final report", llm)
    ]

    # Execute each agent step-by-step
    for agent in agents:
        t1 = time.time()
        state = agent.run(state)
        t2 = time.time()

        # Log execution time
        logging.info(f"{agent.name} took {t2 - t1:.2f}s")

    # Return the enriched state after all agents run
    return state


# Example patient description
user_text = (
    "My father is 62 years old, male, diabetic, with breathing problems after COVID-19, "
    "living in Brazil. We want to know if there are any COVID-19 clinical trials "
    "that might be relevant to his situation."
)

# Run the full multi-agent pipeline
result_state = run_pipeline(user_text)

# Print the final report generated by the agents
print(result_state["final_report"])


import time
import json
import traceback
from IPython.display import display, Markdown

def evaluate_agent(input_text):
    """
    Evaluation harness for the multi-agent pipeline.
    Runs the full pipeline, measures performance,
    validates outputs, and returns a structured report.
    """

    print("ğŸ”� Starting Agent Evaluation...\n")

    # Main evaluation container
    eval_report = {
        "input_text": input_text,
        "steps": [],
        "errors": [],
        "start_time": time.time(),
    }

    # Run the full pipeline
    t0 = time.time()
    try:
        state_output = run_pipeline(input_text)
        t1 = time.time()

        eval_report["steps"].append({
            "agent": "Pipeline (All Agents Sequential)",
            "duration_sec": round(t1 - t0, 3),
            "state_keys": list(memory.store.keys()),
        })

    except Exception:
        t1 = time.time()
        error_msg = traceback.format_exc()

        eval_report["errors"].append(error_msg)
        eval_report["steps"].append({
            "agent": "Pipeline FAILED",
            "duration_sec": round(t1 - t0, 3),
        })

        print("â�Œ Pipeline Failed\n")
        print(error_msg)
        return eval_report

    # Extract final outputs
    patient = memory.get("patient_profile")
    trials = memory.get("tool_result")
    explanation = memory.get("explanation")

    # Validation rules
    validation = {
        "patient_parsed": isinstance(patient, str) and len(patient) > 5,
        "tool_used": isinstance(trials, dict),
        "has_examples": isinstance(trials.get("examples", []), list),
        "explanation_ok": isinstance(explanation, str) and len(explanation) > 20,
    }

    score = sum(validation.values()) / len(validation)

    eval_report["validation"] = validation
    eval_report["final_score"] = round(score, 2)
    eval_report["total_runtime_sec"] = round(time.time() - eval_report["start_time"], 2)

    # UI summary
    display(Markdown("## âœ… Agent Evaluation Summary"))
    display(Markdown(f"**Final Score:** `{round(score * 100)}%`"))
    display(Markdown(f"**Runtime:** `{eval_report['total_runtime_sec']} sec`"))

    display(Markdown("### âœ” Validation Results"))
    display(Markdown("```json\n" + json.dumps(validation, indent=2) + "\n```"))

    display(Markdown("### ğŸ”� Pipeline State Keys"))
    display(Markdown("```json\n" + json.dumps(list(memory.store.keys()), indent=2) + "\n```"))

    display(Markdown("### ğŸ§ª Scenario Input"))
    display(Markdown(f"```text\n{input_text}\n```"))

    display(Markdown("### ğŸ“¦ Raw Trial Tool Output"))
    display(Markdown("```json\n" + json.dumps(trials, indent=2) + "\n```"))

    return eval_report


evaluate_agent("""
Patient: 62-year-old male with long COVID and breathing problems.
Comorbidities: diabetes.
Location: Brazil.
""")



# Filter condition keyword
condition = "COVID"

# Select rows containing the condition
subset = df_small[df_small["Conditions"].str.contains(condition, case=False, na=False)]

# Replace NaN with None for JSON compatibility
subset = subset.replace({np.nan: None})

# Display filtered DataFrame
subset


# Create output DataFrame
df_output = pd.DataFrame([{
    "patient_profile": memory.get("patient_profile"),
    "tool_result": memory.get("tool_result"),
    "explanation": memory.get("explanation"),
    "final_report": memory.get("final_report")
}])

# Save as Excel (optional)
df_output.to_excel("/kaggle/working/\covid_trials_agent_results.xlsx",index=False)

# Display DataFrame
df_output


def plot_trial_status():
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Count and sort status categories
    counts = df_small["Status"].value_counts().sort_values()
    total = counts.sum()
    pct = (counts / total * 100).round(1)

    # 2. Advanced color palette (colors by category)
    color_map = {
        "Recruiting": "#2ecc71",
        "Completed": "#3498db",
        "Not yet recruiting": "#9b59b6",
        "Active, not recruiting": "#f1c40f",
        "Enrolling by invitation": "#1abc9c",
        "Withdrawn": "#e67e22",
        "Terminated": "#e74c3c",
        "Suspended": "#d35400",
        "No longer available": "#7f8c8d"
    }

    colors = [color_map.get(status, "#4ea8de") for status in counts.index]

    # 3. Plot horizontal bar chart
    plt.figure(figsize=(12, 7), dpi=120)
    bars = plt.barh(counts.index, counts.values, color=colors)

    plt.title("COVID-19 Clinical Trials by Recruitment Status", fontsize=16, weight="bold")
    plt.xlabel("Number of Trials", fontsize=13)
    plt.ylabel("Recruitment Status", fontsize=13)

    # 4. Add labels with count + percentage
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label = f"{int(width)}  ({pct.iloc[i]}%)"
        plt.text(width + 2,
                 bar.get_y() + bar.get_height() / 2,
                 label,
                 va='center',
                 fontsize=11)

    # Clean grid
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("trial_status.png", dpi=120)
    plt.close()

    return "trial_status.png"


TOOLS["plot_trial_status"] = plot_trial_status


from IPython.display import Image
plot_trial_status()
Image("/kaggle/working/trial_status.png")


import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)

# ============================================================
# Corrigir IntakeAgent para usar input_text
# ============================================================

def fix_intake_agent():
    # sobrescreve mÃ©todo run do IntakeAgent
    def run_fixed(self, state):
        logging.info("Running IntakeAgent")
        text = state["input_text"]   # <<< corrigido aqui

        prompt = f"""
Extract a structured patient profile from this text.
Return JSON only.

{text}
"""
        response = self.model.generate_content(prompt)
        profile = response.text.strip()

        # Salvar no estado
        state["patient_profile"] = json.loads(profile)
        return state

    IntakeAgent.run = run_fixed

fix_intake_agent()


# ============================================================
# 1. MÃ©tricas avanÃ§adas
# ============================================================

def measure_agent(agent_name, fn, state):
    start = time.time()
    new_state = fn(state)
    end = time.time()

    duration = round(end - start, 3)
    logging.info(f"{agent_name} took {duration}s")
    return new_state, duration


# ============================================================
# 2. ExecuÃ§Ã£o paralela (corrigida)
# ============================================================

def run_parallel_intake_retrieval(state):

    intake = IntakeAgent("intake", "Parse patient", llm)
    retr   = RetrievalAgent("retrieval", "Find trials", llm)

    logging.info("Running Parallel Agents: Intake + Retrieval")

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(intake.run, state)
        f2 = executor.submit(retr.run, state)

        s1 = f1.result()
        s2 = f2.result()

    return {**state, **s1, **s2}


# ============================================================
# 3. ValidaÃ§Ã£o A2A
# ============================================================

def validate_state(state):
    return {
        "patient_parsed": isinstance(state.get("patient_profile"), dict),
        "tool_used": "tool_result" in state,
        "has_examples": bool(state.get("tool_result", {}).get("examples")),
        "explanation_ok": len(state.get("explanation", "")) > 10
    }

def compute_score(v):
    return sum(v.values()) / len(v)


# ============================================================
# 4. JSON Export
# ============================================================

def export_run(full_state, filename="agent_run_export.json"):
    with open(filename, "w") as f:
        json.dump(full_state, f, indent=2)


# ============================================================
# 5. AvaliaÃ§Ã£o FINAL (Sem Markdown)
# ============================================================

LEADERBOARD = []

def evaluate_agent_advanced(user_text):

    state = {"input_text": user_text}
    steps = []

    print("\n=== Advanced Agent Evaluation Started ===\n")
    total_start = time.time()

    # Parallel step ----------------
    state, t = measure_agent(
        "Parallel(Intake + Retrieval)",
        run_parallel_intake_retrieval,
        state
    )
    steps.append(("Parallel Intake/Retrieval", t))

    # Explainer --------------------
    expl = ExplainerAgent("explainer", "Explain trials", llm)
    state, t = measure_agent("Explainer", expl.run, state)
    steps.append(("Explainer", t))

    # Report -----------------------
    rep = ReportAgent("report", "Generate report", llm)
    state, t = measure_agent("Report", rep.run, state)
    steps.append(("Report", t))

    total_runtime = round(time.time() - total_start, 3)

    # Validation
    validation = validate_state(state)
    score = compute_score(validation)

    # Save final structure
    full_state = {
        "input": user_text,
        "steps": steps,
        "state": state,
        "validation": validation,
        "score": score,
        "runtime_sec": total_runtime
    }

    # Leaderboard
    LEADERBOARD.append({
        "input": user_text[:50],
        "score": score,
        "runtime": total_runtime
    })

    # Export JSON
    export_run(full_state)

    # Output simples (sem markdown)
    print("\n=== Evaluation Complete ===")
    print("Score:", round(score*100, 1), "%")
    print("Runtime:", total_runtime, "sec")
    print("\nValidation:", validation)
    print("\nSteps:", steps)
    print("\nState keys:", list(state.keys()))

    return full_state


# FIX â€” JSON SAFE PARSER WITH FALLBACK

import re
import json
import logging

def safe_json(text):
    """
    Extracts valid JSON from an LLM response.
    If JSON cannot be parsed, returns a minimal safe dictionary.
    """
    if not text or not isinstance(text, str):
        return {"error": "empty response"}

    # 1) Attempt direct JSON load
    try:
        return json.loads(text)
    except:
        pass

    # 2) Try extracting JSON between braces { }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    # 3) Safe minimal fallback
    return {"raw_text": text.strip(), "parsed": False}


def fix_intake_agent():
    def run_fixed(self, state):
        logging.info("Running IntakeAgent")

        text = state["input_text"]

        prompt = f"""
Extract a structured patient profile from the following text.
Return JSON only.

{text}
"""
        response = self.model.generate_content(prompt)
        raw_profile = response.text.strip()

        # Safe JSON parsing
        profile_json = safe_json(raw_profile)

        state["patient_profile"] = profile_json
        return state

    IntakeAgent.run = run_fixed


fix_intake_agent()



# Test Example
test_input = """
Patient: 62-year-old male with long COVID and breathing problems.
Comorbidities: diabetes.
Location: Brazil.
"""

# Run the full multi-agent evaluation
result = evaluate_agent_advanced(test_input)

# Print final output report
print(result["state"]["final_report"])

