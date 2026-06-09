# --- 1. INSTALL LIBRARIES (Required after restart) ---
!pip install -q -U langchain-google-genai langgraph langchain langchain-core
print("âœ… Libraries installed.")


# --- 2. IMPORTS & SETUP ---
import os
import google.generativeai as genai
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph import StateGraph, END
from IPython.display import display, Markdown
from kaggle_secrets import UserSecretsClient

print("âš™ï¸� Setting up Agent...")


# --- SETUP API KEY ---
from kaggle_secrets import UserSecretsClient
import os

try:
    user_secrets = UserSecretsClient()
    # We retrieve the key you saved in Add-ons -> Secrets
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
    print("âœ… Google API Key loaded.")
except Exception as e:
    print("â�Œ Error: Could not find 'GOOGLE_API_KEY'. Check the Add-ons -> Secrets menu.")

model = genai.GenerativeModel('gemini-2.5-flash')


# --- 2. HELPER FUNCTION ---
def call_gemini(prompt: str):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- 3. STATE ---
class AgentState(TypedDict):
    user_request: str
    energy_level: str
    micro_steps: List[str]
    current_step: str
    user_feedback: str
    final_response: str
    music_link: str

# --- 4. NODES ---

def intake_node(state: AgentState):
    """Observer: Checks energy"""
    # Mocking a 'busy' calendar to trigger the logic
    energy = "Low" 
    return {"energy_level": energy}

def decomposer_node(state: AgentState):
    """Planner: Breaks goal into steps"""
    prompt = f"""
    Task: {state['user_request']}
    Energy Level: {state['energy_level']}
    Break this into 3 simple steps. Return ONLY a Python list of strings.
    """
    response_text = call_gemini(prompt)
    clean_text = response_text.replace("```python", "").replace("```", "").strip()
    steps = [line.strip('- "[]\'') for line in clean_text.split('\n') if line.strip()]
    if not steps: steps = ["Start Task", "Middle Task", "Finish Task"]
    
    return {"micro_steps": steps, "current_step": steps[0]}

def negotiator_node(state: AgentState):
    """Interface: Proposes the step"""
    prompt = f"""
    The user needs to do: "{state['current_step']}".
    Write ONE short, encouraging sentence asking if they can do this small step right now.
    """
    msg = call_gemini(prompt)
    print(f"\nğŸ¤– Agent: {msg.strip()}") # Print directly so user sees it before input
    return {"final_response": msg.strip()}

def human_input_node(state: AgentState):
    """The STOP Sign: Waits for user input"""
    user_input = input("You (Yes/No): ")
    return {"user_feedback": user_input}

def replanner_node(state: AgentState):
    """Loop Logic: Breaks it down further"""
    rejected = state['current_step']
    prompt = f"""
    The user refused to do: "{rejected}". It was too hard.
    Break "{rejected}" into 2 TINY micro-steps. Return ONLY a list of strings.
    """
    response_text = call_gemini(prompt)
    new_steps = [line.strip('- "[]\'') for line in response_text.split('\n') if line.strip()]
    if not new_steps: new_steps = ["Do part 1", "Do part 2"]
    
    updated_plan = new_steps + state['micro_steps'][1:]
    return {
        "micro_steps": updated_plan, 
        "current_step": updated_plan[0],
        "final_response": "I hear you. That was too much. Let's make it smaller."
    }

def music_node(state: AgentState):
    """Reward Tool"""
    if state['energy_level'] == "Low":
        url = "https://www.youtube.com/watch?v=jfKfPfyJRdk"
        vibe = "Gentle Lo-Fi"
    else:
        url = "https://www.youtube.com/watch?v=xcCpW6-Wp7s"
        vibe = "High Energy Focus"
        
    # NEW: Clearer explanation
    explanation = f"Great job committing to the task! I'm playing some {vibe} background music to help you focus while you work on that step."
    
    print(f"\nğŸ�§ DJ Agent: {explanation}")
    display(Markdown(f"**[CLICK HERE TO START MUSIC]({url})**"))
    
    return {"music_link": url, "final_response": explanation}

# --- 5. ROUTER & GRAPH ---
def router(state: AgentState) -> Literal["replanner", "music_node"]:
    feedback = state['user_feedback'].lower()
    if any(x in feedback for x in ["no", "hard", "can't"]):
        return "replanner"
    else:
        return "music_node"

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("observer", intake_node)
workflow.add_node("decomposer", decomposer_node)
workflow.add_node("negotiator", negotiator_node)
workflow.add_node("human", human_input_node) 
workflow.add_node("replanner", replanner_node)
workflow.add_node("music_node", music_node)

# Set Edges
workflow.set_entry_point("observer")
workflow.add_edge("observer", "decomposer")
workflow.add_edge("decomposer", "negotiator")
workflow.add_edge("negotiator", "human")
workflow.add_conditional_edges("human", router, {"replanner": "replanner", "music_node": "music_node"})
workflow.add_edge("replanner", "negotiator") 
workflow.add_edge("music_node", END)

app = workflow.compile()
print("âœ… Agent Compiled Successfully.")

# --- 6. RUN IT ---
def run_session():
    print("\nğŸ¤– EXECUTIVE FUNCTION EXOSKELETON (Ready)")
    print("------------------------------------------")
    goal = input("What is your goal? ")
    
    state = {
        "user_request": goal,
        "energy_level": "Unknown",
        "micro_steps": [],
        "current_step": "",
        "user_feedback": "", 
        "final_response": "",
        "music_link": ""
    }
    
    app.invoke(state)

if __name__ == "__main__":
    run_session()




