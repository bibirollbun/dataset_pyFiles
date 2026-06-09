import time
import json
from typing import Dict, Any, List, Callable

# ==============================================================================
# 1. TOOLS (Simulating Built-in and Custom Tools)
# ==============================================================================

class Tool:
    """Base class for all tools."""
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def use(self, *args, **kwargs) -> Any:
        """Execute the tool function."""
        print(f"    [TOOL: {self.name}] Executing with args: {kwargs}")
        # Simulate latency for long-running operations
        time.sleep(0.5)
        return self.func(*args, **kwargs)

# Built-in Tool Simulation (Google Search)
def google_search_func(query: str) -> str:
    """Simulates a real-time Google Search query."""
    if "latest trend" in query.lower():
        return "Search Result: The latest trend in AI is the rise of 'Micro-Agents' specialized in single tasks."
    return f"Search Result: General data for '{query}' found. Summary: Multi-Agent Systems enhance AI by distributing complex tasks."

# Custom Tool Simulation (MCP - Model Context Protocol)
# This simulates a service that formats output for another model
def mcp_format_output_func(data: Dict[str, Any]) -> str:
    """Formats structured data (like a JSON log) into a concise string for the next agent."""
    summary = f"Formatted Report: Task completed by {data.get('agent_name', 'N/A')}. Result: {data.get('result', 'No result.')}"
    return summary

# Instantiate Tools
google_search_tool = Tool("Google_Search", "A tool to search the internet for current information.", google_search_func)
mcp_tool = Tool("MCP_Output_Formatter", "Formats detailed logs into a simplified output string for subsequent agents.", mcp_format_output_func)
TOOL_KIT = {
    "Google_Search": google_search_tool,
    "MCP_Output_Formatter": mcp_tool
}

# ==============================================================================
# 2. AGENT (LLM-Powered and Specialized)
# ==============================================================================

def simulated_llm_call(prompt: str, session_state: Dict[str, Any]) -> str:
    """
    Simulates the core reasoning/planning step of an LLM, which now uses session state for context.
    This fixes the infinite loop by making the LLM check the state before deciding the next action.
    """
    print(f"    [LLM Thinking] Processing prompt length: {len(prompt)}...")
    
    # Planner LLM logic must be aware of the state to avoid infinite loops
    if "PLAN" in prompt:
        
        # 1. Check if research is done and we need to summarize (Sequential Step 3 decision)
        if session_state.get('status') == 'RESEARCH_DONE':
            print("    [LLM Decision] Research is complete. Transitioning to Summarize.")
            return json.dumps({
                "action": "SUMMARIZE",
                "target_agent": "Summarizer",
                "task": f"Synthesize all state information into a final answer.",
                "tools_required": ["MCP_Output_Formatter"]
            })
        
        # 2. Otherwise, if the initial task is about research and it hasn't been done yet, delegate research (Sequential Step 1 decision)
        elif "research" in prompt.lower() and 'research_result' not in session_state:
            print("    [LLM Decision] Starting research task.")
            return json.dumps({
                "action": "DELEGATE_RESEARCH",
                "target_agent": "Researcher",
                "task": "Find the latest trend in AI.",
                "tools_required": ["Google_Search"]
            })
            
    return "Final Answer: The system execution completed successfully."

class Agent:
    """An autonomous entity with a role, LLM capability, and tools."""
    def __init__(self, name: str, role: str, tools: List[str] = None):
        self.name = name
        self.role = role
        self.tools = tools or []
        # Observability: Basic Agent Logging
        self.log: List[str] = []

    def run(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """The agent's main execution method, taking the session state as input."""
        task = session_state.get('current_task', 'No task defined.')
        print(f"\n[{self.name} - {self.role}] Running task: '{task}'")
        self.log.append(f"Received task: {task}")

        if self.name == "Planner":
            # Agent powered by LLM for dynamic decision-making
            llm_prompt = f"PLAN: I am the Orchestrator. The user's goal is to research a topic and summarize it. The current state is: {session_state}. Decide the next step for task: '{task}'."
            
            # FIX: Pass the session_state to the LLM simulation for context-aware routing
            decision_json_str = simulated_llm_call(llm_prompt, session_state)
            
            try:
                decision = json.loads(decision_json_str)
                session_state['llm_decision'] = decision
                self.log.append(f"LLM Decision: {decision.get('action', 'N/A')}")
            except json.JSONDecodeError:
                session_state['llm_decision'] = {"action": "ERROR", "message": "Invalid LLM output."}

        elif self.name == "Researcher":
            # Agent executing a Tool Call
            if "Google_Search" in self.tools and task:
                result = TOOL_KIT["Google_Search"].use(query=task)
                session_state['research_result'] = result
                self.log.append(f"Tool Result Stored: {result}")
            else:
                session_state['research_result'] = "Error: Tool or task missing."
                self.log.append("Error: Tool or task missing.")

        elif self.name == "Summarizer":
            # Agent using MCP logic to format data from previous agents
            data_to_format = {
                "agent_name": "Researcher",
                "result": session_state.get('research_result', 'N/A'),
                "context": session_state.get('initial_query', 'N/A')
            }
            formatted_output = TOOL_KIT["MCP_Output_Formatter"].use(data=data_to_format)
            session_state['final_summary'] = formatted_output
            self.log.append(f"Final Summary generated: {formatted_output}")

        # Basic Observability: Add the agent's log history to the session state
        session_state[f'{self.name}_log'] = self.log

        return session_state

# ==============================================================================
# 3. CONTROLLER (Sequential & Loop Logic with Session/State Management)
# ==============================================================================

class SequentialAgentController:
    """Orchestrates agents, manages the session, and enforces workflow logic."""
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        # Sessions & Memory: In-Memory Session Service simulation
        self.session_state: Dict[str, Any] = {
            'session_id': 'mas_101_demo',
            'status': 'STARTING',
            'history': []
        }
        # Observability: Tracing/Execution Log
        self.trace_log: List[Dict[str, Any]] = []

    def update_state(self, key: str, value: Any, trace_step: str):
        """Context Engineering/State Management - Updates state and logs the change."""
        self.session_state[key] = value
        self.trace_log.append({"step": trace_step, "key_updated": key, "value_preview": str(value)[:50] + '...'})

    def run_workflow(self, initial_query: str):
        """Executes a sequential workflow with a loop component."""
        print(f"--- Workflow Start (Session ID: {self.session_state['session_id']}) ---")
        self.update_state('initial_query', initial_query, 'Initial User Prompt')
        self.update_state('current_task', 'Research and summarize the latest trend in AI.', 'Initial Task Defined')
        
        current_agent_name = "Planner"
        loop_counter = 0
        MAX_ITERATIONS = 5 # Increased for safety, though only 3 are now needed
        self.update_state('status', 'PLANNING_LOOP', 'Starting Loop Agent Logic')

        while self.session_state['status'] != 'COMPLETED' and loop_counter < MAX_ITERATIONS:
            loop_counter += 1
            
            # --- Sequential Agent Execution ---
            if current_agent_name not in self.agents:
                print(f"ERROR: Agent '{current_agent_name}' not found.")
                break
            
            agent = self.agents[current_agent_name]
            self.session_state = agent.run(self.session_state)
            
            decision = self.session_state.get('llm_decision', {})
            action = decision.get('action')
            
            # --- Conditional/Routing Logic (Sequential Flow) ---
            if action == "DELEGATE_RESEARCH":
                # Sequential Step 1: Planner delegates to Researcher
                self.update_state('current_task', decision['task'], 'Delegated Research Task')
                current_agent_name = "Researcher"
                
            elif current_agent_name == "Researcher":
                # Sequential Step 2: Researcher completes, hands off to Planner for next step decision
                # NOTE: The status update here is key for the Planner's LLM to route correctly next turn.
                self.update_state('status', 'RESEARCH_DONE', 'Research Completed')
                current_agent_name = "Planner"
                
            elif current_agent_name == "Planner" and self.session_state['status'] == 'RESEARCH_DONE':
                # Planner's second turn - LLM routing logic should now return "SUMMARIZE"
                if action == "SUMMARIZE":
                    self.update_state('status', 'SUMMARIZING', 'Planner decides to Summarize')
                    current_agent_name = "Summarizer"
                
            elif current_agent_name == "Summarizer":
                # Sequential Step 3: Summarizer completes, workflow ends
                self.update_state('status', 'COMPLETED', 'Final Summary Generated')
                break # Exit the loop (Loop Agent Termination Condition)

            elif action == "ERROR":
                self.update_state('status', 'FAILED', 'LLM returned error')
                break

            print(f"--- Iteration {loop_counter} finished. Next Agent: {current_agent_name} ---")
        
        # Add final iteration count for evaluation
        self.session_state['total_iterations'] = loop_counter

        # --- Final Output and Observability ---
        print("\n--- Workflow Execution Trace (Observability) ---")
        for log_entry in self.trace_log:
            print(f"[{log_entry['step']}]: {log_entry['key_updated']} -> {log_entry['value_preview']}")
        
        print("\n--- Final Session State (Long Term Memory/Context) ---")
        print(json.dumps(self.session_state, indent=2))
        print("--- Workflow End ---")

# ==============================================================================
# 4. EXECUTION
# ==============================================================================

# 1. Define Agents (Each with specialized tools)
planner_agent = Agent("Planner", "Orchestrates the workflow and determines the next step (LLM-Powered).", [])
researcher_agent = Agent("Researcher", "Executes data retrieval tasks.", ["Google_Search"])
summarizer_agent = Agent("Summarizer", "Synthesizes final reports.", ["MCP_Output_Formatter"])

all_agents = {
    "Planner": planner_agent,
    "Researcher": researcher_agent,
    "Summarizer": summarizer_agent,
}

# 2. Initialize Controller
controller = SequentialAgentController(all_agents)

# 3. Run the complete Multi-Agent Workflow
controller.run_workflow("I need a concise report on the latest AI trend.")

# Agent Evaluation (A simple post-run check)
print("\n--- Agent Evaluation (Post-Run Metrics) ---")
print(f"Total loop iterations: {controller.session_state.get('total_iterations', 'N/A')}")
print(f"Planner Log Entries: {len(planner_agent.log)}")
print(f"Researcher Log Entries: {len(researcher_agent.log)}")
print(f"Final Status: {controller.session_state['status']}")

