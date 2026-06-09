import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from kaggle_secrets import UserSecretsClient
from IPython.display import display, HTML, clear_output

print("âœ“ Libraries Loaded")


# Load API Key from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ“ API Key Configured")
except Exception as e:
    print(f"âš  API Key Error: {str(e)}")
    print("ğŸ“Œ To fix: Go to Add-ons â†’ Secrets â†’ Add 'GOOGLE_API_KEY'")
    GOOGLE_API_KEY = None

# Agent Configuration
CONFIG = {
    "team": "GlobeTrotter",
    "model": "models/gemini-2.5-flash",
    "max_tokens": 2000,
    "temperature": 0.3,
    "version": "2.0.0"
}

print(f"\n{'='*60}")
print(f"{'AGENT CONFIGURATION':^60}")
print(f"{'='*60}")
for k, v in CONFIG.items():
    print(f"{k:.<25} {v}")
print(f"{'='*60}")


def propose_model_upgrades(model_name: str, current_score: float, target_score: float) -> str:
    """
    Produce 5 practical, high-impact suggestions to improve a model's performance.

    Returns:
        A short actionable list covering architecture, training workflow, augmentations,
        ensembling ideas, and hyperparameter strategies.
    """
    prompt = (
        "You are an expert ML competitor. Provide **5 concrete, ranked suggestions** to boost model performance.\n\n"
        f"Model: {model_name}\n"
        f"Current validation score: {current_score}\n"
        f"Desired target score: {target_score}\n\n"
        "Each suggestion should be 1â€“2 sentences and include:\n"
        "- what to change (e.g., architecture or augmentation)\n"
        "- why it helps\n"
        "- a short implementation hint or example\n"
    )

    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text


def build_competition_plan(objective: str, days_available: int, status_summary: str) -> str:
    """
    Create a daily step-by-step plan for a competition sprint.

    Returns:
        A day-by-day checklist with time allocations, success criteria, risks, and required resources.
    """
    prompt = (
        "Draft a clear day-by-day competition plan for a data-science sprint.\n\n"
        f"Objective: {objective}\n"
        f"Time horizon (days): {days_available}\n"
        f"Current position / notes: {status_summary}\n\n"
        "For each day include:\n"
        "- Key tasks\n"
        "- Estimated hours\n"
        "- Metrics to show progress\n"
        "- Potential blockers and mitigations\n"
        "- Tools/compute needed\n"
    )

    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text


def troubleshoot_code(error_message: str, code_snippet: str, ecosystem: str = "general") -> str:
    """
    Analyze an error and propose a corrective patch plus testing suggestions.

    Returns:
        Root cause, corrected code (or patch), best practices, and how to verify the fix.
    """
    prompt = (
        "You are a senior engineer who debugs production ML code.\n\n"
        f"Ecosystem: {ecosystem}\n"
        f"Observed error: {error_message}\n\n"
        "Suspect code (begin block):\n"
        "```python\n"
        f"{code_snippet}\n"
        "```\n\n"
        "Deliverables:\n"
        "- Most likely root cause (brief)\n"
        "- Minimal corrected code or patch (marked clearly)\n"
        "- One or two best-practice recommendations\n"
        "- A simple test/validation to confirm the fix\n"
    )

    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text


def recommend_features(data_summary: str, target: str, existing_cols: str) -> str:
    """
    Suggest advanced feature engineering options.

    Returns:
        Exactly 10 feature ideas, interaction features, brief sample code for each,
        and an impact ranking (high/medium/low).
    """
    prompt = (
        "Act as a senior feature engineer. Propose **10 new features** with short code snippets.\n\n"
        f"Dataset summary: {data_summary}\n"
        f"Predicting: {target}\n"
        f"Current features: {existing_cols}\n\n"
        "For each feature include:\n"
        "- Name and short rationale\n"
        "- One-line example (pandas / numpy)\n"
        "- Expected impact (High / Medium / Low)\n    "
    )

    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text


def synthesize_community_insights(topic: str, source_count: int = 5) -> str:
    """
    Summarize common and advanced approaches from community discussions.

    Returns:
        A concise synthesis of winning patterns, usual pitfalls, novel tricks, code idioms, and recommended next steps.
    """
    prompt = (
        "Synthesize community wisdom on the following competition topic.\n\n"
        f"Topic: {topic}\n"
        f"Aggregate knowledge from roughly {source_count} representative threads/papers/solutions.\n\n"
        "Output sections:\n"
        "- Top successful strategies\n"
        "- Recurrent mistakes to avoid\n"
        "- One or two novel techniques worth trying\n"
        "- Useful code patterns or libs\n"
        "- Tactical recommendations for a competitor\n"
    )

    model = genai.GenerativeModel(CONFIG["model"])
    return model.generate_content(prompt).text
# --- Compatibility wrappers (create old names pointing to the new implementations) ---
# Only define wrappers if original names are missing to avoid overriding existing implementations.

if 'suggest_model_improvements' not in globals():
    def suggest_model_improvements(model_type: str, current_score: float, target_score: float) -> str:
        # delegate to new function name
        return propose_model_upgrades(model_name=model_type, current_score=current_score, target_score=target_score)

if 'create_competition_strategy' not in globals():
    def create_competition_strategy(goal: str, timeframe_days: int, current_position: str) -> str:
        return build_competition_plan(objective=goal, days_available=timeframe_days, status_summary=current_position)

if 'debug_code_issue' not in globals():
    def debug_code_issue(error_message: str, code_context: str, framework: str = "general") -> str:
        return troubleshoot_code(error_message=error_message, code_snippet=code_context, ecosystem=framework)

if 'suggest_features' not in globals():
    def suggest_features(dataset_description: str, target_variable: str, current_features: str) -> str:
        return recommend_features(data_summary=dataset_description, target=target_variable, existing_cols=current_features)

if 'analyze_competition_insights' not in globals():
    def analyze_competition_insights(topic: str, num_sources: int = 5) -> str:
        return synthesize_community_insights(topic=topic, source_count=num_sources)
# -------------------------------------------------------------------------------------


# summary printout
print("âœ“ 5 tool functions ready")
print("  â€¢ propose_model_upgrades")
print("  â€¢ build_competition_plan")
print("  â€¢ troubleshoot_code")
print("  â€¢ recommend_features")
print("  â€¢ synthesize_community_insights")



function_declarations = [
    FunctionDeclaration(
        name="propose_model_upgrades",
        description="Provides targeted, practical recommendations to boost a model's competition performance.",
        parameters={
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Model architecture or family (e.g., XGBoost, CNN, Transformer)"},
                "current_score": {"type": "number", "description": "Current validation or leaderboard score"},
                "target_score": {"type": "number", "description": "Desired score to reach"}
            },
            "required": ["model_name", "current_score", "target_score"]
        }
    ),
    FunctionDeclaration(
        name="build_competition_plan",
        description="Generates a step-by-step, day-by-day plan to improve competition standing.",
        parameters={
            "type": "object",
            "properties": {
                "objective": {"type": "string", "description": "High-level competition goal (e.g., reach top 10%)"},
                "days_available": {"type": "integer", "description": "Number of days available for the sprint"},
                "current_status": {"type": "string", "description": "Brief summary of current progress or ranking"}
            },
            "required": ["objective", "days_available", "current_status"]
        }
    ),
    FunctionDeclaration(
        name="troubleshoot_code",
        description="Analyzes an error and returns likely root cause, a minimal patch, and tests to validate the fix.",
        parameters={
            "type": "object",
            "properties": {
                "error_message": {"type": "string", "description": "Full error text or stack trace"},
                "code_snippet": {"type": "string", "description": "Relevant code block where the error appears"},
                "framework": {"type": "string", "description": "Optional: runtime or library (e.g., PyTorch, pandas)"}
            },
            "required": ["error_message", "code_snippet"]
        }
    ),
    FunctionDeclaration(
        name="recommend_features",
        description="Suggests advanced feature engineering ideas with brief code examples and impact estimates.",
        parameters={
            "type": "object",
            "properties": {
                "data_overview": {"type": "string", "description": "Short description of the dataset/domain"},
                "target_column": {"type": "string", "description": "Target variable to predict"},
                "existing_columns": {"type": "string", "description": "List or description of current feature columns"}
            },
            "required": ["data_overview", "target_column", "existing_columns"]
        }
    ),
    FunctionDeclaration(
        name="summarize_competition_insights",
        description="Aggregates community and published solutions to surface common winning patterns and novel tactics.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic or competition theme to analyze"},
                "source_limit": {"type": "integer", "description": "Number of sources to aggregate (approximate, default: 5)"}
            },
            "required": ["topic"]
        }
    )
]

tools = Tool(function_declarations=function_declarations)
print(f"âœ“ Function Declarations Created ({len(function_declarations)} tools)")



@dataclass
class ConversationMemory:
    """
    Lightweight manager for tracking recent conversation history.
    Stores message turns, timestamps, and provides compact context
    useful for agent reasoning.
    """
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_history: int = 30

    def add_message(self, role: str, content: str):
        """Append a message to memory while enforcing the history limit."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only the most recent N messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_context(self) -> str:
        """
        Return a compact summary of recent conversation turns.
        Useful for feeding into LLM prompts or agent routing logic.
        """
        if not self.messages:
            return "No prior conversation available."

        context = "Recent conversation summary:\n"
        for msg in self.messages[-5:]:
            preview = msg["content"][:120]  # longer but meaningful
            context += f"{msg['role']}: {preview}...\n"
        return context

    def clear(self):
        """Reset memory completely."""
        self.messages.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return basic statistics about stored messages."""
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m["role"] == "user"),
            "agent_messages": sum(1 for m in self.messages if m["role"] == "agent")
        }


memory = ConversationMemory(max_history=30)
print(f"âœ“ Conversation Memory Ready (limit: {memory.max_history} messages)")



@dataclass
class AgentLogger:
    """
    Lightweight, structured logger for agent activity.
    Captures timestamped events and supports simple log-level filtering.
    Ideal for debugging multi-agent workflows.
    """
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, level: str, event: str, details: Dict[str, Any] = None):
        """Record a log event with optional metadata."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "event": event,
            "details": details or {}
        })

    def info(self, event: str, **kwargs):
        """Log an informational message."""
        self.log("INFO", event, kwargs)

    def error(self, event: str, **kwargs):
        """Log an error-level message."""
        self.log("ERROR", event, kwargs)

    def warning(self, event: str, **kwargs):
        """Log a warning-level message."""
        self.log("WARNING", event, kwargs)

    def get_recent_logs(self, count: int = 10) -> List[Dict]:
        """Return the most recent N log entries."""
        return self.logs[-count:]

    def get_stats(self) -> Dict[str, Any]:
        """Summarize total log volume and counts per log level."""
        return {
            "total_logs": len(self.logs),
            "info_count": sum(1 for log in self.logs if log["level"] == "INFO"),
            "error_count": sum(1 for log in self.logs if log["level"] == "ERROR"),
            "warning_count": sum(1 for log in self.logs if log["level"] == "WARNING"),
        }

    def export_logs(self, filename: str = "agent_logs.json"):
        """Export logs to a JSON file for offline review."""
        with open(filename, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"âœ“ Logs exported â†’ {filename}")


# Initialize the logger
logger = AgentLogger()
logger.info("Logger initialized")
print("âœ“ Logging System Ready")



class KaggleCompetitionAgent:
    """Main orchestrating agent for competition assistance"""
    
    def __init__(self, config: Dict, tools: Tool, memory: ConversationMemory, logger: AgentLogger):
        self.config = config
        self.tools = tools
        self.memory = memory
        self.logger = logger
        self.model = genai.GenerativeModel(model_name=config['model'], tools=[tools])
        
        self.stats = {
            "queries_processed": 0,
            "tools_called": 0,
            "total_response_time": 0.0,
            "errors": 0
        }
        self.logger.info("Agent initialized", model=config['model'])
    
    def _call_function(self, function_call) -> str:
        """Execute tool function and return result"""
        function_name = function_call.name
        function_args = dict(function_call.args)
        
        self.logger.info("Function called", function=function_name, args=str(function_args))
        
        function_map = {
            "suggest_model_improvements": suggest_model_improvements,
            "create_competition_strategy": create_competition_strategy,
            "debug_code_issue": debug_code_issue,
            "suggest_features": suggest_features,
            "analyze_competition_insights": analyze_competition_insights
        }
        
        if function_name in function_map:
            try:
                result = function_map[function_name](**function_args)
                self.stats["tools_called"] += 1
                return result
            except Exception as e:
                self.logger.error("Function execution failed", error=str(e))
                return f"Error executing {function_name}: {str(e)}"
        return f"Unknown function: {function_name}"
    
    def run(self, user_query: str) -> str:
        start_time = time.time()
        
        try:
            self.logger.info("Query received", query=user_query[:100])
            self.memory.add_message("user", user_query)
            
            system_prompt = f"""You are an expert Kaggle Competition Assistant for Team {self.config['team']}.

Capabilities: Model improvements, strategy planning, debugging, feature engineering, insights analysis

Context: {self.memory.get_context()}

Provide specific, actionable guidance."""

            # Start chat WITHOUT automatic function calling
            chat = self.model.start_chat()
            response = chat.send_message(f"{system_prompt}\n\nUser Query: {user_query}")
            
            # Check if function was called
            function_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)
            
            # Execute functions and get results
            if function_calls:
                function_responses = []
                for fc in function_calls:
                    result = self._call_function(fc)
                    function_responses.append(result)
                
                # Send function results back to model
                response = chat.send_message(function_responses)
            
            # Extract final text response
            try:
                response_text = response.text
            except Exception:
                if hasattr(response, 'candidates') and response.candidates:
                    parts = response.candidates[0].content.parts
                    response_text = ""
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                    if not response_text:
                        response_text = "Response generated successfully."
                else:
                    response_text = "Unable to extract response."
            
            self.memory.add_message("agent", response_text)
            
            elapsed = time.time() - start_time
            self.stats["queries_processed"] += 1
            self.stats["total_response_time"] += elapsed
            
            self.logger.info("Query completed", response_time=f"{elapsed:.2f}s")
            return response_text
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Query failed", error=str(e))
            return f"Error: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        avg_response_time = (
            self.stats["total_response_time"] / self.stats["queries_processed"]
            if self.stats["queries_processed"] > 0 else 0
        )
        
        return {
            **self.stats,
            "avg_response_time": round(avg_response_time, 2),
            "memory_stats": self.memory.get_stats(),
            "logger_stats": self.logger.get_stats()
        }
    
    def reset(self):
        self.memory.clear()
        self.stats = {"queries_processed": 0, "tools_called": 0, "total_response_time": 0.0, "errors": 0}
        self.logger.info("Agent reset")

if GOOGLE_API_KEY:
    agent = KaggleCompetitionAgent(config=CONFIG, tools=tools, memory=memory, logger=logger)
    print("âœ“ Agent Initialized")
    print("âœ“ Ready for Competition Assistance")
else:
    agent = None
    print("âš  Agent initialization skipped - Configure API key")


def test_agent(query: str) -> Optional[str]:
    """
    Simple utility to exercise the KaggleCompetitionAgent with a single query.

    - Logs the query via the agent logger (if available)
    - Runs the agent and prints a neatly formatted interaction
    - Returns the agent's raw text response (or None if agent not available)
    """
    if agent is None:
        print("âš  Agent not initialized. Set GOOGLE_API_KEY and initialize the agent first.")
        return None

    # Log and store the incoming query
    try:
        agent.logger.info("Running test query", snippet=query[:200])
    except Exception:
        # Defensive: if logger missing or fails, continue
        pass

    divider = "=" * 70
    print(f"\n{divider}")
    print("TEST INPUT (user):")
    print(f"{query}\n")
    print(divider + "\n")

    # Run the agent and measure elapsed time
    start = time.time()
    try:
        response = agent.run(query)
    except Exception as exc:
        agent.logger.error("Test run failed", error=str(exc))
        print("â�Œ Agent execution raised an exception:")
        print(str(exc))
        return None
    elapsed = time.time() - start

    # Print the agent's response cleanly
    print("AGENT RESPONSE:")
    print("-" * 70)
    print(response)
    print("\n" + divider)
    print(f"Execution time: {elapsed:.2f}s")
    print(divider + "\n")

    # Optionally log completion
    try:
        agent.logger.info("Test query completed", response_time=f"{elapsed:.2f}s")
    except Exception:
        pass

    return response


print("âœ“ test_agent utility loaded")
print("Usage example: test_agent('How can I improve my LightGBM model to reach a 0.92 AUC?')")



test_agent("What are the top 3 strategies for winning Kaggle competitions?")


test_agent("What are the top 3 strategies to be a cool guy?")



def display_statistics():
    """Display agent performance metrics"""
    if not agent:
        print("âš  Agent not initialized")
        return
    
    stats = agent.get_stats()
    
    print(f"\n{'='*60}")
    print(f"{'AGENT PERFORMANCE DASHBOARD':^60}")
    print(f"{'='*60}")
    
    print(f"\nğŸ“Š Query Statistics:")
    print(f"  Total Queries: {stats['queries_processed']}")
    print(f"  Tools Called: {stats['tools_called']}")
    print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
    print(f"  Errors: {stats['errors']}")
    
    print(f"\nğŸ’­ Memory Statistics:")
    mem = stats['memory_stats']
    print(f"  Total Messages: {mem['total_messages']}")
    print(f"  User Messages: {mem['user_messages']}")
    print(f"  Agent Messages: {mem['agent_messages']}")
    
    print(f"\nğŸ“� Logger Statistics:")
    log = stats['logger_stats']
    print(f"  Total Logs: {log['total_logs']}")
    print(f"  Info: {log['info_count']} | Warning: {log['warning_count']} | Error: {log['error_count']}")
    
    print(f"{'='*60}\n")

if agent:
    display_statistics()


# Example 1: General Strategy Question (This one already worked!)
print("="*60)
print("DEMO 1: Kaggle Competition Strategies")
print("="*60)
test_agent("What are the top 3 strategies for winning Kaggle competitions?")

print("\n" + "="*60)
print("DEMO 2: Model Improvement Suggestion")
print("="*60)
test_agent("How can I improve my XGBoost model from 0.87 to 0.92 accuracy?")

print("\n" + "="*60)
print("DEMO 3: Feature Engineering Ideas")
print("="*60)
test_agent("Suggest 5 features for customer churn prediction")

print("\n" + "="*60)
print("UPDATED PERFORMANCE METRICS")
print("="*60)

# Show updated statistics
if agent:
    stats = agent.get_stats()
    
    print(f"\nğŸ“Š Query Statistics:")
    print(f"  Total Queries: {stats['queries_processed']}")
    print(f"  Tools Called: {stats['tools_called']}")
    print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
    print(f"  Errors: {stats['errors']}")
    
    print(f"\nğŸ’­ Memory Statistics:")
    mem = stats['memory_stats']
    print(f"  Total Messages: {mem['total_messages']}")
    print(f"  User Messages: {mem['user_messages']}")
    print(f"  Agent Messages: {mem['agent_messages']}")
    
    print(f"\nğŸ“� Logger Statistics:")
    log = stats['logger_stats']
    print(f"  Total Logs: {log['total_logs']}")
    print(f"  Info: {log['info_count']} | Warning: {log['warning_count']} | Error: {log['error_count']}")




