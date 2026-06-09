import sys
import os
import time
import json
import textwrap
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from kaggle_secrets import UserSecretsClient
from IPython.display import display, HTML, clear_output

BASE_DIR = Path.cwd()
EXPORT_DIR = BASE_DIR / "mindmatrix_exports"
EXPORT_DIR.mkdir(exist_ok=True)

print("âœ“ Libraries Loaded")


# Load API Key from Kaggle Secrets
def load_google_api_key(secret_name: str = "GOOGLE_API_KEY") -> Optional[str]:
    """Safely load and configure the Google API key."""
    try:
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret(secret_name)
        genai.configure(api_key=api_key)
        print("âœ“ API Key Configured")
        return api_key
    except Exception as exc:
        print(f"âš  API Key Error: {exc}")
        print("ğŸ“Œ To fix: Add 'GOOGLE_API_KEY' via Add-ons â†’ Secrets")
        return None


@dataclass(frozen=True)
class AgentConfig:
    team: str = "MindMatrix"
    model: str = "models/gemini-2.5-flash"
    max_tokens: int = 2000
    temperature: float = 0.3
    version: str = "2.1.0"
    max_history: int = 20
    response_timeout: int = 90

    def describe(self) -> str:
        lines = ["=" * 60, f"AGENT CONFIGURATION (v{self.version})".center(60), "=" * 60]
        for label, value in self.__dict__.items():
            lines.append(f"{label.replace('_', ' ').title():.<25} {value}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class AgentStats:
    queries_processed: int = 0
    tools_called: int = 0
    total_response_time: float = 0.0
    errors: int = 0

    @property
    def avg_response_time(self) -> float:
        if not self.queries_processed:
            return 0.0
        return round(self.total_response_time / self.queries_processed, 2)

    def record_success(self, elapsed: float, tools_used: int):
        self.queries_processed += 1
        self.tools_called += tools_used
        self.total_response_time += elapsed

    def record_error(self):
        self.errors += 1


GOOGLE_API_KEY = load_google_api_key()
CONFIG = AgentConfig()
print(CONFIG.describe())


class MindMatrixToolkit:
    """Encapsulates all function-calling utilities exposed to the agent."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._function_declarations = self._build_function_declarations()
        self._function_map: Dict[str, Callable[..., str]] = {
            "suggest_model_improvements": self.suggest_model_improvements,
            "create_competition_strategy": self.create_competition_strategy,
            "debug_code_issue": self.debug_code_issue,
            "suggest_features": self.suggest_features,
            "analyze_competition_insights": self.analyze_competition_insights
        }

    @staticmethod
    def _safe_text(response: Any) -> str:
        if hasattr(response, "text") and response.text:
            return response.text
        candidates = getattr(response, "candidates", [])
        text_chunks: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                text_value = getattr(part, "text", None)
                if text_value:
                    text_chunks.append(text_value)
        return "\n".join(text_chunks) if text_chunks else "Response generated successfully."

    def _call_model(self, prompt: str) -> str:
        model = genai.GenerativeModel(self.config.model)
        response = model.generate_content(prompt)
        return self._safe_text(response)

    def suggest_model_improvements(self, model_type: str, current_score: float, target_score: float) -> str:
        """Generate ML model improvement suggestions"""
        prompt = textwrap.dedent(f"""
            As a Kaggle Grandmaster, provide 5 specific improvements for:
            Model: {model_type}
            Current Score: {current_score}
            Target Score: {target_score}

            Include architecture changes, training techniques, data augmentation, ensemble methods, and hyperparameter tuning.
        """).strip()
        return self._call_model(prompt)

    def create_competition_strategy(self, goal: str, timeframe_days: int, current_position: str) -> str:
        """Create detailed competition strategy plan"""
        prompt = textwrap.dedent(f"""
            Create a day-by-day competition strategy.
            Goal: {goal}
            Timeframe: {timeframe_days} days
            Current Position: {current_position}

            Include daily tasks, time estimates, success metrics, risk mitigation, and resource requirements.
        """).strip()
        return self._call_model(prompt)

    def debug_code_issue(self, error_message: str, code_context: str, framework: str = "general") -> str:
        """Debug code with detailed solutions"""
        prompt = textwrap.dedent(f"""
            Debug this issue.
            Framework: {framework}
            Error: {error_message}

            Code:
            ``````
            {code_context}
            ``````

            Provide the root cause, corrected code, best practices, and a testing approach.
        """).strip()
        return self._call_model(prompt)

    def suggest_features(self, dataset_description: str, target_variable: str, current_features: str) -> str:
        """Suggest advanced feature engineering techniques"""
        prompt = textwrap.dedent(f"""
            Suggest feature engineering ideas for:
            Dataset: {dataset_description}
            Target: {target_variable}
            Current Features: {current_features}

            Provide 10 new features, interactions, domain transformations, code snippets, and impact ranking.
        """).strip()
        return self._call_model(prompt)

    def analyze_competition_insights(self, topic: str, num_sources: int = 5) -> str:
        """Analyze competition discussions and techniques"""
        prompt = textwrap.dedent(f"""
            Analyze Kaggle insights for: {topic}.
            Reference {num_sources} sources and summarize winning techniques, novel methods, code patterns, and recommendations.
        """).strip()
        return self._call_model(prompt)

    def call(self, function_name: str, **kwargs) -> str:
        if function_name not in self._function_map:
            raise ValueError(f"Unknown function: {function_name}")
        return self._function_map[function_name](**kwargs)

    def _build_function_declarations(self) -> List[FunctionDeclaration]:
        return [
            FunctionDeclaration(
                name="suggest_model_improvements",
                description="Suggests ML model improvements to increase competition score",
                parameters={
                    "type": "object",
                    "properties": {
                        "model_type": {"type": "string", "description": "Model architecture (e.g., XGBoost)"},
                        "current_score": {"type": "number", "description": "Current competition score"},
                        "target_score": {"type": "number", "description": "Target score to achieve"}
                    },
                    "required": ["model_type", "current_score", "target_score"]
                }
            ),
            FunctionDeclaration(
                name="create_competition_strategy",
                description="Creates day-by-day strategy plan for competitions",
                parameters={
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Competition goal"},
                        "timeframe_days": {"type": "integer", "description": "Days available"},
                        "current_position": {"type": "string", "description": "Current standing"}
                    },
                    "required": ["goal", "timeframe_days", "current_position"]
                }
            ),
            FunctionDeclaration(
                name="debug_code_issue",
                description="Debugs code issues with solutions",
                parameters={
                    "type": "object",
                    "properties": {
                        "error_message": {"type": "string", "description": "Error description"},
                        "code_context": {"type": "string", "description": "Code snippet"},
                        "framework": {"type": "string", "description": "Framework (e.g., pandas, sklearn)"}
                    },
                    "required": ["error_message", "code_context"]
                }
            ),
            FunctionDeclaration(
                name="suggest_features",
                description="Suggests feature engineering techniques",
                parameters={
                    "type": "object",
                    "properties": {
                        "dataset_description": {"type": "string", "description": "Dataset description"},
                        "target_variable": {"type": "string", "description": "Target variable"},
                        "current_features": {"type": "string", "description": "Current features"}
                    },
                    "required": ["dataset_description", "target_variable", "current_features"]
                }
            ),
            FunctionDeclaration(
                name="analyze_competition_insights",
                description="Analyzes competition discussions and techniques",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic to analyze"},
                        "num_sources": {"type": "integer", "description": "Number of sources (default: 5)"}
                    },
                    "required": ["topic"]
                }
            )
        ]

    @property
    def function_declarations(self) -> List[FunctionDeclaration]:
        return self._function_declarations

    @property
    def tool(self) -> Tool:
        return Tool(function_declarations=self._function_declarations)

print("âœ“ MindMatrix toolkit ready")


toolkit = MindMatrixToolkit(CONFIG)
function_declarations = toolkit.function_declarations
tools = toolkit.tool
print(f"âœ“ Function Declarations Created ({len(function_declarations)} tools)")


@dataclass
class ConversationMemory:
    """Manages conversation history and context."""
    max_history: int = CONFIG.max_history
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(entry)
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_context(self, window: int = 5) -> str:
        if not self.messages:
            return "No previous conversation."
        recent = self.messages[-window:]
        context_lines = ["Recent conversation:"]
        for msg in recent:
            snippet = msg['content'].strip().replace("\n", " ")[:160]
            context_lines.append(f"{msg['role']}: {snippet}...")
        return "\n".join(context_lines)

    def clear(self):
        self.messages.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m['role'] == 'user'),
            "agent_messages": sum(1 for m in self.messages if m['role'] == 'agent')
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "messages": list(self.messages),
            "stats": self.get_stats()
        }

memory = ConversationMemory(max_history=CONFIG.max_history)
print(f"Memory System Initialized (Max: {memory.max_history} messages)")


@dataclass
class LogEntry:
    timestamp: str
    level: str
    event: str
    details: Dict[str, Any]


class AgentLogger:
    """Structured logger with export helpers."""

    def __init__(self):
        self.logs: List[LogEntry] = []

    def log(self, level: str, event: str, details: Optional[Dict[str, Any]] = None):
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            event=event,
            details=details or {}
        )
        self.logs.append(entry)

    def info(self, event: str, **kwargs):
        self.log("INFO", event, kwargs)

    def error(self, event: str, **kwargs):
        self.log("ERROR", event, kwargs)

    def warning(self, event: str, **kwargs):
        self.log("WARNING", event, kwargs)

    def get_recent_logs(self, count: int = 10) -> List[Dict[str, Any]]:
        return [log.__dict__ for log in self.logs[-count:]]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_logs": len(self.logs),
            "info_count": sum(1 for log in self.logs if log.level == 'INFO'),
            "error_count": sum(1 for log in self.logs if log.level == 'ERROR'),
            "warning_count": sum(1 for log in self.logs if log.level == 'WARNING')
        }

    def export(self, filename: str = "agent_logs.json") -> Path:
        path = EXPORT_DIR / filename
        with open(path, 'w') as f:
            json.dump([log.__dict__ for log in self.logs], f, indent=2)
        print(f"Logs exported to {path}")
        return path

logger = AgentLogger()
logger.info("Logger initialized")
print("Logging System Ready")


class PromptBuilder:
    """Creates rich prompts using config + memory context."""

    def __init__(self, config: AgentConfig, memory: ConversationMemory):
        self.config = config
        self.memory = memory

    def build(self, user_query: str) -> str:
        context = self.memory.get_context()
        prompt = textwrap.dedent(f"""
            You are an expert Kaggle Competition Assistant for Team {self.config.team}.
            Capabilities: Model improvements, strategy planning, debugging, feature engineering, insights analysis.
            Context: {context}

            Provide specific, actionable guidance.

            User Query: {user_query}
        """).strip()
        return prompt


@contextmanager
def measure_time():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


class KaggleCompetitionAgent:
    """Main orchestrating agent for competition assistance."""

    def __init__(self, config: AgentConfig, toolkit: MindMatrixToolkit, memory: ConversationMemory, logger: AgentLogger, prompt_builder: Optional[PromptBuilder] = None):
        self.config = config
        self.toolkit = toolkit
        self.memory = memory
        self.logger = logger
        self.prompt_builder = prompt_builder or PromptBuilder(config, memory)
        self.stats = AgentStats()
        self._model = genai.GenerativeModel(
            model_name=config.model,
            tools=[toolkit.tool],
            generation_config={
                "max_output_tokens": config.max_tokens,
                "temperature": config.temperature
            }
        )
        self.logger.info("Agent initialized", model=config.model)

    @staticmethod
    def _extract_function_calls(response) -> List[Any]:
        calls = []
        candidates = getattr(response, "candidates", [])
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                function_call = getattr(part, "function_call", None)
                if function_call:
                    calls.append(function_call)
        return calls

    def _execute_function_calls(self, function_calls: List[Any]) -> (List[str], int):
        results: List[str] = []
        for fc in function_calls:
            function_name = getattr(fc, "name", "")
            args = dict(getattr(fc, "args", {}))
            try:
                self.logger.info("Function called", function=function_name, args=str(args))
                result = self.toolkit.call(function_name, **args)
                results.append(result)
            except Exception as exc:
                self.logger.error("Function execution failed", function=function_name, error=str(exc))
                results.append(f"Error executing {function_name}: {exc}")
        return results, len(results)

    @staticmethod
    def _extract_response_text(response) -> str:
        if hasattr(response, "text") and response.text:
            return response.text
        candidates = getattr(response, "candidates", [])
        text_chunks: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                text_value = getattr(part, "text", None)
                if text_value:
                    text_chunks.append(text_value)
        return "\n".join(text_chunks) if text_chunks else "Response generated successfully."

    def run(self, user_query: str) -> str:
        if not GOOGLE_API_KEY:
            return "âš  Agent not initialized - configure API key first"
        if not user_query or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")

        self.memory.add_message("user", user_query)
        prompt = self.prompt_builder.build(user_query)
        chat = self._model.start_chat()
        tools_used = 0

        with measure_time() as elapsed_timer:
            try:
                response = chat.send_message(prompt)
                function_calls = self._extract_function_calls(response)
                if function_calls:
                    payload, tools_used = self._execute_function_calls(function_calls)
                    response = chat.send_message(payload)
                response_text = self._extract_response_text(response)
            except Exception as exc:
                self.stats.record_error()
                self.logger.error("Query failed", error=str(exc))
                return f"Error: {exc}"

        elapsed = elapsed_timer()
        self.memory.add_message("agent", response_text)
        self.stats.record_success(elapsed, tools_used)
        self.logger.info("Query completed", response_time=f"{elapsed:.2f}s", tools_used=tools_used)
        return response_text

    def get_stats(self) -> Dict[str, Any]:
        return {
            "queries_processed": self.stats.queries_processed,
            "tools_called": self.stats.tools_called,
            "avg_response_time": self.stats.avg_response_time,
            "errors": self.stats.errors,
            "memory_stats": self.memory.get_stats(),
            "logger_stats": self.logger.get_stats()
        }

    def reset(self):
        self.memory.clear()
        self.stats = AgentStats()
        self.logger.info("Agent reset")

if GOOGLE_API_KEY:
    agent = KaggleCompetitionAgent(config=CONFIG, toolkit=toolkit, memory=memory, logger=logger)
    print("âœ“ Agent Initialized")
    print("âœ“ Ready for Competition Assistance")
else:
    agent = None
    print("âš  Agent initialization skipped - Configure API key")


class AgentConsole:
    """Lightweight helper for running interactive tests with the agent."""
    def __init__(self, agent_instance: Optional[KaggleCompetitionAgent]):
        self.agent = agent_instance

    def ask(self, query: str) -> Optional[str]:
        if not self.agent:
            print("Agent not initialized")
            return None
        print(f"\n{'='*60}")
        print(f"USER: {query}")
        print(f"{'='*60}\n")
        response = self.agent.run(query)
        print("AGENT RESPONSE:")
        print(f"{'-'*60}")
        print(response)
        print(f"{'='*60}\n")
        return response

    def display_stats(self):
        if not self.agent:
            print("Agent not initialized")
            return
        stats = self.agent.get_stats()
        print(f"\n{'='*60}")
        print(f"{'AGENT PERFORMANCE DASHBOARD':^60}")
        print(f"{'='*60}")
        print(f"\nQuery Statistics:")
        print(f"  Total Queries: {stats['queries_processed']}")
        print(f"  Tools Called: {stats['tools_called']}")
        print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
        print(f"  Errors: {stats['errors']}")
        print(f"\nMemory Statistics:")
        mem = stats['memory_stats']
        print(f"  Total Messages: {mem['total_messages']}")
        print(f"  User Messages: {mem['user_messages']}")
        print(f"  Agent Messages: {mem['agent_messages']}")
        print(f"\nLogger Statistics:")
        log = stats['logger_stats']
        print(f"  Total Logs: {log['total_logs']}")
        print(f"  Info: {log['info_count']} | Warning: {log['warning_count']} | Error: {log['error_count']}")
        print(f"{'='*60}\n")

    def run_scripted_demo(self, prompts: List[str]):
        if not self.agent:
            print("Agent not initialized")
            return
        for idx, prompt in enumerate(prompts, start=1):
            print(f"\n{'='*60}")
            print(f"DEMO {idx}: {prompt}")
            print(f"{'='*60}")
            self.ask(prompt)

agent_console = AgentConsole(agent)
print("Agent console ready. Use agent_console.ask('question') to interact.")


if agent:
    agent_console.ask("What are the top 3 strategies for winning Kaggle competitions?")
else:
    print("âš  Agent not initialized")


def display_statistics():
    """Display agent performance metrics via the console helper."""
    agent_console.display_stats()

if agent:
    display_statistics()
else:
    print("âš  Agent not initialized")


# Export Conversation History & Logs

def export_conversation_history(filename="conversation_history.txt"):
    """Export the full conversation history to a text file"""
    if not agent:
        print("âš  Agent not initialized")
        return None
    
    try:
        stats = agent.get_stats()
        memory_stats = stats['memory_stats']
        
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MINDMATRIX AI AGENT - CONVERSATION HISTORY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Session Statistics:\n")
            f.write(f"  Total Queries: {stats['queries_processed']}\n")
            f.write(f"  Tools Called: {stats['tools_called']}\n")
            f.write(f"  Average Response Time: {stats['avg_response_time']:.2f}s\n")
            f.write(f"  Errors: {stats['errors']}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("CONVERSATION LOG\n")
            f.write("=" * 60 + "\n\n")
            
            for msg in agent.memory.messages:
                role = msg['role'].upper()
                timestamp = msg.get('timestamp', 'N/A')
                content = msg['content']
                
                f.write(f"[{timestamp}] {role}:\n")
                f.write(f"{content}\n")
                f.write("-" * 60 + "\n\n")
            
            f.write("=" * 60 + "\n")
            f.write(f"Total Messages: {memory_stats['total_messages']}\n")
            f.write(f"User Messages: {memory_stats['user_messages']}\n")
            f.write(f"Agent Messages: {memory_stats['agent_messages']}\n")
            f.write("=" * 60 + "\n")
        
        print(f"âœ“ Conversation history exported to: {filename}")
        print(f"ğŸ“Š Total messages: {memory_stats['total_messages']}")
        return filename
    
    except Exception as e:
        print(f"â�Œ Error exporting conversation: {str(e)}")
        return None

def export_agent_logs(filename="agent_logs.json"):
    """Export detailed agent logs to JSON"""
    import json
    
    if not agent:
        print("âš  Agent not initialized")
        return None
    
    try:
        stats = agent.get_stats()
        
        export_data = {
            "performance_metrics": {
                "queries_processed": stats['queries_processed'],
                "tools_called": stats['tools_called'],
                "avg_response_time": stats['avg_response_time'],
                "errors": stats['errors']
            },
            "memory_stats": stats['memory_stats'],
            "logger_stats": stats['logger_stats'],
            "logs": agent.logger.logs,
            "conversation": agent.memory.messages
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"âœ“ Agent logs exported to: {filename}")
        print(f"ğŸ“� Total log entries: {stats['logger_stats']['total_logs']}")
        return filename
    
    except Exception as e:
        print(f"â�Œ Error exporting logs: {str(e)}")
        return None

print("âœ“ Export Functions Ready!")
print("\nğŸ“¤ Available Commands:")
print("  â€¢ export_conversation_history('filename.txt')")
print("  â€¢ export_agent_logs('filename.json')")


demo_prompts = [
    "What are the top 3 strategies for winning Kaggle competitions?",
    "How can I improve my XGBoost model from 0.87 to 0.92 accuracy?",
    "Suggest 5 features for customer churn prediction"
 ]

if agent:
    print("="*60)
    print("LIVE DEMO: Running scripted prompts")
    print("="*60)
    agent_console.run_scripted_demo(demo_prompts)
    print("\n" + "="*60)
    print("UPDATED PERFORMANCE METRICS")
    print("="*60)
    agent_console.display_stats()
else:
    print("âš  Agent not initialized")


# Export Demo

print("=" * 60)
print("EXPORT FUNCTIONALITY DEMO")
print("=" * 60)
print()

# Export conversation history
conv_file = export_conversation_history("mindmatrix_conversation.txt")

print()

# Export detailed logs
log_file = export_agent_logs("mindmatrix_logs.json")

if conv_file and log_file:
    print("\n" + "=" * 60)
    print("âœ“ EXPORT SUCCESSFUL!")
    print("=" * 60)
    print("\n Files created:")
    print(f"  1. {conv_file} - Human-readable conversation history")
    print(f"  2. {log_file} - Detailed JSON logs for analysis")
    print("\n Tip: Click the folder icon on the right to download these files")
else:
    print("\nâš  Export failed - check error messages above")

