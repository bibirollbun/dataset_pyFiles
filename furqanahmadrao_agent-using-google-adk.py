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

print("✓ Libraries Loaded")



# Load API Key from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✓ API Key Configured")
except Exception as e:
    print(f"⚠ API Key Error: {str(e)}")
    print("📌 To fix: Go to Add-ons → Secrets → Add 'GOOGLE_API_KEY'")
    GOOGLE_API_KEY = None

# Agent Configuration
CONFIG = {
    "team": "MindMatrix",
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


def suggest_model_improvements(model_type: str, current_score: float, target_score: float) -> str:
    """Generate ML model improvement suggestions"""
    prompt = (
        f"As a Kaggle Grandmaster, provide 5 specific improvements for:\n\n"
        f"Model: {model_type}\n"
        f"Current Score: {current_score}\n"
        f"Target Score: {target_score}\n\n"
        f"Include: Architecture changes, training techniques, data augmentation, "
        f"ensemble methods, hyperparameter tuning."
    )
    model = genai.GenerativeModel(CONFIG['model'])
    return model.generate_content(prompt).text


def create_competition_strategy(goal: str, timeframe_days: int, current_position: str) -> str:
    """Create detailed competition strategy plan"""
    prompt = (
        f"Create a day-by-day competition strategy:\n\n"
        f"Goal: {goal}\n"
        f"Timeframe: {timeframe_days} days\n"
        f"Current Position: {current_position}\n\n"
        f"Provide: Daily tasks, time estimates, success metrics, risk mitigation, "
        f"resource requirements."
    )
    model = genai.GenerativeModel(CONFIG['model'])
    return model.generate_content(prompt).text


def debug_code_issue(error_message: str, code_context: str, framework: str = "general") -> str:
    """Debug code with detailed solutions"""
    prompt = (
        f"Debug this issue:\n\n"
        f"Framework: {framework}\n"
        f"Error: {error_message}\n\n"
        f"Code:\n``````\n\n"
        f"Provide: Root cause, corrected code, best practices, testing approach."
    )
    model = genai.GenerativeModel(CONFIG['model'])
    return model.generate_content(prompt).text


def suggest_features(dataset_description: str, target_variable: str, current_features: str) -> str:
    """Suggest advanced feature engineering techniques"""
    prompt = (
        f"Suggest feature engineering for:\n\n"
        f"Dataset: {dataset_description}\n"
        f"Target: {target_variable}\n"
        f"Current Features: {current_features}\n\n"
        f"Provide: 10 new features, interactions, domain transformations, "
        f"code snippets, impact ranking."
    )
    model = genai.GenerativeModel(CONFIG['model'])
    return model.generate_content(prompt).text


def analyze_competition_insights(topic: str, num_sources: int = 5) -> str:
    """Analyze competition discussions and techniques"""
    prompt = (
        f"Analyze Kaggle insights for: {topic}\n\n"
        f"Synthesize: Winning techniques, common approaches, novel methods, "
        f"code patterns, recommendations."
    )
    model = genai.GenerativeModel(CONFIG['model'])
    return model.generate_content(prompt).text


print("✓ 5 Tool Functions Defined")
print("  • suggest_model_improvements")
print("  • create_competition_strategy")
print("  • debug_code_issue")
print("  • suggest_features")
print("  • analyze_competition_insights")


function_declarations = [
    FunctionDeclaration(
        name="suggest_model_improvements",
        description="Suggests ML model improvements to increase competition score",
        parameters={
            "type": "object",
            "properties": {
                "model_type": {"type": "string", "description": "Model architecture (e.g., XGBoost, Neural Network)"},
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

tools = Tool(function_declarations=function_declarations)
print(f"✓ Function Declarations Created ({len(function_declarations)} tools)")


@dataclass
class ConversationMemory:
    """Manages conversation history and context"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_history: int = 20
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_context(self) -> str:
        if not self.messages:
            return "No previous conversation."
        context = "Recent conversation:\n"
        for msg in self.messages[-5:]:
            context += f"{msg['role']}: {msg['content'][:100]}...\n"
        return context
    
    def clear(self):
        self.messages.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m['role'] == 'user'),
            "agent_messages": sum(1 for m in self.messages if m['role'] == 'agent')
        }

memory = ConversationMemory(max_history=20)
print(f"✓ Memory System Initialized (Max: {memory.max_history} messages)")


@dataclass
class AgentLogger:
    """Comprehensive logging for agent operations"""
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def log(self, level: str, event: str, details: Dict[str, Any] = None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            "details": details or {}
        })
    
    def info(self, event: str, **kwargs):
        self.log("INFO", event, kwargs)
    
    def error(self, event: str, **kwargs):
        self.log("ERROR", event, kwargs)
    
    def warning(self, event: str, **kwargs):
        self.log("WARNING", event, kwargs)
    
    def get_recent_logs(self, count: int = 10) -> List[Dict]:
        return self.logs[-count:]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_logs": len(self.logs),
            "info_count": sum(1 for log in self.logs if log['level'] == 'INFO'),
            "error_count": sum(1 for log in self.logs if log['level'] == 'ERROR'),
            "warning_count": sum(1 for log in self.logs if log['level'] == 'WARNING')
        }
    
    def export_logs(self, filename: str = "agent_logs.json"):
        with open(filename, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"✓ Logs exported to {filename}")

logger = AgentLogger()
logger.info("Logger initialized")
print("✓ Logging System Ready")


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
    print("✓ Agent Initialized")
    print("✓ Ready for Competition Assistance")
else:
    agent = None
    print("⚠ Agent initialization skipped - Configure API key")


def test_agent(query: str):
    """Test agent with a query"""
    if not agent:
        print("⚠ Agent not initialized")
        return
    
    print(f"\n{'='*60}")
    print(f"USER: {query}")
    print(f"{'='*60}\n")
    
    response = agent.run(query)
    
    print("AGENT RESPONSE:")
    print(f"{'-'*60}")
    print(response)
    print(f"{'='*60}\n")

print("✓ Test function ready")
print("📌 Usage: test_agent('your question here')")


test_agent("What are the top 3 strategies for winning Kaggle competitions?")


def display_statistics():
    """Display agent performance metrics"""
    if not agent:
        print("⚠ Agent not initialized")
        return
    
    stats = agent.get_stats()
    
    print(f"\n{'='*60}")
    print(f"{'AGENT PERFORMANCE DASHBOARD':^60}")
    print(f"{'='*60}")
    
    print(f"\n📊 Query Statistics:")
    print(f"  Total Queries: {stats['queries_processed']}")
    print(f"  Tools Called: {stats['tools_called']}")
    print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
    print(f"  Errors: {stats['errors']}")
    
    print(f"\n💭 Memory Statistics:")
    mem = stats['memory_stats']
    print(f"  Total Messages: {mem['total_messages']}")
    print(f"  User Messages: {mem['user_messages']}")
    print(f"  Agent Messages: {mem['agent_messages']}")
    
    print(f"\n📝 Logger Statistics:")
    log = stats['logger_stats']
    print(f"  Total Logs: {log['total_logs']}")
    print(f"  Info: {log['info_count']} | Warning: {log['warning_count']} | Error: {log['error_count']}")
    
    print(f"{'='*60}\n")

if agent:
    display_statistics()


# Export Conversation History & Logs

def export_conversation_history(filename="conversation_history.txt"):
    """Export the full conversation history to a text file"""
    if not agent:
        print("⚠ Agent not initialized")
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
        
        print(f"✓ Conversation history exported to: {filename}")
        print(f"📊 Total messages: {memory_stats['total_messages']}")
        return filename
    
    except Exception as e:
        print(f"❌ Error exporting conversation: {str(e)}")
        return None

def export_agent_logs(filename="agent_logs.json"):
    """Export detailed agent logs to JSON"""
    import json
    
    if not agent:
        print("⚠ Agent not initialized")
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
        
        print(f"✓ Agent logs exported to: {filename}")
        print(f"📝 Total log entries: {stats['logger_stats']['total_logs']}")
        return filename
    
    except Exception as e:
        print(f"❌ Error exporting logs: {str(e)}")
        return None

print("✓ Export Functions Ready!")
print("\n📤 Available Commands:")
print("  • export_conversation_history('filename.txt')")
print("  • export_agent_logs('filename.json')")


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
    
    print(f"\n📊 Query Statistics:")
    print(f"  Total Queries: {stats['queries_processed']}")
    print(f"  Tools Called: {stats['tools_called']}")
    print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
    print(f"  Errors: {stats['errors']}")
    
    print(f"\n💭 Memory Statistics:")
    mem = stats['memory_stats']
    print(f"  Total Messages: {mem['total_messages']}")
    print(f"  User Messages: {mem['user_messages']}")
    print(f"  Agent Messages: {mem['agent_messages']}")
    
    print(f"\n📝 Logger Statistics:")
    log = stats['logger_stats']
    print(f"  Total Logs: {log['total_logs']}")
    print(f"  Info: {log['info_count']} | Warning: {log['warning_count']} | Error: {log['error_count']}")


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
    print("✓ EXPORT SUCCESSFUL!")
    print("=" * 60)
    print("\n📁 Files created:")
    print(f"  1. {conv_file} - Human-readable conversation history")
    print(f"  2. {log_file} - Detailed JSON logs for analysis")
    print("\n💡 Tip: Click the folder icon on the right to download these files")
else:
    print("\n⚠ Export failed - check error messages above")

