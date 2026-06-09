pip install google-generativeai rich


import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.theme import Theme
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("âš   Libraries missing. Please run: pip install google-generativeai rich")

print("âœ“ Libraries Loaded")




# API Configuration
GOOGLE_API_KEY = "YOUR_API_KEY"  # <--- REPLACE THIS

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("âœ“ API Key Configured")
    except Exception as e:
        print(f" âš   API Key Error: {str(e)}")


# Agent Configuration
CONFIG = {
    "agent_name": "Bureaucracy Breaker",
    # FIX: Changed 'gemini-2.5-flash' (which may have access issues or be region-locked)
    # to the recommended and widely accessible 'gemini-2.5-flash'.
    "model": "gemini-2.5-flash", 
    "max_tokens": 2000,
    "temperature": 0.4,
    "version": "2.1.1 (FIXED)"
}


# UI Configuration (Rich)
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "danger": "bold red",
    "success": "green",
    "advice": "white"
})
console = Console(theme=custom_theme) if RICH_AVAILABLE else None

print(f"\n{'='*60}")
print(f"{'AGENT CONFIGURATION':^60}")
print(f"{'='*60}")
for k, v in CONFIG.items():
    print(f"{k:.<25} {v}")
print(f"{'='*60}")


def classify_intent(query: str) -> Tuple[str, str]:
    """
    [Triage Agent] Classifies the user query intent and urgency.
    Returns: (Category, Urgency_Level)
    """
    text = query.lower()
    
    # Critical Urgency Detection
    critical_keywords = ["detained", "arrested", "deport", "jail", "police", "stuck at airport", "emergency", "ban", "seized"]
    if any(w in text for w in critical_keywords):
        return "LEGAL_JEOPARDY", "critical"
    
    # Category Detection
    if "passport" in text:
        return "PASSPORT", "medium"
    elif any(w in text for w in ["visa", "immigration", "embassy", "consulate", "permit"]):
        return "VISA", "medium"
    elif any(w in text for w in ["aadhar", "pan", "voter", "license", "id", "birth certificate"]):
        return "CIVIC_DOCS", "low"
    elif any(w in text for w in ["law", "court", "fine", "ticket", "judge", "fir"]):
        return "GENERAL_LAW", "medium"
    
    return "GENERAL_INQUIRY", "low"

def fetch_regulations(category: str, query: str) -> str:
    """
    [Research Agent] Simulates fetching official data or uses GenAI to hallucinate plausible constraints.
    In a production env, this would hit real APIs.
    """
    # Simulated Knowledge Base
    knowledge_base = {
        "passport": "Indian Passport Lost: File FIR at nearest station. Apply for 'Re-issue' on Passport Seva. Tatkaal scheme available.",
        "visa": "Visa Rejections: Check rejection letter code (e.g., 214b for USA). Do not reapply immediately without changing circumstances.",
        "civic_docs": "Aadhar: Update address online via myAadhaar. Biometrics require physical visit. PAN: Apply via NSDL/Protean.",
        "legal_jeopardy": "CRITICAL: Do not sign anything without a lawyer. If abroad, contact Indian Mission/Embassy immediately.",
        "general_inquiry": "General bureaucratic guidelines apply. Check official government portals."
    }
    
    # If we have API access, we could do a grounding search here, 
    # but for now we return the KB entry + the query context
    base_info = knowledge_base.get(category.lower(), knowledge_base["general_inquiry"])
    return f"Official Guidelines for {category}: {base_info}"

def assess_escalation(category: str, urgency: str) -> Dict[str, Any]:
    """
    [Escalation Agent] Checks if human intervention is strictly required.
    
    *FIXED ERROR: Removed the redundant 'ban in category' check, as 'ban' is already handled by
    the 'critical' urgency in the Triage Agent.*
    """
    if urgency == "critical":
        return {
            "required": True,
            "reason": "Immediate Legal Jeopardy / Detention Risk",
            "contact": "Contact: Indian Embassy Emergency Line or a Criminal Lawyer immediately."
        }
    return {"required": False, "reason": "", "contact": ""}

print("âœ“ 3 Tool Functions Defined")
print("  â€¢ classify_intent (Triage)")
print("  â€¢ fetch_regulations (Research)")
print("  â€¢ assess_escalation (Safety)")


@dataclass
class ConversationMemory:
    """Manages conversation history and context"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_history: int = 15

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
            context += f"{msg['role'].upper()}: {msg['content'][:200]}...\n"
        return context

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m['role'] == 'user'),
            "agent_messages": sum(1 for m in self.messages if m['role'] == 'assistant')
        }

memory = ConversationMemory()
print(f"âœ“ Memory System Initialized (Max: {memory.max_history} messages)")


@dataclass
class AgentLogger:
    """Comprehensive logging for agent operations"""
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, level: str, event: str, details: Dict[str, Any] = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            "details": details or {}
        }
        self.logs.append(entry)
        # Optional: Print to console if needed, or keep silent for clean UI
        # print(f"[{level}] {event}") 

    def info(self, event: str, **kwargs):
        self.log("INFO", event, kwargs)

    def error(self, event: str, **kwargs):
        self.log("ERROR", event, kwargs)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_logs": len(self.logs),
            "errors": sum(1 for log in self.logs if log['level'] == 'ERROR')
        }

logger = AgentLogger()
print("âœ“ Logging System Ready")


class BureaucracyCoordinator:
    """Main orchestrating agent for Bureaucracy/Legal assistance"""

    def __init__(self, config: Dict, memory: ConversationMemory, logger: AgentLogger):
        self.config = config
        self.memory = memory
        self.logger = logger
        
        # Initialize Model
        if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
            self.model = genai.GenerativeModel(config['model'])
        else:
            self.model = None

        self.stats = {
            "queries_processed": 0,
            "total_response_time": 0.0,
            "errors": 0
        }
        self.logger.info("Agent initialized", model=config['model'])

    def run(self, user_query: str) -> Dict[str, Any]:
        start_time = time.time()
        self.memory.add_message("user", user_query)
        self.logger.info("Query received", query=user_query[:50])

        try:
            # 1. Execute Triage Tool
            category, urgency = classify_intent(user_query)
            self.logger.info("Triage complete", category=category, urgency=urgency)

            # 2. Execute Research Tool
            research_data = fetch_regulations(category, user_query)
            self.logger.info("Research complete", source=category)

            # 3. Execute Escalation Check
            escalation = assess_escalation(category, urgency)

            # 4. Construct Prompt for Advisor (The LLM)
            system_prompt = f"""
            You are 'Bureaucracy Breaker', an empathetic, expert legal & immigration assistant.
            
            CONTEXT:
            - Intent Category: {category}
            - Urgency: {urgency.upper()}
            - Research: {research_data}
            - History: {self.memory.get_context()}
            - Escalation Needed: {escalation['required']}
            
            INSTRUCTIONS:
            - Break answer into: **The Situation**, **Immediate Actions**, **Documents Needed**, and **Pro Tip**.
            - Use Markdown heavily.
            - If specific forms are needed, insert a tag like <FORM: Aadhar Update> or <DOCUMENT: Police Report>.
            - Be concise.
            """
            
            full_prompt = f"{system_prompt}\n\nUser Query: {user_query}"

            # 5. Generate Response
            if self.model:
                response = self.model.generate_content(full_prompt)
                advice_text = response.text
            else:
                advice_text = f"**Simulation Mode**\nAdvice based on {category}: {research_data}\n\n[NOTE: Full AI features are disabled. Please provide a valid API key.]"

            self.memory.add_message("assistant", advice_text)
            
            elapsed = time.time() - start_time
            self.stats["queries_processed"] += 1
            self.stats["total_response_time"] += elapsed
            
            return {
                "category": category,
                "urgency": urgency,
                "response_text": advice_text,
                "escalation": escalation
            }

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Query failed", error=str(e))
            return {
                "category": "ERROR",
                "urgency": "high",
                "response_text": f"Error processing request: {str(e)}",
                "escalation": {"required": False}
            }

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "memory_stats": self.memory.get_stats(),
            "logger_stats": self.logger.get_stats()
        }

# Initialize the Coordinator
bot = BureaucracyCoordinator(config=CONFIG, memory=memory, logger=logger)
print("âœ“ Agent Initialized")
print("âœ“ Ready for Bureaucracy Assistance")


def display_dashboard(agent: BureaucracyCoordinator):
    stats = agent.get_stats()
    
    if RICH_AVAILABLE:
        grid = f"""
        [bold]Query Stats:[/bold] {stats['queries_processed']} queries | {stats['errors']} errors
        [bold]Memory Stats:[/bold] {stats['memory_stats']['total_messages']} msgs stored
        [bold]Logger Stats:[/bold] {stats['logger_stats']['total_logs']} events logged
        """
        console.print(Panel(grid, title="ğŸ“Š System Dashboard", style="green"))
    else:
        print("\n--- DASHBOARD ---")
        print(stats)
        print("-----------------")

def start_interactive_session():
    """Starts the chat loop"""
    if RICH_AVAILABLE:
        console.clear()
        console.print(Panel.fit(
            "[bold yellow]Bureaucracy Breaker AI[/bold yellow]\n"
            "[italic white]Your Guide through Red Tape, Visas, and Legal Mazes[/italic white]",
            border_style="blue"
        ))

    while True:
        try:
            if RICH_AVAILABLE:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
            else:
                user_input = input("\nYou: ")

            if user_input.lower() in ['exit', 'quit']:
                print("\nâœ“ Session Ended. Goodbye!")
                display_dashboard(bot)
                break

            if RICH_AVAILABLE:
                with console.status("[bold blue]Consulting rulebooks...[/bold blue]", spinner="dots"):
                    result = bot.run(user_input)
            else:
                print("... Processing ...")
                result = bot.run(user_input)

            # Display Output
            header = f"Category: {result['category']} | Urgency: {result['urgency'].upper()}"
            
            if RICH_AVAILABLE:
                console.print(f"\n[dim]{header}[/dim]")
                console.print(Panel(Markdown(result['response_text']), title="ğŸ¤– Bureaucracy Breaker Advice", style="blue"))
                
                if result['escalation']['required']:
                    console.print(Panel(
                        f"[bold]{result['escalation']['reason']}[/bold]\n{result['escalation']['contact']}",
                        title="âš ï¸� ESCALATION REQUIRED", style="danger"
                    ))
            else:
                print(f"\n--- {header} ---")
                print(result['response_text'])
                if result['escalation']['required']:
                    print(f"!!! ESCALATE: {result['escalation']['reason']} !!!")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        print("\nâš   NOTE: No API Key detected. Running in SIMULATION MODE.")
        print("    Edit the 'GOOGLE_API_KEY' variable to enable full AI features.\n")
    
    start_interactive_session()

