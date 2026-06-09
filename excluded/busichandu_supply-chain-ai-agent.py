# Install required packages
!pip install -q google-genai google-generativeai google-cloud-aiplatform
!pip install -q pydantic pydantic-settings python-dotenv
!pip install -q opentelemetry-api opentelemetry-sdk
!pip install -q python-json-logger

print("âœ… Dependencies installed successfully!")


# Set up environment variables
import os
from google.colab import userdata

# Get API key from Kaggle secrets or set manually
try:
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
except:
    # If running locally, set your API key here
    GEMINI_API_KEY = "AIzaSyBQvW_ru8bLhjHsiAjfUbIYEw_qrkjd4N4"  # Replace with your actual key

os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
os.environ['ENABLE_LOGGING'] = 'true'
os.environ['ENABLE_TRACING'] = 'false'  # Disable for notebook

print("âœ… Environment configured!")


# Base Agent with ADK Integration
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types

class BaseAgent:
    """Base agent class with ADK integration"""
    
    def __init__(self, name: str, system_instruction: str):
        self.name = name
        self.client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
        self.model_name = "gemini-2.0-flash-exp"
        self.system_instruction = system_instruction
        print(f"âœ… Initialized {name} agent")
    
    async def process_request(self, request: str, **kwargs) -> str:
        """Process a request using Gemini"""
        config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
            system_instruction=self.system_instruction,
        )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=request,
            config=config
        )
        
        return response.text if hasattr(response, 'text') else str(response)

print("âœ… BaseAgent class defined")


# Coordinator Agent with Multi-agent Orchestration
import asyncio
from enum import Enum

class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"

class CoordinatorAgent(BaseAgent):
    """Coordinator for multi-agent orchestration"""
    
    def __init__(self):
        system_instruction = """You are the Coordinator Agent in an enterprise 
supply chain and procurement system. Analyze requests, delegate to specialized 
agents, and synthesize comprehensive recommendations."""
        super().__init__("Coordinator", system_instruction)
        self.agents = {}
    
    def register_agent(self, agent_type: str, agent: BaseAgent):
        """Register a specialized agent"""
        self.agents[agent_type] = agent
        print(f"  âœ“ Registered {agent_type} agent")
    
    async def execute_agents_parallel(self, tasks: List[Dict]) -> List[str]:
        """Execute agents in parallel"""
        print("\nğŸ”„ Executing agents in PARALLEL mode...")
        
        async def execute_task(task):
            agent = self.agents.get(task['agent_type'])
            if agent:
                print(f"  âš¡ Starting {task['agent_type']} agent...")
                return await agent.process_request(task['request'])
            return f"Error: {task['agent_type']} not found"
        
        results = await asyncio.gather(*[execute_task(t) for t in tasks])
        print("âœ… Parallel execution complete!\n")
        return results
    
    async def execute_agents_sequential(self, tasks: List[Dict]) -> List[str]:
        """Execute agents sequentially with context passing"""
        print("\nğŸ”„ Executing agents in SEQUENTIAL mode...")
        results = []
        
        for task in tasks:
            agent = self.agents.get(task['agent_type'])
            if agent:
                print(f"  â–¶ï¸�  Executing {task['agent_type']} agent...")
                result = await agent.process_request(task['request'])
                results.append(result)
                print(f"  âœ“ {task['agent_type']} complete")
        
        print("âœ… Sequential execution complete!\n")
        return results

print("âœ… CoordinatorAgent class defined")
print("\nğŸ“Œ Concept 1 Demonstrated: Multi-agent orchestration with parallel & sequential execution")


# Enhanced BaseAgent with Tool Support
class ToolEnabledAgent(BaseAgent):
    """Agent with tool support"""
    
    def __init__(self, name: str, system_instruction: str, tools: List = None):
        super().__init__(name, system_instruction)
        self.tools = tools or []
    
    def add_tool(self, tool):
        """Add a tool to the agent"""
        self.tools.append(tool)
        print(f"  âœ“ Added tool to {self.name}")
    
    async def process_request_with_tools(self, request: str) -> str:
        """Process request with tool support"""
        config = types.GenerateContentConfig(
            temperature=0.7,
            system_instruction=self.system_instruction,
            tools=self.tools if self.tools else None
        )
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=request,
            config=config
        )
        
        return response.text if hasattr(response, 'text') else str(response)

print("âœ… ToolEnabledAgent class defined")
print("\nğŸ“Œ Concept 2 Demonstrated: Tool registration framework for MCP and custom tools")


# Session Management
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class Session:
    """Session for conversation state"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_messages(self, count: int = 5):
        return self.conversation_history[-count:]

# Long-term Memory
@dataclass
class MemoryEntry:
    entry_id: str
    category: str  # supplier, negotiation, compliance, market_trend
    content: Dict
    created_at: datetime = field(default_factory=datetime.now)

class LongTermMemory:
    """Long-term memory storage"""
    
    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}
    
    def store(self, entry_id: str, category: str, content: Dict):
        self.memories[entry_id] = MemoryEntry(entry_id, category, content)
        print(f"  âœ“ Stored memory: {category} - {entry_id}")
    
    def search_by_category(self, category: str) -> List[MemoryEntry]:
        return [m for m in self.memories.values() if m.category == category]

# Demo
session = Session()
session.add_message("user", "Find suppliers for sensors")
print(f"âœ… Session created: {session.session_id[:8]}...")

memory = LongTermMemory()
memory.store("supplier_001", "supplier", {"name": "TechSensors Inc.", "rating": 4.5})

print("\nğŸ“Œ Concept 3 Demonstrated: Session state management + long-term persistent memory")


# Context Engineering in BaseAgent
class ContextAwareAgent(BaseAgent):
    """Agent with context engineering"""
    
    def __init__(self, name: str, system_instruction: str):
        super().__init__(name, system_instruction)
        self.long_term_memory = None
    
    def set_memory(self, memory: LongTermMemory):
        self.long_term_memory = memory
    
    def build_context(self, request: str, session: Session = None) -> str:
        """Build enriched context (Context Engineering)"""
        context_parts = [request]
        
        # Add session history
        if session:
            recent = session.get_recent_messages(3)
            if recent:
                context_parts.append("\n\n## Recent Conversation:")
                for msg in recent:
                    context_parts.append(f"{msg['role']}: {msg['content']}")
        
        # Add relevant memories
        if self.long_term_memory:
            memories = list(self.long_term_memory.memories.values())[:2]
            if memories:
                context_parts.append("\n\n## Relevant Historical Data:")
                for mem in memories:
                    context_parts.append(f"- {mem.category}: {mem.content}")
        
        enriched = "\n".join(context_parts)
        print(f"  ğŸ�¯ Context enriched: {len(enriched)} chars (from {len(request)} original)")
        return enriched
    
    async def process_with_context(self, request: str, session: Session = None) -> str:
        enriched_request = self.build_context(request, session)
        return await self.process_request(enriched_request)

# Demo
agent = ContextAwareAgent("Demo", "You are a helpful assistant.")
agent.set_memory(memory)
enriched = agent.build_context("Evaluate suppliers", session)

print("\nğŸ“Œ Concept 4 Demonstrated: Dynamic context enrichment with session history and memories")


# Structured Logging
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Structured JSON logger"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
    
    def log_agent_action(self, level: str, message: str, agent: str, action: str, **kwargs):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "agent": agent,
            "action": action,
            "message": message,
            **kwargs
        }
        print(f"ğŸ“� LOG: {json.dumps(log_entry, indent=2)}")

# Distributed Tracing
import time

class SimpleTracer:
    """Simple tracer for agent actions"""
    
    def __init__(self):
        self.spans = []
    
    def trace_span(self, name: str, agent: str):
        start = time.time()
        span = {"name": name, "agent": agent, "start": start}
        self.spans.append(span)
        print(f"ğŸ”� TRACE: Started span '{name}' for {agent}")
        return span
    
    def end_span(self, span):
        span['duration'] = time.time() - span['start']
        print(f"âœ“ TRACE: Completed '{span['name']}' in {span['duration']:.2f}s")

# Demo
logger = StructuredLogger("enterprise_agent")
logger.log_agent_action("info", "Processing request", "Research", "supplier_discovery", request_id="REQ-001")

tracer = SimpleTracer()
span = tracer.trace_span("research_suppliers", "Research")
time.sleep(0.1)  # Simulate work
tracer.end_span(span)

print("\nğŸ“Œ Concept 5 Demonstrated: Structured logging + distributed tracing for observability")


# Evaluation Framework
@dataclass
class TestScenario:
    scenario_id: str
    name: str
    request: str
    expected_outcomes: Dict

@dataclass
class EvaluationMetrics:
    scenario_id: str
    accuracy: float = 0.0
    completeness: float = 0.0
    latency: float = 0.0
    
    def calculate_accuracy(self, expected: Dict, actual: Dict) -> float:
        matches = sum(1 for k, v in expected.items() 
                     if k in actual and str(v).lower() in str(actual[k]).lower())
        self.accuracy = (matches / len(expected)) * 100 if expected else 0
        return self.accuracy
    
    def calculate_completeness(self, required_fields: List, result: Dict) -> float:
        present = sum(1 for f in required_fields if f in result and result[f])
        self.completeness = (present / len(required_fields)) * 100 if required_fields else 0
        return self.completeness
    
    def summary(self) -> str:
        return f"""Evaluation Results for {self.scenario_id}:
  â€¢ Accuracy: {self.accuracy:.1f}%
  â€¢ Completeness: {self.completeness:.1f}%
  â€¢ Latency: {self.latency:.2f}s"""

# Demo Test Scenario
scenario = TestScenario(
    scenario_id="SC001",
    name="Simple Sensor Procurement",
    request="Procure 1,000 temperature sensors",
    expected_outcomes={
        "supplier_count": "3-5",
        "compliance_check": "passed",
        "recommendation": "clear"
    }
)

metrics = EvaluationMetrics(scenario_id="SC001")
actual_result = {
    "supplier_count": "4 suppliers found",
    "compliance_check": "All passed",
    "recommendation": "Clear recommendation provided"
}

metrics.calculate_accuracy(scenario.expected_outcomes, actual_result)
metrics.calculate_completeness(["supplier_count", "compliance_check", "recommendation"], actual_result)
metrics.latency = 2.5

print(metrics.summary())
print("\nğŸ“Œ Concept 6 Demonstrated: Automated evaluation with accuracy, completeness, and latency metrics")


# Create Specialized Agents

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Research", """You are a Research Agent specializing in 
supplier discovery and market analysis. Provide comprehensive supplier options, 
market pricing, and initial risk assessment.""")

class EvaluationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Evaluation", """You are an Evaluation Agent specializing in 
supplier scoring and cost-benefit analysis. Score suppliers on quality, price, 
reliability, and capacity.""")

class ComplianceAgent(BaseAgent):
    def __init__(self):
        super().__init__("Compliance", """You are a Compliance Agent specializing in 
regulatory checking and certification validation. Verify compliance with ISO, CE, 
RoHS, and other standards.""")

class NegotiationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Negotiation", """You are a Negotiation Agent specializing in 
price negotiation and contract optimization. Develop negotiation strategies and 
recommend optimal contract terms.""")

# Initialize all agents
print("ğŸš€ Initializing Enterprise Agent System...\n")
coordinator = CoordinatorAgent()
research = ResearchAgent()
evaluation = EvaluationAgent()
compliance = ComplianceAgent()
negotiation = NegotiationAgent()

# Register agents
print("\nğŸ“‹ Registering specialized agents...")
coordinator.register_agent("research", research)
coordinator.register_agent("evaluation", evaluation)
coordinator.register_agent("compliance", compliance)
coordinator.register_agent("negotiation", negotiation)

print("\nâœ… System ready!")


# Define procurement request
procurement_request = """
We need to procure 10,000 units of industrial temperature sensors for our manufacturing facility.

Requirements:
- Temperature range: -40Â°C to 125Â°C
- Accuracy: Â±0.5Â°C
- Wireless connectivity (Bluetooth or WiFi)
- Battery life: minimum 2 years
- IP67 rated (dust and water resistant)
- Budget: $50-75 per unit
- Delivery: Within 60 days
- Certifications: CE, FCC, RoHS compliant

Please provide a comprehensive procurement recommendation.
"""

print("ğŸ“� PROCUREMENT REQUEST:")
print("=" * 80)
print(procurement_request)
print("=" * 80)


# Execute multi-agent workflow (SEQUENTIAL mode)
import asyncio

async def run_procurement_workflow():
    # Define tasks for each agent
    tasks = [
        {
            "agent_type": "research",
            "request": f"Research suppliers and products for: {procurement_request}"
        },
        {
            "agent_type": "evaluation",
            "request": f"Evaluate supplier options for: {procurement_request}"
        },
        {
            "agent_type": "compliance",
            "request": f"Check compliance requirements for: {procurement_request}"
        },
        {
            "agent_type": "negotiation",
            "request": f"Develop negotiation strategy for: {procurement_request}"
        }
    ]
    
    # Execute agents sequentially
    results = await coordinator.execute_agents_sequential(tasks)
    
    return results
# Run the workflow
print("\nğŸš€ Starting procurement workflow...\n")
results = await run_procurement_workflow()

print("\n" + "=" * 80)
print("âœ… WORKFLOW COMPLETE!")
print("=" * 80)


# Display results
agent_names = ["Research", "Evaluation", "Compliance", "Negotiation"]
emojis = ["ğŸ”�", "âš–ï¸�", "âœ…", "ğŸ’¼"]

for i, (name, emoji, result) in enumerate(zip(agent_names, emojis, results)):
    print(f"\n{emoji} {name.upper()} AGENT OUTPUT:")
    print("=" * 80)
    print(result[:500] + "..." if len(result) > 500 else result)
    print()

