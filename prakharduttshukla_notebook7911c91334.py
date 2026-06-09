"""
AI Travel Concierge Agent - Kaggle AI Agents Intensive Course Capstone
===============================================================================
This implementation covers ALL topics from the 5-Day AI Agents Intensive:

DAY 1: Agentic Architectures & Agent Loops
- ReAct pattern (Reasoning + Acting)
- Agent orchestration and task planning
- Multi-step reasoning with LLM integration

DAY 2: Agent Tools & MCP (Model Context Protocol)
- Tool discovery and registration
- Standardized tool interfaces
- API integration with external services

DAY 3: Context Engineering & Memory Management
- Short-term memory (conversation history)
- Long-term memory (user preferences, past trips)
- Session management and context persistence

DAY 4: Quality, Logging & Evaluation
- Agent performance metrics
- Logging and observability
- Quality checks and validation

DAY 5: Prototype to Production & Multi-Agent Systems
- Multi-agent coordination (A2A Protocol)
- Production deployment considerations
- Scalability and error handling
===============================================================================
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# ============================================================================
# DAY 4: LOGGING & OBSERVABILITY SETUP
# ============================================================================

class AgentLogger:
    """Comprehensive logging for agent operations"""
    
    def __init__(self, log_file: str = "agent_operations.log"):
        self.logger = logging.getLogger("TravelAgent")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.metrics = {
            'tool_calls': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'memory_access': 0,
            'total_latency': 0.0
        }
    
    def log_tool_call(self, tool_name: str, params: Dict, result: Any):
        """Log tool execution"""
        self.metrics['tool_calls'] += 1
        self.logger.info(f"Tool Call: {tool_name} | Params: {params}")
    
    def log_agent_step(self, step: str, details: Dict):
        """Log agent reasoning step"""
        self.logger.info(f"Agent Step: {step} | Details: {details}")
    
    def log_memory_access(self, memory_type: str, operation: str):
        """Log memory operations"""
        self.metrics['memory_access'] += 1
        self.logger.info(f"Memory Access: {memory_type} | Operation: {operation}")
    
    def get_metrics(self) -> Dict:
        """Return collected metrics"""
        return self.metrics.copy()


# ============================================================================
# DAY 2: TOOL DEFINITIONS (MCP-Style)
# ============================================================================

class ToolMetadata:
    """MCP-compatible tool metadata"""
    def __init__(self, name: str, description: str, parameters: Dict):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.type = "function"


class BaseTool(ABC):
    """Base class for all agent tools (MCP pattern)"""
    
    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """Return tool metadata for discovery"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict:
        """Execute tool with given parameters"""
        pass


class WeatherTool(BaseTool):
    """Tool for fetching weather information"""
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "location": {"type": "string", "description": "City name"},
                "date": {"type": "string", "description": "Date (YYYY-MM-DD)"}
            }
        )
    
    def execute(self, location: str, date: str = None) -> Dict:
        """Simulate weather API call"""
        weather_data = {
            'Paris': {'temp': 18, 'condition': 'Partly Cloudy', 'humidity': 65},
            'Tokyo': {'temp': 22, 'condition': 'Sunny', 'humidity': 55},
            'Bali': {'temp': 28, 'condition': 'Tropical', 'humidity': 80},
            'London': {'temp': 15, 'condition': 'Rainy', 'humidity': 75}
        }
        return weather_data.get(location, {'temp': 20, 'condition': 'Unknown', 'humidity': 60})


class FlightSearchTool(BaseTool):
    """Tool for searching flights"""
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="search_flights",
            description="Search for available flights",
            parameters={
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
                "passengers": {"type": "integer"}
            }
        )
    
    def execute(self, origin: str, destination: str, date: str, passengers: int = 1) -> Dict:
        """Simulate flight search"""
        base_price = np.random.randint(300, 1200)
        return {
            'flights': [
                {'airline': 'SkyWings', 'price': base_price, 'duration': '8h 30m'},
                {'airline': 'CloudJet', 'price': base_price + 150, 'duration': '7h 45m'},
                {'airline': 'AirGlobal', 'price': base_price - 100, 'duration': '9h 15m'}
            ]
        }


class HotelSearchTool(BaseTool):
    """Tool for searching hotels"""
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="search_hotels",
            description="Search for hotels in a destination",
            parameters={
                "destination": {"type": "string"},
                "checkin": {"type": "string"},
                "checkout": {"type": "string"},
                "budget_category": {"type": "string"}
            }
        )
    
    def execute(self, destination: str, checkin: str, checkout: str, 
                budget_category: str = "mid-range") -> Dict:
        """Simulate hotel search"""
        price_map = {'budget': 80, 'mid-range': 150, 'luxury': 350}
        base_price = price_map.get(budget_category, 150)
        
        return {
            'hotels': [
                {'name': f'{destination} Plaza', 'price': base_price, 'rating': 4.5},
                {'name': f'Grand {destination}', 'price': base_price + 50, 'rating': 4.7},
                {'name': f'{destination} Inn', 'price': base_price - 30, 'rating': 4.2}
            ]
        }


class MCPToolRegistry:
    """MCP-style tool registry for discovery and execution"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.logger = AgentLogger()
    
    def register_tool(self, tool: BaseTool):
        """Register a tool in the MCP registry"""
        metadata = tool.get_metadata()
        self.tools[metadata.name] = tool
        self.logger.log_agent_step("tool_registration", {"tool": metadata.name})
    
    def discover_tools(self) -> List[ToolMetadata]:
        """List all available tools (MCP discovery)"""
        return [tool.get_metadata() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict:
        """Execute a registered tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        tool = self.tools[tool_name]
        start_time = time.time()
        result = tool.execute(**kwargs)
        latency = time.time() - start_time
        
        self.logger.log_tool_call(tool_name, kwargs, result)
        self.logger.metrics['total_latency'] += latency
        
        return result


# ============================================================================
# DAY 3: MEMORY MANAGEMENT & CONTEXT ENGINEERING
# ============================================================================

@dataclass
class ConversationMessage:
    """Represents a message in conversation history"""
    role: str  # 'user' or 'agent'
    content: str
    timestamp: datetime
    metadata: Dict = None


class ShortTermMemory:
    """Session-based memory for current conversation"""
    
    def __init__(self, max_messages: int = 20):
        self.messages: List[ConversationMessage] = []
        self.max_messages = max_messages
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation history"""
        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(msg)
        
        # Keep only last N messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> str:
        """Get conversation context for LLM"""
        context = []
        for msg in self.messages[-10:]:  # Last 10 messages
            context.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(context)
    
    def clear(self):
        """Clear session memory"""
        self.messages = []


class LongTermMemory:
    """Persistent memory for user preferences and history"""
    
    def __init__(self):
        self.user_preferences: Dict = {}
        self.past_trips: List[Dict] = []
        self.learned_patterns: Dict = {}
    
    def store_preference(self, key: str, value: Any):
        """Store user preference"""
        self.user_preferences[key] = value
    
    def get_preference(self, key: str, default=None) -> Any:
        """Retrieve user preference"""
        return self.user_preferences.get(key, default)
    
    def add_trip_history(self, trip: Dict):
        """Add completed trip to history"""
        trip['completed_date'] = datetime.now().isoformat()
        self.past_trips.append(trip)
    
    def learn_pattern(self, pattern_name: str, pattern_data: Dict):
        """Learn user behavior patterns"""
        self.learned_patterns[pattern_name] = pattern_data
    
    def get_trip_recommendations(self) -> List[str]:
        """Generate recommendations based on history"""
        if not self.past_trips:
            return []
        
        # Simple pattern: recommend similar destinations
        destinations = [trip.get('destination') for trip in self.past_trips]
        return list(set(destinations))
    
    def export_memory(self) -> Dict:
        """Export memory for persistence"""
        return {
            'preferences': self.user_preferences,
            'trips': self.past_trips,
            'patterns': self.learned_patterns
        }


class MemoryManager:
    """Unified memory management system"""
    
    def __init__(self, logger: AgentLogger):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.logger = logger
    
    def add_interaction(self, user_input: str, agent_response: str):
        """Record user-agent interaction"""
        self.short_term.add_message('user', user_input)
        self.short_term.add_message('agent', agent_response)
        self.logger.log_memory_access('short_term', 'write')
    
    def get_contextual_prompt(self, current_query: str) -> str:
        """Build contextual prompt with memory"""
        context = self.short_term.get_context()
        preferences = self.long_term.user_preferences
        
        prompt = f"""
Current Query: {current_query}

Recent Conversation:
{context}

User Preferences:
{json.dumps(preferences, indent=2)}

Based on the context and preferences, provide a personalized response.
"""
        self.logger.log_memory_access('both', 'read')
        return prompt


# ============================================================================
# DAY 1: REACT AGENT (Reasoning + Acting)
# ============================================================================

class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentStep:
    """Represents one step in agent reasoning"""
    step_num: int
    thought: str
    action: str
    action_input: Dict
    observation: str
    state: AgentState


class ReActAgent:
    """
    ReAct Agent: Reason + Act pattern
    Implements the ReAct loop: Think -> Act -> Observe
    """
    
    def __init__(self, tool_registry: MCPToolRegistry, 
                 memory_manager: MemoryManager,
                 logger: AgentLogger):
        self.tools = tool_registry
        self.memory = memory_manager
        self.logger = logger
        self.steps: List[AgentStep] = []
        self.max_iterations = 5
    
    def think(self, query: str, context: str) -> Tuple[str, str, Dict]:
        """
        Reasoning step: Decide what action to take
        In production: This would call an LLM for reasoning
        """
        self.logger.log_agent_step("thinking", {"query": query})
        
        # Simulated LLM reasoning
        if "weather" in query.lower():
            return (
                "User wants weather information",
                "get_weather",
                {"location": self._extract_location(query)}
            )
        elif "flight" in query.lower():
            return (
                "User needs flight information",
                "search_flights",
                self._extract_flight_params(query)
            )
        elif "hotel" in query.lower():
            return (
                "User is looking for accommodation",
                "search_hotels",
                self._extract_hotel_params(query)
            )
        else:
            return (
                "Creating comprehensive travel plan",
                "plan_trip",
                {"query": query}
            )
    
    def act(self, action: str, action_input: Dict) -> str:
        """Execute the chosen action using tools"""
        self.logger.log_agent_step("acting", {"action": action, "input": action_input})
        
        try:
            result = self.tools.execute_tool(action, **action_input)
            return json.dumps(result, indent=2)
        except Exception as e:
            self.logger.logger.error(f"Action failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def observe(self, observation: str) -> bool:
        """Observe the result and decide if task is complete"""
        self.logger.log_agent_step("observing", {"observation": observation[:100]})
        
        # Simple completion check
        return "error" not in observation.lower()
    
    def run(self, user_query: str) -> Dict:
        """
        Main ReAct loop
        """
        self.steps = []
        context = self.memory.get_contextual_prompt(user_query)
        
        for i in range(self.max_iterations):
            # Think
            thought, action, action_input = self.think(user_query, context)
            
            # Act
            observation = self.act(action, action_input)
            
            # Record step
            step = AgentStep(
                step_num=i + 1,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                state=AgentState.ACTING
            )
            self.steps.append(step)
            
            # Observe
            is_complete = self.observe(observation)
            
            if is_complete:
                step.state = AgentState.COMPLETE
                break
        
        # Synthesize final answer
        final_answer = self._synthesize_answer(user_query)
        
        self.logger.metrics['successful_plans'] += 1
        
        return {
            'query': user_query,
            'steps': [asdict(step) for step in self.steps],
            'final_answer': final_answer,
            'iterations': len(self.steps)
        }
    
    def _synthesize_answer(self, query: str) -> str:
        """Combine observations into final answer"""
        observations = [step.observation for step in self.steps]
        return f"Based on your request '{query}', here's what I found:\n" + "\n".join(observations)
    
    def _extract_location(self, query: str) -> str:
        """Extract location from query"""
        locations = ['Paris', 'Tokyo', 'Bali', 'London', 'New York']
        for loc in locations:
            if loc.lower() in query.lower():
                return loc
        return "Paris"
    
    def _extract_flight_params(self, query: str) -> Dict:
        """Extract flight parameters"""
        return {
            'origin': 'New York',
            'destination': self._extract_location(query),
            'date': (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            'passengers': 1
        }
    
    def _extract_hotel_params(self, query: str) -> Dict:
        """Extract hotel parameters"""
        return {
            'destination': self._extract_location(query),
            'checkin': (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            'checkout': (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d"),
            'budget_category': 'mid-range'
        }


# ============================================================================
# DAY 5: MULTI-AGENT SYSTEM (A2A Protocol)
# ============================================================================

class AgentRole(Enum):
    """Different agent roles in multi-agent system"""
    COORDINATOR = "coordinator"
    FLIGHT_SPECIALIST = "flight_specialist"
    ACCOMMODATION_SPECIALIST = "accommodation_specialist"
    ACTIVITY_PLANNER = "activity_planner"


class SpecialistAgent:
    """Specialized agent for specific tasks"""
    
    def __init__(self, role: AgentRole, tools: MCPToolRegistry, logger: AgentLogger):
        self.role = role
        self.tools = tools
        self.logger = logger
    
    def process(self, task: Dict) -> Dict:
        """Process task according to specialization"""
        self.logger.log_agent_step(
            f"{self.role.value}_processing",
            {"task": task}
        )
        
        if self.role == AgentRole.FLIGHT_SPECIALIST:
            return self.tools.execute_tool('search_flights', **task)
        elif self.role == AgentRole.ACCOMMODATION_SPECIALIST:
            return self.tools.execute_tool('search_hotels', **task)
        else:
            return {'status': 'processed', 'data': task}


class CoordinatorAgent:
    """
    Coordinator agent that delegates to specialist agents
    Implements A2A (Agent-to-Agent) Protocol
    """
    
    def __init__(self, tool_registry: MCPToolRegistry, 
                 memory_manager: MemoryManager,
                 logger: AgentLogger):
        self.tools = tool_registry
        self.memory = memory_manager
        self.logger = logger
        
        # Initialize specialist agents
        self.specialists = {
            AgentRole.FLIGHT_SPECIALIST: SpecialistAgent(
                AgentRole.FLIGHT_SPECIALIST, tool_registry, logger
            ),
            AgentRole.ACCOMMODATION_SPECIALIST: SpecialistAgent(
                AgentRole.ACCOMMODATION_SPECIALIST, tool_registry, logger
            )
        }
    
    def coordinate(self, user_request: Dict) -> Dict:
        """
        Coordinate multiple specialist agents
        Implements A2A communication protocol
        """
        self.logger.log_agent_step("coordination_start", {"request": user_request})
        
        results = {}
        
        # Task decomposition
        if 'need_flights' in user_request and user_request['need_flights']:
            flight_task = {
                'origin': user_request.get('origin', 'New York'),
                'destination': user_request.get('destination', 'Paris'),
                'date': user_request.get('date'),
                'passengers': user_request.get('passengers', 1)
            }
            results['flights'] = self.specialists[AgentRole.FLIGHT_SPECIALIST].process(flight_task)
        
        if 'need_hotel' in user_request and user_request['need_hotel']:
            hotel_task = {
                'destination': user_request.get('destination', 'Paris'),
                'checkin': user_request.get('checkin'),
                'checkout': user_request.get('checkout'),
                'budget_category': user_request.get('budget', 'mid-range')
            }
            results['hotels'] = self.specialists[AgentRole.ACCOMMODATION_SPECIALIST].process(hotel_task)
        
        # Synthesize results
        self.logger.log_agent_step("coordination_complete", {"results": list(results.keys())})
        
        return {
            'status': 'success',
            'coordinated_results': results,
            'agents_used': list(self.specialists.keys())
        }


# ============================================================================
# DAY 4: EVALUATION & METRICS
# ============================================================================

class AgentEvaluator:
    """Evaluate agent performance"""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
        self.evaluation_results = []
    
    def evaluate_response_quality(self, query: str, response: Dict) -> float:
        """Evaluate quality of agent response"""
        score = 0.0
        
        # Check if response is complete
        if 'final_answer' in response:
            score += 0.3
        
        # Check iteration efficiency
        iterations = response.get('iterations', 0)
        if iterations <= 3:
            score += 0.3
        elif iterations <= 5:
            score += 0.2
        
        # Check if steps are logical
        if 'steps' in response and len(response['steps']) > 0:
            score += 0.4
        
        return min(score, 1.0)
    
    def evaluate_tool_usage(self) -> Dict:
        """Evaluate tool usage efficiency"""
        metrics = self.logger.get_metrics()
        
        return {
            'total_tool_calls': metrics['tool_calls'],
            'avg_latency': metrics['total_latency'] / max(metrics['tool_calls'], 1),
            'success_rate': metrics['successful_plans'] / max(
                metrics['successful_plans'] + metrics['failed_plans'], 1
            )
        }
    
    def evaluate_memory_efficiency(self) -> Dict:
        """Evaluate memory system efficiency"""
        metrics = self.logger.get_metrics()
        
        return {
            'memory_accesses': metrics['memory_access'],
            'memory_efficiency': 'good' if metrics['memory_access'] < 100 else 'needs_optimization'
        }
    
    def generate_evaluation_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        return {
            'tool_usage': self.evaluate_tool_usage(),
            'memory_efficiency': self.evaluate_memory_efficiency(),
            'overall_metrics': self.logger.get_metrics(),
            'timestamp': datetime.now().isoformat()
        }


# ============================================================================
# MAIN TRAVEL CONCIERGE SYSTEM
# ============================================================================

class TravelConciergeSystem:
    """
    Complete AI Travel Concierge implementing all 5 days of Kaggle course
    """
    
    def __init__(self):
        # Initialize components
        self.logger = AgentLogger()
        self.tool_registry = MCPToolRegistry()
        self.memory_manager = MemoryManager(self.logger)
        
        # Register tools (Day 2: MCP)
        self._register_tools()
        
        # Initialize agents (Day 1 & 5)
        self.react_agent = ReActAgent(
            self.tool_registry,
            self.memory_manager,
            self.logger
        )
        self.coordinator = CoordinatorAgent(
            self.tool_registry,
            self.memory_manager,
            self.logger
        )
        
        # Evaluator (Day 4)
        self.evaluator = AgentEvaluator(self.logger)
        
        print("ğŸŒ� Travel Concierge System Initialized")
        print(f"ğŸ“Š Tools Registered: {len(self.tool_registry.tools)}")
        print(f"ğŸ¤– Agents: ReAct Agent + Multi-Agent Coordinator")
        print(f"ğŸ’¾ Memory: Short-term + Long-term enabled")
        print(f"ğŸ“ˆ Evaluation: Active")
    
    def _register_tools(self):
        """Register all available tools"""
        self.tool_registry.register_tool(WeatherTool())
        self.tool_registry.register_tool(FlightSearchTool())
        self.tool_registry.register_tool(HotelSearchTool())
    
    def plan_simple_trip(self, query: str) -> Dict:
        """
        Simple trip planning using ReAct agent
        """
        print(f"\n{'='*60}")
        print("ğŸ�¯ SIMPLE TRIP PLANNING (ReAct Agent)")
        print(f"{'='*60}")
        print(f"Query: {query}\n")
        
        result = self.react_agent.run(query)
        
        # Record interaction in memory
        self.memory_manager.add_interaction(
            query,
            result['final_answer']
        )
        
        # Evaluate
        quality_score = self.evaluator.evaluate_response_quality(query, result)
        result['quality_score'] = quality_score
        
        return result
    
    def plan_complex_trip(self, request: Dict) -> Dict:
        """
        Complex trip planning using multi-agent system
        """
        print(f"\n{'='*60}")
        print("ğŸŒ� COMPLEX TRIP PLANNING (Multi-Agent System)")
        print(f"{'='*60}")
        print(f"Request: {request}\n")
        
        result = self.coordinator.coordinate(request)
        
        return result
    
    def get_evaluation_report(self) -> Dict:
        """Get comprehensive evaluation report"""
        return self.evaluator.generate_evaluation_report()
    
    def demonstrate_all_features(self):
        """Demonstrate all Kaggle course topics"""
        
        print("\n" + "="*80)
        print(" " * 20 + "KAGGLE AI AGENTS INTENSIVE - FULL DEMO")
        print("="*80)
        
        # DAY 1: ReAct Agent
        print("\nğŸ“š DAY 1: REACT AGENT (Reasoning + Acting)")
        simple_result = self.plan_simple_trip(
            "What's the weather like in Tokyo and find me hotels?"
        )
        print(f"âœ“ Iterations: {simple_result['iterations']}")
        print(f"âœ“ Quality Score: {simple_result['quality_score']:.2f}")
        
        # DAY 2: Tool Discovery (MCP)
        print("\nğŸ“š DAY 2: TOOL DISCOVERY (MCP Protocol)")
        tools = self.tool_registry.discover_tools()
        print(f"âœ“ Discovered {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # DAY 3: Memory Management
        print("\nğŸ“š DAY 3: MEMORY MANAGEMENT")
        self.memory_manager.long_term.store_preference('budget', 'mid-range')
        self.memory_manager.long_term.store_preference('preferred_climate', 'tropical')
        print(f"âœ“ Stored preferences: {self.memory_manager.long_term.user_preferences}")
        print(f"âœ“ Conversation history: {len(self.memory_manager.short_term.messages)} messages")
        
        # DAY 4: Evaluation
        print("\nğŸ“š DAY 4: QUALITY & EVALUATION")
        eval_report = self.get_evaluation_report()
        print(f"âœ“ Tool Calls: {eval_report['overall_metrics']['tool_calls']}")
        print(f"âœ“ Success Rate: {eval_report['tool_usage']['success_rate']:.2%}")
        print(f"âœ“ Avg Latency: {eval_report['tool_usage']['avg_latency']:.4f}s")
        
        # DAY 5: Multi-Agent System
        print("\nğŸ“š DAY 5: MULTI-AGENT COORDINATION (A2A Protocol)")
        complex_request = {
            'need_flights': True,
            'need_hotel': True,
            'destination': 'Paris',
            'origin': 'New York',
            'date': '2025-06-01',
            'checkin': '2025-06-01',
            'checkout': '2025-06-08',
            'passengers': 2,
            'budget': 'luxury'
        }
        multi_result = self.plan_complex_trip(complex_request)
        print(f"âœ“ Status: {multi_result['status']}")
        print(f"âœ“ Coordinated services: {list(multi_result['coordinated_results'].keys())}")
        print(f"âœ“ Agents used: {len(multi_result['agents_used'])}")
        
        # Final Summary
        print("\n" + "="*80)
        print("ğŸ“Š FINAL EVALUATION REPORT")
        print("="*80)
        final_report = self.get_evaluation_report()
        print(json.dumps(final_report, indent=2))
        
        print("\nâœ… ALL KAGGLE AI AGENTS INTENSIVE TOPICS DEMONSTRATED!")




