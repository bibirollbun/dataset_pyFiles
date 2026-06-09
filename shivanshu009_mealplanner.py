import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import random
from collections import deque
import time

# Optional: Claude API integration
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ============================================================================
# Part 1: Core Agent Components
# ============================================================================

class AgentState(Enum):
    """Different operational states for the agent"""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    ACTING = "acting"
    REFLECTING = "reflecting"

@dataclass
class Environment:
    """Represents the world state that our agent operates in"""
    user_preferences: Dict[str, Any]
    available_ingredients: List[str]
    meal_history: List[Dict]
    time_of_day: str
    season: str
    constraints: Dict[str, Any]
    
    def get_state(self) -> Dict:
        return {
            "preferences": self.user_preferences,
            "ingredients": self.available_ingredients,
            "history_length": len(self.meal_history),
            "time": self.time_of_day,
            "season": self.season
        }

@dataclass
class Perception:
    timestamp: datetime
    environment_state: Dict
    user_intent: str
    context: Dict[str, Any]

@dataclass
class Action:
    action_type: str
    parameters: Dict[str, Any]
    timestamp: datetime
    expected_outcome: str
    tool_used: Optional[str] = None

# ============================================================================
# Part 2: Tools for the Agent
# ============================================================================

class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class NutritionCalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="nutrition_calculator",
            description="Calculate nutritional values for meals"
        )
    
    def execute(self, ingredients: List[str], portions: int = 1) -> Dict[str, Any]:
        # Basic nutrition database - could be expanded
        nutrition_db = {
            "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fats": 3.6},
            "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fats": 0.3},
            "broccoli": {"calories": 55, "protein": 3.7, "carbs": 11, "fats": 0.6},
            "salmon": {"calories": 208, "protein": 20, "carbs": 0, "fats": 13},
            "quinoa": {"calories": 120, "protein": 4.4, "carbs": 21, "fats": 1.9}
        }
        
        total = {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}
        for ingredient in ingredients:
            key = ingredient.lower()
            if key in nutrition_db:
                for nutrient in total:
                    total[nutrient] += nutrition_db[key][nutrient] * portions
        
        return total

class RecipeSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="recipe_search",
            description="Search for recipes based on criteria"
        )
        self.recipe_db = self._load_recipes()
    
    def _load_recipes(self) -> List[Dict]:
        # In a real system, this would connect to a database
        return [
            {
                "name": "Grilled Salmon with Quinoa",
                "ingredients": ["salmon", "quinoa", "lemon", "olive oil"],
                "cuisine": "mediterranean",
                "difficulty": "medium",
                "time": 30,
                "tags": ["healthy", "protein-rich"]
            },
            {
                "name": "Chicken Stir Fry",
                "ingredients": ["chicken", "broccoli", "soy sauce", "ginger"],
                "cuisine": "asian",
                "difficulty": "easy",
                "time": 25,
                "tags": ["quick", "protein-rich"]
            }
        ]
    
    def execute(self, cuisine: Optional[str] = None, 
                max_time: Optional[int] = None,
                dietary_restrictions: List[str] = None) -> List[Dict]:
        results = self.recipe_db.copy()
        
        if cuisine:
            results = [r for r in results if r["cuisine"] == cuisine]
        if max_time:
            results = [r for r in results if r["time"] <= max_time]
        
        return results

class ShoppingListTool(Tool):
    def __init__(self):
        super().__init__(
            name="shopping_list_generator",
            description="Generate shopping lists from meal plans"
        )
    
    def execute(self, meal_plan: Dict) -> Dict[str, int]:
        shopping_list = {}
        for date, meals in meal_plan.items():
            for meal in meals:
                for ingredient in meal.get("ingredients", []):
                    shopping_list[ingredient] = shopping_list.get(ingredient, 0) + 1
        return shopping_list

class ToolRegistry:
    """Manages available tools"""
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

# ============================================================================
# Part 3: Prompt Engineering
# ============================================================================

class PromptTemplate:
    
    @staticmethod
    def chain_of_thought_prompt(task: str, context: Dict) -> str:
        return f"""Let's solve this step by step:

Task: {task}

Context:
{json.dumps(context, indent=2)}

Please follow these steps:
1. Understand the requirements
2. Break down the problem
3. Consider constraints and preferences
4. Generate solutions
5. Evaluate and select the best option
6. Explain your reasoning

Your response:"""

    @staticmethod
    def react_prompt(observation: str, previous_actions: List[str]) -> str:
        # ReAct pattern: Reasoning + Acting
        return f"""You are a meal planning agent. Use the ReAct framework:

Previous Actions: {previous_actions}

Current Observation: {observation}

Think step by step:
Thought: [Your reasoning about what to do next]
Action: [The action you will take]
Action Input: [Parameters for the action]

Format your response as JSON with keys: thought, action, action_input"""

    @staticmethod
    def few_shot_prompt(examples: List[Dict], query: str) -> str:
        example_str = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}" 
            for ex in examples
        ])
        return f"""Here are some examples:

{example_str}

Now solve this:
Input: {query}
Output:"""

# ============================================================================
# Part 4: Memory Systems
# ============================================================================

@dataclass
class MemoryItem:
    content: Any
    timestamp: datetime
    importance: float  # scale from 0 to 1
    access_count: int = 0
    memory_type: str = "episodic"

class ShortTermMemory:
    """Working memory with limited capacity (Miller's Law: 7Â±2)"""
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.memory: deque = deque(maxlen=capacity)
    
    def add(self, item: MemoryItem):
        self.memory.append(item)
    
    def get_recent(self, n: int = 5) -> List[MemoryItem]:
        return list(self.memory)[-n:]
    
    def clear(self):
        self.memory.clear()

class LongTermMemory:
    def __init__(self):
        self.semantic_memory: Dict[str, Any] = {}  # facts
        self.episodic_memory: List[MemoryItem] = []  # experiences
        self.procedural_memory: Dict[str, Callable] = {}  # procedures
    
    def store_semantic(self, key: str, value: Any):
        self.semantic_memory[key] = value
    
    def store_episodic(self, item: MemoryItem):
        item.memory_type = "episodic"
        self.episodic_memory.append(item)
    
    def retrieve_semantic(self, key: str) -> Optional[Any]:
        return self.semantic_memory.get(key)
    
    def retrieve_episodic(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        # Simple keyword matching - in production use embeddings
        relevant = [m for m in self.episodic_memory 
                   if query.lower() in str(m.content).lower()]
        return sorted(relevant, key=lambda x: x.importance, reverse=True)[:top_k]

class MemoryManager:
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
    
    def consolidate(self):
        # Move important items to long-term storage
        for item in self.short_term.get_recent():
            if item.importance > 0.7:
                self.long_term.store_episodic(item)

# ============================================================================
# Part 5: Planning and Reasoning
# ============================================================================

class PlanningStrategy(Enum):
    FORWARD = "forward_chaining"
    BACKWARD = "backward_chaining"
    HIERARCHICAL = "hierarchical_task_network"
    REACTIVE = "reactive"

@dataclass
class Plan:
    goal: str
    steps: List[Dict[str, Any]]
    strategy: PlanningStrategy
    estimated_time: int
    success_criteria: Dict[str, Any]

class Planner:
    
    def __init__(self):
        self.current_plan: Optional[Plan] = None
        self.plan_history: List[Plan] = []
    
    def create_plan(self, goal: str, context: Dict, 
                    strategy: PlanningStrategy = PlanningStrategy.HIERARCHICAL) -> Plan:
        
        if strategy == PlanningStrategy.HIERARCHICAL:
            return self._hierarchical_planning(goal, context)
        elif strategy == PlanningStrategy.FORWARD:
            return self._forward_planning(goal, context)
        else:
            return self._reactive_planning(goal, context)
    
    def _hierarchical_planning(self, goal: str, context: Dict) -> Plan:
        """Hierarchical Task Network planning"""
        steps = [
            {
                "step": 1,
                "task": "Analyze user preferences and constraints",
                "subtasks": ["Load preferences", "Identify restrictions", "Check available ingredients"],
                "status": "pending"
            },
            {
                "step": 2,
                "task": "Search for suitable recipes",
                "subtasks": ["Query recipe database", "Filter by constraints", "Rank by preferences"],
                "status": "pending"
            },
            {
                "step": 3,
                "task": "Generate meal schedule",
                "subtasks": ["Assign meals to days", "Balance nutrition", "Ensure variety"],
                "status": "pending"
            },
            {
                "step": 4,
                "task": "Create shopping list",
                "subtasks": ["Extract ingredients", "Aggregate quantities", "Categorize items"],
                "status": "pending"
            }
        ]
        
        return Plan(
            goal=goal,
            steps=steps,
            strategy=PlanningStrategy.HIERARCHICAL,
            estimated_time=30,
            success_criteria={"meals_generated": 7, "nutrition_balanced": True}
        )
    
    def _forward_planning(self, goal: str, context: Dict) -> Plan:
        """Forward chaining from current state to goal"""
        steps = [
            {"step": 1, "action": "perceive_environment", "status": "pending"},
            {"step": 2, "action": "identify_constraints", "status": "pending"},
            {"step": 3, "action": "generate_options", "status": "pending"},
            {"step": 4, "action": "evaluate_options", "status": "pending"},
            {"step": 5, "action": "execute_best_option", "status": "pending"}
        ]
        
        return Plan(
            goal=goal,
            steps=steps,
            strategy=PlanningStrategy.FORWARD,
            estimated_time=20,
            success_criteria={"goal_achieved": True}
        )
    
    def _reactive_planning(self, goal: str, context: Dict) -> Plan:
        """Reactive planning - respond to immediate stimuli"""
        steps = [
            {"step": 1, "action": "observe_and_react", "status": "pending"}
        ]
        
        return Plan(
            goal=goal,
            steps=steps,
            strategy=PlanningStrategy.REACTIVE,
            estimated_time=5,
            success_criteria={"response_generated": True}
        )
    
    def execute_step(self, step_index: int) -> Dict[str, Any]:
        """Execute a plan step with self-monitoring"""
        if not self.current_plan or step_index >= len(self.current_plan.steps):
            return {"success": False, "error": "Invalid step"}
        
        step = self.current_plan.steps[step_index]
        step["status"] = "executing"
        
        # Simulate step execution
        result = {
            "success": True,
            "step": step_index,
            "output": f"Executed: {step.get('task', step.get('action'))}",
            "timestamp": datetime.now()
        }
        
        step["status"] = "completed"
        return result

# ============================================================================
# TOPIC 6: MULTI-AGENT SYSTEMS & COLLABORATION
# ============================================================================

class AgentRole(Enum):
    """Different agent roles in multi-agent system"""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    EXECUTOR = "executor"
    CRITIC = "critic"

@dataclass
class Message:
    """Inter-agent communication message"""
    sender: str
    receiver: str
    content: Dict[str, Any]
    message_type: str  # request, response, broadcast, query
    timestamp: datetime
    priority: int = 1

class BaseAgent:
    """Base class for agents in multi-agent system"""
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.knowledge_base: Dict[str, Any] = {}
    
    def receive_message(self, message: Message):
        self.inbox.append(message)
    
    def send_message(self, message: Message):
        self.outbox.append(message)
    
    def process_messages(self):
        """Process incoming messages"""
        processed = []
        for msg in self.inbox:
            response = self.handle_message(msg)
            if response:
                processed.append(response)
        self.inbox.clear()
        return processed
    
    def handle_message(self, message: Message) -> Optional[Message]:
        raise NotImplementedError

class NutritionistAgent(BaseAgent):
    """Specialist agent for nutrition analysis"""
    def __init__(self):
        super().__init__("nutritionist_001", AgentRole.SPECIALIST)
        self.tool = NutritionCalculatorTool()
    
    def handle_message(self, message: Message) -> Optional[Message]:
        if message.message_type == "request" and "analyze_nutrition" in message.content:
            ingredients = message.content.get("ingredients", [])
            nutrition = self.tool.execute(ingredients=ingredients)
            
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                content={"nutrition_analysis": nutrition},
                message_type="response",
                timestamp=datetime.now()
            )
        return None

class ChefAgent(BaseAgent):
    """Specialist agent for recipe selection"""
    def __init__(self):
        super().__init__("chef_001", AgentRole.SPECIALIST)
        self.tool = RecipeSearchTool()
    
    def handle_message(self, message: Message) -> Optional[Message]:
        if message.message_type == "request" and "find_recipes" in message.content:
            criteria = message.content.get("criteria", {})
            recipes = self.tool.execute(**criteria)
            
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                content={"recipes": recipes},
                message_type="response",
                timestamp=datetime.now()
            )
        return None

class CoordinatorAgent(BaseAgent):
    """Coordinator agent for orchestrating multi-agent collaboration"""
    def __init__(self):
        super().__init__("coordinator_001", AgentRole.COORDINATOR)
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent: BaseAgent):
        self.agents[agent.agent_id] = agent
    
    def delegate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate tasks to appropriate specialist agents"""
        results = {}
        
        # Delegate to nutritionist
        if "nutrition_check" in task:
            msg = Message(
                sender=self.agent_id,
                receiver="nutritionist_001",
                content={"analyze_nutrition": True, "ingredients": task.get("ingredients", [])},
                message_type="request",
                timestamp=datetime.now()
            )
            self.send_message(msg)
            if "nutritionist_001" in self.agents:
                self.agents["nutritionist_001"].receive_message(msg)
                responses = self.agents["nutritionist_001"].process_messages()
                if responses:
                    results["nutrition"] = responses[0].content
        
        # Delegate to chef
        if "recipe_search" in task:
            msg = Message(
                sender=self.agent_id,
                receiver="chef_001",
                content={"find_recipes": True, "criteria": task.get("criteria", {})},
                message_type="request",
                timestamp=datetime.now()
            )
            self.send_message(msg)
            if "chef_001" in self.agents:
                self.agents["chef_001"].receive_message(msg)
                responses = self.agents["chef_001"].process_messages()
                if responses:
                    results["recipes"] = responses[0].content
        
        return results

# ============================================================================
# TOPIC 7: AGENT EVALUATION & MONITORING
# ============================================================================

@dataclass
class Metric:
    """Performance metric"""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentEvaluator:
    """Evaluates agent performance"""
    def __init__(self):
        self.metrics: List[Metric] = []
        self.benchmarks: Dict[str, float] = {
            "response_time": 2.0,  # seconds
            "success_rate": 0.9,
            "user_satisfaction": 0.8,
            "accuracy": 0.85
        }
    
    def record_metric(self, name: str, value: float, metadata: Dict = None):
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.metrics.append(metric)
    
    def evaluate_performance(self) -> Dict[str, Any]:
        """Comprehensive performance evaluation"""
        if not self.metrics:
            return {"status": "no_data"}
        
        # Calculate aggregates
        metric_groups = {}
        for metric in self.metrics:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = []
            metric_groups[metric.name].append(metric.value)
        
        evaluation = {}
        for name, values in metric_groups.items():
            avg_value = sum(values) / len(values)
            benchmark = self.benchmarks.get(name, 0.5)
            
            evaluation[name] = {
                "average": avg_value,
                "benchmark": benchmark,
                "meets_benchmark": avg_value >= benchmark,
                "sample_size": len(values)
            }
        
        return evaluation
    
    def generate_report(self) -> str:
        """Generate evaluation report"""
        eval_results = self.evaluate_performance()
        
        report = "AGENT PERFORMANCE REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for metric_name, results in eval_results.items():
            report += f"{metric_name.upper()}:\n"
            report += f"  Average: {results.get('average', 0):.3f}\n"
            report += f"  Benchmark: {results.get('benchmark', 0):.3f}\n"
            report += f"  Status: {'âœ“ PASS' if results.get('meets_benchmark') else 'âœ— FAIL'}\n\n"
        
        return report

# ============================================================================
# TOPIC 8: RETRIEVAL AUGMENTED GENERATION (RAG)
# ============================================================================

class DocumentStore:
    """Simple vector store for RAG"""
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self._initialize_documents()
    
    def _initialize_documents(self):
        """Initialize with meal planning knowledge"""
        self.documents = [
            {
                "id": "doc1",
                "content": "Mediterranean diet emphasizes olive oil, fish, vegetables, and whole grains.",
                "metadata": {"topic": "nutrition", "diet_type": "mediterranean"}
            },
            {
                "id": "doc2",
                "content": "Protein requirements: 0.8g per kg body weight for adults, higher for athletes.",
                "metadata": {"topic": "nutrition", "category": "protein"}
            },
            {
                "id": "doc3",
                "content": "Meal prep tips: Cook in batches, store properly, plan variety for the week.",
                "metadata": {"topic": "meal_planning", "category": "tips"}
            }
        ]
    
    def add_document(self, content: str, metadata: Dict):
        doc_id = f"doc{len(self.documents) + 1}"
        self.documents.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata
        })
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simplified retrieval (in production, use embeddings)"""
        query_terms = set(query.lower().split())
        
        scored_docs = []
        for doc in self.documents:
            doc_terms = set(doc["content"].lower().split())
            score = len(query_terms & doc_terms)
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:top_k]]

class RAGSystem:
    """Retrieval Augmented Generation system"""
    def __init__(self, api_key: Optional[str] = None):
        self.document_store = DocumentStore()
        if ANTHROPIC_AVAILABLE and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None
    
    def generate_with_context(self, query: str) -> str:
        """Generate response with retrieved context"""
        # Retrieve relevant documents
        relevant_docs = self.document_store.retrieve(query)
        
        # Build context
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        
        if not self.client:
            return f"[Context-based answer]\n\nRelevant Information:\n{context}\n\nQuery: {query}\n\nBased on the context above, the meal planning system can provide recommendations using the retrieved knowledge."
        
        # Generate with Claude
        prompt = f"""Use the following context to answer the question:

Context:
{context}

Question: {query}

Answer based on the context provided:"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"

# ============================================================================
# TOPIC 9: AGENTIC WORKFLOWS (Sequential, Parallel, Hierarchical)
# ============================================================================

class WorkflowType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    CONDITIONAL = "conditional"

@dataclass
class WorkflowTask:
    """Individual task in a workflow"""
    task_id: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Any] = None

class Workflow:
    """Agentic workflow orchestration"""
    def __init__(self, workflow_type: WorkflowType):
        self.workflow_type = workflow_type
        self.tasks: List[WorkflowTask] = []
        self.execution_log: List[Dict] = []
    
    def add_task(self, task: WorkflowTask):
        self.tasks.append(task)
    
    def execute(self) -> Dict[str, Any]:
        """Execute workflow based on type"""
        if self.workflow_type == WorkflowType.SEQUENTIAL:
            return self._execute_sequential()
        elif self.workflow_type == WorkflowType.PARALLEL:
            return self._execute_parallel()
        elif self.workflow_type == WorkflowType.HIERARCHICAL:
            return self._execute_hierarchical()
        else:
            return self._execute_conditional()
    
    def _execute_sequential(self) -> Dict[str, Any]:
        """Execute tasks in sequence"""
        results = {}
        for task in self.tasks:
            print(f"Executing: {task.task_id}")
            task.status = "running"
            # Simulate execution
            task.result = f"Result of {task.action}"
            task.status = "completed"
            results[task.task_id] = task.result
            
            self.execution_log.append({
                "task_id": task.task_id,
                "timestamp": datetime.now(),
                "status": "completed"
            })
        
        return {"workflow_type": "sequential", "results": results}
    
    def _execute_parallel(self) -> Dict[str, Any]:
        """Simulate parallel execution"""
        results = {}
        print("Executing tasks in parallel...")
        
        for task in self.tasks:
            task.status = "running"
            task.result = f"Parallel result of {task.action}"
            task.status = "completed"
            results[task.task_id] = task.result
        
        return {"workflow_type": "parallel", "results": results}
    
    def _execute_hierarchical(self) -> Dict[str, Any]:
        """Execute hierarchical task decomposition"""
        results = {}
        
        # Group by dependencies (simplified)
        levels = self._organize_by_dependencies()
        
        for level, tasks in levels.items():
            print(f"Executing level {level}")
            for task in tasks:
                task.status = "running"
                task.result = f"Hierarchical result of {task.action}"
                task.status = "completed"
                results[task.task_id] = task.result
        
        return {"workflow_type": "hierarchical", "results": results}
    
    def _execute_conditional(self) -> Dict[str, Any]:
        """Execute with conditional branching"""
        results = {}
        
        for task in self.tasks:
            # Check dependencies
            deps_met = all(
                any(t.task_id == dep and t.status == "completed" 
                    for t in self.tasks)
                for dep in task.dependencies
            )
            
            if deps_met or not task.dependencies:
                task.status = "running"
                task.result = f"Conditional result of {task.action}"
                task.status = "completed"
                results[task.task_id] = task.result
        
        return {"workflow_type": "conditional", "results": results}
    
    def _organize_by_dependencies(self) -> Dict[int, List[WorkflowTask]]:
        """Organize tasks by dependency levels"""
        levels = {}
        processed = set()
        
        level = 0
        while len(processed) < len(self.tasks):
            levels[level] = []
            for task in self.tasks:
                if task.task_id not in processed:
                    deps_met = all(dep in processed for dep in task.dependencies)
                    if deps_met:
                        levels[level].append(task)
                        processed.add(task.task_id)
            level += 1
        
        return levels

# ============================================================================
# TOPIC 10: ERROR HANDLING & SELF-CORRECTION
# ============================================================================

class ErrorType(Enum):
    """Types of errors agents can encounter"""
    VALIDATION_ERROR = "validation_error"
    TOOL_ERROR = "tool_error"
    PLANNING_ERROR = "planning_error"
    EXECUTION_ERROR = "execution_error"
    CONSTRAINT_VIOLATION = "constraint_violation"

@dataclass
class AgentError:
    """Error with context for recovery"""
    error_type: ErrorType
    message: str
    context: Dict[str, Any]
    timestamp: datetime
    recoverable: bool = True
    recovery_strategy: Optional[str] = None

class ErrorHandler:
    """Handles errors and implements self-correction"""
    def __init__(self):
        self.error_log: List[AgentError] = []
        self.recovery_strategies: Dict[ErrorType, Callable] = {
            ErrorType.VALIDATION_ERROR: self._recover_validation,
            ErrorType.TOOL_ERROR: self._recover_tool,
            ErrorType.PLANNING_ERROR: self._recover_planning,
            ErrorType.EXECUTION_ERROR: self._recover_execution,
            ErrorType.CONSTRAINT_VIOLATION: self._recover_constraint
        }
    
    def handle_error(self, error: AgentError) -> Dict[str, Any]:
        """Handle error with appropriate recovery strategy"""
        self.error_log.append(error)
        
        print(f"âš ï¸�  Error detected: {error.error_type.value}")
        print(f"   Message: {error.message}")
        
        if not error.recoverable:
            return {"recovered": False, "reason": "Error not recoverable"}
        
        # Apply recovery strategy
        strategy = self.recovery_strategies.get(error.error_type)
        if strategy:
            recovery_result = strategy(error)
            print(f"âœ“ Recovery attempted: {recovery_result['strategy']}")
            return recovery_result
        
        return {"recovered": False, "reason": "No recovery strategy found"}
    
    def _recover_validation(self, error: AgentError) -> Dict[str, Any]:
        """Recover from validation errors"""
        return {
            "recovered": True,
            "strategy": "Relaxed validation constraints",
            "action": "Retry with adjusted parameters"
        }
    
    def _recover_tool(self, error: AgentError) -> Dict[str, Any]:
        """Recover from tool errors"""
        return {
            "recovered": True,
            "strategy": "Fallback to alternative tool",
            "action": "Use backup method"
        }
    
    def _recover_planning(self, error: AgentError) -> Dict[str, Any]:
        """Recover from planning errors"""
        return {
            "recovered": True,
            "strategy": "Replan with simpler strategy",
            "action": "Switch from hierarchical to sequential planning"
        }
    
    def _recover_execution(self, error: AgentError) -> Dict[str, Any]:
        """Recover from execution errors"""
        return {
            "recovered": True,
            "strategy": "Retry with exponential backoff",
            "action": "Re-execute failed step"
        }
    
    def _recover_constraint(self, error: AgentError) -> Dict[str, Any]:
        """Recover from constraint violations"""
        return {
            "recovered": True,
            "strategy": "Adjust constraints or find alternative solution",
            "action": "Negotiate constraint relaxation"
        }
    
    def self_correct(self, action: Action, feedback: Dict[str, Any]) -> Action:
        """Self-correction based on feedback"""
        if feedback.get("success"):
            return action
        
        # Create corrected action
        corrected = Action(
            action_type=action.action_type,
            parameters=action.parameters.copy(),
            timestamp=datetime.now(),
            expected_outcome=action.expected_outcome,
            tool_used=action.tool_used
        )
        
        # Apply corrections based on feedback
        if "parameter_error" in feedback:
            corrected.parameters.update(feedback.get("suggested_parameters", {}))
        
        print(f"ğŸ”„ Self-correction applied to {action.action_type}")
        return corrected

# ============================================================================
# MAIN AGENT: Integrating All Topics
# ============================================================================

class MealPlanningAgent:
    """
    Complete AI Agent demonstrating all Kaggle course topics:
    - Agent Fundamentals
    - Tool Use
    - Prompt Engineering
    - Memory Systems
    - Planning & Reasoning
    - Multi-Agent Collaboration
    - Evaluation
    - RAG
    - Workflows
    - Error Handling
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Core components
        self.agent_id = "meal_planner_main"
        self.state = AgentState.IDLE
        
        # Topic 1: Environment
        self.environment = Environment(
            user_preferences={},
            available_ingredients=[],
            meal_history=[],
            time_of_day="morning",
            season="spring",
            constraints={}
        )
        
        # Topic 2: Tools
        self.tool_registry = ToolRegistry()
        self._register_tools()
        
        # Topic 4: Memory
        self.memory_manager = MemoryManager()
        
        # Topic 5: Planning
        self.planner = Planner()
        
        # Topic 6: Multi-Agent
        self.coordinator = CoordinatorAgent()
        self._setup_multi_agent_system()
        
        # Topic 7: Evaluation
        self.evaluator = AgentEvaluator()
        
        # Topic 8: RAG
        self.rag_system = RAGSystem(api_key=api_key)
        
        # Topic 9: Workflows
        self.current_workflow: Optional[Workflow] = None
        
        # Topic 10: Error Handling
        self.error_handler = ErrorHandler()
        
        # API client
        if ANTHROPIC_AVAILABLE and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None
        
        print("âœ“ Meal Planning Agent initialized with all course topics!")
    
    def _register_tools(self):
        """Register all available tools"""
        self.tool_registry.register(NutritionCalculatorTool())
        self.tool_registry.register(RecipeSearchTool())
        self.tool_registry.register(ShoppingListTool())
    
    def _setup_multi_agent_system(self):
        """Setup multi-agent collaboration"""
        nutritionist = NutritionistAgent()
        chef = ChefAgent()
        
        self.coordinator.register_agent(nutritionist)
        self.coordinator.register_agent(chef)
    
    def perceive(self, user_input: Dict[str, Any]) -> Perception:
        """Topic 1: Perceive environment and user intent"""
        self.state = AgentState.PERCEIVING
        
        perception = Perception(
            timestamp=datetime.now(),
            environment_state=self.environment.get_state(),
            user_intent=user_input.get("intent", "generate_meal_plan"),
            context=user_input
        )
        
        # Store in short-term memory
        memory_item = MemoryItem(
            content=perception,
            timestamp=datetime.now(),
            importance=0.8
        )
        self.memory_manager.short_term.add(memory_item)
        
        return perception
    
    def reason_and_plan(self, perception: Perception) -> Plan:
        """Topic 5: Reason about the situation and create a plan"""
        self.state = AgentState.PLANNING
        
        # Use chain of thought reasoning
        context = {
            "user_intent": perception.user_intent,
            "environment": perception.environment_state,
            "available_tools": self.tool_registry.list_tools()
        }
        
        # Create hierarchical plan
        plan = self.planner.create_plan(
            goal=perception.user_intent,
            context=context,
            strategy=PlanningStrategy.HIERARCHICAL
        )
        
        self.planner.current_plan = plan
        
        # Record in memory
        memory_item = MemoryItem(
            content=plan,
            timestamp=datetime.now(),
            importance=0.9,
            memory_type="procedural"
        )
        self.memory_manager.short_term.add(memory_item)
        
        return plan
    
    def execute_with_workflow(self, plan: Plan) -> Dict[str, Any]:
        """Topic 9: Execute plan using agentic workflow"""
        self.state = AgentState.ACTING
        
        # Create workflow based on plan
        workflow = Workflow(WorkflowType.HIERARCHICAL)
        
        for i, step in enumerate(plan.steps):
            task = WorkflowTask(
                task_id=f"task_{i}",
                action=step.get("task", "unknown"),
                parameters={"step_data": step},
                dependencies=[f"task_{j}" for j in range(i) if i > 0]
            )
            workflow.add_task(task)
        
        self.current_workflow = workflow
        
        # Execute with error handling
        try:
            start_time = time.time()
            results = workflow.execute()
            execution_time = time.time() - start_time
            
            # Record metrics
            self.evaluator.record_metric("response_time", execution_time)
            self.evaluator.record_metric("success_rate", 1.0)
            
            return results
        
        except Exception as e:
            # Topic 10: Handle errors
            error = AgentError(
                error_type=ErrorType.EXECUTION_ERROR,
                message=str(e),
                context={"plan": plan, "workflow": workflow},
                timestamp=datetime.now(),
                recoverable=True
            )
            
            recovery = self.error_handler.handle_error(error)
            
            if recovery.get("recovered"):
                # Retry with simpler workflow
                simple_workflow = Workflow(WorkflowType.SEQUENTIAL)
                for task in workflow.tasks:
                    simple_workflow.add_task(task)
                return simple_workflow.execute()
            
            return {"error": str(e), "recovery": recovery}
    
    def collaborate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Topic 6: Multi-agent collaboration"""
        print("\nğŸ¤� Delegating to specialist agents...")
        
        # Delegate to coordinator
        results = self.coordinator.delegate_task(task)
        
        # Record collaboration metrics
        self.evaluator.record_metric("collaboration_success", 1.0)
        
        return results
    
    def retrieve_knowledge(self, query: str) -> str:
        """Topic 8: Use RAG for knowledge retrieval"""
        print(f"\nğŸ“š Retrieving knowledge for: {query}")
        response = self.rag_system.generate_with_context(query)
        return response
    
    def reflect(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Topic 7: Evaluate performance and reflect"""
        self.state = AgentState.REFLECTING
        
        # Consolidate memories
        self.memory_manager.consolidate()
        
        # Evaluate performance
        evaluation = self.evaluator.evaluate_performance()
        
        # Store episodic memory
        episode = MemoryItem(
            content={
                "execution_results": execution_results,
                "evaluation": evaluation
            },
            timestamp=datetime.now(),
            importance=0.85,
            memory_type="episodic"
        )
        self.memory_manager.long_term.store_episodic(episode)
        
        return evaluation
    
    def run(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Main agent loop integrating all topics"""
        print("\n" + "="*70)
        print("ğŸ¤– MEAL PLANNING AGENT - COMPLETE SYSTEM")
        print("="*70)
        
        try:
            # 1. PERCEIVE (Topic 1)
            print("\n1ï¸�âƒ£  PERCEPTION PHASE")
            perception = self.perceive(user_input)
            print(f"   âœ“ Perceived intent: {perception.user_intent}")
            
            # 2. REASON & PLAN (Topic 5)
            print("\n2ï¸�âƒ£  PLANNING PHASE")
            plan = self.reason_and_plan(perception)
            print(f"   âœ“ Created {plan.strategy.value} plan with {len(plan.steps)} steps")
            
            # 3. RETRIEVE KNOWLEDGE (Topic 8 - RAG)
            print("\n3ï¸�âƒ£  KNOWLEDGE RETRIEVAL (RAG)")
            knowledge = self.retrieve_knowledge(user_input.get("query", "meal planning tips"))
            print(f"   âœ“ Retrieved relevant knowledge")
            
            # 4. COLLABORATE (Topic 6)
            print("\n4ï¸�âƒ£  MULTI-AGENT COLLABORATION")
            collab_task = {
                "nutrition_check": True,
                "recipe_search": True,
                "ingredients": ["salmon", "quinoa", "broccoli"],
                "criteria": {"cuisine": "mediterranean", "max_time": 30}
            }
            collab_results = self.collaborate(collab_task)
            print(f"   âœ“ Received responses from {len(collab_results)} specialist agents")
            
            # 5. EXECUTE WITH WORKFLOW (Topic 9)
            print("\n5ï¸�âƒ£  WORKFLOW EXECUTION")
            execution_results = self.execute_with_workflow(plan)
            print(f"   âœ“ Workflow completed: {execution_results.get('workflow_type')}")
            
            # 6. REFLECT & EVALUATE (Topic 7)
            print("\n6ï¸�âƒ£  REFLECTION & EVALUATION")
            evaluation = self.reflect(execution_results)
            print(f"   âœ“ Performance evaluation completed")
            
            # Final results
            final_results = {
                "perception": asdict(perception),
                "plan": asdict(plan),
                "knowledge": knowledge[:200] + "..." if len(knowledge) > 200 else knowledge,
                "collaboration": collab_results,
                "execution": execution_results,
                "evaluation": evaluation,
                "agent_state": self.state.value
            }
            
            self.state = AgentState.IDLE
            return final_results
        
        except Exception as e:
            # Topic 10: Error handling
            error = AgentError(
                error_type=ErrorType.EXECUTION_ERROR,
                message=str(e),
                context={"user_input": user_input},
                timestamp=datetime.now()
            )
            recovery = self.error_handler.handle_error(error)
            return {"error": str(e), "recovery": recovery}
    
    def generate_meal_plan(self, preferences: Dict[str, Any], days: int = 7) -> Dict[str, Any]:
        """Generate complete meal plan using all agent capabilities"""
        
        user_input = {
            "intent": "generate_meal_plan",
            "preferences": preferences,
            "days": days,
            "query": "healthy meal planning strategies"
        }
        
        # Update environment
        self.environment.user_preferences = preferences
        self.environment.constraints = {
            "days": days,
            "dietary_restrictions": preferences.get("dietary_restrictions", [])
        }
        
        # Run agent
        results = self.run(user_input)
        
        # Generate actual meal plan using tools
        recipe_tool = self.tool_registry.get_tool("recipe_search")
        nutrition_tool = self.tool_registry.get_tool("nutrition_calculator")
        
        meal_plan = {}
        start_date = datetime.now()
        
        for day in range(days):
            date = start_date + timedelta(days=day)
            date_str = date.strftime("%Y-%m-%d")
            
            # Search recipes
            recipes = recipe_tool.execute(
                cuisine=preferences.get("cuisine"),
                max_time=preferences.get("max_cooking_time", 60)
            )
            
            if recipes:
                selected = random.choice(recipes)
                nutrition = nutrition_tool.execute(
                    ingredients=selected["ingredients"],
                    portions=1
                )
                
                meal_plan[date_str] = {
                    "breakfast": {"name": "Greek Yogurt Parfait", "calories": 350},
                    "lunch": selected,
                    "dinner": {"name": "Grilled Chicken", "calories": 450},
                    "daily_nutrition": nutrition
                }
        
        results["meal_plan"] = meal_plan
        return results
    
    def get_evaluation_report(self) -> str:
        """Get comprehensive evaluation report"""
        return self.evaluator.generate_report()


# ============================================================================
# DEMONSTRATION & USAGE
# ============================================================================

def demonstrate_all_topics():
    """Comprehensive demonstration of all Kaggle course topics"""
    
    print("\n" + "="*70)
    print("KAGGLE AI AGENT INTENSIVE COURSE - COMPLETE DEMONSTRATION")
    print("="*70)
    
    # Initialize agent
    agent = MealPlanningAgent()
    
    # Demo user preferences
    preferences = {
        "dietary_restrictions": ["vegetarian"],
        "cuisine": "mediterranean",
        "max_cooking_time": 45,
        "calorie_target": 2000,
        "allergies": []
    }
    
    print("\nğŸ“‹ USER PREFERENCES:")
    for key, value in preferences.items():
        print(f"   {key}: {value}")
    
    # Generate meal plan demonstrating all topics
    print("\n" + "="*70)
    results = agent.generate_meal_plan(preferences, days=7)
    
    # Display results
    print("\n\nğŸ“Š FINAL RESULTS:")
    print("="*70)
    
    if "meal_plan" in results:
        print("\nğŸ�½ï¸�  GENERATED MEAL PLAN:")
        for date, meals in list(results["meal_plan"].items())[:3]:
            print(f"\n   ğŸ“… {date}")
            for meal_type, meal_data in meals.items():
                if meal_type != "daily_nutrition":
                    print(f"      {meal_type.title()}: {meal_data.get('name', 'N/A')}")
        print(f"\n   ... ({len(results['meal_plan']) - 3} more days)")
    
    # Show evaluation
    print("\n\nğŸ“ˆ PERFORMANCE EVALUATION:")
    print("="*70)
    print(agent.get_evaluation_report())
    
    # Memory statistics
    print("\nğŸ’¾ MEMORY STATISTICS:")
    print("="*70)
    print(f"   Short-term memory items: {len(agent.memory_manager.short_term.memory)}")
    print(f"   Long-term episodic memories: {len(agent.memory_manager.long_term.episodic_memory)}")
    print(f"   Semantic knowledge entries: {len(agent.memory_manager.long_term.semantic_memory)}")
    
    # Error log
    print("\nâš ï¸�  ERROR HANDLING LOG:")
    print("="*70)
    if agent.error_handler.error_log:
        for error in agent.error_handler.error_log:
            print(f"   {error.timestamp}: {error.error_type.value} - {error.message}")
    else:
        print("   âœ“ No errors encountered")
    
    print("\n" + "="*70)
    print("âœ… DEMONSTRATION COMPLETE")
    print("="*70)
    
    print("\n\nğŸ“š TOPICS COVERED:")
    topics = [
        "1. Agent Fundamentals (Perception, Action, Environment)",
        "2. Tool Use & Function Calling",
        "3. Prompt Engineering & Chain of Thought",
        "4. Memory Systems (Short-term, Long-term, Episodic)",
        "5. Planning & Reasoning (ReAct, Hierarchical Planning)",
        "6. Multi-Agent Systems & Collaboration",
        "7. Agent Evaluation & Monitoring",
        "8. Retrieval Augmented Generation (RAG)",
        "9. Agentic Workflows (Sequential, Parallel, Hierarchical)",
        "10. Error Handling & Self-Correction"
    ]
    
    for topic in topics:
        print(f"   âœ“ {topic}")
    
    print("\n" + "="*70 + "\n")
    
    return results


if __name__ == "__main__":
    # Run comprehensive demonstration
    demonstrate_all_topics()

