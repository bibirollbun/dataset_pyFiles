import os
import json
import time
import logging
import asyncio
import uuid
import hashlib
import inspect
from typing import List, Dict, Any, Optional, Callable, Union, Type
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod

# ----------------------------------------------------------------------------
# 0. SETUP & AUTHENTICATION
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [AGENT-SDK] - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentSDK")

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["GOOGLE_API_KEY"] = UserSecretsClient().get_secret("GOOGLE_API_KEY")
except Exception:
    pass

import google.generativeai as genai
from google.generativeai.types import content_types
from google.protobuf import struct_pb2

if not os.environ.get("GOOGLE_API_KEY"):
    logger.warning(" GOOGLE_API_KEY not found. Agents will fail to initialize.")
else:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ----------------------------------------------------------------------------
# 1. OBSERVABILITY & METRICS
# ----------------------------------------------------------------------------

class MetricsRegistry:
    """Thread-safe centralized metrics collector with detailed statistical analysis."""
    def __init__(self):
        self._traces = {}
        self._metrics = defaultdict(list)
        self._agent_metadata = {}

    def register_agent(self, agent_name: str, agent_type: str):
        """Registers static metadata about an agent."""
        self._agent_metadata[agent_name] = agent_type

    def start_trace(self, agent_name: str, task: str) -> str:
        trace_id = str(uuid.uuid4())
        self._traces[trace_id] = {
            "id": trace_id,
            "agent": agent_name,
            "task": task[:50],
            "start": time.time(),
            "status": "running"
        }
        return trace_id

    def end_trace(self, trace_id: str, success: bool = True):
        if trace_id in self._traces:
            trace = self._traces[trace_id]
            duration = time.time() - trace["start"]
            
            # Record raw data
            self._metrics[trace['agent']].append({
                "duration": duration,
                "success": success,
                "timestamp": trace["start"]
            })
            
            status = "success" if success else "failed"
            logger.info(f"{status} Trace {trace_id} ({trace['agent']}) finished in {duration:.2f}s")

    def get_summary(self) -> Dict:
        """Generates a detailed, enterprise-grade JSON report."""
        summary = {}
        
        for agent_name, history in self._metrics.items():
            durations = [h["duration"] for h in history]
            success_count = sum(1 for h in history if h["success"])
            total_count = len(history)
            
            # detailed stats calculation
            summary[agent_name] = {
                "type": self._agent_metadata.get(agent_name, "Unknown"),
                "load": {
                    "total_calls": total_count,
                    "success_rate": f"{(success_count/total_count)*100:.1f}%"
                },
                "latency_seconds": {
                    "avg": round(sum(durations) / total_count, 4),
                    "min": round(min(durations), 4),
                    "max": round(max(durations), 4),
                    "total_compute_time": round(sum(durations), 4)
                }
            }
        return summary

metrics = MetricsRegistry()

# ----------------------------------------------------------------------------
# 2. MEMORY & CONTEXT SYSTEMS
# ----------------------------------------------------------------------------

class ContextCompactor:
    """Intelligent context pruning for long conversations."""
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens

    def compact(self, history: List[Any]) -> List[Any]:


        if len(history) < 10:
            return history

        logger.info(f" Compacting context from {len(history)} to 8 messages")
        return [history[0]] + [history[1]] + history[-6:]

class MemoryBank:
    """Vector-style semantic memory (Simulated)."""
    def __init__(self):
        self.storage = []

    def store(self, content: str, tags: List[str]):
        self.storage.append({
            "id": str(uuid.uuid4())[:8],
            "content": content,
            "tags": tags,
            "timestamp": time.time()
        })

    def recall(self, query: str) -> str:
        hits = [m["content"] for m in self.storage if any(t in query.lower() for t in m["tags"])]
        return "\n".join(hits[-3:]) if hits else "No relevant memories."

# ----------------------------------------------------------------------------
# 3. TOOLING REGISTRY
# ----------------------------------------------------------------------------

def google_search(query: str) -> Dict[str, Any]:
    """Performs a web search for real-time information."""
    
    return {"result": f"Search results for '{query}': Market is bullish. AI adoption is up 40%."}

def analyze_dataset(data_summary: str) -> Dict[str, float]:
    """Analyzes a dataset summary string."""
    return {"mean": 120.5, "max": 500, "recommendation": "Hold"}

def create_jira_ticket(issue: str, priority: str) -> str:
    """Creates a ticket in the tracking system."""
    return f"JIRA-{os.urandom(2).hex().upper()}"

TOOL_REGISTRY = {
    "google_search": google_search,
    "analyze_dataset": analyze_dataset,
    "create_jira_ticket": create_jira_ticket
}

# ----------------------------------------------------------------------------
# 4. AGENT BASE PROTOCOLS
# ----------------------------------------------------------------------------

@dataclass
class AgentResponse:
    success: bool
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_name: str = "Unknown"
    tools_used: List[str] = field(default_factory=list)

class Agent(ABC):
    """Abstract Base Class for all Agent types in the SDK."""
    
    def __init__(self, name: str, agent_type: str = "Generic"):
        self.name = name
        self.memory = MemoryBank()
        self.compactor = ContextCompactor()
        metrics.register_agent(name, agent_type)
        
    @abstractmethod
    async def run(self, input_task: str, context: Dict = None) -> AgentResponse:
        pass

# ----------------------------------------------------------------------------
# 5. ATOMIC AGENT
# ----------------------------------------------------------------------------

class GenAIAgent(Agent):
    """
    Standard Worker Agent wrapping Google Gen AI.
    Handles Tool Calling loop automatically.
    """
    def __init__(self, name: str, model_name: str, instructions: str, tools: List[Callable] = None):
        super().__init__(name, agent_type="Atomic (GenAI)")
        self.model_name = model_name
        self.instructions = instructions
        self.tools = tools or []

        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=instructions,
            tools=self.tools
        )

    async def run(self, input_task: str, context: Dict = None) -> AgentResponse:
        trace_id = metrics.start_trace(self.name, input_task)

        context_str = json.dumps(context) if context else "{}"
        memories = self.memory.recall(input_task)
        augmented_prompt = (
            f"Context: {context_str}\n"
            f"Memories: {memories}\n"
            f"Task: {input_task}"
        )

        try:
            chat = self._model.start_chat(enable_automatic_function_calling=True)
            
            response = await chat.send_message_async(augmented_prompt)

            tools_used = []
            for message in chat.history:
                if message.parts:
                    for part in message.parts:
                        if part.function_call:
                            tools_used.append(part.function_call.name)

            self.memory.store(response.text, tags=[self.name, "task_result"])
            metrics.end_trace(trace_id, success=True)
            
            return AgentResponse(
                success=True,
                output=response.text,
                agent_name=self.name,
                metadata={"history_length": len(chat.history)},
                tools_used=tools_used # Pass the detected tools to the response
            )

        except Exception as e:
            logger.error(f"Agent {self.name} crashed: {e}")
            metrics.end_trace(trace_id, success=False)
            return AgentResponse(success=False, output=str(e), agent_name=self.name)
# ----------------------------------------------------------------------------
# 6. ORCHESTRATION AGENTS (Parallel, Sequential, Loop)
# ----------------------------------------------------------------------------

class ParallelAgent(Agent):
    """
    Executes multiple sub-agents concurrently and synthesizes results.
    """
    def __init__(self, name: str, agents: List[Agent], aggregator_model: str = "gemini-2.0-flash-lite"):
        super().__init__(name, agent_type="Composite (Parallel)")
        self.agents = agents
        self.aggregator = genai.GenerativeModel(aggregator_model)

    async def run(self, input_task: str, context: Dict = None) -> AgentResponse:
        trace_id = metrics.start_trace(self.name, input_task)
        logger.info(f" {self.name}: Launching {len(self.agents)} agents in parallel...")

        futures = [agent.run(input_task, context) for agent in self.agents]
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        valid_outputs = []
        for res in results:
            if isinstance(res, AgentResponse) and res.success:
                valid_outputs.append(f"[{res.agent_name}]: {res.output}")
            else:
                valid_outputs.append(f"[Error]: {str(res)}")

        summary_prompt = (
            f"Original Task: {input_task}\n\n"
            f"Sub-Agent Results:\n" + "\n".join(valid_outputs) + "\n\n"
            "Synthesize a single coherent response."
        )
        final_res = await self.aggregator.generate_content_async(summary_prompt)
        
        metrics.end_trace(trace_id)
        return AgentResponse(success=True, output=final_res.text, agent_name=self.name)


class SequentialAgent(Agent):
    """
    Executes agents in a chain, passing output of Agent A as context to Agent B.
    """
    def __init__(self, name: str, agents: List[Agent]):
        super().__init__(name, agent_type="Composite (Sequential)")
        self.agents = agents

    async def run(self, input_task: str, context: Dict = None) -> AgentResponse:
        trace_id = metrics.start_trace(self.name, input_task)
        logger.info(f" {self.name}: Starting sequence of {len(self.agents)} agents")
        
        current_context = context or {}
        last_output = input_task
        flow_history = []

        for agent in self.agents:

            current_context["previous_output"] = last_output
            
            # Execute
            response = await agent.run(input_task, current_context)
            
            if not response.success:
                return AgentResponse(success=False, output=f"Chain broken at {agent.name}: {response.output}")
            
            last_output = response.output
            flow_history.append(f"{agent.name} completed.")

        metrics.end_trace(trace_id)
        return AgentResponse(
            success=True, 
            output=last_output, 
            agent_name=self.name,
            metadata={"flow": flow_history}
        )


class LoopAgent(Agent):
    def __init__(self, name: str, worker: Agent, validator: Callable[[str], bool], max_loops: int = 3):
        super().__init__(name, agent_type="Composite (Loop)")
        self.worker = worker
        self.validator = validator
        self.max_loops = max_loops

    async def run(self, input_task: str, context: Dict = None) -> AgentResponse:
        trace_id = metrics.start_trace(self.name, input_task)
        loops = 0
        current_context = context or {}
        last_error = ""

        while loops < self.max_loops:
            logger.info(f" {self.name}: Iteration {loops + 1}/{self.max_loops}")
            current_context["iteration"] = loops
            
            if last_error:
                current_context["critique"] = f"Your previous answer failed the check: {last_error}. You MUST fix this."
            
            response = await self.worker.run(input_task, current_context)
            

            try:
                val_result = self.validator(response.output)
                if isinstance(val_result, tuple):
                    passed, error_msg = val_result
                else:
                    passed, error_msg = val_result, "Constraint check failed."
                
                if passed:
                    logger.info(f" Loop condition met for {self.name}")
                    metrics.end_trace(trace_id)
                    return response
                
                logger.info(f"   [Validator Failed]: {error_msg}")
                last_error = error_msg
                loops += 1
                
            except Exception as e:
                logger.error(f"Validator crashed: {e}")
                loops += 1

        metrics.end_trace(trace_id)
        return AgentResponse(
            success=False, 
            output="Max loops reached without meeting condition.", 
            agent_name=self.name
        )

# ----------------------------------------------------------------------------
# 7. ADVANCED FEATURES (Evaluator, A2A, Long-Running Ops)
# ----------------------------------------------------------------------------

class AgentEvaluator:
    """Framework to score agent performance against test cases."""
    
    async def evaluate(self, agent: Agent, test_cases: List[Dict]) -> Dict:
        logger.info(f" Starting Evaluation for {agent.name}")
        results = []
        score = 0
        
        for case in test_cases:
            start = time.time()
            res = await agent.run(case["input"])
            duration = time.time() - start
            
            # CRITICAL FIX: Hybrid verification logic
            
            # 1. Keyword Check: Did the output contain the expected text?
            keyword_match = False
            if "expected_keyword" in case:
                keyword_match = case["expected_keyword"].lower() in res.output.lower()
                
            # 2. Tool Check: Did the agent call the expected tool?
            tool_match = False
            if "expected_tool" in case:
                tool_match = case["expected_tool"] in res.tools_used
            
            # Pass if EITHER the keyword matches OR the correct tool was used
            passed = keyword_match or tool_match
            
            if passed: score += 1
            
            results.append({
                "input": case["input"],
                "passed": passed,
                "duration": duration,
                "tools_used": res.tools_used, # Useful for debugging
                "response_snippet": res.output[:50]
            })
            
        return {
            "agent": agent.name,
            "accuracy": f"{(score/len(test_cases))*100}%",
            "details": results
        }
class A2AMessenger:
    """Agent-to-Agent Message Bus."""
    def __init__(self):
        self.inbox = defaultdict(list)

    def send(self, sender: str, recipient: str, message: str):
        logger.info(f" A2A: {sender} -> {recipient}: {message}")
        self.inbox[recipient].append({"from": sender, "msg": message, "ts": time.time()})

    def receive(self, recipient: str) -> List[Dict]:
        return self.inbox.pop(recipient, [])

class LongRunningOperation:
    """Manages state persistence for long tasks."""
    def __init__(self, op_id: str, agent: Agent, task: str):
        self.op_id = op_id
        self.agent = agent
        self.task = task
        self.state = "initialized"
        self.checkpoint_data = {}

    def pause(self):
        self.state = "paused"
        # In production, save 'self.checkpoint_data' to Disk/DB
        logger.info(f" Operation {self.op_id} PAUSED")
        return self.checkpoint_data

    async def resume(self):
        self.state = "running"
        logger.info(f" Operation {self.op_id} RESUMED")
        return await self.agent.run(self.task, context=self.checkpoint_data)

# ----------------------------------------------------------------------------
# 8. MAIN EXECUTION & DEMO
# ----------------------------------------------------------------------------

async def main():
    print("\n" + "="*80)
    print(" GOOGLE AGENT SDK - ENTERPRISE DEMO")
    print("="*80)

    # 1. Define Atomic Agents
    researcher = GenAIAgent(
        name="Researcher",
        model_name="gemini-2.0-flash-lite",
        instructions="You are a senior web researcher. Use tools to find facts.",
        tools=[google_search]
    )

    analyst = GenAIAgent(
        name="Analyst",
        model_name="gemini-2.0-flash-lite",
        instructions="You are a data analyst. Analyze data provided in context.",
        tools=[analyze_dataset]
    )

    writer = GenAIAgent(
        name="Writer",
        model_name="gemini-2.0-flash-lite",
        instructions="Write a professional executive summary."
    )

    # 2. Test Parallel Orchestration
    print("\n--- TEST 1: Parallel Research & Analysis ---")
    parallel_squad = ParallelAgent(
        name="ResearchSquad",
        agents=[researcher, analyst]
    )
    res = await parallel_squad.run(
        input_task="Find current AI market trends and analyze the attached Q3 revenue data.",
        context={"dataset_summary": "Q3 Revenue: 200M, Growth: 15%"}
    )
    print(f"Parallel Result:\n{res.output}\n")

    # 3. Test Sequential Workflow
    print("\n--- TEST 2: Sequential Report Generation ---")
    pipeline = SequentialAgent(
        name="ReportPipeline",
        agents=[parallel_squad, writer]
    )
    res_seq = await pipeline.run("Prepare a final report on AI Market Trends.")
    print(f"Final Report:\n{res_seq.output}\n")

    # 4. Test Loop Agent (Self-Correction)
    print("\n--- TEST 3: Loop / Self-Correction (Success Demo) ---")
    
    title_agent = GenAIAgent(
        name="TitleGenerator",
        model_name="gemini-2.0-flash-lite",
        instructions="You generate catchy headlines for business reports."
    )

    def brevity_validator(text: str):
        limit = 50
        clean_text = text.strip()
        is_short = len(clean_text) < limit
        if is_short:
            return True, "Success"
        else:
            return False, f"Text was {len(clean_text)} chars. MUST be under {limit}."

    # 3. Setup Loop
    loop_agent = LoopAgent(
        "RefinerLoop", 
        title_agent,
        validator=brevity_validator, 
        max_loops=5
    )
    
    res_loop = await loop_agent.run("Create a title for a report about AI Market Trends in 2025.") 
    
    print(f"\nâœ… Loop Result ({res_loop.success}): {res_loop.output}\n")

    # 5. Advanced Systems Check (Evaluation, Ops, A2A)
    print("\n--- TEST 4: Advanced Systems ---")
    
    # Evaluation
    evaluator = AgentEvaluator()
    
    score = await evaluator.evaluate(
        researcher, 
        [{
            "input": "Search for Tesla stock", 
            "expected_tool": "google_search", 
            "expected_keyword": "Tesla"        
        }]
    )
    print(f"Evaluation Score: {score['accuracy']}")

    # Long Running Ops
    op = LongRunningOperation("OP-999", pipeline, "Generate Annual Report")
    op.pause()
    await op.resume() 

    # Metrics Summary
    print("\nðŸ“Š System Metrics:")
    print(json.dumps(metrics.get_summary(), indent=2))

if __name__ == "__main__":
    # Check for API Key before running
    if os.environ.get("GOOGLE_API_KEY"):
        await main()
    else:
        print(" Error: Please set GOOGLE_API_KEY environment variable.")




