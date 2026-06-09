# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


class OrchestratorAgent:
    def __init__(self, a2a_client, planner, agents_registry):
        self.a2a_client = a2a_client
        self.planner = planner
        self.agents_registry = agents_registry

    async def handle_user_goal(self, user_goal, session_id):
        # Step 1: Decompose goal into subtasks
        subtasks = await self.planner.plan(user_goal)
        print(f"[Planner] Decomposed goal into {len(subtasks)} subtasks: {subtasks}")

        # Step 2: Assign subtasks to agents (parallel/sequential as needed)
        results = []
        for idx, subtask in enumerate(subtasks, start=1):
            agent = self.agents_registry.get_agent_for_task(subtask)
            print(f"[Orchestrator] Assigning Subtask {idx} → Agent: {agent.name}")
            result = await self.a2a_client.send_task(agent, subtask, session_id)
            print(f"[Agent:{agent.name}] Result: {result}")
            results.append(result)

        # Step 3: Aggregate results and return
        final_output = self.aggregate_results(results)
        print(f"[Final Output] {final_output}")
        return final_output

    def aggregate_results(self, results):
        # Example aggregation: join results into a single string
        return " | ".join(str(r) for r in results)



class PlannerAgent:
    def __init__(self, llm, memory_agent):
        self.llm = llm
        self.memory_agent = memory_agent

    async def plan(self, user_goal):
        # Step 1: Retrieve relevant memories/context
        context = await self.memory_agent.retrieve_context(user_goal)
        print(f"[MemoryAgent] Retrieved context for goal '{user_goal}': {context}")

        # Step 2: Use LLM to decompose goal
        prompt = f"Decompose the following goal into subtasks:\nGoal: {user_goal}\nContext: {context}"
        print(f"[PlannerAgent] Sending prompt to LLM:\n{prompt}")

        subtasks = await self.llm.generate(prompt)
        print(f"[LLM] Raw subtasks output: {subtasks}")

        # Step 3: Parse subtasks
        parsed = parse_subtasks(subtasks)
        print(f"[PlannerAgent] Parsed subtasks: {parsed}")

        return parsed


class ResearchAgent:
    def __init__(self, search_tool, openapi_tools):
        self.search_tool = search_tool
        self.openapi_tools = openapi_tools

    async def handle_task(self, task, session_id):
        # Step 1: Decide which tool to use
        if task.requires_web_search:
            print(f"[ResearchAgent] Using SearchTool for query: '{task.query}' (Session: {session_id})")
            results = await self.search_tool.search(task.query)
            print(f"[SearchTool] Results: {results}")

        elif task.api_endpoint:
            print(f"[ResearchAgent] Calling OpenAPI endpoint: {task.api_endpoint} with params {task.params} (Session: {session_id})")
            results = await self.openapi_tools.call(task.api_endpoint, task.params)
            print(f"[OpenAPI] Results: {results}")

        else:
            print("[ResearchAgent] No valid task type provided.")
            results = None

        # Step 2: Return results
        print(f"[ResearchAgent] Final Results Returned: {results}")
        return results


class AnalysisAgent:
    def __init__(self, code_executor, analytics_tools):
        self.code_executor = code_executor
        self.analytics_tools = analytics_tools

    async def handle_task(self, task, session_id):
        print(f"[AnalysisAgent] Handling task (Session: {session_id}) → Type: {task.type}")

        if task.type == "data_analysis":
            print(f"[AnalysisAgent] Running data analysis on: {task.data}")
            result = await self.analytics_tools.analyze(task.data)
            print(f"[AnalyticsTools] Analysis Result: {result}")

        elif task.type == "code_execution":
            print(f"[AnalysisAgent] Executing code:\n{task.code}")
            result = await self.code_executor.run(task.code)
            print(f"[CodeExecutor] Execution Result: {result}")

        else:
            print("[AnalysisAgent] Unknown task type provided.")
            result = None

        print(f"[AnalysisAgent] Final Result Returned: {result}")
        return result


class ExecutorAgent:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def handle_task(self, task, session_id):
        print(f"[ExecutorAgent] Handling task (Session: {session_id}) → Tool: {task.tool_name}")

        # Step 1: Discover tool
        tool = await self.mcp_client.discover_tool(task.tool_name)
        print(f"[MCP Client] Discovered Tool: {tool}")

        # Step 2: Call tool with arguments
        print(f"[ExecutorAgent] Calling tool '{task.tool_name}' with arguments: {task.arguments}")
        result = await self.mcp_client.call_tool(tool, task.arguments)
        print(f"[MCP Client] Tool Result: {result}")

        # Step 3: Return result
        print(f"[ExecutorAgent] Final Result Returned: {result}")
        return result


class CriticAgent:
    def __init__(self, llm_judge, agent_judge):
        self.llm_judge = llm_judge
        self.agent_judge = agent_judge

    async def review(self, output, criteria):
        print(f"[CriticAgent] Reviewing output against criteria: {criteria}")

        # LLM-as-judge for quick evaluation
        score = await self.llm_judge.evaluate(output, criteria)
        print(f"[LLM Judge] Score: {score} (Threshold: {criteria.threshold})")

        if score < criteria.threshold:
            print("[CriticAgent] Score below threshold → Escalating to Agent Judge")
            feedback = await self.agent_judge.evaluate_process(output)
            print(f"[Agent Judge] Feedback: {feedback}")
            return feedback, False

        print("[CriticAgent] Output Approved")
        return "Approved", True




class MemoryAgent:
    def __init__(self, session_service, memory_bank):
        self.session_service = session_service
        self.memory_bank = memory_bank

    async def retrieve_context(self, query):
        print(f"[MemoryAgent] Retrieving context for query: {query}")

        # Short-term: session state
        session_context = await self.session_service.get_state(query.session_id)
        print(f"[SessionService] Retrieved session context: {session_context}")

        # Long-term: semantic search in memory bank
        long_term_context = await self.memory_bank.search(query)
        print(f"[MemoryBank] Retrieved long-term context: {long_term_context}")

        merged = merge_contexts(session_context, long_term_context)
        print(f"[MemoryAgent] Merged Context: {merged}")
        return merged

    async def update_memory(self, key, value, scope="long_term"):
        print(f"[MemoryAgent] Updating memory → Key: {key}, Value: {value}, Scope: {scope}")

        if scope == "short_term":
            await self.session_service.update_state(key, value)
            print(f"[SessionService] Updated short-term memory with {key}: {value}")
        else:
            await self.memory_bank.add_memory(key, value)
            print(f"[MemoryBank] Added long-term memory with {key}: {value}")


class ObservabilityAgent:
    def __init__(self, telemetry_client):
        self.telemetry_client = telemetry_client

    def log_event(self, agent_id, event_type, details):
        print(f"[ObservabilityAgent] Logging Event → Agent: {agent_id}, Type: {event_type}, Details: {details}")
        self.telemetry_client.log(agent_id, event_type, details)

    def trace_span(self, span_name, attributes):
        print(f"[ObservabilityAgent] Starting Trace Span → {span_name}, Attributes: {attributes}")
        span = self.telemetry_client.start_span(span_name, attributes)
        return span

    def record_metric(self, metric_name, value, tags):
        print(f"[ObservabilityAgent] Recording Metric → {metric_name} = {value}, Tags: {tags}")
        self.telemetry_client.record_metric(metric_name, value, tags)


class HITLAgent:
    def __init__(self, checkpoint_service):
        self.checkpoint_service = checkpoint_service

    async def pause_for_review(self, state, checkpoint_id):
        print(f"[HITLAgent] Saving checkpoint {checkpoint_id} with state: {state}")
        await self.checkpoint_service.save_checkpoint(state, checkpoint_id)

        print(f"[HITLAgent] Waiting for human approval on checkpoint {checkpoint_id}...")
        approval = await wait_for_human_approval(checkpoint_id)

        if approval:
            print(f"[HITLAgent] Approved by human reviewer → Resuming from checkpoint {checkpoint_id}")
            state = await self.checkpoint_service.load_checkpoint(checkpoint_id)
            print(f"[HITLAgent] Loaded state: {state}")
            return state
        else:
            print(f"[HITLAgent] Rejected by human reviewer")
            return "Rejected"


import requests

class A2AClient:
    def __init__(self, base_url, auth_token=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        print(f"[A2AClient] Initialized with base_url: {self.base_url}")

    async def discover_agent(self):
        url = f"{self.base_url}/.well-known/agent.json"
        print(f"[A2AClient] Discovering agent at: {url}")
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        agent_info = response.json()
        print(f"[A2AClient] Discovered Agent Info: {agent_info}")
        return agent_info

    async def send_task(self, task_data):
        print(f"[A2AClient] Preparing to send task: {task_data}")
        agent_info = await self.discover_agent()
        task_endpoint = agent_info["endpoints"]["tasks"]
        url = f"{self.base_url}{task_endpoint}"
        print(f"[A2AClient] Sending task to endpoint: {url}")
        response = requests.post(url, json=task_data, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        print(f"[A2AClient] Task Result: {result}")
        return result


from flask import Flask, request, jsonify
import uuid
from datetime import datetime

app = Flask(__name__)
tasks = {}

@app.route('/.well-known/agent.json', methods=['GET'])
def get_agent_card():
    # Return agent metadata
    agent_info = {
        "name": "Research Agent",
        "version": "1.0.0",
        "capabilities": ["web-search", "api-calls"],
        "endpoints": {"tasks": "/api/tasks/send"}
    }
    print(f"[AgentCard] Returning agent metadata: {agent_info}")
    return jsonify(agent_info)

@app.route('/api/tasks/send', methods=['POST'])
def send_task():
    task_data = request.json
    print(f"[Task Received] {task_data}")

    # Generate or reuse task_id
    task_id = task_data.get('task_id', str(uuid.uuid4()))
    print(f"[TaskID] Assigned: {task_id}")

    # Mark as working
    tasks[task_id] = {"status": "working", "created_at": datetime.now().isoformat()}
    print(f"[TaskStatus] Task {task_id} marked as working")

    # Simulate processing
    result = {"summary": "Task completed successfully."}
    print(f"[Processing] Simulated result: {result}")

    # Mark as completed
    tasks[task_id] = {
        "status": "completed",
        "artifacts": [result],
        "completed_at": datetime.now().isoformat()
    }
    print(f"[TaskStatus] Task {task_id} completed with artifacts: {tasks[task_id]['artifacts']}")

    return jsonify(tasks[task_id])

