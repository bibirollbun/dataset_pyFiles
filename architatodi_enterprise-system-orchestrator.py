import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Setup and authentication complete.")
except Exception as e:
    print("Authentication Error")


!pip install google-generativeai

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
print("âœ… Google AI configured.")



class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {'created': datetime.now(), 'state': {}}
        return session_id
    
    def get_state(self, session_id: str) -> Dict:
        return self.sessions.get(session_id, {}).get('state', {})
    
    def update_state(self, session_id: str, state: Dict):
        if session_id in self.sessions:
            self.sessions[session_id]['state'].update(state)

print("âœ… SessionManager ready.")



class MemoryBank:
    def __init__(self):
        self.memory = []
    
    def store(self, key: str, data: Any):
        entry = {'key': key, 'data': data, 'timestamp': datetime.now()}
        self.memory.append(entry)
    
    def retrieve(self, query: str, limit: int = 5) -> List[Dict]:
        results = [m for m in self.memory if query.lower() in str(m['data']).lower()]
        return results[-limit:]
    
    def compact_context(self, max_items: int = 100):
        if len(self.memory) > max_items:
            self.memory = self.memory[-max_items:]

print("âœ… MemoryBank ready.")



class GoogleAITool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def execute(self, **kwargs) -> Any:
        raise NotImplementedError

class GeminiSearchTool(GoogleAITool):
    def __init__(self):
        super().__init__("gemini_search", "Search using Gemini AI")
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    async def execute(self, query: str) -> Dict:
        response = self.model.generate_content(f"Search and analyze: {query}")
        return {"results": response.text, "source": "gemini-2.0-flash"}

class VertexAITool(GoogleAITool):
    def __init__(self):
        super().__init__("vertex_analysis", "Analyze data using Vertex AI")
    
    async def execute(self, data: str) -> Dict:
        return {"analysis": f"Vertex AI processed: {data[:100]}...", "confidence": 0.95}

print("âœ… Tools ready.")



class BaseAgent:
    def __init__(self, agent_id: str, model_name: str = "gemini-2.0-flash"):
        self.agent_id = agent_id
        self.model = genai.GenerativeModel(model_name)
        self.tools = {}
        self.memory = []
        self.state = "idle"
    
    def add_tool(self, tool: GoogleAITool):
        self.tools[tool.name] = tool
    
    async def process_task(self, task: Dict) -> Dict:
        self.state = "processing"
        try:
            result = await self._execute_task(task)
            self.memory.append({'task': task, 'result': result, 'timestamp': datetime.now()})
            self.state = "idle"
            return result
        except Exception as e:
            self.state = "error"
            return {'error': str(e), 'agent': self.agent_id}
    
    async def _execute_task(self, task: Dict) -> Dict:
        prompt = task.get('prompt', '')
        response = self.model.generate_content(prompt)
        return {'response': response.text, 'agent': self.agent_id}

print("âœ… BaseAgent ready.")



class DataAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__("data_analyst", "gemini-2.0-flash")
        self.add_tool(VertexAITool())

class CustomerSupportAgent(BaseAgent):
    def __init__(self):
        super().__init__("customer_support", "gemini-2.0-flash")
        self.add_tool(GeminiSearchTool())

class WorkflowAgent(BaseAgent):
    def __init__(self):
        super().__init__("workflow_optimizer", "gemini-2.0-flash")

print("âœ… Specialized Agents ready.")



class AgentOrchestrator:
    def __init__(self):
        self.agents = {}
        self.session_manager = SessionManager()
        self.memory_bank = MemoryBank()
        self.metrics = {'tasks_completed': 0, 'errors': 0}
    
    def register_agent(self, agent: BaseAgent):
        self.agents[agent.agent_id] = agent
    
    async def parallel_execution(self, tasks: List[Dict]) -> List[Dict]:
        agent_tasks = []
        agent_list = list(self.agents.values())
        
        for i, task in enumerate(tasks):
            agent = agent_list[i % len(agent_list)]
            agent_tasks.append(agent.process_task(task))
        
        results = await asyncio.gather(*agent_tasks)
        self.metrics['tasks_completed'] += len(results)
        return results
    
    async def sequential_execution(self, tasks: List[Dict], agent_id: str) -> List[Dict]:
        if agent_id not in self.agents:
            return [{'error': f'Agent {agent_id} not found'}]
        
        agent = self.agents[agent_id]
        results = []
        
        for task in tasks:
            result = await agent.process_task(task)
            results.append(result)
        
        return results
    
    async def loop_execution(self, task: Dict, iterations: int = 3) -> List[Dict]:
        results = []
        current_task = task
        
        for i in range(iterations):
            result = await self.parallel_execution([current_task])
            results.extend(result)
            
            if result and 'response' in result[0]:
                current_task['prompt'] = f"Improve on this: {result[0]['response'][:200]}"
        
        return results

print("âœ… AgentOrchestrator ready.")



class A2AProtocol:
    def __init__(self):
        self.message_queue = []
    
    async def send_message(self, from_agent: str, to_agent: str, message: str) -> str:
        msg_id = str(uuid.uuid4())
        msg = {
            'id': msg_id,
            'from': from_agent,
            'to': to_agent,
            'message': message,
            'timestamp': datetime.now()
        }
        self.message_queue.append(msg)
        return msg_id
    
    def get_messages(self, agent_id: str) -> List[Dict]:
        return [msg for msg in self.message_queue if msg['to'] == agent_id]

print("âœ… A2A Protocol ready.")



import sys
from io import StringIO
import json
import asyncio

# Enterprise System Demo
orchestrator = AgentOrchestrator()
orchestrator.register_agent(DataAnalystAgent())
orchestrator.register_agent(CustomerSupportAgent())
orchestrator.register_agent(WorkflowAgent())

a2a = A2AProtocol()
memory = MemoryBank()

async def demo():
    print("ğŸš€ Enterprise AI Agent System Demo")
    
    scenarios = [
        {"prompt": "Analyze Q4 sales data showing 15% decline"},
        {"prompt": "Customer complaint about delayed order #12345"},
        {"prompt": "Optimize inventory management process"}
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\nğŸ“‹ Scenario {i+1}: {scenario['prompt']}")
        
        results = await orchestrator.parallel_execution([scenario])
        
        for result in results:
            if "response" in result:
                print(f"\nâœ… {result['agent']} FULL RESPONSE:")
                print(f"{result['response']}")
                print("-" * 80)
            elif "error" in result:
                print(f"  â�Œ Error: {result['error']}")
        
        memory.store(f"scenario_{i+1}", results)
    
    print("\nğŸ“Š Metrics:")
    print(f"Tasks: {orchestrator.metrics['tasks_completed']}")
    print(f"Errors: {orchestrator.metrics['errors']}")
    print(f"Memory: {len(memory.memory)}")


old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()


await demo()

sys.stdout = old_stdout

output_path = "/kaggle/working/output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(mystdout.getvalue())

print("âœ… Output saved to:", output_path)

