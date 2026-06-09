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


import json

from datetime import datetime

from dataclasses import dataclass, field

from typing import List, Dict

print("Import successfull")


# capstone_agents.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import uuid
import json

# ----------------------------
# Memory and Session Management
# ----------------------------
@dataclass
class Memory:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 50

    def add(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "time": datetime.utcnow().isoformat(),
        })
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_summary(self, last_k: int = 10) -> str:
        # Simple concatenation; could be replaced with summarization
        out = ""
        for m in self.messages[-last_k:]:
            out += f"{m['role']}: {m['content']}\n"
        return out

@dataclass
class Session:
    session_id: str
    user_id: str
    memory: Memory = field(default_factory=Memory)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}

    def create_session(self, user_id: str) -> Session:
        sid = str(uuid.uuid4())
        s = Session(session_id=sid, user_id=user_id)
        self.sessions[sid] = s
        return s

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

# ----------------------------
# Tool abstractions (account management)
# ----------------------------
class Tool:
    """Base tool class. Tools expose a run() method and a name."""
    name: str = "base"

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class AccountTool(Tool):
    name = "account_tool"

    def __init__(self):
        # very simple in-memory account store
        self.accounts: Dict[str, Dict[str, Any]] = {}

    def run(self, action: str, **kwargs) -> Dict[str, Any]:
        # Supported actions: create, deposit, withdraw, balance, list
        if action == "create":
            return self._create(kwargs["user_id"], kwargs.get("initial_deposit", 0.0))
        if action == "deposit":
            return self._deposit(kwargs["account_id"], kwargs["amount"])
        if action == "withdraw":
            return self._withdraw(kwargs["account_id"], kwargs["amount"])
        if action == "balance":
            return self._balance(kwargs["account_id"])
        if action == "list":
            return {"accounts": list(self.accounts.values())}
        return {"error": "unsupported_action"}

    def _create(self, user_id: str, initial_deposit: float):
        acc_id = str(uuid.uuid4())
        acc = {
            "account_id": acc_id,
            "user_id": user_id,
            "balance": float(initial_deposit),
            "created_at": datetime.utcnow().isoformat(),
        }
        self.accounts[acc_id] = acc
        return {"status": "ok", "account": acc}

    def _deposit(self, account_id: str, amount: float):
        if account_id not in self.accounts:
            return {"error": "account_not_found"}
        self.accounts[account_id]["balance"] += float(amount)
        return {"status": "ok", "account": self.accounts[account_id]}

    def _withdraw(self, account_id: str, amount: float):
        if account_id not in self.accounts:
            return {"error": "account_not_found"}
        if self.accounts[account_id]["balance"] < amount:
            return {"error": "insufficient_funds"}
        self.accounts[account_id]["balance"] -= float(amount)
        return {"status": "ok", "account": self.accounts[account_id]}

    def _balance(self, account_id: str):
        if account_id not in self.accounts:
            return {"error": "account_not_found"}
        return {"status": "ok", "balance": self.accounts[account_id]["balance"]}

# ----------------------------
# LLM Adapter (mockable)
# ----------------------------
class MockLLM:
    """Simple deterministic 'LLM' for demo. Replace with real LLM integration."""
    def generate(self, prompt: str) -> str:
        # For demo: very naive rule-based output
        prompt_lower = prompt.lower()
        if "create account" in prompt_lower or "open account" in prompt_lower:
            return "CALL_TOOL: account_tool create user_id={user_id} initial_deposit=50"
        if "balance" in prompt_lower:
            return "CALL_TOOL: account_tool balance account_id={account_id}"
        if "deposit" in prompt_lower:
            return "CALL_TOOL: account_tool deposit account_id={account_id} amount=100"
        # If nothing matches, respond conversationally
        return "I can help manage your account. Say 'create account' or ask for 'balance'."

# ----------------------------
# Agent base classes
# ----------------------------
class Agent:
    def __init__(self, name: str):
        self.name = name

    def receive(self, session: Session, message: str) -> str:
        raise NotImplementedError

class ToolAgent(Agent):
    def __init__(self, tool: Tool):
        super().__init__(name=f"tool-{tool.name}")
        self.tool = tool

    def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.tool.run(action=action, **params)

# Assistant agent uses an LLM to generate responses or tool-call directives
class AssistantAgent(Agent):
    def __init__(self, name: str, llm: MockLLM):
        super().__init__(name=name)
        self.llm = llm

    def receive(self, session: Session, message: str) -> str:
        # Add to memory
        session.memory.add("user", message)
        # Compose prompt (very simple)
        prompt = f"Session summary:\n{session.memory.get_summary(6)}\nUser: {message}\nAssistant: "
        response = self.llm.generate(prompt)
        session.memory.add("assistant", response)
        return response

# Coordinator orchestrates assistants and tools based on LLM output
class Coordinator:
    def __init__(self, assistant: AssistantAgent, tools: Dict[str, ToolAgent]):
        self.assistant = assistant
        self.tools = tools
        self.audit_log: List[Dict[str, Any]] = []

    def handle(self, session: Session, user_message: str) -> Dict[str, Any]:
        """Process user message and possibly call tools. Returns a structured reply."""
        # ask assistant (LLM) for next action
        llm_output = self.assistant.receive(session, user_message)
        # Expect a specific textual directive format for tooling:
        # Example: "CALL_TOOL: account_tool create user_id=... initial_deposit=..."
        if llm_output.startswith("CALL_TOOL:"):
            # parse directive
            try:
                payload = llm_output[len("CALL_TOOL:"):].strip()
                # split first token => tool_name, second => action
                parts = payload.split()
                tool_name = parts[0]
                action = parts[1]
                # remaining are key=value
                kvs = {}
                for token in parts[2:]:
                    if "=" in token:
                        k, v = token.split("=", 1)
                        # basic type conversion: try float or int
                        if v.replace('.', '', 1).isdigit():
                            if '.' in v:
                                v_parsed = float(v)
                            else:
                                v_parsed = int(v)
                            kvs[k] = v_parsed
                        else:
                            # allow placeholders like {user_id}
                            kvs[k] = v
                # Replace placeholders from session or metadata
                for k, v in list(kvs.items()):
                    if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                        key = v.strip("{}")
                        # try session metadata, memory or session fields
                        kvs[k] = session.metadata.get(key, session.__dict__.get(key, v))
                # find tool agent
                tool_agent = self.tools.get(tool_name)
                if not tool_agent:
                    return {"status": "error", "error": "tool_not_found", "raw": llm_output}
                # execute tool
                result = tool_agent.execute(action, kvs)
                # record audit
                self.audit_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": session.session_id,
                    "user_message": user_message,
                    "llm_output": llm_output,
                    "tool": tool_name,
                    "action": action,
                    "params": kvs,
                    "result": result,
                })
                # write result into session memory
                session.memory.add("system", f"Tool {tool_name}.{action} -> {json.dumps(result)}")
                # produce user-friendly reply
                reply = {"status": "ok", "result": result, "message": f"Executed {tool_name}.{action}"}
                return reply
            except Exception as e:
                return {"status": "error", "error": "parse_or_execute_failed", "detail": str(e), "raw": llm_output}
        else:
            # Not a tool call — return assistant text
            return {"status": "ok", "result": None, "message": llm_output}

# ----------------------------
# Demo: build system and run a scenario
# ----------------------------
def demo():
    print("=== Capstone Generative AI: Multi-Agent Demo ===")
    # Build managers
    session_manager = SessionManager()
    account_tool = AccountTool()
    tools = {"account_tool": ToolAgent(account_tool)}
    # LLM: mock for demo. Replace with real LLM integration as needed.
    llm = MockLLM()
    assistant = AssistantAgent(name="assistant-ai", llm=llm)
    coordinator = Coordinator(assistant, tools)

    # Create a user session
    session = session_manager.create_session(user_id="user_ankit")
    print("Created session:", session.session_id)

    # 1) User asks to create an account
    user_msg1 = "Please create account for me with initial deposit 50"
    print("\nUser:", user_msg1)
    # MockLLM is naive: it returns a directive with placeholders. We substitute user_id into metadata
    session.metadata["user_id"] = session.user_id
    # Note: our MockLLM returns "CALL_TOOL: account_tool create user_id={user_id} initial_deposit=50"
    out1 = coordinator.handle(session, user_msg1)
    print("Coordinator output:", out1)

    # If tool created an account, capture account id for next steps
    if out1.get("result") and "account" in out1["result"]:
        acc = out1["result"]["account"]
        acc_id = acc["account_id"]
        print("New account id:", acc_id)
        # Store account id in session metadata so subsequent LLM outputs can reference {account_id}
        session.metadata["account_id"] = acc_id
    else:
        # maybe result nested differently depending on tool return
        if isinstance(out1.get("result"), dict) and out1["result"].get("account"):
            acc = out1["result"]["account"]
            acc_id = acc["account_id"]
            session.metadata["account_id"] = acc_id

    # 2) User asks balance (assistant will call tool)
    user_msg2 = "What's my balance?"
    print("\nUser:", user_msg2)
    out2 = coordinator.handle(session, user_msg2)
    print("Coordinator output:", out2)

    # 3) User asks natural question
    user_msg3 = "Can you explain what you did?"
    print("\nUser:", user_msg3)
    out3 = coordinator.handle(session, user_msg3)
    print("Coordinator output:", out3)

    # Print audit log
    print("\nAudit log (most recent entries):")
    for entry in coordinator.audit_log[-5:]:
        print(json.dumps(entry, indent=2))

    # Print session memory summary
    print("\nSession memory summary:")
    print(session.memory.get_summary(20))

if __name__ == "__main__":
    demo()



{"call_tool": {"name": "account_tool", "action": "create", "params": {"user_id": "...", "initial_deposit": 50}}}


