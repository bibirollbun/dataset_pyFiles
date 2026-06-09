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


# tools.py
import json
from typing import Optional, Dict, Any
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

class OrderLookupTool:
    """
    Custom tool to lookup mock orders by order_id or customer phone.
    In production, replace with DB queries or API calls.
    """
    def __init__(self, orders_file: str = os.path.join(DATA_DIR, "mock_orders.json")):
        with open(orders_file, "r", encoding="utf-8") as f:
            self.orders = json.load(f)

    def find_order(self, order_id: Optional[str] = None, phone: Optional[str] = None) -> Optional[Dict[str, Any]]:
        for o in self.orders:
            if order_id and o.get("order_id") == order_id:
                return o
            if phone and o.get("phone") == phone:
                return o
        return None

class FAQTool:
    """
    Simple FAQ database lookup tool. Returns canned responses and metadata.
    """
    def __init__(self):
        self.faqs = {
            "delivery_time": {
                "q": "How long does delivery take?",
                "a": "Deliveries typically take 2-4 business days for local orders; express shipping available on request."
            },
            "ingredients": {
                "q": "Are your masalas natural?",
                "a": "Yes — our masalas are made with natural spices, no artificial preservatives."
            },
            "return_policy": {
                "q": "What is your return policy?",
                "a": "Contact us at sprhomemademasalas@gmail.com within 7 days with photos for a returns evaluation."
            }
        }

    def lookup(self, intent_key: str):
        return self.faqs.get(intent_key)



# memory.py
from typing import Dict, Any, Optional
import time
import threading

class InMemorySessionService:
    """
    Short-lived session store: holds a session per customer for conversation continuity.
    Not persistent; suitable for fast demos and HF Spaces with ephemeral state.
    """
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_or_get(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {"created_at": time.time(), "messages": []}
            return self._sessions[session_id]

    def append_message(self, session_id: str, message: Dict[str, Any]):
        session = self.create_or_get(session_id)
        session["messages"].append(message)

    def get_messages(self, session_id: str):
        return self._sessions.get(session_id, {}).get("messages", [])

    def clear_session(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

class MemoryBank:
    """
    Long-term memory to store recurring themes, trending complaints, and high-value customers.
    Simple in-memory aggregator — replace with Redis or DB for production.
    """
    def __init__(self):
        self._topics: Dict[str, int] = {}
        self._lock = threading.Lock()

    def add_topic(self, topic: str):
        with self._lock:
            self._topics[topic] = self._topics.get(topic, 0) + 1

    def top_topics(self, n: int = 10):
        return sorted(self._topics.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def get_count(self, topic: str) -> int:
        return self._topics.get(topic, 0)



# agents.py
import asyncio
from typing import Dict, Any, Optional, List
import logging
from tools import OrderLookupTool, FAQTool
from memory import InMemorySessionService, MemoryBank

# Simple LLM wrapper placeholder for answer generation
class LLMWrapper:
    """
    Replace with actual LLM call. Here it's a deterministic mock generator.
    """
    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        # Mock behavior: mirror prompt with a canned suffix
        await asyncio.sleep(0.05)  # simulate latency
        return f"[LLM Reply based on prompt] {prompt[:200]}"

# A2A message structure
class AgentMessage:
    def __init__(self, sender: str, payload: Dict[str, Any]):
        self.sender = sender
        self.payload = payload

# Planner
class PlannerAgent:
    def __init__(self, session_service: InMemorySessionService, memory_bank: MemoryBank):
        self.session_service = session_service
        self.memory_bank = memory_bank
        self.logger = logging.getLogger("PlannerAgent")

    async def plan(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        Basic routing plan: returns dictionary with route and metadata.
        A real planner could use an LLM to create the plan.
        """
        self.logger.info("Planning for session %s", session_id)
        # Very simple heuristics:
        text = user_message.lower()
        if "order" in text or "track" in text or "delivered" in text:
            route = "order_query"
        elif "price" in text or "discount" in text or "offer" in text:
            route = "pricing_query"
        elif "ingredients" in text or "natural" in text:
            route = "product_query"
        elif "complaint" in text or "bad" in text or "too spicy" in text:
            route = "complaint"
        else:
            route = "general_query"

        plan = {"route": route, "original_text": user_message}
        self.logger.debug("Plan created: %s", plan)
        return plan

# Classification agent
class QueryClassifierAgent:
    def __init__(self):
        self.logger = logging.getLogger("QueryClassifierAgent")

    async def classify(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        # For demonstration, just return plan with a few extracted fields
        text = plan["original_text"]
        self.logger.info("Classifying text: %s", text)
        # naive extraction
        extracted = {}
        tokens = text.split()
        for tok in tokens:
            if tok.startswith("ORD"):
                extracted["order_id"] = tok
            if tok.isdigit() and len(tok) >= 10:
                extracted["phone"] = tok
        # minimal intent mapping
        intent = plan["route"]
        return {"intent": intent, "extracted": extracted, "text": text}

# Answer generation agent
class AnswerGeneratorAgent:
    def __init__(self, llm: LLMWrapper, order_tool: OrderLookupTool, faq_tool: FAQTool,
                 session_service: InMemorySessionService, memory_bank: MemoryBank):
        self.llm = llm
        self.order_tool = order_tool
        self.faq_tool = faq_tool
        self.session_service = session_service
        self.memory_bank = memory_bank
        self.logger = logging.getLogger("AnswerGeneratorAgent")

    async def answer(self, message: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Generate an answer by using tools and LLM.
        Demonstrates A2A protocol via message dict input and output.
        """
        intent = message["intent"]
        extracted = message.get("extracted", {})
        text = message.get("text", "")

        # Tool usage
        tool_info = {}
        if intent == "order_query":
            order = None
            if "order_id" in extracted:
                order = self.order_tool.find_order(order_id=extracted["order_id"])
            elif "phone" in extracted:
                order = self.order_tool.find_order(phone=extracted["phone"])
            tool_info["order"] = order
        elif intent == "product_query":
            # try to match a FAQ
            faq = None
            if "ingredient" in text or "natural" in text:
                faq = self.faq_tool.lookup("ingredients")
            tool_info["faq"] = faq

        # Update memory with intent/topic
        # small topic normalization
        topic = intent
        self.memory_bank.add_topic(topic)

        # Build prompt (simplified)
        prompt_parts = [
            f"User: {text}",
            f"Intent: {intent}",
            f"Tools: {tool_info}",
            "Respond helpfully with steps or actions."
        ]
        prompt = "\n".join(prompt_parts)
        llm_resp = await self.llm.generate(prompt)

        # record into session
        self.session_service.append_message(session_id, {"user": text, "agent": llm_resp, "timestamp": __import__("time").time()})

        return {"response": llm_resp, "tools": tool_info}

# Analytics agent
class AnalyticsAgent:
    def __init__(self, memory_bank: MemoryBank):
        self.memory_bank = memory_bank
        self.logger = logging.getLogger("AnalyticsAgent")

    async def record(self, message: Dict[str, Any]):
        """
        Update analytics / metrics. For demo, we just log top topics occasionally.
        """
        # In production this could push metrics to a monitoring system.
        top = self.memory_bank.top_topics(5)
        self.logger.info("Top topics (sample): %s", top)
        return {"top": top}

# Orchestrator (Planner -> parallel workers -> evaluator)
class Orchestrator:
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.memory_bank = MemoryBank()
        self.planner = PlannerAgent(self.session_service, self.memory_bank)
        self.classifier = QueryClassifierAgent()
        self.llm = LLMWrapper()
        self.order_tool = OrderLookupTool()
        self.faq_tool = FAQTool()
        self.answerer = AnswerGeneratorAgent(self.llm, self.order_tool, self.faq_tool,
                                             self.session_service, self.memory_bank)
        self.analytics = AnalyticsAgent(self.memory_bank)

    async def handle_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        # 1. Planner
        plan = await self.planner.plan(session_id, user_message)
        # 2. classifier
        classification = await self.classifier.classify(plan)
        # 3. Parallel workers (answer + analytics)
        answer_task = asyncio.create_task(self.answerer.answer(classification, session_id))
        analytics_task = asyncio.create_task(self.analytics.record(classification))
        # Wait for both
        answer_res, analytics_res = await asyncio.gather(answer_task, analytics_task)
        # 4. Compose reply (A2A messaging in simplified form)
        return {
            "answer": answer_res["response"],
            "tools_used": answer_res["tools"],
            "analytics": analytics_res["top"]
        }



# evaluator.py
import logging
from typing import List, Dict

class SimpleEvaluator:
    """
    Simple evaluator that compares predicted answers to expected (if available).
    Computes basic metrics and logs them.
    """
    def __init__(self):
        self.logger = logging.getLogger("SimpleEvaluator")
        self.total = 0
        self.correct = 0

    def evaluate(self, predicted: str, expected: str):
        # naive exact-match or substring heuristic
        self.total += 1
        predicted_norm = predicted.strip().lower()
        expected_norm = expected.strip().lower()
        is_correct = expected_norm in predicted_norm or predicted_norm in expected_norm
        if is_correct:
            self.correct += 1
        self.logger.info("Eval: predicted contains expected? %s", is_correct)
        return {"is_correct": is_correct, "total": self.total, "correct": self.correct}

    def summary(self):
        acc = (self.correct / self.total) if self.total else 0.0
        return {"total": self.total, "correct": self.correct, "accuracy": acc}



# main.py
import asyncio
import logging
import json
from agents import Orchestrator
from evaluator import SimpleEvaluator
import gradio as gr
import os

# Setup logging & simple metrics to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Instantiate orchestrator and evaluator
orchestrator = Orchestrator()
evaluator = SimpleEvaluator()

# For demo: a few sample expected answers for evaluation
EXPECTED_RESPONSES = {
    "Where is my order ORD123?": "Your order ORD123 is out for delivery",
    "Are your masalas natural?": "Yes — our masalas are made with natural spices",
    "How long does delivery take?": "2-4 business days"
}

async def handle_async(session_id: str, user_text: str):
    res = await orchestrator.handle_message(session_id, user_text)
    # If expected exists, evaluate
    expected = EXPECTED_RESPONSES.get(user_text)
    eval_res = None
    if expected:
        eval_res = evaluator.evaluate(res["answer"], expected)
    return {
        "reply": res["answer"],
        "tools_used": json.dumps(res["tools_used"]),
        "analytics_top": str(res["analytics"]),
        "evaluation": eval_res or {}
    }

def handle(session_id: str, user_text: str):
    # run the async handler in event loop
    return asyncio.get_event_loop().run_until_complete(handle_async(session_id or "anon", user_text))

# Build a minimal Gradio UI to demo usage (deployable to Hugging Face Space)
with gr.Blocks() as demo:
    gr.Markdown("# SPR Customer Agent Demo")
    session_id = gr.Textbox(label="Session ID (use phone or email to maintain session)", value="anon")
    user_input = gr.Textbox(label="Customer message", placeholder="Type customer question like 'Where is my order ORD123?'")
    send = gr.Button("Send")
    reply = gr.Textbox(label="Agent reply")
    tools_used = gr.Textbox(label="Tools used (json)")
    analytics = gr.Textbox(label="Analytics (top topics)")
    evaluation = gr.Textbox(label="Evaluation")

    def on_click(session_id_val, user_input_val):
        out = handle(session_id_val.strip() or "anon", user_input_val)
        return out["reply"], out["tools_used"], out["analytics_top"], json.dumps(out["evaluation"])

    send.click(on_click, inputs=[session_id, user_input], outputs=[reply, tools_used, analytics, evaluation])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))



#data/mock_orders.json
[
  {
    "order_id": "ORD123",
    "phone": "9998887776",
    "items": ["Tandoori Masala (200g)", "Biryani Masala (100g)"],
    "status": "Out for delivery",
    "delivery_eta": "2025-12-02"
  },
  {
    "order_id": "ORD124",
    "phone": "9998887777",
    "items": ["Sambar Powder (250g)"],
    "status": "Delivered",
    "delivery_eta": "2025-11-28"
  }
]





