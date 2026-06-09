# ADK style Planner agent (simplified)
from adk.agent import Agent
from a2a.message import Message

class PlannerAgent(Agent):
    """Decomposes enterprise request into tasks and assigns to agents."""
    def handle(self, request: dict):
        topic = request.get("topic")
        # simple decomposition - in real use call the model
        steps = [
            {"task":"discover_docs","query": f"{topic} policies, contracts"},
            {"task":"extract_info","fields": ["vendor_name","compliance_status"]},
            {"task":"analyze_risk","metrics": ["financial","security"]},
            {"task":"execute_onboard","systems":["erp","slack"]}
        ]
        # publish via A2A
        self.send("connector_agent", Message(payload={"tasks": steps}))



from adk.agent import Agent
from a2a.message import Message
from tools.web_search import web_search
from tools.pdf_parser import parse_pdfs

class ConnectorAgent(Agent):
    def handle(self, msg: Message):
        tasks = msg.payload["tasks"]
        docs = []
        for t in tasks:
            if t["task"]=="discover_docs":
                docs += web_search(t["query"])
        parsed = parse_pdfs(docs)
        self.send("extractor_agent", Message(payload={"documents": parsed}))



from adk.agent import Agent
from a2a.message import Message

class ExtractorAgent(Agent):
    def handle(self, msg: Message):
        documents = msg.payload["documents"]
        # call LLM via ADK model wrapper to extract structured facts
        facts = []
        for doc in documents:
            facts.append(self.model.extract(doc, schema={"vendor_name":"str","lic_status":"str"}))
        self.send("analyst_agent", Message(payload={"facts": facts}))



from adk.agent import Agent
from a2a.message import Message

class AnalystAgent(Agent):
    def handle(self, msg: Message):
        facts = msg.payload["facts"]
        # simple risk scoring logic stub
        risk = sum(1 for f in facts if f.get("lic_status","ok")!="ok")
        self.send("executor_agent", Message(payload={"facts": facts, "risk": risk}))



from adk.agent import Agent
from a2a.message import Message
from tools.slack_notifier import notify
from tools.gsuite_connector import create_vendor_entry

class ExecutorAgent(Agent):
    def handle(self, msg: Message):
        facts = msg.payload["facts"]
        risk = msg.payload["risk"]
        if risk > 0:
            self.send("validator_agent", Message(payload={"action":"halt","reason":"risk"}))
            return
        # perform actions (create ERP vendor, notify slack)
        create_vendor_entry(facts[0])
        notify(channel="#vendor-onboard", text=f"Onboarded: {facts[0]['vendor_name']}")
        self.send("validator_agent", Message(payload={"action":"done","details":facts}))



from adk.agent import Agent
from a2a.message import Message

class ValidatorAgent(Agent):
    def handle(self, msg: Message):
        # check that actions performed match approvals, log audit
        # return final report
        return {"status":"completed","payload": msg.payload}



{
  "name": "planner-agent",
  "description": "Decomposes enterprise tasks into agent jobs",
  "capabilities": ["plan","decompose","assign"],
  "input_modes": ["text"],
  "output_modes": ["a2a"]
}





