# Paste this into a code cell and run it (it's a shell command, note the leading !)
!pip install --upgrade pip
!pip install openai>=0.27.0 pandas rich python-dotenv



# WorkFlow Zero - Kaggle Demo Notebook
# Paste this entire cell into Kaggle after running the pip install cell.

import os
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from rich import print as rprint
from rich.table import Table
import pandas as pd

# Optional: use OpenAI LLM if API key present.
USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
if USE_OPENAI:
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")

# ----------------------------
# Data structures & utilities
# ----------------------------
@dataclass
class ActionLog:
    step: int
    name: str
    tool: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    status: str
    timestamp: float

class AuditTrail:
    def __init__(self):
        self.logs: List[ActionLog] = []

    def add(self, log: ActionLog):
        self.logs.append(log)

    def to_dict(self):
        return [asdict(l) for l in self.logs]

    def pretty(self):
        table = Table(title="Audit Trail")
        table.add_column("Step", justify="right")
        table.add_column("Action")
        table.add_column("Tool")
        table.add_column("Status")
        table.add_column("Result Summary")
        table.add_column("Time")
        for l in self.logs:
            table.add_row(str(l.step), l.name, l.tool, l.status, json.dumps(l.output)[:50], time.strftime("%X", time.localtime(l.timestamp)))
        rprint(table)

# ----------------------------
# Simulated Tools (safe demo)
# ----------------------------
def tool_search_vendors(item: str, quantity: int, region: str = "global"):
    vendors = [
        {"name": f"Acme Supplies Ltd.", "price_per_unit": 9.5, "lead_days": 14, "contact": "sales@acme.example"},
        {"name": f"Global Widgets Co.", "price_per_unit": 10.1, "lead_days": 10, "contact": "sales@global.example"},
        {"name": f"Local Makers", "price_per_unit": 11.0, "lead_days": 7, "contact": "sales@local.example"},
    ]
    for v in vendors:
        v["total_price"] = round(v["price_per_unit"] * quantity, 2)
        v["score"] = round(v["price_per_unit"] + (v["lead_days"] * 0.05), 3)
    return {"vendors": vendors, "query": {"item": item, "quantity": quantity, "region": region}}

def tool_draft_email(to: str, subject: str, body: str):
    draft_id = f"draft_{int(time.time())}"
    return {"draft_id": draft_id, "to": to, "subject": subject, "body": body}

def tool_schedule_calendar(event_title: str, start_date: str, participants: List[str]):
    event_id = f"evt_{int(time.time())}"
    return {"event_id": event_id, "title": event_title, "start": start_date, "participants": participants, "status": "confirmed"}

def tool_spreadsheet_analyze(csv_text: str):
    from io import StringIO
    df = pd.read_csv(StringIO(csv_text))
    summary = df.describe(include='all').to_dict()
    top_insights = []
    if "price" in df.columns:
        avg = df["price"].mean()
        top_insights.append({"insight": "Average price", "value": float(avg)})
    return {"summary": summary, "insights": top_insights, "rows": len(df)}

# ----------------------------
# LLM Planner / Fallback
# ----------------------------
def llm_plan(prompt: str) -> str:
    if USE_OPENAI:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"You are an expert business workflow planner. Produce a numbered step-by-step plan describing actions and the name of tools to call (e.g., search_vendors, draft_email, schedule_calendar, spreadsheet_analyze)."},
                {"role":"user","content": prompt},
            ],
            max_tokens=400,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    else:
        plan_lines = []
        plan_lines.append("1) Clarify intent if ambiguous. (skip for demo)")
        if "quote" in prompt or "quotes" in prompt or "vendor" in prompt or "supplier" in prompt:
            plan_lines.append("2) search_vendors -> gather 3 vendor quotes (tool_search_vendors).")
            plan_lines.append("3) rank vendors by price+lead time and build comparison table.")
            plan_lines.append("4) draft_email -> prepare an email to top vendor to request formal quote (tool_draft_email).")
            plan_lines.append("5) produce final decision and summary.")
        elif "analyze" in prompt or "csv" in prompt or "dataset" in prompt:
            plan_lines.append("2) spreadsheet_analyze -> analyze uploaded CSV (tool_spreadsheet_analyze).")
            plan_lines.append("3) produce insights, charts (simulated).")
            plan_lines.append("4) generate summary report and suggested next actions.")
        else:
            plan_lines.append("2) propose an initial plan of 3 steps: research, prepare deliverable, schedule follow-up.")
            plan_lines.append("3) call tools as needed (tool_search_vendors, tool_draft_email, tool_schedule_calendar).")
        return "\n".join(plan_lines)

# ----------------------------
# WorkFlowZero Agent
# ----------------------------
class WorkFlowZero:
    def __init__(self):
        self.audit = AuditTrail()
        self.step_counter = 0

    def _log(self, name, tool, inp, out, status="ok"):
        self.step_counter += 1
        log = ActionLog(step=self.step_counter, name=name, tool=tool, input=inp, output=out, status=status, timestamp=time.time())
        self.audit.add(log)

    def plan(self, user_request: str) -> str:
        rprint("[bold cyan]Planning...[/bold cyan]")
        plan_text = llm_plan(user_request)
        self._log("plan", "LLM", {"request": user_request}, {"plan": plan_text})
        return plan_text

    def execute_plan(self, plan_text: str, user_request: str) -> Dict[str, Any]:
        results = {"request": user_request, "plan": plan_text, "actions": []}
        if "search_vendors" in plan_text or "vendor" in plan_text or "quote" in user_request:
            item = "widgets"
            quantity = 100
            if "for" in user_request:
                parts = user_request.split("for")
                if len(parts) > 1:
                    tail = parts[1]
                    import re
                    m = re.search(r"(\d+)", tail)
                    if m:
                        quantity = int(m.group(1))
                    words = tail.strip().split()
                    if len(words) > 0:
                        item = words[-1].strip().strip(".,")
            out = tool_search_vendors(item=item, quantity=quantity)
            self._log("search_vendors", "tool_search_vendors", {"item": item, "quantity": quantity}, out)
            results["actions"].append({"name": "search_vendors", "result": out})
            vendors = out["vendors"]
            vendors_sorted = sorted(vendors, key=lambda v: (v["price_per_unit"], v["lead_days"]))
            top = vendors_sorted[0]
            self._log("rank_vendors", "internal", {"count": len(vendors)}, {"top_vendor": top})
            results["top_vendor"] = top
            subject = f"Request for Quote: {quantity} {item}"
            body = f"Hello {top['name']},\n\nWe are interested in purchasing {quantity} {item}. Please send your best quote and lead time.\n\nThanks."
            draft = tool_draft_email(to=top["contact"], subject=subject, body=body)
            self._log("draft_email", "tool_draft_email", {"to": top["contact"], "subject": subject}, draft)
            results["actions"].append({"name": "draft_email", "result": draft})
            calendar = tool_schedule_calendar(event_title="Vendor Follow-up", start_date=time.strftime("%Y-%m-%d"), participants=[top["contact"], "ops@company.example"])
            self._log("schedule_followup", "tool_schedule_calendar", {"event_title": "Vendor Follow-up"}, calendar)
            results["actions"].append({"name": "schedule_followup", "result": calendar})
        elif "spreadsheet_analyze" in plan_text or "analyze" in user_request:
            sample_csv = "product,price,units\nA,10,5\nB,12,7\nC,9,10\n"
            out = tool_spreadsheet_analyze(sample_csv)
            self._log("spreadsheet_analyze", "tool_spreadsheet_analyze", {"rows": out["rows"]}, out)
            results["actions"].append({"name": "spreadsheet_analyze", "result": out})
            subject = "Data Analysis Summary"
            body = f"Hi,\n\nFound {out['rows']} rows. Top insight: {out['insights']}\n\nRegards."
            draft = tool_draft_email(to="team@example.com", subject=subject, body=body)
            self._log("draft_email", "tool_draft_email", {"to":"team@example.com", "subject":subject}, draft)
            results["actions"].append({"name": "draft_email", "result": draft})
        else:
            research = {"notes": ["Researched competitors", "Found 3 potential approaches"], "sources": ["web.sim/1","web.sim/2"]}
            self._log("research", "simulated_research", {"query": user_request}, research)
            results["actions"].append({"name": "research", "result": research})
            draft = tool_draft_email(to="pm@example.com", subject="Deliverable: " + user_request, body="Draft deliverable attached.")
            self._log("draft_email", "tool_draft_email", {"to": "pm@example.com"}, draft)
            results["actions"].append({"name": "draft_email", "result": draft})
        decision_summary = {"recommendation": "Select vendor " + (results.get("top_vendor", {}).get("name", "TBD")),
                            "confidence": 0.81}
        self._log("final_decision", "internal", {}, decision_summary)
        results["decision"] = decision_summary
        return results

    def run(self, user_request: str):
        plan = self.plan(user_request)
        results = self.execute_plan(plan, user_request)
        rprint("[bold green]Execution Results[/bold green]")
        rprint(results["decision"])
        rprint("\n[bold yellow]Actions Summary[/bold yellow]")
        for a in results["actions"]:
            rprint(f"- {a['name']}: {list(a['result'].keys())}")
        self.audit.pretty()
        return {"plan": plan, "results": results, "audit": self.audit.to_dict()}

# ----------------------------
# Demo runs (Kaggle-friendly)
# ----------------------------
if __name__ == "__main__":
    agent = WorkFlowZero()
    rprint("[bold]WorkFlow Zero Demo[/bold]")
    rprint("Examples to try:")
    rprint("1) 'Get 3 supplier quotes for 100 widgets for my small shop.'")
    rprint("2) 'Analyze this CSV and summarize insights.'")
    rprint("3) 'Plan a marketing launch for our new serum.'")
    rprint("\nRunning demo #1...\n")

    request1 = "Get 3 supplier quotes for 100 widgets for my small shop."
    output1 = agent.run(request1)

    rprint("\nRunning demo #2...\n")
    request2 = "Analyze this CSV sales dataset and summarize insights."
    output2 = agent.run(request2)

    with open("workflow_zero_outputs.json", "w") as f:
        json.dump({"demo1": output1, "demo2": output2}, f, indent=2)

    rprint("\nSaved outputs to [bold]workflow_zero_outputs.json[/bold].")



agent = WorkFlowZero()
agent.run("Get 3 quotes for 50 bottles for my cosmetics brand.")


