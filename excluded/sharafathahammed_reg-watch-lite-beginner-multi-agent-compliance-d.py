# Optional - only if you plan to add a small local UI in the notebook
!pip install -q fastapi uvicorn


# Core imports and lightweight schemas
import json, uuid, time, os
from dataclasses import dataclass, asdict
from typing import List, Dict

@dataclass
class Draft:
    summary: str
    risk_score: int
    remediation_steps: List[str]
    suggested_policy_text: str
    references: List[str]

@dataclass
class Critique:
    decision: str   # "accept"|"refine"|"escalate"
    severity: str   # "low"|"med"|"high"
    comments: str
    required_changes: List[str]

class Session:
    def __init__(self, case_id, reg_text, context):
        self.id = str(uuid.uuid4())
        self.case_id = case_id
        self.regulation_text = reg_text
        self.company_context = context
        self.iterations = []
        self.start_time = time.time()
    def log_iteration(self, draft, critique):
        self.iterations.append({"draft": asdict(draft), "critique": asdict(critique), "ts": time.time()})

print("success")



golden = [
  {"id":"case_001",
   "regulation_snippet":"Services that process children's data require parental consent for analytics.",
   "company_context":"App collects ages but has no parental consent flow.",
   "expected_risk":"high","expected_action":"add_parental_consent"
  },
  {"id":"case_002",
   "regulation_snippet":"Health data must be stored encrypted at rest for 7 years.",
   "company_context":"Health logs are stored in plaintext for 1 year.",
   "expected_risk":"high","expected_action":"encrypt_and_extend_retention"},
  {"id":"case_003",
   "regulation_snippet":"Marketing emails require an opt-in and an unsubscribe link.",
   "company_context":"Marketing uses implied consent, no unsubscribe link.",
   "expected_risk":"medium","expected_action":"add_unsubscribe_optin"}
]
os.makedirs("data", exist_ok=True)
with open("data/golden_dataset.json","w") as f:
    json.dump(golden, f, indent=2)
print("Created golden dataset with", len(golden), "cases")



# Very simple mock drafting & critique
def mock_drafting(reg_text, company_context, prev_summary=None):
    summary = f"Summary: {reg_text[:140]}"
    risk = 80 if "child" in reg_text.lower() or "health" in reg_text.lower() else 50
    remediation = []
    if "consent" in reg_text.lower() or "parent" in reg_text.lower():
        remediation.append("Implement explicit parental consent flow; record timestamps.")
    if "encrypt" in reg_text.lower() or "encrypted" in reg_text.lower():
        remediation.append("Encrypt data at rest and update retention to 7 years.")
    if "unsubscribe" in reg_text.lower() or "opt-in" in reg_text.lower():
        remediation.append("Add clear unsubscribe link and opt-in checkbox.")
    if not remediation:
        remediation.append("Legal review required.")
    suggested = remediation[0]
    return Draft(summary=summary, risk_score=risk, remediation_steps=remediation, suggested_policy_text=suggested, references=[])

def mock_critique(draft, procedural_rules):
    # If high risk
    if draft.risk_score >= 75:
        # If remediation contains strong keywords → accept
        if ("consent" in " ".join(draft.remediation_steps).lower()) or \
           ("encrypt" in " ".join(draft.remediation_steps).lower()):
            # After refinement, accept
            return Critique(
                decision="accept",
                severity="high",
                comments="Remediation is now complete.",
                required_changes=[]
            )
        else:
            # First iteration: refine
            return Critique(
                decision="refine",
                severity="high",
                comments="Remediation missing or incomplete.",
                required_changes=["add_details"]
            )
    # Medium or low risk → accept
    return Critique(decision="accept", severity="low", comments="Looks good.", required_changes=[])

print('Success')



class MemoryBank:
    def __init__(self, path="data/memory_bank.json"):
        self.path = path
        if os.path.exists(path):
            with open(path,"r") as f: self.store = json.load(f)
        else:
            self.store = {"consent_rule": {"id":"consent_rule","desc":"If child data, require parental consent","threshold":70}}
            self._commit()
    def get_rule(self, id): return self.store.get(id)
    def list_rules(self): return list(self.store.keys())
    def update_rule(self, id, obj):
        self.store[id]=obj; self._commit()
    def _commit(self):
        with open(self.path,"w") as f: json.dump(self.store,f,indent=2)
memory = MemoryBank()
print("Memory rules:", memory.list_rules())



class DraftingAgent:
    def __init__(self, llm_fn=None): self.llm = llm_fn or mock_drafting
    def produce(self, reg_text, context, prev=None): return self.llm(reg_text, context, prev)

class CritiqueAgent:
    def __init__(self, llm_fn=None, memory_bank=None):
        self.llm = llm_fn or mock_critique
        self.memory = memory_bank
    def evaluate(self, draft):
        rules = {r: self.memory.get_rule(r) for r in self.memory.list_rules()}
        return self.llm(draft, rules)

print('success')


class Supervisor:
    def __init__(self, drafting, critique, memory, max_iters=3):
        self.drafting = drafting
        self.critique = critique
        self.memory = memory
        self.max_iters = max_iters

    def run(self, case):
        session = Session(case["id"], case["regulation_snippet"], case["company_context"])
        prev = None
        last_draft = None

        for i in range(self.max_iters):
            draft = self.drafting.produce(session.regulation_text, session.company_context, prev)
            last_draft = draft
            critique = self.critique.evaluate(draft)
            session.log_iteration(draft, critique)

            print(f"Iter {i+1} Decision:{critique.decision} Severity:{critique.severity}")

            if critique.decision == "accept":
                return {"status": "accepted", "final": draft, "session": session}

            if critique.decision == "escalate":
                return {"status": "escalate", "draft": draft, "critique": critique, "session": session}

            prev = draft.summary + " | refined"

        # Timeout now returns last draft
        return {"status": "timeout", "draft": last_draft, "session": session}

d_agent = DraftingAgent()
c_agent = CritiqueAgent(memory_bank=memory)
supervisor = Supervisor(d_agent, c_agent, memory)



with open("data/golden_dataset.json") as f: cases = json.load(f)
results=[]
for c in cases:
    print("\n--- Case:", c["id"])
    res = supervisor.run(c)
    results.append({"case":c["id"], "status":res["status"]})
print("\nSummary:", results)



def simple_eval(cases, supervisor):
    # synonym map for better beginner-friendly matching
    synonym_map = {
        "extend": ["extend", "update"],   # treat "update retention" as "extend retention"
        "optin": ["optin", "opt-in", "opt in"],  # normalize opt-in variations
    }

    results = []

    for c in cases:
        r = supervisor.run(c)

        # choose remediation text
        if r["status"] == "accepted":
            pred = r["final"].remediation_steps
        else:
            pred = r["draft"].remediation_steps

        pred_text = " ".join(pred).lower().replace("-", "").replace("/", " ")

        # Extract expected keywords except the first verb
        expected_keywords = c["expected_action"].split("_")[1:]  # ex: ["extend","retention"]

        matched_keywords = []

        for kw in expected_keywords:
            kw_norm = kw.lower().replace("-", "")

            # if keyword has synonyms, check any synonym
            if kw_norm in synonym_map:
                synonyms = synonym_map[kw_norm]
                if any(s in pred_text for s in synonyms):
                    matched_keywords.append(True)
                else:
                    matched_keywords.append(False)
            else:
                # normal keyword check
                matched_keywords.append(kw_norm in pred_text)

        matched = all(matched_keywords)

        results.append({
            "id": c["id"],
            "expected": c["expected_action"],
            "matched": matched,
            "status": r["status"]
        })

    return results


# ⭐ IMPORTANT: print must be OUTSIDE the function (no indentation!)
print(json.dumps(simple_eval(cases, supervisor), indent=2))



# Collect logs to data/session_logs.json for review & submission
logs=[]
for c in cases:
    r = supervisor.run(c)
    s = r.get("session")
    if s:
        logs.append({"case_id":c["id"], "session": s.__dict__})
with open("data/session_logs.json","w") as f: json.dump(logs,f, default=str, indent=2)
print("Saved session logs")


