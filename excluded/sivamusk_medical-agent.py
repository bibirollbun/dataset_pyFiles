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
import csv
from datetime import datetime
from pathlib import Path


DATA_CSV = Path("symptoms_dataset.csv")
MEMORY_JSON = Path("triage_memory.json")

csv_text = """symptom,base_score,advice,red_flag
fever,2,"Stay hydrated; monitor temperature; rest",False
headache,1,"Rest, hydrate; seek care if sudden severe headache",False
chest pain,5,"Seek emergency medical attention immediately",True
shortness_of_breath,4,"Avoid exertion; seek urgent medical care",True
cough,1,"Warm fluids; rest; see care if severe or productive",False
dizziness,2,"Sit or lie down; monitor; seek care if fainting occurs",True
abdominal pain,3,"Avoid solid food until mild; seek care if severe/persistent",False
nausea,1,"Oral rehydration; small sips; seek care if persistent",False
vomiting,2,"Oral rehydration; seek care if unable to keep fluids",False
high_fever,3,"Seek medical evaluation if fever > 39°C or prolonged",True
bleeding,5,"Apply direct pressure and seek emergency care",True
seizure,5,"Protect from injury; call emergency services",True
loss_of_consciousness,5,"Call emergency services immediately",True
rapid_heart_rate,4,"Stop exertion; seek urgent evaluation",True
confusion,4,"Seek urgent medical evaluation",True
fatigue,1,"Rest, nutrition, hydration",False
"""

# Save dataset to file
DATA_CSV.write_text(csv_text, encoding="utf-8")



def load_symptom_db(path=DATA_CSV):
    db = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = r["symptom"].strip().lower()
            db[key] = {
                "base_score": int(r["base_score"]),
                "advice": r["advice"],
                "red_flag": r.get("red_flag", "False").lower() in ("true", "1", "yes")
            }
    return db

SYM_DB = load_symptom_db()


def save_memory(report, path=MEMORY_JSON):
    mem = []
    if path.exists():
        try:
            mem = json.loads(path.read_text(encoding="utf-8") or "[]")
        except:
            mem = []
    mem.append(report)
    path.write_text(json.dumps(mem, indent=2), encoding="utf-8")


def load_memory(path=MEMORY_JSON):
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8") or "[]")
    except:
        return []



def tool_parse_symptoms(raw_text):
    tokens = [t.strip().lower().replace(" ", "_") for t in raw_text.split(",") if t.strip()]
    return tokens


def tool_calculate_severity(tokens, db=SYM_DB):
    breakdown = []
    total = 0
    red_flags = []
    unknown = []

    for t in tokens:
        if t in db:
            score = db[t]["base_score"]
            breakdown.append((t, score))
            total += score
            if db[t]["red_flag"]:
                red_flags.append(t)
        else:
            breakdown.append((t, 1))
            total += 1
            unknown.append(t)

    return {
        "total": total,
        "breakdown": breakdown,
        "red_flags": red_flags,
        "unknown": unknown
    }


 


def tool_get_advice(tokens, db=SYM_DB):
    adv = []
    for t in tokens:
        if t in db:
            adv.append(f"{t.replace('_',' ').title()}: {db[t]['advice']}")
        else:
            adv.append(f"{t.replace('_',' ').title()}: No specific advice available.")
    return adv


class SymptomAnalyzerAgent:
    def run(self, raw_text):
        tokens = tool_parse_symptoms(raw_text)
        calc = tool_calculate_severity(tokens)
        return {"tokens": tokens, **calc}


class SeverityClassifierAgent:
    def run(self, analysis):
        score = analysis["total"]
        if score >= 8:
            severity = "HIGH"
        elif score >= 4:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        return {
            "severity": severity,
            "total": score,
            "red_flags": analysis["red_flags"],
            "unknown": analysis["unknown"]
        }


class AdviceAgent:
    def run(self, tokens):
        return tool_get_advice(tokens)



class ReportAgent:
    def run(self, user_name, analysis, classification, advice):
        now = datetime.utcnow().isoformat() + "Z"
        return {
            "user": user_name,
            "timestamp_utc": now,
            "symptoms": [t.replace("_", " ") for t, _ in analysis["breakdown"]],
            "breakdown": [
                {"symptom": t.replace("_", " "), "score": s}
                for t, s in analysis["breakdown"]
            ],
            "total_score": classification["total"],
            "severity": classification["severity"],
            "red_flags": [r.replace("_", " ") for r in classification["red_flags"]],
            "unknown_symptoms": [u.replace("_", " ") for u in classification["unknown"]],
            "advice": advice
        }


def run_triage(symptom_text, user="User"):
    analyzer = SymptomAnalyzerAgent()
    classifier = SeverityClassifierAgent()
    advisor = AdviceAgent()
    reporter = ReportAgent()

    analysis = analyzer.run(symptom_text)
    classification = classifier.run(analysis)
    advice = advisor.run(analysis["tokens"])
    report = reporter.run(user, analysis, classification, advice)

    save_memory(report)
    return report


if __name__ == "__main__":
    print("\n=== SAMPLE RUN ===\n")
    out = run_triage("fever, headache, chest pain", user="DemoUser")
    print(json.dumps(out, indent=2))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline Medical Triage Agent - All-in-one Python Capstone Code
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# -------------------------
# Paths and dataset
# -------------------------
BASE_DIR = Path.cwd()
DATA_CSV = BASE_DIR / "symptoms_dataset.csv"
MEMORY_JSON = BASE_DIR / "triage_memory.json"
LOGS_JSON = BASE_DIR / "triage_logs.json"

EMBEDDED_CSV = """symptom,base_score,advice,red_flag,notes
fever,2,"Stay hydrated; monitor temperature; rest",False,"fever severity variable"
headache,1,"Rest and hydrate; seek care if sudden severe headache",False,"common"
chest_pain,5,"Seek emergency medical attention immediately",True,"major red flag"
shortness_of_breath,4,"Avoid exertion; seek urgent medical care",True,"major red flag"
cough,1,"Warm fluids and rest; seek care if severe or productive",False,"respiratory"
dizziness,2,"Sit or lie down; monitor; seek care if fainting occurs",True,"may indicate serious causes"
abdominal_pain,3,"Avoid solid food until mild; seek care if severe/persistent",False,"abdomen"
nausea,1,"Oral rehydration; small sips; seek care if persistent",False,"GI"
vomiting,2,"Oral rehydration; seek care if unable to keep fluids",False,"GI"
high_fever,3,"Seek medical evaluation if fever > 39°C or prolonged",True,"high fever red flag"
bleeding,5,"Apply direct pressure and seek emergency care",True,"external bleeding"
seizure,5,"Protect from injury; call emergency services",True,"neurologic emergency"
loss_of_consciousness,5,"Call emergency services immediately",True,"emergency"
rapid_heart_rate,4,"Stop exertion; seek urgent evaluation",True,"cardiac"
confusion,4,"Seek urgent medical evaluation",True,"neurological"
fatigue,1,"Rest, nutrition, hydration",False,"nonspecific"
"""

def ensure_dataset(force: bool = False):
    if force or not DATA_CSV.exists():
        DATA_CSV.write_text(EMBEDDED_CSV, encoding="utf-8")
        print(f"[dataset] Written embedded dataset to {DATA_CSV}")

# -------------------------
# Observability
# -------------------------
class Observability:
    def __init__(self, log_path: Path = LOGS_JSON):
        self.log_path = log_path
        self.session_id = f"session-{int(time.time())}"
        self.events: List[Dict[str, Any]] = []
        if not self.log_path.exists():
            self.log_path.write_text("[]", encoding="utf-8")

    def log_event(self, level: str, component: str, message: str, payload: Dict[str, Any] = None):
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "level": level,
            "component": component,
            "message": message,
            "payload": payload or {}
        }
        self.events.append(event)
        try:
            existing = json.loads(self.log_path.read_text(encoding="utf-8") or "[]")
        except Exception:
            existing = []
        existing.append(event)
        self.log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def trace_tool_call(self, tool_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        self.log_event("INFO", "TOOL_CALL", f"Tool {tool_name} invoked", {"inputs": inputs, "outputs": outputs})

    def trace_agent_action(self, agent_name: str, action: str, details: Dict[str, Any] = None):
        self.log_event("INFO", "AGENT_ACTION", f"Agent {agent_name} performed action {action}", details or {})

OBS = Observability()

# -------------------------
# Tools
# -------------------------
class ConditionDBTool:
    def __init__(self, csv_path: Path = DATA_CSV):
        self.csv_path = csv_path
        self.db = self._load_db()

    def _load_db(self) -> Dict[str, Dict[str, Any]]:
        db = {}
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = r["symptom"].strip().lower()
                db[key] = {
                    "base_score": int(r["base_score"]),
                    "advice": r["advice"],
                    "red_flag": r.get("red_flag", "False").lower() in ("true", "1", "yes"),
                    "notes": r.get("notes", "")
                }
        OBS.log_event("DEBUG", "ConditionDBTool", "Loaded condition DB", {"rows": len(db)})
        return db

    def query(self, symptom: str) -> Dict[str, Any]:
        key = symptom.strip().lower().replace(" ", "_")
        res = self.db.get(key)
        OBS.log_event("DEBUG", "ConditionDBTool", f"Query for {key}", {"found": bool(res)})
        return res

class SymptomCheckerTool:
    def parse(self, raw_text: str) -> List[str]:
        tokens = [t.strip().lower().replace(" ", "_") for t in raw_text.split(",") if t.strip()]
        OBS.log_event("DEBUG", "SymptomCheckerTool", "Parsed symptoms", {"input": raw_text, "tokens": tokens})
        return tokens

class RiskScoreTool:
    def __init__(self, db_tool: ConditionDBTool):
        self.db_tool = db_tool

    def compute(self, tokens: List[str]) -> Dict[str, Any]:
        breakdown = []
        total = 0
        red_flags = []
        unknown = []
        for t in tokens:
            row = self.db_tool.query(t)
            if row:
                sc = row["base_score"]
                breakdown.append((t, sc))
                total += sc
                if row.get("red_flag"):
                    red_flags.append(t)
            else:
                breakdown.append((t, 1))
                total += 1
                unknown.append(t)
        result = {"total": total, "breakdown": breakdown, "red_flags": red_flags, "unknown": unknown}
        OBS.trace_tool_call("RiskScoreTool.compute", {"tokens": tokens}, result)
        return result

# -------------------------
# Agents
# -------------------------
class IntakeAgent:
    def __init__(self, parser: SymptomCheckerTool):
        self.parser = parser

    def receive(self, raw_input: str) -> Dict[str, Any]:
        OBS.trace_agent_action("IntakeAgent", "receive", {"raw_input": raw_input})
        tokens = self.parser.parse(raw_input)
        return {"raw": raw_input, "tokens": tokens, "received_at": datetime.utcnow().isoformat() + "Z"}

class TriageAgent:
    def __init__(self, risk_tool: RiskScoreTool):
        self.risk_tool = risk_tool

    def classify(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        OBS.trace_agent_action("TriageAgent", "classify", {"tokens": analysis["tokens"]})
        risk = self.risk_tool.compute(analysis["tokens"])
        total = risk["total"]
        if total >= 8:
            severity = "HIGH"
        elif total >= 4:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        out = {"severity": severity, **risk}
        OBS.trace_agent_action("TriageAgent", "classified", out)
        return out

class RecommendationAgent:
    def __init__(self, db_tool: ConditionDBTool):
        self.db_tool = db_tool

    def recommend(self, tokens: List[str], severity: str, red_flags: List[str], unknowns: List[str]) -> Dict[str, Any]:
        OBS.trace_agent_action("RecommendationAgent", "recommend_start", {"tokens": tokens, "severity": severity})
        adv_lines = []
        for t in tokens:
            row = self.db_tool.query(t)
            if row:
                adv_lines.append(f"{t.replace('_',' ').title()}: {row['advice']}")
            else:
                adv_lines.append(f"{t.replace('_',' ').title()}: No specific advice available.")
        overall = []
        if severity == "HIGH" or len(red_flags) > 0:
            overall.append("Seek urgent medical attention or emergency services.")
        elif severity == "MEDIUM":
            overall.append("Schedule medical consultation soon; monitor symptoms closely.")
        else:
            overall.append("Self-care: rest, hydration, monitor symptoms. Seek care if worsen.")
        overall.append("This tool is educational only — not a substitute for professional medical advice.")
        result = {"per_symptom": adv_lines, "overall": overall, "red_flags": [r.replace("_"," ") for r in red_flags], "unknowns": [u.replace("_"," ") for u in unknowns]}
        OBS.trace_agent_action("RecommendationAgent", "recommend_done", result)
        return result

# -------------------------
# Memory helpers
# -------------------------
def persist_report(report: Dict[str, Any]):
    OBS.trace_agent_action("Memory", "persist_report", {"user": report.get("user"), "severity": report.get("severity")})
    save_list = []
    if MEMORY_JSON.exists():
        try:
            save_list = json.loads(MEMORY_JSON.read_text(encoding="utf-8") or "[]")
        except Exception:
            save_list = []
    save_list.append(report)
    MEMORY_JSON.write_text(json.dumps(save_list, indent=2), encoding="utf-8")

def load_reports() -> List[Dict[str, Any]]:
    if not MEMORY_JSON.exists():
        return []
    try:
        return json.loads(MEMORY_JSON.read_text(encoding="utf-8") or "[]")
    except Exception:
        return []

# -------------------------
# Pipeline
# -------------------------
def run_triage(symptoms_text: str, user: str = "Anonymous", save: bool = True) -> Dict[str, Any]:
    OBS.trace_agent_action("Pipeline", "start", {"user": user, "input": symptoms_text})
    condition_db = ConditionDBTool()
    parser_tool = SymptomCheckerTool()
    risk_tool = RiskScoreTool(condition_db)
    intake_agent = IntakeAgent(parser_tool)
    triage_agent = TriageAgent(risk_tool)
    recommend_agent = RecommendationAgent(condition_db)

    intake = intake_agent.receive(symptoms_text)
    classification = triage_agent.classify(intake)
    recommendations = recommend_agent.recommend(intake["tokens"], classification["severity"], classification["red_flags"], classification["unknown"])

    report = {
        "user": user,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "input_raw": intake["raw"],
        "tokens": intake["tokens"],
        "breakdown": [{ "symptom": t.replace("_"," "), "score": s } for t, s in classification["breakdown"]],
        "total_score": classification["total"],
        "severity": classification["severity"],
        "red_flags": [r.replace("_"," ") for r in classification["red_flags"]],
        "unknown_symptoms": [u.replace("_"," ") for u in classification["unknown"]],
        "advice_per_symptom": recommendations["per_symptom"],
        "overall_recommendations": recommendations["overall"]
    }

    if save:
        persist_report(report)
    OBS.trace_agent_action("Pipeline", "end", {"user": user, "severity": report["severity"]})
    return report

# -------------------------
# Pretty-print helper
# -------------------------
def pretty_print(report: Dict[str, Any]):
    print("\n=== TRIAGE REPORT ===")
    print(f"User: {report.get('user')} | Time (UTC): {report.get('timestamp_utc')}")
    print(f"Severity: {report.get('severity')} | Score: {report.get('total_score')}")
    if report.get("red_flags"):
        print("** RED FLAGS: ")
        for rf in report["red_flags"]:
            print(" -", rf)
    print("\nBreakdown:")
    for b in report.get("breakdown", []):
        print(f" - {b['symptom']}: {b['score']}")
    print("\nAdvice per symptom:")
    for a in report.get("advice_per_symptom", []):
        print(" •", a)
    print("\nOverall Recommendations:")
    for o in report.get("overall_recommendations", []):
        print(" •", o)
    print("=====================")

# -------------------------
# Evaluator
# -------------------------
def evaluator_run_tests():
    tests = [
        ("fever, headache", "LOW"),
        ("fever, chest_pain", "MEDIUM"),
        ("chest_pain, shortness_of_breath", "HIGH"),
        ("cough, fatigue", "LOW"),
        ("nausea, vomiting", "LOW"),
    ]
    results = []
    for inp, expected in tests:
        rpt = run_triage(inp, user="test", save=False)
        got = rpt["severity"]
        ok = (got == expected)
        results.append({"input": inp, "expected": expected, "got": got, "ok": ok, "score": rpt["total_score"]})
    print("\n[EVALUATOR] Test results:")
    for r in results:
        print(f" - Input: '{r['input']}' | expected: {r['expected']} | got: {r['got']} | score: {r['score']} | PASS: {r['ok']}")
    pass_count = sum(1 for r in results if r["ok"])
    print(f"[EVALUATOR] Passed {pass_count}/{len(results)} tests")

# -------------------------
# CLI interactive
# -------------------------
def cli_interactive():
    print("Offline Medical Triage Agent — Interactive CLI")
    print("Type 'quit' to exit. Type 'demo' to run sample inputs. Type 'memory' to view stored reports.")
    while True:
        inp = input("\nEnter symptoms (comma-separated): ").strip()
        if inp.lower() in ("quit", "exit"):
            print("Exiting.")
            break
        if inp.lower() == "demo":
            demo_inputs = [
                ("fever, headache", "DemoUser1"),
                ("cough, nausea", "DemoUser2"),
                ("shortness_of_breath, chest_pain", "DemoUser3"),
            ]
            for text, user in demo_inputs:
                rpt = run_triage(text, user=user)
                pretty_print(rpt)
            continue
        if inp.lower() == "memory":
            mem = load_reports()
            print(json.dumps(mem[-10:], indent=2))
            continue
        user = input("Enter user name (or press Enter for Anonymous): ").strip() or "Anonymous"
        report = run_triage(inp, user=user)
        pretty_print(report)

# -------------------------
# Main guard
# -------------------------
def main(argv):
    ensure_dataset(force=False)
    if "--run-tests" in argv:
        evaluator_run_tests()
        return
    if "--demo" in argv:
        demo_inputs = [
            ("fever, headache, chest_pain", "DemoUser1"),
            ("cough, fatigue", "DemoUser2"),
            ("shortness_of_breath, chest_pain", "DemoUser3")
        ]
        for txt, user in demo_inputs:
            rpt = run_triage(txt, user=user)
            pretty_print(rpt)
        return
    if "--interactive" in argv:
        cli_interactive()
        return
    print("\nUsage options:")
    print("  --interactive        start interactive CLI")
    print("  --demo               run built-in demo examples")
    print("  --run-tests          run evaluator unit tests")

if __name__ == "__main__":
    main(sys.argv[1:])



# Run demo examples
run_triage("fever, headache", user="DemoUser")
run_triage("cough, fatigue", user="DemoUser2")
run_triage("shortness_of_breath, chest_pain", user="DemoUser3")


