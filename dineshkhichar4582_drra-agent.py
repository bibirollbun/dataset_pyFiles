# 3. Setup & Imports

from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
import json
import math
import random
import os

import google.generativeai as genai  # Gemini import

# For reproducibility
random.seed(42)



# Gemini safely using Kaggle Secrets

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

api_key = user_secrets.get_secret("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    print("âœ”ï¸� Gemini API key loaded successfully")
else:
    print("â�Œ GEMINI_API_KEY not found. Please add it in Kaggle Secrets.")



# Data Model & Long-Term Memory Implementation

from dataclasses import dataclass

from typing import List, Dict, Any
import json


MEMORY_FILE = "disaster_memory.json"

@dataclass
class DisasterContext:
    location: str
    disaster_type: str
    severity_level: str
    summary: str
    timestamp: str

def load_memory() -> List[Dict[str, Any]]:
    """Load long-term disaster memory."""
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_memory(records: List[Dict[str, Any]]) -> None:
    """Save long-term disaster memory."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(records, f, indent=2)

def add_disaster_to_memory(ctx: DisasterContext) -> None:
    """Append a new disaster context to long-term memory."""
    records = load_memory()
    records.append(ctx.__dict__)
    save_memory(records)



# Custom Tool: Risk Score Calculator

def calculate_risk_score(people_affected: int,
                         severity_index: int,
                         infra_damage_index: int) -> float:
    """
    Simple demo risk model:
    - people_affected: 0â€“100 (normalized scale)
    - severity_index: 1â€“10
    - infra_damage_index: 1â€“10
    
    Returns a risk score from 0 to 100.
    """
    score = (people_affected * 0.5) + (severity_index * 3.0) + (infra_damage_index * 2.5)
    return float(min(score, 100.0))



# Observability Utilities

class SimpleLogger:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log(self, agent_name: str, message: str) -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent_name,
            "message": message
        }
        self.events.append(event)
        print(f"[{agent_name}] {message}")

    def to_json(self) -> str:
        return json.dumps(self.events, indent=2)


logger = SimpleLogger()



# Base Agent

class BaseAgent:
    def __init__(self, name: str, description: str, logger: SimpleLogger):
        self.name = name
        self.description = description
        self.logger = logger

    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement run().")



# Data Collector Agent

from typing import Dict, Any
import random

class DataCollectorAgent(BaseAgent):
    def run(self, query: str) -> Dict[str, Any]:
        """
        Simulated data collection.
        query example: 'Flood in coastal city of X'
        """
        self.logger.log(self.name, f"Collecting data for query: '{query}'")

        # Fake "parsed" fields from the query
        disaster_type = "flood" if "flood" in query.lower() else "disaster"
        location = query

        # Simulated metrics (could be derived from real APIs)
        people_affected = random.randint(40, 100)  # 0â€“100 scale
        severity_index = random.randint(6, 10)     # 1â€“10
        infra_damage_index = random.randint(5, 10) # 1â€“10

        raw_reports = [
            "Heavy rainfall in low-lying areas.",
            "Roads blocked in multiple districts.",
            "Reports of power cuts and contaminated water.",
        ]

        collected = {
            "query": query,
            "location": location,
            "disaster_type": disaster_type,
            "people_affected": people_affected,
            "severity_index": severity_index,
            "infra_damage_index": infra_damage_index,
            "raw_reports": raw_reports,
        }

        self.logger.log(self.name, f"Collected metrics: {collected}")
        return collected



# Resource Analyzer Agent

class ResourceAnalyzerAgent(BaseAgent):
    def run(self, collected: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log(self.name, "Analyzing collected data...")

        risk = calculate_risk_score(
            collected["people_affected"],
            collected["severity_index"],
            collected["infra_damage_index"]
        )

        # Simple rules for priority needs
        needs = set()

        if collected["people_affected"] > 60:
            needs.add("Food supplies")
            needs.add("Clean drinking water")

        if collected["infra_damage_index"] >= 7:
            needs.add("Temporary shelters")
            needs.add("Road clearing equipment")

        if collected["severity_index"] >= 8:
            needs.add("Medical camps")
            needs.add("Emergency rescue teams")

        if not needs:
            needs.add("Basic monitoring")
            needs.add("Local support coordination")

        analysis = {
            "location": collected["location"],
            "disaster_type": collected["disaster_type"],
            "risk_score": risk,
            "priority_needs": sorted(list(needs)),
            "people_affected": collected["people_affected"],
            "severity_index": collected["severity_index"],
            "infra_damage_index": collected["infra_damage_index"],
        }

        self.logger.log(self.name, f"Analysis result: {analysis}")
        return analysis



# Strategy Planner Agent (Gemini-enabled)

class StrategyPlannerAgent(BaseAgent):
    def __init__(self, name: str, description: str, logger: SimpleLogger,
                 use_gemini: bool = False):
        super().__init__(name, description, logger)
        self.use_gemini = use_gemini
        # configuration is already done globally via genai.configure()

    def _template_plan(self, analysis: Dict[str, Any]) -> str:
        """Deterministic template-based plan (no external API)."""
        loc = analysis["location"]
        risk = analysis["risk_score"]
        needs = ", ".join(analysis["priority_needs"])
        people = analysis["people_affected"]
        sev = analysis["severity_index"]
        infra = analysis["infra_damage_index"]

        plan = f"""
Disaster Relief Plan for {loc}

1. Situation Summary
   - Disaster Type       : {analysis['disaster_type']}
   - Estimated People    : {people} (0â€“100 scale)
   - Severity Index      : {sev}/10
   - Infrastructure Damage Index: {infra}/10
   - Overall Risk Score  : {risk:.1f}/100

2. Priority Needs
   Based on the current impact, the highest priority needs are:
   - {needs}

3. Recommended Immediate Actions (0â€“24 hours)
   - Deploy assessment teams to verify the worst-affected localities.
   - Provide clean drinking water and ready-to-eat food to affected clusters.
   - Set up temporary shelters in schools/community halls located in safer zones.
   - Arrange medical camps for basic health checks and first aid.
   - Coordinate with local authorities for road clearing and power restoration.

4. Short-Term Actions (1â€“3 days)
   - Establish a command center to coordinate NGOs, volunteers, and government units.
   - Map vulnerable groups (elderly, children, people with disabilities) and prioritize support.
   - Ensure safe sanitation and waste management around shelters.
   - Maintain updated situation reports every 6â€“12 hours.

5. Medium-Term Actions (3â€“7 days)
   - Start damage assessment for homes and infrastructure.
   - Plan phased return of families from shelters once areas are safe.
   - Provide mental health and counseling support where possible.

6. Assumptions & Limitations
   - This plan assumes roads are partially accessible.
   - Actual field data may modify priorities.
   - This is intended as a decision-support tool, not a replacement for human judgment.
        """.strip()
        return plan

    def _gemini_plan(self, analysis: Dict[str, Any]) -> str:
        """
        Uses Gemini (gemini-1.5-flash) to generate a richer, more natural plan.
        """
        prompt = f"""
You are an expert disaster-relief planner.

Generate a complete, structured disaster relief strategy based on this data:

{analysis}

Your response MUST include these sections:

1. Situation Summary
2. Priority Needs
3. Immediate Actions (0â€“24 hours)
4. Short-Term Actions (1â€“3 days)
5. Medium-Term Actions (3â€“7 days)
6. Risks, Assumptions and Limitations

Write clearly and concisely so that emergency responders can follow it.
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    def run(self, analysis: Dict[str, Any]) -> str:
        self.logger.log(self.name, "Planning strategy...")

        if self.use_gemini:
            try:
                plan = self._gemini_plan(analysis)
                self.logger.log(self.name, "Relief plan generated using Gemini.")
                return plan
            except Exception as e:
                self.logger.log(
                    self.name,
                    f"Gemini failed ({e}). Falling back to template planner."
                )

        # Fallback or when use_gemini = False
        plan = self._template_plan(analysis)
        self.logger.log(self.name, "Relief plan generated using template planner.")
        return plan



# Deployment Agent

from typing import List, Dict, Any
import json

class DeploymentAgent(BaseAgent):
    def run(self, plan: str) -> List[str]:
        self.logger.log(self.name, "Converting plan to checklist...")

        # For demo, we just create a structured checklist manually.
        checklist = [
            "Deploy assessment teams to worst-affected localities.",
            "Distribute clean drinking water and food packets.",
            "Set up temporary shelters in safe public buildings.",
            "Arrange medical camps and first-aid stations.",
            "Coordinate road clearing and power restoration with authorities.",
            "Establish central command center for coordination.",
            "Schedule regular situation reports (every 6â€“12 hours).",
        ]

        self.logger.log(self.name, f"Generated checklist with {len(checklist)} items.")
        return checklist



# Multi-Agent Orchestrator

class DRRAAgentSystem:
    def __init__(self, logger: SimpleLogger, use_gemini: bool = False):
        self.logger = logger

        self.collector = DataCollectorAgent(
            name="DataCollectorAgent",
            description="Collects structured disaster information.",
            logger=self.logger
        )
        self.analyzer = ResourceAnalyzerAgent(
            name="ResourceAnalyzerAgent",
            description="Analyzes risk and priority needs.",
            logger=self.logger
        )
        self.planner = StrategyPlannerAgent(
            name="StrategyPlannerAgent",
            description="Generates a disaster relief plan.",
            logger=self.logger,
            use_gemini=use_gemini
        )
        self.deployer = DeploymentAgent(
            name="DeploymentAgent",
            description="Converts plan into actionable checklist.",
            logger=self.logger
        )

    def run(self, query: str) -> Dict[str, Any]:
        self.logger.log("DRRAAgentSystem", "=== Starting full pipeline ===")

        collected = self.collector.run(query)
        analysis = self.analyzer.run(collected)
        plan = self.planner.run(analysis)
        checklist = self.deployer.run(plan)

        # Save to memory
        ctx = DisasterContext(
            location=analysis["location"],
            disaster_type=analysis["disaster_type"],
            severity_level="high" if analysis["risk_score"] > 60 else "moderate",
            summary=f"Risk {analysis['risk_score']:.1f}, needs: {analysis['priority_needs']}",
            timestamp=datetime.utcnow().isoformat()
        )
        add_disaster_to_memory(ctx)
        self.logger.log("DRRAAgentSystem", "Disaster context saved to long-term memory.")

        result = {
            "collected": collected,
            "analysis": analysis,
            "plan": plan,
            "checklist": checklist,
            "logs": self.logger.events,
            "memory_snapshot": load_memory()
        }

        self.logger.log("DRRAAgentSystem", "=== Pipeline completed ===")
        return result



# 24. Demo Run

system = DRRAAgentSystem(logger=logger, use_gemini=True)

user_query = "Severe flood in coastal city of X"
result = system.run(user_query)

print("\n" + "="*60)
print("RELIEF PLAN\n" + "="*60)
print(result["plan"])

print("\n" + "="*60)
print("CHECKLIST\n" + "="*60)
for i, item in enumerate(result["checklist"], start=1):
    print(f"{i}. {item}")

print("\n" + "="*60)
print("MEMORY SNAPSHOT (last 3 entries)\n" + "="*60)
mem = result["memory_snapshot"][-3:]
print(json.dumps(mem, indent=2))



# Simple Evaluation of the Generated Plan

def evaluate_plan(analysis: Dict[str, Any], checklist: List[str]) -> Dict[str, Any]:
    """
    Very simple heuristic evaluation:
    - base score from risk (higher risk = needs stronger response)
    - plus a bonus for number of checklist items
    All normalized to a 0â€“100 scale.
    """
    risk_component = min(analysis["risk_score"], 100.0)  # 0â€“100
    checklist_component = min(len(checklist) * 5, 30)   # up to +30 points

    total = min(risk_component * 0.7 + checklist_component, 100.0)
    return {
        "risk_component": risk_component,
        "checklist_component": checklist_component,
        "total_score": total
    }

eval_result = evaluate_plan(result["analysis"], result["checklist"])
print("Evaluation Result:", eval_result)


