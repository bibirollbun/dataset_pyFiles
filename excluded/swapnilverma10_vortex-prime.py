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


import datetime
import json
import operator
from typing import TypedDict, List, Dict, Any


# PART 1: MEMORY & OBSERVABILITY (vortex_memory.py)

class MemoryManager:
    """
    Handles session state and audit logging for the agent.
    Demonstrates 'Sessions & Memory' and 'Observability' requirements.
    """
    def __init__(self):
        # In-Memory Session Store (Short Term)
        self.session_store = {}
        # Log Store (Observability)
        self.audit_log = []

    def get_session(self, session_id: str) -> Dict:
        return self.session_store.get(session_id, {})

    def save_session(self, session_id: str, state: Dict):
        self.session_store[session_id] = state

    # Observability: Structured Logging
    def log_decision(self, agent_name: str, decision: str, reasoning: str):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": agent_name,
            "decision": decision,
            "reasoning_trace": reasoning  # Critical for "Agent Evaluation"
        }
        self.audit_log.append(entry)
        # Print for Kaggle output visibility
        print(f"\n[AUDIT LOG] Agent: {agent_name} | Decision: {decision}")
        print(f"            Trace: {reasoning}")

    # Long Term Memory (Simulated Vector DB Retrieval)
    def retrieve_customer_dna(self, email: str) -> List[str]:
        """
        Retrieves long-term context that doesn't fit in the current context window.
        e.g., "User historically hates upsells in Q4."
        """
        # In a real app, this queries Pinecone/Weaviate
        if "acme" in email:
             return ["Historical sensitivity to pricing", "Prefers email over phone"]
        return []

# PART 2: CUSTOM TOOLS (vortex_tools.py)

class CRMTools:
    """Mock Salesforce/HubSpot Interactions"""
    
    def get_customer_profile(self, email: str) -> Dict:
        # Mock API Call to Salesforce
        print(f"   [Tool] Querying Salesforce for {email}...")
        if "acme_corp" in email:
            return {
                "name": "Alice Exec", 
                "company": "Acme Corp", 
                "ltv": 120000, 
                "status": "Active",
                "billing_email": "alice@acme.com" # Old email
            }
        return {"name": "Unknown", "ltv": 0}

    def block_marketing_emails(self, email: str) -> str:
        # Mock API Call to Marketo
        return f"Marketing automation PAUSED for {email} in Marketo."

    def update_field(self, email: str, field: str, value: str) -> str:
        # Mock API Call to update record
        return f"Updated {field} to '{value}' for {email} in Salesforce."

class SentimentTools:
    """Mock Zendesk/NLP Interactions"""
    
    def analyze_recent_tickets(self, email: str) -> Dict:
        # In production, this uses an LLM to read the last 3 tickets
        # Returns a 'Compacted Context' summary
        print(f"   [Tool] Analyzing recent Zendesk tickets for {email}...")
        return {
            "summary": "User is experiencing repeated login failures on the enterprise SSO. Requested billing change.",
            "sentiment_score": 0.2, # Very low/Angry (Scale 0-1)
            "urgency": "high",
            "flags": ["billing_request"] 
        }

class SearchTools:
    """External Search Tools (Ghost Detector)"""
    
    def check_career_move(self, email: str) -> Dict:
        # Connects to Google Search / LinkedIn Scraper
        # Logic: Search for "User Name + Company" and see if recent profile says "Former"
        print(f"   [Tool] Scanning LinkedIn/Web for career changes...")
        return {"new_employer": "Unknown", "probability_left": 0.0}


# PART 3: AGENT ORCHESTRATION (vortex_core.py)

#  State Management (Context Engineering)
# This defines the "Session" memory that passes between agents.
class AgentState(TypedDict):
    ticket_id: str
    user_email: str
    crm_data: Dict
    support_data: Dict
    external_data: Dict
    consolidated_view: Dict
    actions_taken: List[str]
    sentiment_score: float
    is_vip: bool

class VortexOrchestrator:
    def __init__(self):
        self.memory = MemoryManager()
        self.crm_tool = CRMTools()
        self.sentiment_tool = SentimentTools()
        self.search_tool = SearchTools()

    # Step 1: Parallel Gathering (The "Gatherers")
    def gather_intelligence(self, state: AgentState) -> AgentState:
        print("\n--- STEP 1: [Supervisor] Dispatching parallel agents ---")
        
        # 1. Fetch CRM Data
        crm_data = self.crm_tool.get_customer_profile(state['user_email'])
        
        # 2. Analyze Support Context
        support_context = self.sentiment_tool.analyze_recent_tickets(state['user_email'])
        
        # 3. Ghost Detector (Optional Trigger)
        external_data = {}
        if "bounce" in support_context.get('flags', []):
            external_data = self.search_tool.check_career_move(state['user_email'])
        else:
            print("   [Supervisor] No bounce detected, skipping Ghost Detector.")

        # Log this gathering phase
        self.memory.log_decision(
            "Gatherer Agent", 
            "Data Collected", 
            f"Found CRM profile for {crm_data.get('name')} and analyzed sentiment."
        )
            
        return {
            **state,
            "crm_data": crm_data,
            "support_data": support_context,
            "external_data": external_data
        }

    # Step 2: The "Thinker" (Conflict Resolution & Context Compaction)
    def synthesize_golden_record(self, state: AgentState) -> AgentState:
        print("\n--- STEP 2: [Analyst] Resolving conflicts and compacting context ---")
        
        # Context Compaction: Summarizing verbose support logs
        summary = state['support_data'].get('summary', 'No interaction')
        
        # Logic: Determine VIP status and Sentiment Score
        ltv = state['crm_data'].get('ltv', 0)
        is_vip = ltv > 50000
        sentiment = state['support_data'].get('sentiment_score', 0.5)
        
        reasoning = f"LTV is ${ltv} (VIP Threshold: $50k). Sentiment is {sentiment} (Low < 0.3)."
        
        self.memory.log_decision(
            "Analyst Agent", 
            f"Classified User: VIP={is_vip}, Risk=HIGH", 
            reasoning
        )
        
        return {
            **state,
            "is_vip": is_vip,
            "sentiment_score": sentiment,
            "consolidated_view": {
                "name": state['crm_data'].get('name'),
                "risk_level": "HIGH" if sentiment < 0.3 else "LOW",
                "key_ask": summary
            }
        }

    # Step 3: The "Guardian" (Action Executor with Guardrails)
    def execute_guardrails(self, state: AgentState) -> AgentState:
        print("\n--- STEP 3: [Guardian] Checking policies and executing tools ---")
        actions = []

        # Feature: Sentiment Shield
        # Block marketing if sentiment is low and user is VIP
        if state['is_vip'] and state['sentiment_score'] < 0.4:
            result = self.crm_tool.block_marketing_emails(state['user_email'])
            actions.append(result)
            self.memory.log_decision("Guardian Agent", "Activated Sentiment Shield", "User is VIP and Angry. Pausing Marketing to prevent churn.")

        # Feature: VIP Alarm
        if state['is_vip'] and state['support_data'].get('urgency') == 'high':
             # Draft email Logic would go here
             alert_msg = "VIP ALARM: Slack notification sent to Account Manager with draft apology."
             actions.append(alert_msg)
             self.memory.log_decision("Guardian Agent", "Triggered VIP Alarm", "Urgency is HIGH for a VIP client.")

        # Feature: Data Correction (The "Golden Record" Fix)
        # In a real scenario, the LLM extracts 'billing@acme.com' from the text
        # and compares it to the CRM 'alice@acme.com'
        actions.append(self.crm_tool.update_field(state['user_email'], "Billing Email", "billing@acme.com"))

        return {**state, "actions_taken": actions}

# MAIN EXECUTION SIMULATION

if __name__ == "__main__":
    print("Initializing Vortex Prime System...")
    
    # 1. Simulate an incoming webhook payload
    initial_state = {
        "ticket_id": "TICK-992",
        "user_email": "alice@acme_corp.com",
        "crm_data": {},
        "support_data": {},
        "external_data": {},
        "consolidated_view": {},
        "actions_taken": [],
        "sentiment_score": 0.0,
        "is_vip": False
    }

    # 2. Instantiate the Orchestrator
    vortex = VortexOrchestrator()
    
    # 3. Run the Sequential Flow (The Graph)
    # In LangGraph, these would be nodes connected by edges
    state_step_1 = vortex.gather_intelligence(initial_state)
    state_step_2 = vortex.synthesize_golden_record(state_step_1)
    final_state = vortex.execute_guardrails(state_step_2)

    # 4. Final Report
    print("\n" + "="*40)
    print("FINAL VORTEX PRIME REPORT")
    print("="*40)
    print(f"User Identity: {final_state['consolidated_view'].get('name')}")
    print(f"VIP Status:    {final_state['is_vip']}")
    print(f"Risk Level:    {final_state['consolidated_view'].get('risk_level')}")
    print("\nActions Executed:")
    for action in final_state['actions_taken']:
        print(f" - {action}")
    print("="*40)




