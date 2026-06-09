"""
Enterprise Agent Framework – Capstone Project
---------------------------------------------
This program simulates an enterprise multi-agent system with
perception, reasoning, action, and memory layers.

Agents:
1. CustomerSupportAgent – handles customer tickets
2. FinanceAgent – analyzes transactions and flags anomalies
3. ITOpsAgent – monitors system logs and self-remediates issues

Each agent:
- Observes environment
- Reasons about tasks
- Executes actions
- Stores memory logs
"""

import time
import random
import datetime


# ---------------------------------------------------------
# Memory Layer
# ---------------------------------------------------------
class Memory:
    def __init__(self):
        self.log = []

    def store(self, message):
        entry = f"{datetime.datetime.now()} - {message}"
        self.log.append(entry)
        print("[MEMORY]", entry)

    def get_recent(self, n=5):
        return self.log[-n:]


# ---------------------------------------------------------
# Base Agent Class
# ---------------------------------------------------------
class EnterpriseAgent:
    def __init__(self, name):
        self.name = name
        self.memory = Memory()

    def perceive(self, data):
        self.memory.store(f"{self.name} perceived: {data}")
        return data

    def reason(self, data):
        raise NotImplementedError

    def act(self, decision):
        raise NotImplementedError

    def process(self, data):
        perceived = self.perceive(data)
        decision = self.reason(perceived)
        action_result = self.act(decision)
        self.memory.store(f"{self.name} completed action: {action_result}\n")
        return action_result


# ---------------------------------------------------------
# Customer Support Agent
# ---------------------------------------------------------
class CustomerSupportAgent(EnterpriseAgent):
    def reason(self, ticket):
        severity = "high" if "urgent" in ticket.lower() else "normal"
        decision = f"routing to {'Level 2' if severity == 'high' else 'Level 1'} support"
        self.memory.store(f"Decision: {decision}")
        return decision

    def act(self, decision):
        return f"Ticket processed and {decision}"


# ---------------------------------------------------------
# Finance Agent
# ---------------------------------------------------------
class FinanceAgent(EnterpriseAgent):
    def reason(self, transaction):
        is_anomaly = transaction > 5000 or transaction < 0
        decision = "FLAG" if is_anomaly else "APPROVE"
        self.memory.store(f"Decision: {decision}")
        return decision

    def act(self, decision):
        if decision == "FLAG":
            return "Fraud team alerted"
        return "Transaction approved"


# ---------------------------------------------------------
# IT Operations Agent
# ---------------------------------------------------------
class ITOpsAgent(EnterpriseAgent):
    def reason(self, log_message):
        if "error" in log_message.lower():
            decision = "restart_service"
        else:
            decision = "no_action"
        self.memory.store(f"Decision: {decision}")
        return decision

    def act(self, decision):
        if decision == "restart_service":
            return "Service restarted automatically"
        return "System stable – no action required"


# ---------------------------------------------------------
# Multi-Agent Orchestrator
# ---------------------------------------------------------
class EnterpriseAgentSystem:
    def __init__(self):
        self.support_agent = CustomerSupportAgent("CustomerSupportAgent")
        self.finance_agent = FinanceAgent("FinanceAgent")
        self.itops_agent = ITOpsAgent("ITOpsAgent")

    def run_cycle(self):
        print("\n=== ENTERPRISE AGENT SYSTEM CYCLE START ===")

        # Simulated incoming data
        ticket = random.choice([
            "Customer cannot login – urgent issue",
            "Password reset request",
            "Billing inquiry"
        ])
        transaction_value = random.randint(-200, 8000)
        system_log = random.choice([
            "INFO: System running normally.",
            "ERROR: Service timeout detected.",
            "INFO: Backup completed."
        ])

        print("\nProcessing customer support ticket...")
        self.support_agent.process(ticket)

        print("\nProcessing financial transaction...")
        self.finance_agent.process(transaction_value)

        print("\nProcessing IT operations log...")
        self.itops_agent.process(system_log)

        print("=== ENTERPRISE AGENT SYSTEM CYCLE END ===\n")


# ---------------------------------------------------------
# Main Program Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    system = EnterpriseAgentSystem()

    print("\n*** Enterprise Multi-Agent System Running ***\n")

    # Run 3 cycles for demonstration
    for i in range(3):
        system.run_cycle()
        time.sleep(1)
        """
Enterprise Agent Framework – Capstone Project
---------------------------------------------
This program simulates an enterprise multi-agent system with
perception, reasoning, action, and memory layers.

Agents:
1. CustomerSupportAgent – handles customer tickets
2. FinanceAgent – analyzes transactions and flags anomalies
3. ITOpsAgent – monitors system logs and self-remediates issues

Each agent:
- Observes environment
- Reasons about tasks
- Executes actions
- Stores memory logs
"""

import time
import random
import datetime


# ---------------------------------------------------------
# Memory Layer
# ---------------------------------------------------------
class Memory:
    def __init__(self):
        self.log = []

    def store(self, message):
        entry = f"{datetime.datetime.now()} - {message}"
        self.log.append(entry)
        print("[MEMORY]", entry)

    def get_recent(self, n=5):
        return self.log[-n:]


# ---------------------------------------------------------
# Base Agent Class
# ---------------------------------------------------------
class EnterpriseAgent:
    def __init__(self, name):
        self.name = name
        self.memory = Memory()

    def perceive(self, data):
        self.memory.store(f"{self.name} perceived: {data}")
        return data

    def reason(self, data):
        raise NotImplementedError

    def act(self, decision):
        raise NotImplementedError

    def process(self, data):
        perceived = self.perceive(data)
        decision = self.reason(perceived)
        action_result = self.act(decision)
        self.memory.store(f"{self.name} completed action: {action_result}\n")
        return action_result


# ---------------------------------------------------------
# Customer Support Agent
# ---------------------------------------------------------
class CustomerSupportAgent(EnterpriseAgent):
    def reason(self, ticket):
        severity = "high" if "urgent" in ticket.lower() else "normal"
        decision = f"routing to {'Level 2' if severity == 'high' else 'Level 1'} support"
        self.memory.store(f"Decision: {decision}")
        return decision

    def act(self, decision):
        return f"Ticket processed and {decision}"


# ---------------------------------------------------------
# Finance Agent
# ---------------------------------------------------------
class FinanceAgent(EnterpriseAgent):
    def reason(self, transaction):
        is_anomaly = transaction > 5000 or transaction < 0
        decision = "FLAG" if is_anomaly else "APPROVE"
        self.memory.store(f"Decision: {decision}")
        return decision

    def act(self, decision):
        if decision == "FLAG":
            return "Fraud team alerted"
        return "Transaction approved"


# ---------------------------------------------------------
# IT Operations Agent
# ---------------------------------------------------------
class ITOpsAgent(EnterpriseAgent):
    def reason(self, log_message):
        if "error" in log_message.lower():
            decision = "restart_service"
        else:
            decision = "no_action"
        self.memory.store(f"Decision: {decision}")
        return decision

    def act(self, decision):
        if decision == "restart_service":
            return "Service restarted automatically"
        return "System stable – no action required"


# ---------------------------------------------------------
# Multi-Agent Orchestrator
# ---------------------------------------------------------
class EnterpriseAgentSystem:
    def __init__(self):
        self.support_agent = CustomerSupportAgent("CustomerSupportAgent")
        self.finance_agent = FinanceAgent("FinanceAgent")
        self.itops_agent = ITOpsAgent("ITOpsAgent")

    def run_cycle(self):
        print("\n=== ENTERPRISE AGENT SYSTEM CYCLE START ===")

        # Simulated incoming data
        ticket = random.choice([
            "Customer cannot login – urgent issue",
            "Password reset request",
            "Billing inquiry"
        ])
        transaction_value = random.randint(-200, 8000)
        system_log = random.choice([
            "INFO: System running normally.",
            "ERROR: Service timeout detected.",
            "INFO: Backup completed."
        ])

        print("\nProcessing customer support ticket...")
        self.support_agent.process(ticket)

        print("\nProcessing financial transaction...")
        self.finance_agent.process(transaction_value)

        print("\nProcessing IT operations log...")
        self.itops_agent.process(system_log)

        print("=== ENTERPRISE AGENT SYSTEM CYCLE END ===\n")


# ---------------------------------------------------------
# Main Program Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    system = EnterpriseAgentSystem()

    print("\n*** Enterprise Multi-Agent System Running ***\n")

    # Run 3 cycles for demonstration
    for i in range(3):
        system.run_cycle()
        time.sleep(1)
        

