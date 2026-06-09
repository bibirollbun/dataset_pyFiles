"""
InnoDrive – Simple Multi-Agent Demo
Clean, error-free version for Kaggle Notebook.
Features included:
1. LLM-powered agent (mock, replaceable)
2. LoopAgent (retry + validation)
3. Custom Tools (sensor + driver-state simulator)
4. Simple memory system
5. Logging for observability
"""

import random
import time
import logging

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("InnoDrive")

# ----------------------------
# Simple Memory Store
# ----------------------------
class SimpleMemory:
    def __init__(self):
        self.sessions = {}

    def create(self, session_id):
        self.sessions[session_id] = {"events": []}

    def append(self, session_id, data):
        self.sessions[session_id]["events"].append(data)

    def get(self, session_id):
        return self.sessions.get(session_id, {})

memory = SimpleMemory()

# ----------------------------
# Mock LLM Tool
# ----------------------------
class MockLLM:
    def run(self, prompt):
        log.info("[LLM] Running mock LLM...")
        return "LLM Advice: Reduce speed, check engine temperature, stay alert."

# ----------------------------
# Custom Tools
# ----------------------------
def simulate_sensors():
    return {
        "speed": random.randint(40, 140),
        "engine_temp": random.randint(70, 200),
        "brake_pressure": random.uniform(10, 100),
        "vibration": random.uniform(0.0, 5.0)
    }

def simulate_driver_state():
    eyes_closed = random.choice([0, 1, 2, 3])
    return {
        "drowsy": eyes_closed >= 2,
        "eyes_closed_seconds": eyes_closed,
        "intoxicated": random.random() < 0.05
    }

# ----------------------------
# Base Agent Class
# ----------------------------
class BaseAgent:
    def __init__(self, name):
        self.name = name

    def run(self, context):
        raise NotImplementedError

# ----------------------------
# Loop Agent
# ----------------------------
class LoopAgent(BaseAgent):
    def __init__(self, name, agent, validator, max_retries=3):
        super().__init__(name)
        self.agent = agent
        self.validator = validator
        self.max_retries = max_retries

    def run(self, context):
        for i in range(1, self.max_retries + 1):
            log.info(f"[{self.name}] Try {i}/{self.max_retries}")
            out = self.agent.run(context)
            if self.validator(out):
                return out
            log.warning(f"[{self.name}] Validation failed. Retrying…")
        return out

# ----------------------------
# Individual Agents
# ----------------------------
class VehicleHealthAgent(BaseAgent):
    def run(self, context):
        sensors = simulate_sensors()
        alerts = []
        if sensors["engine_temp"] > 150:
            alerts.append("engine_overheat")
        if sensors["brake_pressure"] < 20:
            alerts.append("low_brake_pressure")
        if sensors["vibration"] > 3.5:
            alerts.append("high_vibration")

        result = {"sensors": sensors, "alerts": alerts}
        log.info(f"[VehicleHealth] {result}")
        return result

class DriverMonitorAgent(BaseAgent):
    def run(self, context):
        state = simulate_driver_state()
        log.info(f"[DriverMonitor] {state}")
        return state

class RiskAgent(BaseAgent):
    def run(self, context):
        vehicle = context["vehicle"]
        driver = context["driver"]
        s = vehicle["sensors"]

        risk = 0
        if s["speed"] > 100:
            risk += (s["speed"] - 100) * 0.5
        if s["engine_temp"] > 140:
            risk += (s["engine_temp"] - 140) * 0.7
        if driver["drowsy"]:
            risk += 30
        if driver["intoxicated"]:
            risk += 40

        risk = min(max(int(risk), 0), 100)
        log.info(f"[RiskAgent] score={risk}")

        return {"risk_score": risk}

class ActionAgent(BaseAgent):
    def run(self, context):
        risk = context["risk"]["risk_score"]
        if risk >= 80:
            actions = ["SOS", "ENGINE_SHUTDOWN"]
        elif risk >= 50:
            actions = ["ALERT_DRIVER", "SPEED_REDUCTION"]
        elif risk >= 20:
            actions = ["CAUTION"]
        else:
            actions = ["NO_ACTION"]

        log.info(f"[ActionAgent] actions={actions}")
        return {"actions": actions}

# ----------------------------
# Orchestrator (Master Agent)
# ----------------------------
class Orchestrator(BaseAgent):
    def __init__(self, llm=None):
        super().__init__("Orchestrator")
        self.llm = llm

        self.vehicle_agent = LoopAgent(
            "VehicleHealthLoop",
            VehicleHealthAgent("Vehicle"),
            validator=lambda out: out["sensors"]["engine_temp"] < 250
        )

        self.driver_agent = DriverMonitorAgent("Driver")
        self.risk_agent = RiskAgent("Risk")
        self.action_agent = ActionAgent("Action")

    def run(self, session_id="session1"):
        if session_id not in memory.sessions:
            memory.create(session_id)

        context = {}

        # 1. Vehicle check
        context["vehicle"] = self.vehicle_agent.run(context)
        memory.append(session_id, context["vehicle"])

        # 2. Driver monitor
        context["driver"] = self.driver_agent.run(context)
        memory.append(session_id, context["driver"])

        # 3. Risk evaluation
        context["risk"] = self.risk_agent.run(context)
        memory.append(session_id, context["risk"])

        # 4. Actions
        context["actions"] = self.action_agent.run(context)
        memory.append(session_id, context["actions"])

        # 5. LLM summary (optional)
        if self.llm:
            prompt = f"Vehicle:{context['vehicle']} Driver:{context['driver']} Risk:{context['risk']}"
            context["llm_summary"] = self.llm.run(prompt)
            memory.append(session_id, {"llm": context["llm_summary"]})

        return context

# ----------------------------
# Demo Runner
# ----------------------------
def run_demo(times=3):
    orch = Orchestrator(llm=MockLLM())

    for i in range(times):
        log.info(f"\n====== CYCLE {i+1} ======")
        out = orch.run("session1")

        print("\n--- SUMMARY ---")
        print("Vehicle:", out["vehicle"])
        print("Driver:", out["driver"])
        print("Risk:", out["risk"])
        print("Actions:", out["actions"])
        print("LLM:", out.get("llm_summary", None))
        print("----------------\n")
        time.sleep(1)


# Run demo
run_demo(3)


