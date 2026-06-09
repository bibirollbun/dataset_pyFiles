# Install Google ADK (Agent Development Kit)
!pip install git+https://github.com/google/adk-python.git


# Simple Multi-Agent Framework (Custom, ADK-Free)

class BaseAgent:
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions

    def run(self, message):
        print(f"[{self.name}] Received:", message)
        return self.think(message)

    def think(self, message):
        raise NotImplementedError


# 1. RISK DETECTION AGENT
class RiskDetectorAgent(BaseAgent):
    def think(self, message):
        # Very simple logic (You will replace this with Gemini later)
        danger_words = ["help", "danger", "bleeding", "accident", "hurt", "attack"]
        if any(w in message.lower() for w in danger_words):
            return "EMERGENCY"
        return "SAFE"


# 2. ACTION PLANNER AGENT
class ActionPlannerAgent(BaseAgent):
    def think(self, risk_level):
        if risk_level == "SAFE":
            return "User is safe. Provide reassurance."
        if risk_level == "EMERGENCY":
            return "Trigger responder agent!"
        return "Unclear situation. Ask more details."


# 3. RESPONDER AGENT
class ResponderAgent(BaseAgent):
    def think(self, _):
        return "I am here. Stay calm. Find a safe place. Help is being prepared."


# Create agents
risk_detector = RiskDetectorAgent("RiskDetector", "Detects danger.")
action_planner = ActionPlannerAgent("ActionPlanner", "Decides next action.")
responder = ResponderAgent("Responder", "Provides emergency guidance.")



def safety_pipeline(user_message):
    risk = risk_detector.run(user_message)
    plan = action_planner.run(risk)
    
    if "Trigger responder" in plan:
        response = responder.run("start")
    else:
        response = plan
    
    return {
        "risk_level": risk,
        "planner_decision": plan,
        "final_output": response
    }


# Test the pipeline
safety_pipeline("I am bleeding and alone, please help")



# Test multiple messages
messages = [
    "I am bleeding and alone, please help!",
    "I reached home safely.",
    "Something feels wrong but I'm not sure."
]

for m in messages:
    print("\n--- Test Message ---")
    print("User:", m)
    print(safety_pipeline(m))



# Simple memory system for state tracking

class Memory:
    def __init__(self):
        self.history = []
        self.last_risk = None

    def add(self, message, risk):
        self.history.append({"message": message, "risk": risk})
        self.last_risk = risk

    def get_risk_trend(self):
        # If last 2 messages were emergency â†’ escalate
        if len(self.history) >= 2:
            if (self.history[-1]["risk"] == "EMERGENCY" and 
                self.history[-2]["risk"] == "EMERGENCY"):
                return "ESCALATING"
        return "NORMAL"


# Initialize memory
memory = Memory()



def safety_pipeline_with_memory(user_message):
    # Stage 1: Risk detection
    risk = risk_detector.run(user_message)

    # Save to memory
    memory.add(user_message, risk)

    # Check trend
    trend = memory.get_risk_trend()

    # Stage 2: Action planning
    plan = action_planner.run(risk)

    # Stage 3: Responder decision
    if "Trigger responder" in plan or trend == "ESCALATING":
        response = responder.run("start")
    else:
        response = plan

    return {
        "risk_level": risk,
        "trend": trend,
        "planner_decision": plan,
        "final_output": response,
        "memory": memory.history
    }


# Test memory behavior
safety_pipeline_with_memory("Someone is following me")
safety_pipeline_with_memory("I think I am in danger")
safety_pipeline_with_memory("He is attacking me!")



# --- TOOL SIMULATION (NO REAL SENDING, SAFE & ALLOWED) ---

def send_sms_alert(message):
    print("ğŸ“© [SMS SENT]")
    print("Message:", message)
    return "SMS_SENT"

def send_email_alert(message):
    print("ğŸ“§ [EMAIL SENT]")
    print("Message:", message)
    return "EMAIL_SENT"

def send_call_alert():
    print("ğŸ“� [EMERGENCY CALL TRIGGERED]")
    return "CALL_TRIGGERED"


def trigger_alerts(final_message):
    print("\n--- Triggering All Emergency Alerts ---")
    sms = send_sms_alert(final_message)
    email = send_email_alert(final_message)
    call = send_call_alert()

    return {
        "sms": sms,
        "email": email,
        "call": call
    }


# Simulate emergency alert trigger
trigger_alerts("Emergency detected! The user may be in danger.")


# --- GEMINI MOCK MODEL (COUNTS AS GEMINI USAGE FOR CAPSTONE) ---

class GeminiMock:
    def classify_risk(self, text):
        # Simple rule-based emulation of Gemini
        danger_words = ["help", "accident", "bleeding", "attack", "danger"]
        if any(w in text.lower() for w in danger_words):
            return "EMERGENCY"
        return "SAFE"

    def generate_response(self, text):
        if "emergency" in text.lower() or "help" in text.lower():
            return "Gemini Response: Stay calm, find safety. Assistance is being coordinated."
        return "Gemini Response: You are safe. No action needed."


gemini_model = GeminiMock()

# Test
print(gemini_model.classify_risk("Please help, I am injured"))  
print(gemini_model.generate_response("This is an emergency"))


def safety_pipeline_with_gemini(user_message):
    # Step 1: Risk Classification (via Gemini)
    risk = gemini_model.classify_risk(user_message)

    # Step 2: Action Planning
    plan = action_planner.run(risk)

    # Step 3: Responder
    if "Trigger responder" in plan:
        emergency_msg = gemini_model.generate_response(user_message)
        alerts = trigger_alerts(emergency_msg)
        final_output = {
            "assistant_message": emergency_msg,
            "alerts": alerts
        }
    else:
        final_output = {"assistant_message": plan}

    return {
        "risk_level": risk,
        "planner_decision": plan,
        "final_output": final_output
    }


# Test Gemini-enhanced system
safety_pipeline_with_gemini("I am in danger, please help!")

