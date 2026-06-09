# ================================================================
# AUTOHEALTH ASSIST V3 â€” FULL REAL MULTI-AGENT SYSTEM (OPTION C)
# Updated for Gemini 2.5 Flash â€” Works on Kaggle (2025)
# ================================================================

!pip install google-generativeai --quiet

from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# Load API key
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

print("Gemini Connected Successfully âœ”ï¸�")


# ================================================================
# BASE AGENT CLASS (all agents use this)
# ================================================================
# ================================================================
# SMART BASE AGENT (Hybrid Model Strategy)
# ================================================================

class BaseAgent:
    def __init__(self, name, instructions, model_type="heavy"):
        self.name = name
        self.instructions = instructions

        # Use heavy model only when needed
        if model_type == "heavy":
            self.model_name = "gemini-2.5-flash"
        else:
            self.model_name = "gemini-flash-lite-latest"

        self.model = genai.GenerativeModel(self.model_name)

    def run(self, user_input):
        prompt = f"""
You are {self.name}. Follow instructions:

{self.instructions}

User Input:
{user_input}
"""
        response = self.model.generate_content(prompt)
        return response.text

# ================================================================
# INTENT DETECTION
# Intent Detection (lite)
IntentAgent = BaseAgent(
    "IntentAgent",
    """
Classify user input into ONE intent:
- SYMPTOM_ANALYSIS
- FIND_DOCTOR
- FIND_HOSPITAL
- BOOK_APPOINTMENT
- INSURANCE_QUERY
- EMERGENCY
- REMINDER
- GENERAL_HEALTH
Return ONLY the intent.
""",
    model_type="lite"
)

# Symptom Agent (heavy)
SymptomAgent = BaseAgent(
    "SymptomAgent",
    """
Extract symptoms and return JSON.
""",
    model_type="heavy"
)

# Doctor Recommend (heavy)
DoctorRecommendAgent = BaseAgent(
    "DoctorRecommendAgent",
    """
Recommend specialist based on symptoms.
""",
    model_type="heavy"
)

# Doctor Search (heavy)
DoctorSearchAgent = BaseAgent(
    "DoctorSearchAgent",
    """
List real doctors by specialty and state.
""",
    model_type="heavy"
)

# Hospital Search (heavy)
HospitalSearchAgent = BaseAgent(
    "HospitalSearchAgent",
    """
Return top hospitals in US state.
""",
    model_type="heavy"
)

# Appointment Agent (lite)
AppointmentAgent = BaseAgent(
    "AppointmentAgent",
    """
Give 5-step appointment plan.
""",
    model_type="lite"
)

# Insurance Agent (lite)
InsuranceAgent = BaseAgent(
    "InsuranceAgent",
    """
Explain which doctors accept insurance.
""",
    model_type="lite"
)

# Emergency Agent (lite)
EmergencyAgent = BaseAgent(
    "EmergencyAgent",
    """
Give emergency warnings and actions.
""",
    model_type="lite"
)

# Reminder Agent (lite)
ReminderAgent = BaseAgent(
    "ReminderAgent",
    """
Generate medicine reminders.
""",
    model_type="lite"
)

# ================================================================
# MEMORY SYSTEM
# ================================================================
Memory = []

def add_to_memory(entry):
    Memory.append(entry)


# ================================================================
# ORCHESTRATOR â€” ROUTES TO THE RIGHT AGENTS
# ================================================================
def orchestrator(user_message):

    intent = IntentAgent.run(user_message).strip()
    print(f"\nğŸ”� Detected Intent: {intent}\n")

    if intent == "SYMPTOM_ANALYSIS":
        s = SymptomAgent.run(user_message)
        d = DoctorRecommendAgent.run(s)
        r = ReminderAgent.run(s)
        add_to_memory({"symptoms": s})
        return f"{s}\n\n{d}\n\n{r}"

    elif intent == "FIND_DOCTOR":
        return DoctorSearchAgent.run(user_message)

    elif intent == "FIND_HOSPITAL":
        return HospitalSearchAgent.run(user_message)

    elif intent == "BOOK_APPOINTMENT":
        return AppointmentAgent.run(user_message)

    elif intent == "INSURANCE_QUERY":
        return InsuranceAgent.run(user_message)

    elif intent == "EMERGENCY":
        return EmergencyAgent.run(user_message)

    elif intent == "REMINDER":
        return ReminderAgent.run(user_message)

    else:
        return "I can help with healthcare, hospitals, doctors, symptoms, appointments, insurance, or emergencies."


# ================================================================
# TESTS â€” REAL USA HEALTHCARE ACTIONS
# ================================================================
print("===== Test 1: Symptom Analysis =====")
print(orchestrator("I have chest pain and shortness of breath."))

print("\n===== Test 2: Doctor Search =====")
print(orchestrator("Find cardiologists in Texas."))

print("\n===== Test 3: Hospital Search =====")
print(orchestrator("List top hospitals in California."))

print("\n===== Test 4: Appointment Booking =====")
print(orchestrator("Book an appointment with an orthopedic doctor tomorrow."))

print("\n===== Test 5: Insurance =====")
print(orchestrator("Which doctors accept Blue Cross insurance in Alabama?"))

print("\n===== Test 6: Reminder =====")
print(orchestrator("Remind me to take my thyroid medicine every morning."))

print("\n===== Test 7: Emergency =====")
print(orchestrator("I cannot breathe and my chest is hurting badly."))


