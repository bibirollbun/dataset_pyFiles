from typing import List, Dict

# Base Agent Class
class Agent:
    def __init__(self, name):
        self.name = name

    def run(self, input_data):
        raise NotImplementedError

# SNOMED CT mock mapping
SNOMED_MAP = {
    "fever": "386661006",  # Fever (finding)
    "rash": "271807003",   # Rash (finding)
    "cough": "49727002",   # Cough (finding)
}

def map_to_snomed(symptoms: List[str]) -> List[str]:
    return [SNOMED_MAP.get(s, "unknown") for s in symptoms]

# MIMIC-IV mock query
def query_mimic_for_conditions(snomed_codes: List[str]) -> List[str]:
    condition_lookup = {
        "386661006": ["flu", "malaria"],
        "271807003": ["measles", "allergy"],
        "49727002": ["bronchitis", "COVID-19"]
    }
    conditions = []
    for code in snomed_codes:
        conditions.extend(condition_lookup.get(code, []))
    return list(set(conditions))

# Triage Agent
class TriageAgent(Agent):
    def run(self, input_data):
        symptoms = input_data.get("symptoms", "")
        parsed = [s for s in SNOMED_MAP if s in symptoms.lower()]
        return {"parsed_symptoms": parsed, "needs_imaging": "rash" in parsed}

# Imaging Agent
class ImagingAgent(Agent):
    def run(self, input_data):
        return {"image_findings": ["lesion", "inflammation"]}

# Diagnosis Agent
class DiagnosisAgent(Agent):
    def run(self, input_data):
        symptoms = input_data.get("parsed_symptoms", [])
        snomed_codes = map_to_snomed(symptoms)
        conditions = query_mimic_for_conditions(snomed_codes)
        return {"diagnosis": [(cond, round(1.0 / len(conditions), 2)) for cond in conditions]}

# Treatment Agent
class TreatmentAgent(Agent):
    def run(self, input_data):
        diagnosis = input_data.get("diagnosis", [])
        top_condition = diagnosis[0][0] if diagnosis else "unknown"
        plans = {
            "flu": "Rest, fluids, and paracetamol.",
            "COVID-19": "Isolation, monitoring, and antiviral therapy.",
            "measles": "Supportive care and vitamin A.",
            "malaria": "Antimalarial medication and hydration.",
            "allergy": "Antihistamines and avoid triggers.",
            "bronchitis": "Rest, fluids, and cough suppressants."
        }
        return {"treatment_plan": plans.get(top_condition, "Consult a specialist.")}

# Feedback Agent
class FeedbackAgent(Agent):
    def __init__(self):
        super().__init__("Feedback")
        self.memory = {}

    def run(self, diagnosis_output, user_feedback):
        confirmed = user_feedback.get("confirmed_condition")
        if confirmed:
            for i, (cond, score) in enumerate(diagnosis_output["diagnosis"]):
                if cond == confirmed:
                    diagnosis_output["diagnosis"][i] = (cond, round(score + 0.5, 2))
            self.memory[confirmed] = self.memory.get(confirmed, 0) + 1
        return diagnosis_output

# Orchestrator Agent with Feedback
class OrchestratorAgent:
    def __init__(self):
        self.triage = TriageAgent("Triage")
        self.imaging = ImagingAgent("Imaging")
        self.diagnosis = DiagnosisAgent("Diagnosis")
        self.treatment = TreatmentAgent("Treatment")
        self.feedback = FeedbackAgent()
        self.state = {}

    def run_with_feedback(self, user_input: Dict, user_feedback: Dict) -> Dict:
        # Step 1: Triage
        triage_output = self.triage.run(user_input)
        self.state.update(triage_output)

        # Step 2: Imaging
        if triage_output.get("needs_imaging"):
            imaging_output = self.imaging.run(user_input)
            self.state.update(imaging_output)

        # Step 3: Diagnosis
        diagnosis_output = self.diagnosis.run(self.state)
        self.state.update(diagnosis_output)

        # Step 4: Feedback
        updated_diagnosis = self.feedback.run(diagnosis_output, user_feedback)
        self.state["diagnosis"] = updated_diagnosis["diagnosis"]

        # Step 5: Treatment
        treatment_output = self.treatment.run(self.state)
        self.state.update(treatment_output)

        return {
            "parsed_symptoms": self.state.get("parsed_symptoms"),
            "diagnosis": self.state.get("diagnosis"),
            "treatment": self.state.get("treatment_plan")
        }

# Example usage
user_input = {"symptoms": "I have a fever and rash"}
user_feedback = {"confirmed_condition": "measles"}
orchestrator = OrchestratorAgent()
result = orchestrator.run_with_feedback(user_input, user_feedback)
print(result)

