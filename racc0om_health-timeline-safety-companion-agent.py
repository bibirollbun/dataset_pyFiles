## 4️⃣ Imports & Config (Code cell)

# 3. Imports & Configuration

from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
import json

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# In Kaggle: add a Secret named "GOOGLE_API_KEY" in the notebook settings.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")

if genai is not None and GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
    genai.configure(api_key=GOOGLE_API_KEY)
    GEMINI_MODEL_NAME = "gemini-1.5-flash"  # or "gemini-1.5-pro" if enabled
    llm_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
else:
    # Fallback: notebook will still run with mocked LLM responses
    llm_model = None
    print("⚠️ Gemini not configured. Using mocked LLM responses.")



# 4. Data Models & Memory Store (Long-Term Patient Record)

@dataclass
class LabResult:
    name: str
    value: float
    unit: str
    reference_range: Optional[str] = None
    flag: Optional[str] = None  # e.g., "high", "low", "normal"


@dataclass
class Diagnosis:
    label: str           # e.g., "Type 2 Diabetes" (from doctor's report)
    source: str          # e.g., "Endocrinology clinic report"
    date: datetime


@dataclass
class Medication:
    name: str
    dose: str            # e.g., "500 mg"
    frequency: str       # e.g., "twice daily"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class SymptomLog:
    date: datetime
    description: str
    severity: Optional[int] = None  # 1–10 scale


@dataclass
class PatientRecord:
    patient_id: str
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    diagnoses: List[Diagnosis] = field(default_factory=list)
    lab_results: Dict[str, List[LabResult]] = field(default_factory=dict)  # key = test name
    medications: List[Medication] = field(default_factory=list)
    symptoms: List[SymptomLog] = field(default_factory=list)
    emergency_contacts: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_lab_result(self, lab: LabResult):
        self.lab_results.setdefault(lab.name.lower(), []).append(lab)
        self.updated_at = datetime.utcnow()

    def add_diagnosis(self, diag: Diagnosis):
        self.diagnoses.append(diag)
        self.updated_at = datetime.utcnow()

    def add_medication(self, med: Medication):
        self.medications.append(med)
        self.updated_at = datetime.utcnow()

    def add_symptom(self, symptom: SymptomLog):
        self.symptoms.append(symptom)
        self.updated_at = datetime.utcnow()


class PatientMemoryStore:
    """
    Simple in-memory store.
    In a real deployment, this can be replaced with a persistent database.
    """
    def __init__(self):
        self._store: Dict[str, PatientRecord] = {}

    def get_or_create(self, patient_id: str, name: Optional[str] = None) -> PatientRecord:
        if patient_id not in self._store:
            self._store[patient_id] = PatientRecord(patient_id=patient_id, name=name)
        return self._store[patient_id]

    def save(self, record: PatientRecord):
        self._store[record.patient_id] = record

    def to_json(self) -> str:
        def convert(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)
        return json.dumps({pid: asdict(rec) for pid, rec in self._store.items()}, default=convert, indent=2)



# 5. LLM Helper & Non-LLM Tools

def call_llm(system_prompt: str, user_prompt: str, model=None) -> str:
    """
    Helper to call Gemini. If no model is configured, returns a mock response.
    """
    if model is None:
        return f"[MOCKED LLM RESPONSE]\nSystem: {system_prompt[:80]}...\nUser: {user_prompt[:160]}..."
    response = model.generate_content(
        [
            {"role": "system", "parts": [system_prompt]},
            {"role": "user", "parts": [user_prompt]},
        ]
    )
    return response.text


def compute_lab_trends(record: PatientRecord) -> Dict[str, Dict[str, float]]:
    """
    Simple numeric trend calculator: compares first and last values.
    Returns: {test_name: {"first": v1, "last": v2, "delta": d}}
    """
    trends: Dict[str, Dict[str, float]] = {}
    for test_name, series in record.lab_results.items():
        numeric_values = [lr.value for lr in series if isinstance(lr.value, (int, float))]
        if len(numeric_values) >= 2:
            first = numeric_values[0]
            last = numeric_values[-1]
            trends[test_name] = {
                "first": first,
                "last": last,
                "delta": last - first,
            }
    return trends



# 6. Agent Implementations (Multi-Agent System)

class BaseAgent:
    def __init__(self, name: str, model=None):
        self.name = name
        self.model = model or llm_model

    def log(self, message: str):
        # Simple observability – can be swapped with real logging.
        print(f"[{self.name}] {message}")


class ReportExtractorAgent(BaseAgent):
    """
    Agent 1: Parses medical reports into structured data.
    Only extracts information explicitly present in the text.
    """
    def run(self, report_text: str, record: PatientRecord) -> PatientRecord:
        self.log("Starting report extraction.")
        system_prompt = (
            "You are a medical report parsing assistant. "
            "Extract ONLY information explicitly present in the text. "
            "Do NOT invent diagnoses or medicines. "
            "Return JSON with keys: 'labs', 'diagnoses', 'medications'. "
            "Each lab: {name, value, unit, reference_range, flag?}. "
            "Each diagnosis: {label, source, date?}. "
            "Each medication: {name, dose, frequency}."
        )
        user_prompt = f"Report text:\n{report_text}\n\nReturn JSON only."
        raw = call_llm(system_prompt, user_prompt, model=self.model)
        self.log(f"Raw LLM output (truncated): {raw[:200]}")

        try:
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            data = json.loads(raw[json_start:json_end])
        except Exception as e:
            self.log(f"Failed to parse JSON from LLM: {e}")
            return record

        for lab in data.get("labs", []):
            try:
                lr = LabResult(
                    name=lab.get("name", "unknown"),
                    value=float(lab.get("value")),
                    unit=lab.get("unit", ""),
                    reference_range=lab.get("reference_range"),
                    flag=lab.get("flag"),
                )
                record.add_lab_result(lr)
            except Exception as e:
                self.log(f"Skipping lab due to error: {e}")

        for d in data.get("diagnoses", []):
            try:
                diag = Diagnosis(
                    label=d.get("label", "unspecified"),
                    source=d.get("source", "report"),
                    date=datetime.fromisoformat(d.get("date")) if d.get("date") else datetime.utcnow(),
                )
                record.add_diagnosis(diag)
            except Exception as e:
                self.log(f"Skipping diagnosis due to error: {e}")

        for m in data.get("medications", []):
            try:
                med = Medication(
                    name=m.get("name", "unknown"),
                    dose=m.get("dose", ""),
                    frequency=m.get("frequency", ""),
                )
                record.add_medication(med)
            except Exception as e:
                self.log(f"Skipping medication due to error: {e}")

        self.log("Finished report extraction.")
        return record


class HealthExplainerAgent(BaseAgent):
    """
    Agent 2: Explains the record in simple language.
    Does NOT diagnose or prescribe; only general education + disclaimer.
    """
    def run(self, record: PatientRecord) -> str:
        self.log("Generating plain-language explanation.")
        summary_context = self._summarize_record(record)
        system_prompt = (
            "You are a helpful health explainer for patients.\n"
            "You ONLY explain information already in the patient's record.\n"
            "You must NOT diagnose, must NOT prescribe, and must NOT give personalized treatment plans.\n"
            "You can give general educational information and safety guidance.\n"
            "Always include this disclaimer at the end: "
            "'This information is for education only and is not medical advice. "
            "Please consult a doctor for diagnosis and treatment.'"
        )
        user_prompt = (
            "Here is a structured summary of the patient's health record:\n"
            f"{summary_context}\n\n"
            "Explain this in simple language for the patient, grouped into sections like "
            "Lab Results, Diagnoses (from doctor), Medications, and General Precautions."
        )
        explanation = call_llm(system_prompt, user_prompt, model=self.model)
        self.log("Explanation generated.")
        return explanation

    def _summarize_record(self, record: PatientRecord) -> str:
        parts = [f"Patient ID: {record.patient_id}", f"Name: {record.name}"]
        if record.diagnoses:
            parts.append("Diagnoses:")
            for d in record.diagnoses[-5:]:
                parts.append(f"- {d.label} (source: {d.source}, date: {d.date.date()})")
        if record.lab_results:
            parts.append("Recent lab results (last value per test):")
            for test_name, series in record.lab_results.items():
                last = series[-1]
                parts.append(
                    f"- {test_name}: {last.value} {last.unit} "
                    f"(ref: {last.reference_range}, flag: {last.flag})"
                )
        if record.medications:
            parts.append("Current medications:")
            for m in record.medications[-10:]:
                parts.append(f"- {m.name}: {m.dose}, {m.frequency}")
        if record.symptoms:
            parts.append("Recent symptom logs (last 5):")
            for s in record.symptoms[-5:]:
                parts.append(f"- {s.date.date()}: {s.description} (severity {s.severity})")
        return "\n".join(parts)


class TrendAnalyzerAgent(BaseAgent):
    """
    Agent 3: Computes and explains numeric trends from lab results.
    """
    def run(self, record: PatientRecord) -> str:
        self.log("Computing lab trends.")
        trends = compute_lab_trends(record)
        system_prompt = (
            "You are a health trends explainer.\n"
            "Given lab test trends over time, you highlight general patterns and generic precautions.\n"
            "You must NOT diagnose or provide specific treatment.\n"
        )
        user_prompt = (
            "Here are lab test trends for a patient (first, last, delta):\n"
            f"{json.dumps(trends, indent=2)}\n\n"
            "Explain any important patterns in simple language, and suggest generic healthy habits "
            "that are usually recommended (diet, exercise, sleep, follow-ups with doctor, etc.)."
        )
        text = call_llm(system_prompt, user_prompt, model=self.model)
        self.log("Trend explanation generated.")
        return text


class EmergencyHelperAgent(BaseAgent):
    """
    Agent 4: Provides general emergency/safety guidance and uses emergency contacts.
    """
    def run(self, record: PatientRecord, city: Optional[str] = None) -> str:
        self.log("Preparing emergency and safety guidance.")
        contacts_text = "\n".join(
            [f"- {c.get('name')} ({c.get('relation')}): {c.get('phone')}" for c in record.emergency_contacts]
        ) or "No emergency contacts stored."

        system_prompt = (
            "You are a safety-focused assistant. You provide general instructions on when to seek emergency help.\n"
            "You do NOT diagnose. You always recommend calling local emergency services if severe symptoms occur.\n"
        )
        user_prompt = (
            f"The patient has these emergency contacts:\n{contacts_text}\n\n"
            f"Their city (if provided) is: {city or 'unknown'}.\n\n"
            "1. List the general warning signs when anyone should seek urgent care.\n"
            "2. Explain how this patient can keep their emergency info handy.\n"
            "3. Remind them to follow their own doctor's instructions first.\n"
        )
        guidance = call_llm(system_prompt, user_prompt, model=self.model)
        self.log("Emergency guidance generated.")
        return guidance



# 7. Orchestrator: Wiring Agents into a Multi-Agent Workflow

class HealthCompanionOrchestrator:
    """
    High-level orchestrator that wires the agents together.
    Demonstrates a multi-agent system with shared memory.
    """
    def __init__(self, memory_store: Optional[PatientMemoryStore] = None, model=None):
        self.memory = memory_store or PatientMemoryStore()
        self.model = model or llm_model

        self.report_extractor = ReportExtractorAgent("ReportExtractor", model=self.model)
        self.health_explainer = HealthExplainerAgent("HealthExplainer", model=self.model)
        self.trend_analyzer = TrendAnalyzerAgent("TrendAnalyzer", model=self.model)
        self.emergency_helper = EmergencyHelperAgent("EmergencyHelper", model=self.model)

    def process_new_report(
        self,
        patient_id: str,
        report_text: str,
        patient_name: Optional[str] = None,
        city: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Main pipeline when a new medical report is uploaded.
        """
        print("\n=== New Report Pipeline Start ===")
        record = self.memory.get_or_create(patient_id, name=patient_name)

        # 1) Extract structured info from report (Agent 1)
        record = self.report_extractor.run(report_text, record)
        self.memory.save(record)

        # 2) Generate patient-friendly explanation (Agent 2)
        explanation = self.health_explainer.run(record)

        # 3) Analyze numeric trends (Agent 3)
        trend_text = self.trend_analyzer.run(record)

        # 4) Provide general emergency & safety guidance (Agent 4)
        safety_text = self.emergency_helper.run(record, city=city)

        print("=== Pipeline Complete ===\n")
        return {
            "explanation": explanation,
            "trends": trend_text,
            "safety": safety_text,
        }

    def log_symptom(self, patient_id: str, description: str, severity: Optional[int] = None):
        record = self.memory.get_or_create(patient_id)
        record.add_symptom(SymptomLog(date=datetime.utcnow(), description=description, severity=severity))
        self.memory.save(record)

    def add_emergency_contact(self, patient_id: str, name: str, phone: str, relation: str):
        record = self.memory.get_or_create(patient_id)
        record.emergency_contacts.append({"name": name, "phone": phone, "relation": relation})
        record.updated_at = datetime.utcnow()
        self.memory.save(record)



# 8. Better Demo: Two Visits Over Time + Trends + Memory View

orchestrator = HealthCompanionOrchestrator()

patient_id = "patient-001"

# Add some emergency contacts
orchestrator.add_emergency_contact(patient_id, "Mother", "+91-1234567890", "Family")
orchestrator.add_emergency_contact(patient_id, "Primary Doctor", "+91-9876543210", "Doctor")

# ---- Visit 1: Older report ----
sample_report_1 = """
Patient: Om
Date: 2025-07-01
Summary:
The patient has a history of Type 2 Diabetes documented by the endocrinology clinic.
Lab results:
- HbA1c: 7.0 % (reference: 4.0 - 5.6 %) [high]
- Fasting glucose: 130 mg/dL (reference: 70 - 99 mg/dL) [high]
Current medications:
- Metformin 500 mg, twice daily
Plan:
Follow up in 4 months with repeat labs.
"""

print("\n##### VISIT 1 (Older Report) #####")
_ = orchestrator.process_new_report(
    patient_id=patient_id,
    report_text=sample_report_1,
    patient_name="Om",
    city="Pune, India",
)

# Manually add lab values to memory (since LLM is mocked)
record = orchestrator.memory.get_or_create(patient_id)
record.add_lab_result(LabResult(
    name="HbA1c",
    value=7.0,
    unit="%",
    reference_range="4.0 - 5.6",
    flag="high",
))
record.add_lab_result(LabResult(
    name="Fasting glucose",
    value=130,
    unit="mg/dL",
    reference_range="70 - 99",
    flag="high",
))

# ---- Visit 2: Newer report ----
sample_report_2 = """
Patient: Om
Date: 2025-11-20
Summary:
The patient continues follow-up for Type 2 Diabetes.
Lab results:
- HbA1c: 7.8 % (reference: 4.0 - 5.6 %) [high]
- Fasting glucose: 140 mg/dL (reference: 70 - 99 mg/dL) [high]
Current medications:
- Metformin 500 mg, twice daily
- Atorvastatin 10 mg, once daily
Plan:
Encouraged lifestyle modification. Repeat labs in 3 months.
"""

print("\n##### VISIT 2 (Newer Report) #####")
output = orchestrator.process_new_report(
    patient_id=patient_id,
    report_text=sample_report_2,
    patient_name="Om",
    city="Pune, India",
)

# Add newer lab values for trends
record = orchestrator.memory.get_or_create(patient_id)
record.add_lab_result(LabResult(
    name="HbA1c",
    value=7.8,
    unit="%",
    reference_range="4.0 - 5.6",
    flag="high",
))
record.add_lab_result(LabResult(
    name="Fasting glucose",
    value=140,
    unit="mg/dL",
    reference_range="70 - 99",
    flag="high",
))

# Log a recent symptom entry
orchestrator.log_symptom(
    patient_id=patient_id,
    description="Felt tired after minimal activity",
    severity=5,
)

print("\n=== EXPLANATION (after Visit 2) ===")
print(output["explanation"])

print("\n=== TRENDS (HbA1c / Fasting Glucose over time) ===")
print(output["trends"])

print("\n=== SAFETY & EMERGENCY GUIDANCE ===")
print(output["safety"])

print("\n=== INTERNAL PATIENT MEMORY (JSON view) ===")
print(orchestrator.memory.to_json())

# Create the required submission file for Kaggle
with open("submission.txt", "w") as f:
    f.write("Notebook executed successfully.")


