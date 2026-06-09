!pip install -q google-genai google-adk geopy python-dotenv uvicorn


import os
import json
import uuid
import logging
import warnings
from datetime import datetime, timedelta
from kaggle_secrets import UserSecretsClient
from typing import List, Dict, Optional, Literal

from geopy.distance import geodesic
from geopy.geocoders import Nominatim

from pydantic import BaseModel, Field

from google.adk.apps import App
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("healthcare_notebook")


user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()

# Set kredensial agar library google-auth menemukannya
user_secrets.set_tensorflow_credential(user_credential)

print("Authentication Success")


# Suppress Google Cloud metadata warnings
warnings.filterwarnings('ignore')


# Set your Gemini API key. On Kaggle: use Secrets for safety.
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    GEMINI_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    # Optional: Set project (not required for Gemini API)
    os.environ["GOOGLE_CLOUD_PROJECT"] = "healthcare-agent-demo-479919"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


class PatientInput(BaseModel):
    symptoms: str = Field(description="Patient's symptoms")
    location: str = Field(description="Patient's location (address or coordinates)")
    emergency: bool = Field(default=False, description="If the condition is an emergency")
    patient_id: Optional[str] = None
    insurance_type: Optional[str] = None
    preferred_language: str = Field(default="en", description="Preferred language")

class TriageResult(BaseModel):
    urgency_level: Literal["emergency", "urgent", "routine"]
    recommended_facility: Literal["hospital", "clinic", "pharmacy", "telemedicine"]
    estimated_wait_time: str
    reasoning: str
    red_flags: List[str] = []

class HealthFacility(BaseModel):
    name: str
    type: str
    address: str
    distance_km: float
    available_doctors: int
    services: List[str]
    cost_range: str
    accepts_insurance: bool
    contact: str

class AppointmentSlot(BaseModel):
    facility_name: str
    doctor_name: str
    specialty: str
    datetime: datetime
    duration_minutes: int
    estimated_cost: float
    booking_id: Optional[str] = None

class TransportationOption(BaseModel):
    type: Literal["ambulance", "community_transport", "ride_share", "public"]
    cost: float
    eta_minutes: int
    contact: Optional[str] = None
    availability: bool

class MedicationInfo(BaseModel):
    medication_name: str
    available_at: List[str]
    price_range: str
    requires_prescription: bool
    alternative_generic: Optional[str] = None


class HealthcareTools:
    def __init__(self, maps_api_key: str = None):
        self.maps_api_key = maps_api_key
        self.geolocator = Nominatim(user_agent="healthcare_agent_demo")
        self.facilities_db = self._load_facilities_database()
        self.doctors_schedule_db = self._load_doctors_schedule()
        self.medication_db = self._load_medication_database()
    
    def _load_facilities_database(self) -> List[Dict]:
        return [
            {
                "id": "fac_001",
                "name": "Puskesmas Desa Makmur",
                "type": "clinic",
                "lat": -6.2088,
                "long": 106.8456,
                "services": ["general_checkup", "minor_emergency", "vaccination"],
                "doctors_available": 2,
                "cost_range": "Rp 10,000 - 50,000",
                "accepts_insurance": True,
                "contact": "+62-21-1234567",
                "operating_hours": "08:00-16:00"
            },
            {
                "id": "fac_002",
                "name": "RS Harapan Sehat",
                "type": "hospital",
                "lat": -6.1751,
                "long": 106.8650,
                "services": ["emergency", "surgery", "specialist", "imaging"],
                "doctors_available": 15,
                "cost_range": "Rp 100,000 - 500,000",
                "accepts_insurance": True,
                "contact": "+62-21-7654321",
                "operating_hours": "24/7"
            },
            {
                "id": "fac_003",
                "name": "Klinik Pratama Sejahtera",
                "type": "clinic",
                "lat": -6.2297,
                "long": 106.8172,
                "services": ["general_checkup", "pediatrics", "dental"],
                "doctors_available": 3,
                "cost_range": "Rp 30,000 - 100,000",
                "accepts_insurance": False,
                "contact": "+62-21-9876543",
                "operating_hours": "09:00-18:00"
            }
        ]
    
    def _load_doctors_schedule(self) -> List[Dict]:
        base_date = datetime.now()
        return [
            {
                "facility_id": "fac_001",
                "doctor_name": "dr. Budi Santoso",
                "specialty": "General Practitioner",
                "available_slots": [
                    (base_date + timedelta(hours=2)).isoformat(),
                    (base_date + timedelta(hours=4)).isoformat(),
                    (base_date + timedelta(days=1, hours=2)).isoformat()
                ],
                "cost": 50000
            },
            {
                "facility_id": "fac_002",
                "doctor_name": "dr. Siti Aminah, Sp.PD",
                "specialty": "Internal Medicine",
                "available_slots": [
                    (base_date + timedelta(hours=3)).isoformat(),
                    (base_date + timedelta(days=1, hours=1)).isoformat()
                ],
                "cost": 250000
            }
        ]
    
    def _load_medication_database(self) -> List[Dict]:
        return [
            {
                "name": "Paracetamol 500mg",
                "generic": True,
                "requires_prescription": False,
                "available_at": ["Apotek Sehat", "Apotek 24 Jam", "Puskesmas Desa Makmur"],
                "price_range": "Rp 3,000 - 8,000"
            },
            {
                "name": "Amoxicillin 500mg",
                "generic": True,
                "requires_prescription": True,
                "available_at": ["Apotek Sehat", "RS Harapan Sehat"],
                "price_range": "Rp 15,000 - 30,000"
            }
        ]
    
    def assess_triage(self, symptoms: str, age: int = None, emergency: bool = False) -> str:
        emergency_keywords = [
            "chest pain", "nyeri dada", "sesak napas", "stroke",
            "perdarahan hebat", "trauma kepala", "kejang", "tidak sadar"
        ]
        urgent_keywords = [
            "demam tinggi", "high fever", "muntah terus", "diare berat",
            "nyeri perut hebat", "bengkak tiba-tiba"
        ]
        symptoms_lower = symptoms.lower()
        
        if emergency or any(kw in symptoms_lower for kw in emergency_keywords):
            urgency = "emergency"
            facility_type = "hospital"
            wait_time = "Immediate - <30 minutes"
        elif any(kw in symptoms_lower for kw in urgent_keywords):
            urgency = "urgent"
            facility_type = "clinic"
            wait_time = "Same day - 1-4 hours"
        else:
            urgency = "routine"
            facility_type = "clinic"
            wait_time = "1-3 days"
        
        result = {
            "urgency_level": urgency,
            "recommended_facility": facility_type,
            "estimated_wait_time": wait_time,
            "reasoning": f"Based on symptoms: {symptoms}",
            "red_flags": [kw for kw in emergency_keywords if kw in symptoms_lower]
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def find_nearby_facilities(self, patient_location: str, facility_type: str,
                               max_distance_km: float = 20) -> str:
        try:
            location = self.geolocator.geocode(patient_location)
            if not location:
                return json.dumps({"error": "Location not found"})
            
            patient_coords = (location.latitude, location.longitude)
            nearby_facilities = []
            for facility in self.facilities_db:
                if facility["type"] != facility_type:
                    continue
                facility_coords = (facility["lat"], facility["long"])
                distance = geodesic(patient_coords, facility_coords).kilometers
                if distance <= max_distance_km:
                    nearby_facilities.append({
                        **facility,
                        "distance_km": round(distance, 2)
                    })
            nearby_facilities.sort(key=lambda x: x["distance_km"])
            return json.dumps(nearby_facilities[:5], indent=2, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def check_available_appointments(self, facility_id: str, specialty: str = None) -> str:
        available_slots = []
        for doc_schedule in self.doctors_schedule_db:
            if doc_schedule["facility_id"] == facility_id:
                if specialty is None or specialty.lower() in doc_schedule["specialty"].lower():
                    for slot in doc_schedule["available_slots"]:
                        available_slots.append({
                            "facility_id": facility_id,
                            "doctor_name": doc_schedule["doctor_name"],
                            "specialty": doc_schedule["specialty"],
                            "datetime": slot,
                            "duration_minutes": 30,
                            "estimated_cost": doc_schedule["cost"]
                        })
        return json.dumps(available_slots, indent=2, ensure_ascii=False)
    
    def book_appointment(self, facility_id: str, doctor_name: str,
                         datetime_slot: str, patient_id: str) -> str:
        booking_id = f"BK{datetime.now().strftime('%Y%m%d%H%M%S')}"
        result = {
            "status": "confirmed",
            "booking_id": booking_id,
            "facility_id": facility_id,
            "doctor_name": doctor_name,
            "datetime": datetime_slot,
            "patient_id": patient_id,
            "confirmation_message": f"Appointment successfully created. Booking ID: {booking_id}",
            "reminder": "Please arrive 15 minutes early. Bring ID and insurance card if available."
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def find_transportation(self, from_location: str, to_location: str,
                            urgency: str = "routine") -> str:
        options = []
        if urgency == "emergency":
            options.append({
                "type": "ambulance",
                "cost": 150000,
                "eta_minutes": 15,
                "contact": "+62-21-118-119",
                "availability": True
            })
        options.append({
            "type": "community_transport",
            "cost": 25000,
            "eta_minutes": 30,
            "contact": "+62-812-3456-7890",
            "availability": True
        })
        options.append({
            "type": "ride_share",
            "cost": 35000,
            "eta_minutes": 20,
            "contact": "Book via app: Gojek/Grab",
            "availability": True
        })
        return json.dumps(options, indent=2, ensure_ascii=False)
    
    def check_medication_availability(self, medication_name: str, location: str) -> str:
        medication_info = []
        for med in self.medication_db:
            if medication_name.lower() in med["name"].lower():
                medication_info.append(med)
        if not medication_info:
            return json.dumps({
                "error": f"Medication '{medication_name}' not found in database",
                "suggestion": "Please consult with a doctor for alternatives"
            })
        return json.dumps(medication_info, indent=2, ensure_ascii=False)

healthcare_tools = HealthcareTools()


# TRIAGE AGENT
triage_agent = LlmAgent(
    name="triage_agent",
    model=LiteLlm("gemini-2.0-flash"),
    description="Medical triage specialist for assessing urgency and facility type.",
    instruction="""You are a medical triage specialist.
Use the assess_triage tool to produce a structured triage result in JSON.
Please return your reasoning in English.""",
    tools=[healthcare_tools.assess_triage]
)

# NAVIGATION AGENT
navigation_agent = LlmAgent(
    name="navigation_agent",
    model=LiteLlm("gemini-2.0-flash"),
    description="Find nearby healthcare facilities based on location and type.",
    instruction="""You are a navigation expert.
Use find_nearby_facilities to list the closest facilities in JSON.
Briefly explain your results in English.""",
    tools=[healthcare_tools.find_nearby_facilities]
)

# APPOINTMENT AGENT
appointment_agent = LlmAgent(
    name="appointment_agent",
    model=LiteLlm("gemini-2.0-flash"),
    description="Check and book appointment slots at facilities.",
    instruction="""You are an appointment coordinator.
First, call check_available_appointments, then optionally book_appointment if patient_id is specified.
Summarize the process for the patient in English.""",
    tools=[healthcare_tools.check_available_appointments,
           healthcare_tools.book_appointment]
)

# TRANSPORTATION AGENT
transportation_agent = LlmAgent(
    name="transportation_agent",
    model=LiteLlm("gemini-2.0-flash"),
    description="Arrange transportation options for patients.",
    instruction="""You are a transport coordinator.
Use find_transportation and present the best 2-3 options in English.""",
    tools=[healthcare_tools.find_transportation]
)

# MEDICATION AGENT
medication_agent = LlmAgent(
    name="medication_agent",
    model=LiteLlm("gemini-2.0-flash"),
    description="Check medication availability and generic alternatives.",
    instruction="""You are a pharmacy specialist.
Use check_medication_availability and summarize options and prices in English.""",
    tools=[healthcare_tools.check_medication_availability]
)

# FOLLOW-UP AGENT
followup_agent = LlmAgent(
    name="followup_agent",
    model=LiteLlm("gemini-2.0-flash"),
    description="Create reminder and follow-up plan.",
    instruction="""You are a patient care coordinator.
Create reminders (one day before, one hour before) and simple follow-up guidance in English.""",
    tools=[]
)

# COORDINATOR AGENT
coordinator_agent = LlmAgent(
    name="healthcare_coordinator",
    model=LiteLlm("gemini-2.0-flash"),
    description="Orchestrates all healthcare sub-agents.",
    instruction="""You are the main coordinator.
Workflow:
1) Delegate to triage_agent.
2) In parallel, delegate to navigation_agent, appointment_agent, transportation_agent, and medication_agent (if needed).
3) Summarize all results in a clear, step-by-step plan for the patient (English, please).
4) Finally, delegate to followup_agent to generate reminders.

Always communicate clearly, empathetically, and concisely.""",
    sub_agents=[
        triage_agent,
        navigation_agent,
        appointment_agent,
        transportation_agent,
        medication_agent,
        followup_agent
    ]
)

app = App(name="healthcare_navigation_app", root_agent=coordinator_agent)

session_service = InMemorySessionService()
runner = Runner(app=app, session_service=session_service)

print("âœ… Agents and Runner initialized successfully!")


# Diagnostic: Check Runner.run() signature
import inspect
print("Runner.run() signature:")
print(inspect.signature(runner.run))
print("\nRunner.run() parameters:")
print(inspect.signature(runner.run).parameters.keys())


# Quick diagnostic - run this first
print("InMemorySessionService methods:")
print([m for m in dir(session_service) if not m.startswith('_')])


def run_healthcare_query(symptoms: str,
                         location: str,
                         emergency: bool = False,
                         insurance_type: Optional[str] = None,
                         preferred_language: str = "en",
                         session_id: Optional[str] = None,
                         user_id: Optional[str] = None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    if user_id is None:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
    
    # âœ… Get app_name from the app we created earlier
    app_name = app.name  # Should be "healthcare_navigation_app"
    
    user_message = f"""
    Patient Information:
    - Symptoms: {symptoms}
    - Location: {location}
    - Emergency: {emergency}
    - Insurance: {insurance_type or 'None'}
    - Language: {preferred_language}
    
    Please help this patient navigate appropriate healthcare services.
    """
    
    start_time = datetime.now()
    
    from google.genai import types
    
    # âœ… FIX: Create session with app_name parameter
    try:
        session = session_service.create_session_sync(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
        logger.info(f"âœ“ Session created: {session_id}")
    except Exception as e:
        # Session might already exist, try to get it
        try:
            session = session_service.get_session_sync(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id
            )
            logger.info(f"âœ“ Session retrieved: {session_id}")
        except Exception as e2:
            logger.error(f"Failed to create/get session: {str(e2)}")
            return {
                "session_id": session_id,
                "user_id": user_id,
                "response": f"Session error: {str(e2)}",
                "execution_time_seconds": 0,
                "error": str(e2)
            }
    
    # Create the message
    new_message = types.Content(
        role="user",
        parts=[types.Part(text=user_message)]
    )
    
    # Now run with the created session
    try:
        events = []
        logger.info("Starting agent execution...")
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            events.append(event)
            # Log progress every 10 events
            if len(events) % 10 == 0:
                logger.info(f"Processed {len(events)} events...")
        
        logger.info(f"âœ“ Total events received: {len(events)}")
        
    except Exception as e:
        logger.error(f"Error during run: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "session_id": session_id,
            "user_id": user_id,
            "response": f"Runtime error: {str(e)}",
            "execution_time_seconds": (datetime.now() - start_time).total_seconds(),
            "error": str(e)
        }
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Extract response from events - try multiple strategies
    final_response = ""
    
    # Strategy 1: Look for content in last events
    for event in reversed(events[-10:] if len(events) > 10 else events):
        if hasattr(event, 'content'):
            content = event.content
            if hasattr(content, 'parts'):
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        final_response = part.text
                        break
            elif isinstance(content, str) and content:
                final_response = content
                break
        if final_response:
            break
    
    # Strategy 2: Check for 'text' attribute directly
    if not final_response:
        for event in reversed(events):
            if hasattr(event, 'text') and event.text:
                final_response = event.text
                break
    
    # Strategy 3: Look for message content
    if not final_response:
        for event in reversed(events):
            if hasattr(event, 'message'):
                msg = event.message
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, str):
                        final_response = msg.content
                        break
                    elif hasattr(msg.content, 'parts'):
                        for part in msg.content.parts:
                            if hasattr(part, 'text') and part.text:
                                final_response = part.text
                                break
                if final_response:
                    break
    
    # Strategy 4: Convert entire event to string if it's substantial
    if not final_response and len(events) > 0:
        for event in reversed(events):
            event_str = str(event)
            if len(event_str) > 100:  # Has substantial content
                final_response = event_str
                break
    
    if not final_response:
        # Last resort: show event structure for debugging
        event_info = []
        for i, event in enumerate(events[:5]):
            event_info.append(f"Event {i}: {type(event).__name__}")
        
        final_response = f"âš ï¸� Processed {len(events)} events but could not extract text response.\n\nEvent types: {', '.join(event_info)}\n\nPlease check logs for details."
    
    logger.info(f"[{session_id}] âœ“ Completed in {elapsed:.2f}s with {len(events)} events")
    
    return {
        "session_id": session_id,
        "user_id": user_id,
        "response": final_response,
        "event_count": len(events),
        "execution_time_seconds": elapsed
    }

# Example usage
print("=" * 60)
print("STARTING HEALTHCARE NAVIGATION DEMO")
print("=" * 60)

demo_result = run_healthcare_query(
    symptoms="Severe chest pain, shortness of breath, cold sweats",
    location="Sukamaju Village, Bogor",
    emergency=True
)

print("\n" + "=" * 60)
print("HEALTHCARE NAVIGATION RESULT")
print("=" * 60)
print(demo_result["response"])
print("=" * 60)
print(f"Session ID: {demo_result['session_id']}")
print(f"User ID: {demo_result['user_id']}")
print(f"Events processed: {demo_result.get('event_count', 0)}")
print(f"Execution time: {demo_result['execution_time_seconds']:.2f}s")
print("=" * 60)


test_cases = [
    {
        "name": "Emergency - Chest Pain",
        "symptoms": "Severe chest pain, shortness of breath, cold sweats",
        "location": "Sukamaju Village, Bogor",
        "emergency": True,
        "expected_keywords": ["emergency", "hospital", "immediate"]
    },
    {
        "name": "Routine - Common Cold",
        "symptoms": "Cough, mild fever for 2 days",
        "location": "Makmur Village, East Jakarta",
        "emergency": False,
        "expected_keywords": ["clinic", "routine"]
    },
    {
        "name": "Urgent - High Fever",
        "symptoms": "High fever for 3 days, vomiting",
        "location": "Sejahtera Village, Bandung",
        "emergency": False,
        "expected_keywords": ["urgent", "today", "clinic"]
    }
]

results = []
for case in test_cases:
    print("="*60)
    print(f"Test: {case['name']}")
    res = run_healthcare_query(
        symptoms=case["symptoms"],
        location=case["location"],
        emergency=case["emergency"]
    )
    text = res["response"].lower()
    hit = any(kw.lower() in text for kw in case["expected_keywords"])
    results.append({
        "name": case["name"],
        "success": hit,
        "time": res["execution_time_seconds"]
    })
    print(f"Success: {hit}, Time: {res['execution_time_seconds']:.2f}s")

success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
avg_time = sum(r["time"] for r in results) / len(results)

print("\nSummary:")
print(f"Success rate: {success_rate:.1f}%")
print(f"Average response time: {avg_time:.2f}s")

