from IPython.display import Image, display

# Replace this string with the EXACT path you copied from Step 1
image_path = "/kaggle/input/architecture-diagram/architecture_diagram.png"

try:
    display(Image(filename=image_path))
    print("âœ… Image loaded successfully!")
except FileNotFoundError:
    print(f"â�Œ Error: Could not find file at {image_path}")
    print("Please check the 'Path Finder' step above again.")


!pip install -U google-generativeai


# ============================================================
# âš™ï¸� VOLTGUARD - SYSTEM SETUP & MODEL LOADING
# ============================================================

!pip install -q google-generativeai pillow requests python-dotenv

import os
import requests
from PIL import Image
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION (Your Snippet) ---
print("âš¡ Initializing VOLTGUARD Neural Core...")

# âœ… SECURE: Use Kaggle Secrets or .env
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
except:
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("â�Œ ERROR: GOOGLE_API_KEY not found!")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
print("âœ… API Configured Successfully")

# --- VISION SERVICE ---

class GeminiVisionService:
    """
    Singleton class to manage the Gemini Vision Model.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiVisionService, cls).__new__(cls)
            # Using 'gemini-1.5-flash-latest' for speed/efficiency in agent workflows
            # You can also use 'gemini-pro-vision' or 'gemini-1.5-pro'
            cls._instance.model_name = "gemini-2.5-flash" 
            cls._instance.model = genai.GenerativeModel(cls._instance.model_name)
            print(f"   ğŸ”„ Connected to {cls._instance.model_name}...")
            print("   âœ… Neural Core Online.")
        return cls._instance

    def analyze(self, image_url, prompt_text):
        """
        Generic analysis function used by all agents
        """
        try:
            # Load Image from URL
            image = Image.open(requests.get(image_url, stream=True).raw)
            
            # Generate Content
            response = self.model.generate_content([prompt_text, image])
            
            return response.text
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

# Initialize the Model Once
vision_core = GeminiVisionService()


from dataclasses import dataclass
from typing import List
from enum import Enum

# ==========================================
# ğŸ—‚ï¸� VOLTGUARD DATA ARCHITECTURE
# ==========================================

class FaultType(str, Enum):
    INSULATOR_BREAK = "Broken Insulator"
    RUST_CORROSION = "Rust/Corrosion"
    THERMAL_HOTSPOT = "Thermal Hotspot"
    BIOLOGICAL_INTRUSION = "Bird Nest/Wildlife"
    NORMAL = "Normal Operation"

class SeverityLevel(int, Enum):
    MINOR = 1      # Monitor
    MODERATE = 2   # Schedule Maintenance
    CRITICAL = 3   # Immediate Intervention
    EMERGENCY = 4  # Shutdown Required

@dataclass
class GridAssetReport:
    """Input Model: Incoming image from drone/camera"""
    asset_id: str
    location: str
    image_url: str
    sensor_type: str # 'Optical' or 'Thermal'

@dataclass
class FaultAssessment:
    """Output Model: Analysis results"""
    fault_type: FaultType
    severity: SeverityLevel
    description: str
    confidence: float

@dataclass
class MaintenancePlan:
    """Output Model: Action plan"""
    crew_type: str 
    equipment_needed: List[str]
    estimated_downtime: str
    priority_status: str

print("âœ… Data Models & Enums Configured")


import time
import json
import re


# ==========================================
# ğŸ¤– VOLTGUARD AGENT DEFINITIONS
# ==========================================

class BaseAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.core = vision_core # Link to Gemini Singleton
        print(f"   âœ“ {name} initialized ({role})")


class VisualInspectionAgent(BaseAgent):
    """
    ğŸ‘�ï¸� Primary Agent: Uses Gemini 1.5 Flash in JSON Mode for precision.
    """
    def __init__(self):
        super().__init__("VisualSentinel", "Fault Detection")
        
        # Configure the model to output JSON specifically
        self.generation_config = {
            "temperature": 0.2,
            "response_mime_type": "application/json"
        }

    def inspect_asset(self, report: GridAssetReport) -> FaultAssessment:
        print(f"   ...{self.name} analyzing image feed...")
        
        # 1. Precise Prompt for JSON Output
        prompt = """
        Analyze this power grid infrastructure image for faults.
        Return ONLY a JSON object with this exact schema:
        {
            "detected_fault": "Normal" | "Broken Insulator" | "Rust" | "Bird Nest" | "Thermal Hotspot",
            "severity_score": 1 to 4,
            "description": "Short factual summary of observations."
        }
        
        Rules:
        - If the equipment is healthy, set fault to "Normal" and severity to 1.
        - Be extremely strict. Do not hallucinate faults.
        """

        try:
            # 2. Call Gemini with Image & JSON Config
            # We bypass the generic core.analyze to pass specific config
            image = Image.open(requests.get(report.image_url, stream=True).raw)
            response = self.core.model.generate_content(
                [prompt, image], 
                generation_config=self.generation_config
            )
            
            # 3. Parse JSON (with robust cleanup)
            data = self._clean_and_parse_json(response.text)
            
            # 4. Map JSON to your Enums
            fault_str = data.get("detected_fault", "Normal").upper()
            
            if "BROKEN" in fault_str:
                fault_enum = FaultType.INSULATOR_BREAK
                severity = SeverityLevel.CRITICAL
            elif "RUST" in fault_str:
                fault_enum = FaultType.RUST_CORROSION
                severity = SeverityLevel.MODERATE
            elif "NEST" in fault_str or "BIRD" in fault_str:
                fault_enum = FaultType.BIOLOGICAL_INTRUSION
                severity = SeverityLevel.MODERATE
            elif "THERMAL" in fault_str or "HOTSPOT" in fault_str:
                fault_enum = FaultType.THERMAL_HOTSPOT
                severity = SeverityLevel.CRITICAL
            else:
                fault_enum = FaultType.NORMAL
                severity = SeverityLevel.MINOR

            return FaultAssessment(
                fault_type=fault_enum,
                severity=severity,
                description=data.get("description", "Analysis complete."),
                confidence=0.95
            )

        except Exception as e:
            print(f"   âš ï¸� Analysis Failed: {e}")
            return FaultAssessment(
                fault_type=FaultType.NORMAL,
                severity=SeverityLevel.MINOR,
                description=f"Error parsing analysis: {str(e)}",
                confidence=0.0
            )

    def _clean_and_parse_json(self, text):
        """Helper to handle cases where LLM wraps JSON in markdown"""
        try:
            # Remove ```json ... ``` wrappers if present
            clean_text = re.sub(r'```json\s*|\s*```', '', text)
            return json.loads(clean_text)
        except json.JSONDecodeError:
            # Fallback: simple text return if JSON fails
            return {"detected_fault": "Normal", "description": text}
class MaintenanceCrewAgent(BaseAgent):
    """
    ğŸ› ï¸� Logistics Agent: Plans the fix based on the assessment.
    """
    def __init__(self):
        super().__init__("GridMechanic", "Resource Allocation")

    def create_plan(self, assessment: FaultAssessment) -> MaintenancePlan:
        # Logic to assign resources
        if assessment.fault_type == FaultType.THERMAL_HOTSPOT:
            return MaintenancePlan(
                crew_type="Specialist High-Voltage Team",
                equipment_needed=["IR Camera", "Replacement Breaker", "Bucket Truck"],
                estimated_downtime="4 Hours",
                priority_status="IMMEDIATE"
            )
        elif assessment.fault_type == FaultType.INSULATOR_BREAK:
             return MaintenancePlan(
                crew_type="Line Maintenance Crew",
                equipment_needed=["Replacement Insulator String", "Climbing Gear"],
                estimated_downtime="3 Hours",
                priority_status="HIGH PRIORITY"
            )
        elif assessment.fault_type == FaultType.BIOLOGICAL_INTRUSION:
            return MaintenancePlan(
                crew_type="Environmental Control",
                equipment_needed=["Insulated Pole", "Wildlife Safe Containment"],
                estimated_downtime="1 Hour",
                priority_status="SCHEDULED"
            )
        elif assessment.fault_type == FaultType.RUST_CORROSION:
             return MaintenancePlan(
                crew_type="Structural Inspection Team",
                equipment_needed=["Anti-Corrosion Treatment", "Drone Scanner"],
                estimated_downtime="2 Hours",
                priority_status="ROUTINE"
            )
        else:
            return MaintenancePlan(
                crew_type="None",
                equipment_needed=["None"],
                estimated_downtime="0 Hours",
                priority_status="NO ACTION"
            )

print("âœ… Agents Ready.")


# ==========================================================
# ğŸ§  ORCHESTRATOR â€“ THE BRAIN OF VOLTGUARD
# ==========================================================

class VoltGuardOrchestrator:
    def __init__(self):
        print("ğŸš€ Initializing VOLTGUARD Multi-Agent System...\n")
        self.inspector = VisualInspectionAgent()
        self.planner = MaintenanceCrewAgent()
        print("\n" + "="*60)
        print("âœ… System Online.")

    def process_asset(self, report: GridAssetReport):
        print(f"\n{'='*60}")
        print(f"âš¡ PROCESSING ASSET: {report.asset_id}")
        print(f"{'='*60}")
        print(f"ğŸ“� Location: {report.location}")
        
        start_time = time.time()

        # STEP 1: VISUAL INSPECTION (Gemini Vision)
        print("\nğŸ‘�ï¸� Step 1: Visual Inspection (Gemini Vision)...")
        assessment = self.inspector.inspect_asset(report)
        print(f"   âœ“ Detection: {assessment.fault_type.value}")
        print(f"   âœ“ Severity: {assessment.severity.name}")
        print(f"   âœ“ Analysis: {assessment.description[:120]}...")

        # STEP 2: MAINTENANCE PLANNING
        print("\nğŸ› ï¸� Step 2: Maintenance Logistics...")
        plan = self.planner.create_plan(assessment)
        print(f"   âœ“ Crew Dispatched: {plan.crew_type}")
        print(f"   âœ“ Equipment: {plan.equipment_needed}")

        total_time = round(time.time() - start_time, 2)
        
        self.display_dashboard(report, assessment, plan, total_time)

    def display_dashboard(self, report, assessment, plan, duration):
        print("\n" + "="*70)
        print("ğŸ“Š VOLTGUARD INTELLIGENCE DASHBOARD")
        print("="*70)
        
        print(f"ğŸ“¸ ASSET ID: {report.asset_id} | SENSOR: {report.sensor_type}")
        print("-" * 70)
        print(f"DETECTED FAULT   : {assessment.fault_type.value.upper()}")
        print(f"SEVERITY LEVEL   : {assessment.severity.value}/4")
        print(f"RECOMMENDED CREW : {plan.crew_type.upper()}")
        print(f"PRIORITY STATUS  : {plan.priority_status}")
        print(f"PROCESSING TIME  : {duration}s")
        print("="*70 + "\n")

# Initialize
sentinel = VoltGuardOrchestrator()


from PIL import Image 




# Load the image
url = "https://raw.githubusercontent.com/naimul011/Electric/refs/heads/main/broken2.JPG"
image = Image.open(requests.get(url, stream=True).raw)
image


# Load the image
url = "https://raw.githubusercontent.com/naimul011/Electric/refs/heads/main/thermal.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image


# Load the image
url = "https://raw.githubusercontent.com/naimul011/Electric/refs/heads/main/nest.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image


# ==========================================
# ğŸ§ª TEST SCENARIOS
# ==========================================

# Scenario 1: Broken Insulator
report_broken = GridAssetReport(
    asset_id="TOWER-Alpha-01",
    location="Sector 7, North Grid",
    image_url="https://raw.githubusercontent.com/naimul011/Electric/refs/heads/main/broken2.JPG",
    sensor_type="Optical"
)

# Scenario 2: Thermal Hotspot
report_thermal = GridAssetReport(
    asset_id="SUBSTATION-X9",
    location="Downtown Transformer Bank",
    image_url="https://raw.githubusercontent.com/naimul011/Electric/refs/heads/main/thermal.jpeg",
    sensor_type="Thermal"
)

# Scenario 3: Bird Nest
report_nest = GridAssetReport(
    asset_id="POLE-Rural-44",
    location="Route 66 Corridor",
    image_url="https://raw.githubusercontent.com/naimul011/Electric/refs/heads/main/nest.jpg",
    sensor_type="Optical"
)

# EXECUTE
sentinel.process_asset(report_broken)
time.sleep(1) # Pause to respect API rate limits
sentinel.process_asset(report_thermal)
time.sleep(1)
sentinel.process_asset(report_nest)

