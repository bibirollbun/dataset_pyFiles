# @title 1. Install Dependencies
%pip install -q google-generativeai python-dotenv requests


# @title 2. Configure API Key
import os
from getpass import getpass

# Prompt for API Key securely
GEMINI_API_KEY = getpass("Enter your Google Gemini API Key: ")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["SENTINEL_API_KEY"] = "mock_key" # We use mock data for Sentinel


# @title 3. Define Services (Memory, Sentinel, Gemini)
import time
import json
import logging
import google.generativeai as genai

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger('GreenPulse')

# --- Memory Service ---
class MemoryBank:
    def __init__(self):
        self.alerts = []

    def add_alert(self, alert):
        alert['timestamp'] = time.time()
        self.alerts.append(alert)

    def get_recent_alerts(self, location, seconds=3600):
        current_time = time.time()
        return [
            a for a in self.alerts
            if a['location'] == location and (current_time - a['timestamp']) < seconds
        ]

# --- Sentinel Service (Mock) ---
class SentinelService:
    def fetch_data(self, location):
        # Simulating satellite data retrieval
        return {
            "location": location,
            "image_data": "mock_binary_image_data",
            "metadata": {"cloud_coverage": 10.5, "timestamp": "2023-10-27T10:00:00Z"}
        }

# --- Gemini Service ---
class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")

    def analyze_image(self, image_data, prompt="Analyze this"):
        if not self.api_key:
            return self._get_mock_analysis()

        try:
            genai.configure(api_key=self.api_key)
            # Using 2.0 Flash as discovered in our testing
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            full_prompt = f"{prompt}. Return a JSON object with keys: 'flood_risk_score' (float 0-1), 'deforestation_detected' (bool), and 'reason' (string). Output ONLY JSON."
            
            response = model.generate_content(full_prompt)
            text = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(text)
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return self._get_mock_analysis()

    def _get_mock_analysis(self):
        return {
            "flood_risk_score": 0.85,
            "deforestation_detected": True,
            "reason": "Mock analysis: High flood risk detected."
        }


# @title 4. Define Agents

class BaseAgent:
    def __init__(self, name):
        self.name = name

class DataCollectorAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataCollector")
        self.sentinel_service = SentinelService()

    def run(self, location):
        print(f"[{self.name}] Fetching data for {location}...")
        return self.sentinel_service.fetch_data(location)

class AnalyzerAgent(BaseAgent):
    def __init__(self, memory_bank):
        super().__init__("Analyzer")
        self.gemini_service = GeminiService()
        self.memory_bank = memory_bank

    def run(self, data):
        if not data: return None
        print(f"[{self.name}] Analyzing data...")
        
        # A2A Protocol: Check memory
        if self.memory_bank.get_recent_alerts(data['location']):
            print(f"[{self.name}] NOTICE: Recent alerts found. Adjusting sensitivity.")
            
        analysis = self.gemini_service.analyze_image(
            data['image_data'], 
            prompt=f"Analyze environmental situation for {data['location']} based on satellite data"
        )
        analysis['location'] = data['location']
        print(f"[{self.name}] Analysis complete: Risk Score {analysis.get('flood_risk_score')}")
        return analysis

class AlertAgent(BaseAgent):
    def __init__(self, memory_bank):
        super().__init__("Alert")
        self.memory_bank = memory_bank
        self.threshold = 0.7

    def run(self, analysis):
        if not analysis: return
        
        score = analysis.get('flood_risk_score', 0)
        if score > self.threshold:
            msg = f"HIGH RISK ALERT: Flood risk at {analysis['location']}. Reason: {analysis['reason']}"
            print(f"\n>>> [{self.name}] SENDING ALERT: {msg} <<<\n")
            self.memory_bank.add_alert({"message": msg, "location": analysis['location'], "type": "flood"})
        else:
            print(f"[{self.name}] Risk normal ({score}). No alert.")


# @title 5. Orchestrator & Main Execution

class Orchestrator:
    def __init__(self):
        self.memory = MemoryBank()
        self.collector = DataCollectorAgent()
        self.analyzer = AnalyzerAgent(self.memory)
        self.alert = AlertAgent(self.memory)

    def run_cycle(self, location):
        print(f"\n--- Monitoring {location} ---")
        data = self.collector.run(location)
        if data:
            analysis = self.analyzer.run(data)
            if analysis:
                self.alert.run(analysis)
        print("--- Cycle Complete ---")

# --- Run Simulation ---
orchestrator = Orchestrator()
locations = ["Assam", "Bihar", "Uttar Pradesh", "West Bengal", "Kerala"]

print("Starting Green Pulse Sentinel Simulation...")
for loc in locations:
    orchestrator.run_cycle(loc)
    time.sleep(1)

