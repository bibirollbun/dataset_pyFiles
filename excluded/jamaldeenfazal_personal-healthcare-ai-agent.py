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


# Step 1: Install library (uncomment if needed)
# !pip install -q google-generativeai

# Step 2: Imports
import os
import time
import json
import re
from datetime import datetime
import google.generativeai as genai

# Step 3: Set API key (Kaggle Secrets)
try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = API_KEY
    genai.configure(api_key=API_KEY)
    print("âœ… Gemini API key loaded and configured.")
except Exception as e:
    print(f"ğŸ”‘ Error loading API key: {e}")
    API_KEY = None

# Step 3.5: List available models (for debugging)
def list_available_models():
    """Check which models are available"""
    try:
        models = genai.list_models()
        print("\nğŸ“‹ Available Models:")
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"  - {model.name}")
        return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    except Exception as e:
        print(f"â�Œ Could not list models: {e}")
        return []

# Step 4: Memory class
class HealthMemory:
    def __init__(self):
        self.medications = []
        self.symptoms = []
        self.metrics = []
        self.conversations = []

    def add_medication(self, name, dosage, frequency="Daily", time_str="8:00 AM"):
        self.medications.append({
            "name": name,
            "dosage": dosage,
            "frequency": frequency,
            "time": time_str,
            "added": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

    def add_symptom(self, description):
        self.symptoms.append({
            "description": description,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

    def add_metric(self, metric_type, value, unit):
        self.metrics.append({
            "type": metric_type,
            "value": value,
            "unit": unit,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

# Step 5: Health Agent
class HealthAgent:
    def __init__(self, model_name="gemini-pro"):
        """
        Initialize with correct model name.
        Common options:
        - gemini-pro (standard)
        - gemini-1.5-pro (advanced)
        - models/gemini-pro (full path)
        """
        self.model_name = model_name
        self.memory = HealthMemory()
        self.last_request_time = 0
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Gemini model with error handling"""
        try:
            # Try different model name formats (updated for Gemini 2.x)
            model_variations = [
                "models/gemini-2.5-flash",
                "models/gemini-2.5-pro",
                "models/gemini-2.0-flash-exp",
                "models/gemini-flash-latest",
                "models/gemini-pro-latest",
                self.model_name,
                f"models/{self.model_name}"
            ]
            
            for model_name in model_variations:
                try:
                    self.model = genai.GenerativeModel(model_name=model_name)
                    # Test with a simple generation
                    test = self.model.generate_content("Hi")
                    print(f"âœ… Healthcare AI Agent Ready! Using model: {model_name}")
                    self.model_name = model_name
                    return
                except Exception as e:
                    continue
            
            print("â�Œ Could not initialize any model. Please check available models.")
            
        except Exception as e:
            print(f"â�Œ Error initializing model: {e}")

    def _wait_rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < 3:
            time.sleep(3 - elapsed)
        self.last_request_time = time.time()

    def _call_gemini(self, message):
        """Call Gemini API with proper error handling"""
        if self.model is None:
            return "â�Œ Model not initialized. Please check your API key and model availability."
        
        self._wait_rate_limit()
        
        meds = json.dumps(self.memory.medications[-3:], indent=2) if self.memory.medications else "None"
        symptoms = json.dumps(self.memory.symptoms[-3:], indent=2) if self.memory.symptoms else "None"
        metrics = json.dumps(self.memory.metrics[-5:], indent=2) if self.memory.metrics else "None"

        system_prompt = f"""You are a helpful Healthcare AI Assistant.

Current patient data:
- Medications: {meds}
- Recent symptoms: {symptoms}
- Recent metrics: {metrics}

Guidelines:
- Be empathetic and supportive
- Provide practical health guidance
- Always include disclaimer to consult healthcare professionals
- Keep responses concise (2-3 paragraphs max)
- Current date: {datetime.now().strftime('%Y-%m-%d')}

User message: {message}"""

        try:
            response = self.model.generate_content(
                system_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                    top_p=0.95
                )
            )
            
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                return f"â�Œ Model '{self.model_name}' not found. Try: gemini-pro or models/gemini-pro"
            elif "429" in error_msg:
                return "â�Œ Rate limit exceeded. Please wait a moment and try again."
            else:
                return f"â�Œ Error: {error_msg}"

    def chat(self, message):
        """Main chat interface"""
        print(f"\nğŸ§‘ YOU: {message}")
        msg_lower = message.lower()

        # Medication detection (improved)
        med_pattern = r'add medication\s+(\w+)\s+(\d+\s*mg)'
        med_match = re.search(med_pattern, message, re.IGNORECASE)
        if med_match:
            name = med_match.group(1)
            dosage = med_match.group(2)
            self.memory.add_medication(name, dosage)
            print(f"âœ… Added medication: {name} {dosage}")

        # Blood pressure detection
        bp_match = re.search(r'(\d{2,3})[/\s](\d{2,3})', message)
        if bp_match:
            bp_value = f"{bp_match.group(1)}/{bp_match.group(2)}"
            self.memory.add_metric("Blood Pressure", bp_value, "mmHg")
            print(f"âœ… Logged blood pressure: {bp_value} mmHg")

        # Weight detection
        weight_match = re.search(r'(\d+\.?\d*)\s*(kg|lbs)', message.lower())
        if weight_match:
            weight_value = weight_match.group(1)
            weight_unit = weight_match.group(2)
            self.memory.add_metric("Weight", weight_value, weight_unit)
            print(f"âœ… Logged weight: {weight_value} {weight_unit}")

        # Symptom detection
        symptom_keywords = ["fever", "cough", "pain", "headache", "sick", "symptom", "ache", "sore"]
        if any(word in msg_lower for word in symptom_keywords):
            self.memory.add_symptom(message)
            print(f"âœ… Logged symptom")

        # Call Gemini AI
        response = self._call_gemini(message)
        self.memory.conversations.append({
            "user": message, 
            "agent": response, 
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        
        print(f"ğŸ¤– AGENT: {response}")
        return response

    def show_data(self):
        """Display health data summary"""
        print("\nğŸ“Š Health Data Summary")
        print("="*50)
        
        print("\nğŸ’Š Medications:")
        if self.memory.medications:
            for m in self.memory.medications:
                print(f"  - {m['name']} {m['dosage']} ({m['frequency']} at {m['time']})")
        else:
            print("  - None recorded")
        
        print("\nğŸ¤’ Symptoms:")
        if self.memory.symptoms:
            for s in self.memory.symptoms:
                print(f"  - {s['description'][:50]}... at {s['date']}")
        else:
            print("  - None recorded")
        
        print("\nğŸ“ˆ Metrics:")
        if self.memory.metrics:
            for met in self.memory.metrics:
                print(f"  - {met['type']}: {met['value']} {met['unit']} at {met['date']}")
        else:
            print("  - None recorded")
        
        print("="*50)

# Step 6: Check available models first (optional)
print("\nğŸ”� Checking available models...")
available = list_available_models()

# Step 7: Initialize agent
print("\nğŸš€ Initializing Healthcare AI Agent...")
agent = HealthAgent()  # Will auto-detect correct model

# Step 8: Example usage
if agent.model:  # Only run if model initialized successfully
    print("\n" + "="*50)
    print("ğŸ�¥ Healthcare AI Agent - Demo")
    print("="*50)

    agent.chat("I have a mild headache and cough")
    agent.chat("Add medication Paracetamol 500mg")
    agent.chat("Log my blood pressure 120/80")
    agent.chat("Give me advice to improve my sleep")
    agent.show_data()
else:
    print("\nâ�Œ Could not start agent. Check your API key and model availability.")

