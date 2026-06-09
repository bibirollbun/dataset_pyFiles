# ============================================================
# INSTALLS (
# ============================================================
!pip install google-generativeai wikipedia aiohttp nest_asyncio



# ============================================================
# IMPORTS
# ============================================================
import google.generativeai as genai
import asyncio
import nest_asyncio
import wikipedia
import json
from datetime import datetime

nest_asyncio.apply()



import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )



import google.generativeai as genai

api_key = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-pro")






# ============================================================
# MEMORY BANK
# ============================================================
class MemoryBank:
    def __init__(self):
        self.journal_path = "health_journal.json"
        try:
            with open(self.journal_path, "r") as f:
                self.data = json.load(f)
        except:
            self.data = {"logs": []}

    def add_log(self, entry):
        self.data["logs"].append(entry)
        with open(self.journal_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_logs(self):
        return self.data["logs"]

memory_bank = MemoryBank()



class SymptomAgent:
    async def triage(self, symptoms):
        prompt = f"""
        A user reports the symptoms: {symptoms}

        *Do NOT provide medical diagnosis.*
        Provide:
        - Possible lifestyle factors
        - Possible risk categories (non-medical)
        - Red flag signs (non-medical)
        - General well-being advice, hydration, rest, etc.
        """
        res = model.generate_content(prompt)
        return res.text



class DailyLogAgent:
    def log_day(self, mood, symptoms, energy):
        entry = {
            "date": str(datetime.now()),
            "mood": mood,
            "symptoms": symptoms,
            "energy": energy
        }
        memory_bank.add_log(entry)
        return "Daily log saved."



class DietAgent:
    async def suggest(self, symptoms):
        prompt = f"""
        User has symptoms: {symptoms}

        Provide:
        - Hydration advice  
        - Gentle foods  
        - Foods to avoid  
        - Lifestyle adjustments  
        
        Must not give medical diagnosis.
        """
        res = model.generate_content(prompt)
        return res.text



class HospitalAgent:
    async def suggest_department(self, symptoms):
        prompt = f"""
        Based on symptoms: {symptoms}

        Suggest the most relevant HOSPITAL DEPARTMENT ONLY
        (e.g., ENT, Gastro, Cardio, Skin, Emergency)

        Do NOT diagnose.
        """
        res = model.generate_content(prompt)
        return res.text



class FAQAgent:
    async def search_faq(self, query):
        try:
            page = wikipedia.summary(query, sentences=3)
            return page
        except:
            return "No Wikipedia summary found. Try a different query."



async def parallel_triage(symptoms):
    s_agent = SymptomAgent()
    h_agent = HospitalAgent()

    sym_task = asyncio.create_task(s_agent.triage(symptoms))
    dep_task = asyncio.create_task(h_agent.suggest_department(symptoms))

    triage, department = await asyncio.gather(sym_task, dep_task)
    return triage, department



class Orchestrator:
    def __init__(self):
        self.symptom_agent = SymptomAgent()
        self.log_agent = DailyLogAgent()
        self.diet_agent = DietAgent()
        self.faq_agent = FAQAgent()

    async def run(self, symptoms, mood="Neutral", energy="Average"):
        memory_bank.add_log({
            "date": str(datetime.now()),
            "event": "symptom_check",
            "symptoms": symptoms
        })

        triage, department = await parallel_triage(symptoms)
        diet = await self.diet_agent.suggest(symptoms)

        return {
            "symptom_analysis": triage,
            "recommended_department": department,
            "diet_lifestyle": diet,
            "user_journal": memory_bank.get_logs()
        }



output = await orch.run(symptoms)

print("ğŸ©º Symptom Summary:")
print(output["symptom_analysis"].split("\n")[0])

print("\nğŸ�¥ Recommended Department:")
print(output["recommended_department"].strip())

print("\nğŸ¥— Diet & Lifestyle Suggestion:")
print(output["diet_lifestyle"].split('\n')[0])





