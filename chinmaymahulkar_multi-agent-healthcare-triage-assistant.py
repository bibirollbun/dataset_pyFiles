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


import re
import json

# --- Simulated Tools ---

def Knowledge_Lookup_Tool(query: str) -> str:
    """Simulates a RAG call to a non-diagnostic health knowledge base."""
    if "fever" in query.lower() and "cough" in query.lower():
        return "CDC Guidelines: Common symptoms of viral infections (like the flu or common cold) include fever, cough, fatigue, and body aches. Rest, hydration, and over-the-counter medication are typically recommended for mild cases. Consult a doctor if breathing is difficult or fever is persistent."
    elif "severe pain" in query.lower() or "chest" in query.lower():
        return "Emergency Protocol: Severe, sudden pain, especially in the chest or abdomen, requires immediate medical attention (ER visit). Do not wait for self-care to be effective."
    else:
        return "General Advice: For mild, non-specific symptoms, monitor your condition, stay hydrated, and rest. Use an internal symptom checker for further guidance."

def Risk_Scoring_Tool(symptoms: list, severity: str) -> str:
    """Internal function for the Triage Agent to assign a risk score."""
    if "severe" in severity.lower() or "emergency" in symptoms:
        return "High Risk: Seek immediate emergency medical care."
    elif len(symptoms) >= 3 and severity.lower() in ["moderate", "persistent"]:
        return "Moderate Risk: Schedule a non-emergency appointment with a primary care physician within 1-2 days."
    else:
        return "Low Risk: Home care and monitoring are recommended."


class IntakeAgent:
    """The Coordinator Agent: Gathers input and starts the workflow."""
    def greet_and_get_query(self, user_input: str) -> str:
        print("Intake Agent: Gathering initial data...")
        # Simulate a multi-turn conversation where a simple input is gathered
        return user_input

class AnalyzerAgent:
    """Extracts structured entities from the raw text input."""
    def extract_entities(self, raw_text: str) -> dict:
        print("Analyzer Agent: Extracting key medical entities...")
        
        # Simple regex and keyword-based extraction (LLM would be used here)
        symptoms = re.findall(r'(fever|cough|headache|pain|nausea|emergency)', raw_text.lower())
        severity = "moderate" if "bad" in raw_text.lower() or "persistent" in raw_text.lower() else "mild"
        
        # Crisis Guardrail
        if "severe" in raw_text.lower() or "life-threatening" in raw_text.lower():
            symptoms.append("emergency")
            severity = "severe"
            
        # Deduplicate symptoms and return structured data
        return {
            "symptoms": list(set(symptoms)),
            "severity": severity,
            "query_for_knowledge": raw_text # Pass the full query for RAG context
        }

class KnowledgeAgent:
    """Retrieves relevant health information using a tool."""
    def retrieve_knowledge(self, structured_data: dict) -> str:
        print("Knowledge Agent: Consulting knowledge base...")
        query = " ".join(structured_data['symptoms']) + f" and severity {structured_data['severity']}"
        knowledge = Knowledge_Lookup_Tool(query)
        return knowledge

class TriageAgent:
    """Assesses risk and generates the final, safe response."""
    def generate_response(self, structured_data: dict, knowledge: str) -> str:
        print("Triage Agent: Assessing risk and generating response...")
        
        # 1. Determine Risk
        risk_level = Risk_Scoring_Tool(structured_data['symptoms'], structured_data['severity'])
        
        # 2. Craft Final Message
        final_message = f"""
        ***
        **Triage Assistant Result**

        **Risk Assessment:** {risk_level}

        **Extracted Symptoms:** {', '.join(structured_data['symptoms']).title() if structured_data['symptoms'] else 'None reported'}
        
        **Relevant Health Guidance (Non-Diagnostic):**
        {knowledge}

        **NEXT STEPS (IMPORTANT DISCLAIMER):**
        {risk_level} - Based on your description, we recommend the following action.
        
        ***

        **⚠️ CRITICAL SAFETY NOTICE:** This AI is an assistant and **NOT** a medical professional. This information is **NOT A DIAGNOSIS** and cannot replace a consultation with a licensed doctor. Always seek emergency medical help (call 911 or equivalent) for severe or life-threatening symptoms.
        """
        
        return final_message


def run_healthcare_triage_system(user_input: str):
    """Main function to run the multi-agent system."""
    
    # 1. Initialize Agents
    intake = IntakeAgent()
    analyzer = AnalyzerAgent()
    knowledge = KnowledgeAgent()
    triage = TriageAgent()
    
    print(f"\n--- New Patient Session ---\nUser Input: '{user_input}'\n")

    # 2. Intake
    raw_query = intake.greet_and_get_query(user_input)
    
    # 3. Analyze
    structured_data = analyzer.extract_entities(raw_query)
    
    # Check for immediate crisis and short-circuit the flow
    if "emergency" in structured_data['symptoms']:
        print("\n*** SAFETY ESCALATION TRIGGERED ***")
        return triage.generate_response(structured_data, "Immediate emergency symptoms detected.")

    # 4. Knowledge Retrieval
    retrieved_knowledge = knowledge.retrieve_knowledge(structured_data)
    
    # 5. Triage and Response
    final_response = triage.generate_response(structured_data, retrieved_knowledge)
    
    print("\n--- Final Agent Response ---\n")
    print(final_response)
    
    return final_response


print("## Test Case 1: Low Risk (Mild Symptoms) ##")
run_healthcare_triage_system("I have a mild headache and a runny nose for about 2 days.")

print("\n\n" + "="*80 + "\n\n")

print("## Test Case 2: Moderate Risk (Multiple Symptoms) ##")
run_healthcare_triage_system("I have a persistent fever, a bad cough, and I feel very fatigued. It started three days ago.")

print("\n\n" + "="*80 + "\n\n")

print("## Test Case 3: High Risk (Emergency Trigger) ##")
run_healthcare_triage_system("I have severe, sudden chest pain and I'm having difficulty breathing. I need life-threatening help!")

