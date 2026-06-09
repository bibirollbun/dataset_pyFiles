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


# Install required packages
!pip install fastapi uvicorn pyngrok sqlalchemy python-dotenv openai gradio flask nest-asyncio

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import uvicorn
import sqlite3
import json
from typing import List, Optional, Dict, Any
import logging
import gradio as gr


# Create FastAPI app
app = FastAPI(title="Smart Health Companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
def init_db():
    conn = sqlite3.connect('health_companion.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            symptom_analysis TEXT,
            confidence_score REAL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS symptom_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptoms TEXT NOT NULL,
            severity TEXT NOT NULL,
            duration TEXT NOT NULL,
            user_age INTEGER,
            existing_conditions TEXT,
            analysis_result TEXT NOT NULL,
            recommendations TEXT NOT NULL,
            urgency_level TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# AI Service
class AIService:
    def __init__(self):
        self.symptom_keywords = {
            'headache': ['headache', 'head pain', 'migraine'],
            'fever': ['fever', 'temperature', 'hot'],
            'cough': ['cough', 'coughing'],
            'fatigue': ['tired', 'fatigue', 'exhausted'],
            'nausea': ['nausea', 'sick to stomach', 'queasy']
        }
    
    async def get_health_response(self, user_message: str):
        symptom_analysis = self._analyze_symptoms_in_text(user_message)
        response = self._generate_health_response(user_message)
        
        return {
            "response": response,
            "symptom_analysis": symptom_analysis,
            "confidence_score": 0.8
        }
    
    def _generate_health_response(self, user_message: str) -> str:
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ['headache', 'head pain']):
            return "I understand you're experiencing a headache. For general relief, try resting in a quiet room, staying hydrated, and applying a cool compress. If the headache is severe, persistent, or accompanied by other symptoms like vision changes or fever, please consult a healthcare provider."
        
        elif any(word in user_lower for word in ['fever', 'temperature']):
            return "For fever management, ensure you stay well-hydrated and get plenty of rest. You can use over-the-counter fever reducers as directed. If the fever is high (above 103Â°F/39.4Â°C), lasts more than 3 days, or is accompanied by other concerning symptoms, please seek medical attention."
        
        elif any(word in user_lower for word in ['cough', 'coughing']):
            return "For cough relief, try staying hydrated, using a humidifier, and avoiding irritants like smoke. Honey in warm tea can help soothe throat irritation. If the cough is severe, persistent for more than a week, or accompanied by breathing difficulties, please consult a healthcare provider."
        
        elif any(word in user_lower for word in ['pain', 'hurt']):
            return "I understand you're experiencing pain. For general pain management, rest the affected area and consider over-the-counter pain relief as directed. If the pain is severe, sudden, or accompanied by other symptoms like swelling or fever, please seek medical attention immediately."
        
        else:
            return "Thank you for sharing your health concerns. I can provide general wellness information, but please remember I'm not a medical professional. For personalized medical advice, diagnosis, or treatment, it's important to consult with a qualified healthcare provider. Is there specific wellness information I can help you with today?"
    
    def _analyze_symptoms_in_text(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        
        detected_symptoms = []
        for symptom, keywords in self.symptom_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_symptoms.append(symptom)
        
        return {
            "detected_symptoms": detected_symptoms,
            "symptom_count": len(detected_symptoms)
        }

# Symptom Analyzer
class SymptomAnalyzer:
    def analyze(self, symptoms: List[str], severity: str, duration: str, 
                user_age: Optional[int] = None, 
                existing_conditions: Optional[List[str]] = None):
        
        urgency_level = self._calculate_urgency(symptoms, severity, duration, user_age)
        analysis = self._generate_analysis(symptoms, severity, duration, urgency_level)
        recommendations = self._get_recommendations(urgency_level, user_age, existing_conditions)
        
        return {
            "analysis": analysis,
            "recommendations": recommendations,
            "urgency_level": urgency_level
        }
    
    def _calculate_urgency(self, symptoms: List[str], severity: str, 
                          duration: str, user_age: Optional[int]) -> str:
        symptom_database = {
            'headache': {'mild': 'low', 'moderate': 'medium', 'severe': 'high'},
            'fever': {'mild': 'medium', 'moderate': 'high', 'severe': 'emergency'},
            'cough': {'mild': 'low', 'moderate': 'medium', 'severe': 'high'}
        }
        
        urgency_scores = []
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for known_symptom, severity_map in symptom_database.items():
                if known_symptom in symptom_lower:
                    urgency_level = severity_map.get(severity, 'medium')
                    urgency_scores.append(self._urgency_to_score(urgency_level))
                    break
            else:
                urgency_scores.append(self._urgency_to_score('medium'))
        
        duration_score = self._duration_to_score(duration)
        age_score = self._age_to_score(user_age)
        
        if not urgency_scores:
            final_score = 2
        else:
            final_score = max(urgency_scores) + duration_score + age_score
        
        if final_score >= 4:
            return 'emergency'
        elif final_score >= 3:
            return 'high'
        elif final_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _urgency_to_score(self, urgency: str) -> int:
        mapping = {'low': 1, 'medium': 2, 'high': 3, 'emergency': 4}
        return mapping.get(urgency, 2)
    
    def _duration_to_score(self, duration: str) -> int:
        mapping = {'hours': 0, 'days': 1, 'weeks': 2, 'months': 2}
        return mapping.get(duration, 1)
    
    def _age_to_score(self, age: Optional[int]) -> int:
        if age is None: return 0
        if age < 5 or age > 65: return 1
        return 0
    
    def _generate_analysis(self, symptoms: List[str], severity: str, 
                          duration: str, urgency_level: str) -> str:
        analysis_parts = []
        analysis_parts.append(f"Based on your reported symptoms ({', '.join(symptoms)}), ")
        analysis_parts.append(f"which you describe as {severity} in severity ")
        analysis_parts.append(f"and lasting for {duration}, ")
        
        if urgency_level == 'low':
            analysis_parts.append("this appears to be a minor issue that may resolve with self-care.")
        elif urgency_level == 'medium':
            analysis_parts.append("this may require monitoring and possibly professional consultation if symptoms persist.")
        elif urgency_level == 'high':
            analysis_parts.append("this appears to be a significant concern that should be evaluated by a healthcare provider.")
        else:
            analysis_parts.append("this appears to be a serious situation that requires immediate medical attention.")
        
        analysis_parts.append("\n\nRemember: This is general information and not medical advice. Always consult with healthcare professionals for proper evaluation.")
        
        return "".join(analysis_parts)
    
    def _get_recommendations(self, urgency_level: str, user_age: Optional[int],
                           existing_conditions: Optional[List[str]]) -> List[str]:
        recommendations_db = {
            'low': ["Rest and hydrate", "Monitor symptoms", "Consider over-the-counter remedies if appropriate"],
            'medium': ["Consult with a healthcare provider if symptoms persist", "Rest and maintain hydration", "Monitor for worsening symptoms"],
            'high': ["Seek medical attention soon", "Contact healthcare provider", "Monitor closely for emergency signs"],
            'emergency': ["Seek immediate medical attention", "Go to emergency room if severe", "Contact emergency services for critical symptoms"]
        }
        
        base_recommendations = recommendations_db.get(urgency_level, [])
        specific_recommendations = []
        
        if user_age and user_age < 12:
            specific_recommendations.append("Consult with a pediatrician for children's health concerns")
        elif user_age and user_age > 65:
            specific_recommendations.append("Consider geriatric-specific health considerations")
        
        if existing_conditions:
            specific_recommendations.append(f"Discuss with your healthcare provider given your existing conditions: {', '.join(existing_conditions)}")
        
        return base_recommendations + specific_recommendations

# Initialize services
ai_service = AIService()
symptom_analyzer = SymptomAnalyzer()

# API Routes
@app.get("/")
async def root():
    return {"message": "Smart Health Companion API - Running on Kaggle"}

@app.post("/api/chat")
async def chat_with_companion(request: dict):
    try:
        ai_response = await ai_service.get_health_response(
            user_message=request.get("message", "")
        )
        
        # Store in database
        conn = sqlite3.connect('health_companion.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_message, ai_response, symptom_analysis, confidence_score)
            VALUES (?, ?, ?, ?)
        ''', (
            request.get("message", ""),
            ai_response["response"],
            json.dumps(ai_response["symptom_analysis"]),
            ai_response["confidence_score"]
        ))
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "response": ai_response["response"],
            "symptom_analysis": ai_response["symptom_analysis"],
            "confidence_score": ai_response["confidence_score"],
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/analyze-symptoms")
async def analyze_symptoms(request: dict):
    try:
        analysis = symptom_analyzer.analyze(
            symptoms=request.get("symptoms", []),
            severity=request.get("severity", "mild"),
            duration=request.get("duration", "days"),
            user_age=request.get("user_age"),
            existing_conditions=request.get("existing_conditions", [])
        )
        
        # Store in database
        conn = sqlite3.connect('health_companion.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO symptom_analyses (symptoms, severity, duration, user_age, existing_conditions, analysis_result, recommendations, urgency_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            json.dumps(request.get("symptoms", [])),
            request.get("severity", "mild"),
            request.get("duration", "days"),
            request.get("user_age"),
            json.dumps(request.get("existing_conditions", [])),
            analysis["analysis"],
            json.dumps(analysis["recommendations"]),
            analysis["urgency_level"]
        ))
        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "analysis": analysis["analysis"],
            "recommendations": analysis["recommendations"],
            "urgency_level": analysis["urgency_level"],
            "analysis_id": analysis_id
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/conversations")
async def get_conversation_history():
    try:
        conn = sqlite3.connect('health_companion.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM conversations ORDER BY created_at DESC LIMIT 50')
        conversations = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": conv[0],
                "user_message": conv[1],
                "ai_response": conv[2],
                "symptom_analysis": json.loads(conv[3]) if conv[3] else None,
                "confidence_score": conv[4],
                "created_at": conv[5]
            }
            for conv in conversations
        ]
    except Exception as e:
        return {"error": str(e)}

print("âœ… Backend API setup complete!")


# Create requirements.txt for submission
requirements = """
gradio>=3.50.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
plotly>=5.10.0
transformers>=4.20.0
torch>=1.12.0
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements)

print("âœ… requirements.txt created")


{
  "title": "Smart Health Companion - LLM Healthcare AI",
  "id": "your-username/smart-health-companion",
  "keywords": [
    "healthcare",
    "llm",
    "ai",
    "symptom-checker",
    "medical-ai"
  ],
  "competition": "llm-healthcare-challenge",
  "dataset_sources": [],
  "kernel_sources": [],
  "model_sources": [],
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": true,
  "language": "python",
  "kernel_type": "notebook"
}


evaluation_criteria = {
    "technical_implementation": [
        "LLM integration and fine-tuning",
        "Algorithm complexity and efficiency", 
        "Error handling and robustness",
        "Code quality and documentation"
    ],
    "innovation": [
        "Novelty of healthcare AI approach",
        "Safety mechanisms and ethical considerations",
        "User experience design",
        "Scalability and deployment readiness"
    ],
    "impact": [
        "Practical healthcare applications",
        "User accessibility and inclusivity",
        "Potential for real-world deployment",
        "Social benefit and healthcare improvement"
    ]
}


# Simple all-in-one version
import gradio as gr
import json
from typing import List, Optional, Dict, Any

class SimpleHealthCompanion:
    def __init__(self):
        self.symptom_keywords = {
            'headache': ['headache', 'head pain', 'migraine'],
            'fever': ['fever', 'temperature', 'hot'],
            'cough': ['cough', 'coughing'],
            'fatigue': ['tired', 'fatigue', 'exhausted'],
            'nausea': ['nausea', 'sick to stomach', 'queasy']
        }
    
    def chat(self, message, history):
        response = self._generate_response(message)
        symptom_count = self._count_symptoms(message)
        
        return f"{response}\n\n*Detected {symptom_count} symptom(s)*"
    
    def _generate_response(self, message):
        message_lower = message.lower()
        
        responses = {
            'headache': "For headaches: Rest in a quiet room, stay hydrated, and consider a cool compress. Consult a doctor if severe or persistent.",
            'fever': "For fever: Stay hydrated, rest, and use fever reducers as directed. Seek medical help if fever is high or lasts more than 3 days.",
            'cough': "For cough: Stay hydrated, use a humidifier, and avoid irritants. See a doctor if coughing persists or causes breathing difficulty.",
            'pain': "For pain: Rest the affected area and use pain relief as directed. Seek immediate help for severe or sudden pain.",
            'default': "I can provide general wellness information. For medical concerns, please consult a healthcare professional. How can I help with your wellness today?"
        }
        
        for keyword, response in responses.items():
            if keyword in message_lower and keyword != 'default':
                return response
        
        return responses['default']
    
    def _count_symptoms(self, text):
        text_lower = text.lower()
        count = 0
        for symptom, keywords in self.symptom_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                count += 1
        return count

# Create and launch simple version
companion = SimpleHealthCompanion()

simple_demo = gr.ChatInterface(
    companion.chat,
    title="ğŸ�¥ Smart Health Companion",
    description="Your AI health assistant for general wellness guidance",
    examples=[
        "I have a headache",
        "What should I do for fever?",
        "I'm feeling very tired",
        "My throat hurts when I swallow"
    ]
)

print("ğŸš€ Launching Simple Health Companion...")
simple_demo.launch(share=True)

