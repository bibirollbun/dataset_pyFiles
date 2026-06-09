!pip install ADK



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


import os
import json
from typing import Any, Dict, List
from datetime import datetime, timedelta

from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
print("âœ… ADK components imported successfully.")


APP_NAME = "agents_for_good_capstone"
USER_ID = "default_user"
MODEL_NAME = "gemini-2.5-flash-lite"

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


from langchain.tools import tool
from datetime import datetime, timedelta

@tool
def generate_study_plan(
    subject: str,
    exam_date: str,
    current_level: str,
    study_hours_per_day: int
) -> dict:
    """
    Generates a personalized study plan for students.
    
    Args:
        subject: The subject to study (e.g., 'Mathematics', 'Biology')
        exam_date: Date of the exam (YYYY-MM-DD format)
        current_level: Student's current level ('beginner', 'intermediate', 'advanced')
        study_hours_per_day: Hours available for study per day
    
    Returns:
        A structured study plan with daily breakdown
    """
    try:
        # Conversion de la date (string -> datetime)
        exam = datetime.strptime(exam_date, "%Y-%m-%d")
        today = datetime.now()
        days_until_exam = (exam - today).days
        
        if days_until_exam <= 0:
            return {"error": "Exam date must be in the future"}
        
        total_hours = days_until_exam * study_hours_per_day
        
        # Define topics based on level
        topics = {
            "beginner": ["Fundamentals", "Basic Concepts", "Simple Problems"],
            "intermediate": ["Core Concepts", "Practice Problems", "Application", "Review"],
            "advanced": ["Advanced Topics", "Complex Problems", "Research", "Mock Exams", "Review"]
        }
        
        subject_topics = topics.get(current_level.lower(), topics["intermediate"])
        # Eviter la division par zÃ©ro si la liste est vide (peu probable ici mais bonne pratique)
        if not subject_topics:
            subject_topics = ["General Review"]
            
        hours_per_topic = max(1, total_hours // len(subject_topics))
        
        plan = {
            "subject": subject,
            "exam_date": exam_date,
            "days_until_exam": days_until_exam,
            "total_study_hours": total_hours,
            "daily_breakdown": []
        }
        
        topic_index = 0
        hours_in_current_topic = 0
        
        for day in range(days_until_exam):
            current_date = today + timedelta(days=day)
            
            # Logique de changement de sujet
            if hours_in_current_topic >= hours_per_topic and topic_index < len(subject_topics) - 1:
                topic_index += 1
                hours_in_current_topic = 0
            
            plan["daily_breakdown"].append({
                "date": current_date.strftime("%Y-%m-%d"),
                "topic": subject_topics[topic_index],
                "hours": study_hours_per_day,
                "focus": f"Focus on {subject_topics[topic_index]} for {subject}"
            })
            
            hours_in_current_topic += study_hours_per_day
        
        return plan
        
    except Exception as e:
        return {"error": f"Failed to generate study plan: {str(e)}"}


import google.adk.tools
print(dir(google.adk.tools))


from datetime import datetime, timedelta

# L'agent lira la signature et la docstring automatiquement.
def generate_study_plan(
    subject: str,
    exam_date: str,
    current_level: str,
    study_hours_per_day: int
) -> dict:
    """
    Generates a personalized study plan for students.
    
    Args:
        subject: The subject to study (e.g., 'Mathematics', 'Biology')
        exam_date: Date of the exam (YYYY-MM-DD format)
        current_level: Student's current level ('beginner', 'intermediate', 'advanced')
        study_hours_per_day: Hours available for study per day
    
    Returns:
        A structured study plan with daily breakdown
    """
    try:
        # Conversion de la date
        exam = datetime.strptime(exam_date, "%Y-%m-%d")
        today = datetime.now()
        days_until_exam = (exam - today).days
        
        if days_until_exam <= 0:
            return {"error": "Exam date must be in the future"}
        
        total_hours = days_until_exam * study_hours_per_day
        
        # Define topics based on level
        topics = {
            "beginner": ["Fundamentals", "Basic Concepts", "Simple Problems"],
            "intermediate": ["Core Concepts", "Practice Problems", "Application", "Review"],
            "advanced": ["Advanced Topics", "Complex Problems", "Research", "Mock Exams", "Review"]
        }
        
        # SÃ©lection des sujets avec une valeur par dÃ©faut
        subject_topics = topics.get(current_level.lower(), topics["intermediate"])
        
        # SÃ©curitÃ© pour Ã©viter la division par zÃ©ro
        if not subject_topics:
            subject_topics = ["General Review"]

        hours_per_topic = max(1, total_hours // len(subject_topics))
        
        plan = {
            "subject": subject,
            "exam_date": exam_date,
            "days_until_exam": days_until_exam,
            "total_study_hours": total_hours,
            "daily_breakdown": []
        }
        
        current_date = today
        topic_index = 0
        hours_in_current_topic = 0
        
        for day in range(days_until_exam):
            current_date = today + timedelta(days=day)
            
            # Logique pour passer au sujet suivant
            if hours_in_current_topic >= hours_per_topic and topic_index < len(subject_topics) - 1:
                topic_index += 1
                hours_in_current_topic = 0
            
            plan["daily_breakdown"].append({
                "date": current_date.strftime("%Y-%m-%d"),
                "topic": subject_topics[topic_index],
                "hours": study_hours_per_day,
                "focus": f"Focus on {subject_topics[topic_index]} for {subject}"
            })
            
            hours_in_current_topic += study_hours_per_day
        
        return plan
        
    except Exception as e:
        return {"error": f"Failed to generate study plan: {str(e)}"}


def calculate_medication_schedule(
    medication_name: str,
    dosage: str,
    frequency_per_day: int,
    duration_days: int,
    start_time: str = "08:00"
) -> dict:
    """
    Creates a medication reminder schedule for patients.
    
    Args:
        medication_name: Name of the medication
        dosage: Dosage amount (e.g., '500mg', '2 tablets')
        frequency_per_day: How many times per day to take medication
        duration_days: Total days of medication
        start_time: First dose time in HH:MM format
    
    Returns:
        A detailed medication schedule with reminders
    """
    try:
        start_hour, start_minute = map(int, start_time.split(':'))
        interval_hours = 24 / frequency_per_day
        
        schedule = {
            "medication": medication_name,
            "dosage": dosage,
            "frequency": f"{frequency_per_day} times per day",
            "duration": f"{duration_days} days",
            "daily_times": [],
            "important_notes": [
                "Take medication at the same times each day",
                "Do not skip doses",
                "Complete the full course even if you feel better",
                "Consult your doctor if you experience side effects"
            ]
        }
        
        # Generate daily times
        for i in range(frequency_per_day):
            dose_time = (start_hour + int(i * interval_hours)) % 24
            dose_minute = start_minute
            schedule["daily_times"].append(f"{dose_time:02d}:{dose_minute:02d}")
        
        # Generate full schedule
        schedule["full_schedule"] = []
        today = datetime.now()
        
        for day in range(duration_days):
            date = today + timedelta(days=day)
            for time in schedule["daily_times"]:
                schedule["full_schedule"].append({
                    "date": date.strftime("%Y-%m-%d"),
                    "time": time,
                    "medication": medication_name,
                    "dosage": dosage
                })
        
        return schedule
        
    except Exception as e:
        return {"error": f"Failed to create medication schedule: {str(e)}"}




def calculate_carbon_footprint(
    transportation_miles: float,
    transportation_type: str,
    electricity_kwh: float,
    meat_meals_per_week: int,
    flights_per_year: int
) -> dict:
    """
    Calculates an individual's estimated carbon footprint and provides reduction suggestions.
    
    Args:
        transportation_miles: Weekly miles traveled
        transportation_type: Type of transport ('car', 'bus', 'train', 'bike', 'walk')
        electricity_kwh: Monthly electricity usage in kWh
        meat_meals_per_week: Number of meals with meat per week
        flights_per_year: Number of flights taken per year
    
    Returns:
        Carbon footprint calculation and personalized reduction recommendations
    """
    # Carbon emission factors (kg CO2 per unit)
    emission_factors = {
        "car": 0.411,  # kg CO2 per mile
        "bus": 0.089,
        "train": 0.041,
        "bike": 0,
        "walk": 0,
        "electricity": 0.92,  # kg CO2 per kWh (US average)
        "meat_meal": 7.0,  # kg CO2 per meal
        "flight": 1000,  # kg CO2 per flight (average)
    }
    
    # Calculate emissions
    transport_emissions = (
        transportation_miles * 52 * emission_factors.get(transportation_type.lower(), 0.411)
    )
    electricity_emissions = electricity_kwh * 12 * emission_factors["electricity"]
    food_emissions = meat_meals_per_week * 52 * emission_factors["meat_meal"]
    flight_emissions = flights_per_year * emission_factors["flight"]
    
    total_emissions = (
        transport_emissions + electricity_emissions + food_emissions + flight_emissions
    )
    
    # Generate recommendations
    recommendations = []
    
    if transportation_type.lower() == "car" and transportation_miles > 50:
        potential_savings = transportation_miles * 52 * (emission_factors["car"] - emission_factors["bus"])
        recommendations.append({
            "category": "Transportation",
            "action": "Switch to public transportation for work commute",
            "potential_reduction_kg": round(potential_savings, 2),
            "impact": "High"
        })
    
    if electricity_kwh > 300:
        potential_savings = (electricity_kwh - 300) * 12 * emission_factors["electricity"]
        recommendations.append({
            "category": "Energy",
            "action": "Reduce electricity usage through LED bulbs and energy-efficient appliances",
            "potential_reduction_kg": round(potential_savings, 2),
            "impact": "Medium"
        })
    
    if meat_meals_per_week > 7:
        potential_savings = (meat_meals_per_week - 7) * 52 * emission_factors["meat_meal"]
        recommendations.append({
            "category": "Diet",
            "action": "Reduce meat consumption to 1 meal per day or less",
            "potential_reduction_kg": round(potential_savings, 2),
            "impact": "High"
        })
    
    if flights_per_year > 2:
        potential_savings = (flights_per_year - 2) * emission_factors["flight"]
        recommendations.append({
            "category": "Travel",
            "action": "Reduce air travel or offset carbon emissions",
            "potential_reduction_kg": round(potential_savings, 2),
            "impact": "Very High"
        })
    
    total_potential_reduction = sum(r["potential_reduction_kg"] for r in recommendations)
    
    return {
        "total_annual_emissions_kg": round(total_emissions, 2),
        "total_annual_emissions_tons": round(total_emissions / 1000, 2),
        "comparison": f"Average US citizen: ~16 tons/year",
        "breakdown": {
            "transportation": round(transport_emissions, 2),
            "electricity": round(electricity_emissions, 2),
            "food": round(food_emissions, 2),
            "flights": round(flight_emissions, 2)
        },
        "recommendations": recommendations,
        "potential_total_reduction_kg": round(total_potential_reduction, 2),
        "potential_reduction_percentage": round(
            (total_potential_reduction / total_emissions * 100) if total_emissions > 0 else 0,
            1
        )
    }


def assess_health_risk(
    age: int,
    symptoms: List[str],
    duration_days: int,
    severity: str
) -> dict:
    """
    Provides preliminary health risk assessment based on symptoms.
    DISCLAIMER: This is not medical advice. Always consult a healthcare professional.
    
    Args:
        age: Patient's age
        symptoms: List of symptoms (e.g., ['fever', 'cough', 'fatigue'])
        duration_days: How many days symptoms have persisted
        severity: Symptom severity ('mild', 'moderate', 'severe')
    
    Returns:
        Risk assessment and recommendations
    """
    # Define symptom urgency levels
    urgent_symptoms = [
        'chest pain', 'difficulty breathing', 'severe headache', 
        'confusion', 'loss of consciousness', 'severe bleeding',
        'sudden vision loss', 'seizure'
    ]
    
    moderate_symptoms = [
        'high fever', 'persistent vomiting', 'severe pain',
        'dehydration', 'rapid heart rate'
    ]
    
    # Check for urgent symptoms
    has_urgent = any(s.lower() in ' '.join(symptoms).lower() for s in urgent_symptoms)
    has_moderate = any(s.lower() in ' '.join(symptoms).lower() for s in moderate_symptoms)
    
    if has_urgent or severity.lower() == 'severe':
        urgency = "URGENT - SEEK IMMEDIATE MEDICAL ATTENTION"
        action = "Go to emergency room or call emergency services"
        risk_level = "High"
    elif has_moderate or (severity.lower() == 'moderate' and duration_days > 3):
        urgency = "IMPORTANT - Consult a doctor soon"
        action = "Schedule a doctor's appointment within 24-48 hours"
        risk_level = "Medium"
    elif duration_days > 7:
        urgency = "ATTENTION - Consider medical consultation"
        action = "If symptoms persist, consult a healthcare provider"
        risk_level = "Low to Medium"
    else:
        urgency = "MONITOR - Continue self-care"
        action = "Monitor symptoms and seek care if they worsen"
        risk_level = "Low"
    
    return {
        "disclaimer": "âš ï¸� THIS IS NOT MEDICAL ADVICE. Always consult a healthcare professional.",
        "urgency_level": urgency,
        "risk_assessment": risk_level,
        "recommended_action": action,
        "symptoms_reported": symptoms,
        "duration": f"{duration_days} days",
        "severity": severity,
        "age_consideration": "Higher risk" if age > 65 or age < 5 else "Standard risk",
        "general_advice": [
            "Stay hydrated",
            "Get adequate rest",
            "Monitor temperature regularly",
            "Keep a symptom diary",
            "Avoid self-medication without professional guidance"
        ]
    }


# Education Specialist Agent
education_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="education_specialist",
    description=(
        "Specialized agent for educational support. Creates study plans, generates quizzes, "
        "provides learning resources, and helps students with academic challenges. "
        "Expert in personalized learning and educational best practices."
    ),
    tools=[generate_study_plan]
)



# Healthcare Assistant Agent
healthcare_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="healthcare_assistant",
    description=(
        "Healthcare support agent that helps with medication schedules, health risk assessments, "
        "and general health information. Provides preliminary guidance and emphasizes consulting "
        "healthcare professionals for medical decisions."
    ),
    tools=[calculate_medication_schedule, assess_health_risk]
)



# Sustainability Advisor Agent
sustainability_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="sustainability_advisor",
    description=(
        "Environmental sustainability expert that calculates carbon footprints, provides "
        "eco-friendly recommendations, and helps individuals and organizations reduce their "
        "environmental impact through actionable insights."
    ),
    tools=[calculate_carbon_footprint]
)


root_agent = LlmAgent(
    model=Gemini(
        model=MODEL_NAME,
        retry_options=retry_config,
        tools=["google_search"]  # Built-in Google Search tool
    ),
    name="agents_for_good_coordinator",
    description=(
        "Main coordination agent for 'Agents for Good' platform. Routes user queries to "
        "specialized agents (education, healthcare, sustainability) and provides general support. "
        "Can search the web for current information when needed."
    ),
    sub_agents=[education_agent, healthcare_agent, sustainability_agent]
)



# Use persistent database storage
db_url = "sqlite:///agents_for_good.db"
session_service = DatabaseSessionService(db_url=db_url)

# Create app with Events Compaction for context management
app = App(
    name=APP_NAME,
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,  # Compact every 5 turns
        overlap_size=2,  # Keep 2 previous turns for context
    ),
)


# Create runner
runner = Runner(app=app, session_service=session_service)


async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    """Execute a conversation session with the agent system"""
    print(f"\n{'='*80}")
    print(f"SESSION: {session_name}")
    print(f"{'='*80}\n")

    app_name = runner_instance.app_name

    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    if user_queries:
        if type(user_queries) == str:
            user_queries = [user_queries]

        for query in user_queries:
            print(f"\nğŸ‘¤ USER: {query}\n")
            print(f"ğŸ¤– AGENT: ", end="")

            query_content = types.Content(
                role="user",
                parts=[types.Part(text=query)]
            )

            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query_content
            ):
                if event.content and event.content.parts:
                    if event.content.parts[0].text and event.content.parts[0].text != "None":
                        print(event.content.parts[0].text)
            
            print("\n" + "-"*80)
    else:
        print("â�Œ No queries provided!")



print("âœ… Agents for Good Capstone Project Initialized!")
print(f"   ğŸ“± Application: {APP_NAME}")
print(f"   ğŸ‘¤ User: {USER_ID}")
print(f"   ğŸ—„ï¸�  Database: agents_for_good.db")
print(f"   ğŸ¤– Root Agent: {root_agent.name}")
print(f"   ğŸ‘¥ Sub-Agents: {len(root_agent.sub_agents)}")
print(f"      - {education_agent.name}")
print(f"      - {healthcare_agent.name}")
print(f"      - {sustainability_agent.name}")
print(f"   ğŸ”§ Custom Tools: 4")
print(f"   ğŸ”� Built-in Tools: Google Search")
print(f"   ğŸ’¾ Session Management: DatabaseSessionService")
print(f"   ğŸ“¦ Context Management: Events Compaction (interval=5, overlap=2)")


async def demo_education():
    """Demonstrates the education agent with study plan generation"""
    await run_session(
        runner,
        [
            "Hi! I'm a college student struggling with organic chemistry. I have a final exam on December 15, 2025.",
            "Can you create a detailed study plan for me? I can study 3 hours per day, and I'm at an intermediate level.",
            "What topics should I focus on first?"
        ],
        "education_demo"
    )

# To run: await demo_education()


async def demo_healthcare():
    """Demonstrates the healthcare agent with medication scheduling"""
    await run_session(
        runner,
        [
            "My doctor prescribed me Amoxicillin 500mg, three times a day for 10 days.",
            "Can you help me create a medication schedule? I'd like to start at 8:00 AM.",
            "I sometimes forget to take medications. What reminders should I set?"
        ],
        "healthcare_demo"
    )

# To run: await demo_healthcare()



async def demo_sustainability():
    """Demonstrates the sustainability agent with carbon footprint calculation"""
    await run_session(
        runner,
        [
            "I want to reduce my environmental impact. Can you help me calculate my carbon footprint?",
            "I drive 150 miles per week by car, use about 400 kWh electricity monthly, eat meat 10 times a week, and take 4 flights per year.",
            "What are the most impactful changes I can make?"
        ],
        "sustainability_demo"
    )

# To run: await demo_sustainability()


async def demo_multi_domain():
    """Demonstrates coordination between multiple specialized agents"""
    await run_session(
        runner,
        [
            "I'm a medical student preparing for exams while also trying to live more sustainably.",
            "Can you help me: 1) Create a study schedule for my pathology exam in 3 weeks, 2) Calculate my carbon footprint from commuting 50 miles/week by car, and 3) Suggest ways to balance both goals?",
            "Also, I'm on medication for anxiety - can you remind me about maintaining a regular schedule?"
        ],
        "multi_domain_demo"
    )

# To run: await demo_multi_domain()


async def demo_memory_test():
    """Tests session memory and context persistence"""
    
    # First session
    await run_session(
        runner,
        [
            "Hi! My name is Sarah and I'm studying biology. I need to reduce my carbon footprint.",
            "Can you calculate it for me? I drive 100 miles/week, use 300 kWh/month, eat meat 5x/week, and fly twice a year."
        ],
        "memory_test_session"
    )
    
    # Second query in same session - should remember context
    await run_session(
        runner,
        [
            "What was my name again?",
            "What was my carbon footprint?",
            "Can you create a study plan for my biology exam on December 20th? I can study 2 hours/day at intermediate level."
        ],
        "memory_test_session"
    )
    
    # New session - should NOT remember previous context
    await run_session(
        runner,
        [
            "What's my name?",
            "Do you know anything about my carbon footprint?"
        ],
        "new_session_test"
    )

# To run: await demo_memory_test()



async def demo_web_search():
    """Demonstrates Google Search integration for current information"""
    await run_session(
        runner,
        [
            "What are the latest developments in AI for education in 2025?",
            "Are there any new sustainable energy technologies announced recently?",
            "What are current best practices for managing student mental health?"
        ],
        "web_search_demo"
    )

# To run: await demo_web_search()


import time
from typing import Dict, List

class AgentEvaluator:
    """Evaluation framework for measuring agent performance"""
    
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "tool_calls": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "agent_transitions": 0
        }
    
    async def evaluate_scenario(
        self,
        runner_instance: Runner,
        scenario_name: str,
        queries: List[str],
        expected_tools: List[str] = None
    ) -> Dict:
        """
        Evaluate a specific scenario
        
        Args:
            runner_instance: The agent runner
            scenario_name: Name of the scenario
            queries: List of queries to test
            expected_tools: Tools that should be called
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING: {scenario_name}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        tools_used = set()
        agents_used = set()
        
        try:
            session = await session_service.create_session(
                app_name=runner_instance.app_name,
                user_id=USER_ID,
                session_id=f"eval_{scenario_name}"
            )
        except:
            session = await session_service.get_session(
                app_name=runner_instance.app_name,
                user_id=USER_ID,
                session_id=f"eval_{scenario_name}"
            )
        
        for query in queries:
            query_start = time.time()
            query_content = types.Content(
                role="user",
                parts=[types.Part(text=query)]
            )
            
            response_text = ""
            async for event in runner_instance.run_async(
                user_id=USER_ID,
                session_id=session.id,
                new_message=query_content
            ):
                if event.content and event.content.parts:
                    if event.content.parts[0].text:
                        response_text += event.content.parts[0].text
                
                # Track tool usage
                if hasattr(event, 'actions') and event.actions:
                    if hasattr(event.actions, 'tool_calls'):
                        for tool in event.actions.tool_calls:
                            tools_used.add(tool.name)
                            self.metrics["tool_calls"] += 1
            
            query_time = time.time() - query_start
            self.metrics["response_times"].append(query_time)
        
        total_time = time.time() - start_time
        
        # Check if expected tools were used
        tool_success = True
        if expected_tools:
            tool_success = all(tool in tools_used for tool in expected_tools)
        
        if tool_success and len(response_text) > 0:
            self.metrics["successful_tasks"] += 1
        else:
            self.metrics["failed_tasks"] += 1
        
        evaluation_result = {
            "scenario": scenario_name,
            "success": tool_success,
            "total_time_seconds": round(total_time, 2),
            "average_response_time": round(sum(self.metrics["response_times"]) / len(self.metrics["response_times"]), 2),
            "tools_used": list(tools_used),
            "expected_tools_met": tool_success,
            "queries_processed": len(queries)
        }
        
        print(f"\nâœ… Evaluation Complete:")
        print(f"   - Success: {evaluation_result['success']}")
        print(f"   - Total Time: {evaluation_result['total_time_seconds']}s")
        print(f"   - Avg Response: {evaluation_result['average_response_time']}s")
        print(f"   - Tools Used: {evaluation_result['tools_used']}")
        
        return evaluation_result
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        if not self.metrics["response_times"]:
            return {"error": "No evaluations run yet"}
        
        total_tasks = self.metrics["successful_tasks"] + self.metrics["failed_tasks"]
        success_rate = (self.metrics["successful_tasks"] / total_tasks * 100) if total_tasks > 0 else 0
        
        report = {
            "overall_metrics": {
                "total_tasks": total_tasks,
                "successful_tasks": self.metrics["successful_tasks"],
                "failed_tasks": self.metrics["failed_tasks"],
                "success_rate_percent": round(success_rate, 2),
                "total_tool_calls": self.metrics["tool_calls"],
                "agent_transitions": self.metrics["agent_transitions"]
            },
            "performance_metrics": {
                "average_response_time": round(sum(self.metrics["response_times"]) / len(self.metrics["response_times"]), 2),
                "min_response_time": round(min(self.metrics["response_times"]), 2),
                "max_response_time": round(max(self.metrics["response_times"]), 2),
                "total_response_time": round(sum(self.metrics["response_times"]), 2)
            }
        }
        
        return report



async def run_full_evaluation():
    """Run comprehensive evaluation of all agent capabilities"""
    
    evaluator = AgentEvaluator()
    
    # Test 1: Education Agent
    result1 = await evaluator.evaluate_scenario(
        runner,
        "education_study_plan",
        [
            "Create a study plan for calculus exam on 2025-12-20, intermediate level, 3 hours/day"
        ],
        expected_tools=["generate_study_plan"]
    )
    
    # Test 2: Healthcare Agent
    result2 = await evaluator.evaluate_scenario(
        runner,
        "healthcare_medication",
        [
            "Create medication schedule for Aspirin 100mg, twice daily, 7 days, starting 09:00"
        ],
        expected_tools=["calculate_medication_schedule"]
    )
    
    # Test 3: Sustainability Agent
    result3 = await evaluator.evaluate_scenario(
        runner,
        "sustainability_footprint",
        [
            "Calculate carbon footprint: 80 miles/week car, 350 kWh/month, 6 meat meals/week, 3 flights/year"
        ],
        expected_tools=["calculate_carbon_footprint"]
    )
    
    # Test 4: Multi-Agent Coordination
    result4 = await evaluator.evaluate_scenario(
        runner,
        "multi_agent_coordination",
        [
            "I need both a study plan for physics (exam 2025-12-15, 2 hrs/day, beginner) and my carbon footprint (100 miles car/week, 400 kWh/month, 8 meat meals/week, 2 flights/year)"
        ],
        expected_tools=["generate_study_plan", "calculate_carbon_footprint"]
    )
    
    # Generate final report
    print("\n" + "="*80)
    print("FINAL EVALUATION REPORT")
    print("="*80)
    
    report = evaluator.generate_report()
    print(json.dumps(report, indent=2))
    
    return report

# To run: await run_full_evaluation()


import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agents_for_good.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('agents_for_good')

async def run_session_with_logging(
    runner_instance: Runner,
    user_queries: list[str] | str,
    session_name: str = "default",
):
    """Enhanced run_session with comprehensive logging"""
    
    logger.info(f"Starting session: {session_name}")
    logger.info(f"User ID: {USER_ID}")
    
    app_name = runner_instance.app_name
    
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
        logger.info(f"Session created: {session.id}")
    except Exception as e:
        logger.info(f"Session exists, retrieving: {session_name}")
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    
    if type(user_queries) == str:
        user_queries = [user_queries]
    
    for idx, query in enumerate(user_queries):
        logger.info(f"Processing query {idx + 1}/{len(user_queries)}: {query[:50]}...")
        
        query_content = types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )
        
        start_time = time.time()
        event_count = 0
        
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            event_count += 1
            
            if event.content and event.content.parts:
                if event.content.parts[0].text:
                    logger.debug(f"Response chunk received: {len(event.content.parts[0].text)} chars")
        
        elapsed = time.time() - start_time
        logger.info(f"Query completed in {elapsed:.2f}s with {event_count} events")
    
    logger.info(f"Session {session_name} completed successfully")

# To run with logging: await run_session_with_logging(runner, ["your query"], "logged_session")


print("\nâœ… Demo scenarios and evaluation framework loaded!")
print("   Available demos:")
print("      - await demo_education()")
print("      - await demo_healthcare()")
print("      - await demo_sustainability()")
print("      - await demo_multi_domain()")
print("      - await demo_memory_test()")
print("      - await demo_web_search()")
print("   Evaluation:")
print("      - await run_full_evaluation()")
print("   Logging:")
print("      - await run_session_with_logging(runner, queries, session_name)")


# Demo 1: Education
await demo_education()

# Demo 2: Healthcare
await demo_healthcare()

# Demo 3: Sustainability
await demo_sustainability()

# Demo 4: Multi-domain coordination
await demo_multi_domain()

# Demo 5: Memory and context
await demo_memory_test()

# Demo 6: Web search integration
await demo_web_search()


# Run comprehensive evaluation
evaluation_results = await run_full_evaluation()

# Display results
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)
print(json.dumps(evaluation_results, indent=2))


print("âœ… KEY CONCEPTS IMPLEMENTED:")
print("\n1. MULTI-AGENT SYSTEM")
print(f"   - Root Agent: {root_agent.name}")
print(f"   - Sub-Agents: {len(root_agent.sub_agents)}")
for agent in root_agent.sub_agents:
    print(f"     â€¢ {agent.name}")

print("\n2. CUSTOM TOOLS")
custom_tools = [
    "generate_study_plan",
    "calculate_medication_schedule", 
    "assess_health_risk",
    "calculate_carbon_footprint"
]
for tool in custom_tools:
    print(f"   âœ“ {tool}")

print("\n3. BUILT-IN TOOLS")
print("   âœ“ Google Search")

print("\n4. SESSIONS & MEMORY")
print(f"   - Session Service: {session_service.__class__.__name__}")
print(f"   - Database: agents_for_good.db")
print("   - Persistent storage: âœ“")

print("\n5. CONTEXT ENGINEERING")
print(f"   - Events Compaction: âœ“")
print(f"   - Compaction Interval: 5 turns")
print(f"   - Overlap Size: 2 turns")

print("\n6. OBSERVABILITY")
print("   - Logging: âœ“")
print("   - Performance Metrics: âœ“")
print("   - Evaluation Framework: âœ“")

print("\nğŸ“Š TOTAL KEY CONCEPTS: 6/8 available")
print("âœ… Exceeds minimum requirement of 3 concepts!")

