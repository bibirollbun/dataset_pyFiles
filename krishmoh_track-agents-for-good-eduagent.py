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
#!pip install -q google-generativeai python-dotenv


# %%
# Import libraries
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict
import json


# For Kaggle secrets
from kaggle_secrets import UserSecretsClient

# Get API key from Kaggle secrets
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

# Google Generative AI
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)




# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EduAgent')



from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from google.genai import types

# Configure Model Retry on errors
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)




class StudentMemoryBank:
    """
    Long-term memory storage for student profiles and learning history
    """
    
    def __init__(self):
        self.students: Dict[str, Dict[str, Any]] = {}
        logger.info("Student Memory Bank initialized")
    
    def create_profile(
        self,
        student_id: str,
        grade_level: Optional[str] = None,
        preferred_subjects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new student profile"""
        if student_id in self.students:
            return self.students[student_id]
        
        profile = {
            'student_id': student_id,
            'grade_level': grade_level,
            'preferred_subjects': preferred_subjects or [],
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'history': [],
            'achievements': [],
            'performance': {},
            'total_sessions': 0,
            'total_queries': 0
        }
        
        self.students[student_id] = profile
        logger.info(f"Created profile for student: {student_id}")
        return profile
    
    def get_profile(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a student's profile"""
        return self.students.get(student_id)
    
    def has_profile(self, student_id: str) -> bool:
        """Check if a profile exists"""
        return student_id in self.students
    
    def add_interaction(self, student_id: str, interaction: Dict[str, Any]) -> bool:
        """Add an interaction to student's history"""
        if student_id not in self.students:
            return False
        
        interaction['recorded_at'] = datetime.now().isoformat()
        self.students[student_id]['history'].append(interaction)
        self.students[student_id]['total_queries'] += 1
        self.students[student_id]['last_activity'] = datetime.now().isoformat()
        return True
    
    def save_session_summary(
        self,
        student_id: str,
        duration_minutes: float,
        queries_processed: int
    ) -> bool:
        """Save session summary to profile"""
        if student_id not in self.students:
            return False
        
        self.students[student_id]['total_sessions'] += 1
        
        if 'session_summaries' not in self.students[student_id]:
            self.students[student_id]['session_summaries'] = []
        
        self.students[student_id]['session_summaries'].append({
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': duration_minutes,
            'queries_processed': queries_processed
        })
        
        return True

print("âœ… StudentMemoryBank class defined")




class MetricsCollector:
    """
    Collects and tracks system performance metrics
    """
    
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'total_sessions': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_response_time': 0.0,
            'min_response_time': float('inf'),
            'max_response_time': 0.0,
            'average_response_time': 0.0,
            'agent_usage': defaultdict(int),
            'intent_distribution': defaultdict(int),
            'active_students': set(),
            'system_start_time': datetime.now(),
            'last_updated': datetime.now()
        }
        self.event_log: List[Dict[str, Any]] = []
        logger.info("Metrics Collector initialized")
    
    def log_query(
        self,
        student_id: str,
        query_type: str,
        processing_time: float,
        success: bool
    ):
        """Log a query execution"""
        self.metrics['total_queries'] += 1
        
        if success:
            self.metrics['successful_queries'] += 1
        else:
            self.metrics['failed_queries'] += 1
        
        self.metrics['total_response_time'] += processing_time
        self.metrics['min_response_time'] = min(
            self.metrics['min_response_time'],
            processing_time
        )
        self.metrics['max_response_time'] = max(
            self.metrics['max_response_time'],
            processing_time
        )
        
        if self.metrics['total_queries'] > 0:
            self.metrics['average_response_time'] = (
                self.metrics['total_response_time'] / self.metrics['total_queries']
            )
        
        self.metrics['intent_distribution'][query_type] += 1
        self.metrics['active_students'].add(student_id)
        self.metrics['last_updated'] = datetime.now()
    
    def log_session_start(self, student_id: str):
        """Log session start"""
        self.metrics['total_sessions'] += 1
        self.metrics['active_students'].add(student_id)
    
    def log_session_end(self, student_id: str, duration_minutes: float):
        """Log session end"""
        pass  # Implementation for tracking
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        success_rate = (
            (self.metrics['successful_queries'] / self.metrics['total_queries'] * 100)
            if self.metrics['total_queries'] > 0 else 0
        )
        
        return {
            'total_queries': self.metrics['total_queries'],
            'success_rate': round(success_rate, 2),
            'avg_response_time': round(self.metrics['average_response_time'], 3),
            'total_sessions': self.metrics['total_sessions'],
            'active_students': len(self.metrics['active_students'])
        }

print("âœ… MetricsCollector class defined")




class SubjectTutorAgent:
    """
    Specialized agent for explaining concepts across all subjects
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        logger.info("Subject Tutor Agent initialized")
    
    def _identify_subject(self, query: str) -> str:
        """Identify the subject area from the query"""
        query_lower = query.lower()
        
        math_keywords = ['math', 'number', 'equation', 'fraction', 'algebra', 'geometry']
        science_keywords = ['science', 'biology', 'chemistry', 'physics', 'cell', 'atom']
        language_keywords = ['grammar', 'writing', 'reading', 'sentence', 'paragraph']
        social_keywords = ['history', 'geography', 'government', 'culture', 'president']
        
        if any(kw in query_lower for kw in math_keywords):
            return 'mathematics'
        elif any(kw in query_lower for kw in science_keywords):
            return 'science'
        elif any(kw in query_lower for kw in language_keywords):
            return 'language_arts'
        elif any(kw in query_lower for kw in social_keywords):
            return 'social_studies'
        else:
            return 'general'
    
    def explain_concept(
        self,
        query: str,
        grade_level: str,
        student_context: Dict[str, Any]
    ) -> str:
        """Explain a concept based on student's query"""
        try:
            subject = self._identify_subject(query)
            
            system_instruction = f"""You are an expert tutor for {subject}.

Student Information:
- Grade Level: {grade_level}

Instructions:
- Explain in grade {grade_level} appropriate language
- Use simple, clear examples
- Be encouraging and supportive
- Keep explanation concise (3-5 paragraphs)
- End with: "Do you understand, or would you like me to explain further?"
"""
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )
            
            response = model.generate_content(query)
            return response.text
            
        except Exception as e:
            logger.error(f"Error in tutor: {str(e)}")
            return "I had trouble explaining that. Could you rephrase your question?"

print("âœ… SubjectTutorAgent class defined")




class ProblemGeneratorAgent:
    """
    Generates adaptive practice problems
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        logger.info("Problem Generator Agent initialized")
    
    def _determine_difficulty(self, performance_history: Dict[str, Any]) -> str:
        """Determine appropriate difficulty level"""
        # Simple adaptive logic
        if not performance_history:
            return 'easy'
        return 'medium'  # Can be enhanced with actual performance data
    
    def generate_problems(
        self,
        query: str,
        grade_level: str,
        performance_history: Dict[str, Any],
        num_problems: int = 3
    ) -> str:
        """Generate practice problems"""
        try:
            difficulty = self._determine_difficulty(performance_history)
            
            prompt = f"""Create {num_problems} practice problems for grade {grade_level}.

Topic: Extract from this request: "{query}"
Difficulty: {difficulty}

For each problem provide:
1. The problem statement
2. The answer
3. Step-by-step solution
4. 2 helpful hints

Format as numbered problems."""
            
            model = genai.GenerativeModel(model_name=self.model_name)
            response = model.generate_content(prompt)
            
            return f"Here are {num_problems} practice problems ({difficulty} difficulty):\n\n{response.text}"
            
        except Exception as e:
            logger.error(f"Error generating problems: {str(e)}")
            return "I had trouble creating problems. Please try again."

print("âœ… ProblemGeneratorAgent class defined")




class ProgressTrackerAgent:
    """
    Tracks and reports student progress
    """
    
    def __init__(self, memory_bank: StudentMemoryBank):
        self.memory_bank = memory_bank
        logger.info("Progress Tracker Agent initialized")
    
    def generate_report(self, student_id: str) -> str:
        """Generate a comprehensive progress report"""
        try:
            profile = self.memory_bank.get_profile(student_id)
            
            if not profile:
                return "I don't have enough data yet. Keep learning!"
            
            history = profile.get('history', [])
            
            report = "ğŸ“Š **Your Learning Progress Report**\n\n"
            report += "**Overview:**\n"
            report += f"- Total Sessions: {profile.get('total_sessions', 0)}\n"
            report += f"- Total Questions: {profile.get('total_queries', 0)}\n"
            
            if history:
                report += f"- Recent Activity: {len(history[-5:])} recent interactions\n"
            
            report += "\n**Keep up the great work! ğŸŒŸ**\n"
            report += "Consistent practice is key to mastery!\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return "I had trouble generating your report."

print("âœ… ProgressTrackerAgent class defined")



class KnowledgeGapAnalyzer:
    """
    Identifies knowledge gaps and provides recommendations
    """
    
    def __init__(self, memory_bank: StudentMemoryBank):
        self.memory_bank = memory_bank
        logger.info("Knowledge Gap Analyzer initialized")
    
    def analyze_gaps(self, student_id: str) -> str:
        """Analyze student's learning history"""
        try:
            profile = self.memory_bank.get_profile(student_id)
            
            if not profile:
                return "Keep practicing! I need more data to analyze your learning patterns."
            
            history = profile.get('history', [])
            
            analysis = "ğŸ”� **Knowledge Gap Analysis**\n\n"
            
            if len(history) < 5:
                analysis += "**Getting Started:**\n"
                analysis += "- You're just beginning your learning journey!\n"
                analysis += "- Keep asking questions and practicing\n"
                analysis += "- I'll provide personalized recommendations as you progress\n"
            else:
                analysis += "**Your Learning Pattern:**\n"
                analysis += f"- Total Interactions: {len(history)}\n"
                analysis += "- You're making steady progress!\n\n"
                analysis += "**ğŸ’¡ Recommendations:**\n"
                analysis += "- Continue practicing consistently\n"
                analysis += "- Try tackling challenging problems\n"
                analysis += "- Review concepts regularly\n"
            
            analysis += "\n**Remember**: Every expert was once a beginner! ğŸ’ª\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing gaps: {str(e)}")
            return "I had trouble analyzing your learning patterns."

print("âœ… KnowledgeGapAnalyzer class defined")




class OrchestratorAgent:
    """
    Main coordinator that routes queries to specialized agents
    """
    
    def __init__(
        self,
        memory_bank: StudentMemoryBank,
        metrics_collector: MetricsCollector
    ):
        self.memory_bank = memory_bank
        self.metrics_collector = metrics_collector
        
        # Initialize specialized agents
        self.subject_tutor = SubjectTutorAgent()
        self.problem_generator = ProblemGeneratorAgent()
        self.progress_tracker = ProgressTrackerAgent(memory_bank)
        self.gap_analyzer = KnowledgeGapAnalyzer(memory_bank)
        
        logger.info("Orchestrator Agent initialized")
    
    def _analyze_intent(self, query: str) -> str:
        """Analyze query to determine intent"""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['explain', 'what is', 'how does', 'why', 'teach']):
            return "explanation"
        elif any(kw in query_lower for kw in ['practice', 'problem', 'quiz', 'exercise']):
            return "practice"
        elif any(kw in query_lower for kw in ['progress', 'report', 'how am i doing']):
            return "progress"
        elif any(kw in query_lower for kw in ['weak', 'struggle', 'difficulty', 'gap']):
            return "gap_analysis"
        else:
            return "general"
    
    def process_query(
        self,
        student_id: str,
        query: str,
        session_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a student query"""
        try:
            intent = self._analyze_intent(query)
            grade_level = session_context.get('grade_level', '8')
            
            if intent == "explanation":
                response = self.subject_tutor.explain_concept(
                    query, grade_level, session_context
                )
                agent_used = 'subject_tutor'
            elif intent == "practice":
                response = self.problem_generator.generate_problems(
                    query, grade_level, {}
                )
                agent_used = 'problem_generator'
            elif intent == "progress":
                response = self.progress_tracker.generate_report(student_id)
                agent_used = 'progress_tracker'
            elif intent == "gap_analysis":
                response = self.gap_analyzer.analyze_gaps(student_id)
                agent_used = 'gap_analyzer'
            else:
                response = "I'm here to help you learn! You can ask me to:\n- Explain concepts\n- Generate practice problems\n- Check your progress\n- Analyze your learning gaps"
                agent_used = 'orchestrator'
            
            # Store interaction
            self.memory_bank.add_interaction(student_id, {
                'query': query,
                'intent': intent,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'response': response,
                'intent': intent,
                'agent_used': agent_used
            }
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {str(e)}")
            return {
                'response': "I encountered an error. Please try rephrasing your question.",
                'intent': 'error',
                'agent_used': 'orchestrator'
            }
    
    def generate_progress_report(self, student_id: str) -> str:
        """Generate progress report"""
        return self.progress_tracker.generate_report(student_id)

print("âœ… OrchestratorAgent class defined")





class EduAgentSystem:
    """
    Main system class coordinating all agents
    """
    
    def __init__(self):
        logger.info("Initializing EduAgent System...")
        
        self.memory_bank = StudentMemoryBank()
        self.metrics_collector = MetricsCollector()
        self.orchestrator = OrchestratorAgent(
            memory_bank=self.memory_bank,
            metrics_collector=self.metrics_collector
        )
        
        logger.info("EduAgent System initialized successfully")
    
    def start_session(
        self,
        student_id: str,
        grade_level: Optional[str] = None,
        preferred_subjects: Optional[list] = None
    ) -> Dict[str, Any]:
        """Start a new learning session"""
        try:
            if not self.memory_bank.has_profile(student_id):
                self.memory_bank.create_profile(
                    student_id=student_id,
                    grade_level=grade_level,
                    preferred_subjects=preferred_subjects
                )
            
            self.metrics_collector.log_session_start(student_id)
            
            return {
                'status': 'success',
                'message': "Welcome! I'm your AI learning assistant. How can I help you today?",
                'student_id': student_id
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to start session: {str(e)}'
            }
    
    def process_query(self, student_id: str, query: str) -> Dict[str, Any]:
        """Process a student query"""
        try:
            start_time = datetime.now()
            
            session_context = {
                'grade_level': self.memory_bank.get_profile(student_id).get('grade_level', '8')
            }
            
            response = self.orchestrator.process_query(
                student_id=student_id,
                query=query,
                session_context=session_context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.metrics_collector.log_query(
                student_id=student_id,
                query_type=response.get('intent', 'general'),
                processing_time=processing_time,
                success=True
            )
            
            return {
                'status': 'success',
                'response': response.get('response', ''),
                'intent': response.get('intent', 'general'),
                'agent_used': response.get('agent_used', 'orchestrator'),
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.metrics_collector.log_query(
                student_id=student_id,
                query_type='error',
                processing_time=0,
                success=False
            )
            
            return {
                'status': 'error',
                'message': f'Failed to process query: {str(e)}'
            }
    
    def get_progress_report(self, student_id: str) -> Dict[str, Any]:
        """Generate progress report"""
        try:
            report = self.orchestrator.generate_progress_report(student_id)
            return {
                'status': 'success',
                'report': report
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to generate report: {str(e)}'
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return self.metrics_collector.get_summary()

print("âœ… EduAgentSystem class defined")




# Initialize the system
print("Initializing EduAgent System...")
system = EduAgentSystem()
print("âœ… System ready!")



# Start a session
student_id = "demo_student_001"
grade_level = "8"

session_response = system.start_session(
    student_id=student_id,
    grade_level=grade_level
)

print("=" * 60)
print(session_response['message'])
print("=" * 60)


# Query 1: Explanation
query1 = "Can you explain what photosynthesis is?"

print(f"\nğŸ§‘â€�ğŸ�“ Student: {query1}\n")

response1 = system.process_query(student_id, query1)

if response1['status'] == 'success':
    print(f"ğŸ¤– EduAgent: {response1['response']}\n")
    print(f"[Intent: {response1['intent']} | Agent: {response1['agent_used']} | Time: {response1['processing_time']:.2f}s]")
else:
    print(f"Error: {response1['message']}")





# Query 2: Practice Problems
query2 = "Give me practice problems on fractions"

print(f"\nğŸ§‘â€�ğŸ�“ Student: {query2}\n")

response2 = system.process_query(student_id, query2)

if response2['status'] == 'success':
    print(f"ğŸ¤– EduAgent: {response2['response']}\n")
    print(f"[Intent: {response2['intent']} | Agent: {response2['agent_used']} | Time: {response2['processing_time']:.2f}s]")
else:
    print(f"Error: {response2['message']}")




# Query 3: Progress Report
query3 = "How am I doing?"

print(f"\nğŸ§‘â€�ğŸ�“ Student: {query3}\n")

response3 = system.process_query(student_id, query3)

if response3['status'] == 'success':
    print(f"ğŸ¤– EduAgent: {response3['response']}\n")
    print(f"[Intent: {response3['intent']} | Agent: {response3['agent_used']} | Time: {response3['processing_time']:.2f}s]")
else:
    print(f"Error: {response3['message']}")



# Query 4: Gap Analysis
query4 = "What areas do I need to work on?"

print(f"\nğŸ§‘â€�ğŸ�“ Student: {query4}\n")

response4 = system.process_query(student_id, query4)

if response4['status'] == 'success':
    print(f"ğŸ¤– EduAgent: {response4['response']}\n")
    print(f"[Intent: {response4['intent']} | Agent: {response4['agent_used']} | Time: {response4['processing_time']:.2f}s]")
else:
    print(f"Error: {response4['message']}")





# Get system metrics
metrics = system.get_system_metrics()

print("\nğŸ“Š **System Performance Metrics**")
print("=" * 60)
print(f"Total Queries Processed: {metrics['total_queries']}")
print(f"Success Rate: {metrics['success_rate']}%")
print(f"Average Response Time: {metrics['avg_response_time']:.3f}s")
print(f"Total Sessions: {metrics['total_sessions']}")
print(f"Active Students: {metrics['active_students']}")
print("=" * 60)





# Try your own query
your_query = "Explain the water cycle"  # Modify this!

print(f"\nğŸ§‘â€�ğŸ�“ Student: {your_query}\n")

response = system.process_query(student_id, your_query)

if response['status'] == 'success':
    print(f"ğŸ¤– EduAgent: {response['response']}\n")
    print(f"[Intent: {response['intent']} | Agent: {response['agent_used']} | Time: {response['processing_time']:.2f}s]")
else:
    print(f"Error: {response['message']}")





# %%
# Final metrics
final_metrics = system.get_system_metrics()

print("\nğŸ�‰ **Session Complete!**")
print("=" * 60)
print(f"âœ… Total Queries Processed: {final_metrics['total_queries']}")
print(f"âœ… Success Rate: {final_metrics['success_rate']}%")
print(f"âœ… Avg Response Time: {final_metrics['avg_response_time']:.3f}s")
print(f"âœ… Students Helped: {final_metrics['active_students']}")
print("=" * 60)
print("\nThank you for using EduAgent! Keep learning! ğŸ“šâœ¨")


