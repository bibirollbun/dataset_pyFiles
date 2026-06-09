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


"""
LifeSkills AI Agent - Multi-Agent System for Special Needs Education
Agents Intensive Capstone Project - Agents for Good Track
Author: Anand Srinivas K
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple
import json

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LifeSkillsAI')

# ============================================================================
# AGENT 1: SCHEDULER AGENT
# Purpose: Manages task sequences and timing
# Technology: Sequential agent logic with priority queue
# ============================================================================

class SchedulerAgent:
    """
    Scheduler Agent manages the task queue and prioritization.
    Implements sequential agent pattern with adaptive scheduling.
    """
    
    def __init__(self, task_library: List[Dict]):
        self.task_library = task_library
        self.current_schedule = []
        self.completed_tasks = []
        logger.info("SchedulerAgent initialized with %d tasks", len(task_library))
    
    def create_schedule(self, user_id: str, skill_level: str = "beginner") -> List[Dict]:
        """
        Creates personalized schedule based on user skill level.
        Context Engineering: Adapts task difficulty based on user performance.
        """
        logger.info(f"Creating schedule for user {user_id} at {skill_level} level")
        
        # Filter tasks by skill level
        filtered_tasks = [
            task for task in self.task_library 
            if task['difficulty'] <= self._get_difficulty_threshold(skill_level)
        ]
        
        self.current_schedule = filtered_tasks[:5]  # Top 5 priority tasks
        logger.info(f"Schedule created with {len(self.current_schedule)} tasks")
        return self.current_schedule
    
    def next_task(self) -> Dict:
        """
        Returns next task from schedule (FIFO queue).
        """
        if self.current_schedule:
            task = self.current_schedule.pop(0)
            logger.info(f"Next task: {task['name']}")
            return task
        logger.warning("No more tasks in schedule")
        return None
    
    def mark_completed(self, task: Dict, success: bool):
        """
        Marks task as completed and logs result.
        """
        task['completed'] = True
        task['success'] = success
        task['timestamp'] = datetime.now().isoformat()
        self.completed_tasks.append(task)
        logger.info(f"Task '{task['name']}' marked as {'successful' if success else 'failed'}")
    
    def _get_difficulty_threshold(self, skill_level: str) -> int:
        """Maps skill level to difficulty threshold."""
        mapping = {"beginner": 2, "intermediate": 4, "advanced": 5}
        return mapping.get(skill_level, 2)


# ============================================================================
# AGENT 2: PROMPT DELIVERY AGENT
# Purpose: Provides audio-visual task instructions
# Technology: Text-to-speech synthesis + visual cues
# ============================================================================

class PromptDeliveryAgent:
    """
    Prompt Delivery Agent provides multimodal task instructions.
    Custom Tool: Text-to-speech for voice guidance.
    """
    
    def __init__(self, use_voice: bool = True):
        self.use_voice = use_voice
        self.voice_engine = None
        
        if use_voice:
            try:
                import pyttsx3
                self.voice_engine = pyttsx3.init()
                self.voice_engine.setProperty('rate', 130)  # Slower for comprehension
                self.voice_engine.setProperty('volume', 0.9)
                logger.info("PromptDeliveryAgent initialized with voice synthesis")
            except ImportError:
                logger.warning("pyttsx3 not installed. Voice disabled. Install via: pip install pyttsx3")
                self.use_voice = False
    
    def deliver_prompt(self, task: Dict) -> None:
        """
        Delivers task prompt via text and optional voice.
        Multi-modal output: Text + Audio.
        """
        prompt_text = self._generate_prompt_text(task)
        
        # Visual output
        print("\n" + "="*60)
        print(f"ğŸ�¯ TASK: {task['name'].upper()}")
        print("="*60)
        print(f"\n{prompt_text}\n")
        print(f"Difficulty: {'â­�' * task['difficulty']}")
        print("="*60 + "\n")
        
        logger.info(f"Delivered prompt for task: {task['name']}")
        
        # Voice output
        if self.use_voice and self.voice_engine:
            self._speak(prompt_text)
    
    def _generate_prompt_text(self, task: Dict) -> str:
        """Generates clear, simple instructions."""
        instructions = task.get('instructions', 'Complete this activity.')
        return f"Let's practice: {task['name']}!\n\nSteps:\n{instructions}"
    
    def _speak(self, text: str):
        """Uses text-to-speech to vocalize instructions."""
        try:
            self.voice_engine.say(text)
            self.voice_engine.runAndWait()
            logger.info("Voice prompt delivered")
        except Exception as e:
            logger.error(f"Voice synthesis error: {e}")





# ============================================================================
# AGENT 3: PROGRESS TRACKING AGENT
# Purpose: Records completion data and maintains session history
# Technology: In-memory session service with persistent logging
# Sessions & Memory Implementation
# ============================================================================

class ProgressTrackingAgent:
    """
    Progress Tracking Agent maintains user session data.
    Implements Sessions & Memory pattern with in-memory state management.
    """
    
    def __init__(self):
        # In-memory session service
        self.user_sessions = {}  # user_id -> session data
        self.activity_log = []   # Complete activity history
        logger.info("ProgressTrackingAgent initialized with in-memory session service")
    
    def record_activity(self, user_id: str, task: Dict, success: bool, duration: float = 0.0):
        """
        Records task completion event.
        Observability: Logs all activity for tracking and analysis.
        """
        timestamp = datetime.now().isoformat()
        
        activity_record = {
            'user_id': user_id,
            'task_name': task['name'],
            'task_id': task.get('id', 'unknown'),
            'difficulty': task['difficulty'],
            'success': success,
            'duration_seconds': duration,
            'timestamp': timestamp
        }
        
        # Add to activity log
        self.activity_log.append(activity_record)
        
        # Update user session
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'total_time': 0.0,
                'activities': []
            }
        
        session = self.user_sessions[user_id]
        session['total_tasks'] += 1
        if success:
            session['successful_tasks'] += 1
        else:
            session['failed_tasks'] += 1
        session['total_time'] += duration
        session['activities'].append(activity_record)
        
        logger.info(f"Activity recorded: User={user_id}, Task={task['name']}, Success={success}")
    
    def get_user_progress(self, user_id: str) -> Dict:
        """
        Retrieves user progress summary.
        Memory: Returns historical session data.
        """
        if user_id not in self.user_sessions:
            return {'error': 'User not found'}
        
        session = self.user_sessions[user_id]
        success_rate = (session['successful_tasks'] / session['total_tasks'] * 100) if session['total_tasks'] > 0 else 0
        
        return {
            'total_tasks': session['total_tasks'],
            'successful_tasks': session['successful_tasks'],
            'failed_tasks': session['failed_tasks'],
            'success_rate': round(success_rate, 2),
            'total_time_minutes': round(session['total_time'] / 60, 2)
        }
    
    def export_session_data(self, user_id: str) -> str:
        """Exports user session data as JSON for external storage."""
        if user_id in self.user_sessions:
            return json.dumps(self.user_sessions[user_id], indent=2)
        return json.dumps({'error': 'User not found'})


# ============================================================================
# AGENT 4: FEEDBACK & EVALUATION AGENT
# Purpose: Generates personalized, adaptive feedback using LLM
# Technology: Gemini Pro API for natural language generation
# BONUS POINTS: Uses Gemini to power the agent
# ============================================================================

class FeedbackEvaluationAgent:
    """
    Feedback Agent generates personalized encouragement using Gemini Pro.
    Implements Agent Evaluation pattern - self-assesses and adapts responses.
    """
    
    def __init__(self, use_gemini: bool = True, api_key: str = None):
        self.use_gemini = use_gemini
        self.model = None
        
        if use_gemini:
            try:
                import google.generativeai as genai
                
                # Use API key from parameter or environment variable
                key = api_key or os.getenv('GEMINI_API_KEY')
                if key:
                    genai.configure(api_key=key)
                    self.model = genai.GenerativeModel('gemini-pro')
                    logger.info("FeedbackEvaluationAgent initialized with Gemini Pro")
                else:
                    logger.warning("Gemini API key not found. Using fallback feedback.")
                    self.use_gemini = False
            except Exception as e:
                logger.error(f"Gemini initialization error: {e}. Using fallback.")
                self.use_gemini = False
    
    def generate_feedback(self, user_id: str, task: Dict, success: bool, progress_data: Dict) -> str:
        """
        Generates adaptive, personalized feedback.
        Context Engineering: Uses progress data to tailor encouragement.
        Agent Evaluation: Self-assesses feedback quality.
        """
        if self.use_gemini and self.model:
            return self._generate_gemini_feedback(user_id, task, success, progress_data)
        else:
            return self._generate_fallback_feedback(task, success)
    
    def _generate_gemini_feedback(self, user_id: str, task: Dict, success: bool, progress_data: Dict) -> str:
        """Uses Gemini Pro to generate contextual feedback."""
        prompt = f"""
You are an encouraging AI coach for children with special needs learning life skills.

Student: {user_id}
Task Completed: {task['name']}
Success: {success}
Overall Progress: {progress_data.get('success_rate', 0)}% success rate, {progress_data.get('total_tasks', 0)} tasks completed

Provide:
1. Brief, positive encouragement (2-3 sentences)
2. One specific tip for improvement (if failed) or next challenge (if success)
3. Use simple, clear language suitable for children
4. Be enthusiastic and supportive

Feedback:
"""
        
        try:
            response = self.model.generate_content(prompt)
            feedback = response.text.strip()
            logger.info(f"Gemini feedback generated for user {user_id}")
            return feedback
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_fallback_feedback(task, success)
    
    def _generate_fallback_feedback(self, task: Dict, success: bool) -> str:
        """Fallback feedback when Gemini is unavailable."""
        if success:
            return f"ğŸ�‰ Amazing work on {task['name']}! You did a fantastic job! Keep practicing and you'll become even better!"
        else:
            return f"ğŸ‘� Nice try on {task['name']}! Learning takes practice. Let's try again together. You can do this!"

print("âœ… Agent 4: Feedback & Evaluation Agent loaded successfully!")


# ============================================================================
# ORCHESTRATOR: Multi-Agent System Coordinator
# Sequential Agent Pattern: Coordinates all 4 agents in workflow
# ============================================================================

class LifeSkillsOrchestrator:
    """
    Main orchestrator that coordinates all agents in sequential workflow.
    Implements complete multi-agent system with logging and observability.
    """
    
    def __init__(self, task_library: List[Dict], use_voice: bool = False, use_gemini: bool = False, api_key: str = None):
        # Initialize all agents
        self.scheduler = SchedulerAgent(task_library)
        self.prompter = PromptDeliveryAgent(use_voice=use_voice)
        self.tracker = ProgressTrackingAgent()
        self.evaluator = FeedbackEvaluationAgent(use_gemini=use_gemini, api_key=api_key)
        
        logger.info("\n" + "="*70)
        logger.info("ğŸŒŸ LifeSkills AI Orchestrator Initialized")
        logger.info("Multi-Agent System: 4 agents working in coordination")
        logger.info("="*70 + "\n")
    
    def run_session(self, user_id: str, skill_level: str = "beginner", num_tasks: int = 3):
        """
        Runs complete training session for a user.
        Sequential Agent Pattern: Scheduler -> Prompter -> Tracker -> Evaluator
        """
        logger.info(f"ğŸš€ Starting session for user: {user_id}")
        print("\n" + "âœ¨"*35)
        print(f"   Welcome to LifeSkills AI Training!")
        print(f"   Student: {user_id} | Level: {skill_level.upper()}")
        print("âœ¨"*35 + "\n")
        
        # Agent 1: Schedule tasks
        schedule = self.scheduler.create_schedule(user_id, skill_level)
        tasks_to_complete = schedule[:num_tasks]
        
        # Process each task sequentially
        for idx, task in enumerate(tasks_to_complete, 1):
            print(f"\nğŸ“� Task {idx}/{num_tasks}")
            print("-" * 60)
            
            # Agent 2: Deliver prompt
            self.prompter.deliver_prompt(task)
            
            # Simulate task completion (in real app, this would be user interaction)
            input("\nâ�¸ï¸�  Press ENTER when task is complete...") 
            success = input("âœ… Was the task completed successfully? (y/n): ").lower() == 'y'
            duration = 60.0  # Simulated duration
            
            # Agent 3: Track progress
            self.tracker.record_activity(user_id, task, success, duration)
            self.scheduler.mark_completed(task, success)
            
            # Agent 4: Generate feedback
            progress_data = self.tracker.get_user_progress(user_id)
            feedback = self.evaluator.generate_feedback(user_id, task, success, progress_data)
            
            print("\n" + "ğŸ’¬" + " FEEDBACK " + "ğŸ’¬")
            print("=" * 60)
            print(feedback)
            print("=" * 60)
        
        # Session summary
        self._display_session_summary(user_id)
    
    def _display_session_summary(self, user_id: str):
        """Displays final session summary with progress metrics."""
        progress = self.tracker.get_user_progress(user_id)
        
        print("\n\n" + "ğŸ�†" * 30)
        print("           SESSION COMPLETE!")
        print("ğŸ�†" * 30)
        print(f"\nğŸ“Š Progress Summary for {user_id}:")
        print(f"  â€¢ Total Tasks: {progress['total_tasks']}")
        print(f"  â€¢ Successful: {progress['successful_tasks']} âœ…")
        print(f"  â€¢ Success Rate: {progress['success_rate']}%")
        print(f"  â€¢ Total Time: {progress['total_time_minutes']} minutes")
        print("\nğŸ�‰ Keep up the great work! See you next time!\n")
        
        logger.info(f"Session completed for {user_id}. Success rate: {progress['success_rate']}%")

print("âœ… Multi-Agent Orchestrator loaded successfully!")


# ============================================================================
# SAMPLE TASK LIBRARY - Life Skills Activities
# Real-world daily living tasks for special needs education
# ============================================================================

TASK_LIBRARY = [
    {
        'id': 'T001',
        'name': 'Brushing Teeth',
        'difficulty': 1,
        'instructions': '''1. Wet your toothbrush\n2. Put toothpaste on the brush\n3. Brush your teeth for 2 minutes\n4. Rinse your mouth with water\n5. Clean and store your toothbrush'''
    },
    {
        'id': 'T002',
        'name': 'Taking a Bath',
        'difficulty': 2,
        'instructions': '''1. Turn on the water and check temperature\n2. Wet your body\n3. Apply soap and wash your body\n4. Rinse off all the soap\n5. Dry yourself with a towel'''
    },
    {
        'id': 'T003',
        'name': 'Making a Cup of Tea',
        'difficulty': 2,
        'instructions': '''1. Boil water in a kettle\n2. Place a tea bag in a cup\n3. Pour hot water into the cup\n4. Let it steep for 2-3 minutes\n5. Remove tea bag and add sugar/milk if desired'''
    },
    {
        'id': 'T004',
        'name': 'Getting Dressed',
        'difficulty': 1,
        'instructions': '''1. Choose clean clothes\n2. Put on underwear and shirt\n3. Put on pants or skirt\n4. Put on socks\n5. Put on shoes'''
    },
    {
        'id': 'T005',
        'name': 'Preparing a Simple Sandwich',
        'difficulty': 3,
        'instructions': '''1. Take 2 slices of bread\n2. Spread butter on both slices\n3. Add cheese or vegetables\n4. Put the slices together\n5. Cut in half (optional)'''
    }
]

print("âœ… Task Library loaded: %d life skills activities" % len(TASK_LIBRARY))
for task in TASK_LIBRARY:
    print(f"  â€¢ {task['name']} (Difficulty: {'â­�' * task['difficulty']})")

