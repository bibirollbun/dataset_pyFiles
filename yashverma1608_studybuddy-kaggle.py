# Install dependencies (warnings about other Kaggle packages can be ignored)
!pip install google-adk>=1.0.0 --quiet 2>/dev/null

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# For Kaggle, use secrets:
# from kaggle_secrets import UserSecretsClient
# os.environ['GEMINI_API_KEY'] = UserSecretsClient().get_secret('GEMINI_API_KEY')

print('Setup complete!')


class SpacedRepetitionScheduler:
    """
    Implements spaced repetition based on the forgetting curve.
    Intervals increase as mastery improves.
    """
    
    INTERVALS = [1, 3, 7, 14, 30, 60, 120]  # days
    
    def __init__(self):
        self.topic_data: Dict[str, dict] = {}
    
    def update_topic(self, topic: str, score: float) -> dict:
        """Update review schedule based on quiz performance."""
        if topic not in self.topic_data:
            self.topic_data[topic] = {
                'level': 0,
                'last_reviewed': datetime.now(),
                'scores': []
            }
        
        data = self.topic_data[topic]
        data['scores'].append(score)
        data['last_reviewed'] = datetime.now()
        
        # Adjust level based on performance
        if score >= 0.8:  # Great - move up
            data['level'] = min(data['level'] + 1, len(self.INTERVALS) - 1)
        elif score < 0.6:  # Struggling - move down
            data['level'] = max(data['level'] - 1, 0)
        # 0.6-0.8 stays at same level
        
        interval = self.INTERVALS[data['level']]
        next_review = datetime.now() + timedelta(days=interval)
        
        return {
            'topic': topic,
            'score': f'{score*100:.0f}%',
            'level': data['level'],
            'interval_days': interval,
            'next_review': next_review.strftime('%Y-%m-%d')
        }
    
    def get_due_topics(self) -> List[str]:
        """Get topics that need review."""
        due = []
        now = datetime.now()
        for topic, data in self.topic_data.items():
            interval = self.INTERVALS[data['level']]
            next_review = data['last_reviewed'] + timedelta(days=interval)
            if now >= next_review:
                due.append(topic)
        return due

print('SpacedRepetitionScheduler defined!')


scheduler = SpacedRepetitionScheduler()

print('=' * 60)
print('SPACED REPETITION DEMO')
print('=' * 60)

# Simulate different performance levels
test_cases = [
    ('Python Lists', 0.85),      # Good score - interval increases
    ('Binary Search', 0.45),     # Poor score - needs immediate review
    ('Recursion', 0.72),         # Medium score - maintains interval
    ('Data Structures', 0.95),   # Excellent - big interval increase
]

for topic, score in test_cases:
    result = scheduler.update_topic(topic, score)
    print(f"\nTopic: {result['topic']}")
    print(f"  Score: {result['score']}")
    print(f"  Level: {result['level']} / 6")
    print(f"  Next review in: {result['interval_days']} days")
    print(f"  Review date: {result['next_review']}")

print('\n' + '=' * 60)
print('Higher scores -> longer intervals')
print('Lower scores -> immediate review')
print('=' * 60)


class StudyBuddySession:
    """Manages a student's learning session with progress tracking."""
    
    def __init__(self, user_id: str = 'student'):
        self.user_id = user_id
        self.current_topic: Optional[str] = None
        self.quiz_results: List[dict] = []
        self.scheduler = SpacedRepetitionScheduler()
        self.created_at = datetime.now()
    
    def record_quiz(self, topic: str, score: float, questions: int) -> dict:
        """Record a quiz result and update spaced repetition."""
        result = {
            'topic': topic,
            'score': score,
            'questions': questions,
            'timestamp': datetime.now().isoformat()
        }
        self.quiz_results.append(result)
        
        # Update spaced repetition
        sr_update = self.scheduler.update_topic(topic, score)
        result['next_review'] = sr_update['next_review']
        result['interval_days'] = sr_update['interval_days']
        
        return result
    
    def get_progress_summary(self) -> dict:
        """Get overall learning progress."""
        if not self.quiz_results:
            return {'message': 'No quizzes taken yet!'}
        
        topics = set(r['topic'] for r in self.quiz_results)
        avg_score = sum(r['score'] for r in self.quiz_results) / len(self.quiz_results)
        
        # Find strengths and weaknesses
        topic_scores = {}
        for r in self.quiz_results:
            if r['topic'] not in topic_scores:
                topic_scores[r['topic']] = []
            topic_scores[r['topic']].append(r['score'])
        
        topic_avgs = {t: sum(s)/len(s) for t, s in topic_scores.items()}
        strengths = [t for t, s in topic_avgs.items() if s >= 0.8]
        weaknesses = [t for t, s in topic_avgs.items() if s < 0.6]
        
        return {
            'topics_studied': len(topics),
            'quizzes_taken': len(self.quiz_results),
            'average_score': f'{avg_score*100:.1f}%',
            'strengths': strengths,
            'needs_work': weaknesses,
            'due_for_review': self.scheduler.get_due_topics()
        }

print('StudyBuddySession defined!')


# Create a session and simulate study activity
session = StudyBuddySession('demo_student')

print('=' * 60)
print('PROGRESS TRACKING DEMO')
print('=' * 60)

# Simulate quiz results
quizzes = [
    ('Python Basics', 0.92, 10),
    ('Lists and Loops', 0.85, 8),
    ('Recursion', 0.62, 5),
    ('OOP Concepts', 0.58, 6),
    ('Python Basics', 0.95, 10),  # Improved!
]

print('\nRecording quiz results...\n')
for topic, score, questions in quizzes:
    result = session.record_quiz(topic, score, questions)
    print(f"Quiz: {topic} - {score*100:.0f}% ({questions} questions)")
    print(f"  -> Next review: {result['next_review']} ({result['interval_days']} days)")

print('\n' + '=' * 60)
print('PROGRESS SUMMARY')
print('=' * 60)

summary = session.get_progress_summary()
print(f"\nTopics studied: {summary['topics_studied']}")
print(f"Quizzes taken: {summary['quizzes_taken']}")
print(f"Average score: {summary['average_score']}")
print(f"\nStrengths: {', '.join(summary['strengths']) if summary['strengths'] else 'Keep practicing!'}")
print(f"Needs work: {', '.join(summary['needs_work']) if summary['needs_work'] else 'Great job!'}")
print(f"Due for review: {summary['due_for_review'] if summary['due_for_review'] else 'All caught up!'}")



# Agent system prompts (simplified for demo)

TUTOR_INSTRUCTION = '''
You are a patient, knowledgeable tutor. When explaining concepts:

1. EXPLANATION: Clear explanation with real-world analogies
2. KEY POINTS: 3-5 bullet points summarizing the essentials
3. FLASHCARDS: 2-3 Q&A pairs for review
4. QUICK CHECK: 1-2 questions to verify understanding

Adapt your explanations to the student's level.
'''

QUIZ_INSTRUCTION = '''
You create educational assessments. For each quiz:

1. Generate questions appropriate to the topic and difficulty
2. Include a mix of question types when possible
3. Provide clear answer keys with explanations
4. Give constructive feedback on performance
'''

PLANNER_INSTRUCTION = '''
You create personalized study plans. Each plan must include:

1. GOALS: Clear learning objectives
2. TIMELINE: Realistic schedule based on available time
3. TOPICS: Broken down into manageable chunks
4. REVIEW SESSIONS: Spaced at days 1, 3, 7, 14, 21, 28
5. MILESTONES: Checkpoints to measure progress
'''

ORCHESTRATOR_INSTRUCTION = '''
You are StudyBuddy, an AI learning companion. Route requests:

- Study plans, schedules -> Planner
- Explanations, concepts -> Tutor
- Quizzes, tests -> Quiz Agent
- Progress, stats -> Progress Tracker

Be encouraging and supportive!
'''

print('Agent instructions defined!')
print('\nIn the full system, these become LlmAgent instances with tools.')


# Tool implementations (simplified)

def record_quiz_result(topic: str, score: float, total_questions: int) -> str:
    """Record a quiz result to the student's session."""
    return f'Recorded: {topic} quiz with score {score*100:.0f}% ({total_questions} questions)'

def update_spaced_repetition(topic: str, score: float) -> str:
    """Update the spaced repetition schedule for a topic."""
    scheduler = SpacedRepetitionScheduler()
    result = scheduler.update_topic(topic, score)
    return f"Updated {topic}: next review in {result['interval_days']} days"

def get_progress_summary() -> str:
    """Get the student's overall progress summary."""
    return 'Progress: 5 topics studied, 78% average, recursion needs work'

def get_review_schedule() -> str:
    """Get upcoming review schedule."""
    return 'Reviews due: Recursion (overdue), OOP (tomorrow), Lists (in 5 days)'

def save_study_plan_to_file(plan: str, filename: str) -> str:
    """Save a study plan to a file."""
    return f'Study plan saved to {filename}'

print('Custom tools defined!')
print('\nThese tools let agents interact with session state.')


# Example: Study Plan Request
print('=' * 60)
print('EXAMPLE: Study Plan Request')
print('=' * 60)

user_request = 'I want to learn Python basics in 4 weeks, 2 hours a day'
print(f'\nUser: {user_request}')
print('\nStudyBuddy (Planner):')
print('''
### Overview
4 weeks with 2 hours daily = 56 hours total.
That's plenty for solid Python fundamentals.

### Weekly Breakdown
- Week 1: Variables, data types, basic operations
- Week 2: Control flow (if/else, loops), functions
- Week 3: Data structures - lists, dictionaries, sets
- Week 4: File handling, error handling, mini project

### Review Sessions (Spaced Repetition)
- Day 3: Quick review of Week 1 basics
- Day 7: First checkpoint
- Day 14: Cumulative review + practice
- Day 21: Everything up to data structures
- Day 28: Final review before project

### Milestones
- End of Week 2: Write 3 small programs
- End of Week 4: Build a working CLI app
''')


# Example: Tutoring Request
print('=' * 60)
print('EXAMPLE: Tutoring Request')
print('=' * 60)

user_request = 'Explain how binary search works'
print(f'\nUser: {user_request}')
print('\nStudyBuddy (Tutor):')
print('''
### The Idea
Imagine looking for a word in a dictionary. You don't start 
at page 1 - you open somewhere in the middle and adjust.
Binary search does the same thing with sorted data.

### Key Points
- Data MUST be sorted first
- Each step cuts search space in half
- Time complexity: O(log n) - very fast

### Flashcards
Q: What's the prerequisite for binary search?
A: Data must be sorted

Q: What's the time complexity?
A: O(log n) - logarithmic

### Quick Check
Searching for 7 in [1, 3, 5, 7, 9]:
1. What's the first element you compare?
2. How many comparisons to find 7?
(Answers: 5, then just 2 comparisons total!)
''')


# Example: After Quiz
print('=' * 60)
print('EXAMPLE: After Quiz Result')
print('=' * 60)

print('\n[Student just scored 85% on Python Lists quiz]')
print('\nStudyBuddy:')
print('''
Nice work on that quiz!

Here's what happened with your review schedule:
- Topic: Python Lists
- You scored: 85% (solid!)
- Old interval: 1 day
- New interval: 3 days

Since you scored above 80%, I'm spacing out your next review.
You'll see this topic again on December 5th.

Your retention is improving. Keep it up!
''')

